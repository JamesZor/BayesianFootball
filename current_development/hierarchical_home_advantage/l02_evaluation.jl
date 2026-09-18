# ==============================================================================
# Task 008 Phase 1 loader — proper scores, flat-vs-hierarchical contrasts, ground effects
# ==============================================================================
#
# Definitions only. `r04_evaluate.jl` and `r05_slate_repricing.jl` execute. Nothing
# here samples: every posterior is loaded from `mcmc_experiments` by UUID.
#
# The scoring machinery (Betfair TWA(−20, 0] close, panel restriction, `evaluate_
# predictions` scopes, fixture-clustered paired bootstrap) is Task 013's
# `l02_evaluation.jl`, included, so a number here is on the same scale as a number
# in the Task 013 and Exp 06 READMEs.
# ==============================================================================

if !isdefined(@__MODULE__, :HHAConfig)
    include(joinpath(@__DIR__, "l01_loader.jl"))
end
if !isdefined(@__MODULE__, :GPHArm)
    include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l02_evaluation.jl"))
end

# ==============================================================================
# 1. Arms
# ==============================================================================

"""
    hha_arms(c) -> Vector{GPHArm}

The three candidates by name from this task's namespace, then each one's pinned
flat-HA control. `role` is `"candidate"` or `"control"`; the pairing is
`HHA_CONTROLS[candidate].label`.
"""
function hha_arms(c::HHAConfig)
    db = PostgresStorage(c.experiment)
    arms = GPHArm[]
    for name in HHA_MODEL_NAMES
        run_id = gph_run_by_name(db, name)
        run_id === nothing && error("no completed run named $name in $(c.experiment) — run r02 first")
        push!(arms, GPHArm(name, c.experiment, run_id, HHA_DYNAMICS[name], "candidate"))
    end
    for name in HHA_MODEL_NAMES
        ctl = HHA_CONTROLS[name]
        push!(arms, GPHArm(ctl.label, ctl.experiment, ctl.run_id, HHA_DYNAMICS[name], "control"))
    end
    return arms
end

# ==============================================================================
# 2. Fold team maps without rebuilding features
# ==============================================================================

"""
    hha_fold_team_map(ds, splitter, fold) -> Dict{String,Int}

The `team_map` fold `fold` was fitted with, rebuilt from its split boundary by the
feature builder's own rule (`src/features/builder.jl:69-70`: sorted unique home and
away names over history + target matches).

Rebuilding the full feature set would cost the RAPM ridge fit for nothing. The
rebuild is only trusted after `hha_ground_draws` checks its size against the chain's
γ_team_raw width, which catches the rule drifting.
"""
function hha_fold_team_map(ds, splitter, fold::Int)
    boundaries = Data.create_id_boundaries(ds, splitter)
    b = first(boundaries[fold])
    ids = Set(Int.(vcat(b.history_match_ids, b.target_match_ids)))
    rows = ds.matches[[Int(m) in ids for m in ds.matches.match_id], :]
    nrow(rows) == length(ids) || error("fold $fold: $(nrow(rows)) match rows for $(length(ids)) ids")
    teams = sort(unique(vcat(String.(rows.home_team), String.(rows.away_team))))
    return Dict(t => i for (i, t) in enumerate(teams))
end

"""
    hha_ground_draws(fit, fold, team_map) -> (γ::Matrix, base::Vector, sigma::Vector, teams)

Per-draw γ_i (draws × clubs) for one fold of a loaded hierarchical-HA fit.
"""
function hha_ground_draws(fit, fold::Int, team_map::AbstractDict)
    model = fit.config.model
    model.home_advantage isa HierarchicalTeamHomeAdvantage || error(
        "$(fit.config.name) is not a hierarchical-HA fit")
    chain = fit.folds[fold].chain
    width = count(p -> startswith(String(p), "ha.γ_team_raw["), names(chain, :parameters))
    width == length(team_map) || error(
        "fold $fold: chain has $width γ_team_raw sites, rebuilt team_map has $(length(team_map))")
    γ = HHA_PG.extract_home_advantage(chain, model.home_advantage, width)
    base = vec(Array(chain[Symbol("ha.γ_base")]))
    sigma = vec(Array(chain[Symbol("ha.σ_γ")]))
    team_of = Dict(i => t for (t, i) in team_map)
    return γ, base, sigma, [team_of[i] for i in 1:width]
end

"""
    hha_turf_contrast(fit, fold, team_map) -> (clubs::DataFrame, contrast::NamedTuple)

The Phase 1 surface test on one fold. Per draw,

    Δ = mean(γ_i : turf clubs) − mean(γ_i : grass clubs)

so `p_turf_above` is a posterior probability, not a comparison of two marginals.
Clubs are classified by `hha_classify_surface` (the work package's list).
"""
function hha_turf_contrast(fit, fold::Int, team_map::AbstractDict)
    γ, base, sigma, teams = hha_ground_draws(fit, fold, team_map)
    turf = hha_classify_surface(teams)
    any(turf) && !all(turf) || error("fold $fold: needs both turf and grass clubs; turf = $(count(turf))")
    Δ = vec(mean(γ[:, turf]; dims = 2)) .- vec(mean(γ[:, .!turf]; dims = 2))
    clubs = DataFrame(team = teams, surface = [t ? "turf" : "grass" for t in turf],
                      gamma_mean = vec(mean(γ; dims = 1)), gamma_sd = vec(std(γ; dims = 1)),
                      gamma_q05 = [quantile(γ[:, i], 0.05) for i in eachindex(teams)],
                      gamma_q95 = [quantile(γ[:, i], 0.95) for i in eachindex(teams)],
                      home_multiplier = vec(mean(exp.(γ); dims = 1)),
                      p_above_base = [mean(γ[:, i] .> base) for i in eachindex(teams)])
    sort!(clubs, :gamma_mean; rev = true)
    contrast = (; fold, n_turf = count(turf), n_grass = count(.!turf),
                  delta_mean = mean(Δ), delta_q05 = quantile(Δ, 0.05), delta_q95 = quantile(Δ, 0.95),
                  p_turf_above = mean(Δ .> 0),
                  gamma_base_mean = mean(base),
                  sigma_q05 = quantile(sigma, 0.05), sigma_q50 = quantile(sigma, 0.50),
                  sigma_q95 = quantile(sigma, 0.95),
                  p_sigma_below_0p02 = mean(sigma .< 0.02),
                  prior_p_sigma_below_0p02 = cdf(truncated(Normal(0, 0.1), lower = 0.0), 0.02))
    return clubs, contrast
end

# ==============================================================================
# 3. Row annotations for split scoring
# ==============================================================================

"""
    hha_annotate!(obs, ds, unmapped_ids) -> obs

Adds `home_team`, `home_surface` and `unmapped_home` to an observation frame so the
bootstrap can be cut by the hypothesis (turf home grounds) and by the T003 exclusion.
"""
function hha_annotate!(obs::DataFrame, ds, unmapped_ids::AbstractSet{<:Integer})
    home_of = Dict(Int(r.match_id) => String(r.home_team) for r in eachrow(ds.matches))
    obs.home_team = [home_of[m] for m in obs.match_id]
    clubs = unique(obs.home_team)
    surface = Dict(zip(clubs, hha_classify_surface(clubs)))
    obs.home_surface = [surface[t] ? "turf" : "grass" for t in obs.home_team]
    obs.unmapped_home = [m in unmapped_ids for m in obs.match_id]
    return obs
end

"""
    hha_paired(a, b; subset, B, seed, family) -> NamedTuple

`gph_paired_bootstrap` on the rows of `a` (and `b`) selected by the Bool vector
function `subset(frame)`. Both frames are cut by the same predicate on their own
columns, so a fixture is in or out of both.
"""
function hha_paired(a::DataFrame, b; subset = df -> trues(nrow(df)), B::Int = 10_000,
                    seed::Int = 20260912, family = nothing)
    aa = a[subset(a), :]
    bb = b === :market ? :market : b[subset(b), :]
    return gph_paired_bootstrap(aa, bb; B, seed, family)
end
