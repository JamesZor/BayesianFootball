# ==============================================================================
# Task 008 Phase 1 loader — HierarchicalTeamHomeAdvantage across three production tiers
# ==============================================================================
#
# Definitions only. `r01_smoke.jl`, `r02_production_grid.jl`, `r04_evaluate.jl` and
# `r05_slate_repricing.jl` execute.
#
# THE LADDER. Three candidates, each the exact twin of a persisted flat-HA control
# with ONE slot changed: `GlobalHomeAdvantage()` → `HierarchicalTeamHomeAdvantage()`.
#
#   candidate                              dynamics           control (flat HA)
#   m05_joint_production_wealth_hier_ha    TimeDecay(180)     m05_joint_td_raw     (Exp 06)
#   m12_joint_hybrid_synergy_hier_ha       TimeDecay(180)     m12_hybrid_td_raw    (Exp 06)
#   m12_joint_hybrid_synergy_grw_hier_ha   MultiScaleGRW      m12_joint_hybrid_synergy_grw (Task 013)
#
#   γ_i = γ_base + σ_γ · z_i,   z_i ~ N(0, 1),   γ_base ~ N(0.2, 0.2),   σ_γ ~ N⁺(0, 0.1)
#
# `i` indexes the HOME club, so γ_i is a ground effect only to the extent a club
# plays at one ground — which is true for every 56/57 club in the panel.
#
# SHARED MACHINERY. Every non-HA component, the splitter, the gradient audit, the
# filtration report, thinning, latent audit and the save/load round-trip come from
# Task 013's `l01_loader.jl` by include. That file already pins the Exp 06 recipes
# verbatim (joint observation, production wealth, shots pillar) and they are what
# the three controls were fitted with — reusing them rather than copying them is
# what keeps "exactly one slot differs" true.
#
# KNOWN EXTRACTION ASYMMETRY (ticket T003). `extract_parameters` prices a held-out
# fixture whose home team is absent from the fold's `team_map` with γ = 0 under a
# hierarchical HA, but with γ_global under the flat control
# (src/models/pregame/builder/engine.jl:629). The posterior-predictive answer for an
# unseen club is γ_base. `hha_unmapped_home_report` counts the affected fixtures so
# every score can be read with, and without, them.
# ==============================================================================

if !isdefined(@__MODULE__, :GPHConfig)
    include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l01_loader.jl"))
end

const HHA_PG = BayesianFootball.Models.PreGame

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

"""
    HHAConfig

Every number that governs Phase 1, in one place. Runners read `config.samples`,
never `ENV` directly.

The production sampler is Task 013's (4 × (500 warmup + 1000 retained), δ = 0.80,
max depth 10), which is what the GRW control was fitted with. The two TimeDecay
controls were fitted at Exp 06's 4 × (800 + 800), δ = 0.65; a sampler difference
changes Monte-Carlo noise, not the posterior, and is reported beside every score.

`persist_stride = 2` for the same reason as Task 013: a GRW fit at 4,000 draws per
fold exceeds PostgreSQL's 1 GB field limit once serialised. The audit runs on every
draw; only the persisted panel is thinned.
"""
Base.@kwdef struct HHAConfig
    experiment::String = "scottish_lower_hierarchical_ha"
    smoke_experiment::String = "smoke_hier_ha"
    save_root::String = joinpath(@__DIR__, "results")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    extension_seasons::Vector{String} = ["24/25", "25/26", "26/27"]
    expected_folds::Int = 40
    expected_oos::Int = 710

    smoke_folds::Int = 2
    smoke_samples::Int = 400
    smoke_warmup::Int = 400
    smoke_chains::Int = 4

    samples::Int = 1000
    warmup::Int = 500
    chains::Int = 4
    accept_rate::Float64 = 0.80
    max_depth::Int = 10
    max_concurrent_tasks::Int = 16

    persist_stride::Int = 2
    gradient_replays::Int = 200

    # The six-part audit. The work package writes the R̂ gate as ≤ 1.05.
    max_rhat::Float64 = 1.05
    strict_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end

const HHA_MODEL_NAMES = [
    "m05_joint_production_wealth_hier_ha",
    "m12_joint_hybrid_synergy_hier_ha",
    "m12_joint_hybrid_synergy_grw_hier_ha",
]

const HHA_DESCRIPTIONS = Dict(
    "m05_joint_production_wealth_hier_ha" =>
        "Exp 06 m05 control with hierarchical team home advantage: TimeDecay(180) + production wealth + joint.",
    "m12_joint_hybrid_synergy_hier_ha" =>
        "Gen 4 TimeDecay hybrid with hierarchical team home advantage: lineup + wealth + joint.",
    "m12_joint_hybrid_synergy_grw_hier_ha" =>
        "Gen 4 MultiScaleGRW hybrid with hierarchical team home advantage: lineup + wealth + joint.",
)

const HHA_DYNAMICS = Dict(
    "m05_joint_production_wealth_hier_ha" => "TimeDecay(180)",
    "m12_joint_hybrid_synergy_hier_ha" => "TimeDecay(180)",
    "m12_joint_hybrid_synergy_grw_hier_ha" => "MultiScaleGRW",
)

const HHA_TAGS = [
    "scottish-lower", "24/25", "25/26", "hierarchical-home-advantage",
    "todo008", "phase1", "reversediff",
]

"""
The persisted flat-HA twin of each candidate, pinned by immutable UUID.

The two TimeDecay pins are the runs the Exp 06 README and Task 013 scored; the GRW
pin is Task 013's 43-fold `m12_joint_hybrid_synergy_grw`.
"""
const HHA_CONTROLS = Dict(
    "m05_joint_production_wealth_hier_ha" => GPHControl(
        "m05_joint_td_raw", "scottish_lower_joint_player_2426",
        UUID("ed541a7c-01e2-447e-a771-783517728d47"), "m05_joint_production_wealth",
        "Exp 06 control: TimeDecay(180) + production wealth + joint, flat HA"),
    "m12_joint_hybrid_synergy_hier_ha" => GPHControl(
        "m12_hybrid_td_raw", "scottish_lower_joint_player_2426",
        UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"), "m12_joint_hybrid_synergy",
        "Gen 4 production hybrid (Run 67 lineage), flat HA"),
    "m12_joint_hybrid_synergy_grw_hier_ha" => GPHControl(
        "m12_grw_raw", "scottish_lower_grw_player_hybrid",
        UUID("3a9a4c7e-378b-45d0-a2d2-c8b69b46786b"), "m12_joint_hybrid_synergy_grw",
        "Task 013 GRW hybrid, flat HA"),
)

# ==============================================================================
# 2. Models
# ==============================================================================

"The slot under test. Its priors are the component defaults, stated in the header."
hha_home_advantage() = HierarchicalTeamHomeAdvantage()

"""
    hha_models(; home_advantage = hha_home_advantage()) -> Vector{Tuple{String,Any}}

The three candidates in ladder order. `home_advantage = GlobalHomeAdvantage()` builds
the flat twins, which the gradient audit uses to attribute tape cost to the one slot
that changed. Add order is Exp 06's: interception, dynamics, HA, pillar, wealth,
observation.
"""
function hha_models(; home_advantage = hha_home_advantage())
    m05 = CountModelBuilder(:m05_joint_production_wealth_hier_ha) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(home_advantage) |>
        add(gph_production_wealth()) |>
        add(gph_joint_observation()) |>
        build

    m12 = CountModelBuilder(:m12_joint_hybrid_synergy_hier_ha) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(home_advantage) |>
        add(gph_shots_pillar()) |>
        add(gph_production_wealth()) |>
        add(gph_joint_observation()) |>
        build

    m12_grw = CountModelBuilder(:m12_joint_hybrid_synergy_grw_hier_ha) |>
        add(GlobalInterception()) |>
        add(gph_dynamics()) |>
        add(home_advantage) |>
        add(gph_shots_pillar()) |>
        add(gph_production_wealth()) |>
        add(gph_joint_observation()) |>
        build

    return Tuple{String,Any}[
        (HHA_MODEL_NAMES[1], m05),
        (HHA_MODEL_NAMES[2], m12),
        (HHA_MODEL_NAMES[3], m12_grw),
    ]
end

# ==============================================================================
# 3. Sampler, execution, thresholds, recipes
# ==============================================================================

hha_smoke_sampler(c::HHAConfig) = QueuedNUTSConfig(
    n_samples = c.smoke_samples,
    n_warmup = c.smoke_warmup,
    n_chains = c.smoke_chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    show_progress = false,
)

hha_production_sampler(c::HHAConfig) = QueuedNUTSConfig(
    n_samples = c.samples,
    n_warmup = c.warmup,
    n_chains = c.chains,
    accept_rate = c.accept_rate,
    max_depth = c.max_depth,
    show_progress = false,
)

hha_execution(c::HHAConfig) = QueuedExecution(max_concurrent_tasks = c.max_concurrent_tasks)

hha_thresholds(c::HHAConfig) = ConvergenceThresholds(
    max_rhat = c.max_rhat,
    min_ess = c.min_ess,
    max_divergence_rate = c.max_divergence_rate,
    min_bfmi = c.min_bfmi,
    max_treedepth_rate = c.max_treedepth_rate,
)

function hha_fit_configs(c::HHAConfig, models, splitter, sampler;
                         name_suffix::AbstractString = "")
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = hha_execution(c),
        tags = copy(HHA_TAGS),
        description = HHA_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

"Register every canonical component and assembled recipe in `config_registry`."
function hha_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = save_model(db, name, model;
                                     description = HHA_DESCRIPTIONS[name], tags = HHA_TAGS)
        fit_hashes[name] = save_config(db, name * "_fit", configs[name];
                                       description = HHA_DESCRIPTIONS[name] * " Task 008 Phase 1 recipe.",
                                       tags = HHA_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_hier_ha_40fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25 and 25/26.",
        tags = HHA_TAGS)
    sampler_id = save_sampler(db, "queued_nuts_4x1000_w500_a080", sampler;
        description = "ReverseDiff queued NUTS: 4 chains, 500 warmup, 1000 retained, target acceptance 0.80.",
        tags = HHA_TAGS)
    return (; model_ids, fit_hashes, splitter_id, sampler_id)
end

"One run-level convergence row, identical in columns to Task 013's so reports stack."
hha_convergence_row(name::AbstractString, fit, c::HHAConfig; run_id = nothing) =
    gph_convergence_row(name, fit, GPHConfig(strict_rhat = c.strict_rhat); run_id)

"Sample every fold in `inputs`, checkpointing each fold as it lands."
function hha_sample(fit_config::FitConfig, inputs, c::HHAConfig; checkpoint_dir = nothing)
    return fit_model(fit_config;
                     feature_sets = inputs.feature_sets,
                     oos_fixtures = inputs.oos,
                     thresholds = hha_thresholds(c),
                     checkpoint_dir = checkpoint_dir,
                     cleanup_checkpoints = false,
                     quiet = false)
end

# ==============================================================================
# 4. Home-advantage diagnostics
# ==============================================================================

_hha_is_ha_site(p) = startswith(String(p), "ha.")

"""
    hha_ha_site_report(fit) -> DataFrame

R̂, bulk ESS and tail ESS for every `ha.*` site, worst value over folds.

Same estimators as `audit_fold` (`MCMCChains.rhat`, `MCMCChains.ess(kind = …)`), so a
site that fails here also moved the run-level numbers. The run-level audit reports
only the single worst parameter; this names the HA sites explicitly, which is what
the work package's gate asks for.
"""
function hha_ha_site_report(fit)
    worst = Dict{String,NamedTuple}()
    for f in fit.folds
        chain = f.chain
        rh = DataFrame(MCMCChains.rhat(chain))
        eb = DataFrame(MCMCChains.ess(chain; kind = :bulk))
        et = DataFrame(MCMCChains.ess(chain; kind = :tail))
        bulk = Dict(String(p) => v for (p, v) in zip(eb.parameters, eb.ess))
        tail = Dict(String(p) => v for (p, v) in zip(et.parameters, et.ess))
        for (p, r) in zip(rh.parameters, rh.rhat)
            name = String(p)
            _hha_is_ha_site(name) || continue
            row = (; rhat = Float64(r), ess_bulk = Float64(bulk[name]),
                     ess_tail = Float64(tail[name]), fold = Int(f.fold))
            prev = get(worst, name, nothing)
            worst[name] = prev === nothing ? row :
                (; rhat = max(prev.rhat, row.rhat),
                   ess_bulk = min(prev.ess_bulk, row.ess_bulk),
                   ess_tail = min(prev.ess_tail, row.ess_tail),
                   fold = row.rhat > prev.rhat ? row.fold : prev.fold)
        end
    end
    isempty(worst) && error("no ha.* sites in the chain — is this a hierarchical-HA fit?")
    names_sorted = sort!(collect(keys(worst)))
    return DataFrame(site = names_sorted,
                     max_rhat = [worst[n].rhat for n in names_sorted],
                     min_ess_bulk = [worst[n].ess_bulk for n in names_sorted],
                     min_ess_tail = [worst[n].ess_tail for n in names_sorted],
                     worst_rhat_fold = [worst[n].fold for n in names_sorted])
end

"""
    hha_unmapped_home_report(inputs) -> DataFrame

One row per held-out fixture whose HOME team is absent from its fold's `team_map`.
Those fixtures are priced with γ = 0 by the hierarchical extraction and γ_global by
the flat one (ticket T003), so every comparison must be readable without them.
"""
function hha_unmapped_home_report(inputs)
    rows = NamedTuple[]
    for (i, fs) in enumerate(inputs.feature_sets)
        oos = inputs.oos[i]
        oos === nothing && continue
        team_map = first(fs).data[:team_map]
        for r in eachrow(oos)
            haskey(team_map, r.home_team) && continue
            push!(rows, (; fold = i, match_id = Int(r.match_id),
                           home_team = String(r.home_team), away_team = String(r.away_team),
                           away_mapped = haskey(team_map, r.away_team)))
        end
    end
    isempty(rows) && return DataFrame(fold = Int[], match_id = Int[], home_team = String[],
                                      away_team = String[], away_mapped = Bool[])
    return DataFrame(rows)
end

"""
    hha_ground_effects(fit, fold, feature_set) -> (clubs::DataFrame, hyper::NamedTuple)

Posterior of every club's γ_i on one fold, plus the hyperparameters. `feature_set` is
the fold's `FeatureCollection` — the chain's `γ_team_raw[i]` index is only meaningful
through that fold's `team_map`, which `FoldFit.meta` does not carry.

`clubs` carries mean, sd, 5/50/95% quantiles and `p_above_base = P(γ_i > γ_base)`
(the per-club draw comparison, not the comparison of two marginals). `hyper`
carries γ_base and σ_γ summaries and `p_sigma_below_0p02`, the posterior mass of
σ_γ near the boundary — a truncated-at-zero scale never places mass at exactly zero,
so "excludes zero" is read as the lower 5% quantile being away from the boundary.
"""
function hha_ground_effects(fit, fold::Int, feature_set)
    model = fit.config.model
    model.home_advantage isa HierarchicalTeamHomeAdvantage || error(
        "fold $fold: model home advantage is $(typeof(model.home_advantage))")
    f = fit.folds[fold]
    team_map = first(feature_set).data[:team_map]
    n_teams = length(team_map)
    width = count(p -> startswith(String(p), "ha.γ_team_raw["), names(f.chain, :parameters))
    width == n_teams || error("fold $fold: chain holds $width γ_team_raw sites but the " *
                              "feature set's team_map has $n_teams clubs — wrong feature set")
    ha = HHA_PG.extract_home_advantage(f.chain, model.home_advantage, n_teams)
    base = vec(Array(f.chain[Symbol("ha.γ_base")]))
    sigma = vec(Array(f.chain[Symbol("ha.σ_γ")]))
    team_of = Dict(i => t for (t, i) in team_map)
    q(v, p) = quantile(v, p)
    clubs = DataFrame(
        team = [team_of[i] for i in 1:n_teams],
        mean = [mean(ha[:, i]) for i in 1:n_teams],
        sd = [std(ha[:, i]) for i in 1:n_teams],
        q05 = [q(ha[:, i], 0.05) for i in 1:n_teams],
        q50 = [q(ha[:, i], 0.50) for i in 1:n_teams],
        q95 = [q(ha[:, i], 0.95) for i in 1:n_teams],
        p_above_base = [mean(ha[:, i] .> base) for i in 1:n_teams],
    )
    sort!(clubs, :mean; rev = true)
    hyper = (; fold,
               gamma_base_mean = mean(base), gamma_base_sd = std(base),
               gamma_base_q05 = q(base, 0.05), gamma_base_q95 = q(base, 0.95),
               sigma_mean = mean(sigma), sigma_sd = std(sigma),
               sigma_q05 = q(sigma, 0.05), sigma_q50 = q(sigma, 0.50), sigma_q95 = q(sigma, 0.95),
               p_sigma_below_0p02 = mean(sigma .< 0.02),
               prior_p_sigma_below_0p02 = cdf(truncated(Normal(0, 0.1), lower = 0.0), 0.02))
    return clubs, hyper
end

# ==============================================================================
# 5. Surface classification
# ==============================================================================

"""
Clubs the work package classifies as playing on synthetic 3G/4G turf.

These are name fragments, matched case-insensitively against the store's team names by
`hha_classify_surface`, which refuses a fragment that matches zero or several clubs.
The classification is the work package's, not re-derived here; Phase 2 replaces it
with a dated `is_synthetic_pitch` feature, because a club can relay its pitch.
"""
const HHA_TURF_FRAGMENTS = [
    "airdrie", "east kilbride", "montrose", "spartans", "edinburgh city",
    "clyde", "alloa", "cove rangers",
]

function hha_classify_surface(teams::AbstractVector{<:AbstractString})
    # DataStore names are slugs ("east-kilbride", "cove-rangers"); normalise to words first.
    lowered = [replace(lowercase(String(t)), r"[^a-z]+" => " ") for t in teams]
    turf = falses(length(teams))
    for frag in HHA_TURF_FRAGMENTS
        hits = findall(t -> occursin(frag, t), lowered)
        # "clyde" is a substring of "clydebank"; prefer an exact-word hit when several match.
        if length(hits) > 1
            exact = filter(i -> any(==(frag), split(lowered[i], r"[^a-z]+")), hits)
            hits = length(exact) == 1 ? exact : hits
        end
        length(hits) <= 1 || error("turf fragment \"$frag\" matches several clubs: " *
                                   join(teams[hits], ", "))
        isempty(hits) || (turf[only(hits)] = true)
    end
    return turf
end
