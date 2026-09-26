# ==============================================================================
# r04 — Score the pyramid GRW arms against their controls on the 710-fixture panel
# ==============================================================================
#
# Proper scores vs the de-vigged Betfair TWA(−20,0] close (grw_player_hybrid's exact
# evaluation path), fixture-clustered paired bootstrap, and the compression scorecard:
#   * market-on-model slope of log-rate supremacy (1 = calibrated; > 1 = compressed)
#   * outcome-on-model slope (realised GD on model expected GD)
# plus the same cuts on the TRANSITION subgroup (fixtures with a club in its first
# season after changing tier), which is what cross-tier pooling is meant to fix.
#
#   julia --project -t 16 current_development/grw_pyramid_cups/r04_evaluate.jl
# ==============================================================================

# No thread pinning: this is scoring, not sampling, and may run beside a pinned grid.
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball, CSV, DataFrames, Dates, Printf, Statistics, UUIDs, Optim, SpecialFunctions
include(joinpath(@__DIR__, "l01_loader.jl"))
include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l02_evaluation.jl"))

const R04_CFG = PCXConfig()
const R04_OUT = joinpath(R04_CFG.save_root, "evaluation"); mkpath(R04_OUT)
const R04_B = 10_000

arms = GPHArm[]
pdb = PostgresStorage(R04_CFG.experiment)
for name in PCX_ARMS
    rid = gph_run_by_name(pdb, name)
    rid === nothing && (@warn "no completed run for $name — skipped"; continue)
    push!(arms, GPHArm(name, R04_CFG.experiment, rid, "MultiScaleGRW pooled", "candidate"))
end
push!(arms, GPHArm("m00_baseline_grw", "scottish_lower_multiscale_grw_2426",
                   UUID("f64a00a2-34a0-4f31-8c58-c093c92d54b7"), "MultiScaleGRW 56/57", "control"))
push!(arms, GPHArm("m05_joint_grw", "scottish_lower_multiscale_grw_2426",
                   UUID("f870dbb7-9df0-4dae-a84a-cf570cf8113e"), "MultiScaleGRW 56/57", "control"))
push!(arms, GPHArm("m12_grw", "scottish_lower_grw_player_hybrid",
                   UUID("3a9a4c7e-378b-45d0-a2d2-c8b69b46786b"), "MultiScaleGRW 56/57", "control"))
push!(arms, GPHArm("m12_td", "scottish_lower_joint_player_2426",
                   UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"), "TimeDecay(180) 56/57", "control"))

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
odds = gph_betfair_closing_odds(ds)
families = gph_family_selections(odds)

fits = Dict{String,Any}()
for a in arms
    f = gph_load_arm(a)
    panel = gph_season_panel(ds, f, R04_CFG.target_seasons)
    length(panel) == R04_CFG.expected_oos || error("$(a.label): $(length(panel)) panel fixtures")
    fits[a.label] = gph_restrict(f, panel)
end
panel = Set(fits[arms[1].label].latents.match_ids)
all(Set(f.latents.match_ids) == panel for f in values(fits)) || error("arms cover different panels")

# ---- proper scores + bootstrap -------------------------------------------------
score_rows = NamedTuple[]; obs = Dict{String,DataFrame}()
for a in arms
    ctx = gph_context(fits[a.label], odds, ds)
    append!(score_rows, [(; r..., dynamics = a.dynamics, role = a.role) for r in gph_scores(a.label, ctx, families)])
    obs[a.label] = gph_observation_frame(a.label, ctx, odds)
end
scores = DataFrame(score_rows)
pairs = [("g1_grw_all_spfl", "m00_baseline_grw"), ("g2_grw_all_spfl_cups", "g1_grw_all_spfl"),
         ("g2_grw_all_spfl_cups", "m00_baseline_grw"), ("g3_grw_joint_all_spfl_cups", "g2_grw_all_spfl_cups"),
         ("g3_grw_joint_all_spfl_cups", "m05_joint_grw"), ("g3_grw_joint_all_spfl_cups", "m12_grw"),
         ("g3_grw_joint_all_spfl_cups", "m12_td")]
boot = NamedTuple[]
for scope in ("all", "1X2"), (l, r) in pairs
    (haskey(obs, l) && haskey(obs, r)) || continue
    b = gph_paired_bootstrap(obs[l], obs[r]; B = R04_B, family = scope == "all" ? nothing : scope)
    push!(boot, (; left = l, right = r, scope, b...))
end
boot = DataFrame(boot)

# ---- compression scorecard ------------------------------------------------------
# market log-rate supremacy: exact independent-Poisson inversion of the de-vigged 1X2 close
function p1x2(λ, ν; K = 12)
    ph = pd = pa = 0.0
    for x in 0:K, y in 0:K
        q = exp(-λ - ν + x * log(λ) + y * log(ν) - loggamma(x + 1.0) - loggamma(y + 1.0))
        x > y ? (ph += q) : x == y ? (pd += q) : (pa += q)
    end
    s = ph + pd + pa; (ph / s, pd / s, pa / s)
end
function invert(ph, pa)
    r = optimize(v -> (q = p1x2(exp(v[1]), exp(v[2])); (q[1] - ph)^2 + (q[3] - pa)^2), [0.3, 0.1], NelderMead())
    v = Optim.minimizer(r); (exp(v[1]), exp(v[2]))
end
x12 = filter(r -> lowercase(r.market_name) == "1x2", odds)
mkt = Dict{Int,Float64}()
for g in groupby(x12, :match_id)
    ph = g.prob_fair_close[g.selection .== :home]; pa = g.prob_fair_close[g.selection .== :away]
    (length(ph) == 1 && length(pa) == 1) || continue
    λ, ν = invert(ph[1], pa[1]); mkt[Int(g.match_id[1])] = log(λ) - log(ν)
end
gd = Dict(Int(r.match_id) => Float64(r.home_score - r.away_score) for r in eachrow(ds.matches) if !ismissing(r.home_score))

# transition clubs: first season in 56/57 after a season in 54/55 (or no prior 56/57 season)
tier = Dict{Tuple{String,String},Int}()
pm = pcx_load_data(; max_age_hours = 10_000)   # pyramid store for league membership
for r in eachrow(pm.matches[in.(pm.matches.tournament_id, Ref(PCX_LEAGUES)), :])
    tier[(String(r.home_team), String(r.season))] = r.tournament_id
    tier[(String(r.away_team), String(r.season))] = r.tournament_id
end
prevs(s) = (y = parse(Int, s[1:2]); @sprintf("%02d/%02d", y - 1, y))
is_transition(team, season) = (p = get(tier, (team, prevs(season)), 0); c = get(tier, (team, season), 0); p != 0 && p != c)
trans = Set(Int(r.match_id) for r in eachrow(ds.matches) if Int(r.match_id) in panel &&
            (is_transition(String(r.home_team), String(r.season)) || is_transition(String(r.away_team), String(r.season))))

slope(y, x) = cov(x, y) / var(x)
card = NamedTuple[]
for a in arms, (sub, ids) in (("all", panel), ("transition", trans))
    lat = fits[a.label].latents
    rows = [i for (i, m) in enumerate(lat.match_ids) if Int(m) in ids]
    msup = [mean(log.(lat.λ_home[i, :]) .- log.(lat.λ_away[i, :])) for i in rows]
    megd = [mean(lat.λ_home[i, :] .- lat.λ_away[i, :]) for i in rows]
    mids = [Int(lat.match_ids[i]) for i in rows]
    km = [haskey(mkt, m) for m in mids]
    kg = [haskey(gd, m) for m in mids]
    push!(card, (model = a.label, subset = sub, n = length(rows),
                 market_on_model = slope([mkt[m] for m in mids[km]], msup[km]),
                 model_on_market = slope(msup[km], [mkt[m] for m in mids[km]]),
                 outcome_on_model = slope([gd[m] for m in mids[kg]], megd[kg]),
                 sd_model_sup = std(msup[km]), sd_market_sup = std([mkt[m] for m in mids[km]])))
end
card = DataFrame(card)

CSV.write(joinpath(R04_OUT, "r04_proper_scores.csv"), scores)
CSV.write(joinpath(R04_OUT, "r04_paired_bootstrap.csv"), boot)
CSV.write(joinpath(R04_OUT, "r04_compression_scorecard.csv"), card)
println("\n=== PROPER SCORES (all) ===")
show(sort(select(filter(:scope => ==("all"), scores), :model, :n_obs, :logloss, :market_logloss, :brier, :rps, :ece), :logloss); allrows = true); println()
println("\n=== PAIRED ΔLogLoss ===")
show(select(boot, :scope, :left, :right, :n_fixtures, :delta, :lo, :hi, :p_negative); allrows = true); println()
println("\n=== COMPRESSION SCORECARD ===")
show(card; allrows = true); println()
println("R04_DONE")
