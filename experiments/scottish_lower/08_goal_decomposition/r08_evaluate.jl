# ==============================================================================
# r08 — Experiment 08 common-set proper-score benchmark
# ==============================================================================
#
# WHAT THIS IS. A no-MCMC, reproducible paired evaluation of persisted posterior
# draws. It compares the genuine Gen-3 baselines on m05's exact 710 fixtures and,
# only after the immutable production manifest names all four candidates, adds the
# new goal-decomposition family on that same fixed target.
#
# WHAT THIS IS NOT. It does not fabricate a baseline when deserialization fails;
# it does not score a candidate intersection; and its CRPS is explicitly the
# package's plug-in marginal home/away goal-count CRPS, not "scoreline CRPS".
# ===============================================================================

# %%
# ===============================================================================
# 1. Packages and implementation
# ===============================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using Statistics
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_evaluation.jl"))

const R08_EVALUATION_OUTPUT = joinpath(@__DIR__, "results")

# %%
# ===============================================================================
# 2. Evaluation and comparability contract
# ===============================================================================
# Scores use selection-weighted means: every available de-vigged Betfair selection
# in 1X2, O/U 2.5 and BTTS contributes one observation. ECE is ten equal-width
# probability bins, weighted by each bin's selection count. 1X2 RPS is the typed
# package implementation. CRPS is marginal home/away goal-count CRPS at the
# posterior-mean intensity, inherited for comparability from Evaluation.compute_crps.
#
# The baseline-only preflight deliberately does not require a candidate manifest.
# A full result does: production must have persisted all four named candidate UUIDs.

# %%
# ===============================================================================
# 3. Load exact runs and freeze the canonical fixture target
# ===============================================================================
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)

baseline_pairs = [l08_load_fit_checked(entry) for entry in L08_BASELINE_RUNS]
baseline_dbs = Dict(entry.name => pair[1] for (entry, pair) in zip(L08_BASELINE_RUNS, baseline_pairs))
fits = Dict(entry.name => pair[2] for (entry, pair) in zip(L08_BASELINE_RUNS, baseline_pairs))
canonical_ids = l08_canonical_match_ids(fits)

# m00 is intentionally restricted down to m05's published 710 IDs; m08 must already
# cover them. This is not an intersection that can hide a missing candidate fixture.
for (name, fit) in fits
    l08_require_coverage(name, fit, canonical_ids)
end

candidate_entries = l08_production_candidates(R08_EVALUATION_OUTPUT)
mode = candidate_entries === nothing ? "baselinepreflight" : "fullcomparison"
if candidate_entries !== nothing
    for entry in candidate_entries
        candidate_db, candidate_fit = l08_load_fit_checked(entry)
        fits[entry.name] = candidate_fit
        baseline_dbs[entry.name] = candidate_db
        l08_require_coverage(entry.name, candidate_fit, canonical_ids)
    end
end

# %%
# ===============================================================================
# 4. Identical de-vigged Betfair panel and proper scores
# ===============================================================================
# l08_betfair_closing_odds is the fixed TWA[-20, 0] source and de-vigs each market
# group by normalising implied probabilities. No stored bookmaker portfolio/evaluation
# output is read here.
odds = l08_betfair_closing_odds(ds)
common_odds = filter(:match_id => in(Set(canonical_ids)), odds)
common_matches = filter(:match_id => in(Set(canonical_ids)), ds.matches)

score_rows = DataFrame[]
panels = Dict{String,DataFrame}()
contexts = Dict{String,Any}()
for (name, fit) in sort!(collect(fits); by = first)
    context = Evaluation.build_evaluation_context(
        fit.latents, common_odds, common_matches,
        Evaluation.AbstractScoringRule[Evaluation.PredictionScore(), Evaluation.CRPS()];
        threaded = true,
    )
    summary, panel = l08_score_summary(name, context)
    push!(score_rows, summary)
    panels[name] = panel
    contexts[name] = context
end
scores = vcat(score_rows...)
CSV.write(joinpath(R08_EVALUATION_OUTPUT, "r08_$(mode)_common_set_scores.csv"), scores)

# %%
# ===============================================================================
# 5. Paired fixture bootstrap for log-loss differences
# ===============================================================================
# Re-sample fixtures (not individual selections) so 1X2 selections from one match
# remain together. This is an uncertainty summary, not an independent-fixture claim.
function r08_paired_logloss_bootstrap(reference::DataFrame, challenger::DataFrame;
                                      draws::Integer = 2_000, seed::Integer = 8)
    joined = innerjoin(select(reference, :match_id, :selection, :p_model => :p_reference,
                              :is_winner),
                       select(challenger, :match_id, :selection, :p_model => :p_challenger,
                              :is_winner), on = [:match_id, :selection, :is_winner])
    nrow(joined) == nrow(reference) == nrow(challenger) || error(
        "paired bootstrap requires identical scored selections")
    by_match = groupby(joined, :match_id)
    function mean_logloss(p, y)
        y_float = Float64.(y)
        return mean(-((y_float .* log.(clamp.(p, 1e-15, 1.0 - 1e-15))) .+
                       ((1.0 .- y_float) .* log.(clamp.(1.0 .- p, 1e-15, 1.0 - 1e-15)))))
    end
    deltas = [mean_logloss(part.p_challenger, part.is_winner) -
              mean_logloss(part.p_reference, part.is_winner) for part in by_match]
    rng = Xoshiro(seed)
    sampled = [mean(rand(rng, deltas, length(deltas))) for _ in 1:draws]
    return (reference = "m05_joint_production_wealth", challenger = "",
            n_fixtures = length(deltas), delta_logloss = mean(deltas),
            ci95_lo = quantile(sampled, 0.025), ci95_hi = quantile(sampled, 0.975))
end

bootstrap_rows = NamedTuple[]
for name in sort!(collect(keys(panels)))
    name == "m05_joint_production_wealth" && continue
    boot = r08_paired_logloss_bootstrap(panels["m05_joint_production_wealth"], panels[name])
    push!(bootstrap_rows, merge(boot, (challenger = name,)))
end
bootstrap = DataFrame(bootstrap_rows)
CSV.write(joinpath(R08_EVALUATION_OUTPUT, "r08_$(mode)_paired_logloss_bootstrap.csv"), bootstrap)

# %%
# ===============================================================================
# 6. Final report and immutable manifest
# ===============================================================================
l08_registry_obj = l08_registry(ds, PostgresStorage(L08_EXPERIMENT);
    output_dir = R08_EVALUATION_OUTPUT,
    source_files = [joinpath(@__DIR__, "l08_evaluation.jl"), @__FILE__])
manifest = l08_write_manifest!(l08_registry_obj; stage = mode,
    extra = Dict(
        "canonical_fixture_count" => length(canonical_ids),
        "canonical_match_ids" => canonical_ids,
        "odds_source" => "Betfair de-vigged TWA[-20,0]",
        "selection_weighting" => "equal weight per available selection",
        "ece" => "10 equal-width bins, count-weighted",
        "crps" => "Evaluation.compute_crps plug-in marginal home/away goal-count CRPS",
        "models" => sort!(collect(keys(fits))),
    ))

println("\nMODE: ", mode)
println("CANONICAL m05 SET: ", length(canonical_ids), " fixtures")
show(scores; allrows = true, allcols = true)
println("\nWrote score, bootstrap, and manifest artifacts under ", R08_EVALUATION_OUTPUT)
println("Manifest: ", manifest)
