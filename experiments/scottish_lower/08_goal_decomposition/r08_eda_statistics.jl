# This is a full-snapshot, non-MCMC statistical supplement to the incident EDA.
# It tests conditional count variation and sparse-component heterogeneity; it is not an
# OOS predictive comparison or evidence that a decomposed model should be promoted.
#
# The registry is immutable input from r08_eda.jl. If its recorded hash changes, rerun this
# supplement and interpret only artifacts carrying the new hash.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "l08_incident_data.jl"))
include(joinpath(@__DIR__, "l08_eda_statistics.jl"))
using .GoalDecompositionIncidentData
using .GoalDecompositionEDAStatistics

# %%
# ===================================================================
# 2. Configuration and reproducibility contract
# ===================================================================
const GD08_STATS_OUTPUT_DIR = joinpath(@__DIR__, "results")
const GD08_STATS_SEED = 20260908
const GD08_POISSON_REPLICATES = 1_000
const GD08_PENALTY_REPLICATES = 2_000
const GD08_MINIMUM_PRESSURE_HISTORY = 5
const GD08_REPEATABILITY_REPLICATES = 2_000

# GLM and parametric-bootstrap only: no Turing/MCMC is run here.
# All bootstrap p-values retain their Monte-Carlo standard error and use fixed seeds.
LinearAlgebra.BLAS.set_num_threads(1)

# %%
# ===================================================================
# 3. Frozen registry input and hash gate
# ===================================================================
registry, registry_hash = load_registry(GD08_STATS_OUTPUT_DIR)
matches = registry.matches
usable_matches = count(matches.usable_for_components)

println("Frozen registry SHA-256: ", registry_hash)
println("Finished matches: ", nrow(matches), "; component-usable: ", usable_matches)

# %%
# ===================================================================
# 4. Deterministic synthetic support checks
# ===================================================================
# These verify that side expansion and conversion no-attempt handling retain their intended
# support before touching the frozen operational registry.
@assert run_synthetic_tests()
println("G-S synthetic support checks: PASS")

# %%
# ===================================================================
# 5. Conditional non-penalty/non-own Poisson check
# ===================================================================
# The mean model includes receiving-team attack proxy, opponent defence proxy, home, season
# and division fixed effects. Each null replicate samples from that fitted conditional model
# then refits exactly the same nuisance design before calculating Pearson dispersion.
poisson_result = conditional_poisson_bootstrap(
    matches;
    replicates = GD08_POISSON_REPLICATES,
    seed = GD08_STATS_SEED,
)

# %%
# ===================================================================
# 6. Penalty drawing and conceding heterogeneity
# ===================================================================
# The exposure-weighted team Pearson statistic is assessed after a Poisson mean adjustment for
# home/away, season, division, strictly history-only non-penalty scoring pressure, and BBC
# referee identity (UNKNOWN is retained as a level). Drawing and conceding are paired views of
# the same events, so their p-values are descriptive rather than independent evidence.
drawing_result = penalty_heterogeneity_test(
    matches,
    :drawing;
    replicates = GD08_PENALTY_REPLICATES,
    seed = GD08_STATS_SEED,
)
conceding_result = penalty_heterogeneity_test(
    matches,
    :conceding;
    replicates = GD08_PENALTY_REPLICATES,
    seed = GD08_STATS_SEED,
)

# %%
# ===================================================================
# 7. Conversion pooling and exploratory repeatability
# ===================================================================
# Team rates with zero attempts remain missing, never zero. The heterogeneity bootstrap fixes
# every team's observed attempts and refits the pooled conversion probability in each replicate.
conversion_result = conversion_heterogeneity_test(
    matches;
    replicates = GD08_PENALTY_REPLICATES,
    seed = GD08_STATS_SEED,
)
repeatability_result = penalty_repeatability(
    matches;
    replicates = GD08_REPEATABILITY_REPLICATES,
    seed = GD08_STATS_SEED,
)

# %%
# ===================================================================
# 8. BBC referee coverage and observed penalty-rate variation
# ===================================================================
# This is the authoritative referee source. The test below will fail loudly if a stale registry
# predating the BBC match_officials join is supplied.
referee_result = referee_coverage_and_rates(matches)

# %%
# ===================================================================
# 9. History-only own-goal pressure association
# ===================================================================
# Pressure uses strictly earlier kickoffs. Same-kickoff matches update only after their block,
# and cold-start observations are excluded rather than interpreted as zero pressure.
own_goal_result = own_goal_pressure_test(
    matches;
    minimum_history = GD08_MINIMUM_PRESSURE_HISTORY,
)

# %%
# ===================================================================
# 10. Quarantine-selection outcome audit
# ===================================================================
# Component usability is incident/reconciliation driven, but it can still select on final-score
# outcomes. This table is descriptive and prevents a component-only summary being mistaken for
# an all-fixture distributional claim.
quarantine_result = quarantine_outcome_summary(matches)

# %%
# ===================================================================
# 11. Persisted result manifest and final summary
# ===================================================================
results = (; poisson = poisson_result, drawing = drawing_result, conceding = conceding_result,
           conversion = conversion_result, repeatability = repeatability_result, referee = referee_result,
           own_goal = own_goal_result, quarantine = quarantine_result)
write_statistics_artifacts(GD08_STATS_OUTPUT_DIR, results)

manifest = DataFrame(
    registry_hash = [registry_hash],
    generated_utc = [string(Dates.now(Dates.UTC))],
    seed = [GD08_STATS_SEED],
    poisson_refit_replicates = [GD08_POISSON_REPLICATES],
    penalty_null_replicates = [GD08_PENALTY_REPLICATES],
    conversion_null_replicates = [GD08_PENALTY_REPLICATES],
    repeatability_null_replicates = [GD08_REPEATABILITY_REPLICATES],
    minimum_pressure_history = [GD08_MINIMUM_PRESSURE_HISTORY],
)
CSV.write(joinpath(GD08_STATS_OUTPUT_DIR, "eda_stats_manifest.csv"), manifest)

println("G-1 conditional Poisson bootstrap p = ", @sprintf("%.4f", poisson_result.summary.p_value[1]))
println("G-2 penalty drawing p = ", @sprintf("%.4f", drawing_result.summary.p_value[1]),
        "; conceding p = ", @sprintf("%.4f", conceding_result.summary.p_value[1]))
println("G-3 pooled conversion = ", @sprintf("%.4f", conversion_result.summary.beta_posterior_mean[1]))
println("G-4 exploratory block correlation / null p = ", @sprintf("%.4f", repeatability_result.summary.correlation[1]),
        " / ", @sprintf("%.4f", repeatability_result.summary.p_value_positive[1]))
println("G-R named BBC-referee rows = ", sum(referee_result.coverage.named_referee_matches),
        "; unique referee levels = ", nrow(referee_result.rates))
println("G-5 history-only own-goal pressure coefficient = ", @sprintf("%.4f", own_goal_result.summary.pressure_coefficient[1]))
println("Statistical artifacts: ", GD08_STATS_OUTPUT_DIR)
