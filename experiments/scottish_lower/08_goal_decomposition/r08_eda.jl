# This is an incident-semantic and distributional EDA runner.
# It is not an MCMC, model-selection, or portfolio experiment.
#
# Question: does the operational SofaScore feed provide a score-reconciling, temporally safe
# decomposition of Scottish Lower goals that can be passed to a model without fabricated zeros?
# Provider `regular` is named non-penalty/non-own: it includes set pieces and is not open play.
# The model-facing answer is a frozen registry plus explicit usable mask and snapshot hash.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf
using Statistics
import DotEnv

# Module precompilation does not preserve environment mutations: load runtime credentials
# from the ignored project environment only when the caller has not supplied the DSN.
if !haskey(ENV, "BF_DB_URL")
    DotEnv.load!(ENV, joinpath(pkgdir(BayesianFootball), ".env"))
end

include(joinpath(@__DIR__, "l08_incident_data.jl"))
using .GoalDecompositionIncidentData

# %%
# ===================================================================
# 2. Configuration and artifact contract
# ===================================================================
const GD08_OUTPUT_DIR = joinpath(@__DIR__, "results")
const GD08_PERMUTATIONS = 10_000
const GD08_SEED = 20260908

# All extraction is read-only. The loader starts a read-only transaction and reads BF_DB_URL
# only from the environment. The registry retains final-score data for all matches but component
# likelihoods may consume rows only where component_usable_mask=true.

# %%
# ===================================================================
# 3. Data snapshot and frozen match-component registry
# ===================================================================
registry, orientation = extract_registry()
snapshot_hash = write_registry_artifacts(registry, orientation, GD08_OUTPUT_DIR)
features = model_feature_view(registry, snapshot_hash)

println("Goal-decomposition registry snapshot: ", snapshot_hash)
println("Finished matches: ", nrow(registry.matches))
println("Component-usable matches: ", count(registry.matches.usable_for_components))
println("Quarantined matches: ", nrow(registry.quarantines))

# %%
# ===================================================================
# 4. Incident semantics and score-reconciliation gates
# ===================================================================
orientation_summary = combine(groupby(orientation, [:recipient, :expected_recipient, :agrees]), nrow => :n)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_own_goal_orientation_summary.csv"), orientation_summary)

if nrow(orientation) > 0 && !all(orientation.agrees)
    @warn "G-A own-goal semantics has $(count(!, orientation.agrees)) ambiguous/disagreeing incidents; their matches are quarantined. See eda_own_goal_orientation.csv."
end

# %%
# ===================================================================
# 5. Penalty-attempt quality gate and marginal component distribution
# ===================================================================
attempt_summary, attempt_flags = attempt_quality_audit(registry)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_attempt_quality_summary.csv"), attempt_summary)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_attempt_quality_flags.csv"), attempt_flags)
println("G-B1 penalty-attempt audit: ", nrow(attempt_flags), " candidate flags; inspect before two-stage promotion.")

summary = component_summary(registry)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_component_distribution.csv"), summary)

# VMR is a marginal descriptive statistic. It is deliberately not interpreted as conditional
# Poisson overdispersion because team strength, home advantage, division and time are unadjusted.

# %%
# ===================================================================
# 6. Team penalty generation/concession and conversion heterogeneity
# ===================================================================
usable = subset(registry.matches, :usable_for_components => ByRow(identity))
team_penalty = vcat(
    DataFrame(team = usable.home_team, role = "awarded", penalties = usable.penalty_awarded_home,
              converted = usable.penalty_goal_home, exposure = ones(Int, nrow(usable))),
    DataFrame(team = usable.away_team, role = "awarded", penalties = usable.penalty_awarded_away,
              converted = usable.penalty_goal_away, exposure = ones(Int, nrow(usable))),
    DataFrame(team = usable.home_team, role = "conceded", penalties = usable.penalty_awarded_away,
              converted = usable.penalty_goal_away, exposure = ones(Int, nrow(usable))),
    DataFrame(team = usable.away_team, role = "conceded", penalties = usable.penalty_awarded_home,
              converted = usable.penalty_goal_home, exposure = ones(Int, nrow(usable))),
)
team_penalty_summary = combine(groupby(team_penalty, [:team, :role]),
    :penalties => sum => :penalties,
    :converted => sum => :converted,
    :exposure => sum => :exposure)
team_penalty_summary.rate = team_penalty_summary.penalties ./ team_penalty_summary.exposure
team_penalty_summary.conversion = [attempts == 0 ? missing : converted / attempts
                                   for (converted, attempts) in zip(team_penalty_summary.converted, team_penalty_summary.penalties)]
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_team_penalty_rates.csv"), team_penalty_summary)

# The dedicated statistics runner owns adjusted dispersion, generation/concession nulls,
# conversion heterogeneity, repeatability and own-goal-pressure inference. It consumes the
# frozen registry produced above rather than re-querying the operational database.

# %%
# ===================================================================
# 7. Own-goal pressure, division and referee coverage audit
# ===================================================================
pressure = pressure_feature(registry)
own_pressure = leftjoin(features, pressure, on = :match_id)
own_pressure = subset(own_pressure, :component_usable_mask => ByRow(identity))
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_own_goal_pressure_rows.csv"), own_pressure)

division_rates = combine(groupby(usable, :tournament_id),
    :non_penalty_non_own_goal_home => sum => :non_penalty_non_own_goal_home,
    :non_penalty_non_own_goal_away => sum => :non_penalty_non_own_goal_away,
    :penalty_goal_home => sum => :penalty_goal_home,
    :penalty_goal_away => sum => :penalty_goal_away,
    :own_goal_home => sum => :own_goal_home,
    :own_goal_away => sum => :own_goal_away,
    nrow => :matches)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_division_rates.csv"), division_rates)

referee_audit = combine(groupby(registry.matches, :referee_present), nrow => :matches)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_referee_coverage.csv"), referee_audit)
referee_seasons = combine(groupby(registry.matches, [:season, :tournament_id]),
    nrow => :matches,
    :referee_present => sum => :bbc_named_referees,
    :sofascore_referee_present => sum => :sofascore_json_referees,
    :usable_for_components => sum => :component_usable)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_referee_coverage_by_season.csv"), referee_seasons)
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_referee_rates_usable.csv"), referee_rates(registry))
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_referee_rates_raw.csv"), referee_rates(registry; usable_only = false))
CSV.write(joinpath(GD08_OUTPUT_DIR, "eda_referee_deviance.csv"), referee_deviance_summary(registry))
println("BBC named referees: ", count(registry.matches.referee_present), "/", nrow(registry.matches))

# %%
# ===================================================================
# 8. Final report for the model agent
# ===================================================================
println("G-A own-goal orientation: ", all(orientation.agrees) ? "PASS" : "QUARANTINE — ambiguous rows retained in audit.")
println("G-B score reconciliation: ", nrow(registry.quarantines) == 0 ? "PASS" : "QUARANTINE PRESENT")
println("G-B1 valid-attempt completeness: NOT PROVEN — raw feed has no independent award/retake/shootout marker.")
println("G-C frozen feature interface: model_feature_view(registry, snapshot_hash) verified.")
println("Artifacts: ", GD08_OUTPUT_DIR)
