module GoalDecompositionEDAStatistics

import CSV
import DataFrames
import Dates
import Distributions
import GLM
import Random
import Statistics
import StatsModels

export nonpenalty_side_rows, conditional_poisson_bootstrap, penalty_heterogeneity_test,
       conversion_heterogeneity_test, penalty_repeatability, own_goal_pressure_test,
       referee_coverage_and_rates, quarantine_outcome_summary, write_statistics_artifacts,
       run_synthetic_tests

const DEFAULT_SEED = 20260908

function _quantile_interval(values::AbstractVector{<:Real}; level::Float64 = 0.95)
    α = (1.0 - level) / 2.0
    return (Statistics.quantile(values, α), Statistics.quantile(values, 1.0 - α))
end

function nonpenalty_side_rows(matches::DataFrames.DataFrame)
    usable = DataFrames.subset(matches, :usable_for_components => DataFrames.ByRow(identity))
    home = DataFrames.DataFrame(
        y = Int.(usable.non_penalty_non_own_goal_home),
        team = String.(usable.home_team),
        opponent = String.(usable.away_team),
        home = ones(Int, DataFrames.nrow(usable)),
        season = String.(usable.season),
        division = string.(usable.tournament_id),
    )
    away = DataFrames.DataFrame(
        y = Int.(usable.non_penalty_non_own_goal_away),
        team = String.(usable.away_team),
        opponent = String.(usable.home_team),
        home = zeros(Int, DataFrames.nrow(usable)),
        season = String.(usable.season),
        division = string.(usable.tournament_id),
    )
    return vcat(home, away)
end

function _poisson_fit(rows::DataFrames.DataFrame)
    return GLM.glm(StatsModels.@formula(y ~ 1 + team + opponent + home + season + division),
                   rows, Distributions.Poisson(), GLM.LogLink())
end

function _pearson_statistic(y::Vector{Int}, μ::Vector{Float64}, dof::Int)
    statistic = sum((Float64.(y) .- μ) .^ 2 ./ max.(μ, eps()))
    return statistic, statistic / dof
end

"""Parametric-bootstrap conditional-Poisson check that refits all nuisance fixed effects."""
function conditional_poisson_bootstrap(matches::DataFrames.DataFrame;
                                       replicates::Int = 1_000,
                                       seed::Int = DEFAULT_SEED)
    rows = nonpenalty_side_rows(matches)
    fit = _poisson_fit(rows)
    μ = Vector{Float64}(GLM.predict(fit))
    n_parameters = length(GLM.coef(fit))
    dof = DataFrames.nrow(rows) - n_parameters
    dof > 0 || error("conditional Poisson model has non-positive residual degrees of freedom")
    observed_x2, observed_phi = _pearson_statistic(rows.y, μ, dof)
    rng = Random.MersenneTwister(seed)
    null_x2 = Vector{Float64}(undef, replicates)
    null_phi = Vector{Float64}(undef, replicates)
    for replicate in eachindex(null_x2)
        simulated = copy(rows)
        simulated.y = Random.rand.(Ref(rng), Distributions.Poisson.(μ))
        simulated_fit = _poisson_fit(simulated)
        simulated_μ = Vector{Float64}(GLM.predict(simulated_fit))
        null_x2[replicate], null_phi[replicate] = _pearson_statistic(simulated.y, simulated_μ, dof)
    end
    p_value = (count(>=(observed_x2), null_x2) + 1) / (replicates + 1)
    mcse = sqrt(p_value * (1.0 - p_value) / (replicates + 1))
    summary = DataFrames.DataFrame(
        test = ["conditional_poisson_pearson_refit_bootstrap"],
        observations = [DataFrames.nrow(rows)],
        residual_df = [dof],
        observed_pearson_x2 = [observed_x2],
        observed_dispersion = [observed_phi],
        null_x2_mean = [Statistics.mean(null_x2)],
        null_x2_interval_low = [_quantile_interval(null_x2)[1]],
        null_x2_interval_high = [_quantile_interval(null_x2)[2]],
        p_value = [p_value],
        monte_carlo_se = [mcse],
        replicates = [replicates],
        seed = [seed],
    )
    return (; summary, null = DataFrames.DataFrame(replicate = eachindex(null_x2), pearson_x2 = null_x2, dispersion = null_phi), rows)
end

"""Fixture-side penalty rows with strictly earlier non-penalty/non-own scoring pressure."""
function _penalty_side_rows(matches::DataFrames.DataFrame, role::Symbol)
    role in (:drawing, :conceding) || error("role must be :drawing or :conceding")
    usable = sort(DataFrames.subset(matches, :usable_for_components => DataFrames.ByRow(identity)), :match_date)
    goals = Dict{String, Int}()
    exposure = Dict{String, Int}()
    rows = NamedTuple[]
    # Same-kickoff fixtures use pre-block state, then update together: no contemporaneous leak.
    for block in DataFrames.groupby(usable, :match_date)
        for match in eachrow(block)
            home_pressure = get(goals, match.home_team, 0) / max(get(exposure, match.home_team, 0), 1)
            away_pressure = get(goals, match.away_team, 0) / max(get(exposure, match.away_team, 0), 1)
            home_count = role === :drawing ? match.penalty_awarded_home : match.penalty_awarded_away
            away_count = role === :drawing ? match.penalty_awarded_away : match.penalty_awarded_home
            hasproperty(match, :referee_name) || error("registry has no referee_name column — regenerate it from the BBC-officials audit before referee-adjusted analysis")
            referee = ismissing(match.referee_name) || isempty(strip(String(match.referee_name))) ? "UNKNOWN" : String(match.referee_name)
            push!(rows, (; team = String(match.home_team), count = Int(home_count), home = "home", season = String(match.season), division = string(match.tournament_id), referee, pressure = home_pressure, history_matches = get(exposure, match.home_team, 0)))
            push!(rows, (; team = String(match.away_team), count = Int(away_count), home = "away", season = String(match.season), division = string(match.tournament_id), referee, pressure = away_pressure, history_matches = get(exposure, match.away_team, 0)))
        end
        for match in eachrow(block)
            goals[match.home_team] = get(goals, match.home_team, 0) + match.non_penalty_non_own_goal_home
            goals[match.away_team] = get(goals, match.away_team, 0) + match.non_penalty_non_own_goal_away
            exposure[match.home_team] = get(exposure, match.home_team, 0) + 1
            exposure[match.away_team] = get(exposure, match.away_team, 0) + 1
        end
    end
    rows = DataFrames.DataFrame(rows)
    rows.pressure_z = (rows.pressure .- Statistics.mean(rows.pressure)) ./ Statistics.std(rows.pressure)
    return rows
end

function _team_pearson_statistic(rows::DataFrames.DataFrame, counts::Vector{Int}, μ::Vector{Float64})
    observed = Dict{String, Float64}()
    expected = Dict{String, Float64}()
    for (index, team) in enumerate(rows.team)
        observed[team] = get(observed, team, 0.0) + counts[index]
        expected[team] = get(expected, team, 0.0) + μ[index]
    end
    teams = sort(collect(keys(expected)))
    statistic = sum((observed[team] - expected[team])^2 / max(expected[team], eps()) for team in teams)
    return statistic, teams, observed, expected
end

function _penalty_fit(rows::DataFrames.DataFrame, referee_adjusted::Bool)
    if referee_adjusted
        return GLM.glm(StatsModels.@formula(count ~ 1 + pressure_z + home + season + division + referee), rows,
                       Distributions.Poisson(), GLM.LogLink())
    end
    return GLM.glm(StatsModels.@formula(count ~ 1 + pressure_z + home + season + division), rows,
                   Distributions.Poisson(), GLM.LogLink())
end

"""Exposure-weighted team Pearson test after history-only pressure and optional BBC referee adjustment."""
function penalty_heterogeneity_test(matches::DataFrames.DataFrame, role::Symbol;
                                    replicates::Int = 2_000,
                                    seed::Int = DEFAULT_SEED,
                                    referee_adjusted::Bool = true)
    rows = _penalty_side_rows(matches, role)
    fit = _penalty_fit(rows, referee_adjusted)
    μ = Vector{Float64}(GLM.predict(fit))
    observed_statistic, teams, observed, expected = _team_pearson_statistic(rows, rows.count, μ)
    rng = Random.MersenneTwister(seed + (role === :drawing ? 1 : 2))
    null = Vector{Float64}(undef, replicates)
    for replicate in eachindex(null)
        simulated = Random.rand.(Ref(rng), Distributions.Poisson.(μ))
        simulated_rows = copy(rows)
        simulated_rows.count = simulated
        simulated_fit = _penalty_fit(simulated_rows, referee_adjusted)
        simulated_μ = Vector{Float64}(GLM.predict(simulated_fit))
        null[replicate] = _team_pearson_statistic(rows, simulated, simulated_μ)[1]
    end
    p_value = (count(>=(observed_statistic), null) + 1) / (replicates + 1)
    mcse = sqrt(p_value * (1.0 - p_value) / (replicates + 1))
    effects = DataFrames.DataFrame(team = teams,
        observed = [observed[team] for team in teams],
        expected_under_adjusted_null = [expected[team] for team in teams])
    effects.smr = effects.observed ./ effects.expected_under_adjusted_null
    summary = DataFrames.DataFrame(role = [String(role)], statistic = [observed_statistic],
        null_mean = [Statistics.mean(null)], null_interval_low = [_quantile_interval(null)[1]],
        null_interval_high = [_quantile_interval(null)[2]], p_value = [p_value], monte_carlo_se = [mcse],
        teams = [length(teams)], events = [sum(rows.count)], side_match_exposure = [DataFrames.nrow(rows)],
        cold_start_rows = [count(rows.history_matches .== 0)], pressure_adjustment = ["history_only_nonpenalty_rate"],
        referee_adjustment = [referee_adjusted ? "bbc_referee_fixed_effect; UNKNOWN retained" : "none_pre_bbc_audit"],
        referees = [length(unique(rows.referee))], replicates = [replicates], seed = [seed])
    return (; summary, effects, null = DataFrames.DataFrame(replicate = eachindex(null), statistic = null))
end

function conversion_heterogeneity_test(matches::DataFrames.DataFrame;
                                       replicates::Int = 2_000,
                                       seed::Int = DEFAULT_SEED,
                                       prior_a::Float64 = 1.0,
                                       prior_b::Float64 = 1.0)
    usable = DataFrames.subset(matches, :usable_for_components => DataFrames.ByRow(identity))
    attempts = vcat(Int.(usable.penalty_awarded_home), Int.(usable.penalty_awarded_away))
    converted = vcat(Int.(usable.penalty_goal_home), Int.(usable.penalty_goal_away))
    teams = vcat(String.(usable.home_team), String.(usable.away_team))
    grouped = DataFrames.combine(DataFrames.groupby(DataFrames.DataFrame(; team = teams, attempts, converted), :team),
        :attempts => sum => :attempts, :converted => sum => :converted)
    total_attempts, total_converted = sum(attempts), sum(converted)
    total_attempts > 0 || error("no valid penalty attempts available for conversion analysis")
    posterior = Distributions.Beta(prior_a + total_converted, prior_b + total_attempts - total_converted)
    pooled_mle = total_converted / total_attempts
    active = DataFrames.subset(grouped, :attempts => DataFrames.ByRow(>(0)))
    observed_statistic = sum((active.converted .- active.attempts .* pooled_mle).^2 ./
                             (active.attempts .* pooled_mle .* (1.0 - pooled_mle)))
    rng = Random.MersenneTwister(seed + 3)
    null = Vector{Float64}(undef, replicates)
    for replicate in eachindex(null)
        simulated = Random.rand.(Ref(rng), Distributions.Binomial.(active.attempts, pooled_mle))
        refit = sum(simulated) / sum(active.attempts)
        # At a boundary refit every simulated team count equals its attempts (or zero), so the
        # Pearson denominator vanishes. The correct finite-support discrepancy is zero there.
        null[replicate] = refit == 0.0 || refit == 1.0 ? 0.0 :
            sum((simulated .- active.attempts .* refit).^2 ./
                (active.attempts .* refit .* (1.0 - refit)))
    end
    p_value = (count(>=(observed_statistic), null) + 1) / (replicates + 1)
    mcse = sqrt(p_value * (1.0 - p_value) / (replicates + 1))
    grouped.conversion = Vector{Union{Missing, Float64}}(missing, DataFrames.nrow(grouped))
    has_attempts = grouped.attempts .> 0
    grouped.conversion[has_attempts] .= grouped.converted[has_attempts] ./ grouped.attempts[has_attempts]
    summary = DataFrames.DataFrame(total_attempts = [total_attempts], total_converted = [total_converted],
        pooled_mle = [pooled_mle], beta_prior_a = [prior_a], beta_prior_b = [prior_b],
        beta_posterior_mean = [Statistics.mean(posterior)], beta_interval_low = [Distributions.quantile(posterior, 0.025)],
        beta_interval_high = [Distributions.quantile(posterior, 0.975)], teams = [DataFrames.nrow(grouped)],
        teams_with_no_attempts = [count(.!has_attempts)], observed_pearson_x2 = [observed_statistic],
        null_mean = [Statistics.mean(null)], p_value = [p_value], monte_carlo_se = [mcse], replicates = [replicates], seed = [seed])
    return (; summary, teams = grouped, null = DataFrames.DataFrame(replicate = eachindex(null), statistic = null))
end

"""Exploratory chronological-block correlation against an exposure-aware pooled-Poisson null."""
function penalty_repeatability(matches::DataFrames.DataFrame;
                               replicates::Int = 2_000,
                               seed::Int = DEFAULT_SEED)
    usable = DataFrames.subset(matches, :usable_for_components => DataFrames.ByRow(identity))
    ordered = sort(usable, :match_date)
    midpoint = ordered.match_date[cld(DataFrames.nrow(ordered), 2)]
    # ISO-8601 timestamps are lexicographically ordered; retaining them as strings also avoids
    # silently dropping the explicit +00:00 offset supplied by the frozen registry.
    early = DataFrames.subset(ordered, :match_date => DataFrames.ByRow(x -> x < midpoint))
    late = DataFrames.subset(ordered, :match_date => DataFrames.ByRow(x -> x >= midpoint))
    function rates(frame)
        side = vcat(DataFrames.DataFrame(team = String.(frame.home_team), penalties = Int.(frame.penalty_awarded_home)),
                    DataFrames.DataFrame(team = String.(frame.away_team), penalties = Int.(frame.penalty_awarded_away)))
        return DataFrames.combine(DataFrames.groupby(side, :team), :penalties => sum => :penalties, DataFrames.nrow => :exposure)
    end
    joined = DataFrames.outerjoin(rates(early), rates(late), on = :team, makeunique = true)
    DataFrames.dropmissing!(joined)
    joined.early_rate = joined.penalties ./ joined.exposure
    joined.late_rate = joined.penalties_1 ./ joined.exposure_1
    correlation = Statistics.cor(joined.early_rate, joined.late_rate)
    early_rate = sum(joined.penalties) / sum(joined.exposure)
    late_rate = sum(joined.penalties_1) / sum(joined.exposure_1)
    rng = Random.MersenneTwister(seed + 4)
    null = Vector{Float64}(undef, replicates)
    for replicate in eachindex(null)
        simulated_early = Random.rand.(Ref(rng), Distributions.Poisson.(early_rate .* joined.exposure))
        simulated_late = Random.rand.(Ref(rng), Distributions.Poisson.(late_rate .* joined.exposure_1))
        null[replicate] = Statistics.cor(simulated_early ./ joined.exposure, simulated_late ./ joined.exposure_1)
    end
    p_value = (count(>=(correlation), null) + 1) / (replicates + 1)
    mcse = sqrt(p_value * (1.0 - p_value) / (replicates + 1))
    return (; summary = DataFrames.DataFrame(metric = ["nonoverlapping_block_team_penalty_rate_correlation"],
        early_matches = [DataFrames.nrow(early)], late_matches = [DataFrames.nrow(late)], teams = [DataFrames.nrow(joined)],
        correlation = [correlation], null_mean = [Statistics.mean(null)], null_interval_low = [_quantile_interval(null)[1]],
        null_interval_high = [_quantile_interval(null)[2]], p_value_positive = [p_value], monte_carlo_se = [mcse],
        replicates = [replicates], seed = [seed]),
        null = DataFrames.DataFrame(replicate = eachindex(null), correlation = null))
end

"""BBC referee coverage and match-level penalty award rates; UNKNOWN is reported separately."""
function referee_coverage_and_rates(matches::DataFrames.DataFrame)
    hasproperty(matches, :referee_name) || error("registry has no referee_name column — rerun incident EDA with BBC match_officials joined")
    referee = [ismissing(value) || isempty(strip(String(value))) ? "UNKNOWN" : String(value) for value in matches.referee_name]
    frame = DataFrames.DataFrame(; referee, usable = matches.usable_for_components,
        penalties = Int.(matches.penalty_awarded_home) .+ Int.(matches.penalty_awarded_away))
    coverage = DataFrames.combine(DataFrames.groupby(frame, :usable),
        DataFrames.nrow => :matches,
        :referee => (x -> count(!=("UNKNOWN"), x)) => :named_referee_matches,
        :referee => (x -> count(==("UNKNOWN"), x)) => :unknown_referee_matches)
    usable = DataFrames.subset(frame, :usable => DataFrames.ByRow(identity))
    rates = DataFrames.combine(DataFrames.groupby(usable, :referee), DataFrames.nrow => :matches,
        :penalties => sum => :penalties)
    rates.rate_per_match = rates.penalties ./ rates.matches
    sort!(rates, :rate_per_match)
    return (; coverage, rates)
end

function own_goal_pressure_test(matches::DataFrames.DataFrame; minimum_history::Int = 5)
    usable = sort(DataFrames.subset(matches, :usable_for_components => DataFrames.ByRow(identity)), :match_date)
    rows = NamedTuple[]
    goals = Dict{String, Int}()
    exposures = Dict{String, Int}()
    grouped = DataFrames.groupby(usable, :match_date)
    for block in grouped
        for match in eachrow(block)
            home_n, away_n = get(exposures, match.home_team, 0), get(exposures, match.away_team, 0)
            if home_n >= minimum_history
                push!(rows, (; y = Int(match.own_goal_home), pressure = goals[match.home_team] / home_n, home = 1, season = String(match.season), division = string(match.tournament_id), history_matches = home_n))
            end
            if away_n >= minimum_history
                push!(rows, (; y = Int(match.own_goal_away), pressure = goals[match.away_team] / away_n, home = 0, season = String(match.season), division = string(match.tournament_id), history_matches = away_n))
            end
        end
        for match in eachrow(block)
            goals[match.home_team] = get(goals, match.home_team, 0) + match.non_penalty_non_own_goal_home
            goals[match.away_team] = get(goals, match.away_team, 0) + match.non_penalty_non_own_goal_away
            exposures[match.home_team] = get(exposures, match.home_team, 0) + 1
            exposures[match.away_team] = get(exposures, match.away_team, 0) + 1
        end
    end
    data = DataFrames.DataFrame(rows)
    data.pressure_z = (data.pressure .- Statistics.mean(data.pressure)) ./ Statistics.std(data.pressure)
    flat = GLM.glm(StatsModels.@formula(y ~ 1 + home + season + division), data, Distributions.Poisson(), GLM.LogLink())
    pressure = GLM.glm(StatsModels.@formula(y ~ 1 + pressure_z + home + season + division), data, Distributions.Poisson(), GLM.LogLink())
    coefficient_index = findfirst(==("pressure_z"), GLM.coefnames(pressure))
    coefficient_index === nothing && error("pressure coefficient missing from own-goal model")
    β = GLM.coef(pressure)[coefficient_index]
    se = GLM.stderror(pressure)[coefficient_index]
    likelihood_ratio = 2.0 * (GLM.loglikelihood(pressure) - GLM.loglikelihood(flat))
    summary = DataFrames.DataFrame(observations = [DataFrames.nrow(data)], own_goals = [sum(data.y)], minimum_history = [minimum_history],
        pressure_coefficient = [β], interval_low = [β - 1.96 * se], interval_high = [β + 1.96 * se],
        rate_ratio_per_pressure_sd = [exp(β)], likelihood_ratio = [likelihood_ratio],
        flat_loglikelihood = [GLM.loglikelihood(flat)], pressure_loglikelihood = [GLM.loglikelihood(pressure)])
    return (; summary, rows = data)
end

function quarantine_outcome_summary(matches::DataFrames.DataFrame)
    rows = DataFrames.DataFrame(
        group = ifelse.(matches.usable_for_components, "component_usable", "quarantined"),
        total_goals = Int.(matches.overall_home) .+ Int.(matches.overall_away),
    )
    return DataFrames.combine(DataFrames.groupby(rows, :group),
        DataFrames.nrow => :matches,
        :total_goals => Statistics.mean => :mean_final_goals,
        :total_goals => (x -> Statistics.mean(x .>= 4)) => :share_final_goals_at_least_4,
    )
end

function run_synthetic_tests()
    synthetic = DataFrames.DataFrame(
        usable_for_components = trues(8), home_team = ["A", "B", "A", "B", "A", "B", "A", "B"], away_team = ["B", "A", "B", "A", "B", "A", "B", "A"],
        non_penalty_non_own_goal_home = [1, 0, 2, 1, 1, 0, 1, 2], non_penalty_non_own_goal_away = [0, 1, 0, 2, 0, 1, 1, 0],
        penalty_awarded_home = [0, 1, 0, 0, 1, 0, 0, 1], penalty_awarded_away = [1, 0, 0, 1, 0, 0, 1, 0],
        penalty_goal_home = [0, 1, 0, 0, 0, 0, 0, 1], penalty_goal_away = [1, 0, 0, 1, 0, 0, 1, 0],
        own_goal_home = zeros(Int, 8), own_goal_away = zeros(Int, 8), season = fill("x", 8), tournament_id = fill(56, 8),
        match_date = ["2020-01-0$(i)T12:00:00" for i in 1:8],
    )
    @assert DataFrames.nrow(nonpenalty_side_rows(synthetic)) == 16
    conversion = conversion_heterogeneity_test(synthetic; replicates = 20)
    @assert conversion.summary.total_attempts[1] == 6
    @assert conversion.summary.teams_with_no_attempts[1] == 0
    return true
end

function write_statistics_artifacts(output_dir::AbstractString, results::NamedTuple)
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "eda_stats_conditional_poisson.csv"), results.poisson.summary)
    CSV.write(joinpath(output_dir, "eda_stats_conditional_poisson_null.csv"), results.poisson.null)
    CSV.write(joinpath(output_dir, "eda_stats_penalty_drawing.csv"), results.drawing.summary)
    CSV.write(joinpath(output_dir, "eda_stats_penalty_drawing_effects.csv"), results.drawing.effects)
    CSV.write(joinpath(output_dir, "eda_stats_penalty_conceding.csv"), results.conceding.summary)
    CSV.write(joinpath(output_dir, "eda_stats_penalty_conceding_effects.csv"), results.conceding.effects)
    CSV.write(joinpath(output_dir, "eda_stats_conversion.csv"), results.conversion.summary)
    CSV.write(joinpath(output_dir, "eda_stats_conversion_teams.csv"), results.conversion.teams)
    CSV.write(joinpath(output_dir, "eda_stats_repeatability.csv"), results.repeatability.summary)
    CSV.write(joinpath(output_dir, "eda_stats_repeatability_null.csv"), results.repeatability.null)
    CSV.write(joinpath(output_dir, "eda_stats_referee_coverage.csv"), results.referee.coverage)
    CSV.write(joinpath(output_dir, "eda_stats_referee_rates.csv"), results.referee.rates)
    CSV.write(joinpath(output_dir, "eda_stats_own_goal_pressure.csv"), results.own_goal.summary)
    CSV.write(joinpath(output_dir, "eda_stats_own_goal_pressure_rows.csv"), results.own_goal.rows)
    CSV.write(joinpath(output_dir, "eda_stats_quarantine_outcomes.csv"), results.quarantine)
end

end # module
