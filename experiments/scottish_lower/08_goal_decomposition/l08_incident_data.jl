module GoalDecompositionIncidentData

import CSV
import DataFrames
import Dates
import Distributions
import JSON3
import LibPQ
import Random
import SHA
import Serialization
import Statistics

export GoalComponentRegistry, extract_registry, write_registry_artifacts, load_registry,
       registry_snapshot_hash, model_feature_view, component_summary,
       poisson_homogeneity_permutation, attempt_quality_audit, own_goal_orientation_audit,
       pressure_feature, referee_rates, referee_deviance_summary

"""
Immutable, match-level registry for decomposed-goal models. Counts are only populated for
matches whose incident decomposition reconciles exactly to the published final score.
`overall_home` and `overall_away` remain available for every finished match as the total-score
fallback; `usable_for_components` is the mandatory training filter for component likelihoods.
"""
struct GoalComponentRegistry
    matches::DataFrames.DataFrame
    incidents::DataFrames.DataFrame
    quarantines::DataFrames.DataFrame
    extraction_utc::String
end

const GOAL_CLASSES = Set(["regular", "penalty", "ownGoal"])

_json_string(value) = value === nothing || value === missing ? missing : String(value)

function _json_value(raw::AbstractString, key::String)
    object = JSON3.read(raw)
    return haskey(object, Symbol(key)) ? object[Symbol(key)] : nothing
end

function _bool_json(raw::AbstractString, key::String)
    value = _json_value(raw, key)
    return value === true
end

function _int_json(raw::AbstractString, key::String)
    value = _json_value(raw, key)
    value === nothing && return missing
    try
        return Int(value)
    catch
        return missing
    end
end

function _text_json(raw::AbstractString, key::String)
    value = _json_value(raw, key)
    return _json_string(value)
end

function _provider_id_json(raw::AbstractString)
    value = _json_value(raw, "id")
    value === nothing && return missing
    return string(value)
end

function _player_id_json(raw::AbstractString)
    player = _json_value(raw, "player")
    player === nothing && return missing
    haskey(player, :id) || return missing
    return string(player[:id])
end

function _connect_readonly()
    dsn = get(ENV, "BF_DB_URL", nothing)
    dsn === nothing && error("BF_DB_URL is not set; configure the read-only operational DB connection.")
    conn = LibPQ.Connection(dsn)
    LibPQ.execute(conn, "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
    return conn
end

function _raw_frames()
    conn = _connect_readonly()
    try
        matches = DataFrames.DataFrame(LibPQ.execute(conn, """
            SELECT m.match_id, m.tournament_id, s.year AS season, m.home_team, m.away_team,
                   m.home_score, m.away_score, m.start_timestamp, m.raw_data::text AS raw_match,
                   o.name AS referee_name, o.bbc_official_id AS referee_id
            FROM sofascore.matches AS m
            JOIN sofascore.seasons AS s ON s.season_id = m.season_id
            LEFT JOIN bbc.match_officials AS o
              ON o.match_id = m.match_id AND o.role = 'referee'
            WHERE m.status_type = 'finished' AND m.tournament_id = ANY(ARRAY[56, 57])
            ORDER BY m.start_timestamp, m.match_id
        """))
        length(unique(matches.match_id)) == DataFrames.nrow(matches) || error(
            "BBC referee LEFT JOIN duplicated fixtures; audit multiple role='referee' rows before extraction")
        incidents = DataFrames.DataFrame(LibPQ.execute(conn, """
            SELECT i.id AS incident_id, i.match_id, i.incident_type::text, i.time, i.added_time,
                   i.is_home, i.data::text AS raw_incident
            FROM sofascore.match_incidents AS i
            JOIN sofascore.matches AS m ON m.match_id = i.match_id
            WHERE m.status_type = 'finished' AND m.tournament_id = ANY(ARRAY[56, 57])
            ORDER BY i.match_id, i.time NULLS LAST, i.added_time NULLS LAST, i.id
        """))
        return matches, incidents
    finally
        try
            LibPQ.execute(conn, "ROLLBACK")
        finally
            close(conn)
        end
    end
end

function _incident_rows(raw::DataFrames.DataFrame)
    rows = NamedTuple[]
    for incident in eachrow(raw)
        incident_class = _text_json(incident.raw_incident, "incidentClass")
        home_cumulative = _int_json(incident.raw_incident, "homeScore")
        away_cumulative = _int_json(incident.raw_incident, "awayScore")
        reversed_period_time = _int_json(incident.raw_incident, "reversedPeriodTime")
        provider_incident_id = _provider_id_json(incident.raw_incident)
        player_id = _player_id_json(incident.raw_incident)
        rescinded = _bool_json(incident.raw_incident, "rescinded")
        incident_type = String(incident.incident_type)
        component = if rescinded
            "rescinded"
        elseif incident_type == "goal" && isequal(incident_class, "regular")
            # Provider label means non-penalty/non-own goal; it is not evidence of open play
            # because corners, free-kicks and other set pieces share this class.
            "non_penalty_non_own_goal"
        elseif incident_type == "goal" && isequal(incident_class, "penalty")
            "penalty_goal"
        elseif incident_type == "goal" && isequal(incident_class, "ownGoal")
            "own_goal"
        elseif incident_type == "inGamePenalty" && isequal(incident_class, "missed")
            "penalty_missed"
        elseif incident_type == "goal"
            "unclassified_goal"
        else
            "non_goal_event"
        end
        # Do not assume SofaScore's own-goal side convention. The recipient is resolved by the
        # embedded cumulative-score delta in `own_goal_orientation_audit`.
        push!(rows, (; incident_id = Int(incident.incident_id), match_id = Int(incident.match_id),
                     incident_type, incident_class, rescinded, provider_incident_id, player_id,
                     minute = ismissing(incident.time) ? missing : Int(incident.time),
                     reversed_period_time,
                     added_minute = ismissing(incident.added_time) ? missing : Int(incident.added_time),
                     is_home = ismissing(incident.is_home) ? missing : Bool(incident.is_home),
                     component, home_cumulative, away_cumulative))
    end
    return DataFrames.DataFrame(rows)
end

function own_goal_orientation_audit(matches::DataFrames.DataFrame, incidents::DataFrames.DataFrame)
    audit = NamedTuple[]
    for match_id in unique(incidents.match_id)
        match_incidents = DataFrames.subset(incidents, :match_id => DataFrames.ByRow(==(match_id)))
        home = 0
        away = 0
        for event in eachrow(match_incidents)
            event.incident_type == "goal" && !event.rescinded || continue
            old_home, old_away = home, away
            if !ismissing(event.home_cumulative) && !ismissing(event.away_cumulative)
                home, away = event.home_cumulative, event.away_cumulative
            end
            if event.component == "own_goal"
                delta_home, delta_away = home - old_home, away - old_away
                recipient = delta_home == 1 && delta_away == 0 ? "home" :
                            delta_home == 0 && delta_away == 1 ? "away" : "ambiguous"
                # Measured against the cumulative-score deltas: is_home is the recipient side
                # in this operational feed (contrary to the work-package claim).
                expected_recipient = ismissing(event.is_home) ? "unknown" : event.is_home ? "home" : "away"
                push!(audit, (; incident_id = event.incident_id, match_id, is_home = event.is_home,
                               recipient, expected_recipient,
                               agrees = recipient == expected_recipient))
            end
        end
    end
    return DataFrames.DataFrame(audit)
end

function _referee_string(row, column::Symbol)
    hasproperty(row, column) || return "UNKNOWN"
    value = row[column]
    (ismissing(value) || value === nothing) && return "UNKNOWN"
    text = strip(String(value))
    return isempty(text) ? "UNKNOWN" : text
end

function _match_registry(matches::DataFrames.DataFrame, incidents::DataFrames.DataFrame, orientation::DataFrames.DataFrame)
    rows = NamedTuple[]
    quarantines = NamedTuple[]
    orientation_by_id = Dict(row.incident_id => row for row in eachrow(orientation))
    for match in eachrow(matches)
        match_id = Int(match.match_id)
        events = DataFrames.subset(incidents, :match_id => DataFrames.ByRow(==(match_id)))
        own_rows = DataFrames.subset(events, :component => DataFrames.ByRow(==("own_goal")))
        own_ok = all(row -> haskey(orientation_by_id, row.incident_id) && orientation_by_id[row.incident_id].agrees,
                     eachrow(own_rows))
        # A 0-0 with no incident rows cannot distinguish a genuinely event-free fixture from a
        # collection failure. Do not promote it as a zero component observation.
        zero_score_without_feed = match.home_score == 0 && match.away_score == 0 && DataFrames.nrow(events) == 0
        unknown_side_missed_penalty = any(row -> row.component == "penalty_missed" && ismissing(row.is_home), eachrow(events))
        unclassified = count(==("unclassified_goal"), events.component)
        component_counts = Dict(name => 0 for name in ("non_penalty_non_own_goal", "penalty_goal", "own_home", "own_away", "penalty_awarded_home", "penalty_awarded_away", "penalty_missed_home", "penalty_missed_away"))
        for event in eachrow(events)
            side = ismissing(event.is_home) ? nothing : (event.is_home ? "home" : "away")
            if event.component == "non_penalty_non_own_goal" && side !== nothing
                component_counts["non_penalty_non_own_goal_$(side)"] = get(component_counts, "non_penalty_non_own_goal_$(side)", 0) + 1
            elseif event.component == "penalty_goal" && side !== nothing
                component_counts["penalty_awarded_$(side)"] += 1
            elseif event.component == "penalty_missed" && side !== nothing
                component_counts["penalty_awarded_$(side)"] += 1
                component_counts["penalty_missed_$(side)"] += 1
            elseif event.component == "own_goal" && haskey(orientation_by_id, event.incident_id)
                audit = orientation_by_id[event.incident_id]
                if audit.agrees
                    component_counts["own_$(audit.recipient)"] += 1
                end
            end
        end
        open_home = get(component_counts, "non_penalty_non_own_goal_home", 0)
        open_away = get(component_counts, "non_penalty_non_own_goal_away", 0)
        pen_home = component_counts["penalty_awarded_home"] - component_counts["penalty_missed_home"]
        pen_away = component_counts["penalty_awarded_away"] - component_counts["penalty_missed_away"]
        own_home, own_away = component_counts["own_home"], component_counts["own_away"]
        derived_home, derived_away = open_home + pen_home + own_home, open_away + pen_away + own_away
        final_home = ismissing(match.home_score) ? missing : Int(match.home_score)
        final_away = ismissing(match.away_score) ? missing : Int(match.away_score)
        score_match = !ismissing(final_home) && !ismissing(final_away) && derived_home == final_home && derived_away == final_away
        usable = score_match && own_ok && unclassified == 0 && !zero_score_without_feed && !unknown_side_missed_penalty
        reason = usable ? "" : join(filter(!isempty, [!score_match ? "component_total_mismatch" : "", !own_ok ? "own_goal_orientation_unresolved" : "", unclassified > 0 ? "unclassified_goal" : "", zero_score_without_feed ? "zero_score_without_incident_feed_evidence" : "", unknown_side_missed_penalty ? "missed_penalty_without_side" : ""]), ";")
        if !usable
            push!(quarantines, (; match_id, reason, final_home, final_away, derived_home, derived_away,
                                unclassified_goals = unclassified, own_orientation_ok = own_ok))
        end
        referee_name = _referee_string(match, :referee_name)
        referee_id = _referee_string(match, :referee_id)
        referee_present = referee_name != "UNKNOWN"
        sofascore_referee_present = _json_value(match.raw_match, "referee") !== nothing
        push!(rows, (; match_id, tournament_id = Int(match.tournament_id), season = String(match.season),
                     match_date = string(match.start_timestamp), home_team = String(match.home_team), away_team = String(match.away_team),
                     overall_home = final_home, overall_away = final_away,
                     non_penalty_non_own_goal_home = open_home, non_penalty_non_own_goal_away = open_away,
                     penalty_goal_home = pen_home, penalty_goal_away = pen_away,
                     penalty_awarded_home = component_counts["penalty_awarded_home"], penalty_awarded_away = component_counts["penalty_awarded_away"],
                     penalty_missed_home = component_counts["penalty_missed_home"], penalty_missed_away = component_counts["penalty_missed_away"],
                     own_goal_home = own_home, own_goal_away = own_away,
                     usable_for_components = usable, quarantine_reason = reason,
                     referee = referee_name, referee_name, referee_id, referee_present,
                     sofascore_referee_present))
    end
    return DataFrames.DataFrame(rows), DataFrames.DataFrame(quarantines)
end

"""Stable SHA-256 identity of the immutable match-component rows, including quarantines."""
function registry_snapshot_hash(registry::GoalComponentRegistry)
    io = IOBuffer()
    for row in eachrow(sort(registry.matches, :match_id))
        print(io, join(string.(Tuple(row)), '\u001f'), '\n')
    end
    for row in eachrow(sort(registry.quarantines, :match_id))
        print(io, join(string.(Tuple(row)), '\u001f'), '\n')
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

"""
    model_feature_view(registry, snapshot_hash) -> DataFrame

Return the immutable feature contract consumed by decomposition models. The supplied hash must
match `registry_snapshot_hash(registry)`, preventing a model from silently training against a
changed registry. Rows remain present for total-score fallback; component likelihoods must use
`component_usable_mask` and never turn a quarantined component into zero.
"""
function model_feature_view(registry::GoalComponentRegistry, snapshot_hash::AbstractString)
    observed_hash = registry_snapshot_hash(registry)
    snapshot_hash == observed_hash || error("registry snapshot hash mismatch: model received $(snapshot_hash), but registry is $(observed_hash)")
    source = registry.matches
    return DataFrames.DataFrame(
        match_id = source.match_id,
        referee = source.referee_name,
        referee_name = source.referee_name,
        referee_id = source.referee_id,
        overall_home = source.overall_home,
        overall_away = source.overall_away,
        flat_non_penalty_non_own_goal_home = source.non_penalty_non_own_goal_home,
        flat_non_penalty_non_own_goal_away = source.non_penalty_non_own_goal_away,
        flat_penalty_goals_home = source.penalty_goal_home,
        flat_penalty_goals_away = source.penalty_goal_away,
        flat_penalty_awarded_home = source.penalty_awarded_home,
        flat_penalty_awarded_away = source.penalty_awarded_away,
        flat_own_goals_credited_home = source.own_goal_home,
        flat_own_goals_credited_away = source.own_goal_away,
        component_usable_mask = source.usable_for_components,
        component_quarantine_reason = source.quarantine_reason,
    )
end

"""Extract the all-season, read-only component registry and retain every total-score row."""
function extract_registry()
    matches, raw_incidents = _raw_frames()
    incidents = _incident_rows(raw_incidents)
    orientation = own_goal_orientation_audit(matches, incidents)
    registry, quarantines = _match_registry(matches, incidents, orientation)
    return GoalComponentRegistry(registry, incidents, quarantines, string(Dates.now(Dates.UTC))), orientation
end

"""
    attempt_quality_audit(registry) -> (summary, flags)

Audit the provider's observable scored/missed penalty events before treating their sum as valid
attempts. It detects, but cannot resolve, same-minute possible retakes and duplicates. This feed
has no independently labelled award, shootout, VAR-cancellation, or retake field; absence of a
flag is therefore not proof of complete award coverage.
"""
function attempt_quality_audit(registry::GoalComponentRegistry)
    penalty_rows = DataFrames.subset(registry.incidents,
        :component => DataFrames.ByRow(x -> x == "penalty_goal" || x == "penalty_missed"))
    duplicate_db_ids = DataFrames.nrow(penalty_rows) - length(unique(penalty_rows.incident_id))
    provider_ids = DataFrames.dropmissing(penalty_rows, :provider_incident_id)
    duplicate_provider_ids = DataFrames.nrow(provider_ids) - length(unique(provider_ids.provider_incident_id))
    score_missing = count(ismissing, DataFrames.subset(penalty_rows, :component => DataFrames.ByRow(==("penalty_goal"))).home_cumulative)
    no_side = count(ismissing, penalty_rows.is_home)
    rescinded = count(identity, penalty_rows.rescinded)
    minute_missing = count(ismissing, penalty_rows.minute)
    beyond_120 = count(x -> !ismissing(x) && x > 120, penalty_rows.minute)
    group_columns = [:match_id, :minute, :added_minute, :is_home]
    candidates = DataFrames.combine(DataFrames.groupby(penalty_rows, group_columns),
                         DataFrames.nrow => :event_rows,
                         :incident_id => (x -> join(string.(x), ";")) => :db_incident_ids,
                         :provider_incident_id => (x -> join(string.(coalesce.(x, "missing")), ";")) => :provider_incident_ids,
                         :player_id => (x -> join(string.(coalesce.(x, "missing")), ";")) => :player_ids,
                         :component => (x -> join(x, ";")) => :components)
    same_stamp = DataFrames.subset(candidates, :event_rows => DataFrames.ByRow(>(1)))
    # A scored/missed pair at the same match-time-side is compatible with a retake, but the feed
    # offers no event-link field. Keep it a candidate, never collapse it automatically.
    retake_candidates = DataFrames.subset(same_stamp, :components => DataFrames.ByRow(x -> occursin("penalty_goal", x) && occursin("penalty_missed", x)))
    summary = DataFrames.DataFrame(metric = [
            "penalty_goal_rows", "penalty_missed_rows", "penalty_rows_total",
            "duplicate_database_incident_ids", "duplicate_provider_incident_ids",
            "penalty_goals_missing_cumulative_score", "penalty_rows_missing_side",
            "rescinded_penalty_rows", "penalty_rows_missing_minute", "penalty_rows_minute_gt_120",
            "same_match_time_side_candidates", "scored_missed_same_stamp_retake_candidates",
            "independent_award_event_available", "explicit_shootout_or_retake_marker_available",
            "explicit_var_cancellation_marker_available"],
        value = Any[
            count(==("penalty_goal"), penalty_rows.component), count(==("penalty_missed"), penalty_rows.component), DataFrames.nrow(penalty_rows),
            duplicate_db_ids, duplicate_provider_ids, score_missing, no_side, rescinded,
            minute_missing, beyond_120, DataFrames.nrow(same_stamp), DataFrames.nrow(retake_candidates),
            false, false, false])
    flags = vcat(DataFrames.transform(same_stamp, :components => DataFrames.ByRow(_ -> "same_match_time_side") => :flag),
                 DataFrames.transform(retake_candidates, :components => DataFrames.ByRow(_ -> "possible_retake_scored_missed_same_stamp") => :flag))
    return summary, flags
end

"""Referee exposure and observed penalty-attempt rates; raw and reconciled cohorts stay separate."""
function referee_rates(registry::GoalComponentRegistry; usable_only::Bool = true)
    rows = DataFrames.subset(registry.matches, :referee_present => DataFrames.ByRow(identity))
    if usable_only
        rows = DataFrames.subset(rows, :usable_for_components => DataFrames.ByRow(identity))
    end
    rows.penalty_attempts = rows.penalty_awarded_home .+ rows.penalty_awarded_away
    rates = DataFrames.combine(DataFrames.groupby(rows, [:referee_id, :referee_name]),
        DataFrames.nrow => :matches, :penalty_attempts => sum => :penalty_attempts)
    rates.attempts_per_match = rates.penalty_attempts ./ rates.matches
    rates.cohort = fill(usable_only ? "reconciled_components" : "all_finished_raw_incident_counts", DataFrames.nrow(rates))
    return sort(rates, :attempts_per_match)
end

"""
Exposure-offset referee-only Poisson deviance versus one pooled rate. This is unadjusted
association, not a causal test; the chi-square approximation is weak for very sparse officials.
"""
function referee_deviance_summary(registry::GoalComponentRegistry)
    output = NamedTuple[]
    for usable_only in (false, true)
        rates = referee_rates(registry; usable_only)
        for minimum_matches in (1, 20)
            selected = DataFrames.subset(rates, :matches => DataFrames.ByRow(>=(minimum_matches)))
            n_referees = DataFrames.nrow(selected)
            n_referees > 1 || error("too few referee groups for a deviance comparison")
            pooled = sum(selected.penalty_attempts) / sum(selected.matches)
            expected = selected.matches .* pooled
            deviance = 2sum(n == 0 ? e : n * log(n / e) - (n - e)
                            for (n, e) in zip(selected.penalty_attempts, expected))
            degrees_of_freedom = n_referees - 1
            push!(output, (; cohort = usable_only ? "reconciled_components" : "all_finished_raw_incident_counts",
                minimum_matches, referees = n_referees, matches = sum(selected.matches),
                attempts = sum(selected.penalty_attempts), deviance, degrees_of_freedom,
                asymptotic_p = Distributions.ccdf(Distributions.Chisq(degrees_of_freedom), deviance)))
        end
    end
    return DataFrames.DataFrame(output)
end

function component_summary(registry::GoalComponentRegistry)
    usable = DataFrames.subset(registry.matches, :usable_for_components => DataFrames.ByRow(identity))
    side_counts = vcat(usable.non_penalty_non_own_goal_home, usable.non_penalty_non_own_goal_away)
    penalty_goals = vcat(usable.penalty_goal_home, usable.penalty_goal_away)
    own_goals = vcat(usable.own_goal_home, usable.own_goal_away)
    return DataFrames.DataFrame(component = ["non_penalty_non_own_goal", "penalty_goal", "own_goal"],
        observations = fill(length(side_counts), 3),
        total = [sum(side_counts), sum(penalty_goals), sum(own_goals)],
        mean = [Statistics.mean(side_counts), Statistics.mean(penalty_goals), Statistics.mean(own_goals)],
        variance = [Statistics.var(side_counts), Statistics.var(penalty_goals), Statistics.var(own_goals)],
        vmr = [Statistics.var(side_counts) / Statistics.mean(side_counts), Statistics.var(penalty_goals) / Statistics.mean(penalty_goals), Statistics.var(own_goals) / Statistics.mean(own_goals)],
        zero_rate = [Statistics.mean(side_counts .== 0), Statistics.mean(penalty_goals .== 0), Statistics.mean(own_goals .== 0)])
end

"""
    poisson_homogeneity_permutation(team, counts, strata; draws, seed)

Exposure-controlled team-rate null. Counts are permuted only within the supplied strata (for
this EDA: side × tournament × season), retaining their side/division/season count distributions
and each team's observed fixture exposure. It is not a replacement for an adjusted Poisson GLM.
"""
function poisson_homogeneity_permutation(team::Vector{String}, counts::Vector{Int}, strata::Vector{String}; draws::Int = 10_000, seed::Int = 20260908)
    length(team) == length(counts) == length(strata) || error("team, counts and strata must have equal length")
    team_indices = [findall(==(name), team) for name in unique(team)]
    statistic(values) = Statistics.var([Statistics.mean(values[index]) for index in team_indices])
    strata_indices = [findall(==(name), strata) for name in unique(strata)]
    rng = Random.MersenneTwister(seed)
    observed = statistic(counts)
    shuffled = similar(counts)
    null = Vector{Float64}(undef, draws)
    for draw in eachindex(null)
        for indices in strata_indices
            shuffled[indices] = Random.shuffle(rng, counts[indices])
        end
        null[draw] = statistic(shuffled)
    end
    return (; observed, null_mean = Statistics.mean(null), p_value = (count(>=(observed), null) + 1) / (draws + 1))
end

"""
History-only attacking-pressure feature. Same-kickoff rows are all scored before that kickoff's
outcomes update history, enforcing training kickoff < prediction cutoff. `*_history_matches`
exposes cold starts rather than treating a zero-history team as genuinely zero pressure.
"""
function pressure_feature(registry::GoalComponentRegistry)
    ordered = sort(registry.matches, [:match_date, :match_id])
    cumulative = Dict{String, Int}()
    exposure = Dict{String, Int}()
    home_pressure = Union{Missing, Float64}[]
    away_pressure = Union{Missing, Float64}[]
    home_history_matches = Int[]
    away_history_matches = Int[]
    for date in unique(ordered.match_date)
        indices = findall(==(date), ordered.match_date)
        for index in indices
            row = ordered[index, :]
            home_n, away_n = get(exposure, row.home_team, 0), get(exposure, row.away_team, 0)
            push!(home_history_matches, home_n)
            push!(away_history_matches, away_n)
            push!(home_pressure, home_n == 0 ? missing : get(cumulative, row.home_team, 0) / home_n)
            push!(away_pressure, away_n == 0 ? missing : get(cumulative, row.away_team, 0) / away_n)
        end
        for index in indices
            row = ordered[index, :]
            row.usable_for_components || continue
            cumulative[row.home_team] = get(cumulative, row.home_team, 0) + row.non_penalty_non_own_goal_home
            cumulative[row.away_team] = get(cumulative, row.away_team, 0) + row.non_penalty_non_own_goal_away
            exposure[row.home_team] = get(exposure, row.home_team, 0) + 1
            exposure[row.away_team] = get(exposure, row.away_team, 0) + 1
        end
    end
    return DataFrames.DataFrame(match_id = ordered.match_id,
                                home_pressure = home_pressure,
                                away_pressure = away_pressure,
                                home_history_matches = home_history_matches,
                                away_history_matches = away_history_matches)
end

"""Persist the frozen registry and its human-readable audit files; returns its SHA-256 identity."""
function write_registry_artifacts(registry::GoalComponentRegistry, orientation::DataFrames.DataFrame, output_dir::AbstractString)
    mkpath(output_dir)
    snapshot_hash = registry_snapshot_hash(registry)
    CSV.write(joinpath(output_dir, "eda_match_component_registry.csv"), registry.matches)
    CSV.write(joinpath(output_dir, "eda_incident_audit.csv"), registry.incidents)
    CSV.write(joinpath(output_dir, "eda_quarantine.csv"), registry.quarantines)
    CSV.write(joinpath(output_dir, "eda_own_goal_orientation.csv"), orientation)
    attempt_summary, attempt_flags = attempt_quality_audit(registry)
    CSV.write(joinpath(output_dir, "eda_attempt_quality_summary.csv"), attempt_summary)
    CSV.write(joinpath(output_dir, "eda_attempt_quality_flags.csv"), attempt_flags)
    CSV.write(joinpath(output_dir, "eda_history_only_pressure.csv"), pressure_feature(registry))
    open(joinpath(output_dir, "eda_registry_snapshot.txt"), "w") do io
        println(io, snapshot_hash)
        println(io, "extraction_utc=", registry.extraction_utc)
        println(io, "format=GoalComponentRegistry/v1")
    end
    open(joinpath(output_dir, "eda_match_component_registry.jls"), "w") do io
        Serialization.serialize(io, registry)
    end
    return snapshot_hash
end

"""
Load the frozen registry and verify its hash. With `prefer_binary=false` (or no local binary),
reconstruct from the versioned CSVs: a fresh Git checkout needs no live SQL re-extraction and
never substitutes a newer snapshot. Binary deserialization is only for trusted local artifacts.
"""
function load_registry(output_dir::AbstractString; prefer_binary::Bool = true)
    registry_path = joinpath(output_dir, "eda_match_component_registry.jls")
    lines = readlines(joinpath(output_dir, "eda_registry_snapshot.txt"))
    isempty(lines) && error("registry snapshot file is empty")
    recorded_hash = strip(first(lines))
    registry = if prefer_binary && isfile(registry_path)
        open(Serialization.deserialize, registry_path)
    else
        matches = CSV.read(joinpath(output_dir, "eda_match_component_registry.csv"), DataFrames.DataFrame;
            stringtype = String, types = Dict(:match_date => String, :season => String,
                :home_team => String, :away_team => String, :referee => String,
                :referee_name => String, :referee_id => String), missingstring = nothing)
        incidents = CSV.read(joinpath(output_dir, "eda_incident_audit.csv"), DataFrames.DataFrame;
            stringtype = String, types = Dict(:provider_incident_id => String, :player_id => String))
        quarantines = CSV.read(joinpath(output_dir, "eda_quarantine.csv"), DataFrames.DataFrame;
            stringtype = String)
        stamp_line = findfirst(line -> startswith(line, "extraction_utc="), lines)
        stamp_line === nothing && error("frozen registry lacks its extraction timestamp")
        stamp = split(lines[stamp_line], '='; limit = 2)[2]
        GoalComponentRegistry(matches, incidents, quarantines, stamp)
    end
    registry isa GoalComponentRegistry || error("$(registry_path) does not contain GoalComponentRegistry/v1")
    recorded_hash == registry_snapshot_hash(registry) || error("frozen registry hash verification failed; do not substitute a newer snapshot")
    return registry, recorded_hash
end

end # module
