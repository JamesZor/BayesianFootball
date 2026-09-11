# Pure incident-contract tests. No SQL, sampling or production state is touched.
# Usage: julia --project -t 8 experiments/scottish_lower/08_goal_decomposition/test08_incident_contract.jl
using Test
import DataFrames
import Dates

if !isdefined(@__MODULE__, :GoalDecompositionIncidentData)
    include(joinpath(@__DIR__, "l08_incident_data.jl"))
end
const GD08_TEST_DATA = GoalDecompositionIncidentData

function gd08_test_match(id; home = 1, away = 1,
                         referee_name = "Synthetic Referee",
                         referee_id = "synthetic-referee-id")
    return (; match_id = id, tournament_id = 56, season = "24/25",
            home_team = "Synthetic Home", away_team = "Synthetic Away",
            home_score = home, away_score = away,
            start_timestamp = Dates.DateTime(2024, 8, 1), raw_match = "{}",
            referee_name, referee_id)
end

function gd08_test_incident(id, match_id, kind, side, minute, raw)
    return (; incident_id = id, match_id, incident_type = kind,
            time = minute, added_time = missing, is_home = side,
            raw_incident = raw)
end

@testset "Synthetic incident attribution and missing feeds" begin
    matches = DataFrames.DataFrame([
        gd08_test_match(1), gd08_test_match(2; home = 0, away = 0),
    ])
    raw = DataFrames.DataFrame([
        gd08_test_incident(1, 1, "goal", false, 12,
            "{\"incidentClass\":\"regular\",\"homeScore\":0,\"awayScore\":1}"),
        gd08_test_incident(2, 1, "goal", true, 24,
            "{\"incidentClass\":\"ownGoal\",\"homeScore\":1,\"awayScore\":1}"),
        gd08_test_incident(3, 1, "inGamePenalty", true, 30,
            "{\"incidentClass\":\"missed\"}"),
    ])
    incidents = GD08_TEST_DATA._incident_rows(raw)
    orientation = GD08_TEST_DATA.own_goal_orientation_audit(matches, incidents)
    components, quarantine = GD08_TEST_DATA._match_registry(matches, incidents, orientation)
    @test only(orientation.recipient) == "home"
    @test only(orientation.agrees)
    @test components.usable_for_components == [true, false]
    @test components.referee_name[1] == "Synthetic Referee"
    @test components.referee_id[1] == "synthetic-referee-id"
    @test components.own_goal_home[1] == 1
    @test components.own_goal_away[1] == 0
    @test components.non_penalty_non_own_goal_away[1] == 1
    @test components.penalty_awarded_home[1] == 1
    @test components.penalty_goal_home[1] == 0
    @test components.penalty_missed_home[1] == 1
    @test occursin("zero_score_without", only(quarantine.reason))

    registry = GD08_TEST_DATA.GoalComponentRegistry(components, incidents, quarantine, "synthetic")
    identity = GD08_TEST_DATA.registry_snapshot_hash(registry)
    @test length(identity) == 64
    view = GD08_TEST_DATA.model_feature_view(registry, identity)
    @test DataFrames.nrow(view) == 2  # fallback row must survive
    @test view.component_usable_mask == [true, false]
    @test_throws ErrorException GD08_TEST_DATA.model_feature_view(registry, repeat("0", 64))
    changed = deepcopy(registry)
    changed.matches.penalty_missed_home[1] += 1
    @test GD08_TEST_DATA.registry_snapshot_hash(changed) != identity
    @test registry.matches.penalty_missed_home[1] == 1
end

@testset "Unclassified null goal is quarantined, not imputed" begin
    matches = DataFrames.DataFrame([gd08_test_match(3; home = 1, away = 0)])
    raw = DataFrames.DataFrame([
        gd08_test_incident(4, 3, "goal", true, 12,
            "{\"incidentClass\":null,\"homeScore\":1,\"awayScore\":0}"),
    ])
    incidents = GD08_TEST_DATA._incident_rows(raw)
    @test only(incidents.component) == "unclassified_goal"
    orientation = GD08_TEST_DATA.own_goal_orientation_audit(matches, incidents)
    components, quarantine = GD08_TEST_DATA._match_registry(matches, incidents, orientation)
    @test !only(components.usable_for_components)
    @test occursin("unclassified", only(quarantine.reason))
end

@testset "Unknown-side penalty attempt cannot silently disappear" begin
    matches = DataFrames.DataFrame([gd08_test_match(4; home = 0, away = 0)])
    raw = DataFrames.DataFrame([
        gd08_test_incident(5, 4, "inGamePenalty", missing, 12,
            "{\"incidentClass\":\"missed\"}"),
    ])
    incidents = GD08_TEST_DATA._incident_rows(raw)
    orientation = GD08_TEST_DATA.own_goal_orientation_audit(matches, incidents)
    components, _ = GD08_TEST_DATA._match_registry(matches, incidents, orientation)
    @test !only(components.usable_for_components)
end

@testset "Frozen actual registry conservation" begin
    output_dir = joinpath(@__DIR__, "results")
    registry, identity = GD08_TEST_DATA.load_registry(output_dir)
    @test length(identity) == 64
    csv_registry, csv_identity = GD08_TEST_DATA.load_registry(output_dir; prefer_binary = false)
    @test csv_identity == identity
    @test isequal(csv_registry.matches, registry.matches)
    @test isequal(csv_registry.quarantines, registry.quarantines)
    @test length(unique(registry.matches.match_id)) == DataFrames.nrow(registry.matches)
    @test :referee_name in propertynames(registry.matches)
    @test :referee_id in propertynames(registry.matches)
    named = map(registry.matches.referee_name) do name
        !ismissing(name) && !isempty(strip(name)) && name != "UNKNOWN"
    end
    @test count(named) > 0  # catches a regression back to the empty SofaScore JSON key
    @test count(named) <= DataFrames.nrow(registry.matches)
    referee_test = GD08_TEST_DATA.referee_deviance_summary(registry)
    @test DataFrames.nrow(referee_test) == 4
    @test all(referee_test.degrees_of_freedom .== referee_test.referees .- 1)
    @test all(isfinite, referee_test.deviance)
    @test all(p -> 0.0 <= p <= 1.0, referee_test.asymptotic_p)
    usable = DataFrames.subset(registry.matches, :usable_for_components => DataFrames.ByRow(Base.identity))
    for side in ("home", "away")
        regular = usable[!, "non_penalty_non_own_goal_" * side]
        converted = usable[!, "penalty_goal_" * side]
        own = usable[!, "own_goal_" * side]
        awarded = usable[!, "penalty_awarded_" * side]
        missed = usable[!, "penalty_missed_" * side]
        @test all(regular .+ converted .+ own .== usable[!, "overall_" * side])
        @test all(converted .+ missed .== awarded)
        @test all(awarded .>= 0)
        @test all(converted .>= 0)
        @test all(own .>= 0)
    end
    @test all(isempty, usable.quarantine_reason)
    @test count(.!registry.matches.usable_for_components) == DataFrames.nrow(registry.quarantines)
end
