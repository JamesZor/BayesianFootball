using Test
using BayesianFootball
using DataFrames
using Dates

const ScopedData = BayesianFootball.Data
const ScopedFeatures = BayesianFootball.Features

struct ScopedClockProbe <: BayesianFootball.AbstractFootballModel end
ScopedFeatures.required_features(::ScopedClockProbe) = ScopedFeatures.AbstractFeatureConfig[
    ScopedFeatures.TimeIndicesFeature(),
]

function scoped_store(matches; segment = ScopedData.ScottishPyramid())
    empty = DataFrame()
    return ScopedData.DataStore(segment, matches, empty, empty, empty, empty,
                                empty, empty, empty)
end

function scoped_fixture_rows()
    rows = NamedTuple[]
    add(id, tournament, season, date, hour, home, away; neutral = false) = push!(rows, (;
        match_id = id, tournament_id = tournament, season, match_date = Date(date),
        match_hour = hour, match_week = 1, match_biweek = 1, match_month = 1,
        home_team = home, away_team = away, home_score = 1, away_score = 0,
        neutral_venue = neutral))

    # League rows establish point-in-time memberships in the history and target seasons.
    add(1, 56, "23/24", "2024-05-01", 15, "l1a", "l1b")
    add(2, 57, "23/24", "2024-05-01", 15, "l2a", "l2b")
    add(3, 54, "23/24", "2024-05-01", 15, "upa", "upb")
    add(4, 55, "23/24", "2024-05-01", 15, "cha", "chb")
    for (offset, date) in enumerate(("2024-08-03", "2024-08-17", "2024-08-31"))
        add(10 + offset, 56, "24/25", date, 15, "l1a", "l1b")
        add(20 + offset, 57, "24/25", date, 15, "l2a", "l2b")
        add(30 + offset, 54, "24/25", Date(date) + Day(1), 15, "upa", "upb")
        add(40 + offset, 55, "24/25", Date(date) + Day(1), 16, "cha", "chb")
    end

    # Raw cup season labels deliberately use calendar years.
    add(100, 73, "2024", "2024-08-10", 15, "l1a", "upa")       # accepted
    add(101, 982, "2024", "2024-08-10", 16, "l2a", "cha";
        neutral = true)                                             # neutral
    add(102, 1520, "2024", "2024-08-10", 17, "l1a", "club-b") # B side
    add(103, 1520, "2024", "2024-08-10", 18, "l1a", "guest")  # guest
    add(104, 73, "2024", "2024-08-10", 19, "l1a", "nonleague")
    # A scoreability exclusion from the canonical CSV remains excluded even when the
    # DataFrame also carries a native neutral_venue column.
    add(9724815, 1520, "2024", "2024-08-10", 20, "l1a", "upa")
    return DataFrame(rows)
end

@testset "DataScope validation and copy semantics" begin
    @test_throws ArgumentError ScopedData.DataScope(
        name = "bad", train_tournaments = [56], target_tournaments = [56, 57])
    @test_throws ArgumentError ScopedData.DataScope(
        name = "bad", clock_tournaments = [54])
    @test_throws ArgumentError ScopedData.DataScope(name = "bad", cups = :all)
    @test_throws ArgumentError ScopedData.DataScope(
        name = "bad", monitor_tournaments = [56])
    @test_throws ArgumentError ScopedData.DataScope(name = "bad", history_seasons = -1)

    ds = scoped_store(scoped_fixture_rows())
    scope = ScopedData.DataScope(
        name = "spfl_cups", train_tournaments = [54, 55, 56, 57],
        target_tournaments = [56, 57], monitor_tournaments = [54, 55],
        clock_tournaments = [56, 57], target_seasons = ["24/25"],
        history_seasons = 1, cups = :senior_spfl_ties)
    no_cups = ScopedData.apply_scope(ds, ScopedData.DataScope(
        name = "lower", target_seasons = ["24/25"], history_seasons = 1))
    @test isempty(intersect(Set(no_cups.matches.tournament_id), Set([73, 982, 1520])))

    scoped = ScopedData.apply_scope(ds, scope)
    @test scoped !== ds
    @test scoped.matches !== ds.matches
    @test ds.matches.season[ds.matches.match_id .== 100] == ["2024"]
    @test scoped.matches.season[scoped.matches.match_id .== 100] == ["24/25"]
    @test Set(scoped.matches.match_id[in.(scoped.matches.tournament_id,
                                          Ref([73, 982, 1520]))]) == Set([100])
    @test all(in([54, 55, 56, 57, 73, 982, 1520]), scoped.matches.tournament_id)
end

@testset "Scoped walk-forward widening, monitors, and clock" begin
    raw = scoped_store(scoped_fixture_rows())
    scope = ScopedData.DataScope(
        name = "spfl_cups", train_tournaments = [54, 55, 56, 57],
        cups = :senior_spfl_ties, target_tournaments = [56, 57],
        monitor_tournaments = [54, 55], clock_tournaments = [56, 57],
        target_seasons = ["24/25"], history_seasons = 1,
        dynamics_col = :match_biweek)
    ds = ScopedData.apply_scope(raw, scope)
    splitter = ScopedData.ScopedWalkForwardCV(scope)
    boundaries = ScopedData.create_id_boundaries(ds, splitter)
    @test length(boundaries) == 3

    kickoff = Dict(Int(row.match_id) => DateTime(row.match_date) + Hour(row.match_hour)
                   for row in eachrow(ds.matches))
    for pair in boundaries
        boundary, _ = pair
        heldout = ScopedData.get_next_matches(ds, pair, splitter)
        fitted = vcat(boundary.history_match_ids, boundary.target_match_ids)
        @test isempty(intersect(Set(fitted), Set(Int.(heldout.match_id))))
        @test maximum(kickoff[id] for id in fitted) <
              minimum(kickoff[Int(id)] for id in heldout.match_id)
        @test Set(heldout.tournament_id) == Set([54, 55, 56, 57])
    end
    # The accepted cup tie widens target-season training only after it has happened.
    @test !(100 in boundaries[1][1].target_match_ids)
    @test 100 in boundaries[2][1].target_match_ids

    features = ScopedFeatures.create_features(
        boundaries, ds, ScopedClockProbe(), splitter)
    for ((boundary, meta), (feature_set, feature_meta)) in zip(boundaries, features)
        @test feature_meta === meta
        @test Set(feature_set.data[:ordered_match_ids]) ==
              Set(vcat(boundary.history_match_ids, boundary.target_match_ids))
        @test length(feature_set.data[:ordered_match_ids]) ==
              length(boundary.history_match_ids) + length(boundary.target_match_ids)
        canonical = ScopedData._effective_step_map(
            ds.matches, scope.clock_tournaments, meta.target_season, scope.dynamics_col)
        effective = feature_set.data[:effective_target_steps]
        for id in boundary.target_match_ids
            tournament = only(ds.matches.tournament_id[ds.matches.match_id .== id])
            tournament in scope.clock_tournaments || continue
            @test effective[id] == canonical[id]
        end
        @test feature_set.data[:n_rounds] ==
              feature_set.data[:n_history_steps] + feature_set.data[:n_target_steps]
    end
end

@testset "Monitor-only tournaments are never fitted" begin
    raw = scoped_store(scoped_fixture_rows())
    scope = ScopedData.DataScope(
        name = "monitor_only", train_tournaments = [56, 57],
        target_tournaments = [56, 57], monitor_tournaments = [54, 55],
        clock_tournaments = [56, 57], target_seasons = ["24/25"],
        history_seasons = 1)
    ds = ScopedData.apply_scope(raw, scope)
    splitter = ScopedData.ScopedWalkForwardCV(scope)
    boundaries = ScopedData.create_id_boundaries(ds, splitter)
    tournament_by_id = Dict(Int(row.match_id) => Int(row.tournament_id)
                            for row in eachrow(ds.matches))
    seen_monitors = Set{Int}()
    for pair in boundaries
        fitted = vcat(pair[1].history_match_ids, pair[1].target_match_ids)
        @test all(tournament_by_id[id] in (56, 57) for id in fitted)
        heldout = ScopedData.get_next_matches(ds, pair, splitter)
        union!(seen_monitors, Int.(heldout.match_id[in.(heldout.tournament_id,
                                                       Ref([54, 55]))]))
    end
    @test seen_monitors == Set(vcat(collect(31:33), collect(41:43)))
end

@testset "Lower scope is the canonical grouped walk" begin
    matches = filter(row -> Int(row.tournament_id) in (56, 57), scoped_fixture_rows())
    ds = scoped_store(matches; segment = ScopedData.ScottishLower())
    scope = ScopedData.DataScope(name = "lower", target_seasons = ["24/25"],
                                 history_seasons = 1)
    scoped = ScopedData.ScopedWalkForwardCV(scope)
    canonical = ScopedData.GroupedCVConfig(
        tournament_groups = [[56, 57]], target_seasons = ["24/25"],
        history_seasons = 1, dynamics_col = :match_biweek,
        warmup_period = 0, stop_early = true)
    left = ScopedData.create_id_boundaries(ds, scoped)
    right = ScopedData.create_id_boundaries(ds, canonical)
    @test [(b.history_match_ids, b.target_match_ids) for (b, _) in left] ==
          [(b.history_match_ids, b.target_match_ids) for (b, _) in right]
    @test [Set(ScopedData.get_next_matches(ds, pair, scoped).match_id) for pair in left] ==
          [Set(ScopedData.get_next_matches(ds, pair, canonical).match_id) for pair in right]
end
