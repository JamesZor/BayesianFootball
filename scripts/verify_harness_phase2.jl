#!/usr/bin/env julia

# Deterministic Phase-2 acceptance check. No sampling and no database writes.
using BayesianFootball
using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "current_development", "grw_pyramid_cups", "l01_loader.jl"))

function verify_lower()
    raw = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
    scope = Data.DataScope(name = "lower")
    scoped = Data.apply_scope(raw, scope)
    splitter = Data.ScopedWalkForwardCV(scope)
    actual = Data.create_id_boundaries(scoped, splitter)
    expected_splitter = gph_splitter(["24/25", "25/26"])
    expected = Data.create_id_boundaries(raw, expected_splitter)
    length(actual) == length(expected) == 40 || error("lower fold count differs")
    for (left, right) in zip(actual, expected)
        left[1].history_match_ids == right[1].history_match_ids ||
            error("lower history IDs differ in fold $(left[1].fold_id)")
        left[1].target_match_ids == right[1].target_match_ids ||
            error("lower target IDs differ in fold $(left[1].fold_id)")
        Set(Data.get_next_matches(scoped, left, splitter).match_id) ==
            Set(Data.get_next_matches(raw, right, expected_splitter).match_id) ||
            error("lower held-out IDs differ in fold $(left[1].fold_id)")
    end
    heldout = union((Set(Int.(Data.get_next_matches(scoped, pair, splitter).match_id))
                     for pair in actual)...)
    length(heldout) == 710 || error("lower held-out count is $(length(heldout)), expected 710")
    println("LOWER_OK folds=40 unique_heldout=710")
end

function verify_pyramid()
    raw = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)
    scope = Data.DataScope(
        name = "spfl_cups", train_tournaments = [54, 55, 56, 57],
        cups = :senior_spfl_ties, target_tournaments = [56, 57],
        clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
        history_seasons = 2, dynamics_col = :match_biweek)
    scoped = Data.apply_scope(raw, scope)
    splitter = Data.ScopedWalkForwardCV(scope)
    boundaries = Data.create_id_boundaries(scoped, splitter)
    model = pcx_model("g2_grw_all_spfl_cups")
    features = BayesianFootball.Features.create_features(boundaries, scoped, model, splitter)

    tournament = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(scoped.matches))
    rows = NamedTuple[]
    for (index, ((boundary, meta), (feature_set, _))) in enumerate(zip(boundaries, features))
        ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
        tournaments = Int[tournament[id] for id in ids]
        heldout = Data.get_next_matches(scoped, (boundary, meta), splitter)
        push!(rows, (; fold = index, season = meta.target_season, n_train = length(ids),
            n_lower = count(in((56, 57)), tournaments),
            n_upper = count(in((54, 55)), tournaments),
            n_cup = count(in((73, 982, 1520)), tournaments),
            n_oos = count(in((56, 57)), heldout.tournament_id),
            n_target_steps = Int(feature_set.data[:n_target_steps]),
            n_teams = Int(feature_set.data[:n_teams])))
    end
    actual = DataFrame(rows)
    expected = CSV.read(joinpath(ROOT, "current_development", "grw_pyramid_cups",
        "results", "r02_widening_g2_grw_all_spfl_cups.csv"), DataFrame)
    actual == expected || error("scoped g2 widening report differs from r02")

    prototype = pcx_load_data(max_age_hours = 10_000)
    prototype_splitter = PyramidGRWCV(
        target_seasons = ["24/25", "25/26"],
        extra_tournaments = [54, 55, 73, 982, 1520])
    prototype_boundaries = Data.create_id_boundaries(prototype, prototype_splitter)
    prototype_features = BayesianFootball.Features.create_features(
        prototype_boundaries, prototype, model, prototype_splitter)
    fields = (:ordered_match_ids, :time_indices, :n_history_steps,
              :n_target_steps, :n_teams, :team_map)
    for (index, (left, right)) in enumerate(zip(features, prototype_features)), field in fields
        left[1].data[field] == right[1].data[field] ||
            error("feature $field differs in fold $index")
    end
    println("PYRAMID_OK folds=40 widening=exact features=exact cups=363")
    return raw
end

function verify_monitors(raw)
    scope = Data.DataScope(
        name = "spfl_monitor", train_tournaments = [54, 55, 56, 57],
        target_tournaments = [56, 57], monitor_tournaments = [54, 55],
        clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
        history_seasons = 2)
    ds = Data.apply_scope(raw, scope)
    splitter = Data.ScopedWalkForwardCV(scope)
    boundaries = Data.create_id_boundaries(ds, splitter)
    close = BayesianFootball.Harness._betfair_closing_odds(ds)
    covered = Set(Int.(close.match_id))
    target = Set{Int}()
    totals = Dict(54 => [0, 0], 55 => [0, 0])
    for (index, pair) in enumerate(boundaries)
        heldout = Data.get_next_matches(ds, pair, splitter)
        target_ids = Int.(heldout.match_id[in.(heldout.tournament_id, Ref([56, 57]))])
        union!(target, target_ids)
        counts = String[]
        for tournament in (54, 55)
            ids = Int.(heldout.match_id[heldout.tournament_id .== tournament])
            totals[tournament] .+= [length(ids), count(in(covered), ids)]
            push!(counts, "t$tournament=$(length(ids))/$(count(in(covered), ids))")
        end
        println("MONITOR_FOLD fold=$index season=$(pair[2].target_season) " * join(counts, " "))
    end
    length(target) == 710 || error("monitor scope changed target IDs")
    println("MONITOR_OK target=710 t54=$(join(totals[54], '/')) t55=$(join(totals[55], '/'))")
end

verify_lower()
raw = verify_pyramid()
verify_monitors(raw)
