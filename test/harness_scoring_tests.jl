using Test
using BayesianFootball
using BayesianFootball: Data, CountLatents
using DataFrames
using Dates
using Distributions
using Random
using Statistics

const H = BayesianFootball.Harness

function harness_store(matches)
    empty = DataFrame()
    return Data.DataStore(Data.ScottishAll(), matches, empty, empty, empty, empty,
                          empty, empty, empty)
end

function add_fixture!(rows, id, date, season, tier, home, away)
    push!(rows, (; match_id = id, match_date = date, season = season,
                  tournament_id = tier, home_team = home, away_team = away,
                  home_score = 1, away_score = 0))
end

function toy_transition_store()
    rows = NamedTuple[]
    id = 1
    # Previous-season memberships establish all four transition directions.
    for (tier, club, opponent) in ((55, "R", "R0"), (57, "P", "P0"),
                                    (56, "D", "D0"), (54, "A", "A0"))
        add_fixture!(rows, id, Date(2023, 2, id), "22/23", tier, club, opponent)
        id += 1
    end
    # Intermediate season: R relegates into L1, P promotes into L1, D drops to L2,
    # A changes 54 -> 55 and therefore belongs only to :any.
    current = ((56, "R", "R1"), (56, "P", "P1"),
               (57, "D", "D1"), (55, "A", "A1"))
    current_ids = Dict{String,Vector{Int}}()
    for (tier, club, opponent) in current
        current_ids[club] = Int[]
        for n in 1:12
            push!(current_ids[club], id)
            add_fixture!(rows, id, Date(2024, 1, 1) + Day(id), "23/24",
                         tier, club, opponent)
            id += 1
        end
    end
    # Third season keeps every club in place, proving the previous-season comparison
    # does not label all fixtures of a historically mobile club.
    for (tier, club, opponent) in current
        add_fixture!(rows, id, Date(2025, 1, 1) + Day(id), "24/25",
                     tier, club, opponent)
        id += 1
    end
    return harness_store(DataFrame(rows)), current_ids
end

@testset "Harness transition cohorts" begin
    ds, current = toy_transition_store()
    tiers = H.club_season_tiers(ds)
    panel = collect(Iterators.flatten(values(current)))

    relegated = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :relegated_into_L1)
    promoted = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :promoted_into_L1)
    l1_l2 = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :l1_l2)
    any_move = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :any)

    @test relegated.match_ids == Set(current["R"][1:10])
    @test promoted.match_ids == Set(current["P"][1:10])
    @test l1_l2.match_ids == union(Set(current["P"][1:10]), Set(current["D"][1:10]))
    @test any_move.match_ids == union((Set(current[c][1:10]) for c in ("R", "P", "D", "A"))...)
    @test !(current["R"][11] in relegated.match_ids)
    @test H.transition_cohort(ds, panel, tiers;
        first_n = 20, direction = :relegated_into_L1).match_ids == Set(current["R"])
    @test all(relegated.clubs[m] == ["R"] for m in relegated.match_ids)
    @test_throws ArgumentError H.transition_cohort(ds, panel, tiers;
        first_n = 9, direction = :any)
end

@testset "Harness fixture-clustered bootstrap" begin
    rng = MersenneTwister(9)
    a = DataFrame(match_id = Int[], selection = Symbol[], family = String[],
                  ll_model = Float64[], ll_market = Float64[])
    b = similar(a, 0)
    fixture_effects = repeat([-0.9, 1.1], 20) # known row-weighted mean = 0.1
    selections = (:home, :draw, :away, :over_25)
    for (mid, effect) in enumerate(fixture_effects), selection in selections
        push!(a, (mid, selection, selection === :over_25 ? "OU2.5" : "1X2",
                  1.5 + effect, 1.0))
        push!(b, (mid, selection, selection === :over_25 ? "OU2.5" : "1X2",
                  1.5, 1.0))
    end
    clustered = H._paired_bootstrap(a, b; B = 5_000, seed = 2)
    @test clustered.delta ≈ 0.1 atol = 1e-12
    @test clustered.lo <= 0.1 <= clustered.hi

    d = a.ll_model .- b.ll_model
    row_stats = [mean(rand(rng, d, length(d))) for _ in 1:5_000]
    row_width = quantile(row_stats, 0.975) - quantile(row_stats, 0.025)
    @test clustered.hi - clustered.lo > row_width
end

function poisson_1x2(lambda_home, lambda_away; max_goals = 24)
    home = draw = away = 0.0
    for h in 0:max_goals, a in 0:max_goals
        p = pdf(Poisson(lambda_home), h) * pdf(Poisson(lambda_away), a)
        h > a ? (home += p) : h == a ? (draw += p) : (away += p)
    end
    total = home + draw + away
    return (home / total, draw / total, away / total)
end

@testset "Harness compression slope convention" begin
    model_sup = [-0.30, -0.10, 0.10, 0.30, 0.50]
    market_sup = 2.0 .* model_sup
    ids = collect(101:105)
    draws = 30
    lambda_home = reduce(vcat, [fill(exp(s / 2), 1, draws) for s in model_sup])
    lambda_away = reduce(vcat, [fill(exp(-s / 2), 1, draws) for s in model_sup])
    latents = CountLatents(ids, lambda_home, lambda_away)

    odds = DataFrame(match_id = Int[], market_name = String[], market_line = Float64[],
                     selection = Symbol[], prob_fair_close = Float64[])
    for (id, supremacy) in zip(ids, market_sup)
        p = poisson_1x2(exp(supremacy / 2), exp(-supremacy / 2))
        for (selection, probability) in zip((:home, :draw, :away), p)
            push!(odds, (id, "1x2", 0.0, selection, probability))
        end
    end
    slopes = H.compression_slopes(latents, odds, ids)
    @test slopes.n_fixtures == length(ids)
    @test slopes.compression_slope > 1.0
    @test slopes.compression_slope ≈ 2.0 atol = 0.15
end

@testset "Harness per-subset score row counts" begin
    frame = DataFrame(match_id = repeat([1, 2], inner = 7),
        selection = repeat([:home, :draw, :away, :over_25, :under_25,
                            :btts_yes, :btts_no], 2),
        family = repeat(["1X2", "1X2", "1X2", "OU2.5", "OU2.5", "BTTS", "BTTS"], 2),
        p_model = fill(0.4, 14), p_market = fill(0.4, 14),
        y = repeat([1.0, 0, 0, 1, 0, 1, 0], 2))
    frame.ll_model = BayesianFootball.Evaluation.calc_logloss.(frame.p_model, frame.y)
    frame.ll_market = copy(frame.ll_model)
    @test H._observation_scores(frame, "all").n_obs == 14
    @test H._observation_scores(frame, "1X2").n_obs == 6
    @test H._observation_scores(frame, "OU2.5").n_obs == 4
    @test H._observation_scores(frame, "BTTS").n_obs == 4
    @test H._observation_scores(frame[in.(frame.match_id, Ref(Set([1]))), :], "all").n_fixtures == 1
end
