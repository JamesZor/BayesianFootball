using Test
using BayesianFootball
using BayesianFootball: Data, Harness, Portfolio
using DataFrames, Dates, Random, UUIDs

function hp_latents(ids; seed = 41, draws = 80)
    rng = Xoshiro(seed)
    n = length(ids)
    return CountLatents(Int.(ids), 0.8 .+ rand(rng, n, draws), 0.7 .+ rand(rng, n, draws))
end

function hp_odds(ids; omit = Set{Int}())
    rows = NamedTuple[]
    for id in ids
        id in omit && continue
        for (market, line, selections, probabilities) in (
            ("1X2", 0.0, (:home, :draw, :away), (0.44, 0.27, 0.29)),
            ("OverUnder", 2.5, (:over_25, :under_25), (0.49, 0.51)),
            ("BTTS", 0.0, (:btts_yes, :btts_no), (0.52, 0.48)))
            for (selection, probability) in zip(selections, probabilities)
                push!(rows, (; match_id = Int(id), market_name = market,
                               market_line = line, selection,
                               odds_close = 1.0 / (1.04 * probability)))
            end
        end
    end
    return DataFrame(rows)
end

function hp_fixtures(ids)
    return Dict(Int(id) => (date = Date(2025, 1, 1) + Day(div(i - 1, 2)),
                            score = (i % 3, (i + 1) % 3))
        for (i, id) in enumerate(ids))
end

function hp_contract()
    spec = Portfolio.BookSpec(
        markets = Data.MarketConfig(Data.AbstractMarket[
            Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()]),
        price = Portfolio.DeArb(),
        allocator = Portfolio.KellyLogUtility(),
        shrink = Portfolio.NoShrinkage(),
        exec = Portfolio.ExecutionConfig(
            commission = Portfolio.PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.0001))
    policy = Portfolio.PolicySpec(
        trust = Portfolio.FlatTrust(1.0),
        risk = Portfolio.SlateDrawdown(12.0),
        cap = Portfolio.FixedCap(0.25),
        grouping = Portfolio.DailySlate())
    return spec, policy
end

@testset "Harness finalist common buildable panel" begin
    latent_panels = Dict(:a => [8101, 8102, 8103, 8104, 8105],
                         :b => [8102, 8103, 8104, 8105, 8106])
    @test Harness._portfolio_requested_panel(latent_panels, :common) ==
          [8102, 8103, 8104, 8105]

    ids = collect(8101:8105)
    spec, _ = hp_contract()
    sources = Dict("a" => hp_latents(ids; seed = 1),
                   "b" => hp_latents(ids; seed = 2))
    keep, dropped = Harness._portfolio_buildable_panel(
        spec, sources, hp_odds(ids; omit = Set([8103])), hp_fixtures(ids), ids)
    @test keep == [8101, 8102, 8104, 8105]
    @test dropped.match_id == [8103]
    @test occursin("no quotes", only(dropped.reason))
end

@testset "Harness finalist refuses a skipped common-panel fixture" begin
    ids = collect(8201:8204)
    spec, policy = hp_contract()
    @test_throws ErrorException Harness._portfolio_simulate(
        spec, policy, hp_latents(ids), hp_odds(ids; omit = Set([8204])),
        hp_fixtures(ids), ids; label = "synthetic", B = 20, seed = 7)
end

@testset "Harness finalist summary is seeded and deterministic" begin
    ids = collect(8301:8312)
    spec, policy = hp_contract()
    latents = hp_latents(ids; seed = 9)
    odds = hp_odds(ids)
    fixtures = hp_fixtures(ids)
    first_result = Harness._portfolio_simulate(
        spec, policy, latents, odds, fixtures, ids; label = "fixed", B = 100, seed = 19)
    second_result = Harness._portfolio_simulate(
        spec, policy, latents, odds, fixtures, ids; label = "fixed", B = 100, seed = 19)
    ref = Harness.RunRef("fixed", "synthetic", UUID("00000000-0000-0000-0000-000000000019"), :control)
    first_row = Harness._portfolio_summary_row(ref, first_result; n_panel = length(ids))
    second_row = Harness._portfolio_summary_row(ref, second_result; n_panel = length(ids))
    @test first_row == second_row
    @test first_row.n_panel == 12
    @test first_row.n_slates == 6
    @test isfinite(first_row.total_return_pct)
    @test first_row.growth_lo == second_row.growth_lo
    @test first_row.growth_hi == second_row.growth_hi
end

@testset "Harness finalist rows persist with finalist stage" begin
    control_id = UUID("00000000-0000-0000-0000-000000000030")
    summary = DataFrame([(
        run_id = control_id, model = "control", total_return_pct = 10.0,
        cagr_pct = 11.0, growth_per_slate = 0.01, growth_lo = -0.01,
        growth_hi = 0.03, roi_pct = 4.0, p_roi_positive = 0.9,
        sharpe_ann = 1.2, calmar = 0.8, max_drawdown_pct = -12.0,
        win_rate_pct = 40.0, mean_edge_pp = 2.0, mean_exposure = 0.1,
        capture_ratio = 1.1, n_bets = 20, n_slates = 5, n_panel = 8)])
    rows = Harness._portfolio_score_rows(summary, :close_option_b, control_id)
    store = Harness.InMemoryScoreStore()
    Harness.write_scores!(store, rows)
    @test nrow(store.scores) == 15
    @test all(==("finalist"), store.scores.stage)
    @test all(==("portfolio_close_option_b"), store.scores.subset)
    @test all(==("book"), store.scores.market)
    @test all(==(control_id), store.scores.control_run_id)
    growth = filter(:metric => ==("growth_per_slate"), store.scores)
    @test only(growth.lo) == -0.01
    @test only(growth.hi) == 0.03
end
