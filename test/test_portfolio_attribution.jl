# test/test_portfolio_attribution.jl
#
# Synthetic, deterministic tests for confidence/capture, ledger partitioning, controlled sizing,
# breakdowns, dispatch, and reporting. No database, cache, or model fit is required.

using Test
using BayesianFootball
using DataFrames, Dates

const PAT = BayesianFootball.Portfolio

function pat_bets(; match_id = [1, 2, 3, 4],
                   family = ["1X2_home", "1X2_draw", "O/U 2.5_over_25", "BTTS_yes"],
                   selection = [:home, :draw, :over_25, :btts_yes],
                   odds = [1.8, 2.5, 3.0, 4.0],
                   stake = [1.0, 3.0, 2.0, 4.0],
                   payoff = [0.8, 1.5, -1.0, -1.0],
                   p_model = [0.60, 0.50, 0.35, 0.30],
                   p_market = [0.40, 0.40, 0.30, 0.20])
    return DataFrame(
        match_id = match_id,
        date = fill(Date(2026, 1, 1), length(match_id)),
        family = family,
        selection = selection,
        odds = odds,
        stake = stake,
        pnl = Float64.(stake) .* Float64.(payoff),
        payoff = payoff,
        p_model = p_model,
        p_market = p_market,
    )
end

function pat_trajectory(bets)
    return Trajectory([1.0], Date[], Float64[], Float64[], Float64[], 0,
                      sum(bets.stake), sum(bets.pnl), DataFrame(bets))
end

function pat_result(bets)
    trajectory = pat_trajectory(bets)
    summary = portfolio_summary(PAT.DailyState[], trajectory, 1.0)
    return PortfolioResult(PAT.DailyState[], summary, NamedTuple(), nothing, trajectory,
                           DataFrame(), nothing, String[])
end

@testset "Portfolio attribution" begin
    @testset "Edge summary and capture ratio" begin
        mixed = pat_bets()
        summary = edge_summary(mixed)
        @test summary isa EdgeSummary
        @test summary.n_bets == 4
        @test summary.n_wins == 2
        @test summary.win_rate == 0.5
        @test summary.cap_weighted_win_rate == 0.4
        @test summary.stake_sum == 10.0
        @test summary.pnl_sum ≈ -0.7
        @test summary.roi ≈ -7.0
        @test summary.edge_mean ≈ 11.25
        @test summary.edge_win ≈ 15.0
        @test summary.edge_loss ≈ 7.5
        @test summary.capture_ratio ≈ 2.0
        @test summary.stake_mean == 2.5
        @test summary.odds_mean == 2.825
        @test summary.p_model_mean ≈ 0.4375
        @test summary.p_market_mean == 0.325
        @test capture_ratio(mixed) == summary.capture_ratio

        all_win = pat_bets(payoff = [0.8, 1.5, 2.0, 3.0])
        all_loss = pat_bets(payoff = fill(-1.0, 4))
        @test isnan(edge_summary(all_win).capture_ratio)
        @test isnan(edge_summary(all_win).edge_loss)
        @test isnan(edge_summary(all_loss).capture_ratio)
        @test isnan(edge_summary(all_loss).edge_win)

        zero_loss_edge = pat_bets(p_model = [0.60, 0.50, 0.30, 0.20])
        negative_loss_edge = pat_bets(p_model = [0.60, 0.50, 0.29, 0.19])
        @test edge_summary(zero_loss_edge).edge_loss == 0.0
        @test isnan(capture_ratio(zero_loss_edge))
        @test edge_summary(negative_loss_edge).edge_loss < 0.0
        @test isnan(capture_ratio(negative_loss_edge))

        pushed = pat_bets(payoff = [0.8, 0.0, -1.0, 0.0])
        pushed_summary = edge_summary(pushed)
        @test pushed_summary.n_wins == 1
        @test pushed_summary.win_rate == 0.25
        @test pushed_summary.edge_win ≈ 20.0
        @test pushed_summary.edge_loss ≈ 5.0
        @test pushed_summary.capture_ratio ≈ 4.0

        empty_summary = edge_summary(DataFrame())
        @test empty_summary.n_bets == 0
        @test empty_summary.stake_sum == 0.0
        @test isnan(empty_summary.capture_ratio)
        @test isnan(capture_ratio(DataFrame()))
        @test_throws ErrorException edge_summary(DataFrame(stake = [1.0]))

        # `extend_portfolio` union-merges old and new ledger schemas. Missing attribution fields
        # must disable confidence metrics, not crash or poison still-valid stake/P&L totals.
        legacy = select(mixed[1:1, :], :match_id, :date, :family, :stake, :odds, :pnl)
        merged = vcat(legacy, mixed[2:2, :]; cols = :union)
        merged_summary = edge_summary(merged)
        @test merged_summary.n_bets == 2
        @test merged_summary.stake_sum == 4.0
        @test merged_summary.pnl_sum ≈ 5.3
        @test merged_summary.roi ≈ 132.5
        @test isnan(merged_summary.win_rate)
        @test isnan(merged_summary.edge_mean)
        @test isnan(merged_summary.capture_ratio)
    end

    @testset "Partitioning and duplicate refusal" begin
        a = pat_bets(match_id = [3, 1, 2],
                     family = ["BTTS_yes", "1X2_home", "O/U 2.5_over_25"],
                     selection = [:btts_yes, :home, :over_25],
                     odds = [2.0, 2.0, 2.0], stake = [0.3, 0.1, 0.2],
                     payoff = [-1.0, 1.0, 1.0],
                     p_model = [0.6, 0.6, 0.6], p_market = [0.5, 0.5, 0.5])
        b = pat_bets(match_id = [4, 2, 1],
                     family = ["1X2_away", "O/U 2.5_over_25", "1X2_home"],
                     selection = [:away, :over_25, :home],
                     odds = [2.0, 2.0, 2.0], stake = [0.4, 0.1, 0.3],
                     payoff = [-1.0, 1.0, 1.0],
                     p_model = [0.6, 0.6, 0.6], p_market = [0.5, 0.5, 0.5])

        both_a, both_b, only_a, only_b = partition_bets(a, b)
        @test both_a.match_id == [1, 2]
        @test both_b.match_id == [1, 2]
        @test both_a.family == both_b.family
        @test only_a.match_id == [3]
        @test only_b.match_id == [4]
        @test !(:key in propertynames(both_a))
        @test !(:key in propertynames(only_b))
        @test a.match_id == [3, 1, 2] # inputs were not mutated

        duplicate = vcat(a, a[1:1, :])
        @test_throws ErrorException partition_bets(duplicate, b)
        @test_throws ErrorException partition_bets(a, vcat(b, b[1:1, :]))

        ea, eb, eoa, eob = partition_bets(DataFrame(), DataFrame())
        @test all(isempty, (ea, eb, eoa, eob))
        _, _, all_a, all_b = partition_bets(a, DataFrame())
        @test nrow(all_a) == nrow(a)
        @test isempty(all_b)
    end

    @testset "Shared-bet sizing identity and comparison" begin
        a = pat_bets(match_id = [1, 2, 3],
                     family = ["1X2_home", "O/U 2.5_over_25", "BTTS_yes"],
                     selection = [:home, :over_25, :btts_yes],
                     odds = [2.0, 3.0, 2.5], stake = [0.30, 0.10, 0.20],
                     payoff = [1.0, -1.0, 1.5],
                     p_model = [0.60, 0.40, 0.50], p_market = [0.50, 0.30, 0.40])
        b = pat_bets(match_id = [1, 2, 4],
                     family = ["1X2_home", "O/U 2.5_over_25", "1X2_away"],
                     selection = [:home, :over_25, :away],
                     odds = [2.0, 3.0, 4.0], stake = [0.10, 0.25, 0.10],
                     payoff = [1.0, -1.0, -1.0],
                     p_model = [0.55, 0.35, 0.30], p_market = [0.50, 0.30, 0.20])
        both_a, both_b, _, _ = partition_bets(a, b)
        expected = sum((both_a.stake .- both_b.stake) .* both_a.payoff)
        @test shared_bet_sizing_attribution(both_a, both_b) ≈ expected
        @test expected ≈ 0.35
        @test shared_bet_sizing_attribution(DataFrame(), DataFrame()) == 0.0
        @test_throws ErrorException shared_bet_sizing_attribution(both_a, reverse(both_b))
        bad_settlement = copy(both_b)
        bad_settlement.payoff[1] = -1.0
        @test_throws ErrorException shared_bet_sizing_attribution(both_a, bad_settlement)

        result_a, result_b = pat_result(a), pat_result(b)
        comparison = compare_portfolios(result_a, result_b; name_a = "Alpha", name_b = "Beta")
        @test comparison isa ModelComparisonAttribution
        @test comparison.name_a == "Alpha"
        @test nrow(comparison.shared_a) == 2
        @test nrow(comparison.exclusive_a) == 1
        @test nrow(comparison.exclusive_b) == 1
        @test comparison.sizing_delta_pnl ≈ expected
        @test comparison.shared_roi_a ≈ edge_summary(both_a).roi
        @test comparison.shared_roi_b ≈ edge_summary(both_b).roi
        @test comparison.summary_a.n_bets == 3
        @test comparison.summary_b.n_bets == 3

        @test capture_ratio(result_a.trajectory) == capture_ratio(a)
        @test capture_ratio(result_a) == capture_ratio(a)
        @test edge_summary(result_a.trajectory).n_bets == 3
        @test edge_summary(result_a).n_bets == 3

        @test occursin("EdgeSummary", sprint(show, edge_summary(a)))
        @test occursin("Alpha", sprint(show, comparison))
        @test occursin("sizing ΔPnL", sprint(show, MIME"text/plain"(), comparison))
        @test occursin("Portfolio attribution: Alpha vs Beta", attribution_markdown(comparison))

        # Rendering remains compatible with the documented legacy Trajectory schema, which does
        # not promise payoff/model/market columns.
        legacy_bets = select(a, :match_id, :date, :family, :stake, :odds, :pnl)
        legacy_result = pat_result(legacy_bets)
        rendered = sprint(io -> display_portfolio(legacy_result; io = io))
        @test occursin("capture ratio", rendered)
        @test occursin("| capture ratio | -- |", portfolio_markdown(legacy_result))
    end

    @testset "Odds and market-family breakdowns" begin
        bets = pat_bets()
        odds = odds_breakdown(bets)
        @test odds.odds_bucket == ["< 2.0", "2.0 - 3.5", "≥ 3.5"]
        @test odds.n_bets == [1, 2, 1]
        @test sum(odds.n_bets) == nrow(bets)
        @test isequal(edge_breakdown(bets; by = :odds), odds)

        sparse_odds = odds_breakdown(bets[1:1, :])
        @test sparse_odds.odds_bucket == ["< 2.0", "2.0 - 3.5", "≥ 3.5"]
        @test sparse_odds.n_bets == [1, 0, 0]

        family = family_breakdown(bets)
        @test family.market_family == ["1X2", "OverUnder", "BTTS"]
        @test family.n_bets == [2, 1, 1]
        @test isequal(edge_breakdown(bets; by = :market_family), family)
        sparse_family = family_breakdown(bets[1:1, :])
        @test sparse_family.market_family == ["1X2", "OverUnder", "BTTS"]
        @test sparse_family.n_bets == [1, 0, 0]
        @test isequal(family_breakdown(pat_trajectory(bets)), family)
        @test isequal(family_breakdown(pat_result(bets)), family)
        @test isempty(odds_breakdown(DataFrame()))
        @test isempty(family_breakdown(DataFrame()))
        @test_throws ErrorException edge_breakdown(bets; by = :date)
    end
end
