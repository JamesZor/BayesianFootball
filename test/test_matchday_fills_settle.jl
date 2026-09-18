# Deterministic regression coverage for T008 multi-level fill accounting.

using Test
using BayesianFootball
using Dates, UUIDs

const T008_MD = BayesianFootball.MatchDay

_t008_levels(back, back_size, lay, lay_size) =
    T008_MD.BookLevels(Float64.(back), Float64.(back_size), Float64.(lay),
                       Float64.(lay_size), 1_000.0, DateTime(2026, 9, 7, 14, 30))

function _t008_order(; side = :back, odds = 3.0, risk = 20.0, venue_stake = 20.0,
                       selection = :home, venue_selection = selection, order_id = uuid4())
    leverage = side === :back ? 1.0 : 1.0 / (odds - 1.0)
    return T008_MD.PaperOrder(
        order_id = order_id, slate_id = UUID(8), account_id = "t008", match_id = 8008,
        kickoff = DateTime(2026, 9, 7, 15), market_group = "1X2", market_line = 0.0,
        selection = selection, venue_selection = venue_selection, side = side,
        venue_odds = odds, leverage = leverage,
        effective_odds = side === :lay ? T008_MD.lay_to_back(odds) : odds,
        p_model = 0.5, p_market = 0.5, edge = 0.0, stake_fraction = risk / 1_000,
        risk = risk, venue_stake = venue_stake,
        quote_ts = DateTime(2026, 9, 7, 14, 30))
end

@testset "T008 multi-level fill and settlement accounting" begin
    at = DateTime(2026, 9, 7, 14, 31)

    @testset "back fills use arithmetic payoff VWAP and exact cashflows" begin
        levels = _t008_levels([3.0, 2.98], [10.0, 10.0], [3.1], [20.0])
        fills = T008_MD.simulate_fill(T008_MD.LadderSweepV2(), levels, :back,
                                      20.0, 1.0, at)
        @test length(fills) == 2
        @test all(f -> f.model === :ladder_sweep_v2, fills)
        @test T008_MD.filled_risk(fills) ≈ 20.0
        @test T008_MD.fill_vwap(fills) ≈ 2.99
        @test T008_MD.fill_harmonic_mean(fills) ≈ 20 / (10 / 3.0 + 10 / 2.98)
        @test T008_MD.fill_harmonic_mean(fills) < T008_MD.fill_vwap(fills)

        order = _t008_order()
        settled = T008_MD.settle_order(order, fills, 2, 1, 0.02)
        # £20 at payoff-equivalent odds 2.99 returns £59.80: £39.80 profit before commission.
        @test settled.gross_return ≈ 59.80
        @test settled.commission ≈ 0.796
        @test settled.net_pnl ≈ 39.004
        @test settled.gross_return - settled.commission ≈ 59.004
        lost = T008_MD.settle_order(order, fills, 0, 1, 0.02)
        @test lost.gross_return == 0.0
        @test lost.commission == 0.0
        @test lost.net_pnl ≈ -20.0
    end

    @testset "lay fills cap every child at remaining parent liability" begin
        levels = _t008_levels([2.9], [20.0], [3.0, 3.1], [10.0, 10.0])
        order = _t008_order(side = :lay, risk = 40.0, venue_stake = 20.0,
                            selection = :away, venue_selection = :home)
        fills = T008_MD.simulate_fill(T008_MD.LadderSweepV2(max_slippage = 0.05),
                                      levels, :lay, order.venue_stake, order.leverage, at)
        @test length(fills) == 2
        @test fills[1].size ≈ 10.0
        @test fills[1].risk_filled ≈ 20.0
        @test fills[2].size ≈ 20.0 / 2.1
        @test fills[2].risk_filled ≈ 20.0
        @test T008_MD.filled_risk(fills) ≈ 40.0
        @test T008_MD.filled_risk(fills) <= order.risk + 1e-9

        settled = T008_MD.settle_order(order, fills, 0, 1, 0.0)
        @test settled.outcome === :win
        @test settled.gross_return ≈ 40.0 + 10.0 + 20.0 / 2.1
        @test settled.net_pnl ≈ 10.0 + 20.0 / 2.1
        lost = T008_MD.settle_order(order, fills, 2, 0, 0.02)
        @test lost.gross_return == 0.0
        @test lost.commission == 0.0
        @test lost.net_pnl ≈ -40.0

        entry = T008_MD.clv_for_order(order, fills, 0.7, DateTime(2026, 9, 7, 14, 59))
        arithmetic_price = sum(f.size * f.price for f in fills) / sum(f.size for f in fills)
        @test entry.entry_prob ≈ 1 / T008_MD.lay_to_back(arithmetic_price)
    end

    @testset "adverse lay slippage is refused by V2 but legacy remains reconstructible" begin
        levels = _t008_levels([2.9], [20.0], [3.0, 3.1], [10.0, 10.0])
        leverage = 1 / (3.0 - 1.0)
        fills = T008_MD.simulate_fill(T008_MD.LadderSweepV2(max_slippage = 0.01),
                                      levels, :lay, 20.0, leverage, at)
        @test length(fills) == 1
        @test T008_MD.filled_size(fills) ≈ 10.0
        @test T008_MD.fill_model_name(T008_MD.LadderSweepV2()) === :ladder_sweep_v2
        @test T008_MD.fill_model_name(T008_MD.LadderSweep()) === :ladder_sweep_v1

        back_levels = _t008_levels([3.0, 2.98], [10.0, 10.0], [3.1], [20.0])
        legacy_fills = T008_MD.simulate_fill(T008_MD.LadderSweep(), back_levels, :back,
                                             20.0, 1.0, at)
        legacy = T008_MD.settle_order_legacy_v1(_t008_order(), legacy_fills, 2, 1, 0.0)
        corrected = T008_MD.settle_order(_t008_order(), legacy_fills, 2, 1, 0.0)
        @test legacy.gross_return ≈ 60.0
        @test corrected.gross_return ≈ 59.8
        @test T008_MD.fill_harmonic_mean(legacy_fills) < T008_MD.fill_vwap(legacy_fills)
    end

    @testset "commission is charged on net market winnings" begin
        winner = _t008_order(selection = :home, venue_selection = :home)
        loser = _t008_order(selection = :away, venue_selection = :away)
        winner_fill = [T008_MD.Fill(order_id = winner.order_id, filled_at = at, price = 3.0,
                                    size = 20.0, risk_filled = 20.0,
                                    model = :ladder_sweep_v2)]
        loser_fill = [T008_MD.Fill(order_id = loser.order_id, filled_at = at, price = 3.0,
                                   size = 20.0, risk_filled = 20.0,
                                   model = :ladder_sweep_v2)]
        raw = [T008_MD.settle_order(winner, winner_fill, 2, 1, 0.0),
               T008_MD.settle_order(loser, loser_fill, 2, 1, 0.0)]
        @test raw[1].net_pnl ≈ 40.0
        @test raw[2].net_pnl ≈ -20.0

        netted = T008_MD._net_market_commission(raw, 0.02)
        @test sum(s.commission for s in netted) ≈ 0.02 * (40.0 - 20.0)
        @test sum(s.net_pnl for s in netted) ≈ 19.6
        @test netted[2].commission == 0.0
    end
end
