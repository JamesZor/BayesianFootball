using Test
using Dates
using DataFrames

include(joinpath(@__DIR__, "l01_microstructure_sweeper.jl"))
include(joinpath(@__DIR__, "l02_archive_research.jl"))
const ME_CLOCK = MicrostructureExecution
const AR_CLOCK = ArchiveMicrostructureResearch

@testset "chronological replay shares same-stamp depth in UUID order" begin
    kickoff = DateTime(2026, 9, 5, 14)
    orders = DataFrame(
        order_id = ["a-order", "b-order"],
        match_id = [1, 1],
        kickoff = [kickoff, kickoff],
        side = ["back", "back"],
        p_model = [0.60, 0.60],
        risk = [5.0, 5.0],
        venue_odds = [3.0, 3.0],
        slate_bankroll = [100.0, 100.0],
        selection = ["home", "home"],
        venue_selection = ["home", "home"],
    )
    rows = DataFrame(
        order_id = ["a-order", "b-order"], market_id = ["market", "market"],
        symbol = ["home", "home"], ts = [kickoff - Minute(25), kickoff - Minute(25)],
        kickoff = [kickoff, kickoff], bid_prices = [[30_000], [30_000]],
        bid_volumes = [[50_000], [50_000]], ask_prices = [[31_000], [31_000]],
        ask_volumes = [[50_000], [50_000]], market_matched = [100_000, 100_000],
    )
    parents, children, diagnostics = AR_CLOCK.replay_policy(
        orders, rows, ME_CLOCK.MultiLevelSweep(max_slip = 0.01), ME_CLOCK;
        start_minutes = 25, end_minutes = 5, commission = 0.02, hurdle = 0.02,
        refreshed = false,
    )
    @test nrow(children) == 1
    @test only(children.order_id) == "a-order"
    @test only(children.venue_stake) == 5.0
    evaluated = diagnostics[diagnostics.status .== :evaluated, :]
    @test nrow(evaluated) == 2
    @test evaluated.priority == [1, 2]
    @test parents.simulated_risk == [5.0, 0.0]
end

println("archive clock-order test passed")
