using Test
using Dates
using DataFrames

include(joinpath(@__DIR__, "l02_archive_research.jl"))
const AR_TEST = ArchiveMicrostructureResearch

@testset "archive fixed-point decoding preserves thin ladders" begin
    @test AR_TEST._fixed_tuple([22_400, 22_200], AR_TEST.PRICE_SCALE) == (2.24, 2.22, 0.0)
    @test AR_TEST._fixed_tuple([50_000], AR_TEST.SIZE_SCALE) == (5.0, 0.0, 0.0)
    @test AR_TEST._fixed_tuple(Union{Missing,Int}[10_000, missing, 30_000], AR_TEST.SIZE_SCALE) ==
          (1.0, 0.0, 3.0)
end

@testset "as-of selection cannot see future archive rows" begin
    kickoff = DateTime(2026, 9, 5, 14)
    rows = DataFrame(order_id = ["o", "o"], market_id = ["m", "m"], symbol = ["home", "home"],
                     ts = [kickoff - Minute(26), kickoff - Minute(24)], kickoff = [kickoff, kickoff],
                     bid_prices = [[30_000], [31_000]], bid_volumes = [[10_000], [10_000]],
                     ask_prices = [[31_000], [32_000]], ask_volumes = [[10_000], [10_000]],
                     market_matched = [10_000, 20_000])
    selected = AR_TEST.snapshot_at(rows, "o", kickoff - Minute(25))
    @test selected !== nothing
    @test selected.ts == kickoff - Minute(26)
    @test selected.back[1] == 3.0
    @test AR_TEST.snapshot_at(rows, "o", kickoff - Minute(27)) === nothing
end

@testset "shared absolute-price depletion crosses parent orders" begin
    depleted = Dict{Tuple{String,String,Symbol,Float64},Float64}()
    prices = (3.0, 2.9, 2.8)
    displayed = (5.0, 4.0, 0.0)
    AR_TEST.consume_sizes!(depleted, "market", "home", :back, prices, (3.0, 0.0, 0.0))
    @test AR_TEST.residual_sizes("market", "home", :back, prices, displayed, depleted) == (2.0, 4.0, 0.0)
    # Different side is distinct resting liquidity and cannot be depleted by the back parent.
    @test AR_TEST.residual_sizes("market", "home", :lay, prices, displayed, depleted) == displayed
end

println("archive research pure tests passed")
