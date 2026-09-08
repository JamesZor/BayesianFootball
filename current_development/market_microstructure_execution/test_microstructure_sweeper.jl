using Test, Dates
if !isdefined(@__MODULE__, :MicrostructureExecution)
    include("l01_microstructure_sweeper.jl")
end
import .MicrostructureExecution as ME

const MS_KICKOFF = DateTime(2026, 9, 5, 14)
const MS_START = MS_KICKOFF - Minute(25)
ms_book(; ts=MS_START, back=(3.0, 2.98, 2.96), bs=(10.0, 20.0, 30.0),
        lay=(3.05, 3.10, 3.15), ls=(10.0, 20.0, 30.0), matched=100.0) =
    ME.Ladder(ts, back, bs, lay, ls, matched)
ms_run!(s, policy, order, book=ms_book(); at=MS_START, kwargs...) =
    ME.execute_snapshot!(s, policy, order, book, at, MS_KICKOFF; kwargs...)

@testset "Reservation: independent expected-return identity" begin
    for side in (:back, :lay), p in (0.2, 0.5, 0.8), c in (0.0, 0.02, 0.05), a in (0.0, 0.02)
        order = ME.ExecutionOrder(side, p, 100.0, 3.0, 2400.0; commission=c, hurdle=a)
        r = ME.reservation_price(order)
        b = side === :back ? (r - 1) * (1 - c) : (1 - c) / (r - 1)
        @test p * b - (1 - p) ≈ a atol=1e-14
        bad = side === :back ? r * 0.999 : r * 1.001
        badb = side === :back ? (bad - 1) * (1 - c) : (1 - c) / (bad - 1)
        @test p * badb - (1 - p) < a
    end
    @test_throws ArgumentError ME.ExecutionOrder(:sell, 0.5, 10, 3, 2400)
    @test_throws ArgumentError ME.ExecutionOrder(:back, NaN, 10, 3, 2400)
    @test_throws ArgumentError ME.ExecutionOrder(:lay, 0.5, 2400, 3, 2400)
end

@testset "Three-level venue stake, VWAP and exact liability" begin
    back = ME.ExecutionOrder(:back, 0.5, 50, 3, 2400)
    s = ME.ExecutionState()
    sizes = @inferred ms_run!(s, ME.MultiLevelSweep(max_slip=0.10), back)
    @test sizes == (10.0, 20.0, 20.0)
    @test s.risk == 50
    @test ME.arithmetic_vwap(s) ≈ (30 + 59.6 + 59.2) / 50
    @test s.net_win ≈ (s.notional - s.venue_size) * 0.98
    lay = ME.ExecutionOrder(:lay, 0.9, 75, 3.05, 2400)
    s = ME.ExecutionState()
    sizes = ms_run!(s, ME.MultiLevelSweep(max_slip=0.10), lay)
    @test s.risk ≈ 75
    @test s.risk ≈ sum(sizes .* (ms_book().lay .- 1))
    @test s.net_win ≈ sum(sizes) * 0.98
    @test sizes[3] ≈ (75 - 10*2.05 - 20*2.10) / 2.15
    @test ME.arithmetic_vwap(ME.ExecutionState()) === nothing
end

@testset "VWAP boundary permits a partial deeper level; strict edge still applies" begin
    order = ME.ExecutionOrder(:back, 0.5, 100, 3, 2400)
    s = ME.ExecutionState()
    book = ms_book(back=(3.0, 2.8, 2.6), bs=(10.0, 100.0, 100.0))
    sizes = ms_run!(s, ME.MultiLevelSweep(max_slip=0.01), order, book)
    @test sizes[1] == 10
    @test sizes[2] ≈ 0.3 / 0.17
    @test sizes[3] ≈ 0 atol=1e-12
    @test ME.arithmetic_vwap(s) ≈ 2.97
    lay = ME.ExecutionOrder(:lay, 0.9, 100, 3.05, 2400)
    s = ME.ExecutionState()
    sizes = ms_run!(s, ME.MultiLevelSweep(max_slip=0.01), lay)
    @test ME.arithmetic_vwap(s) ≈ 3.0805
    @test 0 < sizes[2] < 20
    weak = ME.ExecutionOrder(:back, 0.34, 100, 3, 2400)
    @test ms_run!(ME.ExecutionState(), ME.MultiLevelSweep(max_slip=0.5), weak) == (0.0, 0.0, 0.0)
end

@testset "Touch depth ablation shares sweep price tolerance" begin
    for side in (:back, :lay)
        order = ME.ExecutionOrder(side, side === :back ? 0.5 : 0.9, 100,
                                  side === :back ? 3.02 : 3.04, 2400)
        touch = ME.ExecutionState()
        sweep = ME.ExecutionState()
        t = ms_run!(touch, ME.TouchOnly(max_slip=0.01), order)
        u = ms_run!(sweep, ME.MultiLevelSweep(max_slip=0.01), order)
        @test t == (10.0, 0.0, 0.0)
        @test t[1] == u[1]
        @test touch.level_reasons == (:depth, :policy_depth, :policy_depth)
        @test touch.last_status == :evaluated
        strict = ME.ExecutionState()
        @test ms_run!(strict, ME.TouchOnly(max_slip=0.0), order) == (0.0, 0.0, 0.0)
        @test strict.level_reasons[1] == :slip
    end
    weak = ME.ExecutionOrder(:back, 0.34, 100, 3, 2400)
    s = ME.ExecutionState()
    ms_run!(s, ME.TouchOnly(), weak)
    @test s.level_reasons[1] == :reservation
    s = ME.ExecutionState()
    order = ME.ExecutionOrder(:back, 0.5, 5, 3, 2400)
    ms_run!(s, ME.MultiLevelSweep(), order)
    @test s.level_reasons == (:target, :target, :target)
    ms_run!(s, ME.MultiLevelSweep(), order)
    @test s.last_status == :reused_quote
end

@testset "Kelly marginal optimum with frozen position exposure" begin
    order = ME.ExecutionOrder(:back, 0.5, 900, 3, 2400; commission=0, hurdle=0)
    s = ME.ExecutionState()
    ms_run!(s, ME.MultiLevelSweep(max_slip=0.5), order,
            ms_book(bs=(1000.0, 1000.0, 1000.0)))
    @test s.risk ≈ 600
    g(q) = 0.5 * log(2400 + 2q) + 0.5 * log(2400 - q)
    @test g(s.risk) > g(s.risk - 0.01)
    @test g(s.risk) > g(s.risk + 0.01)
    @test s.risk < order.target_risk
end

@testset "Filtration, working window, no quote reuse, invalid depth" begin
    order = ME.ExecutionOrder(:back, 0.5, 100, 3, 2400)
    policy = ME.StagedTWAP()
    for book in (ms_book(ts=MS_START+Second(1)), ms_book(ts=MS_START-Second(91)),
                 ms_book(bs=(-1.0, 20.0, 30.0)), ms_book(back=(3.0, 3.01, 2.9)),
                 ms_book(back=(NaN, 2.98, 2.96)), ms_book(lay=(2.9, 3.1, 3.2)))
        @test ms_run!(ME.ExecutionState(), policy, order, book) == (0.0, 0.0, 0.0)
    end
    for at in (MS_START-Second(1), MS_KICKOFF-Minute(5)+Second(1), MS_KICKOFF)
        @test ms_run!(ME.ExecutionState(), policy, order, ms_book(ts=at); at) == (0.0, 0.0, 0.0)
    end
    s = ME.ExecutionState()
    ms_run!(s, policy, order)
    @test s.risk ≈ 100/21
    oldrisk = s.risk
    @test ms_run!(s, policy, order; at=MS_START+Minute(1)) == (0.0, 0.0, 0.0)
    @test s.risk == oldrisk
    @test ms_run!(ME.ExecutionState(), ME.TouchOnly(), order,
                 ms_book(ts=MS_START+Minute(1)); at=MS_START+Minute(1)) == (0.0, 0.0, 0.0)
    @test_throws ArgumentError ms_run!(ME.ExecutionState(), policy, order;
                                     available_sizes=(11.0, 20.0, 30.0))
    @test_throws ArgumentError ms_run!(ME.ExecutionState(), ME.MultiLevelSweep(max_slip=-0.1), order)
end

@testset "TWAP catches up without exceeding original parent or price guards" begin
    order = ME.ExecutionOrder(:back, 0.5, 100, 3, 2400)
    s = ME.ExecutionState()
    policy = ME.StagedTWAP()
    for minute in 0:20
        at = MS_START + Minute(minute)
        sizes = minute < 10 ? (0.0, 0.0, 0.0) : (20.0, 20.0, 30.0)
        ms_run!(s, policy, order, ms_book(ts=at, bs=sizes); at)
        @test s.risk <= order.target_risk + 1e-9
        @test s.risk <= order.target_risk * (minute+1)/21 + 1e-9
    end
    @test s.risk ≈ 100
    # Unfillable at the deadline stays unfilled, never force-crosses a reservation price.
    s = ME.ExecutionState()
    at = MS_KICKOFF-Minute(5)
    ms_run!(s, policy, order, ms_book(ts=at, back=(1.5, 1.4, 1.3)); at)
    @test s.risk == 0
end

@testset "Depth and flow semantics" begin
    @test ME.wom(ms_book()) == 0.5
    @test ME.wom(ms_book(bs=(0.0,0.0,0.0), ls=(0.0,0.0,0.0))) === nothing
    @test ME.execution_signal(ms_book(bs=(100.0,100.0,100.0))) == :back_depth_heavy
    @test ME.matched_velocity(ms_book(), ms_book(ts=MS_START+Minute(1), matched=160.0)) == 1.0
    @test ME.matched_velocity(ms_book(), ms_book(ts=MS_START+Minute(1), matched=99.0)) === nothing
end

@testset "Independent exhaustive feasible-grid utility bound" begin
    for side in (:back, :lay), slip in (0.0, 0.01, 0.20)
        p = side === :back ? 0.55 : 0.85
        reference = side === :back ? 3.0 : 3.05
        order = ME.ExecutionOrder(side, p, 30, reference, 100)
        book = ms_book(bs=(5.0, 7.0, 9.0), ls=(5.0, 7.0, 9.0))
        s = ME.ExecutionState()
        ms_run!(s, ME.MultiLevelSweep(max_slip=slip), order, book)
        optimum = p * log(100+s.net_win) + (1-p) * log(100-s.risk)
        prices = side === :back ? book.back : book.lay
        for x1 in 0.0:0.5:5.0, x2 in 0.0:0.5:7.0, x3 in 0.0:0.5:9.0
            x = (x1,x2,x3)
            risk = side === :back ? sum(x) : sum(x .* (prices .- 1))
            risk <= 30 || continue
            win = side === :back ? sum(x .* (prices .- 1)) * 0.98 : sum(x) * 0.98
            if sum(x) > 0
                vwap = sum(x .* prices) / sum(x)
                (side === :back ? vwap >= reference*(1-slip)-1e-12 :
                                 vwap <= reference*(1+slip)+1e-12) || continue
            end
            utility = p*log(100+win) + (1-p)*log(100-risk)
            @test optimum >= utility - 1e-12
        end
    end
end

# Warm, function-barrier allocation measurement; the mutable output is created OUTSIDE it.
function ms_kernel_allocations!(s, policy, order, book, available)
    s.risk = s.net_win = s.venue_size = s.notional = 0.0
    s.last_ts = DateTime(1)
    return @allocated ME.execute_snapshot!(s, policy, order, book, MS_START, MS_KICKOFF;
                                          available_sizes=available)
end
@testset "Preallocated sweep kernel" begin
    s = ME.ExecutionState()
    book = ms_book()
    for side in (:back, :lay), policy in (ME.TouchOnly(), ME.MultiLevelSweep(), ME.StagedTWAP()),
        available in (nothing, (5.0,10.0,15.0))
        order = ME.ExecutionOrder(side, side === :back ? 0.5 : 0.9, 50, 3.05, 2400)
        ms_kernel_allocations!(s, policy, order, book, available)
        bytes = ms_kernel_allocations!(s, policy, order, book, available)
        @test bytes == 0
    end
end
