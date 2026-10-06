# Manager-authorised Gaussian-only arithmetic optimisation: deterministic parity.
using Test, LinearAlgebra, Random, BayesianFootball
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__,"l05_pooled.jl"))
end
if !isdefined(@__MODULE__, :FastGaussianBook)
    include(joinpath(@__DIR__,"l05_fast_gaussian.jl"))
end
const GP08 = PooledMarket
const GF08 = FastGaussianBook
@testset "C8 scalar Gaussian arithmetic, unchanged model and native sampler" begin
    p = GP08.MID.toy_panel(Random.Xoshiro(3951))
    toy = GF08.parity_gates(p; scope="toy")
    @test all(toy.pass)
    @test size(toy,1) == 27
    @test_throws ErrorException GF08.evaluator(GP08.FullBookRung(:C1),p)
    for rung in (:C0,:H1,:H2)
        a = GP08.FullBookRung(rung)
        theta = GP08.MID.init_centre(a)
        callback = GF08.evaluator(a,p)
        @test abs(GP08.fullbook_logtarget(a,p,theta)-GP08.fullbook_logtarget(a,p,theta; filter_fn=callback)) <= 1e-9
        default = GP08.fullbook_chain(a,p,5101; warmup=5,samples=10)
        explicit = GP08.fullbook_chain(a,p,5101; warmup=5,samples=10,filter_fn=GP08.fullbook_filter)
        fast = GP08.fullbook_chain(a,p,5101; warmup=5,samples=10,filter_fn=callback)
        @test default.U == explicit.U
        @test default.widths == explicit.widths
        @test maximum(abs,default.U-fast.U) <= 1e-9
    end
    config = GP08.MM.scottish_lower_2425_2526()
    ds = BayesianFootball.Data.load_datastore_cached(config.segment; max_age_hours=10^6)
    real = GP08.CM.TB.phase_b_panel(ds; config).panel
    gates = GF08.parity_gates(real; scope="full real")
    @test all(gates.pass)
    @test size(gates,1) == 27
    for rung in (:C0,:H1,:H2)
        a = GP08.FullBookRung(rung)
        theta = GP08.MID.init_centre(a)
        evaluator = GF08.evaluator(a,real)
        evaluator(a,real,theta)
        GP08.fullbook_filter(a,real,theta)
        dense = @elapsed for _ in 1:10
            GP08.fullbook_filter(a,real,theta)
        end
        scalar = @elapsed for _ in 1:10
            evaluator(a,real,theta)
        end
        println("C8 Gaussian benchmark $rung: dense=$dense scalar=$scalar for10 fixed-target calls (not fits)")
    end
end
println("T05_FAST_GAUSSIAN_DONE")
