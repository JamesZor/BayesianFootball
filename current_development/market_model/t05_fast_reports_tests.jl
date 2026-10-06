# Deterministic fast-stage reporting tests; IID mocks are NOT posterior fits.
using Test, LinearAlgebra, Distributions, DataFrames, Random, BayesianFootball, CSV
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__,"l05_pooled.jl"))
end
if !isdefined(@__MODULE__, :FullBookWorkflow)
    include(joinpath(@__DIR__,"l05_fullbook_workflow.jl"))
end
if !isdefined(@__MODULE__, :FastFullBookReports)
    include(joinpath(@__DIR__,"l05_fast_reports.jl"))
end
const FR08 = FastFullBookReports
const FP08 = PooledMarket
@testset "C8 fast-stage reports, NOT sampled acceptance" begin
    q = FR08.normal_mixture_quantiles(fill(0.3,10),fill(0.2,10))
    @test maximum(abs,q-quantile.(Normal(0.3,0.2),[0.05,0.5,0.95])) <= 1e-12
    @test_throws ErrorException FR08.normal_mixture_quantiles([0.0],[0.0])
    p = FP08.MID.toy_panel(Random.Xoshiro(3951))
    p.matches.season .= "toy"
    p.teams[1] = "east-kilbride" # mock name exercises present and absent figure branches
    fits = Dict{Tuple{Symbol,String},FP08.MID.ArmFit}()
    rng = Random.Xoshiro(3965)
    for rung in (:C0,:H1,:H2), protocol in ("10a","10b")
        a = FP08.FullBookRung(rung)
        centre = FP08.MID.init_centre(a)
        U = 0.01randn(rng,1000,length(centre),4) .+ reshape(centre,1,:,1)
        fits[(rung,protocol)] = FP08.MID.ArmFit(a,FP08.MID.param_names(a),
            FP08.fullbook_physical_draws(a,U),U,Matrix{Float64}[],Vector{Float64}[],zeros(0,0),0.0)
    end
    theta = FP08.MID.init_centre(FP08.FullBookRung(:H1))
    f = FP08.fullbook_filter(FP08.FullBookRung(:H1),p,theta; store=true)
    batch = FP08.fullbook_batch(FP08.FullBookRung(:H1),p,theta)
    n = size(f.m_filt,1)
    @test maximum(abs,f.m_filt[[2,n],end]-batch.post_mean[(p.n_weeks-1)*n .+ [2,n]]) <= 1e-8
    @test maximum(abs,f.P_filt[[2,n],[2,n],end]-batch.post_cov[(p.n_weeks-1)*n .+ [2,n],(p.n_weeks-1)*n .+ [2,n]]) <= 1e-8
    ha = FR08.home_advantage(fits,p,p)
    @test nrow(ha) == 14
    @test all(ha.lo .< ha.median .< ha.hi)
    @test all(ha.hyperdraws .== 4000)
    @test count(==("kappa"),ha.parameter) == 2
    config = FP08.MM.MarketModelConfig(name="toy",segment=BayesianFootball.Data.ScottishLower(),
        tournaments=[1],seasons=["toy"],honest_train=["train"],honest_test=["toy"])
    mktempdir() do out
        raw = FullBookWorkflow.evaluate_fits(p,nothing,config,fits,out)
        @test nrow(raw) == 6*5*FP08.MID.n_fixtures(p)
        FR08.smoothed_total(fits,p,out; mc_draws=100)
        sm = CSV.read(joinpath(out,"smoothed_fit_c.csv"),DataFrame)
        @test nrow(sm) == 6*5*FP08.MID.n_fixtures(p)
        @test all(isfinite,sm.predicted)
        CSV.write(joinpath(out,"home_advantage_rungs.csv"),ha)
        FR08.figures(out)
        inventory = CSV.read(joinpath(out,"path_club_inventory.csv"),DataFrame)
        @test nrow(inventory) == 4
        @test count(inventory.present) == 1
        @test inventory.team[inventory.present] == ["east-kilbride"]
        for file in ("C_fast_coverage.png","C_fast_home_advantage.png","C_fast_paths.png")
            @test filesize(joinpath(out,file)) > 1000
        end
        ref = copy(raw[raw.rung .== "C0",:])
        ref.rung .= "R6"
        CSV.write(joinpath(out,"mock_r6.csv"),ref)
        paired = FR08.compare_r6(raw,out; reference=joinpath(out,"mock_r6.csv"))
        @test all(paired.logpd .== paired.r6_logpd)
        gap = CSV.read(joinpath(out,"paired_c0_vs_r6.csv"),DataFrame)
        @test gap.mean_delta == [0.0,0.0]
        ref.observed[1] += 1.0
        CSV.write(joinpath(out,"bad_r6.csv"),ref)
        @test_throws ErrorException FR08.compare_r6(raw,out; reference=joinpath(out,"bad_r6.csv"))
    end
    for file in ("r05_fast_rungs.jl","r05_c1_pending.jl")
        parsed = Meta.parseall(read(joinpath(@__DIR__,file),String))
        @test !any(x -> x isa Expr && x.head in (:error,:incomplete),parsed.args)
    end
end
println("T05_FAST_REPORTS_DONE")
