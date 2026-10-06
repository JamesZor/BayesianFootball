using Test, LinearAlgebra, Random, Statistics, Dates
using BayesianFootball, DataFrames, CSV, Optim, Distributions
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :CopulaGridMarket)
    include(joinpath(@__DIR__,"l04_copula_grid.jl"))
end
const CGT = CopulaGridMarket

function cgt_book(g,theta; id=1)
    q = CGT.grid_probabilities(g,theta)
    rows = NamedTuple[]
    for (line,selections) in (("1X2",(:home,:draw,:away)),("BTTS",(:btts_yes,:btts_no)),
        ("OU0.5",(:under_05,:over_05)),("OU1.5",(:under_15,:over_15)),
        ("OU2.5",(:under_25,:over_25)),("OU3.5",(:under_35,:over_35)))
        for s in selections
            push!(rows,(; match_id=id,selection=s,prob_fair_close=q[s]))
        end
    end
    return DataFrame(rows)
end

"Independent BigFloat CDF rectangle differences using Distributions Poisson CDFs."
function cgt_frank_reference(h,a,kappa)
    fh = BigFloat.([0.0; cdf.(Poisson(h),0:10)])
    fa = BigFloat.([0.0; cdf.(Poisson(a),0:10)])
    k = BigFloat(kappa)
    C(u,v) = -log(1+(exp(-k*u)-1)*(exp(-k*v)-1)/(exp(-k)-1))/k
    P = [C(fh[x+2],fa[y+2])-C(fh[x+1],fa[y+2])-C(fh[x+2],fa[y+1])+C(fh[x+1],fa[y+1])
        for x in 0:10,y in 0:10]
    return Float64.(P/sum(P))
end

"Dense connected fixture panel for deterministic PPC verification (not MCMC)."
function cgt_panel(; teams=12,weeks=100,seed=3943)
    rng = Xoshiro(seed)
    ow,oa,od,om = Int[],Int[],Int[],Int[]
    oh = Float64[]
    rows = NamedTuple[]
    for w in 1:weeks
        order = randperm(rng,teams)
        for j in 1:2:teams
            h,a = order[j:j+1]
            id = length(rows)+1
            push!(rows,(; match_id=id,match_date=Date(2024,1,1)+Week(w-1),week=w,
                season="toy",home_team=string(h),away_team=string(a)))
            append!(ow,(w,w))
            append!(oa,(h,a))
            append!(od,(a,h))
            append!(om,(id,id))
            append!(oh,(1.0,0.0))
        end
    end
    ptr = [1; 1 .+ cumsum([count(==(w),ow) for w in 1:weeks])]
    return CGT.MID.MarketPanel(DataFrame(rows),DataFrame(),string.(1:teams),weeks,
        [Date(2024,1,1)+Week(w-1) for w in 1:weeks],ow,oh,oa,od,zeros(length(ow)),om,
        fill("toy",length(ow)),ptr,zeros(length(ow),0),String[])
end

@testset "Phase B3 grids and refiltered PPC" begin
    @testset "Normalisation, marginals and independence reductions" begin
        theta = log.([1.5,1.1])
        baseline = CGT.score_grid(CGT.ScoreGrid(0),theta)
        for g in (CGT.ScoreGrid(0),CGT.ScoreGrid(1,-0.08),CGT.ScoreGrid(2,0.12),CGT.ScoreGrid(3,0.6),CGT.ScoreGrid(3,-0.6))
            P = CGT.score_grid(g,theta)
            @test abs(sum(P)-1) <= 1e-14
            @test minimum(P) >= 0
            h,a = CGT.reported_rates(g,theta)
            mean_h = sum((x-1)*P[x,y] for x in 1:11,y in 1:11)
            mean_a = sum((y-1)*P[x,y] for x in 1:11,y in 1:11)
            # Truncation is measured, not mistaken for exact untruncated marginal means.
            @test abs(mean_h-h) < 2e-5
            @test abs(mean_a-a) < 2e-5
            println("$(CGT.grid_name(g)) $(g.parameter) marginal truncation error: ",mean_h-h," / ",mean_a-a)
        end
        for g in (CGT.ScoreGrid(1,0),CGT.ScoreGrid(2,0),CGT.ScoreGrid(3,1e-8))
            @test maximum(abs.(CGT.score_grid(g,theta)-baseline)) <= 1e-10
        end
        for x in 0:3,y in 0:3
            expected = CGT.MM.FEAT.dixon_coles_tau(x,y,1.5,1.1,-0.08)
            P = CGT.score_grid(CGT.ScoreGrid(1,-0.08),theta)
            @test abs(P[x+1,y+1]/baseline[x+1,y+1]-expected) <= 1e-12
        end
    end
    @testset "Synthetic recovery including globally profiled parameter" begin
        for g in (CGT.ScoreGrid(0),CGT.ScoreGrid(1,-0.08),CGT.ScoreGrid(2,0.12),CGT.ScoreGrid(3,0.6))
            theta = log.([1.7,0.9])
            book = cgt_book(g,theta)
            fit = CGT.solve_grid(g,book)
            @test fit.converged
            @test fit.spread <= 1e-6
            @test maximum(abs.(fit.theta-theta)) <= 1e-5
            @test abs(fit.kl) <= 1e-12
            if g.kind > 0
                lo,hi = g.kind == 1 ? (-0.15,0.0) : g.kind == 2 ? (0.01,0.25) : (-1.0,2.0)
                objective = k -> CGT.solve_grid(CGT.ScoreGrid(g.kind,k),book).kl
                profiled = Optim.optimize(objective,lo,hi,Optim.Brent(); abs_tol=1e-9,rel_tol=1e-9)
                recovered = Optim.minimizer(profiled)
                @test abs(recovered-g.parameter) <= 1e-5
                println("Synthetic $(CGT.grid_name(g)) dependence recovery: ",recovered)
            end
        end
    end
    @testset "Frank versus independent brute force, positive and negative" begin
        rng = Xoshiro(3944)
        for j in 1:20
            h,a = 0.3 .+ 2.7rand(rng,2)
            k = (isodd(j) ? -1 : 1)*(0.1+3rand(rng))
            P = CGT.score_grid(CGT.ScoreGrid(3,k),log.([h,a]))
            @test maximum(abs.(P-cgt_frank_reference(h,a,k))) <= 1e-12
        end
    end
    @testset "Active weeks, Kendall and Gaussian/heavy-tail PPC discrimination" begin
        @test CGT.kendall_tau([1.,2.,3.],[3.,2.,1.]) == -1
        @test CGT.kendall_tau([1.,2.,3.],[1.,2.,3.]) == 1
        base = cgt_panel()
        theta = vcat(log.([0.005,0.08,0.08]),atanh(-0.8))
        gaussian,_ = CGT.simulate_panel(base,theta,Xoshiro(3945))
        heavy,_ = CGT.simulate_panel(base,theta,Xoshiro(3946); heavy_quality=true)
        fixed = repeat(theta',200)
        nullraw,_,keys = CGT.tail_ppc(gaussian,fixed; seed=3947)
        heavraw,_,_ = CGT.tail_ppc(heavy,fixed; seed=3948)
        nullraw.protocol .= "synthetic Gaussian"
        heavraw.protocol .= "synthetic t3 quality"
        summary = CGT.tail_summary(vcat(nullraw,heavraw))
        show(stdout,MIME"text/plain"(),summary; allrows=true,allcols=true)
        println()
        @test length(keys) == 12*99
        # Check requested primary statistics, not every redundant transformed statistic.
        primary = ["kurtosis_quality","kurtosis_style","kendall_alpha_beta",
            "joint_improvement_95","joint_collapse_95","lag1_squared_quality"]
        null = filter(r->r.protocol == "synthetic Gaussian" && r.statistic in primary,summary)
        @test all(.!null.extreme)
        heavyq = filter(r->r.protocol == "synthetic t3 quality" && r.statistic == "kurtosis_quality",summary)
        @test only(heavyq.ppp) < 0.01
    end
end
println("T04_DONE")
