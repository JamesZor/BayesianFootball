using Test, LinearAlgebra, Random, Statistics, Dates
using BayesianFootball, CSV, DataFrames
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :CovarianceMarket)
    include(joinpath(@__DIR__, "l03_covariance.jl"))
end
const TC = CovarianceMarket
const TC_CONFIG = TC.MM.scottish_lower_2425_2526()
const TC_REFERENCE_SHA = "767dbe7ec28eefe8ac04c3a25a9e9085a8350be6"

# Independent frozen pre-B2 source, not a reference refactored to call new code.
module PhaseBReference
    const ROOT = @__DIR__
    const SHA = "767dbe7ec28eefe8ac04c3a25a9e9085a8350be6"
    const PATH = "current_development/market_model/"
    source1 = read(`git show $SHA:$(PATH)l01_market_model.jl`,String)
    Base.include_string(@__MODULE__,replace(source1,"module MarketModel"=>"module LegacyMarketModel"),joinpath(ROOT,"l01_market_model.jl"))
    source2 = read(`git show $SHA:$(PATH)l02_two_stage.jl`,String)
    source2 = replace(source2,":MarketModel"=>":LegacyMarketModel",".MarketModel"=>".LegacyMarketModel")
    Base.include_string(@__MODULE__,source2,joinpath(ROOT,"l02_two_stage.jl"))
end

module PhaseARegression end

function tc_synthetic_events(persistent; seed=3925)
    rng = Xoshiro(seed)
    nteams,nrounds = 30,160
    shocks = zeros(nteams)
    events = NamedTuple[]
    id = 0
    for round in 1:nrounds
        shocks .= 0.85shocks+randn(rng,nteams)
        order = randperm(rng,nteams)
        for j in 1:2:nteams
            h,a = order[j:j+1]
            id += 1
            residual = persistent ? shocks[h]-shocks[a]+0.1randn(rng) : randn(rng)
            for (team,sign) in ((h,1),(a,-1))
                push!(events,(; team=string(team),season="synthetic",date=Date(2024,1,1)+Day(round),match_id=id,residual=sign*residual))
            end
        end
    end
    return DataFrame(events)
end

@testset "Phase B2 covariance contracts" begin
    @testset "Phase A config byte regression (runtime exception approved by human)" begin
        root = mktempdir()
        previous = get(ENV,"MM_A_ROOT",nothing)
        ENV["MM_A_ROOT"] = root
        sandbox = PhaseARegression
        try
            Base.include(sandbox,joinpath(@__DIR__,"r01_kl_inversion.jl"))
            published = joinpath(@__DIR__,"results","A")
            files = sort(filter(f -> endswith(f,".csv") && f != "runtime.csv",readdir(published)))
            @test length(files) == 16
            for file in files
                @test read(joinpath(root,"A",file)) == read(joinpath(published,file))
            end
        finally
            previous === nothing ? delete!(ENV,"MM_A_ROOT") : (ENV["MM_A_ROOT"] = previous)
            rm(root; recursive=true)
        end
    end
    @testset "Frozen Phase B array and R2 density regression" begin
        ds = BayesianFootball.Data.load_datastore_cached(TC_CONFIG.segment; max_age_hours=10^6)
        current = TC.TB.phase_b_panel(ds; config=TC_CONFIG).panel
        reference = PhaseBReference.TwoStageMarket.phase_b_panel(ds).panel
        for field in fieldnames(typeof(current))
            @test isequal(getfield(current,field),getfield(reference,field))
        end
        for theta in (log.([0.07,0.05,0.03]),log.([0.12,0.01,0.06]),log.([0.03,0.09,0.02]))
            old = PhaseBReference.TwoStageMarket.rung_filter(PhaseBReference.TwoStageMarket.Rung(2),reference,theta).loglik
            new = TC.TB.rung_filter(TC.TB.Rung(2),current,theta).loglik
            @test abs(old-new) <= 1e-12
        end
        @test TC.TB.season_break_weeks(current,TC_CONFIG) == TC.MID.season_break_weeks(current)
        # Counts are optional, not silently imposed on an arbitrary league configuration.
        tiny = TC.MM.MarketModelConfig(name="toy",segment=TC_CONFIG.segment,tournaments=[56],
            seasons=["24/25"],honest_train=["24/25"],honest_test=String[])
        @test isempty(tiny.expected)
        @test isempty(tiny.excluded_matches)
    end
    @testset "Rotation and positive process covariance" begin
        @test det(TC.ROTATION) == 1.0
        s = 0.07
        @test TC.ROTATION*(s^2*I)*TC.ROTATION' ≈ Diagonal([2s^2,s^2/2])
        p = TC.MID.toy_panel(Xoshiro(3926))
        for level in (5,6,7), z in (-12.0,-4.0,0.0,4.0,12.0)
            a = TC.CovarianceRung(level)
            theta = level == 6 ? log.([0.07,0.05,0.03]) : log.([0.12,0.025,0.05,0.03])
            level in (6,7) && (theta=vcat(theta,z))
            sch = TC.covariance_schedule(a,p,theta)
            Q = TC.MID.process_cov(TC.conditional_arm(a,theta),TC.MID.n_teams(p),sch,2)
            @test isposdef(Symmetric(Q[[3,3+TC.MID.n_teams(p)],[3,3+TC.MID.n_teams(p)]]))
            @test isposdef(Symmetric(TC.side_noise(a,theta)))
        end
        gates = TC.engine_gates_b2()
        show(stdout,MIME"text/plain"(),gates; allrows=true,allcols=true)
        println()
        for r in eachrow(gates)
            @test r.pass
        end
    end
    @testset "Pre-week Gaussian parity and smoothing" begin
        p = TC.MID.toy_panel(Xoshiro(3927))
        theta = log.([0.07,0.05,0.03])
        reference = TC.TB.preweek_predictions(TC.TB.Rung(2),p,theta; mc_draws=400)
        candidate = TC.covariance_predictions(TC.CovarianceRung(6),p,vcat(theta,0.0); mc_draws=400)
        @test abs(reference.loglik-candidate.loglik) <= 1e-9
        @test maximum(abs.(reference.rows.predicted-candidate.rows.predicted)) <= 1e-8
        @test maximum(abs.(reference.rows.variance-candidate.rows.variance)) <= 1e-8
        a = TC.CovarianceRung(7)
        theta7 = vcat(log.([0.12,0.025,0.05,0.03]),atanh(-0.6))
        f = TC.covariance_filter(a,p,theta7; store=true)
        m,P = TC.smoothing_moments(TC.conditional_arm(a,theta7),p,TC.covariance_schedule(a,p,theta7),f)
        batch = TC.batch_gaussian(a,p,theta7)
        @test maximum(abs.(vec(m)-batch.post_mean)) <= 1e-8
        n = size(m,1)
        @test maximum(maximum(abs.(P[:,:,t]-batch.post_cov[(t-1)*n+1:t*n,(t-1)*n+1:t*n])) for t in 1:p.n_weeks) <= 1e-8
    end
    @testset "Persistence distinguishes planted team and fixture shocks" begin
        persistent = TC.persistence_statistic(tc_synthetic_events(true))
        iid = TC.persistence_statistic(tc_synthetic_events(false))
        println("Synthetic persistent: ",persistent)
        println("Synthetic iid fixture: ",iid)
        @test persistent.autocorrelation > 0
        @test persistent.ci_low > 0
        @test iid.ci_low <= 0 <= iid.ci_high
        @test abs(iid.autocorrelation) < 0.05
        # No across-season pair and deterministic tie handling.
        events = DataFrame(team=fill("A",4),season=["s1","s1","s2","s2"],
            date=Date(2024,1,1).+Day.([2,1,4,3]),match_id=[2,1,4,3],residual=[2.0,1.0,4.0,3.0])
        pairs = TC.persistence_pairs(events)
        @test pairs.previous == [1.0,3.0]
        @test pairs.current == [2.0,4.0]
    end
    @testset "Human addendum: quality/style rotation and common weekly shocks" begin
        A = [0.5 -0.5; 0.5 0.5]
        for rho in (-0.8,0.0,0.8)
            sa,sb = 0.04,0.03
            Q = [sa^2 rho*sa*sb; rho*sa*sb sb^2]
            rotated = A*Q*A'
            vq = (sa^2+sb^2-2rho*sa*sb)/4
            vs = (sa^2+sb^2+2rho*sa*sb)/4
            @test diag(rotated) ≈ [vq,vs]
            @test rotated[1,2]/sqrt(vq*vs) ≈ (sa^2-sb^2)/(4sqrt(vq*vs))
        end
        # Balanced synthetic panels with 200 weeks and 8 independent fixtures per week.
        weeks = repeat(collect(1:200); inner=8)
        rng = Xoshiro(3929)
        common = randn(rng,200)
        fixture_noise = randn(rng,length(weeks))
        planted = TC.league_shock_statistic(weeks,common[weeks]+fixture_noise)
        none = TC.league_shock_statistic(weeks,fixture_noise)
        println("Synthetic common weekly shock: ",planted)
        println("Synthetic no weekly shock: ",none)
        @test planted.icc > 0
        @test planted.ci_low > 0
        @test abs(none.icc) < 0.05
        @test none.ci_low <= 0 <= none.ci_high
        # Do not truncate negative MOM components; repeated bootstrap weeks remain distinct.
        signed = TC.week_icc([[-1.0,1.0],[-1.0,1.0]])
        @test signed.icc == -1.0
        @test signed.variance_week == -1.0
        unbalanced = TC.week_icc([[1.0],[2.0,3.0],[4.0,5.0,6.0]])
        @test unbalanced.n == 6
        @test unbalanced.n_weeks == 3
        @test unbalanced.n_multi_fixture_weeks == 2
        @test unbalanced.n_pairs == 4
    end
end
println("T03_DONE")
