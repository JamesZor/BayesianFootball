# Deterministic preparation only. No recovery or posterior promotion.
using Test, LinearAlgebra, Distributions, DataFrames, Random, BayesianFootball, CSV
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__,"l05_pooled.jl"))
end
if !isdefined(@__MODULE__, :FullBookWorkflow)
    include(joinpath(@__DIR__,"l05_fullbook_workflow.jl"))
end
const WF05 = FullBookWorkflow
const WP05 = PooledMarket

@testset "Full-book workflow preparation (not posterior acceptance)" begin
    p = WP05.MID.toy_panel(Random.Xoshiro(3951))
    @testset "C0 forecasts equal matched R6, all B2 axes and RNG convention" begin
        theta = log.([0.07,0.04,0.015])
        c0 = WF05.predictions(WP05.FullBookRung(:C0),p,theta; seed=3906)
        r6 = WP05.CM.covariance_predictions(WP05.CM.CovarianceRung(6),p,
            WP05.c0_parameters(exp.(theta)...); seed=3906)
        @test c0.rows.match_id == r6.rows.match_id
        @test c0.rows.axis == r6.rows.axis
        for name in (:predicted,:variance,:logpd)
            @test maximum(abs,c0.rows[!,name]-r6.rows[!,name]) <= 1e-9
        end
        @test c0.rows.cover90 == r6.rows.cover90
        @test abs(c0.loglik-r6.loglik) <= 1e-9
        raw = copy(c0.rows)
        raw.rung .= "C0"
        raw.protocol .= "10b"
        candidate = copy(raw)
        candidate.rung .= "C1"
        candidate.logpd .+= 0.1
        paired = WF05.paired_scores(vcat(raw,candidate))
        @test only(paired.n) == WP05.MID.n_fixtures(p)
        @test abs(only(paired.mean_delta)-0.2) <= 1e-14
        @test only(paired.se) <= 1e-14
        @test_throws ErrorException WF05.paired_scores(vcat(raw,candidate[Not(3),:]))
    end
    @testset "Conditional theta/u smoothing vs independent augmented joint Gaussian" begin
        theta = log.([0.04,0.015,0.05,1000.0])
        likelihoods = [x -> logpdf(MvNormal(p.obs_y[j:j+1],0.03^2*Matrix(I,2,2)),x)
            for j in 1:2:WP05.MID.n_obs(p)]
        f = WP05.fullbook_filter(WP05.FullBookRung(:C1),p,theta;
            store=true,predict=true,likelihoods)
        means,covs = WP05.fullbook_smoothing(f)
        conditional = WF05.fixture_moments(WP05.FullBookRung(:C1),p,f,means,covs)
        batch = WF05.augmented_batch(p,theta,f)
        for (fixture,row) in enumerate(conditional)
            ix = (2*fixture-1):(2*fixture)
            B = batch.design[ix,:]
            uix = batch.n*batch.T .+ ix
            @test maximum(abs,row.theta_mean-B*batch.mean) <= 1e-8
            @test maximum(abs,row.theta_cov-B*batch.covariance*B') <= 1e-8
            @test maximum(abs,row.u_mean-batch.mean[uix]) <= 1e-8
            @test maximum(abs,row.u_cov-batch.covariance[uix,uix]) <= 1e-8
            @test minimum(eigvals(Symmetric(row.theta_cov))) >= -1e-10
        end
        # No own-book Hessian is added to forecast variance. Initial prediction
        # is the independent Gaussian structure plus the ONE fresh fixture u.
        m0,P0 = WP05.fullbook_initial(WP05.FullBookRung(:C1),WP05.MID.n_teams(p))
        @test maximum(abs,f.pair_mean[1,:]-f.H[1:2,:]*m0) <= 1e-12
        @test maximum(abs,f.pair_cov[1,:,:]-f.H[1:2,:]*P0*f.H[1:2,:]'-f.noise) <= 1e-12
        # Perturb EVERY book in one week: all predictions within it stay fixed,
        # but later weeks must respond. Keep IDs/population/parameters identical.
        week = minimum(p.obs_week)
        changed = [x -> logpdf(MvNormal(p.obs_y[j:j+1] .+
            (p.obs_week[j] == week ? 0.2 : 0.0),0.03^2*Matrix(I,2,2)),x)
            for j in 1:2:WP05.MID.n_obs(p)]
        perturbed = WP05.fullbook_filter(WP05.FullBookRung(:C1),p,theta;
            predict=true,likelihoods=changed)
        keep = p.obs_week[1:2:end] .== week
        @test maximum(abs,f.pair_mean[keep,:]-perturbed.pair_mean[keep,:]) <= 1e-12
        @test maximum(abs,f.pair_cov[keep,:,:]-perturbed.pair_cov[keep,:,:]) <= 1e-12
        @test maximum(abs,f.pair_mean[.!keep,:]-perturbed.pair_mean[.!keep,:]) > 0.01
    end
    @testset "Report schema smoke with IID mock hyperdraws (NOT posterior fits)" begin
        p.matches.season .= "toy"
        config = WP05.MM.MarketModelConfig(name="toy",segment=BayesianFootball.Data.ScottishLower(),
            tournaments=[1],seasons=["toy"],honest_train=["train"],honest_test=["toy"])
        markets = [[(; selections=[:home,:draw,:away],p=[q[:home],q[:draw],q[:away]]),
                    (; selections=[:over_25,:under_25],p=[q[:over_25],q[:under_25]])]
            for q in (WP05.differentiable_probabilities(p.obs_y[j:j+1]) for j in 1:2:WP05.MID.n_obs(p))]
        fits = Dict{Tuple{Symbol,String},WP05.MID.ArmFit}()
        rng = Random.Xoshiro(3966)
        for rung in (:C0,:C1), protocol in ("10a","10b")
            a = WP05.FullBookRung(rung)
            centre = WP05.MID.init_centre(a)
            U = 0.01randn(rng,1000,length(centre),4) .+ reshape(centre,1,:,1)
            fits[(rung,protocol)] = WP05.MID.ArmFit(a,WP05.MID.param_names(a),
                WP05.fullbook_physical_draws(a,U),U,Matrix{Float64}[],Vector{Float64}[],zeros(0,0),0.0)
        end
        mktempdir() do out
            raw = WF05.evaluate_fits(p,markets,config,fits,out)
            @test nrow(raw) == 4*5*WP05.MID.n_fixtures(p)
            paired = CSV.read(joinpath(out,"paired_vs_c0.csv"),DataFrame)
            @test paired.n == fill(WP05.MID.n_fixtures(p),2)
            @test all(isfinite,paired.mean_delta)
            for file in ("onestep_metrics_c.csv","smoothed_fit_c.csv","ladder_summary_c.csv",
                         "team_paths_c.csv","parameter_posteriors_c.csv","shrinkage_fixture_c.csv",
                         "shrinkage_by_type.csv","book_noise_c.csv")
                @test nrow(CSV.read(joinpath(out,file),DataFrame)) > 0
            end
            @test all(CSV.read(joinpath(out,"book_noise_c.csv"),DataFrame).book_type .== "full")
            @test_throws SystemError WF05.require_recovery(out)
            write(joinpath(out,"synthetic_panel.jls"),"not the frozen panel")
            @test_throws ErrorException WF05.require_recovery(out; expected_panel_sha=repeat("0",64))
        end
    end
end
println("T05_WORKFLOW_PREPARATION_DONE")
