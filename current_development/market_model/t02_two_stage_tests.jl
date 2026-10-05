using Test, LinearAlgebra, Random, Statistics
using BayesianFootball, DataFrames
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l02_two_stage.jl"))
const TB = TwoStageMarket
const TBMID = TB.MID

@testset "Phase B two-stage contracts" begin
    @testset "Pinned Phase B panel" begin
        ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours=10^6)
        built = TB.phase_b_panel(ds)
        @test nrow(built.panel.matches) == 517
        @test TBMID.n_obs(built.panel) == 1034
        @test count(built.frame.accepted) == 517
        @test nrow(built.exclusions) == 78
        @test all(built.frame.n_targets[built.frame.accepted] .>= 5)
        byid = Dict(r.match_id => r.kl for r in eachrow(built.rates))
        @test all(r.sse == byid[r.match_id] for r in eachrow(built.panel.matches))
        @test count(==(TB.EXCLUSION), built.panel.refusals.reason) == 78
        home_panel = TB.home_panel(built.panel)
        @test all(iszero,home_panel.obs_X[2:2:end,:])
        @test all(sum(home_panel.obs_X[j,:]) == 1 for j in 1:2:1034)
        @test all(home_panel.obs_X[j,built.panel.obs_att[j]] == 1 for j in 1:2:1034)
        @test built.panel.matches.match_id == built.panel.obs_match[1:2:end]
        folds = TB.fixture_folds(built.panel)
        @test sort(unique(folds)) == collect(1:10)
        obs_folds = repeat(folds; inner=2)
        @test length(obs_folds) == 1034
        @test all(obs_folds[1:2:end] .== obs_folds[2:2:end])
        @test maximum(count(==(f), folds) for f in 1:10) - minimum(count(==(f), folds) for f in 1:10) <= 1
    end
    @testset "Independent Gaussian gates" begin
        gates = vcat(TB.engine_gates(), TB.rbpf_gate())
        show(stdout, MIME"text/plain"(), gates; allrows=true, allcols=true)
        println()
        for row in eachrow(gates)
            @test row.pass
        end
    end
    @testset "Gaussian pre-week predictive parity" begin
        p = TB.phase_b_panel(BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours=10^6)).panel
        for level in 0:3
            a = TB.Rung(level)
            theta = TBMID.init_centre(a)
            pp = level >= 3 ? TB.home_panel(p) : p
            reference = TB.rung_filter(a,pp,theta; predict=true)
            prediction = TB.preweek_predictions(a,p,theta; mc_draws=100)
            @test abs(reference.loglik-prediction.loglik) <= 1e-8
            @test maximum(abs.(reference.pred_mean.-prediction.pred_mean)) <= 1e-8
            @test maximum(abs.(reference.pred_var.-prediction.pred_var)) <= 1e-8
            @test nrow(prediction.rows) == 517*5
            summary = TB.prediction_summary(p,prediction,"10b","R$level")
            @test nrow(summary) == 20
            @test Set(summary.axis) == Set(TB.AXES)
        end
    end
    @testset "Zero in-season R1 paths" begin
        p0 = TBMID.toy_panel(Xoshiro(3904))
        p = TBMID.MarketPanel(p0.matches, p0.refusals, p0.teams, p0.n_weeks, p0.week_start,
            p0.obs_week, p0.obs_home, p0.obs_att, p0.obs_def, p0.obs_y, p0.obs_match,
            [w < 5 ? "24/25" : "25/26" for w in p0.obs_week], p0.week_ptr, p0.obs_X, p0.feature_names)
        a = TB.Rung(1)
        theta = log.([0.1, 0.2])
        sch = TB.rung_schedule(a, p, theta)
        f = TB.rung_filter(a, p, theta; store=true)
        for seed in 1:5
            path = TBMID.ffbs(a, p, sch, f, Xoshiro(seed))
            @test all(iszero, diff(path[:, 1:4]; dims=2))
            @test all(iszero, diff(path[:, 5:7]; dims=2))
        end
    end
    @testset "Observation families and compiled AD" begin
        for (eta,sigma,nu) in ((0.3,0.1,3.0),(-0.5,0.5,5.0),(0.8,0.7,8.0))
            @test mean(TB.DS.Gamma(nu,exp(eta)/nu)) ≈ exp(eta)
            z = collect(range(-80,80; length=100001))
            y = exp.(z)
            for family in (:lognormal,:logt)
                ll = TB.family_ll(Val(family),y,z,fill(eta,length(z)),(; sigma,nu))
                # Integrate on λ scale via λ=exp(z), dλ=exp(z) dz.
                integrand = exp.(ll.+z)
                integral = step(range(-80,80; length=100001))*(sum(integrand)-(first(integrand)+last(integrand))/2)
                @test abs(integral-1) <= 1e-6
            end
        end
        gates = TB.family_gradient_gates(TB.family_data(TBMID.toy_panel(Xoshiro(3908))))
        show(stdout,MIME"text/plain"(),gates; allrows=true,allcols=true)
        println()
        for row in eachrow(gates)
            @test row.pass
        end
    end
    @testset "Deterministic chain family extraction" begin
        data = TB.family_data(TBMID.toy_panel(Xoshiro(3913)))
        for family in (:gamma,:lognormal,:logt)
            names = vcat(["mu","gamma","sigma_alpha","sigma_beta"],
                         ["z_alpha[$i]" for i in 1:data.n],["z_beta[$i]" for i in 1:data.n],
                         family == :gamma ? ["noise.nu"] : family == :lognormal ? ["noise.sigma"] : ["noise.sigma","noise.nu"])
            values = vcat([0.3,0.2,0.15,0.12],collect(1:data.n).*0.1,collect(1:data.n).*(-0.05),
                          family == :gamma ? [100.0] : family == :lognormal ? [0.2] : [0.2,5.0])
            array = repeat(reshape(values,1,length(values),1),3,1,2)
            chain = TB.MCMCChains.Chains(array,names)
            result = TB.family_loglik(chain,data,family)
            @test size(result.ll) == (length(data.y),6)
            eta = 0.3 .+ 0.2.*data.home .+ (0.15.*(data.C*(collect(1:data.n).*0.1)))[data.att] .+
                  (0.12.*(data.C*(collect(1:data.n).*(-0.05))))[data.def]
            noise = family == :gamma ? (; nu=100.0) : family == :lognormal ? (; sigma=0.2) : (; sigma=0.2,nu=5.0)
            expected = TB.family_ll(Val(family),data.y,data.logy,eta,noise)
            @test result.ll ≈ repeat(expected,1,6)
            @test abs(TB.waic_table(result,family).p_waic) <= 1e-20
        end
    end
    @testset "Divergences must include internal section" begin
        array = zeros(1000,2,4)
        array[:,1,:] .= randn(Xoshiro(3914),1000,4)
        array[1:5,2,1] .= 1
        chain = TB.MCMCChains.Chains(array,["mu","numerical_error"],Dict(:internals=>["numerical_error"]))
        _,diagnostic = TB.family_diagnostics(chain,:gamma,0)
        @test diagnostic.divergences == 5
        @test diagnostic.divergence_rate == 5/4000
        @test !diagnostic.gate_pass
    end
    @testset "Poisson IRLS" begin
        rng = Xoshiro(3905)
        x = randn(rng, 10000)
        truth = [0.2, 0.8]
        y = rand.(Ref(rng), TB.DS.Poisson.(exp.(truth[1] .+ truth[2].*x)))
        fit = TB.poisson_glm(y, x)
        @test all(abs.(fit.coef .- truth) .<= 3fit.se)
    end
end
println("T02_DONE")
