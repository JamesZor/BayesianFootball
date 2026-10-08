# Revision-4 full-book deterministic tests. These are necessary checks, not a
# claim to synthetic-panel recovery or completed Phase C acceptance.
# C2-pending checks execute separately and report failures, excluded by manager scope.
using Test, LinearAlgebra, Distributions, DataFrames, Random, CSV, BayesianFootball
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__, "l05_pooled.jl"))
end
const PC05 = PooledMarket

@testset "Phase C revision 4 full-book deterministic engines" begin
    @testset "Gaussian likelihood: integral and moments independently known" begin
        for k in (1,2)
            a = fill(0.15,k)
            z = fill(-0.10,k)
            S = 0.20^2*Matrix(I,k,k)
            R = 0.08^2*Matrix(I,k,k)
            f = x -> logpdf(MvNormal(z,Symmetric(R)),x)
            L = PC05.laplace_update(f,a,S)
            Q = PC05.quadrature_moments(f,a,S; order=32)
            variance = inv(inv(S)+inv(R))
            mean = variance*(S\a+R\z)
            exact = logpdf(MvNormal(a,Symmetric(S+R)),z)
            @test abs(L.marginal-exact) <= 1e-7
            @test abs(Q.marginal-exact) <= 1e-12
            @test norm(L.mean-mean) <= 1e-7
            @test norm(Q.mean-mean) <= 1e-12
            @test norm(L.covariance-variance) <= 1e-9
            @test norm(Q.covariance-variance) <= 1e-12
        end
    end
    @testset "Exact Beta normalisers and reversed large-n KL" begin
        theta = log.([2.0,1.0])
        b = PC05.PF.synthetic_total_book(2.0,1.0)
        markets = PC05.PF.market_vectors(b)
        p = first(b.prob_fair_close)
        @test abs(PC05.PF.book_logdensity(theta,markets,1000.0)-
            logpdf(Beta(1000p,1000(1-p)),p)) <= 1e-10
        b.prob_fair_close .= [0.4,0.6]
        markets = PC05.PF.market_vectors(b)
        x = log.([1.5,1.0])
        n = 1e6
        slope = (PC05.PF.book_logdensity(x,markets,n)-
            PC05.PF.book_logdensity(theta,markets,n))/n
        limit = -PC05.PF.reverse_kl(x,markets)+PC05.PF.reverse_kl(theta,markets)
        @test abs(slope-limit) <= 1e-6
    end
    @testset "Synthetic total book narrow prediction quadrature" begin
        a = log.([2.0,1.0])
        markets = PC05.PF.market_vectors(PC05.PF.synthetic_total_book(2.0,1.0))
        f = x -> PC05.PF.book_logdensity(x,markets,1000.0)
        S = 0.05^2*Matrix(I,2,2)
        adf = x -> PC05.differentiable_logdensity(x,markets,1000.0)
        L = PC05.laplace_update(f,a,S; third_likelihood=adf)
        Q = PC05.quadrature_moments(f,a,S; order=32)
        Q64 = PC05.quadrature_moments(f,a,S; order=64)
        @test abs(Q.marginal-Q64.marginal) <= 1e-8
        @test abs(L.marginal-Q64.marginal) <= 0.01
        @test maximum(abs.(L.mean-Q64.mean)./sqrt.(diag(Q64.covariance))) <= 0.05
        @test maximum(abs.(sqrt.(diag(L.covariance))./sqrt.(diag(Q64.covariance)).-1)) <= 0.05
    end
    @testset "Cubic skewness contraction and information-centre identity" begin
        Sigma = [0.03 0.005; 0.005 0.02]
        f = x -> -sum(x.^2)/2+0.2x[1]^3+0.1x[1]*x[2]^2
        T = PC05.third_ad(f,[0.0,0.0])
        expected = 0.5Sigma*[1.2Sigma[1,1]+0.2Sigma[2,2],0.4Sigma[1,2]]
        @test norm(PC05.skewness_shift(Sigma,T)-expected) <= 1e-14
        @test norm(T-PC05.third_fd(f,[0.0,0.0])) <= 1e-10
        @test PC05.skewness_shift(reshape([0.04],1,1),reshape([2.0],1,1,1)) == [0.0016]
    end
    @testset "Production-grid derivative parity: first 20 full books" begin
        config = PC05.MM.scottish_lower_2425_2526()
        ds = BayesianFootball.Data.load_datastore_cached(config.segment; max_age_hours=10^6)
        book,_ = PC05.MM.gated_close(ds,config)
        books = Dict(Int(first(g.match_id))=>DataFrame(g) for g in groupby(book,:match_id))
        rates = CSV.read(joinpath(@__DIR__,"results","A","rates.csv"),DataFrame)
        full = sort(filter(r->r.accepted && r.n_selections>=config.min_selections_ladder,rates),:match_id)
        for r in eachrow(full[1:20,:])
            theta = log.([r.lambda_h,r.lambda_a])
            raw = PC05.MM.selection_probabilities(theta)
            generic = PC05.differentiable_probabilities(theta)
            @test maximum(abs(raw[s]-generic[s]) for s in keys(raw)) <= 5e-14
            markets = PC05.PF.market_vectors(books[r.match_id])
            f = x -> PC05.PF.book_logdensity(x,markets,1000.0)
            adf = x -> PC05.differentiable_logdensity(x,markets,1000.0)
            @test abs(f(theta)-adf(theta)) <= 5e-9
            _,Hfd = PC05.derivatives(f,theta)
            _,Had = PC05.ad_derivatives(adf,theta)
            @test norm(Hfd-Had)/norm(Had) <= 1e-6
            L = PC05.laplace_update(f,theta,0.20^2*Matrix(I,2,2); third_likelihood=adf)
            T = PC05.third_ad(adf,L.mode)
            @test norm(T-PC05.third_fd(adf,L.mode))/norm(T) <= 1e-6
            @test norm(L.covariance*(L.information+theta/(0.20^2))-L.mean) <= 1e-12
        end
    end
    @testset "Quality/style rotation equals matched R6" begin
        p = PC05.MID.toy_panel(Random.Xoshiro(3950))
        theta = PC05.c0_parameters(0.07,0.04,0.015)
        N = PC05.MID.n_teams(p)
        arm = PC05.CM.conditional_arm(PC05.CM.CovarianceRung(6),theta)
        schedule = PC05.CM.covariance_schedule(arm,p,theta)
        Q = PC05.MID.process_cov(arm,N,schedule,2)
        rotate = [0.5 -0.5; 0.5 0.5]
        for team in 1:N
            ix = [2+team,2+N+team]
            @test norm(rotate*Q[ix,ix]*rotate'-Diagonal([0.04^2,0.015^2])) <= 1e-15
        end
        scalar = PC05.CM.covariance_filter(PC05.CM.CovarianceRung(6),p,theta)
        batch = PC05.CM.batch_gaussian(PC05.CM.CovarianceRung(6),p,theta)
        @test abs(scalar.loglik-batch.loglik) <= 1e-9
    end
    include(joinpath(@__DIR__, "t05_fullbook_engine_tests.jl"))
    include(joinpath(@__DIR__, "t05_fullbook_sampling_tests.jl"))
    @testset "Revision 7 stopping, polish and fail-loud accounting" begin
        PC05.reset_newton_accounting!()
        flat = x -> 0.0
        tiny = (f,x) -> ([1e-7],reshape([-1.0],1,1))
        stalled = (f,x) -> ([1e-5],reshape([-1.0],1,1))
        small_step = (f,x) -> ([1.0],reshape([-1e10],1,1))
        failed = (f,x) -> ([1e-3],reshape([-1.0],1,1))
        increasing = (f,x) -> ([1e-5+x[1]],reshape([-1.0],1,1))
        @test PC05.joint_mode(flat,[0.0]; derivative=tiny).termination == :decrement
        @test PC05.joint_mode(flat,[0.0]; derivative=stalled).termination == :polished
        @test PC05.joint_mode(flat,[0.0]; derivative=small_step).termination == :step
        rounded = x -> 1e10
        zero_motion = PC05.joint_mode(rounded,[0.0]; derivative=stalled)
        @test zero_motion.termination == :zero_motion_polished
        @test zero_motion.polish_steps <= 3
        @test_throws ErrorException PC05.joint_mode(rounded,[0.0]; derivative=failed)
        @test_throws ErrorException PC05.joint_mode(rounded,[0.0]; derivative=increasing)
        @test_throws ErrorException PC05.joint_mode(flat,[0.0]; derivative=failed)
        @test_throws ErrorException PC05.joint_mode(flat,[0.0]; derivative=increasing)
        mktempdir() do out
            counts = PC05.write_newton_accounting(out; run="unit")
            @test counts.count == [1,1,0,1,1]
            @test maximum(counts.max_decrement) <= 1e-9
            @test counts.polish_steps == [0,0,0,1,1]
            @test maximum(counts.max_polish_steps) <= 3
        end
    end
    PC05.reset_newton_accounting!() # isolate actual regression from mock unit accounting
    include(joinpath(@__DIR__, "t05_newton_regression_tests.jl"))
end
println("T05_C7_DETERMINISTIC_DONE")
include(joinpath(@__DIR__, "t05_c2_pending_tests.jl"))
# T05_DONE remains reserved for the complete full-book recovery/acceptance suite.
