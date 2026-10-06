# Deterministic revision-2 update tests. These are necessary checks, not a
# claim to synthetic-panel recovery or completed Phase C acceptance.
using Test, LinearAlgebra, Distributions, DataFrames, Random
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__, "l05_pooled.jl"))
end
const PC05 = PooledMarket

@testset "Phase C revision 2 deterministic likelihood update" begin
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
        L = PC05.laplace_update(f,a,S)
        Q = PC05.quadrature_moments(f,a,S; order=32)
        Q64 = PC05.quadrature_moments(f,a,S; order=64)
        @test abs(Q.marginal-Q64.marginal) <= 1e-8
        @test abs(L.marginal-Q64.marginal) <= 0.01
        @test maximum(abs.(L.mean-Q64.mean)./sqrt.(diag(Q64.covariance))) <= 0.05
        @test maximum(abs.(sqrt.(diag(L.covariance))./sqrt.(diag(Q64.covariance)).-1)) <= 0.05
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
end
println("T05_UPDATE_DONE")
# T05_DONE is reserved for the complete pooled/recovery suite after gate promotion.
