# Mathematical diagnostic tests ONLY; not the unimplemented t05 pooled/recovery suite.
using Test, LinearAlgebra, Distributions, DataFrames
BLAS.set_num_threads(1)
if !isdefined(@__MODULE__, :LaplaceBookPreflight)
    include(joinpath(@__DIR__, "l05_laplace_preflight.jl"))
end
const CPT = LaplaceBookPreflight

@testset "Phase C preflight mathematics (not pooled acceptance)" begin
    @testset "Quadrature and exact Beta identity" begin
        x, w = CPT.normal_quadrature(32)
        @test abs(sum(w)-1) <= 1e-14
        @test abs(sum(w .* x)) <= 1e-14
        @test abs(sum(w .* x.^2)-1) <= 1e-14
        @test abs(sum(w .* x.^4)-3) <= 1e-13
        theta = log.([2.0, 1.0])
        b = CPT.synthetic_total_book(2.0, 1.0)
        markets = CPT.market_vectors(b)
        q = CPT.MM.selection_probabilities(theta)[:under_25]
        p = first(b.prob_fair_close)
        @test abs(CPT.book_logdensity(theta, markets, 1000.0) -
            logpdf(Beta(1000q, 1000(1-q)), p)) <= 1e-10
    end
    @testset "Totals direction is rate-weighted, not equal-weighted in log coordinates" begin
        theta = log.([2.0, 1.0])
        b = CPT.synthetic_total_book(2.0, 1.0)
        H = CPT.hessian_fd(z -> CPT.MM.kl_loss(z, b), theta)
        H2 = CPT.hessian_fd(z -> CPT.MM.kl_loss(z, b), theta; step=2e-4)
        @test norm(H-H2)/norm(H) <= 1e-6
        eigen = LinearAlgebra.eigen(Symmetric(H))
        @test count(eigen.values .> 1e-6maximum(eigen.values)) == 1
        v = eigen.vectors[:, end]
        @test abs(dot(v, [2.0, 1.0]/sqrt(5))) >= 1-1e-8
        angle = acosd(clamp(abs(sum(v)/sqrt(2)), 0, 1))
        @test angle > 10 # This PROVES the brief's <=10 degree requirement fails.
        println("Synthetic total-book angle (brief limit 10 degrees): ", angle)
    end
    @testset "Dirichlet large-n loss has reverse KL orientation" begin
        b = CPT.synthetic_total_book(2.0, 1.0)
        b.prob_fair_close .= [0.4, 0.6]
        markets = CPT.market_vectors(b)
        a, c = log.([1.5, 1.0]), log.([2.0, 1.2])
        n = 1e6
        slope = (CPT.book_logdensity(a, markets, n) - CPT.book_logdensity(c, markets, n))/n
        reverse = -CPT.reverse_kl(a, markets) + CPT.reverse_kl(c, markets)
        forward = -CPT.MM.kl_loss(a, b) + CPT.MM.kl_loss(c, b)
        @test abs(slope-reverse) <= 1e-6
        @test abs(slope-forward) > 1e-4
        println("Exact log-density difference / n: ", slope,
            "; reverse-KL limit: ", reverse, "; brief forward-KL limit: ", forward)
    end
end
println("T05_PREFLIGHT_DONE")
