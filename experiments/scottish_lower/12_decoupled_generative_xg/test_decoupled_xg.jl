# Deterministic contract tests; no database, real data snapshot, or sampling.
using Test, Distributions, LinearAlgebra
include(joinpath(@__DIR__, "l12_loader.jl"))
const D = DecoupledGenerativeXG

@testset "Four-arm architecture" begin
    references = Dict(D.reference_models())
    optimized = Dict(D.models())
    @test references["m01_poisson_time_decay"].observation isa D.PoissonObservation
    @test references["m02_joint_gamma_poisson"].observation isa D.JointGammaPoissonObservation
    @test references["m03_funnel_shared_kappa"].observation isa D.B.SharedKappaJoint
    @test references["m04_funnel_hierarchical_kappa"].observation isa
          D.B.HierarchicalKappaJoint
    @test all(model.guard isa D.ArrayClampGuard for model in values(optimized))
    @test isempty(references["m03_funnel_shared_kappa"].covariates)
    @test references["m03_funnel_shared_kappa"].dynamics.days_half_life == 180.0
end

@testset "m02 and m03 stated laws are identical" begin
    arms = Dict(D.reference_models())
    m02 = arms["m02_joint_gamma_poisson"]
    m03 = arms["m03_funnel_shared_kappa"]
    @test typeof(m02) == typeof(m03)
    @test string(m02.interception) == string(m03.interception)
    @test string(m02.dynamics) == string(m03.dynamics)
    @test string(m02.home_advantage) == string(m03.home_advantage)
    @test string(m02.observation) == string(m03.observation)
    @test mean(m03.observation.log_kappa_prior) == 0.0
    @test std(m03.observation.log_kappa_prior) == 0.2
end

@testset "Hierarchical finishing prior and centring" begin
    observation = D.hierarchical_observation()
    @test observation.kappa isa D.HierarchicalKappa
    @test minimum(observation.kappa.σ_prior) == 0.0
    raw = [-1.0, 0.5, 2.0, -0.25]
    sigma = 0.08
    delta = sigma .* (raw .- mean(raw))
    @test sum(delta) ≈ 0.0 atol = 1.0e-15
    @test all(exp.(0.1 .+ delta) .> 0.0)
end

@testset "Exact finite Poisson score tensor" begin
    grid = D.funnel_score_grid(1.3, 0.9, 1.1, 0.95)
    @test size(grid) == (12, 12)
    @test all(isfinite, grid)
    @test all(>=(0.0), grid)
    λh = 1.3 * 1.1
    λa = 0.9 * 0.95
    expected = [pdf(Poisson(λh), h) * pdf(Poisson(λa), a) for h in 0:11, a in 0:11]
    @test grid ≈ expected atol = 2.0e-16 rtol = 2.0e-15
    @test sum(grid) ≈ cdf(Poisson(λh), 11) * cdf(Poisson(λa), 11) atol = 2.0e-15
    warmed = Matrix{Float64}(undef, 12, 12)
    D.funnel_score_grid!(warmed, 1.3, 0.9, 1.1, 0.95)
    @test @allocated(D.funnel_score_grid!(warmed, 1.3, 0.9, 1.1, 0.95)) == 0
end
