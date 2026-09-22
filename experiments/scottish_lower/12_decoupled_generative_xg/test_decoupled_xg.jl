# Deterministic contract tests; no database, real data snapshot, or sampling.
using Test, Distributions, LinearAlgebra
include(joinpath(@__DIR__, "l12_loader.jl"))
const D = DecoupledGenerativeXG

@testset "Four-arm architecture" begin
    references = Dict(D.reference_models())
    optimized = Dict(D.models())
    @test references["m01_poisson_time_decay"].observation isa D.PoissonObservation
    @test references["m02_joint_gamma_poisson"].observation isa D.JointGammaPoissonObservation
    # m03/m04 are CUT models, not `PoissonCountModel`s: the funnel arms are a
    # two-stage modular posterior, so their identity lives in `CutFunnelModel`.
    m03 = references["m03_funnel_shared_kappa"]
    m04 = references["m04_funnel_hierarchical_kappa"]
    @test m03 isa D.CutFunnelModel
    @test m04 isa D.CutFunnelModel
    @test m03.observation isa D.B.SharedKappaJoint
    @test m04.observation isa D.B.HierarchicalKappaJoint
    @test all(model.guard isa D.ArrayClampGuard
              for model in values(optimized) if model isa D.B.PoissonCountModel)
    @test all(model.chance.guard isa D.ArrayClampGuard
              for model in values(optimized) if model isa D.CutFunnelModel)
    @test isempty(m03.chance.covariates)
    @test m03.chance.dynamics.days_half_life == 180.0
end

# The previous "m02 and m03 stated laws are identical" testset asserted the very
# defect this work removes: m03 WAS m02 re-registered under another name, so the
# suite ran the same posterior twice and called one of them decoupled. The
# replacement asserts the opposite property — that the two laws now differ, and
# differ in the one specific way that matters.
@testset "m03 is a cut posterior, not a second copy of m02" begin
    arms = Dict(D.reference_models())
    m02 = arms["m02_joint_gamma_poisson"]
    m03 = arms["m03_funnel_shared_kappa"]
    @test typeof(m02) != typeof(m03)
    # The chance layer shares m02's structural recipe, so a rating difference is
    # attributable to the cut rather than to a different linear predictor …
    @test string(m02.interception) == string(m03.chance.interception)
    @test string(m02.home_advantage) == string(m03.chance.home_advantage)
    @test m02.dynamics.days_half_life == m03.chance.dynamics.days_half_life
    @test string(m02.observation) == string(m03.observation)
    # … except for the innovation priors, which TODO 025 widens deliberately.
    @test m03.chance.dynamics.σ_att isa Truncated
    @test std(m03.chance.dynamics.σ_att.untruncated) == 0.20
    @test mean(m03.observation.log_kappa_prior) == 0.0
    @test std(m03.observation.log_kappa_prior) == 0.2
end

@testset "Cut structure: goals cannot reach the chance layer" begin
    # The chance engine's sampled sites are ratings + ν, and NOTHING about κ: κ is a
    # Stage B parameter, so it must be absent from Stage A's density entirely.
    m03 = D.cut_model(D.shared_observation())
    m04 = D.cut_model(D.hierarchical_observation())
    for m in (m03, m04)
        @test m.chance.observation === m.observation
        @test m.chance isa D.B.PoissonCountModel
    end
    # Stage B's design reduces the goal likelihood to per-team sufficient statistics;
    # A_i is data-only, so it must not depend on any chance draw.
    z = (; n_teams = 3, n_matches = 2,
           home_ids = [1, 2], away_ids = [2, 3],
           home_goals = [2.0, 0.0], away_goals = [1.0, 3.0],
           match_weights = [1.0, 0.5])
    team_goals = zeros(3)
    for m in 1:2
        team_goals[z.home_ids[m]] += z.match_weights[m] * z.home_goals[m]
        team_goals[z.away_ids[m]] += z.match_weights[m] * z.away_goals[m]
    end
    @test team_goals ≈ [2.0, 1.0, 1.5]
    @test sum(team_goals) ≈ sum(z.match_weights .* (z.home_goals .+ z.away_goals))
end

@testset "Exact shared-κ conditional matches its analytic law" begin
    # With a diffuse prior the conditional posterior on κ is exactly
    # Gamma(S + 1, 1/T). This is an ANALYTIC check on the grid sampler, independent
    # of any other part of the file.
    n = 60
    z = (; n_teams = 4, n_matches = n,
           home_ids = repeat(1:4, inner = 15), away_ids = repeat(1:4, outer = 15),
           home_goals = Float64[(1.0, 2.0, 0.0)[mod1(i, 3)] for i in 1:n],
           away_goals = Float64[(0.0, 1.0, 2.0)[mod1(i, 3)] for i in 1:n],
           match_weights = fill(1.0, n),
           log_mu_h = fill(log(1.25), n), log_mu_a = fill(log(1.05), n))
    st = D.cut_shared_sufficient(z)
    @test st.S ≈ sum(z.home_goals) + sum(z.away_goals)
    @test st.T ≈ n * (1.25 + 1.05)

    report = D.cut_verify_exact_shared(z; n = 20_000, seed = 4)
    # The deterministic quantile comparison is the real assertion: it has no Monte
    # Carlo floor, so it can be held to a tight tolerance.
    @test report.worst_quantile_rel_error < 1.0e-3
    # The sampled moments are a coarse cross-check; at n = 20k their own sampling
    # error is ~1/√(2n) ≈ 0.5% on the sd, so the tolerance reflects that and not the
    # grid's accuracy.
    @test report.mean_rel_error < 1.0e-2
    @test report.sd_rel_error < 2.0e-2
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
