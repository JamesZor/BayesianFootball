# ==============================================================================
# 13 — Stage 5 deterministic pedigree/tier mock verification
# ==============================================================================
#
# This is a mathematical shape and gradient check, not fitting, EDA extraction,
# or evidence that professionalism explains the live-slate discrepancy. It uses a
# synthetic, finite, point-in-time-like fixture set and launches no MCMC.
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

using LinearAlgebra

# `log(n!)` is precomputed data in a production builder. Avoiding an additional
# package keeps this isolated deterministic mock runnable in minimal environments.

include(joinpath(@__DIR__, "l04_pedigree_tier_components.jl"))
using .PedigreeTierMock

# %%
# ===================================================================
# 2. Research contract
# ===================================================================

# The prior score below is predeclared synthetic metadata. In production all tier,
# status and transition records must be selected as of each kickoff. It is never
# legitimate to replace them with final standings or a later club announcement.
# This runner checks the factor-of-two convention and AD shape only.

const P13_TIER = AdjacentTierPrior(
    tier_4 = 0.0,
    increment_12 = 0.30,
    increment_23 = 0.22,
    increment_34 = 0.16,
)
const P13_PRIOR = StatusPedigreePrior(
    status_coefficient = 0.18,
    pedigree_coefficient = 0.35,
    residual_scale = 0.30,
)
const P13_COVARIATE = SupremacyCovariate(coefficient = 0.18)
const P13_GRADIENT_TOLERANCE = 1e-8

# %%
# ===================================================================
# 3. Synthetic causal feature snapshot
# ===================================================================

const P13_TIER_RANK = Int[2, 3, 3, 4]
const P13_STATUS = Float64[1.0, 1.0, 0.0, 0.0]
const P13_PEDIGREE = Float64[0.70, 0.20, -0.10, -0.45]
const P13_PRIOR_MEAN = centred_prior_mean(
    P13_TIER_RANK,
    P13_STATUS,
    P13_PEDIGREE,
    P13_TIER,
    P13_PRIOR,
)

const P13_HOME_IDS = Int[1, 2, 3, 4, 1, 3]
const P13_AWAY_IDS = Int[3, 4, 1, 2, 4, 2]
const P13_HOME_GOALS = Int[2, 1, 0, 1, 3, 0]
const P13_AWAY_GOALS = Int[0, 0, 1, 2, 1, 1]
const P13_MATCH_WEIGHTS = ones(Float64, length(P13_HOME_IDS))
const P13_LOG_FACT_H = Float64[log(factorial(y)) for y in P13_HOME_GOALS]
const P13_LOG_FACT_A = Float64[log(factorial(y)) for y in P13_AWAY_GOALS]

# `supremacy_design` divides by two. The engine's `(q, -q)` placement then
# produces exactly one coefficient times the home-away status contrast.
const P13_STATUS_MATCH = supremacy_design(
    P13_STATUS[P13_HOME_IDS],
    P13_STATUS[P13_AWAY_IDS],
)

# %%
# ===================================================================
# 4. Mathematical invariants
# ===================================================================

@assert isapprox(sum(P13_PRIOR_MEAN), 0.0; atol = 1e-14)
@assert isapprox(P13_STATUS_MATCH[1], 0.5; atol = 0.0)
@assert isapprox(2.0 * P13_COVARIATE.coefficient * P13_STATUS_MATCH[1],
                 P13_COVARIATE.coefficient * (P13_STATUS[1] - P13_STATUS[3]);
                 atol = 0.0)

# Zero-sum centring does not cap contrasts: make the first club arbitrarily
# strong in the supplied causal prior and its contrast remains arbitrarily large.
p13_large = centred_prior_mean(Int[1, 4], Float64[0.0, 0.0],
                                Float64[100.0, 0.0], P13_TIER, P13_PRIOR)
@assert p13_large[1] - p13_large[2] > 30.0

# %%
# ===================================================================
# 5. ReverseDiff / ForwardDiff deterministic gradient gate
# ===================================================================

const P13_THETA = Float64[0.10, 0.15, -0.10, 0.05, -0.10,
                           -0.10, 0.10, -0.05, 0.05]

p13_logjoint = θ -> mock_logjoint(
    θ,
    P13_HOME_IDS,
    P13_AWAY_IDS,
    P13_HOME_GOALS,
    P13_AWAY_GOALS,
    P13_PRIOR_MEAN,
    P13_PRIOR.residual_scale,
    P13_MATCH_WEIGHTS,
    P13_LOG_FACT_H,
    P13_LOG_FACT_A,
    P13_STATUS_MATCH,
    P13_COVARIATE.coefficient,
)

p13_gradient = gradient_report(p13_logjoint, P13_THETA)
@assert p13_gradient.compiled_fresh <= P13_GRADIENT_TOLERANCE
@assert p13_gradient.compiled_forward <= P13_GRADIENT_TOLERANCE
@assert p13_gradient.perturbed_compiled_fresh <= P13_GRADIENT_TOLERANCE

# %%
# ===================================================================
# 6. Warmed allocation measurement
# ===================================================================

p13_allocations = warmed_allocation_report(p13_logjoint, P13_THETA)

# %%
# ===================================================================
# 7. Final report
# ===================================================================

println("Stage 5 pedigree/tier mock: PASS")
println("  tier means (1→4):             ", adjacent_tier_means(P13_TIER))
println("  centred prior means:           ", P13_PRIOR_MEAN)
println("  tape instructions:             ", p13_gradient.instructions)
println("  compiled vs fresh ReverseDiff: ", p13_gradient.compiled_fresh)
println("  compiled vs ForwardDiff:       ", p13_gradient.compiled_forward)
println("  perturbed compiled vs fresh:   ", p13_gradient.perturbed_compiled_fresh)
println("  warmed bytes / gradient:       ", p13_allocations.bytes_per_gradient)
println("  Note: tape-compatible does not mean allocation-free whole program.")
