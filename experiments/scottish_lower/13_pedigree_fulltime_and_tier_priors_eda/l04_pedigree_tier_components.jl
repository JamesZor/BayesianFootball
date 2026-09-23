# ==============================================================================
# 13 — Pedigree / full-time / tier-prior components (Stage 5 mock)
# ==============================================================================
#
# Definitions only. This local module deliberately accepts already-filtered,
# point-in-time Float64 inputs; it neither queries data nor defines production
# Features/PreGame methods. It is therefore a mathematical and AD-shape mock,
# not a claim that the proposed source data contract has been implemented.
# ==============================================================================

module PedigreeTierMock

using ForwardDiff
using LinearAlgebra
using ReverseDiff
using Statistics

export AdjacentTierPrior, StatusPedigreePrior, SupremacyCovariate,
       adjacent_tier_means, centred_prior_mean, supremacy_design,
       mock_logjoint, gradient_report, warmed_allocation_report

"""
    AdjacentTierPrior

Positive adjacent increments make a lower numeric tier rank stronger. `tier_4`
is the reference location; downstream centring removes its common location, so
only tier contrasts identify the within-competition team effects.
"""
Base.@kwdef struct AdjacentTierPrior
    tier_4::Float64 = 0.0
    increment_12::Float64 = 0.30
    increment_23::Float64 = 0.22
    increment_34::Float64 = 0.16
end

"""
    StatusPedigreePrior

Prior-scale coefficients for operational status and causal pedigree. In a real
model these are sampled hyperparameters with regularising priors; fixed values
here make the deterministic gradient comparison concise.
"""
Base.@kwdef struct StatusPedigreePrior
    status_coefficient::Float64 = 0.18
    pedigree_coefficient::Float64 = 0.35
    residual_scale::Float64 = 0.30
end

"""
    SupremacyCovariate

A linear-predictor ablation. `coefficient` multiplies a design column already
divided by two, because `SupremacyRole` placement is `(q, -q)` and hence changes
the home-away log-rate difference by `2q`.
"""
Base.@kwdef struct SupremacyCovariate
    coefficient::Float64 = 0.18
end

"Return tier locations `(tier 1, tier 2, tier 3, tier 4)` from adjacent steps."
function adjacent_tier_means(c::AdjacentTierPrior)
    τ4 = c.tier_4
    τ3 = τ4 + c.increment_34
    τ2 = τ3 + c.increment_23
    τ1 = τ2 + c.increment_12
    return (τ1, τ2, τ3, τ4)
end

"""
    centred_prior_mean(tier_rank, status, pedigree, tier, prior)

Constructs the tier, status and pedigree prior mean, then centres it across
*the supplied fitted competition*. All inputs must be prefiltered as-of the fold
cutoff; this helper cannot establish point-in-time validity itself.
"""
function centred_prior_mean(tier_rank::Vector{Int}, status::Vector{Float64},
                            pedigree::Vector{Float64}, tier::AdjacentTierPrior,
                            prior::StatusPedigreePrior)
    n_teams = length(tier_rank)
    length(status) == n_teams || error("status length must equal tier_rank length")
    length(pedigree) == n_teams || error("pedigree length must equal tier_rank length")
    all(r -> 1 <= r <= 4, tier_rank) || error("tier ranks must lie in 1:4")
    all(isfinite, status) || error("status must be finite Float64 values")
    all(isfinite, pedigree) || error("pedigree must be finite Float64 values")

    τ = adjacent_tier_means(tier)
    raw = Float64[τ[tier_rank[i]] + prior.status_coefficient * status[i] +
                  prior.pedigree_coefficient * pedigree[i] for i in eachindex(tier_rank)]
    return raw .- mean(raw)
end

"""
    supremacy_design(home_value, away_value)

Returns `(home_value - away_value) / 2`. With a composable-engine supremacy role,
adding `w .* x` to home and subtracting it from away produces the intended
`w * (home_value - away_value)` log-rate-difference effect, not twice it.
"""
function supremacy_design(home_value::Vector{Float64}, away_value::Vector{Float64})
    length(home_value) == length(away_value) || error("home and away values must align")
    all(isfinite, home_value) || error("home values must be finite")
    all(isfinite, away_value) || error("away values must be finite")
    return (home_value .- away_value) ./ 2.0
end

# θ = (intercept, attack residuals..., defence residuals...). The residuals are
# centred to preserve the existing zero-sum convention. `m / 2` and `-m / 2`
# mean the expected log-rate difference is m_home - m_away, exactly once.
function mock_logjoint(θ, home_ids::Vector{Int}, away_ids::Vector{Int},
                       home_goals::Vector{Int}, away_goals::Vector{Int},
                       prior_mean::Vector{Float64}, residual_scale::Float64,
                       match_weights::Vector{Float64}, log_fact_h::Vector{Float64},
                       log_fact_a::Vector{Float64}, covariate_column::Vector{Float64},
                       covariate_weight::Float64)
    n_teams = length(prior_mean)
    n_matches = length(home_ids)
    length(θ) == 1 + 2 * n_teams || error("θ must hold intercept plus attack/defence residuals")
    all(v -> length(v) == n_matches,
        (away_ids, home_goals, away_goals, match_weights, log_fact_h, log_fact_a,
         covariate_column)) ||
        error("match-level design vectors must align")
    residual_scale > 0.0 || error("residual_scale must be positive")

    intercept = θ[1]
    raw_attack = θ[2:(1 + n_teams)]
    raw_defence = θ[(2 + n_teams):end]
    attack = raw_attack .- mean(raw_attack) .+ prior_mean ./ 2.0
    defence = raw_defence .- mean(raw_defence) .- prior_mean ./ 2.0

    # Input column must have been constructed outside this function, using only
    # pre-kickoff metadata. No branch or observation loop occurs in this hot path.
    q = covariate_weight .* covariate_column
    η_h = intercept .+ attack[home_ids] .+ defence[away_ids] .+ q
    η_a = intercept .+ attack[away_ids] .+ defence[home_ids] .- q
    ll_h = home_goals .* η_h .- exp.(η_h) .- log_fact_h
    ll_a = away_goals .* η_a .- exp.(η_a) .- log_fact_a
    loglik = sum(ll_h .* match_weights) + sum(ll_a .* match_weights)
    logprior = -0.5 * intercept^2 -
               0.5 * sum((raw_attack ./ residual_scale) .^ 2) -
               0.5 * sum((raw_defence ./ residual_scale) .^ 2) -
               2.0 * n_teams * log(residual_scale)
    return loglik + logprior
end

"Compiled/fresh ReverseDiff and ForwardDiff comparison at base and perturbed θ."
function gradient_report(f, θ::Vector{Float64}; perturbation::Float64 = 1e-3)
    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    compiled = similar(θ)
    ReverseDiff.gradient!(compiled, tape, θ)
    fresh = ReverseDiff.gradient(f, θ)
    forward = ForwardDiff.gradient(f, θ)
    θp = θ .+ perturbation .* sin.(Float64.(eachindex(θ)))
    compiled_perturbed = similar(θp)
    ReverseDiff.gradient!(compiled_perturbed, tape, θp)
    fresh_perturbed = ReverseDiff.gradient(f, θp)
    relerr(a, b) = norm(a .- b) / max(norm(a), norm(b), 1.0)
    return (; instructions = length(raw.tape),
              compiled_fresh = relerr(compiled, fresh),
              compiled_forward = relerr(compiled, forward),
              perturbed_compiled_fresh = relerr(compiled_perturbed, fresh_perturbed))
end

"""
    warmed_allocation_report(f, θ)

Reports allocations of repeated compiled-tape gradients after warm-up. A finite
number is expected; tape compatibility is not an allocation-free whole-program
claim.
"""
function warmed_allocation_report(f, θ::Vector{Float64}; warmup::Int = 10, reps::Int = 20)
    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    gradient = similar(θ)
    for _ in 1:warmup
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    bytes = @allocated begin
        for _ in 1:reps
            ReverseDiff.gradient!(gradient, tape, θ)
        end
    end
    return (; bytes, bytes_per_gradient = bytes / reps, reps)
end

end # module PedigreeTierMock
