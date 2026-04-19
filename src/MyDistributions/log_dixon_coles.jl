using Distributions, Turing, LogExpFunctions, Statistics

# --- Helper: Safe Log ---
# This protects the sampler. If 1+x <= 0, we return -Inf (probability 0).
# This acts as a "soft wall" for the dynamic constraints.
function safe_log_tau(val)
    if val <= 0
        return -Inf
    end
    return log(val)
end

"""
    DixonColesLogGroup(θ_home, θ_away, ρ, group_type)

A distribution representing a BATCH of matches that all belong to a specific
score group (:00, :10, :01, :11).
"""
struct DixonColesLogGroup{T<:Real, V<:AbstractVector{T}} <: DiscreteMultivariateDistribution
    θ₁::V        # Vector of log-rates (Home)
    θ₂::V        # Vector of log-rates (Away)
    ρ::T         # Scalar dependence parameter
    group::Symbol # :00, :10, :01, :11
end

# We calculate the TOTAL logpdf for the whole group at once (Vectorized)
function Distributions.logpdf(d::DixonColesLogGroup, x::AbstractMatrix{<:Real})
    # Note: 'x' is ignored here because the score is implied by d.group!
    # This saves us passing huge arrays of zeros and ones.
    
    λ = exp.(d.θ₁)
    μ = exp.(d.θ₂)
    ρ = d.ρ

    # 1. Calculate Base Independent Log-Likelihoods
    # Formulations derived from Poisson pmf: log(λ^k * e^-λ / k!)
    if d.group == :00
        # k1=0, k2=0 -> -λ - μ
        log_indep = -λ .- μ
        # τ = 1 - λμρ
        τ_term = 1 .- (λ .* μ .* ρ)

    elseif d.group == :10
        # k1=1, k2=0 -> log(λ) - λ - μ
        log_indep = d.θ₁ .- λ .- μ
        # τ = 1 + μρ
        τ_term = 1 .+ (μ .* ρ)

    elseif d.group == :01
        # k1=0, k2=1 -> log(μ) - λ - μ
        log_indep = d.θ₂ .- λ .- μ
        # τ = 1 + λρ
        τ_term = 1 .+ (λ .* ρ)

    elseif d.group == :11
        # k1=1, k2=1 -> log(λ) - λ + log(μ) - μ
        log_indep = d.θ₁ .- λ .+ d.θ₂ .- μ
        # τ = 1 - ρ
        τ_term = 1 .- ρ
    else
        error("Invalid group symbol")
    end

    # 2. Apply Correction safely
    # We broadcast the safety check over the vector
    log_τ = safe_log_tau.(τ_term)

    # 3. Sum it all up to get the scalar log-likelihood for the group
    return sum(log_indep .+ log_τ)
end
