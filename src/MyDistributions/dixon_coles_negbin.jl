# src/MyDistributions/dixon_coles_negbin.jl

using Distributions
using LogExpFunctions

export DixonColesNegBinLogGroup

# --- Helper: Safe Log ---
# This acts as the "soft guard" for the dynamic constraints of rho. 
# If the sampler proposes a rho that creates a negative probability (τ <= 0), 
# we return -Inf to gently reject the proposal without crashing the AD tracer.
function safe_log_tau(val::T) where {T<:Real}
    return val > zero(T) ? log(val) : -T(Inf)
end

"""
    DixonColesNegBinLogGroup(λ, μ, r_h, r_a, ρ, group)

A custom distribution representing a BATCH of matches for the Negative Binomial 
Dixon-Coles model. Applies the correlation parameter ρ to specific low-scoring 
groups (:s00, :s10, :s01, :s11).

Arguments:
- `λ`: Vector of Expected Goals (Home)
- `μ`: Vector of Expected Goals (Away)
- `r_h`: Vector of Dispersion (Home)
- `r_a`: Vector of Dispersion (Away)
- `ρ`: Scalar dependence parameter
- `group`: Symbol :s00, :s10, :s01, :s11
"""
# We inherit from ContinuousUnivariateDistribution so we can easily use `0 ~` in Turing
struct DixonColesNegBinLogGroup{T<:Real, V<:AbstractVector{T}} <: ContinuousUnivariateDistribution
    λ::V          
    μ::V          
    r_h::V        
    r_a::V        
    ρ::T          
    group::Symbol 
end

# Minimal interface requirements
Base.minimum(::DixonColesNegBinLogGroup) = -Inf
Base.maximum(::DixonColesNegBinLogGroup) = Inf

# The x argument is ignored (we will pass `0 ~`) because the scores are implied by d.group
function Distributions.logpdf(d::DixonColesNegBinLogGroup, x::Real)
    λ = d.λ
    μ = d.μ
    r_h = d.r_h
    r_a = d.r_a
    ρ = d.ρ

    # 1. Calculate Base Independent Negative Binomial Log-Likelihoods + Tau Term
    if d.group == :s00
        # Score 0-0: log( P(0|λ,r_h) * P(0|μ,r_a) )
        log_indep = logpdf.(RobustNegativeBinomial.(r_h, λ), 0) .+ 
                    logpdf.(RobustNegativeBinomial.(r_a, μ), 0)
        τ_term = 1 .- (λ .* μ .* ρ)

    elseif d.group == :s10
        # Score 1-0: log( P(1|λ,r_h) * P(0|μ,r_a) )
        log_indep = logpdf.(RobustNegativeBinomial.(r_h, λ), 1) .+ 
                    logpdf.(RobustNegativeBinomial.(r_a, μ), 0)
        τ_term = 1 .+ (μ .* ρ)

    elseif d.group == :s01
        # Score 0-1: log( P(0|λ,r_h) * P(1|μ,r_a) )
        log_indep = logpdf.(RobustNegativeBinomial.(r_h, λ), 0) .+ 
                    logpdf.(RobustNegativeBinomial.(r_a, μ), 1)
        τ_term = 1 .+ (λ .* ρ)

    elseif d.group == :s11
        # Score 1-1: log( P(1|λ,r_h) * P(1|μ,r_a) )
        log_indep = logpdf.(RobustNegativeBinomial.(r_h, λ), 1) .+ 
                    logpdf.(RobustNegativeBinomial.(r_a, μ), 1)
        τ_term = 1 .- ρ
        
    else
        error("Invalid group symbol: $(d.group). Expected :s00, :s10, :s01, or :s11")
    end

    # 2. Apply Correction safely 
    log_τ = safe_log_tau.(τ_term)

    # 3. Sum all individual match log-likelihoods into a single scalar for the gradient
    return sum(log_indep .+ log_τ)
end
