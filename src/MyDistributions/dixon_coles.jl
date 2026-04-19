# src/MyDistributions/dixon_coles.jl

using Distributions
using LogExpFunctions
using Statistics

export DixonColesLogGroup

# --- Helper: Safe Log ---
# This acts as the "soft guard" for the dynamic constraints. 
# If the sampler proposes a rho/rate combo where 1+x <= 0 (impossible), 
# we return -Inf to reject the proposal gracefully.
function safe_log_tau(val::T) where T
    if val <= zero(T)
        return typemin(T) # -Inf
    end
    return log(val)
end

"""
    DixonColesLogGroup(θ_home, θ_away, ρ, group_type)

A custom distribution representing a BATCH of matches that all belong to a specific
score group (:00, :10, :01, :11). 

This allows us to calculate the likelihood for thousands of matches in a single 
vectorized operation without branching logic inside the gradient path.

Arguments:
- `θ₁`: Vector of log-rates (Home)
- `θ₂`: Vector of log-rates (Away)
- `ρ`:  Scalar dependence parameter (tanh-scaled)
- `group`: Symbol :00, :10, :01, :11
"""
struct DixonColesLogGroup{T<:Real, V<:AbstractVector{T}} <: DiscreteMultivariateDistribution
    θ₁::V        # Vector of log-rates (Home)
    θ₂::V        # Vector of log-rates (Away)
    ρ::T         # Scalar dependence parameter
    group::Symbol # :00, :10, :01, :11
end

# Required Distribution interfaces (minimal)
Base.length(d::DixonColesLogGroup) = 2
Base.eltype(d::DixonColesLogGroup) = Int


function Distributions.logpdf(d::DixonColesLogGroup, x::AbstractMatrix{<:Real})
    # Note: 'x' is ignored because the scores are implied by d.group.
    
    λ = exp.(d.θ₁)
    μ = exp.(d.θ₂)
    ρ = d.ρ

    # 1. Calculate Base Independent Log-Likelihoods + Tau Term
    # We branch on the SYMBOL (must use valid identifiers like :s00)
    
    if d.group == :s00
        # Score 0-0
        # log_indep = -λ - μ
        log_indep = -λ .- μ
        # τ = 1 - λμρ
        τ_term = 1 .- (λ .* μ .* ρ)

    elseif d.group == :s10
        # Score 1-0
        # log_indep = log(λ) - λ - μ
        log_indep = d.θ₁ .- λ .- μ
        # τ = 1 + μρ
        τ_term = 1 .+ (μ .* ρ)

    elseif d.group == :s01
        # Score 0-1
        # log_indep = log(μ) - λ - μ
        log_indep = d.θ₂ .- λ .- μ
        # τ = 1 + λρ
        τ_term = 1 .+ (λ .* ρ)

    elseif d.group == :s11
        # Score 1-1
        # log_indep = log(λ) - λ + log(μ) - μ
        log_indep = d.θ₁ .- λ .+ d.θ₂ .- μ
        # τ = 1 - ρ
        τ_term = 1 .- ρ
    else
        error("Invalid group symbol in DixonColesLogGroup: $(d.group). Expected :s00, :s10, :s01, or :s11")
    end

    # 2. Apply Correction safely
    log_τ = safe_log_tau.(τ_term)

    # 3. Sum it all up to get the scalar log-likelihood for the group
    return sum(log_indep .+ log_τ)
end


