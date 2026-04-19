# src/MyDistribution/dixoncoles-dist.jl

# Dixon coles distribution 

# --- 1. Type Definition ---
"""
    DixonColes(λ, μ, ρ)

A discrete multivariate distribution for football scores with a low-score correlation correction.
- `λ`: Expected home goals (Poisson rate)
- `μ`: Expected away goals (Poisson rate)
- `ρ`: Correlation parameter (usually small, e.g., between -0.3 and 0.3)

Samples return a vector of length 2: `[home_score, away_score]`.
"""
struct DixonColes{T<:Real} <: DiscreteMultivariateDistribution
    λ::T
    μ::T
    ρ::T

    # Inner constructor restricts T to Real (needed for AD gradients in Turing)
    function DixonColes{T}(λ::T, μ::T, ρ::T) where {T<:Real}
        # We allow ρ to be outside theoretical bounds during warm-up, 
        # but the logpdf will handle invalid correlations gracefully.
        new{T}(λ, μ, ρ)
    end
end

# --- 2. Outer Constructors (Handling Type Promotion) ---
DixonColes(λ::T, μ::T, ρ::T) where {T<:Real} = DixonColes{T}(λ, μ, ρ)
DixonColes(λ::Real, μ::Real, ρ::Real) = DixonColes(promote(λ, μ, ρ)...)

# --- 3. Base Methods (Metadata) ---
Base.length(d::DixonColes) = 2 
Base.eltype(d::DixonColes) = Int

# --- 4. Support Checking (Good Practice) ---
# Defines the valid domain: vectors of length 2 containing non-negative integers.
function Distributions.insupport(d::DixonColes, x::AbstractVector{<:Real})
    return length(x) == 2 && all(isinteger, x) && all(>=(0), x)
end

# --- 5. Statistics ---
Distributions.mean(d::DixonColes) = [d.λ, d.μ]
Distributions.var(d::DixonColes) = [d.λ, d.μ] 

# --- 6. Probability Density Function (CRITICAL FOR TURING) ---
function Distributions.logpdf(d::DixonColes, x::AbstractVector{<:Real})
    if length(x) != 2
        throw(DimensionMismatch("DixonColes expects a vector of length 2 (Home, Away)"))
    end
    
    # We cast to Int for logic, but keep gradients attached to parameters (λ, μ, ρ)
    h = floor(Int, x[1])
    a = floor(Int, x[2])
    
    return _dixon_coles_logpdf(d.λ, d.μ, d.ρ, h, a)
end

# Implementation separated for clarity
function _dixon_coles_logpdf(λ::T, μ::T, ρ::T, h::Int, a::Int) where T
    # 1. Independent Poisson Log-Likelihoods
    log_p_home = logpdf(Poisson(λ), h)
    log_p_away = logpdf(Poisson(μ), a)
    
    # 2. Dixon-Coles Correction Factor (τ)
    # The correction only applies to scores 0-0, 0-1, 1-0, 1-1
    correction = one(T)
    
    if h == 0 && a == 0
        correction = 1.0 - (λ * μ * ρ)
    elseif h == 0 && a == 1
        correction = 1.0 + (λ * ρ)
    elseif h == 1 && a == 0
        correction = 1.0 + (μ * ρ)
    elseif h == 1 && a == 1
        correction = 1.0 - ρ
    end

    # 3. Handle bounds for numerical stability
    if correction <= zero(T)
        return T(-Inf)
    end

    return log_p_home + log_p_away + log(correction)
end

# --- 7. Sampling (For Predictions) ---
# Used when you call rand(DixonColes(...)) or perform posterior predictive checks
function Distributions._rand!(rng::AbstractRNG, d::DixonColes, x::AbstractVector{<:Real})
    # Rejection sampling strategy
    max_attempts = 1000
    M = 1.5 # Upper bound for correction factor
    
    for _ in 1:max_attempts
        # Propose from independent Poissons
        h = rand(rng, Poisson(d.λ))
        a = rand(rng, Poisson(d.μ))
        
        # Calculate acceptance probability
        correction = 1.0
        if h == 0 && a == 0; correction = 1.0 - (d.λ * d.μ * d.ρ)
        elseif h == 0 && a == 1; correction = 1.0 + (d.λ * d.ρ)
        elseif h == 1 && a == 0; correction = 1.0 + (d.μ * d.ρ)
        elseif h == 1 && a == 1; correction = 1.0 - d.ρ
        end
        
        if correction < 0; correction = 0.0; end
        
        # Accept/Reject
        if rand(rng) < (correction / M)
            x[1] = h
            x[2] = a
            return x
        end
    end
    
    # Fallback
    x[1] = rand(rng, Poisson(d.λ))
    x[2] = rand(rng, Poisson(d.μ))
    return x
end



