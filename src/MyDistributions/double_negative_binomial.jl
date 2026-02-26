# src/MyDistributions/double_negative_binomial.jl
using Distributions, LogExpFunctions
using SpecialFunctions: loggamma

# 1. STRUCT (Add inner constructor for safety)
struct DoubleNegativeBinomial{T<:Real} <: DiscreteMultivariateDistribution
    μ_h::T # Home Mean
    μ_a::T # Away Mean
    r_h::T # Home Shape
    r_a::T # Away Shape

    # Inner constructor to prevent negative parameters from ever existing
    function DoubleNegativeBinomial{T}(μ_h, μ_a, r_h, r_a) where T
        # Clamp to a tiny positive number to prevent NaN
        # This is safer than Exp() transforms alone during warmup
        new{T}(max(μ_h, 1e-9), max(μ_a, 1e-9), max(r_h, 1e-9), max(r_a, 1e-9))
    end
end

# Outer constructors
DoubleNegativeBinomial(μ_h::T, μ_a::T, r_h::T, r_a::T) where {T<:Real} = 
    DoubleNegativeBinomial{T}(μ_h, μ_a, r_h, r_a)
DoubleNegativeBinomial(μ_h, μ_a, r_h, r_a) = 
    DoubleNegativeBinomial(promote(μ_h, μ_a, r_h, r_a)...)

Base.length(d::DoubleNegativeBinomial) = 2
Base.eltype(d::DoubleNegativeBinomial) = Int

# 2. OPTIMIZED LOG PDF (The Robust Math)
function Distributions.logpdf(d::DoubleNegativeBinomial, x::AbstractVector{<:Real})
    if length(x) != 2
        # Return -Inf instead of erroring creates smoother failures in some pipelines
        return -Inf 
    end
    
    return _nbinom_logpdf_robust(d.r_h, d.μ_h, x[1]) + 
           _nbinom_logpdf_robust(d.r_a, d.μ_a, x[2])
end

# 3. ROBUST HELPER (Log-Space Arithmetic)
# This replaces your old direct calculation
function _nbinom_logpdf_robust(r::Real, μ::Real, k::Real)
    # Standard Negative Binomial Log-PMF formula in log-space:
    # log(Γ(k+r)) - log(k!) - log(Γ(r)) + r*log(p) + k*log(1-p)
    # where p = r / (r + μ)
    
    # We use identities:
    # log(p)   = log(r) - log(r+μ)
    # log(1-p) = log(μ) - log(r+μ)

    term1 = loggamma(k + r) - loggamma(k + 1) - loggamma(r)
    term2 = r * (log(r) - log(r + μ))
    term3 = k * (log(μ) - log(r + μ))

    return term1 + term2 + term3
end

# 4. SAMPLER (Keep this as is, random generation doesn't need gradients)
function Distributions.rand(rng::AbstractRNG, d::DoubleNegativeBinomial)
    # We clamp strictly for the random generator to avoid errors
    p_h = clamp(d.r_h / (d.r_h + d.μ_h), 1e-10, 1.0-1e-10)
    p_a = clamp(d.r_a / (d.r_a + d.μ_a), 1e-10, 1.0-1e-10)
    
    return [
        rand(rng, NegativeBinomial(d.r_h, p_h)),
        rand(rng, NegativeBinomial(d.r_a, p_a))
    ]
end
