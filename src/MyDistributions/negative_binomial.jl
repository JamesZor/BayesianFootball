# src/MyDistributions/robust_negative_binomial.jl

using Distributions
using LogExpFunctions
using SpecialFunctions: loggamma
using Random

export RobustNegativeBinomial

"""
    RobustNegativeBinomial(r, μ)

A Negative Binomial distribution parameterized by dispersion `r` and mean `μ`.
This implementation avoids the numerical instability of converting to `p = r/(r+μ)`
during gradient steps, which can trigger DomainErrors in standard Distributions.jl.
"""
struct RobustNegativeBinomial{T<:Real} <: DiscreteUnivariateDistribution
    r::T  # Dispersion (Shape)
    μ::T  # Mean
    
    function RobustNegativeBinomial{T}(r::T, μ::T) where {T<:Real}
        # We allow μ to be very large, but r must be positive
        new{T}(max(r, 1e-6), max(μ, 1e-6)) 
    end
end

RobustNegativeBinomial(r::T, μ::T) where {T<:Real} = RobustNegativeBinomial{T}(r, μ)
RobustNegativeBinomial(r::Real, μ::Real) = RobustNegativeBinomial(promote(r, μ)...)

# --- Standard Interface ---
Base.minimum(::RobustNegativeBinomial) = 0
Base.maximum(::RobustNegativeBinomial) = Inf
Distributions.insupport(::RobustNegativeBinomial, x::Real) = isinteger(x) && x >= 0
Distributions.params(::RobustNegativeBinomial) = (d.r, d.μ)

# 3. Interface: Statistics
# These allow things like mean(d) and var(d) to work
Distributions.mean(d::RobustNegativeBinomial) = d.μ
Distributions.var(d::RobustNegativeBinomial)  = d.μ + (d.μ^2 / d.r)
Distributions.std(d::RobustNegativeBinomial)  = sqrt(Distributions.var(d))

# --- Numerically Stable LogPDF ---
# We compute logpdf directly from r and μ without explicit p
function Distributions.logpdf(d::RobustNegativeBinomial, k::Int)
    if k < 0
        return -Inf
    end
    r, μ = d.r, d.μ
    
    # Standard NegBin LogPMF:
    # log(Γ(k+r)) - log(k!) - log(Γ(r)) + r*log(p) + k*log(1-p)
    # where p = r / (r + μ)
    #
    # Optimisation:
    # log(p)   = log(r) - log(r+μ)
    # log(1-p) = log(μ) - log(r+μ)
    
    term1 = loggamma(k + r) - loggamma(k + 1) - loggamma(r)
    term2 = r * (log(r) - log(r + μ))
    term3 = k * (log(μ) - log(r + μ))
    
    return term1 + term2 + term3
end

# --- Cumulative Distribution Function (CDF) ---
function Distributions.cdf(d::RobustNegativeBinomial, x::Real)
    # If x is below 0, the cumulative probability is 0
    x < 0 && return 0.0
    
    # Convert to standard (r, p) for the CDF calculation
    p = d.r / (d.r + d.μ)
    p_safe = clamp(p, 1e-12, 1.0 - 1e-12)
    
    # Hand off the heavy lifting to the standard NegativeBinomial
    return cdf(NegativeBinomial(d.r, p_safe), x)
end

# Handle non-integer k (needed for Turing generic calls sometimes)
Distributions.logpdf(d::RobustNegativeBinomial, k::Real) = isinteger(k) ? logpdf(d, Int(k)) : -Inf
Distributions.pdf(d::RobustNegativeBinomial, k::Real) = exp(logpdf(d, k))

# --- Sampler ---
function Distributions.rand(rng::AbstractRNG, d::RobustNegativeBinomial)
    # Convert to standard (r, p) for sampling
    # We clamp p to avoid the exact error we are solving, 
    # but strictly for random generation (not gradients).
    p = d.r / (d.r + d.μ)
    p_safe = clamp(p, 1e-12, 1.0 - 1e-12)
    return rand(rng, NegativeBinomial(d.r, p_safe))
end
