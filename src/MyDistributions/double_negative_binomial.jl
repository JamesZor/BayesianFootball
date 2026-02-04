# src/MyDistributions/double_negative_binomial.jl
using Distributions, LogExpFunctions
using SpecialFunctions: loggamma

# Define the struct
struct DoubleNegativeBinomial{T<:Real} <: DiscreteMultivariateDistribution
    μ_h::T # Home Mean
    μ_a::T # Away Mean
    r_h::T # Home Shape (Inverse Dispersion)
    r_a::T # Away Shape
end

# Constructor for broadcasting or easy creation
DoubleNegativeBinomial(μ_h, μ_a, r_h, r_a) = DoubleNegativeBinomial(promote(μ_h, μ_a, r_h, r_a)...)

# Define array limits
Base.length(d::DoubleNegativeBinomial) = 2
Base.eltype(d::DoubleNegativeBinomial) = Int

# --- The Optimized Log PDF ---
# FIX: We calculate the log-prob manually to avoid creating 
#      Intermediate 'NegativeBinomial' structs which confuse ReverseDiff.
function Distributions.logpdf(d::DoubleNegativeBinomial, x::AbstractVector{<:Integer})
    if length(x) != 2
        throw(DimensionMismatch("DoubleNegativeBinomial requires a vector of length 2 (home, away)"))
    end
    
    return _nbinom_logpdf_direct(d.r_h, d.μ_h, x[1]) + 
           _nbinom_logpdf_direct(d.r_a, d.μ_a, x[2])
end

# Helper: Direct LogPDF calculation for Negative Binomial
# Parameterized by Mean (μ) and Shape (r)
# Corresponds to Distributions.NegativeBinomial(r, p) where p = r / (r + μ)
function _nbinom_logpdf_direct(r::Real, μ::Real, k::Real)
    # p is probability of success
    p = r / (r + μ)
    
    # Formula: log( Γ(k+r) / (Γ(k+1)Γ(r)) ) + r*log(p) + k*log(1-p)
    # We use loggamma for numerical stability with gradients
    return loggamma(k + r) - loggamma(k + 1) - loggamma(r) + 
           r * log(p) + k * log(1 - p)
end

# Sampler (Optional, useful for posterior predictive checks)
# It is safe to create structs here because 'rand' is not differentiated.
function Distributions.rand(rng::AbstractRNG, d::DoubleNegativeBinomial)
    p_h = d.r_h / (d.r_h + d.μ_h)
    p_a = d.r_a / (d.r_a + d.μ_a)
    
    return [
        rand(rng, NegativeBinomial(d.r_h, p_h)),
        rand(rng, NegativeBinomial(d.r_a, p_a))
    ]
end
