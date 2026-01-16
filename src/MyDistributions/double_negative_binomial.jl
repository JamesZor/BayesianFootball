# src/MyDistributions/double_negative_binomial.jl
using Distributions, LogExpFunctions

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

# --- The Log PDF ---
# We compute logpdf(NB_h) + logpdf(NB_a)
function Distributions.logpdf(d::DoubleNegativeBinomial, x::AbstractVector{<:Integer})
    if length(x) != 2
        throw(DimensionMismatch("DoubleNegativeBinomial requires a vector of length 2 (home, away)"))
    end
    
    # 1. Convert Mean/Shape -> Standard (r, p)
    # p = μ / (μ + r)
    p_h = d.r_h / (d.r_h + d.μ_h)
    p_a = d.r_a / (d.r_a + d.μ_a)

    # 2. Construct Standard Distributions
    # Note: Distributions.jl NegativeBinomial(r, p)
    # We use a small epsilon for stability if needed, but r > 0 is handled by the model.
    dist_h = NegativeBinomial(d.r_h, p_h)
    dist_a = NegativeBinomial(d.r_a, p_a)

    return logpdf(dist_h, x[1]) + logpdf(dist_a, x[2])
end

# Sampler (Optional, useful for posterior predictive checks)
function Distributions.rand(rng::AbstractRNG, d::DoubleNegativeBinomial)
    p_h = d.r_h / (d.r_h + d.μ_h)
    p_a = d.r_a / (d.r_a + d.μ_a)
    
    return [
        rand(rng, NegativeBinomial(d.r_h, p_h)),
        rand(rng, NegativeBinomial(d.r_a, p_a))
    ]
end
