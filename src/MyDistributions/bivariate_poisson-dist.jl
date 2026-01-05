# src/MyDistributions/bivariate_poisson-dist.jl

using Distributions
using Turing
using SpecialFunctions: logfactorial
using LogExpFunctions: logsumexp
using Random
using Statistics



"""
    BivariateLogPoisson(θ1, θ2, θ3)

A Bivariate Poisson distribution parameterized by log-rates θ.
The actual rates are λ = exp(θ).

- θ₁: Log-rate of independent event 1 (Home)
- θ₂: Log-rate of independent event 2 (Away)
- θ₃: Log-rate of covariance event (Common)
"""
struct BivariateLogPoisson{T<:Real} <: DiscreteMultivariateDistribution
    θ₁::T
    θ₂::T
    θ₃::T

    # Constructor with verification (though log-rates can be any real number)
    function BivariateLogPoisson(θ₁::T, θ₂::T, θ₃::T) where {T<:Real}
        new{T}(θ₁, θ₂, θ₃)
    end
end


# Support promotion (e.g., if user passes Int and Float)
BivariateLogPoisson(θ₁::Real, θ₂::Real, θ₃::Real) = BivariateLogPoisson(promote(θ₁, θ₂, θ₃)...)

# --- Required Distributions.jl Interface ---

# 1. Dimension of the support (2D distribution: Home and Away)
Base.length(d::BivariateLogPoisson) = 2

# 2. Type of the support (Integers)
Base.eltype(d::BivariateLogPoisson) = Int

# 3. Sampling (rand)
# We convert log-rates back to rates only for sampling.
# X = Z1 + Z3, Y = Z2 + Z3
function Distributions.rand(rng::AbstractRNG, d::BivariateLogPoisson)
    # Numerical safety: exp(20) is huge, so we clamp for sampling safety if needed,
    # though usually standard RNG handles large lambdas fine.
    λ₁ = exp(d.θ₁)
    λ₂ = exp(d.θ₂)
    λ₃ = exp(d.θ₃)

    z₁ = rand(rng, Poisson(λ₁))
    z₂ = rand(rng, Poisson(λ₂))
    z₃ = rand(rng, Poisson(λ₃))

    return [z₁ + z₃, z₂ + z₃]
end

# Convenience wrapper for one-off sampling
Distributions.rand(d::BivariateLogPoisson) = rand(Random.default_rng(), d)

# 3. The LogPDF Interface
function Distributions.logpdf(d::BivariateLogPoisson, x::AbstractVector{<:Real})
    if length(x) != 2
        throw(DimensionMismatch("BivariateLogPoisson requires a vector of length 2"))
    end
    
    # Turing might pass data as Floats, so we strictly cast to Int for factorials.
    # Note: If x contains Dual numbers (gradients on data), this cast will fail.
    # But for standard Poisson regression, data is fixed (Integer), parameters vary.
    x_val = round(Int, x[1])
    y_val = round(Int, x[2])

    # Check support
    if x_val < 0 || y_val < 0
        return -Inf
    end

    return _bp_logpdf_kernel(d.θ₁, d.θ₂, d.θ₃, x_val, y_val)
end

# 4. The Computational Kernel
function _bp_logpdf_kernel(θ₁::Real, θ₂::Real, θ₃::Real, x::Int, y::Int)
    k_max = min(x, y)

    # --- Part A: Normalization ---
    # log(exp(-lambda_sum)) = -lambda_sum
    # We use exp(θ) to map from log-space back to rate-space
    log_norm = -(exp(θ₁) + exp(θ₂) + exp(θ₃))

    # --- Part B: The Summation (LogSumExp) ---
    # We reformulate t_k to pull constants out of the loop.
    # Original: t_k = θ1(x-k) + θ2(y-k) + θ3*k - lfact(x-k) - lfact(y-k) - lfact(k)
    # Regrouped: t_k = (θ1*x + θ2*y) + k(θ3 - θ1 - θ2) - lfact...
    
    base_term = θ₁ * x + θ₂ * y
    diff_theta = θ₃ - θ₁ - θ₂

    # # We use 'map' to create a generator-like structure. 
    # # This is often cleaner for AD (Automatic Differentiation) than mutating an array.
    # log_sum_val = logsumexp(0:k_max) do k
    #     # This is the inner function for each k
    #
    #     # Determine drift based on k
    #     drift = k * diff_theta
    #
    #     # Combinatorial penalty
    #     # Note: logfactorial is from SpecialFunctions.jl
    #     denom = logfactorial(x - k) + logfactorial(y - k) + logfactorial(k)
    #
    #     # Return t_k
    #     base_term + drift - denom
    # end
    log_sum_val = logsumexp(
            (base_term + k * diff_theta - (logfactorial(x - k) + logfactorial(y - k) + logfactorial(k)))
            for k in 0:k_max
        )

    return log_norm + log_sum_val
end





