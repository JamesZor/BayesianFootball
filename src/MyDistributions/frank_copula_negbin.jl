# src/MyDistributions/frank_copula_negbin.jl

struct FrankCopulaNegBin{T<:Real} <: DiscreteMultivariateDistribution
    r_h::T
    λ_h::T
    r_a::T
    λ_a::T
    κ::T
end

FrankCopulaNegBin(r_h::Real, λ_h::Real, r_a::Real, λ_a::Real, κ::Real) = FrankCopulaNegBin(promote(r_h, λ_h, r_a, λ_a, κ)...)

Base.length(::FrankCopulaNegBin) = 2

function frank_copula(u, v, κ)
    # Frank Copula: C(u, v) = -1/κ * log(1 + (exp(-κ*u) - 1)*(exp(-κ*v) - 1)/(exp(-κ) - 1))
    
    # AD-safe limit as κ -> 0
    if abs(κ) < 1e-5
        return u * v
    end
    
    # We use LogExpFunctions for stability
    num1 = expm1(-κ * u)
    num2 = expm1(-κ * v)
    den = expm1(-κ)
    
    # max(..., 1e-12) to ensure log argument is strictly > 0 for AD safety.
    inner = 1.0 + (num1 * num2) / den
    return -1.0 / κ * log(max(inner, 1e-12))
end

function Distributions.logpdf(d::FrankCopulaNegBin, y1::Int, y2::Int)
    if y1 < 0 || y2 < 0
        return -Inf
    end
    
    # We use RobustNegativeBinomial for stable CDF evaluations
    dist_h = RobustNegativeBinomial(d.r_h, d.λ_h)
    dist_a = RobustNegativeBinomial(d.r_a, d.λ_a)
    
    # AD-safe CDF computation by summing PMFs. 
    # Football goals are typically small (0-10), making this O(N) summation very fast 
    # and avoiding Rmath.jl which breaks ReverseDiff for CDFs.
    u0 = 0.0
    for k in 0:(y1-1)
        u0 += exp(logpdf(dist_h, k))
    end
    u1 = u0 + exp(logpdf(dist_h, y1))
    
    v0 = 0.0
    for k in 0:(y2-1)
        v0 += exp(logpdf(dist_a, k))
    end
    v1 = v0 + exp(logpdf(dist_a, y2))
    
    # Ensure probabilities are clipped to [0,1]
    u1 = clamp(u1, 0.0, 1.0)
    u0 = clamp(u0, 0.0, 1.0)
    v1 = clamp(v1, 0.0, 1.0)
    v0 = clamp(v0, 0.0, 1.0)
    
    κ = d.κ
    
    C11 = frank_copula(u1, v1, κ)
    C01 = frank_copula(u0, v1, κ)
    C10 = frank_copula(u1, v0, κ)
    C00 = frank_copula(u0, v0, κ)
    
    pmf = C11 - C01 - C10 + C00
    
    # Ensure PMF is strictly positive to avoid domain errors in log
    pmf_safe = max(pmf, 1e-12)
    return log(pmf_safe)
end

function Distributions.logpdf(d::FrankCopulaNegBin, y1::Real, y2::Real)
    return logpdf(d, Int(y1), Int(y2))
end

function Distributions._logpdf(d::FrankCopulaNegBin, x::AbstractVector{<:Real})
    return logpdf(d, x[1], x[2])
end

function Distributions.rand(rng::AbstractRNG, d::FrankCopulaNegBin)
    error("rand not yet implemented for FrankCopulaNegBin")
end
