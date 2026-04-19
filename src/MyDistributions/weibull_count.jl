# src/MyDistributions/weibull_count.jl

using Distributions
using SpecialFunctions: lgamma
using Optim

export WeibullCount

# -------------------------------------------------------------------
# 1. Euler-van Wijngaarden Series Acceleration (Translated from C++)
# -------------------------------------------------------------------
mutable struct Eulsum
    wksp::Vector{Float64}
    n::Int
    ncv::Int
    cnvgd::Bool
    sum::Float64
    eps::Float64
    lastval::Float64
    lasteps::Float64

    function Eulsum(nmax::Int, eps::Float64)
        new(zeros(Float64, nmax), 0, 0, false, 0.0, eps, 0.0, 0.0)
    end
end

function next_term!(e::Eulsum, term::Float64)
    if e.n + 1 > length(e.wksp)
        error("Workspace too small in Eulsum")
    end

    if e.n == 0
        e.n = 1
        e.wksp[1] = term
        e.sum = 0.5 * term
    else
        tmp = e.wksp[1]
        e.wksp[1] = term
        for j in 2:e.n
            dum = e.wksp[j]
            e.wksp[j] = 0.5 * (e.wksp[j - 1] + tmp)
            tmp = dum
        end
        e.wksp[e.n + 1] = 0.5 * (e.wksp[e.n] + tmp)
        
        if abs(e.wksp[e.n + 1]) <= abs(e.wksp[e.n])
            e.sum += 0.5 * e.wksp[e.n + 1]
            e.n += 1
        else
            e.sum += e.wksp[e.n + 1]
        end
    end

    e.lasteps = abs(e.sum - e.lastval)
    if e.lasteps <= e.eps
        e.ncv += 1
    end
    if e.ncv >= 2
        e.cnvgd = true
    end

    e.lastval = e.sum
    return e.sum
end

# -------------------------------------------------------------------
# 2. Alpha Matrix Generator (McShane Eq. 11)
# -------------------------------------------------------------------
function alphagen(c::Float64, jrow::Int, ncol::Int)
    alpha = zeros(Float64, jrow, ncol)
    lgam = 0.0
    
    # First column (n = 0)
    for j in 0:(jrow-1)
        alpha[j+1, 1] = exp(lgamma(c * j + 1.0) - lgam)
        lgam += log(j + 1.0)
    end
    
    # Subsequent columns
    for n in 0:(ncol-2)
        for j in (n+1):(jrow-1)
            sum_val = 0.0
            for m in n:(j-1)
                sum_val += alpha[m+1, n+1] * alpha[j-m+1, 1]
            end
            alpha[j+1, n+2] = sum_val
        end
    end
    
    return alpha
end

# -------------------------------------------------------------------
# 3. The Distributions.jl Interface
# -------------------------------------------------------------------
struct WeibullCount{T<:Real} <: DiscreteUnivariateDistribution
    c::T  # Shape parameter
    λ::T  # Scale/Rate parameter
    
    function WeibullCount{T}(c::T, λ::T) where {T<:Real}
        new{T}(max(c, 1e-6), max(λ, 1e-6))
    end
end

WeibullCount(c::T, λ::T) where {T<:Real} = WeibullCount{T}(c, λ)
WeibullCount(c::Real, λ::Real) = WeibullCount(promote(c, λ)...)

Distributions.params(d::WeibullCount) = (d.c, d.λ)
Distributions.minimum(::WeibullCount) = 0
Distributions.maximum(::WeibullCount) = Inf

# Core Probability Function
function Distributions.pdf(d::WeibullCount, x::Int; time::Float64 = 1.0, jmax::Int = 50, nmax::Int = 300, eps::Float64 = 1e-10)
    x < 0 && return 0.0
    
    # Generate Alpha Matrix for this count
    alpha_all = alphagen(d.c, jmax + x + 1, x + 1)
    
    ltc = d.λ * (time^d.c)
    coeff = 1.0
    eulsum = Eulsum(nmax, eps)
    val = 0.0
    
    for j in x:(x + jmax - 1)
        term = coeff * (ltc^j) * alpha_all[j+1, x+1] * exp(-lgamma(d.c * j + 1.0))
        val = next_term!(eulsum, term)
        coeff = -coeff
        
        if eulsum.cnvgd
            break
        end
    end
    
    # Prevent negative probabilities from floating point errors
    return max(val, 1e-15) 
end

Distributions.logpdf(d::WeibullCount, x::Int) = log(pdf(d, x))
Distributions.pdf(d::WeibullCount, x::Real) = isinteger(x) ? pdf(d, Int(x)) : 0.0
Distributions.logpdf(d::WeibullCount, x::Real) = isinteger(x) ? logpdf(d, Int(x)) : -Inf

# Approximate CDF by summing PDFs
function Distributions.cdf(d::WeibullCount, x::Int)
    x < 0 && return 0.0
    sum(pdf(d, i) for i in 0:x)
end
Distributions.cdf(d::WeibullCount, x::Real) = isinteger(x) ? cdf(d, Int(x)) : cdf(d, Int(floor(x)))
