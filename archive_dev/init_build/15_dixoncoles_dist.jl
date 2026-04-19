using Distributions
using Random
using LinearAlgebra

# Import specific methods we need to extend
import Distributions: length, eltype, _rand!, logpdf, mean, var, cov, insupport

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



#### dev 
using Distributions, LinearAlgebra, Test
using Random

# Include your distribution definition if it's in a file
# include("src/models/pregame/distributions.jl")

function test_dixon_coles_math()
    println("--- Testing DixonColes Distribution Math ---")
    
    λ, μ, ρ = 1.2, 1.0, 0.2
    d = DixonColes(λ, μ, ρ)
    
    # 1. Check if it is a valid PMF (Sums to ~1.0)
    # We sum over a large enough grid (0 to 15) to capture >99.9% of mass
    total_prob = 0.0
    for h in 0:15, a in 0:15
        total_prob += exp(logpdf(d, [h, a]))
    end
    println("Total Probability Sum (should be ≈ 1.0): ", total_prob)
    @test isapprox(total_prob, 1.0, atol=1e-4)

    # 2. Check Correction Logic (0-0 case)
    # With ρ > 0, 0-0 should be LESS likely than independent Poissons
    # Correction for 0-0 is (1 - λ*μ*ρ)
    prob_dc = exp(logpdf(d, [0, 0]))
    prob_indep = pdf(Poisson(λ), 0) * pdf(Poisson(μ), 0)
    
    expected_correction = 1.0 - (λ * μ * ρ)
    println("P(0,0) Dixon-Coles: $prob_dc")
    println("P(0,0) Independent: $prob_indep")
    println("Theoretical Correction: $expected_correction")
    
    # The DC prob should exactly match Indep * Correction
    @test isapprox(prob_dc, prob_indep * expected_correction, atol=1e-8)
    println("✔ Correction logic verified")
end

test_dixon_coles_math()


### test 2 
using Turing, DataFrames, MCMCChains


@model function simple_dc_test(data_matrix)
    # 1. Priors
    λ ~ Gamma(2, 0.7) 
    μ ~ Gamma(2, 0.5) 
    ρ ~ Uniform(-0.3, 0.3)
    
    # 2. Likelihood Preparation
    # We need to know how many matches we have to create the right number of distributions
    # If data_matrix is a Vector of Vectors (e.g. [[1,0], [2,1]]), use length()
    # If data_matrix is a Matrix (2 x N), use size(data_matrix, 2)
    N = size(data_matrix, 2)
    # 3. Create a VECTOR of distributions
    # We create N identical distributions because λ, μ, ρ are constant in this simple test.
    dists = fill(DixonColes(λ, μ, ρ), N)
    
    # 4. Observe
    # Now arraydist gets a Vector{DixonColes}, which is what it wants.
    data_matrix ~ arraydist(dists)
end

# Run the parameter recovery test
function run_parameter_recovery()
    println("\n--- Running Turing Parameter Recovery ---")
    
    # A. Generate Synthetic Data
    true_λ, true_μ, true_ρ = 1.45, 1.05, 0.15
    d_true = DixonColes(true_λ, true_μ, true_ρ)
    
    n_samples = 2000
    data_tuples = [rand(d_true) for _ in 1:n_samples]
    data_matrix = reduce(hcat, data_tuples)
    h_goals = [x[1] for x in data_tuples]
    a_goals = [x[2] for x in data_tuples]
    data_input = stack([home_goals, away_goals], dims=1)
    
    println("Generated $n_samples matches with True ρ = $true_ρ")

    # B. Run Turing
    model = simple_dc_test(data_matrix)
    chain = sample(model, NUTS(0.65), 1000)
    
    # C. Check Results
    display(chain)
end

run_parameter_recovery()



@model function simple_poisson(data_matrix)
    # 1. Priors
    λ ~ Gamma(2, 0.7) 
    μ ~ Gamma(2, 0.5) 

    # 2. Likelihood Preparation
    N = size(data_matrix, 2)
    
    # 3. Create Vector of Distributions
    # Product([P1, P2]) creates a multivariate distribution where P1 and P2 are independent
    # distinct_d = Product([Poisson(λ), Poisson(μ)])
    dists = fill(Product([Poisson(λ), Poisson(μ)]), N)
    
    # 4. Observe
    data_matrix ~ arraydist(dists)
end


# 2. Fit the Independent Model
indep_model = simple_poisson(data_matrix)
chain_indep = sample(indep_model, NUTS(0.65), 1000)

# 3. Compare Results
println("\n--- Comparison ---")
println("True Parameters: λ=$true_λ, μ=$true_μ, ρ=$true_ρ")

println("\n[Independent Poisson Model Results]")
display(chain_indep)

# Check biases
est_λ = mean(chain_indep[:λ])
est_μ = mean(chain_indep[:μ])

println("\nBias Analysis:")
println("λ Bias: ", round(est_λ - true_λ, digits=4))
println("μ Bias: ", round(est_μ - true_μ, digits=4))
println("(Note: The Independent model forces ρ=0, ignoring the correlation)")


using StatsPlots 
density(chain[:λ], label="λ DC")
density!(chain_indep[:λ], label="λ DP")
density!(chain[:μ], label="μ DC")
density!(chain_indep[:μ], label="μ DP")

density(chain[:ρ], label="DC")
density!(chain_indep[:λ], label="DP")


### 
