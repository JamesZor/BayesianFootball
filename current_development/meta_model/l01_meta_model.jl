# current_development/meta_model/l01_meta_model.jl

using Turing
using Distributions
using LogExpFunctions: logit, logistic
using DataFrames

# Struct to hold the input data cleanly
struct MetaModelData
    Y::Vector{Int}              # Binary outcomes (1 = win, 0 = loss)
    P_L1::Vector{Float64}       # Baseline Layer 1 probabilities (e.g. clm_prob)
    W::Vector{Int}              # Week indices (1 to n_weeks)
    n_weeks::Int                # Total number of weeks
end

@model function build_meta_model(data::MetaModelData)
    # Global Bias and Sensitivity to L1
    α ~ Normal(0, 1)
    β ~ Normal(1, 1) # Prior centered on 1, meaning roughly 1-to-1 mapping with logit(P_L1) initially
    
    # Non-centered parameterization for Weekly GRW
    # This prevents Neal's funnel issues during NUTS sampling
    σ_GRW ~ Gamma(2, 0.1)
    z_w ~ filldist(Normal(0, 1), data.n_weeks)
    
    # Construct the raw random walk
    θ_raw = cumsum(z_w .* σ_GRW)
    
    # Center the walk so that alpha absorbs the global mean
    θ = θ_raw .- mean(θ_raw)
    
    # Likelihood
    for i in 1:length(data.Y)
        w = data.W[i]
        
        # logit(π) = α + β * logit(P_L1) + θ_w
        # Clamp P_L1 to avoid logit(0) or logit(1) -> Inf
        p_clamped = clamp(data.P_L1[i], 1e-5, 1.0 - 1e-5)
        logit_p = logit(p_clamped)
        
        logit_pi = α + β * logit_p + θ[w]
        
        # Convert back to probability
        pi = logistic(logit_pi)
        
        # Clamp pi to avoid -Inf logpdf issues in Turing
        pi_safe = clamp(pi, 1e-10, 1.0 - 1e-10)
        
        data.Y[i] ~ Bernoulli(pi_safe)
    end
end

"""
    shift_posterior(l1_posterior::AbstractVector{Float64}, alpha_chain, beta_chain, theta_chain)

Takes an L1 posterior distribution (e.g. from ExactBayesianKelly) and a set of Meta Model
MCMC samples for alpha, beta, and the current week's theta, and applies the shift equation
to return a new, fully convolved posterior distribution.
"""
function shift_posterior(l1_posterior::AbstractVector{Float64}, alpha_chain::AbstractVector, beta_chain::AbstractVector, theta_chain::AbstractVector)
    # Ensure same number of samples
    n_samples = min(length(l1_posterior), length(alpha_chain))
    
    shifted_dist = zeros(Float64, n_samples)
    for i in 1:n_samples
        p_l1 = clamp(l1_posterior[i], 1e-5, 1.0 - 1e-5)
        logit_p = logit(p_l1)
        
        # Apply the meta model shift equation: α + β * logit(P_L1) + θ_t
        logit_pi = alpha_chain[i] + beta_chain[i] * logit_p + theta_chain[i]
        
        # Convert back to probability
        shifted_dist[i] = logistic(logit_pi)
    end
    return shifted_dist
end

"""
    extract_theta(chain::Chains, n_weeks::Int)

Reconstructs the deterministic θ (regime shift) states from the posterior samples 
of z_w and σ_GRW. Returns a [samples, n_weeks] matrix.
"""
function extract_theta(chain::Chains, n_weeks::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    σ_GRW = reshape(vec(Array(chain[:σ_GRW])), n_samples, 1)
    
    Z = zeros(Float64, n_samples, n_weeks)
    for w in 1:n_weeks
        Z[:, w] = vec(Array(chain[Symbol("z_w[$w]")]))
    end
    
    raw = cumsum(Z .* σ_GRW, dims=2)
    centered = raw .- mean(raw, dims=2)
    return centered
end
