# src/predictions/score_computation/mixture_copula.jl

using Distributions
using LinearAlgebra
using Random
using ..Models # Access to PreGame.StaticMixtureCopula

# --- Helper: Sampling Algorithms ---



# 1. Adapter: Extracts Linear Predictors + Covariance Params
function extract_params(model::Models.PreGame.StaticMixtureCopula, row)
    return (
        loc_h = row.loc_h,  # Linear Predictor Home (μ + γ + α_h + β_a)
        loc_a = row.loc_a,  # Linear Predictor Away (μ + α_a + β_h)
        σ_h   = row.σ_h,    # Home Noise Scale
        σ_a   = row.σ_a,    # Away Noise Scale
        w     = row.w,
       θ_clay = row.θ_clay,
       θ_frank = row.θ_frank
    )
end



"""
Sample from Clayton Copula (Bivariate) using Marshall-Olkin method.
θ > 0.
Generator ψ(t) = (1+t)^(-1/θ). 
Sample V ~ Gamma(1/θ, 1), E1, E2 ~ Exp(1). u = ψ(E1/V).
"""
function sample_clayton(θ::Float64)
    # Avoid numerical issues if θ is extremely close to 0 (though model enforces > 0 via exp)
    θ = max(1e-6, θ)
    
    # 1. Sample latent variable V
    # Distributions.Gamma(shape, scale) -> Gamma(1/θ, 1.0)
    V = rand(Gamma(1.0/θ, 1.0))
    
    # 2. Sample independent exponentials
    E1 = rand(Exponential(1.0))
    E2 = rand(Exponential(1.0))
    
    # 3. Transform
    u = (1.0 + E1/V)^(-1.0/θ)
    v = (1.0 + E2/V)^(-1.0/θ)
    
    return u, v
end

"""
Sample from Frank Copula (Bivariate) using Conditional Sampling method.
v = -1/θ * log(1 + (t * (e^(-θ) - 1)) / (e^(-θ*u)*(1-t) + t))
"""
function sample_frank(θ::Float64)
    # If θ is effectively 0, Frank converges to Independence
    if abs(θ) < 1e-6
        return rand(), rand()
    end
    
    u = rand()
    t = rand() # This acts as the conditional probability C_u(v)
    
    # Pre-compute exp terms
    exp_neg_theta = exp(-θ)
    exp_neg_theta_u = exp(-θ * u)
    
    # Derived conditional inverse formula for Frank
    # v = -1/θ * ln( 1 + (t(e^-θ - 1)) / (e^(-θu)(1-t) + t) )
    numerator = t * (exp_neg_theta - 1.0)
    denominator = exp_neg_theta_u * (1.0 - t) + t
    
    arg = 1.0 + numerator / denominator
    
    # Numerical safety for log
    v = -1.0/θ * log(max(1e-9, arg))
    
    return u, v
end


# --- Main Logic ---

function compute_score_matrix(
    model::Models.PreGame.StaticMixtureCopula, 
    params; 
    max_goals::Int=12, 
    n_sims::Int=200
)
    # Unpack parameters (Vectors of length n_samples)
    lh, la = params.loc_h, params.loc_a
    sh, sa = params.σ_h, params.σ_a
    
    # Mixture Params
    w_vec = params.w            # Vector of vectors [[w1, w2], ...]
    theta_c = params.θ_clay     # Vector
    theta_f = params.θ_frank    # Vector
    
    n_samples = length(lh)
    
    # Output container
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    # Buffers
    goals = 0:(max_goals-1)
    p_h_vec = zeros(Float64, max_goals)
    p_a_vec = zeros(Float64, max_goals)
    
    # Standard Normal for Quantile function
    std_norm = Normal(0, 1)

    @inbounds for k in 1:n_samples
        
        # Params for this posterior sample
        μ_h_base = lh[k]
        μ_a_base = la[k]
        σ_h_k = sh[k]
        σ_a_k = sa[k]
        
        ws = w_vec[k]     # [w_clayton, w_frank]
        θ_c_k = theta_c[k]
        θ_f_k = theta_f[k]
        
        # Categorical distribution for component selection
        # We assume order: 1=Clayton, 2=Frank (matching the training loop)
        comp_dist = Categorical(ws)

        for m in 1:n_sims
            # 1. Sample Component
            z = rand(comp_dist)
            
            # 2. Sample Uniforms (u, v) from Copula
            if z == 1
                u, v = sample_clayton(θ_c_k)
            else
                u, v = sample_frank(θ_f_k)
            end
            
            # 3. Transform Uniforms to Correlated Normal Errors
            # ϵ = Φ⁻¹(u)
            ϵ_h = quantile(std_norm, u)
            ϵ_a = quantile(std_norm, v)
            
            # 4. Calculate Realized Rates
            λ_h = exp(μ_h_base + σ_h_k * ϵ_h)
            λ_a = exp(μ_a_base + σ_a_k * ϵ_a)
            
            # 5. Compute Marginal PDFs
            d_h = Poisson(λ_h)
            d_a = Poisson(λ_a)
            
            @. p_h_vec = pdf(d_h, goals)
            @. p_a_vec = pdf(d_a, goals)
            
            # 6. Accumulate Outer Product
            for j in 1:max_goals
                pj = p_a_vec[j]
                for i in 1:max_goals
                    S[i, j, k] += p_h_vec[i] * pj
                end
            end
        end
        
        # Normalize by n_sims
        inv_n = 1.0 / n_sims
        for j in 1:max_goals
            for i in 1:max_goals
                S[i, j, k] *= inv_n
            end
        end
    end
    
    return ScoreMatrix(S)
end
