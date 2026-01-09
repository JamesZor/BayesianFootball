# src/predictions/score_computation/mvpln.jl

using Distributions
using LinearAlgebra
using ..Models # Access to PreGame.StaticMVPLN

# 1. Adapter: Extracts Linear Predictors + Covariance Params
function extract_params(model::Models.PreGame.StaticMVPLN, row)
    return (
        loc_h = row.loc_h,  # Linear Predictor Home (μ + γ + α_h + β_a)
        loc_a = row.loc_a,  # Linear Predictor Away (μ + α_a + β_h)
        σ_h   = row.σ_h,    # Home Noise Scale
        σ_a   = row.σ_a,    # Away Noise Scale
        ρ     = row.ρ       # Correlation
    )
end

# 2. Kernel: Integrates over the latent variable via Monte Carlo
function compute_score_matrix(
    model::Models.PreGame.StaticMVPLN, 
    params; 
    max_goals::Int=12, 
    n_sims::Int=200 # Number of latent draws per posterior sample to approximate the integral
)
    # Unpack parameters (Vectors of length n_samples)
    lh, la = params.loc_h, params.loc_a
    sh, sa = params.σ_h, params.σ_a
    rho = params.ρ
    
    n_samples = length(lh)
    
    # Output container: [Goals x Goals x Samples]
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    # Pre-allocate buffers for the inner loop to avoid GC overhead
    goals = 0:(max_goals-1)
    p_h_vec = zeros(Float64, max_goals)
    p_a_vec = zeros(Float64, max_goals)
    
    # Pre-allocate for the Cholesky noise generation
    # We will generate 2 standard normals per sim
    
    @inbounds for k in 1:n_samples
        
        # --- A. Setup Covariance for this posterior sample ---
        # Construct Cholesky factors manually for the 2x2 matrix
        # Σ = [σ_h²      ρσ_hσ_a]
        #     [ρσ_hσ_a   σ_a²   ]
        # L = [σ_h            0             ]
        #     [ρ*σ_a    σ_a*sqrt(1-ρ^2)]
        
        σ_h_k = sh[k]
        σ_a_k = sa[k]
        ρ_k   = rho[k]
        
        L11 = σ_h_k
        L21 = ρ_k * σ_a_k
        # Clamp sqrt argument to avoiding domain error if ρ is very close to 1/-1
        L22 = σ_a_k * sqrt(max(0.0, 1.0 - ρ_k^2))
        
        μ_h_base = lh[k]
        μ_a_base = la[k]

        # --- B. Monte Carlo Integration ---
        # We average the probability matrices over 'n_sims' realizations of the noise
        
        for m in 1:n_sims
            # 1. Sample Latent Noise (Standard Normals)
            z1 = randn()
            z2 = randn()
            
            # 2. Transform to Correlated Noise
            ϵ_h = L11 * z1
            ϵ_a = L21 * z1 + L22 * z2
            
            # 3. Calculate Realized Rates
            λ_h = exp(μ_h_base + ϵ_h)
            λ_a = exp(μ_a_base + ϵ_a)
            
            # 4. Compute Marginal PDFs
            d_h = Poisson(λ_h)
            d_a = Poisson(λ_a)
            
            @. p_h_vec = pdf(d_h, goals)
            @. p_a_vec = pdf(d_a, goals)
            
            # 5. Accumulate Outer Product
            # We add the probability matrix for this specific "luck" realization
            for j in 1:max_goals
                pj = p_a_vec[j]
                for i in 1:max_goals
                    S[i, j, k] += p_h_vec[i] * pj
                end
            end
        end
        
        # --- C. Normalize ---
        # Divide by n_sims to get the expected probability matrix
        inv_n = 1.0 / n_sims
        for j in 1:max_goals
            for i in 1:max_goals
                S[i, j, k] *= inv_n
            end
        end
    end
    
    return ScoreMatrix(S)
end
