# src/predictions/score_computation/dixon_coles_negbin.jl

using ..Models: PreGame
using Distributions
# using LogExpFunctions: logpdf

# 1. Adapter: Maps the flat model output to named parameters
# Matches the NamedTuple output from your extract_parameters function
function extract_params(model::Models.PreGame.AbstractDynamicDixonColesNegBinModel, row)
    return (
        λ_h = row.λ_h, 
        λ_a = row.λ_a, 
        r_h = row.r_h, 
        r_a = row.r_a, 
        ρ = row.ρ
    )
end

# 2. Kernel: Params -> ScoreMatrix
function compute_score_matrix(model::Models.PreGame.AbstractDynamicDixonColesNegBinModel, params; max_goals::Int=12)
    
    # Extract the arrays of posterior samples
    L_h, L_a = params.λ_h, params.λ_a
    R_h, R_a = params.r_h, params.r_a
    Rho = params.ρ
    
    n_samples = length(L_h)
    
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    # Pre-allocate arrays for marginal log-probs to avoid re-calculating inside the grid
    log_h_marg = zeros(Float64, max_goals)
    log_a_marg = zeros(Float64, max_goals)
    
    @inbounds for k in 1:n_samples
        # 1. Unpack Parameters for this specific MCMC sample
        # Note: In our Turing extract_parameters, these are already exponentiated 
        # to their final actual values, so we don't need exp() here!
        λ = L_h[k]
        μ = L_a[k]
        r_h = R_h[k]
        r_a = R_a[k]
        ρ = Rho[k]
        
        # 2. Pre-calculate Independent Marginals using our Robust Negative Binomial
        dist_h = RobustNegativeBinomial(r_h, λ)
        dist_a = RobustNegativeBinomial(r_a, μ)
        
        for i in 1:max_goals
            log_h_marg[i] = logpdf(dist_h, i - 1)
        end
        for j in 1:max_goals
            log_a_marg[j] = logpdf(dist_a, j - 1)
        end
        
        # 3. Fill Grid with Dixon-Coles Correction
        for j in 1:max_goals
            a_score = j - 1
            lp_a = log_a_marg[j]
            
            for i in 1:max_goals
                h_score = i - 1
                lp_h = log_h_marg[i]
                
                # Base Independent Log-Probability
                log_p_base = lp_h + lp_a
                
                # Calculate Tau (Correction Factor)
                # Derived from Dixon & Coles (1997) Eq 4.2 
                tau = 1.0
                
                if h_score == 0 && a_score == 0
                    tau = 1.0 - (λ * μ * ρ)
                elseif h_score == 1 && a_score == 0
                    tau = 1.0 + (μ * ρ)
                elseif h_score == 0 && a_score == 1
                    tau = 1.0 + (λ * ρ)
                elseif h_score == 1 && a_score == 1
                    tau = 1.0 - ρ
                end
                
                # Safety: If constraint violated (tau <= 0), prob is 0.
                if tau <= 0
                    S[i, j, k] = 0.0
                else
                    # P(x,y) = P_indep(x,y) * tau
                    # log P = log P_base + log(tau)
                    S[i, j, k] = exp(log_p_base + log(tau))
                end
            end
        end
    end
    
    return ScoreMatrix(S)
end
