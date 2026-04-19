# src/predictions/score_computation/bivariate_poisson.jl

using ..MyDistributions: BivariateLogPoisson
using ..Models: PreGame

# 1. Adapter: Now expects Thetas
function extract_params(model::Models.PreGame.AbstractBivariatePoissonModel, row)
    return (θ_1 = row.θ_1, θ_2 = row.θ_2, θ_3 = row.θ_3)
end

# 2. Kernel: Params -> ScoreMatrix
function compute_score_matrix(model::Models.PreGame.AbstractBivariatePoissonModel, params; max_goals::Int=12)
    
    # Unpack Thetas directly
    T1, T2, T3 = params.θ_1, params.θ_2, params.θ_3
    n_samples = length(T1)
    
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    @inbounds for k in 1:n_samples
        # No log() call needed here anymore
        dist = BivariateLogPoisson(T1[k], T2[k], T3[k])
        
        # Grid evaluation remains the same
        for j in 1:max_goals
            a_score = j - 1
            for i in 1:max_goals
                h_score = i - 1
                
                # logpdf uses the Thetas directly
                log_p = logpdf(dist, [h_score, a_score])
                S[i, j, k] = exp(log_p)
            end
        end
    end
    
    return ScoreMatrix(S)
end
