# src/predictions/score_computation/frank_copula.jl

function compute_score_matrix_discrete_copula(
    params; 
    max_goals::Int=12
)
    lh = params.loc_h
    la = params.loc_a
    rh = params.r_h
    ra = params.r_a
    kappa = params.κ
    
    n_samples = length(lh)
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    for k in 1:n_samples
        dist_h = MyDistributions.RobustNegativeBinomial(rh[k], lh[k])
        dist_a = MyDistributions.RobustNegativeBinomial(ra[k], la[k])
        
        κ_val = kappa[k]
        
        # Precompute marginal CDFs
        u = zeros(Float64, max_goals + 1)
        v = zeros(Float64, max_goals + 1)
        
        u[1] = 0.0 # CDF at -1
        v[1] = 0.0
        
        for g in 0:(max_goals-1)
            u[g+2] = cdf(dist_h, g)
            v[g+2] = cdf(dist_a, g)
        end
        
        for i in 1:max_goals
            for j in 1:max_goals
                u1 = u[i+1]
                u0 = u[i]
                v1 = v[j+1]
                v0 = v[j]
                
                C11 = MyDistributions.frank_copula(u1, v1, κ_val)
                C01 = MyDistributions.frank_copula(u0, v1, κ_val)
                C10 = MyDistributions.frank_copula(u1, v0, κ_val)
                C00 = MyDistributions.frank_copula(u0, v0, κ_val)
                
                pmf = C11 - C01 - C10 + C00
                S[i, j, k] = max(pmf, 0.0)
            end
        end
        
        sum_S = sum(S[:, :, k])
        if sum_S > 0
            S[:, :, k] ./= sum_S
        end
    end
    
    return S 
end

# 1. Adapter: DataFrame Row -> NamedTuple
function extract_params(model::TypesInterfaces.AbstractFrankCopulaNegBinModel, row)
    return (
        loc_h = row.λ_h,
        loc_a = row.λ_a,
        r_h = row.r_h,
        r_a = row.r_a,
        κ = row.κ_frank
    )
end

# 2. Kernel Wrapper
function compute_score_matrix(model::TypesInterfaces.AbstractFrankCopulaNegBinModel, params; max_goals::Int=12)
    return compute_score_matrix_discrete_copula(params; max_goals=max_goals)
end
