# src/Models/PreGame/components/interception.jl
# ==========================================
# 1. CONFIGURATIONS
# ==========================================
Base.@kwdef struct GlobalInterception <: AbstractInterceptionConfig
    μ::ContinuousUnivariateDistribution = Normal(0.2, 0.1)
end

# NEW: Seasonal Interception
Base.@kwdef struct SeasonalInterception <: AbstractInterceptionConfig
    # Notice we use a slightly wider prior (0.2 instead of 0.1) 
    # to allow the model to easily "find" extreme high/low scoring seasons
    μ::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================
@model function build_interception(config::GlobalInterception, n_seasons::Int)
    μ ~ config.μ 
    # Even though it's global, we return a vector of the same value 
    # so the downstream indexing logic doesn't break
    return fill(μ, n_seasons) 
end

@model function build_interception(config::SeasonalInterception, n_seasons::Int)
    # Generates a vector of independent intercepts, one for each season
    μ ~ filldist(config.μ, n_seasons)
    return μ 
end

# ==========================================
# 3. EXTRACTORS
# ==========================================
function extract_interception(chain::Chains, ::GlobalInterception, n_seasons::Int)
    val = vec(Array(chain[Symbol("inter.μ")]))
    # Repeat the global value across columns so it matches the seasonal shape
    return repeat(val, 1, n_seasons)
end

function extract_interception(chain::Chains, ::SeasonalInterception, n_seasons::Int)
    n_samples = size(chain, 1) * size(chain, 3) # total draws across all chains
    μ_matrix = zeros(n_samples, n_seasons)
    
    for i in 1:n_seasons
        μ_matrix[:, i] = vec(Array(chain[Symbol("inter.μ[$i]")]))
    end
    
    return μ_matrix
end
