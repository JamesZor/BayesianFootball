# src/Models/PreGame/components/interception.jl

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
Base.@kwdef struct GlobalInterception <: AbstractInterceptionConfig
  μ::ContinuousUnivariateDistribution = Normal(0.2, 0.1)
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================
@model function build_interception(config::GlobalInterception)
    μ ~ config.μ 
    return μ # Sample first, then return
end

# ==========================================
# 3. EXTRACTORS
# ==========================================
function extract_interception(chain::Chains, ::GlobalInterception)
  return vec(Array(chain[Symbol("inter.μ")]))
end
