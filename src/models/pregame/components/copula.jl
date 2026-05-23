# src/models/pregame/components/copula.jl

# ==========================================
# 1. CONFIGURATION
# ==========================================
Base.@kwdef struct HierarchicalFrankCopulaConfig <: AbstractCopulaConfig
    prior_base::ContinuousUnivariateDistribution = Normal(0.0, 1.0)
    prior_σ::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
end

# ==========================================
# 2. TURING SUBMODEL
# ==========================================
@model function build_copula(config::HierarchicalFrankCopulaConfig, n_teams::Int)
    κ_base ~ config.prior_base
    σ_κ ~ config.prior_σ
    
    # Non-centered parameterization
    raw_κ ~ filldist(Normal(0, 1), n_teams)
    
    # Scaled team deviations
    δ_κ_scaled = raw_κ .* σ_κ
    
    # Zero-sum constraint (ensures average team has exactly 0 deviation)
    δ_κ = δ_κ_scaled .- mean(δ_κ_scaled)
    
    return (; κ_base, σ_κ, δ_κ)
end

# ==========================================
# 3. EXTRACTOR
# ==========================================
function extract_copula(chain::Chains, ::HierarchicalFrankCopulaConfig, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    
    κ_base = vec(Array(chain[Symbol("$prefix.κ_base")]))
    σ_κ = vec(Array(chain[Symbol("$prefix.σ_κ")]))
    
    raw_κ_matrix = zeros(n_samples, n_teams)
    for i in 1:n_teams
        raw_κ_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_κ[$i]")]))
    end
    
    δ_κ_scaled = raw_κ_matrix .* σ_κ
    δ_κ = δ_κ_scaled .- mean(δ_κ_scaled, dims=2)
    
    return (; κ_base, σ_κ, δ_κ)
end
