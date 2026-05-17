# src/models/pregame/components/kappa.jl

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
Base.@kwdef struct GlobalKappa <: AbstractKappaConfig
    # Use a truncated normal so Turing automatically maps the parameter to (0, ∞)
    κ_global::ContinuousUnivariateDistribution = truncated(Normal(0.84, 0.2), lower=0.01)
end

Base.@kwdef struct HierarchicalTeamKappa <: AbstractKappaConfig
    # Base rate should also be strictly positive
    κ_base::ContinuousUnivariateDistribution = truncated(Normal(1.0, 0.2), lower=0.01)
    σ_κ::ContinuousUnivariateDistribution = truncated(Normal(0, 0.1), lower=0.0)
end

# ==========================================
# 2. TURING SUBMODELS
# ==========================================
@model function build_kappa(config::GlobalKappa, n_teams::Int)
  κ_global ~ config.κ_global 
  return fill(κ_global, n_teams)
end

@model function build_kappa(config::HierarchicalTeamKappa, n_teams::Int)
    κ_base ~ config.κ_base
    σ_κ ~ config.σ_κ
    
    # Non-centered parameterization for stable gradients
    κ_team_raw ~ filldist(Normal(0, 1), n_teams) 
    
    # Calculate final team-specific conversion rates
    # Using Softplus: log(1 + exp(x))
    # This acts like max(x, 0) but is completely smooth and differentiable!
    κ_team_linear = κ_base .+ (κ_team_raw .* σ_κ)
    κ_team = log1p.(exp.(κ_team_linear)) 
    
    return κ_team 
end

# ==========================================
# 3. EXTRACTORS
# ==========================================

function extract_kappa(chain::Chains, ::GlobalKappa, n_teams::Int)
    # The @submodel macro in the main engine will prefix this with "kap."
    val = vec(Array(chain[Symbol("kap.κ_global")]))
    return repeat(val, 1, n_teams)
end

function extract_kappa(chain::Chains, ::HierarchicalTeamKappa, n_teams::Int)
    base = vec(Array(chain[Symbol("kap.κ_base")]))
    sigma = vec(Array(chain[Symbol("kap.σ_κ")]))
    
    n_samples = length(base)
    κ_matrix = zeros(n_samples, n_teams)
    
    for i in 1:n_teams
        team_raw = vec(Array(chain[Symbol("kap.κ_team_raw[$i]")]))
        # Apply the same softplus transformation used in the model
        linear_val = base .+ (team_raw .* sigma)
        κ_matrix[:, i] = log1p.(exp.(linear_val))
    end
    
    return κ_matrix
end
