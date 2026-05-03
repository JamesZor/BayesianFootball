# src/Models/PreGame/components/home_advantage.jl

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
Base.@kwdef struct GlobalHomeAdvantage <: AbstractHomeAdvantageConfig
    γ_global::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
end

Base.@kwdef struct HierarchicalTeamHomeAdvantage <: AbstractHomeAdvantageConfig
    γ_base::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
    σ_γ::ContinuousUnivariateDistribution = truncated(Normal(0, 0.1), lower=0.0)
end

Base.@kwdef struct HierarchicalLeagueHomeAdvantage <: AbstractHomeAdvantageConfig
    γ_base::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
    σ_γ::ContinuousUnivariateDistribution = truncated(Normal(0, 0.1), lower=0.0)
end


# ==========================================
# 2. TURING SUBMODELS
# ==========================================
@model function build_home_advantage(config::GlobalHomeAdvantage, n_teams::Int)
    γ_global ~ config.γ_global
    return fill(γ_global, n_teams)
end

@model function build_home_advantage(config::HierarchicalTeamHomeAdvantage, n_teams::Int)
    γ_base ~ config.γ_base
    σ_γ ~ config.σ_γ
    γ_team_raw ~ filldist(Normal(0, 1), n_teams) 
    return γ_base .+ (γ_team_raw .* σ_γ)
end



@model function build_home_advantage(config::HierarchicalLeagueHomeAdvantage, n_leagues::Int)
    γ_base ~ config.γ_base
    σ_γ ~ config.σ_γ
    γ_league_raw ~ filldist(Normal(0, 1), n_leagues) 
    return γ_base .+ (γ_league_raw .* σ_γ)
end


# ==========================================
# 3. EXTRACTORS
# ==========================================
function extract_home_advantage(chain::Chains, ::GlobalHomeAdvantage, n_teams::Int)
    # Turing @submodel macro will prefix these with "ha."
    val = vec(Array(chain[Symbol("ha.γ_global")]))
    
    # Repeat the global value into a (Samples x Teams) matrix for consistency
    return repeat(val, 1, n_teams)
end

function extract_home_advantage(chain::Chains, ::HierarchicalTeamHomeAdvantage, n_teams::Int)
    base = vec(Array(chain[Symbol("ha.γ_base")]))
    sigma = vec(Array(chain[Symbol("ha.σ_γ")]))
    
    n_samples = length(base)
    ha_matrix = zeros(n_samples, n_teams)
    
    for i in 1:n_teams
        team_raw = vec(Array(chain[Symbol("ha.γ_team_raw[$i]")]))
        ha_matrix[:, i] = base .+ (team_raw .* sigma)
    end
    
    return ha_matrix
end

function extract_home_advantage(chain::Chains, ::HierarchicalLeagueHomeAdvantage, n_leagues::Int)
    base = vec(Array(chain[Symbol("ha.γ_base")]))
    sigma = vec(Array(chain[Symbol("ha.σ_γ")]))
    
    n_samples = length(base)
    ha_matrix = zeros(n_samples, n_leagues)
    
    for i in 1:n_teams
        team_raw = vec(Array(chain[Symbol("ha.γ_league_raw[$i]")]))
        ha_matrix[:, i] = base .+ (team_raw .* sigma)
    end
    
    return ha_matrix
end
