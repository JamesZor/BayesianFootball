# src/models/maher.jl

using Turing, Distributions, LinearAlgebra

# --- 1. The Turing Model Definition (Unchanged) ---
@model function basic_maher_model_raw(
    home_team_ids, away_team_ids,
    home_goals, away_goals,
    n_teams
)
    # Priors
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    log_γ ~ Normal(log(1.3), 0.2)
    
    # Identifiability Constraint
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    # Transform to original scale
    α = exp.(log_α)
    β = exp.(log_β)
    γ = exp(log_γ)
    
    # Likelihood
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]
        
        λ = α[i] * β[j] * γ
        μ = α[j] * β[i]
        
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
end

# --- 2. Implementation of the ModelDefinition Protocol ---

"""
get_required_features(::MaherBasic)

Declares that the basic Maher model needs team IDs and the total number of teams.
"""
function get_required_features(::MaherBasic)
    return [:home_team_ids, :away_team_ids, :n_teams]
end

"""
build_turing_model(::MaherBasic, features::NamedTuple, home_goals::V, away_goals::V)

Constructs the Turing model instance for the basic Maher model.
"""
function build_turing_model(::MaherBasic, features::NamedTuple, home_goals::V, away_goals::V) where {V<:AbstractVector}
    return basic_maher_model_raw(
        features.home_team_ids,
        features.away_team_ids,
        home_goals,
        away_goals,
        features.n_teams
    )
end
