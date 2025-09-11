# src/models/maher_variants.jl

using Turing, Distributions, LinearAlgebra

# --- 1. The Turing Model Definition (Unchanged) ---
@model function maher_league_ha_model(
    home_team_ids, away_team_ids,
    home_goals, away_goals,
    n_teams, n_leagues, league_ids
)
    # Priors
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    # Home advantage is now a vector of parameters, one for each league
    log_γ_leagues ~ MvNormal(fill(log(1.3), n_leagues), 0.2 * I)
    
    # Identifiability constraints
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    α = exp.(log_α)
    β = exp.(log_β)
    γ_leagues = exp.(log_γ_leagues)

    # Likelihood
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]
        l = league_ids[k] # Get the league index

        # Index into the γ_leagues vector using the match's league index
        λ = α[i] * β[j] * γ_leagues[l]
        μ = α[j] * β[i]
        
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
end

# --- 2. Implementation of the ModelDefinition Protocol ---

"""
get_required_features(::MaherLeagueHA)

Declares the features needed for the league home-advantage model.
"""
function get_required_features(::MaherLeagueHA)
    return [:home_team_ids, :away_team_ids, :n_teams, :n_leagues, :league_ids]
end

"""
build_turing_model(::MaherLeagueHA, features::NamedTuple, home_goals::V, away_goals::V)

Constructs the Turing model instance for the league home-advantage variant.
"""
function build_turing_model(::MaherLeagueHA, features::NamedTuple, home_goals::V, away_goals::V) where {V<:AbstractVector}
    return maher_league_ha_model(
        features.home_team_ids,
        features.away_team_ids,
        home_goals,
        away_goals,
        features.n_teams,
        features.n_leagues,
        features.league_ids
    )
end
