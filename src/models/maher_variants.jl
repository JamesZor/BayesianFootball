# src/models/maher_variants.jl
# using Turing, Distributions, LinearAlgebra

@model function maher_league_ha_model(features)
    # --- Unpack the required features ---
    home_team_ids = features.home_team_ids
    away_team_ids = features.away_team_ids
    home_goals = features.home_goals # Correct generic name
    away_goals = features.away_goals # Correct generic name
    n_teams = features.n_teams
    n_leagues = features.n_leagues
    league_ids = features.league_ids
    # ---
    #
    # --- Priors (Attack and Defense are the same as the base model) ---
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    # --- MODIFICATION START ---
    # Home advantage is now a vector of parameters, one for each league.
    # The prior states that we expect each league's home advantage to be around 1.3.
    log_γ_leagues ~ MvNormal(fill(log(1.3), n_leagues), 0.2 * I)
    # --- MODIFICATION END ---
    
    # Identifiability constraints (same as base model)
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    α = exp.(log_α)
    β = exp.(log_β)
    γ_leagues = exp.(log_γ_leagues)

    # --- Likelihood ---
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]
        l = league_ids[k] # Get the league index for the current match

        # --- MODIFICATION START ---
        # We now index into the γ_leagues vector using the match's league index.
        λ = α[i] * β[j] * γ_leagues[l]
        # --- MODIFICATION END ---
        μ = α[j] * β[i]
        
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
end
