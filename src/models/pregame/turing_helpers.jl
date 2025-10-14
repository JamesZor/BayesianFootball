"""
This module contains the dispatched helper functions and submodels
that build the statistical components of a Turing model.
"""
module TuringHelpers

using Turing
using Distributions
using LinearAlgebra
using ..PreGameInterfaces, ..PreGameComponents

export static_priors, ar1_dynamics, _add_likelihood, _get_static_log_goal_rates


# --- SUBMODEL for Static Priors ---
@model function static_priors(n_teams::Int, has_home_advantage::Bool)
    # Priors for static attack and defense parameters
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    
    home_adv = 0.0
    if has_home_advantage
        home_adv ~ Normal(log(1.3), 0.2)
    end
    
    return (log_α_raw = log_α_raw, log_β_raw = log_β_raw, home_adv = home_adv)
end

# --- PURE HELPER for Static Goal Rate Calculation ---
function _get_static_log_goal_rates(log_α, log_β, home_adv, home_ids, away_ids)
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]
    return log_λs, log_μs
end

# --- LIKELIHOOD HELPER (can remain a pure function) ---
function _add_likelihood(::PoissonGoal, goals_home, goals_away, log_λs, log_μs)
    goals_home .~ LogPoisson.(log_λs)
    goals_away .~ LogPoisson.(log_μs)
    return nothing
end


# --- SUBMODEL for AR1 Dynamics ---
@model function ar1_dynamics(n_teams::Int, n_rounds::Int)
    # Priors for the AR(1) process
    ρ_att ~ Normal(0.0, 0.5)
    ρ_def ~ Normal(0.0, 0.5)
    σ_att ~ Truncated(Normal(0, 1), 0, Inf)
    σ_def ~ Truncated(Normal(0, 1), 0, Inf)

    # Initialize raw parameters
    attack_raw = Matrix{Real}(undef, n_teams, n_rounds)
    defense_raw = Matrix{Real}(undef, n_teams, n_rounds)

    attack_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
    defense_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)

    # Time-varying dynamics
    for t in 2:n_rounds
        attack_raw[:, t] ~ MvNormal(ρ_att * attack_raw[:, t-1], σ_att * I)
        defense_raw[:, t] ~ MvNormal(ρ_def * defense_raw[:, t-1], σ_def * I)
    end
    
    return attack_raw, defense_raw
end

# --- LIKELIHOOD HELPER for Dynamic Models ---
function _add_likelihood(
    ::PoissonGoal, n_rounds,
    round_home_ids, round_away_ids,
    round_home_goals, round_away_goals,
    attack, defense, home_adv
)
    for t in 1:n_rounds
        home_ids = round_home_ids[t]
        away_ids = round_away_ids[t]
        
        λs = exp.(attack[home_ids, t] .+ defense[away_ids, t] .+ home_adv)
        μs = exp.(attack[away_ids, t] .+ defense[home_ids, t])

        round_home_goals[t] .~ Distributions.Poisson.(λs)
        round_away_goals[t] .~ Distributions.Poisson.(μs)
    end
    return nothing
end

end
