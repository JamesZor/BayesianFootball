"""
This module contains the modular submodels that are the building blocks
for the main model defined in the API.
"""
module TuringHelpers

using Turing, Distributions, LinearAlgebra, Statistics
using ..PreGameInterfaces 
using ....Features: FeatureSet  # <--- ADD THIS LINE

export prepare_data

function prepare_data(model::AbstractPregameModel, feature_set) 
    # Prepare all data variants; the model will use what it needs.
    data = (
        n_teams=feature_set.n_teams,
        n_rounds=feature_set.n_rounds,
        r_home_ids=feature_set.round_home_ids,
        r_away_ids=feature_set.round_away_ids,
        r_home_goals=feature_set.round_home_goals,
        r_away_goals=feature_set.round_away_goals,
        f_home_ids=vcat(feature_set.round_home_ids...),
        f_away_ids=vcat(feature_set.round_away_ids...),
        f_home_goals=vcat(feature_set.round_home_goals...),
        f_away_goals=vcat(feature_set.round_away_goals...)
    )
  return data
end



# --- PRIORS SUBMODEL (Static) ---
@model function static_priors(n_teams::Int, has_home_advantage::Bool)
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    
    home_adv = 0.0
    if has_home_advantage
        home_adv ~ Normal(log(1.3), 0.2)
    end
    
    return (; log_α_raw, log_β_raw, home_adv)
end

# --- DYNAMICS SUBMODEL (AR1) ---
@model function ar1_dynamics(n_teams::Int, n_rounds::Int)
    ρ_att ~ Normal(0.0, 0.5)
    ρ_def ~ Normal(0.0, 0.5)
    σ_att ~ Truncated(Normal(0, 1), 0, Inf)
    σ_def ~ Truncated(Normal(0, 1), 0, Inf)

    attack_raw = Matrix{Real}(undef, n_teams, n_rounds)
    defense_raw = Matrix{Real}(undef, n_teams, n_rounds)

    attack_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
    defense_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)

    for t in 2:n_rounds
        attack_raw[:, t] ~ MvNormal(ρ_att * attack_raw[:, t - 1], σ_att * I)
        defense_raw[:, t] ~ MvNormal(ρ_def * defense_raw[:, t - 1], σ_def * I)
    end
    
    return (attack_raw=attack_raw, defense_raw=defense_raw)
end


# ---- link functions 
function calculate_goal_rates(log_α, log_β, log_home_adv, home_ids, away_ids)
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ log_home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]
    return log_λs, log_μs
end


#
# # --- LIKELIHOOD SUBMODEL (Poisson) ---
# @model function add_likelihood(::PoissonGoal, goals_home, goals_away, log_λs, log_μs)
#     for i in eachindex(goals_home)
#         goals_home[i] ~ LogPoisson(log_λs[i])
#         goals_away[i] ~ LogPoisson(log_μs[i])
#     end
#   return nothing
# end
#
# # --- LIKELIHOOD SUBMODEL (Negative Binomial) ---
# @model function add_likelihood(::NegativeBinomialGoal, goals_home, goals_away, log_λs, log_μs)
#     # 1. DEFINE THE PRIOR FOR THE NEW PARAMETER
#   # ϕ ∈ (0, 1)
#     # ϕ ~ Exponential(1.0)
#     ϕ ~ Gamma(1.2, 1.0)
#
#     # pre compute
#     λs = exp.(log_λs)
#     μs = exp.(log_μs)
#
#     p_home = ϕ ./ (ϕ .+ λs)
#     p_away = ϕ ./ (ϕ .+ μs)
#
#
#     for i in eachindex(goals_home)
#     goals_home[i] ~ NegativeBinomial(ϕ, p_home[i])
#     goals_away[i] ~ NegativeBinomial(ϕ, p_away[i])
#     end
#
#   return nothing
#
# end

end
