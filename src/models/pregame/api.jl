"""
Defines the main `PregameModel` struct and the user-facing API functions.
"""
module PreGame

using ..PreGameInterfaces, ..PreGameComponents, ..TuringHelpers
using Turing, DataFrames, Statistics

export PregameModel, build_turing_model, required_features,
       PoissonGoal, NegativeBinomialGoal, AR1, RandomWalk, Static

struct PregameModel{D<:GoalDistribution, T<:TimeDynamic} <: AbstractPregameModel
    distribution::D
    time_dynamic::T
    home_advantage::Bool
end

function required_features(model::PregameModel)
    return [:home_team, :away_team, :home_score, :away_score, :match_date]
end

#= 
-------- helper functions ----------------
=#
#
# function _prepare_data(model::PregameModel, feature_set) 
#     # Prepare all data variants; the model will use what it needs.
#     data = (
#         n_teams=feature_set.n_teams,
#         n_rounds=feature_set.n_rounds,
#         r_home_ids=feature_set.round_home_ids,
#         r_away_ids=feature_set.round_away_ids,
#         r_home_goals=feature_set.round_home_goals,
#         r_away_goals=feature_set.round_away_goals,
#         f_home_ids=vcat(feature_set.round_home_ids...),
#         f_away_ids=vcat(feature_set.round_away_ids...),
#         f_home_goals=vcat(feature_set.round_home_goals...),
#         f_away_goals=vcat(feature_set.round_away_goals...)
#     )
#   return data
# end
#
#
#= 
-------- api functions ----------------
=#

# function build_turing_model(model::PregameModel{<:GoalDistribution, Static}, feature_set)
#
#     data = _prepare_data(model, feature_set)
#
#     @model function static_model(n_teams, home_ids, away_ids, home_goals, away_goals)
#         # 1. Priors (unchanged)
#         priors ~ to_submodel(TuringHelpers.static_priors(n_teams, model.home_advantage))
#
#         # 2. Identifiability (unchanged)
#         log_α = priors.log_α_raw .- mean(priors.log_α_raw)
#         log_β = priors.log_β_raw .- mean(priors.log_β_raw)
#
#         # 3. Calculate Rates (unchanged)
#         log_λs, log_μs = TuringHelpers.calculate_goal_rates(log_α, log_β, priors.home_adv, home_ids, away_ids)
#
#         # 4. Add the likelihood (THIS IS THE KEY)
#        _n ~ to_submodel(TuringHelpers.add_likelihood(model.distribution, home_goals, away_goals, log_λs, log_μs))
#     end
#
#     return static_model(data.n_teams, data.f_home_ids, data.f_away_ids, data.f_home_goals, data.f_away_goals)
# end
#

# function build_turing_model(model::PregameModel, feature_set)
#
#     # Prepare all data variants; the model will use what it needs.
#     data = (
#         n_teams=feature_set.n_teams, n_rounds=feature_set.n_rounds,
#         r_home_ids=feature_set.round_home_ids, r_away_ids=feature_set.round_away_ids,
#         r_home_goals=feature_set.round_home_goals, r_away_goals=feature_set.round_away_goals,
#         f_home_ids=vcat(feature_set.round_home_ids...), f_away_ids=vcat(feature_set.round_away_ids...),
#         f_home_goals=vcat(feature_set.round_home_goals...), f_away_goals=vcat(feature_set.round_away_goals...)
#     )
#
#     @model function final_model(data)
#
#         # --- STATIC MODEL LOGIC ---
#         if model.time_dynamic isa Static
#             priors ~ to_submodel(static_priors(data.n_teams, model.home_advantage))
#
#             log_α = priors.log_α_raw .- mean(priors.log_α_raw)
#             log_β = priors.log_β_raw .- mean(priors.log_β_raw)
#
#             log_λs = log_α[data.f_home_ids] .+ log_β[data.f_away_ids] .+ priors.home_adv
#             log_μs = log_α[data.f_away_ids] .+ log_β[data.f_home_ids]
#
#             # Likelihood choice for static model
#             if model.distribution isa PoissonGoal
#                 dummy ~ to_submodel(poisson_likelihood(data.f_home_goals, data.f_away_goals, log_λs, log_μs))
#             elseif model.distribution isa NegativeBinomialGoal
#                 dummy ~ to_submodel(negative_binomial_likelihood_goal(data.f_home_goals, data.f_away_goals, log_λs, log_μs))
#             end
#
#         # --- AR1 MODEL LOGIC ---
#         elseif model.time_dynamic isa AR1
#             dynamics ~ to_submodel(ar1_dynamics(data.n_teams, data.n_rounds))
#             attack = dynamics.attack_raw .- mean(dynamics.attack_raw, dims=1)
#             defense = dynamics.defense_raw .- mean(dynamics.defense_raw, dims=1)
#
#             home_adv = 0.0
#             if model.home_advantage
#                  home_adv ~ Normal(log(1.3), 0.2)
#             end
#
#             # Loop through time for the likelihood
#             for t in 1:data.n_rounds
#                 if !isempty(data.r_home_ids[t])
#                     log_λs = attack[data.r_home_ids[t], t] .+ defense[data.r_away_ids[t], t] .+ home_adv
#                     log_μs = attack[data.r_away_ids[t], t] .+ defense[data.r_home_ids[t], t]
#
#                     if model.distribution isa PoissonGoal
#                         dummy ~ to_submodel(poisson_likelihood(data.r_home_goals[t], data.r_away_goals[t], log_λs, log_μs))
#                     elseif model.distribution isa NegativeBinomialGoal
#                         dummy ~ to_submodel(negative_binomial_likelihood_goal(data.r_home_goals[t], data.r_away_goals[t], log_λs, log_μs))
#                     end
#                 end
#             end
#         end
#     end
#
#     return final_model(data)
# end
#
end
