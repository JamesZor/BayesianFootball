# --- Model File: static-poisson.jl ---
using DataFrames
using Turing
using LinearAlgebra
using ..PreGameInterfaces # Use the abstract type
using ..TuringHelpers
using Base.Threads

# Export the concrete model struct and its build function
export StaticPoisson, build_turing_model, predict

struct StaticPoisson <: AbstractPregameModel end

# NEW: The main @model block, isolated
@model function static_poisson_model_train(n_teams, home_ids, away_ids, 
                                    home_goals, away_goals, ::Type{T} = Float64) where {T}
        # --- Priors ---
        log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        home_adv ~ Normal(log(1.3), 0.2)

        # --- Identifiability Constraint ---
        log_α := log_α_raw .- mean(log_α_raw) # using := to added to track vars,
        log_β := log_β_raw .- mean(log_β_raw)

        # --- Calculate Goal Rates ---
        log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
        log_μs = log_α[away_ids] .+ log_β[home_ids]

        # --- TRAINING CASE ---
        for i in eachindex(home_goals)
          home_goals[i] ~ LogPoisson(log_λs[i])
          away_goals[i] ~ LogPoisson(log_μs[i])
        end
    return nothing
end


@model function static_poisson_model_predict(n_teams, home_ids, away_ids, 
                                    home_goals, away_goals, ::Type{T} = Float64) where {T}
        # --- Priors ---
        log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        home_adv ~ Normal(log(1.3), 0.2)

        # --- Identifiability Constraint ---
        log_α := log_α_raw .- mean(log_α_raw) # using := to added to track vars,
        log_β := log_β_raw .- mean(log_β_raw)

        # --- Calculate Goal Rates ---
        log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
        log_μs = log_α[away_ids] .+ log_β[home_ids]

        predicted_home_goals ~ arraydist(LogPoisson.(log_λs))
        predicted_away_goals ~ arraydist(LogPoisson.(log_μs))
    return nothing
end

# # NEW: A helper to get the data from the FeatureSet
# function _prepare_data(feature_set::FeatureSet)
#
#     home_ids = vcat(feature_set.round_home_ids...)
#     away_ids = vcat(feature_set.round_away_ids...)
#     home_goals = vcat(feature_set.round_home_goals...)
#     away_goals = vcat(feature_set.round_away_goals...)
#
#     return (
#         n_teams = feature_set.n_teams,
#         f_home_ids = home_ids,
#         f_away_ids = away_ids,
#         f_home_goals = home_goals,
#         f_away_goals = away_goals
#     )
# end


# 3. DEFINE THE API FUNCTION FOR TRAINING
"""
    build_turing_model(model::StaticPoisson, feature_set::FeatureSet)

Builds the Turing model for the **training phase**.
"""
function build_turing_model(model::StaticPoisson, feature_set::FeatureSet)
    # This helper function flattens the round-based data from the FeatureSet
    data = TuringHelpers.prepare_data(model, feature_set)
    
    return static_poisson_model_train(
        data.n_teams, 
        data.f_home_ids, 
        data.f_away_ids, 
        data.f_home_goals, 
        data.f_away_goals
    )
end

# 4. DEFINE THE API FUNCTION FOR PREDICTION
"""
    build_turing_model(model::StaticPoisson, n_teams::Int, home_ids::Vector{Int}, away_ids::Vector{Int})

Builds the Turing model for the **prediction phase**. 
It takes team IDs directly, as goals are unknown.
"""
function build_turing_model(model::StaticPoisson, n_teams::Int, home_ids::Vector{Int}, away_ids::Vector{Int})
    return static_poisson_model_predict(
        n_teams,
        home_ids,
        away_ids,
        missing, # Goals are missing for prediction
        missing
    )
end


"""
    predict(model::StaticPoisson, data_to_predict::DataFrame, feature_set::FeatureSet)

TBW
"""
function predict(model::StaticPoisson, df_to_predict::DataFrame, feature_set::FeatureSet, chains::Chains)
  team_map = feature_set.team_map 
  n_teams = feature_set.n_teams
  home_ids_to_predict = [team_map[name] for name in df_to_predict.home_team]
  away_ids_to_predict = [team_map[name] for name in df_to_predict.away_team]
  turing_pred_model = build_turing_model(model, n_teams, home_ids_to_predict, away_ids_to_predict)
  chains_goals = Turing.predict(turing_pred_model, chains)
  return chains_goals
end 
