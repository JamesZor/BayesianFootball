
# --- Model File: static-poisson.jl ---

using Turing
using DataFrames
using LinearAlgebra
using ..PreGameInterfaces # Use the abstract type
using ..TuringHelpers


# Export the concrete model struct and its build function
export StaticSimplexPoisson, build_turing_model, predict

# 1. DEFINE A CONCRETE STRUCT FOR THE MODEL
struct StaticSimplexPoisson <: AbstractPregameModel end
# 2. DEFINE THE TURING MODEL LOGIC

@model function static_simplex_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors ---
      # log_α_scale ~ Truncated(Normal(0, 1.5), 0, Inf)
      # log_β_scale ~ Truncated(Normal(0, 1.5), 0, Inf)
      log_α_scale ~ LogNormal(0, 1)
      log_β_scale ~ LogNormal(0, 1)
      home_adv ~ Normal(log(1.3), 0.2)

      # --- Non-Centered Parameterization for Identifiability ---
      # Sample n-1 raw parameters from a standard normal distribution
      # These are independent of the scale! This is the key.
      α_raw_offsets ~ MvNormal(n_teams - 1, 1.0)
      β_raw_offsets ~ MvNormal(n_teams - 1, 1.0)

      # Deterministically create the full n-team vectors
      # that sum to zero.
      α_offsets = vcat(α_raw_offsets, -sum(α_raw_offsets))
      β_offsets = vcat(β_raw_offsets, -sum(β_raw_offsets))

      # Apply the scale and mean AFTER sampling.
      # This transformation happens "outside" the sampler's main work.
      log_α := log_α_scale .* α_offsets
      log_β := log_β_scale .* β_offsets


      # --- Calculate Goal Rates ---
      log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
      log_μs = log_α[away_ids] .+ log_β[home_ids]

    if !ismissing(home_goals)
        # # --- TRAINING CASE ---
        # for i in eachindex(home_goals)
        #   home_goals[i] ~ LogPoisson(log_λs[i])
        #   away_goals[i] ~ LogPoisson(log_μs[i])
        # end
        home_goals ~ arraydist(LogPoisson.(log_λs))
        away_goals ~ arraydist(LogPoisson.(log_μs))
    else
    #     # --- PREDICTION CASE ---
        predicted_home_goals ~ arraydist(LogPoisson.(log_λs))
        predicted_away_goals ~ arraydist(LogPoisson.(log_μs))
    end
    #
    return nothing
end


# 3. DEFINE THE API FUNCTION FOR TRAINING
"""
    build_turing_model(model::StaticPoisson, feature_set::FeatureSet)

Builds the Turing model for the **training phase**.
"""
function build_turing_model(model::StaticSimplexPoisson, feature_set::FeatureSet)
    # This helper function flattens the round-based data from the FeatureSet
    data = TuringHelpers.prepare_data(model, feature_set)
    
    return static_simplex_poisson_model(
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
function build_turing_model(model::StaticSimplexPoisson, n_teams::Int, home_ids::Vector{Int}, away_ids::Vector{Int})
    return static_simplex_poisson_model(
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
function predict(model::StaticSimplexPoisson, df_to_predict::DataFrame, feature_set::FeatureSet, chains::Chains)
  team_map = feature_set.team_map 
  n_teams = feature_set.n_teams
  home_ids_to_predict = [team_map[name] for name in df_to_predict.home_team]
  away_ids_to_predict = [team_map[name] for name in df_to_predict.away_team]
  turing_pred_model = build_turing_model(model, n_teams, home_ids_to_predict, away_ids_to_predict)
  chains_goals = Turing.predict(turing_pred_model, chains)
  return chains_goals
end 

