# --- Model File: static-poisson.jl ---

using Turing
using LinearAlgebra
using ..PreGameInterfaces # Use the abstract type
using ..TuringHelpers
using Base.Threads

# Export the concrete model struct and its build function
export StaticPoisson, build_turing_model

struct StaticPoisson <: AbstractPregameModel end

# NEW: The main @model block, isolated
@model function static_poisson_model(n_teams, home_ids, away_ids, 
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

    if !ismissing(home_goals)
        # --- TRAINING CASE ---
        # for i in eachindex(home_goals)
        #   home_goals[i] ~ LogPoisson(log_λs[i])
        #   away_goals[i] ~ LogPoisson(log_μs[i])
        # end
        home_goals ~ arraydist(LogPoisson.(Log_λs))
        home_goals ~ arraydist(LogPoisson.(Log_μs))
    else
        predicted_home_goals ~ arraydist(LogPoisson.(log_λs))
        predicted_away_goals ~ arraydist(LogPoisson.(log_μs))
    #     # --- PREDICTION CASE ---
    #     # We use `T` to help Turing infer the return type
    #     predicted_home_goals = Vector{T}(undef, length(home_ids))
    #     predicted_away_goals = Vector{T}(undef, length(home_ids))
    #
    #     for i in eachindex(home_ids)
    #         predicted_home_goals[i] ~ LogPoisson(log_λs[i])
    #         predicted_away_goals[i] ~ LogPoisson(log_μs[i])
    end
    #
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
    
    return static_poisson_model(
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
    return static_poisson_model(
        n_teams,
        home_ids,
        away_ids,
        missing, # Goals are missing for prediction
        missing
    )
end

# # NEW: The main API function that returns a Turing model
# """
# Builds the Turing model for training or prediction.
#
# - To train: Pass the `feature_set` from training data.
# """
# function build_turing_model(model::StaticPoisson, feature_set::FeatureSet)
#     # This is the "TRAINING" model
#     data = _prepare_data(feature_set)
#     return static_poisson_model(data.n_teams, data.f_home_ids, data.f_away_ids, 
#                                 data.f_home_goals, data.f_away_goals)
# end
#
#
# function predict(model::StaticPoisson, data_to_predict, chains::Chains) 
#   features = _prepare_data(data_to_predict, :predict)
#
#   model = static_poisson_model(features.n_teams, features.f_home_ids, features.f_away_ids, 
#                               missing, missing)
#
#   # 2. Set up for parallel execution
#   n_samples = length(chains)
#   n_threads = Threads.nthreads()
#   # Create an array to hold the prediction chains from each thread
#   split_chains = Vector{Chains}(undef, n_threads)
#
#   # 3. Use a threaded loop to run predict on chunks of the chain
#   Threads.@threads for i in 1:n_threads
#       # Divide the samples among the threads
#       range = i:n_threads:n_samples
#
#       # Each thread predicts using only its subset of the original chain
#       # This is the key step!
#       split_chains[i] = predict(model, chains[range])
#   end
#
#   # 4. Concatenate the results from all threads into a single chain
#   # need to concrete all the treads chains
#   prediction_chain_multi = chainscat(split_chains...)
#
#   return prediction_chain_multi
#
# end 
#
#
# function build_turing_model(model::StaticPoisson, chains::Chains, feature_set_to_predict_on::FeatureSet)
#     # This is the "PREDICTION" model
#     data = _prepare_data(feature_set_to_predict_on)
#     return static_poisson_model(data.n_teams, data.f_home_ids, data.f_away_ids, 
#                                 missing, missing) | chains
# end
#
#
#
#
#
#
