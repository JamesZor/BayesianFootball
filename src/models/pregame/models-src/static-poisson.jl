# --- Model File: static-poisson.jl ---
using DataFrames
using Turing
using LinearAlgebra
# using ..PreGameInterfaces # Use the abstract type
# using ..TuringHelpers
# using ...TypesInterfaces
using Base.Threads

# Export the concrete model struct and its build function
export StaticPoisson, build_turing_model, predict

struct StaticPoisson <: AbstractStaticPoissonModel end

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
        # for i in eachindex(home_goals)
        #   home_goals[i] ~ LogPoisson(log_λs[i])
        #   away_goals[i] ~ LogPoisson(log_μs[i])
        # end

        home_goals ~ arraydist(LogPoisson.(log_λs))
        away_goals ~ arraydist(LogPoisson.(log_μs))
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

# # --- PREDICTION CASE (THE HACK) ---
#
#         # 1. Calculate the deterministic values
#         lambdas = exp.(log_λs)
#         mus = exp.(log_μs)
#
#         # 2. "Sample" them from a Dirac distribution
#         # This tricks Turing.predict() into saving them.
#         lambda ~ arraydist(Dirac.(lambdas))
#         mu ~ arraydist(Dirac.(mus))
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
    
    return static_poisson_model_train(
        data.n_teams, 
        data.flat_home_ids, 
        data.flat_away_ids, 
        data.flat_home_goals, 
        data.flat_away_goals
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
    predict(model::StaticPoisson, df_to_predict::DataFrame, vocabulary::Vocabulary, chains::Chains)

Generates posterior predictive samples for new data.

This function uses the global `Vocabulary` to map team names from the `df_to_predict`
DataFrame to the integer IDs that the trained model understands.
"""
function predict(model::StaticPoisson, df_to_predict::DataFrame, vocabulary::Vocabulary, chains::Chains)
  # To get the team_map and n_teams, we now access the dictionary within the Vocabulary
  team_map = vocabulary.mappings[:team_map]
  n_teams = vocabulary.mappings[:n_teams]
  
  # Filter out any matches where one of the teams was not in the original training vocabulary
  valid_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), df_to_predict)
  
  if nrow(valid_df) < nrow(df_to_predict)
      println("Warning: Some matches were dropped from prediction because they contained teams not in the vocabulary.")
  end

  home_ids_to_predict = [team_map[name] for name in valid_df.home_team]
  away_ids_to_predict = [team_map[name] for name in valid_df.away_team]
  
  turing_pred_model = build_turing_model(model, n_teams, home_ids_to_predict, away_ids_to_predict)
  chains_goals = Turing.predict(turing_pred_model, chains)
  return chains_goals
end



"""
Extracts the predicted parameters (λ_h, λ_a) for each match.

Arguments:
- 'model'
- `chains`: A single result from your model (e.g., results[1][1]),
       containing parameters like `home_adv`, `log_α`, `log_β`.
- `df_to_predict`: A DataFrame (or similar) of matches to predict,
       containing `home_team`, `away_team`, and `match_id`.
- `vocabulary`: An object containing the team-to-index mappings.
"""
function extract_parameters(model::StaticPoisson, df_to_predict::AbstractDataFrame, vocabulary::Vocabulary, chains::Chains)
    
    ValueType = NamedTuple{(:λ_h, :λ_a), Tuple{AbstractVector{Float64}, AbstractVector{Float64}}}
    # 2. Allocate memory for outputs
    extraction_dict = Dict{Int64, ValueType}()

    # 3. Extract and define main parameters (constant for all matches)
    home_adv = vec(chains[Symbol("home_adv")])

    # 4. Iterate over each row in the filtered match data
    for row in eachrow(df_to_predict)
        # 5. Find the team IDs from the vocabulary
        h_id = vocabulary.mappings[:team_map][row.home_team]
        a_id = vocabulary.mappings[:team_map][row.away_team]

        # 6. Calculate the parameters for this specific match
        λ_h = exp.(vec(chains[Symbol("log_α[$h_id]")]) .+ vec(chains[Symbol("log_β[$a_id]")]) .+ home_adv)
        λ_a = exp.(vec(chains[Symbol("log_α[$a_id]")]) .+ vec(chains[Symbol("log_β[$h_id]")]));

        # 7. Store the results in the dictionary
        # The key is the match_id, the value is the NamedTuple
        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end

