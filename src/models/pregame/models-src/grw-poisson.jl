#src/models/pregame/models-src/grw-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics

# Using your interfaces
# using ..PreGameInterfaces
# using ..TuringHelpers
# using ...TypesInterfaces

export GRWPoisson, build_turing_model, predict, extract_parameters

"""
    GRWPoisson

A Gaussian Random Walk Poisson model. 
Team attack and defense strengths evolve over time via a random walk.
"""
struct GRWPoisson <: AbstractGRWPoissonModel end 

# ==============================================================================
# TURING MODEL DEFINITION
# ==============================================================================

@model function grw_poisson_model_train(n_teams, n_rounds, round_home_ids, round_away_ids, 
                                      round_home_goals, round_away_goals) 
    
    # --- Hyperparameters ---
    # Volatility (Standard Deviation of the drift)
    # We keep priors tight as football ratings don't fluctuate wildly week-to-week
    σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
    σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
    
    # Home Advantage (Constant over time)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- State Matrices [n_teams x n_rounds] ---
    # We define these as matrices to hold the history
    att = Matrix{Real}(undef, n_teams, n_rounds)
    def = Matrix{Real}(undef, n_teams, n_rounds)

    # --- Initial State (Round 1) ---
    att_raw_1 ~ MvNormal(zeros(n_teams), 0.5 * I)
    def_raw_1 ~ MvNormal(zeros(n_teams), 0.5 * I)
    
    # Center to ensure identifiability (sum of ratings = 0)
    att[:, 1] = att_raw_1 .- mean(att_raw_1)
    def[:, 1] = def_raw_1 .- mean(def_raw_1)

    # --- Random Walk Dynamics (Round 2..T) ---
    for t in 2:n_rounds
        # Sample step from previous state
        # We use a temporary variable for the 'raw' step before centering
        att_step ~ MvNormal(att[:, t-1], σ_att * I)
        def_step ~ MvNormal(def[:, t-1], σ_def * I)

        # Center the new state
        att[:, t] = att_step .- mean(att_step)
        def[:, t] = def_step .- mean(def_step)
    end

    # --- Likelihood (Observation) ---
    # Iterate over rounds to match the vector-of-vectors data structure
    for t in 1:n_rounds
        # Retrieve data for this specific round
        h_ids = round_home_ids[t]
        a_ids = round_away_ids[t]
        h_goals = round_home_goals[t]
        a_goals = round_away_goals[t]
        
        # Calculate rates using the parameters for time 't'
        # Broadcasting extracts the specific team parameters for the matches in this round
        log_λs = home_adv .+ att[h_ids, t] .+ def[a_ids, t]
        log_μs = att[a_ids, t] .+ def[h_ids, t]
        
        # Vectorized likelihood for this round
        h_goals ~ arraydist(LogPoisson.(log_λs))
        a_goals ~ arraydist(LogPoisson.(log_μs))
    end
    
    return nothing
end

# ==============================================================================
# API IMPLEMENTATION
# ==============================================================================

"""
    build_turing_model(model::GRWPoisson, feature_set::FeatureSet)

Builds the Turing model for the **training phase**.
"""
function build_turing_model(model::GRWPoisson, feature_set::FeatureSet)
    # We access the data dictionary directly from the FeatureSet
    # The GRW model needs the structure (Vector of Vectors) which FeatureSet provides natively
    data = feature_set.data
    
    return grw_poisson_model_train(
        data[:n_teams],
        data[:n_rounds],
        data[:round_home_ids],
        data[:round_away_ids],
        data[:round_home_goals],
        data[:round_away_goals]
    )
end

"""
    extract_parameters(model::GRWPoisson, df_to_predict, vocabulary, chains)

Extracts parameters for prediction. 
For a GRW model, this assumes we are predicting *future* matches (out-of-sample),
so it retrieves the parameters from the **last observed round** in the training data.
"""
function extract_parameters(model::GRWPoisson, df_to_predict::AbstractDataFrame, vocabulary::Vocabulary, chains::Chains)
    
    ValueType = NamedTuple{(:λ_h, :λ_a), Tuple{AbstractVector{Float64}, AbstractVector{Float64}}}
    extraction_dict = Dict{Int64, ValueType}()

    # 1. Determine the last round index 'T' from the chains
    # We search the chain names for "att[1, T]" to find the max T.
    all_keys = string.(names(chains))
    # Regex to find 'att[1, T]'
    rounds_found = Int[]
    for k in all_keys
        m = match(r"att\[1,\s*(\d+)\]", k)
        if !isnothing(m)
            push!(rounds_found, parse(Int, m.captures[1]))
        end
    end
    T_max = isempty(rounds_found) ? 1 : maximum(rounds_found)

    # 2. Extract global parameters
    home_adv = vec(chains[Symbol("home_adv")])

    # 3. Iterate over matches to predict
    for row in eachrow(df_to_predict)
        if !haskey(vocabulary.mappings[:team_map], row.home_team) || !haskey(vocabulary.mappings[:team_map], row.away_team)
            continue
        end

        h_id = vocabulary.mappings[:team_map][row.home_team]
        a_id = vocabulary.mappings[:team_map][row.away_team]

        # 4. Use parameters from the FINAL round (T_max)
        # We assume a Random Walk forecast: E[θ_{T+1}] = θ_T
        att_h = vec(chains[Symbol("att[$h_id, $T_max]")])
        att_a = vec(chains[Symbol("att[$a_id, $T_max]")])
        def_h = vec(chains[Symbol("def[$h_id, $T_max]")])
        def_a = vec(chains[Symbol("def[$a_id, $T_max]")])

        # 5. Calculate Rates
        λ_h = exp.(att_h .+ def_a .+ home_adv)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end

"""
    predict(model::GRWPoisson, df_to_predict, vocabulary, chains)

Wrapper for prediction that calculates goals based on extracted parameters.
"""
function predict(model::GRWPoisson, df_to_predict::DataFrame, vocabulary::Vocabulary, chains::Chains)
    # Since Turing.predict requires the exact model structure (including time steps),
    # and we often want to predict 'next' games not in the training set,
    # we manually generate samples using the extracted rates.
    
    params_dict = extract_parameters(model, df_to_predict, vocabulary, chains)
    
    # We'll just return the rates or sample from them. 
    # To match the StaticPoisson return type (Chains), we might need to conform.
    # However, usually for backtesting we just need the lambdas.
    # If you strictly need goal samples for the backtester:
    
    # (Implementation dependent on your specific backtesting loop requirements)
    return params_dict
end
