#src/models/pregame/models-src/grw-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics
# using ...TypesInterfaces: required_mapping_keys

export GRWPoisson, build_turing_model, predict, extract_parameters

struct GRWPoisson <: AbstractGRWPoissonModel end 

# function required_mapping_keys(::AbstractGRWPoissonModel)
#     return [:team_map, :n_teams]
# end

# ==============================================================================
# TURING MODEL
# ==============================================================================

@model function grw_poisson_model_train(n_teams, n_rounds, round_home_ids, round_away_ids, 
                                      round_home_goals, round_away_goals) 
    
    # --- Hyperparameters ---
    σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
    σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- State Containers ---
    # We use a standard matrix to build the state step-by-step
    att = Matrix{Real}(undef, n_teams, n_rounds)
    def = Matrix{Real}(undef, n_teams, n_rounds)

    # --- Initial State (Round 1) ---
    att_raw_1 ~ MvNormal(zeros(n_teams), 0.5 * I)
    def_raw_1 ~ MvNormal(zeros(n_teams), 0.5 * I)
    
    # Center and assign
    att[:, 1] = att_raw_1 .- mean(att_raw_1)
    def[:, 1] = def_raw_1 .- mean(def_raw_1)

    # --- Random Walk Dynamics (Round 2..T) ---
    for t in 2:n_rounds
        # Sample step
        att_step ~ MvNormal(att[:, t-1], σ_att * I)
        def_step ~ MvNormal(def[:, t-1], σ_def * I)

        # Center and assign
        att[:, t] = att_step .- mean(att_step)
        def[:, t] = def_step .- mean(def_step)
    end

    # --- Likelihood ---
    for t in 1:n_rounds
        h_ids = round_home_ids[t]
        a_ids = round_away_ids[t]
        
        # Calculate Rates using the current time step t
        log_λs = home_adv .+ att[h_ids, t] .+ def[a_ids, t]
        log_μs = att[a_ids, t] .+ def[h_ids, t]
        
        # Observation (Direct indexing on argument to avoid "missing" error)
        round_home_goals[t] ~ arraydist(LogPoisson.(log_λs))
        round_away_goals[t] ~ arraydist(LogPoisson.(log_μs))
    end
    
    # --- TRACKING (The User's Request) ---
    # We use := to explicitly store the full matrices in the chain.
    # This makes 'att_hist' and 'def_hist' appear in the results.
    att_hist := att
    def_hist := def

    return nothing
end

# ==============================================================================
# API IMPLEMENTATION
# ==============================================================================

function build_turing_model(model::GRWPoisson, feature_set::FeatureSet)
    data = feature_set.data
    return grw_poisson_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:round_home_ids],
        data[:round_away_ids],
        collect.(data[:round_home_goals]), 
        collect.(data[:round_away_goals])
    )
end

"""
    extract_parameters(model::GRWPoisson, df_to_predict, vocabulary, chains)

Extracts parameters for prediction. 
Uses the 'att_hist' and 'def_hist' variables we explicitly tracked with :=
"""
function extract_parameters(model::GRWPoisson, df_to_predict::AbstractDataFrame, vocabulary::Vocabulary, chains::Chains)
    
    ValueType = NamedTuple{(:λ_h, :λ_a), Tuple{AbstractVector{Float64}, AbstractVector{Float64}}}
    extraction_dict = Dict{Int64, ValueType}()

    # 1. Determine the last available round 'T' in the chains
    # We look for keys like "att_hist[1, T]"
    all_keys = string.(names(chains))
    
    # Regex to capture T from "att_hist[team_id, T]"
    # Note: Turing formats matrix keys as "var[row,col]"
    # We find the maximum 'col' index.
    max_round = 0
    for k in all_keys
        m = match(r"att_hist\[\d+,\s*(\d+)\]", k)
        if !isnothing(m)
            r = parse(Int, m.captures[1])
            if r > max_round
                max_round = r
            end
        end
    end
    
    # Fallback if regex fails (e.g. if n_rounds=1, sometimes it formats differently or we just assume 1)
    T_final = max_round > 0 ? max_round : 1
    # println("Debug: Extracting GRW parameters for T=$T_final")

    # 2. Extract Home Advantage
    home_adv = vec(chains[Symbol("home_adv")])

    # 3. Predict for each match
    for row in eachrow(df_to_predict)
        if !haskey(vocabulary.mappings[:team_map], row.home_team) || !haskey(vocabulary.mappings[:team_map], row.away_team)
            continue
        end

        h_id = vocabulary.mappings[:team_map][row.home_team]
        a_id = vocabulary.mappings[:team_map][row.away_team]

        # 4. Retrieve parameters for the LAST time step (T_final)
        # Construct the exact symbol name for the chain
        sym_att_h = Symbol("att_hist[$h_id, $T_final]")
        sym_att_a = Symbol("att_hist[$a_id, $T_final]")
        sym_def_h = Symbol("def_hist[$h_id, $T_final]")
        sym_def_a = Symbol("def_hist[$a_id, $T_final]")

        att_h = vec(chains[sym_att_h])
        att_a = vec(chains[sym_att_a])
        def_h = vec(chains[sym_def_h])
        def_a = vec(chains[sym_def_a])

        # 5. Calculate Rates
        λ_h = exp.(att_h .+ def_a .+ home_adv)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end

function predict(model::GRWPoisson, df_to_predict::DataFrame, vocabulary::Vocabulary, chains::Chains)
    return extract_parameters(model, df_to_predict, vocabulary, chains)
end
