#src/models/pregame/models-src/grw-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics
# using ...TypesInterfaces: required_mapping_keys

export GRWPoisson, build_turing_model, predict, extract_parameters, extract_trends

struct GRWPoisson <: AbstractDynamicPoissonModel end 

# function required_mapping_keys(::AbstractGRWPoissonModel)
#     return [:team_map, :n_teams]
# end

# ==============================================================================
# TURING MODEL
# ==============================================================================
@model function grw_poisson_model_train(n_teams, n_rounds, 
                                      flat_home_ids, flat_away_ids, 
                                      flat_home_goals, flat_away_goals, 
                                      time_indices) 
    
    # --- 1. Hyperparameters ---
    # Standard deviations for the random walk steps
    σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
    σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- 2. Non-Centered Random Walk Generation ---
    # A. Initial States (Round 1)
    # Sample from Standard Normal, then scale by 0.5 (initial variance)
    # shape: (n_teams,)
    z_att_init ~ filldist(Normal(0, 1), n_teams)
    z_def_init ~ filldist(Normal(0, 1), n_teams)

    # B. Innovations (Steps for Round 2..T)
    # We sample ALL steps at once. Shape: (n_teams, n_rounds - 1)
    z_att_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
    z_def_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)

    # C. Deterministic Reconstruction (The "Matt Trick")
    # We construct the trajectory using cumulative sums.
    # We concatenate the Initial State with the scaled steps.
    
    # Scale the standard normals by the actual sigmas
    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def
    
    # Reconstruct the raw random walk
    # hcat connects the initial state (col 1) with the steps (cols 2..T)
    # cumsum adds them up along the time dimension (dims=2)
    att_raw = cumsum(hcat(z_att_init .* 0.5, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(z_def_init .* 0.5, scaled_steps_def), dims=2)

    # --- 3. Zero-Sum Constraint (Centering) ---
    # Enforce mean(att) = 0 and mean(def) = 0 at every time step t
    # We subtract the column means from every row
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- 4. Likelihood (Fully Vectorized) ---
    # We use the time_indices to pull the correct strength for every match
    # flat_home_ids and time_indices are vectors of length n_matches
    
    # Note on indexing: ReverseDiff loves simple indexing.
    # We extract the specific team/time strength for every single match.
    
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Calculate Log-Rates
    log_λs = home_adv .+ att_h_flat .+ def_a_flat
    log_μs = att_a_flat .+ def_h_flat

    # Observe
    flat_home_goals ~ arraydist(LogPoisson.(log_λs))
    flat_away_goals ~ arraydist(LogPoisson.(log_μs))
    
    # --- 5. Tracking ---
    # We track the centered parameters for post-processing
    att_hist := att
    def_hist := def

    return nothing
end



# @model function grw_poisson_model_train(n_teams, n_rounds, 
#                                       flat_home_ids, flat_away_ids, 
#                                       flat_home_goals, flat_away_goals, 
#                                       time_indices) # <--- NEW ARGUMENT
#
#     # --- Hyperparameters ---
#     σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
#     σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
#     home_adv ~ Normal(log(1.3), 0.2)
#
#     # --- State Containers ---
#     att_raw = Matrix{Real}(undef, n_teams, n_rounds)
#     def_raw = Matrix{Real}(undef, n_teams, n_rounds)
#     att = Matrix{Real}(undef, n_teams, n_rounds)
#     def = Matrix{Real}(undef, n_teams, n_rounds)
#
#     # --- Initial State (Round 1) ---
#     att_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
#     def_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
#     att[:, 1] = att_raw[:, 1] .- mean(att_raw[:, 1])
#     def[:, 1] = def_raw[:, 1] .- mean(def_raw[:, 1])
#
#     # --- Random Walk Dynamics (Round 2..T) ---
#     for t in 2:n_rounds
#         att_raw[:, t] ~ MvNormal(att[:, t-1], σ_att * I)
#         def_raw[:, t] ~ MvNormal(def[:, t-1], σ_def * I)
#         att[:, t] = att_raw[:, t] .- mean(att_raw[:, t])
#         def[:, t] = def_raw[:, t] .- mean(def_raw[:, t])
#     end
#
#     # --- LIKELIHOOD (Vectorized) ---
#     # We no longer loop over rounds. We calculate everything in one massive batch.
#     # This is much friendlier to the AD engine.
#
#     # 1. Gather parameters for every match based on time_indices
#     # We use view/indexing to pull the correct parameter for the correct time t
#
#     # Note: Turing might struggle with CartesianIndex broadcasting for some backends.
#     # A manual comprehension or loop for rate calculation is often safer and still fast.
#
#     # Constructing the rates:
#     # log_λ[i] = home_adv + att[home_id[i], time[i]] + def[away_id[i], time[i]]
#
#     # We can do this efficiently:
#     n_matches = length(flat_home_goals)
#     log_λs = Vector{Real}(undef, n_matches)
#     log_μs = Vector{Real}(undef, n_matches)
#
#     for i in 1:n_matches
#         t = time_indices[i]
#         h = flat_home_ids[i]
#         a = flat_away_ids[i]
#
#         log_λs[i] = home_adv + att[h, t] + def[a, t]
#         log_μs[i] = att[a, t] + def[h, t]
#     end
#
#     # 2. Observe
#     flat_home_goals ~ arraydist(LogPoisson.(log_λs))
#     flat_away_goals ~ arraydist(LogPoisson.(log_μs))
#
#     # --- TRACKING ---
#     att_hist := att
#     def_hist := def
#
#     return nothing
# end
#
# @model function grw_poisson_model_train(n_teams, n_rounds, round_home_ids, round_away_ids, 
#                                       round_home_goals, round_away_goals) 
#
#     # --- Hyperparameters ---
#     σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
#     σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
#     home_adv ~ Normal(log(1.3), 0.2)
#
#     # --- State Containers ---
#     # att_raw/def_raw: The unconstrained random variables sampled by Turing
#     att_raw = Matrix{Real}(undef, n_teams, n_rounds)
#     def_raw = Matrix{Real}(undef, n_teams, n_rounds)
#
#     # att/def: The centered states used for calculation (Identifiability constrained)
#     att = Matrix{Real}(undef, n_teams, n_rounds)
#     def = Matrix{Real}(undef, n_teams, n_rounds)
#
#     # --- Initial State (Round 1) ---
#     # We sample into the first column of the raw matrix
#     att_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
#     def_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
#
#     # Center
#     att[:, 1] = att_raw[:, 1] .- mean(att_raw[:, 1])
#     def[:, 1] = def_raw[:, 1] .- mean(def_raw[:, 1])
#
#     # --- Random Walk Dynamics (Round 2..T) ---
#     for t in 2:n_rounds
#         # FIX: We index the LHS (att_raw[:, t]) so each step has a unique variable name
#         att_raw[:, t] ~ MvNormal(att[:, t-1], σ_att * I)
#         def_raw[:, t] ~ MvNormal(def[:, t-1], σ_def * I)
#
#         # Center
#         att[:, t] = att_raw[:, t] .- mean(att_raw[:, t])
#         def[:, t] = def_raw[:, t] .- mean(def_raw[:, t])
#     end
#
#     # --- Likelihood ---
#     for t in 1:n_rounds
#         h_ids = round_home_ids[t]
#         a_ids = round_away_ids[t]
#
#         # Calculate Rates using the current time step t
#         log_λs = home_adv .+ att[h_ids, t] .+ def[a_ids, t]
#         log_μs = att[a_ids, t] .+ def[h_ids, t]
#
#         # Observation
#         round_home_goals[t] ~ arraydist(LogPoisson.(log_λs))
#         round_away_goals[t] ~ arraydist(LogPoisson.(log_μs))
#     end
#
#     # --- TRACKING ---
#     # Save the full history matrices to the chain
#     att_hist := att
#     def_hist := def
#
#     return nothing
# end
#
# ==============================================================================
# API IMPLEMENTATION
# ==============================================================================
#
# function build_turing_model(model::GRWPoisson, feature_set::FeatureSet)
#     data = feature_set.data
#     return grw_poisson_model_train(
#         data[:n_teams]::Int,
#         data[:n_rounds]::Int,
#         data[:round_home_ids],
#         data[:round_away_ids],
#         collect.(data[:round_home_goals]), 
#         collect.(data[:round_away_goals])
#     )
# end
#

function build_turing_model(model::GRWPoisson, feature_set::FeatureSet)
    data = feature_set.data
    
    # 1. Flatten the IDs and Goals
    # We can rely on the 'flat_*' keys if they exist and are sorted correctly.
    # However, to be safe and ensure alignment with the round structure, we re-flatten here.
    
    flat_home_ids = vcat(data[:round_home_ids]...)
    flat_away_ids = vcat(data[:round_away_ids]...)
    flat_home_goals = vcat(collect.(data[:round_home_goals])...) # Collect + Flatten
    flat_away_goals = vcat(collect.(data[:round_away_goals])...)
    
    # 2. Create the Time Index Vector
    # If Round 1 has 10 games, we push '1' ten times.
    # If Round 2 has 8 games, we push '2' eight times.
    time_indices = Int[]
    for (t, round_matches) in enumerate(data[:round_home_ids])
        n_matches_in_round = length(round_matches)
        append!(time_indices, fill(t, n_matches_in_round))
    end

    return grw_poisson_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        flat_home_ids,
        flat_away_ids,
        flat_home_goals,
        flat_away_goals,
        time_indices  # <--- NEW INPUT
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

"""
    extract_trends(model::GRWPoisson, vocabulary::Vocabulary, chains::Chains)

Extracts the evolution of Attack and Defense strengths over time for all teams.
Returns a DataFrame with columns: `[:team, :round, :att, :def]`.
"""
function extract_trends(model::GRWPoisson, vocabulary::Vocabulary, chains::Chains)
    n_teams = vocabulary.mappings[:n_teams]
    team_map = vocabulary.mappings[:team_map]
    
    # Create a reverse map (Int -> String) to get names back
    id_to_team = Dict(v => k for (k, v) in team_map)
    
    # 1. Detect number of rounds from the chain keys
    # We look for the highest index T in "att_hist[1, T]"
    all_keys = string.(names(chains))
    max_round = 0
    for k in all_keys
        m = match(r"att_hist\[\d+,\s*(\d+)\]", k)
        if !isnothing(m)
            r = parse(Int, m.captures[1])
            max_round = max(max_round, r)
        end
    end
    
    if max_round == 0
        @warn "No 'att_hist' variables found. Did you add 'att_hist := att' to your model?"
        return DataFrame()
    end

    # 2. Pre-allocate vectors for the DataFrame
    # (n_teams * n_rounds) rows
    total_rows = n_teams * max_round
    teams_col = Vector{String}(undef, total_rows)
    rounds_col = Vector{Int}(undef, total_rows)
    att_col   = Vector{Float64}(undef, total_rows)
    def_col   = Vector{Float64}(undef, total_rows)
    
    idx = 1
    # 3. Iterate and Extract
    for t in 1:max_round
        for i in 1:n_teams
            # Construct symbol names, e.g., :att_hist[1, 1]
            sym_att = Symbol("att_hist[$i, $t]")
            sym_def = Symbol("def_hist[$i, $t]")
            
            # Extract the mean of the posterior samples
            # You could also grab quantiles here if you want error bars
            att_val = mean(vec(chains[sym_att]))
            def_val = mean(vec(chains[sym_def]))
            
            teams_col[idx] = id_to_team[i]
            rounds_col[idx] = t
            att_col[idx]   = att_val
            def_col[idx]   = def_val
            
            idx += 1
        end
    end
    
    return DataFrame(
        team = teams_col,
        round = rounds_col,
        att = att_col,
        def = def_col
    )
end
