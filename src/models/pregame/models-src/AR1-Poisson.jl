# src/models/pregame/models-src/AR1-Poisson.jl

using DataFrames
using Turing
using LinearAlgebra
using Statistics

# We assume unwrap_ntuple is needed. If it's not in a shared helper, we define it here.
# Ideally, move this to turing_helpers.jl to share between GRW and AR1.
function unwrap_ntuple_ar1(tuple_of_arrays)
    n_features = length(tuple_of_arrays)
    n_samples = length(tuple_of_arrays[1])
    out = Matrix{Float64}(undef, n_features, n_samples)
    for (i, arr) in enumerate(tuple_of_arrays)
        out[i, :] .= vec(parent(arr))
    end
    return out
end

export AR1Poisson, build_turing_model, extract_parameters, reconstruct_ar1_path

# ==============================================================================
# 1. THE STRUCT
# ==============================================================================

Base.@kwdef struct AR1Poisson <: AbstractDynamicPoissonModel 
      # --- Hyperparameters ---
      # Volatility (Standard Deviation of the innovation)
      σ_att::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
      σ_def::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
      
      # Persistence (Autoregressive parameter)
      # Beta(10, 2) biases towards 1.0 (high memory/slow evolution), roughly 0.83 mean
      ρ_att::Distribution = Beta(10, 2)
      ρ_def::Distribution = Beta(10, 2)

      home_adv::Distribution = Normal(log(1.3), 0.2)

      # --- Non-Centered Parameterization ---
      # Initial States (Round 1)
      z_att_init::Distribution = Normal(0,1)
      z_def_init::Distribution = Normal(0,1)

      # Innovations (Steps for Round 2:T)
      z_att_steps::Distribution = Normal(0,1)
      z_def_steps::Distribution = Normal(0,1)
end

# ==============================================================================
# 2. TURING MODEL
# ==============================================================================

@model function ar1_poisson_model_train(n_teams, n_rounds, 
                                        flat_home_ids, flat_away_ids, 
                                        flat_home_goals, flat_away_goals, 
                                        time_indices, model::AR1Poisson)

    # --- A. Hyperparameters ---
    σ_att ~ model.σ_att
    σ_def ~ model.σ_def
    
    ρ_att ~ model.ρ_att
    ρ_def ~ model.ρ_def
    
    home_adv ~ model.home_adv

    # --- B. Non-Centered Innovations ---
    
    # 1. Initial States (t=1)
    z_att_init ~ filldist(model.z_att_init, n_teams)
    z_def_init ~ filldist(model.z_def_init, n_teams)

    # 2. Innovations (t=2..T)
    # Shape: (n_teams, n_rounds - 1)
    z_att_steps ~ filldist(model.z_att_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_def_steps, n_teams, n_rounds - 1)

    # --- C. AR1 Reconstruction (The Loop) ---
    # We must iterate because t depends on t-1.
    
    # Initialize strength matrices [Teams x Rounds]
    # We use a Vector of Vectors logic or pre-allocated Matrix for ReverseDiff efficiency
    # Here we build up the raw path.
    
    # Initial Variance: For a stationary AR1, var = σ² / (1-ρ²).
    # However, usually in sports models we just pick an initial sigma or start at 0.
    # We will use the z_init scaled by σ_att to keep it simple and stable.
    
    # We need to handle types for AD (ForwardDiff/ReverseDiff)
    T = typeof(σ_att) # Capture the Dual type if present
    
    # Pre-allocate containers (using Vector of Vectors is often AD-safer for loops)
    att_seq = Vector{Vector{T}}(undef, n_rounds)
    def_seq = Vector{Vector{T}}(undef, n_rounds)

    # t = 1
    att_seq[1] = z_att_init .* σ_att
    def_seq[1] = z_def_init .* σ_def

    # t = 2...T
    for t in 2:n_rounds
        # AR1: X_t = ρ * X_{t-1} + σ * ε_t
        # z_att_steps[:, t-1] accesses the innovation for this step
        att_seq[t] = (att_seq[t-1] .* ρ_att) .+ (z_att_steps[:, t-1] .* σ_att)
        def_seq[t] = (def_seq[t-1] .* ρ_def) .+ (z_def_steps[:, t-1] .* σ_def)
    end

    # Stack into Matrix: (n_teams, n_rounds)
    att_raw = reduce(hcat, att_seq)
    def_raw = reduce(hcat, def_seq)

    # --- D. Zero-Sum Constraint (Centering) ---
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- E. Likelihood (Vectorized) ---
    # Use views to grab the specific strength for each match
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Log-Rates
    log_λs = home_adv .+ att_h_flat .+ def_a_flat
    log_μs = att_a_flat .+ def_h_flat

    # Observe
    flat_home_goals ~ arraydist(LogPoisson.(log_λs))
    flat_away_goals ~ arraydist(LogPoisson.(log_μs))

    return nothing
end

# ==============================================================================
# 3. API IMPLEMENTATION (Build)
# ============================================================================== 

function build_turing_model(model::AR1Poisson, feature_set::FeatureSet)
    data = feature_set.data
    
    # We now trust the feature set to provide these aligned vectors
    return ar1_poisson_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:flat_home_ids],     # Pre-flattened
        data[:flat_away_ids],     # Pre-flattened
        data[:flat_home_goals],   # Pre-flattened
        data[:flat_away_goals],   # Pre-flattened
        data[:time_indices],      # Created in FeatureSet
        model
    )
end

# function build_turing_model(model::AR1Poisson, feature_set::FeatureSet)
#     data = feature_set.data
#
#     # Flatten Logic (Same as GRW)
#     flat_home_ids = vcat(data[:round_home_ids]...)
#     flat_away_ids = vcat(data[:round_away_ids]...)
#     flat_home_goals = vcat(collect.(data[:round_home_goals])...)
#     flat_away_goals = vcat(collect.(data[:round_away_goals])...)
#
#     # Time Indices
#     time_indices = Int[]
#     for (t, round_matches) in enumerate(data[:round_home_ids])
#         append!(time_indices, fill(t, length(round_matches)))
#     end
#
#     return ar1_poisson_model_train(
#         data[:n_teams]::Int,
#         data[:n_rounds]::Int,
#         flat_home_ids,
#         flat_away_ids,
#         flat_home_goals,
#         flat_away_goals,
#         time_indices,
#         model
#     )
# end
#
# ==============================================================================
# 4. EXTRACTION (Inner & Wrapper)
# ==============================================================================

"""
    reconstruct_ar1_path(...)

Helper to reconstruct the AR1 path outside the model using Chains.
"""
function reconstruct_ar1_path(Z_init, Z_steps, σ_vec, ρ_vec)
    # Dimensions
    n_teams, n_steps, n_samples = size(Z_steps)
    # Z_steps is actually n_rounds-1 long in the time dimension
    n_rounds = n_steps + 1 
    
    # Output container: (Teams, Rounds, Samples)
    path = zeros(Float64, n_teams, n_rounds, n_samples)
    
    # Reshape vectors for broadcasting: (1, 1, Samples)
    σ_b = reshape(σ_vec, 1, 1, n_samples)
    ρ_b = reshape(ρ_vec, 1, 1, n_samples)
    
    # t=1
    path[:, 1, :] .= Z_init .* σ_b # Scale init
    
    # t=2..T
    for t in 2:n_rounds
        prev = path[:, t-1, :]
        innov = Z_steps[:, t-1, :]
        
        # AR1 Update: ρ * prev + σ * innov
        path[:, t, :] .= (prev .* ρ_b) .+ (innov .* σ_b)
    end
    
    return path
end


function extract_parameters(
  model::AR1Poisson,
  df_to_predict::AbstractDataFrame,
  vocabulary::Vocabulary,
  chains::Chains)

    # --- STEP 1: Fast Parameter Retrieval ---
    params = get(chains, [:home_adv, :σ_att, :σ_def, :ρ_att, :ρ_def, 
                          :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    # Vectors (Samples,)
    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)
    ρ_att_vec    = vec(params.ρ_att)
    ρ_def_vec    = vec(params.ρ_def)

    # Arrays (Features x Samples) -> Unwrap
    Z_att_init_raw = unwrap_ntuple_ar1(params.z_att_init)
    Z_def_init_raw = unwrap_ntuple_ar1(params.z_def_init)
    Z_att_steps_raw = unwrap_ntuple_ar1(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple_ar1(params.z_def_steps)

    # Reshape for reconstruction
    n_teams = vocabulary.mappings[:n_teams]
    n_samples = length(home_adv_vec)
    
    # Reshape Init: (Teams, Samples)
    Z_att_init = Z_att_init_raw # Already (Features, Samples) where Features=Teams
    Z_def_init = Z_def_init_raw 

    # Reshape Steps: (Teams, Steps, Samples)
    n_innovations = div(size(Z_att_steps_raw, 1), n_teams) # Should be n_rounds - 1
    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_innovations, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_innovations, n_samples)

    # --- STEP 2: Reconstruct Paths ---
    # Need to reshape Init to (Teams, 1, Samples) for the helper if we wanted to be generic,
    # but let's just pass matching shapes.
    
    # We expand Init to (Teams, Samples) -> (Teams, Samples) used inside helper
    raw_att = reconstruct_ar1_path(Z_att_init, Z_att_steps, σ_att_vec, ρ_att_vec)
    raw_def = reconstruct_ar1_path(Z_def_init, Z_def_steps, σ_def_vec, ρ_def_vec)

    # Center: Subtract mean across teams (dim 1)
    # mean(raw_att, dims=1) results in (1, Rounds, Samples)
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- STEP 3: Prediction Loop ---
    # We use the LAST time step for prediction (assuming prediction is for the *next* round? 
    # Or based on the specific match round in df_to_predict?)
    
    # NOTE: Usually `extract_parameters` for backtesting predicts the match *at its specific time*.
    # However, if df_to_predict contains matches from the FUTURE relative to training, 
    # we usually project the AR1 forward or take the last known state.
    # For now, let's assume we use the latent strength at the **End of the Chain** (Last Round trained)
    # for all predictions, OR we match the round ID if available.
    
    # GRWPoisson implementation usually assumes we are extracting the *final* strengths 
    # or the specific strengths if the match is in-sample. 
    # Looking at your GRW code: it calculates `final_att` which seems to be the SUM of all steps.
    # This implies it uses the strength at time T (end of training) for the prediction.
    
    # So we slice the last time step:
    att_final_T = final_att[:, end, :] # (Teams, Samples)
    def_final_T = final_def[:, end, :] # (Teams, Samples)
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))
    
    team_map = vocabulary.mappings[:team_map]

    for row in eachrow(df_to_predict)
        h_team = row.home_team
        a_team = row.away_team
        h_id = get(team_map, h_team, 0)
        a_id = get(team_map, a_team, 0)

        # Skip if team not found (or handle error)
        if h_id == 0 || a_id == 0 
            continue 
        end

        # Use views for speed
        att_h = @view att_final_T[h_id, :]
        def_a = @view def_final_T[a_id, :]
        att_a = @view att_final_T[a_id, :]
        def_h = @view def_final_T[h_id, :]

        λ_h = exp.(att_h .+ def_a .+ home_adv_vec)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end


"""
OVERLOADED METHOD: Wrapper
Iterates through results and prediction dataframes, calling the inner extraction logic for each fold.
"""
function extract_parameters(
    model::AR1Poisson,
    dfs_to_predict::AbstractVector, 
    vocabulary::Vocabulary,
    results_vector::AbstractVector
)
    # 1. Define Types
    PredictionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    DictType = Dict{Int64, PredictionValue}
    
    n_splits = length(dfs_to_predict)
    partial_results = Vector{DictType}(undef, n_splits)

    # 2. Threaded Extraction
    Threads.@threads for i in 1:n_splits
        if i <= length(results_vector)
            result_tuple = results_vector[i]
            df_curr = dfs_to_predict[i]
            chains = result_tuple[1]
            
            partial_results[i] = extract_parameters(model, df_curr, vocabulary, chains)
        end
    end

    # 3. Merge Results
    total_rows = sum(nrow, dfs_to_predict)
    full_extraction_dict = DictType()
    sizehint!(full_extraction_dict, total_rows)

    for i in 1:n_splits
        if isassigned(partial_results, i)
            merge!(full_extraction_dict, partial_results[i])
        end
    end

    return full_extraction_dict
end






"""
old notes 

"""
# using DataFrames
# using Turing
# using LinearAlgebra
# using Statistics
#
# export AR1Poisson, build_turing_model, extract_parameters
#
#
# struct AR1Poisson <: AbstractDynamicPoissonModel end 
#
#
#
# # ==============================================================================
# # turing model
# # ==============================================================================
#
#
# @model function ar1_poisson_model_train(n_teams, n_rounds, 
#                                         flat_home_ids, flat_away_ids, 
#                                         flat_home_goals, flat_away_goals, 
#                                         time_indices)
#
#
#
# end 
#
#
#
#
# function build_turing_model(model::AR1Poisson, feature_set::FeatureSet)
#   println("this is a place holder for the AR1Model build_turing_model")
#
# end 
#
#
# function extract_parameters(
#   model::AR1Poisson,
#   df_to_predict::AbstractDataFrame,
#   vocabulary::Vocabulary,
#   chains::Chains)
#
#   println("this is a place holder for the AR1Model extract_parameters level 1 ")
#
# end
#
#
# """
# OVERLOADED METHOD: Wrapper
# Iterates through results and prediction dataframes, calling the inner extraction logic for each fold.
# """
# function extract_parameters(
#     model::AR1Poisson,
#     dfs_to_predict::AbstractVector, 
#     vocabulary::Vocabulary,
#     results_vector::AbstractVector
# )
#
#   println("this is a place holder for the AR1Model extract_parameters level 2 - the wrapper. ")
#
# end
#
#
#
# # # src/models/pregame/models-src/ar1-poisson.jl
# #
# # using DataFrames, Turing, LinearAlgebra, Statistics
# #
# # export AR1Poisson, build_turing_model
# #
# # struct AR1Poisson <: AbstractDynamicPoissonModel end 
# #
# # @model function ar1_poisson_model_train(n_teams, n_rounds, 
# #                                         flat_home_ids, flat_away_ids, 
# #                                         flat_home_goals, flat_away_goals, 
# #                                         time_indices)
# #
# #     # --- 1. Hyperparameters ---
# #     ρ_att ~ Beta(10, 2) 
# #     ρ_def ~ Beta(10, 2)
# #     σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
# #     σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
# #     home_adv ~ Normal(log(1.3), 0.2)
# #
# #     # --- 2. Random Walk Innovation ---
# #     # T=1
# #     z_att_init ~ filldist(Normal(0, 1), n_teams)
# #     z_def_init ~ filldist(Normal(0, 1), n_teams)
# #     # T=2..N
# #     z_att_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
# #     z_def_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
# #
# #     # --- 3. The Loop (Optimized for Type Stability) ---
# #
# #     # TRICK: We need to tell Julia that 'att_seq' will contain the same type of numbers
# #     # as 'z_att_init' (which might be Float64 or ReverseDiff.TrackedReal).
# #     # We use `eltype` to detect this type dynamically.
# #
# #     T = eltype(z_att_init) 
# #
# #     # Initialize vector of vectors with explicit type T
# #     # (Using Vector of Vectors is friendly to ReverseDiff)
# #     att_seq = Vector{Vector{T}}(undef, n_rounds)
# #     def_seq = Vector{Vector{T}}(undef, n_rounds)
# #
# #     # Initialize t=1
# #     att_seq[1] = z_att_init .* σ_att
# #     def_seq[1] = z_def_init .* σ_def
# #
# #     # Iterate t=2..Round
# #     for t in 2:n_rounds
# #         # Standard AR(1) update
# #         att_seq[t] = (att_seq[t-1] .* ρ_att) .+ (z_att_steps[:, t-1] .* σ_att)
# #         def_seq[t] = (def_seq[t-1] .* ρ_def) .+ (z_def_steps[:, t-1] .* σ_def)
# #     end
# #
# #     # Reduce to Matrix (Efficient Stack)
# #     att_raw = reduce(hcat, att_seq)
# #     def_raw = reduce(hcat, def_seq)
# #
# #     # Center (Identifiability)
# #     att = att_raw .- mean(att_raw, dims=1)
# #     def = def_raw .- mean(def_raw, dims=1)
# #
# #     # Track for extraction
# #     att_hist := att
# #     def_hist := def
# #     rho_att_track := ρ_att
# #
# #     # --- 4. Likelihood (V1 Cartesian Approach) ---
# #     att_h = view(att, CartesianIndex.(flat_home_ids, time_indices))
# #     def_a = view(def, CartesianIndex.(flat_away_ids, time_indices))
# #     att_a = view(att, CartesianIndex.(flat_away_ids, time_indices))
# #     def_h = view(def, CartesianIndex.(flat_home_ids, time_indices))
# #
# #     log_λs = home_adv .+ att_h .+ def_a
# #     log_μs = att_a .+ def_h
# #
# #     flat_home_goals ~ arraydist(LogPoisson.(log_λs))
# #     flat_away_goals ~ arraydist(LogPoisson.(log_μs))
# # end
# #
# # # ... (Include your build_turing_model function here) ...
