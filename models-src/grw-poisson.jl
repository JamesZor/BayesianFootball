#src/models/pregame/models-src/grw-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics
# using ...TypesInterfaces: required_mapping_keys

export GRWPoisson, build_turing_model, predict, extract_parameters, extract_trends, reconstruct_vectorized


# --- 1. THE STRUCT (Fixed) ---
# Removed {D<:Distribution} to allow mixed distribution types
Base.@kwdef struct GRWPoisson <: AbstractDynamicPoissonModel 
      # --- Hyperparameters 
      σ_att::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
      σ_def::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
      
      # Fixed typo: Normal(log(1.3), 0.2) instead of Normal(log(1.3, 0.2))
      home_adv::Distribution = Normal(log(1.3), 0.2)

      # Non-Centered walk Generation ( steps 1  - first ) 
      z_att_init::Distribution = Normal(0,1)
      z_def_init::Distribution = Normal(0,1)

      # Innovations ( steps 2:T) 
      z_att_steps::Distribution = Normal(0,1)
      z_def_steps::Distribution = Normal(0,1)
end

# function required_mapping_keys(::AbstractGRWPoissonModel)
#     return [:team_map, :n_teams]
# end

# ==============================================================================
# TURING MODEL
# ==============================================================================

@model function grw_poisson_model_train(n_teams, n_rounds, 
                                      flat_home_ids, flat_away_ids, 
                                      flat_home_goals, flat_away_goals, 
                                      time_indices, model::GRWPoisson,
                                        ::Type{T} = Float64 ) where {T} 
    
    # --- 1. Hyperparameters ---
    # Standard deviations for the random walk steps
    σ_att ~ model.σ_att
    σ_def ~ model.σ_def
    home_adv ~ model.home_adv

    # --- 2. Non-Centered Random Walk Generation ---
    # A. Initial States (Round 1)
    # Sample from Standard Normal, then scale by 0.5 (initial variance)
    # shape: (n_teams,)
    z_att_init ~ filldist(model.z_att_init, n_teams)
    z_def_init ~ filldist(model.z_def_init, n_teams)

    # B. Innovations (Steps for Round 2..T)
    # We sample ALL steps at once. Shape: (n_teams, n_rounds - 1)
    z_att_steps ~ filldist(model.z_att_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_def_steps, n_teams, n_rounds - 1)

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
    
    return nothing
end

# ==============================================================================
# API IMPLEMENTATION
# ==============================================================================



function build_turing_model(model::GRWPoisson, feature_set::FeatureSet)
    data = feature_set.data
    
    # We now trust the feature set to provide these aligned vectors
    return grw_poisson_model_train(
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


"""
OPTIMIZED HELPER: unwraps NTuple directly into target shape
Avoids hcat and permutedims allocations.
"""
function unwrap_ntuple(tuple_of_arrays)
    # 1. Determine Dimensions
    # tuple_of_arrays is (AxisArray_1, AxisArray_2, ...)
    n_features = length(tuple_of_arrays)
    
    # Peek at the first element to get sample count (length of the array)
    n_samples = length(tuple_of_arrays[1])
    
    # 2. Pre-allocate the FINAL Matrix [Features, Samples]
    # We want Float64, assuming that's what comes out of Turing
    out = Matrix{Float64}(undef, n_features, n_samples)
    
    # 3. Fill directly (No temporary arrays)
    for (i, arr) in enumerate(tuple_of_arrays)
        out[i, :] .= vec(parent(arr))
    end
    
    return out
end




"""
v2 improved 

    extract_parameters(model::GRWPoisson, df_to_predict, vocabulary, chains)

Extracts parameters for prediction. 

"""

function extract_parameters(
  model::GRWPoisson,
  df_to_predict::AbstractDataFrame,
  vocabulary::Vocabulary,
  chains::Chains)
    
    # --- STEP 1: Fast Vectorized Reconstruction
    params = get(chains, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)

    Z_att_init = unwrap_ntuple(params.z_att_init)
    Z_def_init = unwrap_ntuple(params.z_def_init)
    Z_att_steps_raw = unwrap_ntuple(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple(params.z_def_steps)

    n_samples = length(σ_att_vec)
    n_teams   = vocabulary.mappings[:n_teams]
    n_steps   = div(size(Z_att_steps_raw, 1), n_teams)

    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_steps, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_steps, n_samples)

    # 1. Sum Z-steps (Vectorized)
    sum_z_att = dropdims(sum(Z_att_steps, dims=2), dims=2)
    sum_z_def = dropdims(sum(Z_def_steps, dims=2), dims=2)

    # 2. Reshape Sigmas
    σ_att_row = reshape(σ_att_vec, 1, :)
    σ_def_row = reshape(σ_def_vec, 1, :)

    # 3. Calculate Final Strengths (Vectorized)
    raw_att = (Z_att_init .* 0.5) .+ (sum_z_att .* σ_att_row)
    raw_def = (Z_def_init .* 0.5) .+ (sum_z_def .* σ_def_row)

    # 4. Center columns
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- STEP 2: Optimized Loop with Views ---
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    # Pre-allocate Dictionary size to avoid resizing overhead
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))
    
    team_map = vocabulary.mappings[:team_map]

    for row in eachrow(df_to_predict)
        h_team = row.home_team
        a_team = row.away_team

        # Fast lookup with default to avoid try/catch overhead
        h_id = get(team_map, h_team, 0)
        a_id = get(team_map, a_team, 0)

        # --- THE OPTIMIZATION ---
        # @views ensures we DO NOT copy the columns from final_att.
        # We just point to them.
        att_h = @view final_att[h_id, :]
        def_a = @view final_def[a_id, :]
        att_a = @view final_att[a_id, :]
        def_h = @view final_def[h_id, :]

        # Broadcasting results into a new vector (Payload)
        # We do this directly, skipping intermediate 'matrix' creation
        λ_h = exp.(att_h .+ def_a .+ home_adv_vec)
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


# FIX: need to add model for overloading, and sort

function reconstruct_vectorized(chain, n_teams, target_param_step=:z_att_steps, target_param_init=:z_att_init, target_sigma=:σ_att)
    
    # --- 1. PREPARE DIMENSIONS ---
    # Extract raw flattened data
    # Shape: (Samples, N_teams * N_steps) -> e.g., (1000, 350)
    Z_flat = Array(group(chain, target_param_step))
    
    n_total_samples = size(Z_flat, 1) # 1000
    n_steps = div(size(Z_flat, 2), n_teams) # 35
    
    # --- 2. CONSTRUCT TENSORS (RESHAPING) ---
    
    # A. The Steps Tensor (Δ)
    # Reshape logic: (Sample, Team, Time) -> Permute to (Team, Time, Sample)
    Z_steps = permutedims(reshape(Z_flat, n_total_samples, n_teams, n_steps), (2, 3, 1))
    
    # B. The Init Tensor (X0)
    # Shape: (Samples, N_teams) -> Permute to (Team, Sample) -> Reshape to (Team, 1, Sample)
    Z_init_raw = Array(group(chain, target_param_init))
    Z_init = reshape(permutedims(Z_init_raw, (2, 1)), n_teams, 1, n_total_samples)
    
    # C. The Sigma Vector (σ)
    # We must flatten the chains (500x2) into a single vector (1000)
    # Then reshape to (1, 1, 1000) to allow broadcasting across Teams and Time
    σ_vec = vec(Array(chain[target_sigma])) 
    σ_tensor = reshape(σ_vec, 1, 1, n_total_samples)

    # --- 3. ALGEBRAIC OPERATIONS ---

    # A. Scale the steps
    # (N, T, S) .* (1, 1, S) -> Broadcasting magic
    scaled_steps = Z_steps .* σ_tensor
    
    # B. Scale the init (Fixed 0.5 scaler from your model)
    scaled_init = Z_init .* 0.5
    
    # C. Concatenate: Join Init and Steps along Time axis (dim 2)
    # Result shape: (N, T+1, S)
    raw_walk = cat(scaled_init, scaled_steps, dims=2)
    
    # D. Integrate: Cumulative Sum along Time axis
    traj_raw = cumsum(raw_walk, dims=2)
    
    # E. Project: Center the teams (Mean along dim 1)
    # We subtract the mean of the teams from every team
    traj_centered = traj_raw .- mean(traj_raw, dims=1)
    
    return traj_centered
end



# ==============================================================================
# 5. PRETTY PRINTING (Math Notation)
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", m::GRWPoisson)
    # Title
    printstyled(io, "GRW Poisson Model", color=:cyan, bold=true)
    println(io)
    println(io, "=================")

    # 1. Mathematical Structure
    printstyled(io, "[State Space Dynamics]\n", color=:magenta)
    println(io, "  att(t) = att(t-1) + σ_att * ε_t")
    println(io, "  def(t) = def(t-1) + σ_def * ε_t")
    println(io, "  Constraint: Σ(att) = 0, Σ(def) = 0 (per step)")
    println(io)

    printstyled(io, "[Observation Model]\n", color=:magenta)
    println(io, "  y_home ~ Poisson( exp( HA + att_h + def_a ) )")
    println(io, "  y_away ~ Poisson( exp( att_a + def_h ) )")
    println(io)

    # 2. Priors
    printstyled(io, "[Priors]\n", color=:yellow)
    
    # Skip implementation details (z_steps, z_init)
    key_params = [:home_adv, :σ_att, :σ_def]
    
    for name in key_params
        val = getfield(m, name)
        println(io, "  ", rpad(string(name), 10), " ~ ", val)
    end
end
