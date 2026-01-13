#src/models/pregame/mode/grw-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics


export GRWPoisson

Base.@kwdef struct GRWPoisson <: AbstractDynamicPoissonModel 
      γ::Distribution = Normal(log(1.3), 0.2) # home advantage
  
      # --- Dynamic Hyperparameters (Process Noise)
      σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf) 
      
      # --- Initial State Hyperparameters (Hierarchical Prior t=0) [NEW]
      # Prior on the MEAN of the abilities (The "Baseline")
      # Typically small negative numbers (log(1.0) = 0, but usually goal rates are < 1.0 per team)
      μ_s::Distribution = Normal(0, 0.5)  # att, def 

      # Prior on the SPREAD of abilities at t=0
      # Replacing the hardcoded 0.5
      σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

      # --- Latent Variables (Unchanged)
      z_init::Distribution = Normal(0,1)
      z_steps::Distribution = Normal(0,1)
end


@model function grw_poisson_model_train(n_teams, n_rounds, 
                                      flat_home_ids, flat_away_ids, 
                                      flat_home_goals, flat_away_goals, 
                                      time_indices, model::GRWPoisson,
                                      ::Type{T} = Float64 ) where {T} 
    
    # --- 1. Hyperparameters ---
    # Random Walk Innovations
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k
    γ     ~ model.γ
    
    # [NEW] Hierarchical Priors for t=0
    μ_att ~ model.μ_s
    μ_def ~ model.μ_s
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # --- 2. Latent Variables ---
    z_att_init ~ filldist(model.z_init, n_teams)
    z_def_init ~ filldist(model.z_init, n_teams)

    # Sample steps (n_rounds - 1)
    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # --- 3. Trajectory Reconstruction (NCP) ---
    
    # Scale the steps (Innovation Variance)
    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def
    
    # [NEW] Scale the Initial State (Initial Variance)
    # Note: We do NOT add μ here yet, because cumsum would propagate it as a drift (slope).
    # We construct the "shape" of the walk centered at 0 first.
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0

    # Integrate (Random Walk)
    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # --- 4. Centering & Baseline Shift ---
    
    # A. Enforce Zero-Sum on the 'Shape' to prevent drift identifiability issues
    # B. Apply the Global Hierarchical Mean
    # This sets the "Intercept" of the model.
    # att(t) now has mean μ_att, def(t) has mean μ_def
    att = (att_raw .- mean(att_raw, dims=1)) .+ μ_att
    def = (def_raw .- mean(def_raw, dims=1)) .+ μ_def

    # --- 5. Likelihood ---
    # Extract specific match strengths
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Calculate Log-Rates
    # Total Intercept = home_adv + μ_att + μ_def
    log_λₕ =  att_h_flat .+ def_a_flat .+ γ

    # Total Intercept = μ_att + μ_def
    log_λₐ = att_a_flat .+ def_h_flat 

    # Observe
    flat_home_goals ~ arraydist(LogPoisson.(log_λₕ))
    flat_away_goals ~ arraydist(LogPoisson.(log_λₐ))
    
    return nothing
end


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

# ==============================================================================
# 1. HELPER: Reconstruct States (Unchanged logic, just needed for extraction)
# ==============================================================================
"""
    reconstruct_states(chains, n_teams, n_rounds)

Internal helper to reconstruct 'att' and 'def' matrices (Teams x Rounds x Samples)
from the MCMC chains using the hierarchical logic (NCP).
"""
function reconstruct_states(chain::Chains, n_teams::Int, n_rounds::Int)
    # 1. Extract Scalars (Samples)
    μ_att_vec   = vec(chain[:μ_att])
    μ_def_vec   = vec(chain[:μ_def])
    σ_att_vec   = vec(chain[:σ_att])
    σ_def_vec   = vec(chain[:σ_def])
    σ_att_0_vec = vec(chain[:σ_att_0])
    σ_def_0_vec = vec(chain[:σ_def_0])

    # 2. Extract Arrays (Samples x Dimensions)
    z_att_init_raw = Array(group(chain, :z_att_init))
    z_def_init_raw = Array(group(chain, :z_def_init))
    
    n_samples = size(z_att_init_raw, 1)

    # Reshape Init: [Team, Time=1, Sample]
    Z_att_init = permutedims(reshape(z_att_init_raw, n_samples, n_teams, 1), (2, 3, 1))
    Z_def_init = permutedims(reshape(z_def_init_raw, n_samples, n_teams, 1), (2, 3, 1))

    # Reshape Steps: [Team, Time=Steps, Sample]
    z_att_steps_raw = Array(group(chain, :z_att_steps))
    z_def_steps_raw = Array(group(chain, :z_def_steps))
    
    Z_att_steps = permutedims(reshape(z_att_steps_raw, n_samples, n_teams, n_rounds-1), (2, 3, 1))
    Z_def_steps = permutedims(reshape(z_def_steps_raw, n_samples, n_teams, n_rounds-1), (2, 3, 1))

    # 3. Reshape Scalars for Broadcasting
    S_att   = reshape(σ_att_vec, 1, 1, n_samples)
    S_def   = reshape(σ_def_vec, 1, 1, n_samples)
    S_att_0 = reshape(σ_att_0_vec, 1, 1, n_samples)
    S_def_0 = reshape(σ_def_0_vec, 1, 1, n_samples)
    
    M_att   = reshape(μ_att_vec, 1, 1, n_samples)
    M_def   = reshape(μ_def_vec, 1, 1, n_samples)

    # 4. Reconstruction
    scaled_init_att = Z_att_init .* S_att_0
    scaled_init_def = Z_def_init .* S_def_0
    
    scaled_steps_att = Z_att_steps .* S_att
    scaled_steps_def = Z_def_steps .* S_def

    # Integrate
    raw_att = cumsum(cat(scaled_init_att, scaled_steps_att, dims=2), dims=2)
    raw_def = cumsum(cat(scaled_init_def, scaled_steps_def, dims=2), dims=2)

    # Center & Shift
    final_att = (raw_att .- mean(raw_att, dims=1)) .+ M_att
    final_def = (raw_def .- mean(raw_def, dims=1)) .+ M_def

    return final_att, final_def
end


# ==============================================================================
# 2. EXTRACT PARAMETERS (Prediction)
# ==============================================================================
function extract_parameters(
    model::GRWPoisson, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # --- A. Setup ---
    # Using feature_set dictionary syntax
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer n_rounds from chains (Columns in z_att_steps / n_teams + 1)
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # --- B. Reconstruct Full History ---
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    
    # Get Home Advantage
    γ_vec = vec(chain[:γ])

    # --- C. Prepare Output ---
    # Assuming PoissonRates is defined elsewhere as a NamedTuple or similar
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    # --- D. Prediction Loop ---
    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        # Time Indexing: 
        # If 'match_week' > n_rounds (forecasting), clamp to last known state
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        # Extract views (Team, Time, Samples)
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Calculate Rates
        λ_h = exp.(att_h .+ def_a .+ γ_vec)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end


# ==============================================================================
# 3. EXTRACT TRENDS (Analysis)
# ==============================================================================
"""
    extract_trends(model, feature_set, chains)

Extracts the evolution of Attack and Defense strengths over time.
Returns DataFrame: `[:team, :round, :att, :def]`
"""
function extract_trends(model, feature_set, chain)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # --- FIX ---
    # Use group() to isolate the parameter, then count the names
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1
    # -----------

    # 1. Reconstruct
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    
    # 2. Summarize (Mean of samples)
    att_means = dropdims(mean(att_cube, dims=3), dims=3)
    def_means = dropdims(mean(def_cube, dims=3), dims=3)

    # 3. Build DataFrame
    id_to_team = Dict(v => k for (k, v) in team_map)
    
    teams = String[]
    rounds = Int[]
    att_vals = Float64[]
    def_vals = Float64[]

    for i in 1:n_teams
        t_name = id_to_team[i]
        for t in 1:n_rounds
            push!(teams, t_name)
            push!(rounds, t)
            push!(att_vals, att_means[i, t])
            push!(def_vals, def_means[i, t])
        end
    end

    return DataFrame(
        team = teams,
        round = rounds,
        att = att_vals,
        def = def_vals
    )
end



