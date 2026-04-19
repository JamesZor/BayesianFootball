# src/models/pregame/mode/grw-poisson.jl
using DataFrames
using Turing
using LinearAlgebra
using Statistics

export GRWPoisson

Base.@kwdef struct GRWPoisson <: AbstractDynamicPoissonModel 
      # --- Global Baseline (Intercept) ---
      # Represents the average log-goal rate for an away team.
      # Typically around 0.2 (approx 1.2 goals) to -0.3 (approx 0.7 goals).
      μ::Distribution = Normal(0.2, 0.5)

      # --- Home Advantage ---
      γ::Distribution = Normal(log(1.3), 0.2) 
  
      # --- Dynamic Hyperparameters (Process Noise) ---
      # How much ability changes per step
      σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf) 
      
      # --- Initial State Hyperparameters (Hierarchical Prior t=0) ---
      # Prior on the SPREAD of abilities at t=0
      σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

      # --- Latent Variables ---
      z_init::Distribution = Normal(0,1)
      z_steps::Distribution = Normal(0,1)
end

@model function grw_poisson_model_train(n_teams, n_rounds, 
                                      flat_home_ids, flat_away_ids, 
                                      flat_home_goals, flat_away_goals, 
                                      time_indices, model::GRWPoisson,
                                      ::Type{T} = Float64 ) where {T} 
    
    # --- 1. Hyperparameters ---
    # Global Intercept (The Baseline)
    μ ~ model.μ
    
    # Home Advantage
    γ ~ model.γ

    # Random Walk Innovations (Process Noise)
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k
    
    # Initial Spread (t=0)
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # --- 2. Latent Variables ---
    # Initial State (Standardized)
    z_att_init ~ filldist(model.z_init, n_teams)
    z_def_init ~ filldist(model.z_init, n_teams)

    # Steps (Standardized)
    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # --- 3. Trajectory Reconstruction (NCP) ---
    
    # Scale: innovations * σ_k
    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def
    
    # Scale: initial * σ_0
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0

    # Integrate (Random Walk)
    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # --- 4. Centering (The Robust Formulation) ---
    # Enforce strictly Zero-Mean deviations at every time step.
    # This orthogonalizes 'ability' from the 'global intercept' (μ_global).
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- 5. Likelihood ---
    # Extract strengths for specific matches
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Calculate Log-Rates
    # λ_home = exp(Intercept + HomeAdv + Att_Home + Def_Away)
    log_λₕ = μ .+ γ .+ att_h_flat .+ def_a_flat 

    # λ_away = exp(Intercept + Att_Away + Def_Home)
    log_λₐ = μ .+ att_a_flat .+ def_h_flat 

    # Observe
    flat_home_goals ~ arraydist(LogPoisson.(log_λₕ))
    flat_away_goals ~ arraydist(LogPoisson.(log_λₐ))
    
    return nothing
end

function build_turing_model(model::GRWPoisson, feature_set::FeatureSet)
    data = feature_set.data
    
    return grw_poisson_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:flat_home_ids],    
        data[:flat_away_ids],     
        data[:flat_home_goals],   
        data[:flat_away_goals],   
        data[:time_indices],      
        model
    )
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
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer n_rounds 
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # --- B. Reconstruct Full History ---
    # Uses the local helper to ensure Zero-Sum centering
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    
    # Get Global Parameters
    γ = vec(Array(chain[:γ]))
    μ = vec(Array(chain[:μ])) # The new global intercept

    # --- C. Prepare Output ---
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    # --- D. Prediction Loop ---
    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        # Time Indexing
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        # Extract views (Samples vector)
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Calculate Rates (Standard Formulation)
        # λ = exp(μ_global + γ + att + def)
        λ_h = exp.(μ .+ γ  .+ att_h .+ def_a)
        λ_a = exp.(μ .+       att_a .+ def_h)

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
function extract_trends(model::GRWPoisson, feature_set, chain)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer rounds
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # 1. Reconstruct (Centered)
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
