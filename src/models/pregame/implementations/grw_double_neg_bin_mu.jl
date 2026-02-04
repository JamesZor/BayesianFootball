# src/models/pregame/implementations/grw_double_neg_bin_mu.jl

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics

export GRWNegativeBinomialMu

Base.@kwdef struct GRWNegativeBinomialMu <: AbstractDynamicNegBinModel
    # --- Dynamic Global Baseline (Intercept) ---
    # Replaces the static Distribution with RW components
    μ_init::Distribution  = Normal(0.2, 0.2)   # Starting point for global log-rate
    σ_μ::Distribution     = Truncated(Normal(0, 0.05), 0, Inf) # Process noise for league average
    z_μ_steps::Distribution = Normal(0, 1)     # Standard Normal for NCP steps

    # Standard priors for team strength
    γ::Distribution       = Normal(log(1.3), 0.2)
    
    # Static Dispersion (we keep this static as requested, unlike the Phi model)
    log_r_prior::Distribution = Normal(1.5, 1.0) 

    # --- Dynamic Hyperparameters (Team Process Noise) ---
    σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
    
    # --- Initial State Hyperparameters (Hierarchical Prior t=0) ---
    σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

    z_init::Distribution = Normal(0,1)
    z_steps::Distribution = Normal(0,1)
end

@model function grw_negative_binomial_train(
                    n_teams, n_rounds, 
                    flat_home_ids, flat_away_ids, 
                    flat_goals_pairs,
                    time_indices, model::GRWNegativeBinomialMu,
                    ::Type{T} = Float64 ) where {T} 

    # --- 1. Hyperparameters ---
    # Home Advantage & Dispersion (Static)
    γ ~ model.γ
    log_r ~ model.log_r_prior 
    r = exp(log_r)

    # Process Noise (Teams)
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k
    
    # Process Noise (Global Intercept)
    σ_μ ~ model.σ_μ

    # Initial Spread (t=0)
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # --- 2. Latent Variables (Teams) ---
    z_att_init ~ filldist(model.z_init, n_teams)
    z_def_init ~ filldist(model.z_init, n_teams)

    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # --- 3. Latent Variables (Global Intercept μ) ---
    # NCP for the league average goal rate
    μ_init ~ model.μ_init
    z_μ_steps ~ filldist(model.z_μ_steps, n_rounds - 1)

    # --- 4. Trajectory Reconstruction (NCP) ---
    # Teams
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0
    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def

    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # Global Intercept μ
    # No "scaled_init" needed for scalar μ usually, just start at μ_init and walk
    scaled_steps_μ = z_μ_steps .* σ_μ
    
    # Reconstruct trajectory: Start with μ_init, add steps
    # Result is a Vector of length n_rounds
    μ_traj = cumsum(vcat(μ_init, scaled_steps_μ))

    # --- 5. Centering (Robust Formulation) ---
    # Strictly Zero-Mean deviations for teams. 
    # This makes μ_traj identifiable as the "League Average".
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- 6. Likelihood ---
    # Extract specific match strengths
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Extract specific time-varying global intercepts
    # We map time_indices (1...N) to the μ_traj vector
    μ_flat = μ_traj[time_indices]

    # Calculate Log-Rates
    # Note: μ_flat is now a vector matching the length of the data
    λₕ = exp.(μ_flat .+ γ .+ att_h_flat .+ def_a_flat)
    λₐ = exp.(μ_flat .+      att_a_flat .+ def_h_flat)

    flat_goals_pairs ~ arraydist(DoubleNegativeBinomial.(λₕ, λₐ, r, r))
    
    return nothing
end

function build_turing_model(model::GRWNegativeBinomialMu, feature_set::FeatureSet) 
    data_matrix = permutedims(hcat(feature_set[:flat_home_goals], feature_set[:flat_away_goals]))

    return grw_negative_binomial_train(
        feature_set[:n_teams]::Int,
        feature_set[:n_rounds]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        data_matrix::Matrix{Int}, 
        feature_set[:time_indices],
        model
    )
end


function extract_parameters(
    model::GRWNegativeBinomialMu, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # --- A. Setup ---
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # --- B. Reconstruct Full History (Teams) ---
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)

    # --- C. Reconstruct Full History (Global μ) ---
    # 1. Extract raw chains
    μ_init = vec(Array(chain[:μ_init])) # Vector (Samples)
    σ_μ    = vec(Array(chain[:σ_μ]))    # Vector (Samples)
    
    # 2. Extract steps: Matrix (Samples x (N_rounds-1))
    z_μ_steps_mat = Array(group(chain, :z_μ_steps)) 

    # 3. Integrate Random Walk
    scaled_steps_μ = z_μ_steps_mat .* σ_μ
    
    # Combine: [Init, Steps] -> (Samples x Rounds)
    raw_μ = hcat(μ_init, scaled_steps_μ)
    μ_traj = cumsum(raw_μ, dims=2) # Result: (Samples x Rounds)

    # --- D. Static Globals ---
    γ_vec = vec(Array(chain[:γ]))
    r_vec = exp.(vec(Array(chain[:log_r])))

    ExtractionValue = NamedTuple{(:λ_h, :λ_a, :r), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        # Time Indexing
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        # Extract Views
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Extract Time-Specific Intercept (Samples)
        μ_current = view(μ_traj, :, t_idx)

        # Calculate Rates
        # Note: μ_current is now dynamic per week
        λ_h = exp.(μ_current .+ γ_vec .+ att_h .+ def_a)
        λ_a = exp.(μ_current .+          att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=r_vec)
    end

    return extraction_dict
end


"""
    extract_mu_trends(model, feature_set, chain)

Extracts the time-varying global intercept `μ` from the GRWNegativeBinomialMu model.
Returns a DataFrame with [round, mu_mean, mu_lower, mu_upper].
"""
function extract_mu_trends(
    model::GRWNegativeBinomialMu, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # --- 1. Infer n_rounds ---
    # We can look at the number of steps in the μ process directly
    # The number of steps is (n_rounds - 1)
    step_names = names(group(chain, :z_μ_steps))
    n_rounds = length(step_names) + 1

    # --- 2. Reconstruct μ Trajectory ---
    # Extract raw parameters
    μ_init = vec(Array(chain[:μ_init]))   # Vector (Samples)
    σ_μ    = vec(Array(chain[:σ_μ]))      # Vector (Samples)
    
    # Extract steps matrix: (Samples x (n_rounds - 1))
    z_steps_mat = Array(group(chain, :z_μ_steps)) 

    # Integrate Random Walk (NCP)
    # Scale the standard normal steps by the process noise σ_μ
    # Broadcasting: (Samples x Steps) .* (Samples)
    scaled_steps = z_steps_mat .* σ_μ
    
    # Combine initial state with steps
    raw_mu_matrix = hcat(μ_init, scaled_steps)
    
    # Cumulative sum across time (dimension 2)
    # Result: (Samples x Rounds)
    μ_traj = cumsum(raw_mu_matrix, dims=2) 

    # --- 3. Summarize Statistics per Round ---
    rounds = Int[]
    mu_mean = Float64[]
    mu_lower = Float64[]
    mu_upper = Float64[]

    for t in 1:n_rounds
        push!(rounds, t)
        
        # Get all samples for time t (View for efficiency)
        vals = view(μ_traj, :, t)
        
        push!(mu_mean, mean(vals))
        push!(mu_lower, quantile(vals, 0.05))
        push!(mu_upper, quantile(vals, 0.95))
    end

    return DataFrame(
        round = rounds,
        mu_mean = mu_mean,
        mu_lower = mu_lower,
        mu_upper = mu_upper
    )
end
