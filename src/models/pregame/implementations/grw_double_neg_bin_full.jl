# src/models/pregame/implementations/grw_double_neg_bin_full.jl

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics

export GRWNegativeBinomialFull

Base.@kwdef struct GRWNegativeBinomialFull <: AbstractDynamicNegBinModel
    # --- 1. Dynamic Global Baseline (Intercept mu) ---
    # RW for the league average goal rate
    μ_init::Distribution    = Normal(0.2, 0.2)
    σ_μ::Distribution       = Truncated(Normal(0, 0.05), 0, Inf) 
    z_μ_steps::Distribution = Normal(0, 1)

    # --- 2. Standard Home Advantage ---
    γ::Distribution         = Normal(log(1.3), 0.2)
    
    # --- 3. Hierarchical Dispersion (r / rho) ---
    # Structure: log(r_match) = Global + Delta[Home] + Delta[Away]
    # Note: Higher r = Lower variance (closer to Poisson). Lower r = High Overdispersion.
    log_r_global::Distribution = Normal(1.5, 0.5) 
    δ_r::Distribution      = Normal(0, 2) # How much teams vary in "chaos"

    # --- 4. Hierarchical Process Noise (Volatility sigma) ---
    # Structure: log(σ_team) = Global_Mean + Delta[Team]
    log_σ_att_global::Distribution = Normal(-3.0, 0.5) # Target ~0.05
    log_σ_def_global::Distribution = Normal(-3.0, 0.5)
    
    # Delta controls deviation from global volatility
    δ_σ_att::Distribution = Normal(0, 1)
    δ_σ_def::Distribution = Normal(0, 1)
    
    # --- 5. Initial State Hyperparameters (t=0) ---
    σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

    # Standard Latents
    z_init::Distribution = Normal(0,1)
    z_steps::Distribution = Normal(0,1)
end

@model function grw_negative_binomial_train(
                    n_teams, n_rounds, 
                    flat_home_ids, flat_away_ids, 
                    flat_goals_pairs,
                    time_indices, model::GRWNegativeBinomialFull,
                    ::Type{T} = Float64 ) where {T} 

    # ==========================================================
    # 1. PARAMETER DEFINITIONS
    # ==========================================================
    
    # --- A. Home Advantage ---
    γ ~ model.γ

    # --- B. Hierarchical Dispersion (r) ---
    # Global Baseline
    log_r_bar ~ model.log_r_global
    
    # Team-specific contributions to dispersion (Zero-centered)
    # If Team A has negative δ_r, matches involving them are more chaotic (lower r)
    δ_r ~ filldist(model.δ_r, n_teams)

    # --- C. Hierarchical Volatility (Process Noise σ) ---
    # Global Baselines
    log_σ_att_bar ~ model.log_σ_att_global
    log_σ_def_bar ~ model.log_σ_def_global

    # Team Deviations
    δ_σ_att_vec ~ filldist(model.δ_σ_att, n_teams)
    δ_σ_def_vec ~ filldist(model.δ_σ_def, n_teams)

    # Effective Sigma Vectors (n_teams)
    σ_att_vec = exp.(log_σ_att_bar .+ δ_σ_att_vec)
    σ_def_vec = exp.(log_σ_def_bar .+ δ_σ_def_vec)

    # --- D. Dynamic Global Intercept (μ) ---
    μ_init ~ model.μ_init
    σ_μ    ~ model.σ_μ
    z_μ_steps ~ filldist(model.z_μ_steps, n_rounds - 1)

    # --- E. Team Initial Spreads (t=0) ---
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # ==========================================================
    # 2. LATENT VARIABLES & RECONSTRUCTION
    # ==========================================================

    # --- Teams (Random Walk) ---
    z_att_init  ~ filldist(model.z_init, n_teams)
    z_def_init  ~ filldist(model.z_init, n_teams)
    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # Reconstruct Team Paths (NCP)
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0
    
    # Apply team-specific volatilities to steps
    # Broadcasting: (n_teams vector) .* (n_teams x rounds matrix)
    scaled_steps_att = z_att_steps .* σ_att_vec
    scaled_steps_def = z_def_steps .* σ_def_vec

    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # --- Global Mu (Random Walk) ---
    scaled_steps_μ = z_μ_steps .* σ_μ
    μ_traj = cumsum(vcat(μ_init, scaled_steps_μ))

    # ==========================================================
    # 3. CENTERING & LIKELIHOOD
    # ==========================================================
    
    # Zero-centering team strengths so μ_traj captures the true average
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- View Extraction ---
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))
    
    # Dynamic Mu per match
    μ_flat = μ_traj[time_indices]

    # --- Dispersion Calculation per Match ---
    # log(r) = Global + Home_Effect + Away_Effect
    # We construct the specific r for every match in the dataset
    r_matches = exp.(log_r_bar .+ δ_r[flat_home_ids] .+ δ_r[flat_away_ids] )

    # --- Lambda Calculation ---
    λₕ = exp.(μ_flat .+ γ .+ att_h_flat .+ def_a_flat)
    λₐ = exp.(μ_flat .+      att_a_flat .+ def_h_flat)

    # --- Observation ---
    flat_goals_pairs ~ arraydist(DoubleNegativeBinomial.(λₕ, λₐ, r_matches, r_matches))
    
    return nothing
end

function build_turing_model(model::GRWNegativeBinomialFull, feature_set::FeatureSet) 
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
    model::GRWNegativeBinomialFull, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # --- 1. Setup and Dimensions ---
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1
    n_samples = length(chain) # Total MCMC iterations

    # --- 2. Global Scalars (Samples) ---
    γ_vec = vec(Array(chain[:γ])) # Vector of length S

    # --- 3. Hierarchical Volatility Samples ---
    log_σ_att_bar = vec(Array(chain[:log_σ_att_bar]))
    log_σ_def_bar = vec(Array(chain[:log_σ_def_bar]))
    
    # Deltas: Shape (Samples, Teams)
    δ_σ_att_mat = Array(group(chain, :δ_σ_att_vec))
    δ_σ_def_mat = Array(group(chain, :δ_σ_def_vec))
    
    # Sigmas: Shape (Samples, Teams)
    # Using transpose ' to allow broadcasting (S, 1) .+ (S, T)
    σ_att_mat = exp.(log_σ_att_bar .+ δ_σ_att_mat)
    σ_def_mat = exp.(log_σ_def_bar .+ δ_σ_def_mat)

    # --- 4. Hierarchical Dispersion Samples ---
    log_r_bar = vec(Array(chain[:log_r_bar]))
    δ_r_mat   = Array(group(chain, :δ_r)) # (Samples, Teams)

    # --- 5. Dynamic Mu Reconstruction ---
    μ_init = vec(Array(chain[:μ_init]))
    σ_μ    = vec(Array(chain[:σ_μ]))
    z_μ_steps = Array(group(chain, :z_μ_steps)) # (Samples, Rounds-1)
    
    # Integrate Mu: Result (Samples, Rounds)
    raw_μ = hcat(μ_init, z_μ_steps .* σ_μ)
    μ_traj_samples = cumsum(raw_μ, dims=2) 

    # --- 6. Team Strength Reconstruction ---
    # Shapes: (Samples, Teams)
    z_att_init = Array(group(chain, :z_att_init))
    z_def_init = Array(group(chain, :z_def_init))
    
    # Steps: (Samples, Teams * (Rounds-1))
    z_att_steps_raw = Array(group(chain, :z_att_steps))
    z_def_steps_raw = Array(group(chain, :z_def_steps))

    # Initial Scaled State
    σ_att_0 = vec(Array(chain[:σ_att_0]))
    σ_def_0 = vec(Array(chain[:σ_def_0]))
    
    # We create a 3D Cube: [Team, Round, Sample]
    att_cube = zeros(n_teams, n_rounds, n_samples)
    def_cube = zeros(n_teams, n_rounds, n_samples)

    for s in 1:n_samples
        # Reshape steps for this specific sample
        s_att_steps = reshape(z_att_steps_raw[s, :], n_teams, n_rounds-1)
        s_def_steps = reshape(z_def_steps_raw[s, :], n_teams, n_rounds-1)
        
        # Scale steps by team-specific sigmas for this sample
        # σ_att_mat[s, :] is a vector of length n_teams
        s_att_path = cumsum(hcat(z_att_init[s, :] .* σ_att_0[s], s_att_steps .* σ_att_mat[s, :]), dims=2)
        s_def_path = cumsum(hcat(z_def_init[s, :] .* σ_def_0[s], s_def_steps .* σ_def_mat[s, :]), dims=2)
        
        # Mean-center across teams for this sample (Sum-to-zero constraint)
        att_cube[:, :, s] = s_att_path .- mean(s_att_path, dims=1)
        def_cube[:, :, s] = s_def_path .- mean(s_def_path, dims=1)
    end

    # --- 7. Prediction Loop ---
    # Note the return type changed to Vector{Float64} for each parameter
    ExtractionValue = NamedTuple{(:λ_h, :λ_a, :r), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        t_idx = clamp(row.match_week, 1, n_rounds)

        # Retrieve vectors across all samples
        att_h = att_cube[h_id, t_idx, :]
        def_a = def_cube[a_id, t_idx, :]
        att_a = att_cube[a_id, t_idx, :]
        def_h = def_cube[h_id, t_idx, :]
        
        μ_curr = μ_traj_samples[:, t_idx]

        # Calculate Distributions
        λ_h = exp.(μ_curr .+ γ_vec .+ att_h .+ def_a)
        λ_a = exp.(μ_curr .+ att_a .+ def_h)
        
        # Hierarchical Dispersion Samples
        # r = exp(Global_bar + Home_Delta + Away_Delta)
        r_val = exp.(log_r_bar .+ δ_r_mat[:, h_id] .+ δ_r_mat[:, a_id])

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=r_val)
    end

    return extraction_dict
end
