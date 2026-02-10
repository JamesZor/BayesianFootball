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
    # --- A. Setup & Dimensions ---
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer n_rounds and n_samples
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1
    n_samples = size(chain, 1) * size(chain, 3) # Handle multiple chains if present

    # --- B. Reconstruct Full Distributions ---
    
    # 1. Global Scalars (Vectors of samples)
    γ_vec = vec(Array(chain[:γ])) # Shape: (n_samples,)

    # 2. Hierarchical Volatility (σ) Reconstruction
    # Result Shape: (n_teams, n_samples)
    log_σ_att_bar = vec(Array(chain[:log_σ_att_bar]))' # Transpose to (1, n_samples) for broadcasting
    log_σ_def_bar = vec(Array(chain[:log_σ_def_bar]))'
    
    δ_σ_att = Array(group(chain, :δ_σ_att_vec))' # Shape: (n_teams, n_samples)
    δ_σ_def = Array(group(chain, :δ_σ_def_vec))'
    
    σ_att_team = exp.(log_σ_att_bar .+ δ_σ_att) 
    σ_def_team = exp.(log_σ_def_bar .+ δ_σ_def)

    # 3. Hierarchical Dispersion (r) Reconstruction
    # Result Shape: δ_r is (n_teams, n_samples)
    log_r_bar = vec(Array(chain[:log_r_bar])) # (n_samples,)
    δ_r_mat   = Array(group(chain, :δ_r))'    # (n_teams, n_samples)

    # 4. Dynamic Mu Reconstruction
    # Result Shape: (n_rounds, n_samples)
    μ_init = vec(Array(chain[:μ_init]))' # (1, n_samples)
    σ_μ    = vec(Array(chain[:σ_μ]))'    # (1, n_samples)
    
    z_μ_steps = Array(group(chain, :z_μ_steps))' # (n_rounds-1, n_samples)
    
    # Reconstruct Random Walk for Mu
    scaled_steps_μ = z_μ_steps .* σ_μ
    raw_μ = vcat(μ_init, scaled_steps_μ) # (n_rounds, n_samples)
    μ_traj = cumsum(raw_μ, dims=1)       # (n_rounds, n_samples)

    # 5. Team Strength Reconstruction
    # Result Shape: (n_teams, n_rounds, n_samples)
    
    # 5a. Initial States
    z_att_init = Array(group(chain, :z_att_init))' # (n_teams, n_samples)
    z_def_init = Array(group(chain, :z_def_init))' 
    
    σ_att_0 = vec(Array(chain[:σ_att_0]))' # (1, n_samples)
    σ_def_0 = vec(Array(chain[:σ_def_0]))'

    # 5b. Steps (Reshape required)
    # Raw comes out as (n_samples, n_teams * (n_rounds-1))
    # We want (n_teams, n_rounds-1, n_samples)
    raw_z_att = Array(group(chain, :z_att_steps))
    raw_z_def = Array(group(chain, :z_def_steps))
    
    # Reshape helper: Permute to (Teams, Time, Samples)
    z_att_steps = permutedims(reshape(raw_z_att, n_samples, n_teams, n_rounds-1), [2, 3, 1])
    z_def_steps = permutedims(reshape(raw_z_def, n_samples, n_teams, n_rounds-1), [2, 3, 1])

    # 5c. Integration (Random Walk)
    # Note: σ_att_team is (n_teams, n_samples). We need to broadcast over the Time dimension (dim 2)
    # Reshape Sigma to (n_teams, 1, n_samples)
    σ_att_team_3d = reshape(σ_att_team, n_teams, 1, n_samples)
    σ_def_team_3d = reshape(σ_def_team, n_teams, 1, n_samples)

    scaled_init_att = z_att_init .* σ_att_0         # (n_teams, n_samples) -- broadcast ok
    scaled_init_def = z_def_init .* σ_def_0
    
    # Broadcast multiply steps by team-specific volatilities
    scaled_steps_att = z_att_steps .* σ_att_team_3d 
    scaled_steps_def = z_def_steps .* σ_def_team_3d

    # Concatenate Init + Steps along Time dimension (dim 2)
    # Init needs to be reshaped to (n_teams, 1, n_samples) to cat
    att_raw = cumsum(cat(reshape(scaled_init_att, n_teams, 1, n_samples), scaled_steps_att, dims=2), dims=2)
    def_raw = cumsum(cat(reshape(scaled_init_def, n_teams, 1, n_samples), scaled_steps_def, dims=2), dims=2)

    # 5d. Centering (Zero-Sum constraint per round, per sample)
    # Mean across teams (dim 1)
    att_cube = att_raw .- mean(att_raw, dims=1)
    def_cube = def_raw .- mean(def_raw, dims=1)

    # --- C. Prediction Loop ---
    ExtractionValue = NamedTuple{(:λ_h, :λ_a, :r), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        # 1. Retrieve Team Strengths for this specific match
        # Views: (n_samples,)
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)
        
        # 2. Retrieve Dynamic Global Mu for this time step
        μ_curr = view(μ_traj, t_idx, :) # (n_samples,)

        # 3. Calculate Rates (Vectorized over samples)
        # λ = exp(μ + γ + att + def)
        λ_h = exp.(μ_curr .+ γ_vec .+ att_h .+ def_a)
        λ_a = exp.(μ_curr .+          att_a .+ def_h)
        
        # 4. Calculate Match-Specific Dispersion
        # log(r) = Global + Home_Delta + Away_Delta
        # We perform this calculation per sample
        δ_r_h = view(δ_r_mat, h_id, :)
        δ_r_a = view(δ_r_mat, a_id, :)
        
        log_r_match = log_r_bar .+ δ_r_h .+ δ_r_a
        r_val = exp.(log_r_match)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=r_val)
    end

    return extraction_dict
end
