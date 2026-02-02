
# src/models/pregame/implementations/grw_double_neg_bin_phi.jl

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics

export GRWNegativeBinomial

Base.@kwdef struct GRWNegativeBinomialPhi <: AbstractDynamicNegBinModel
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.5)

    # Standard priors for team strength
    γ::Distribution   = Normal(log(1.3), 0.2)
    
    # Dispersion parameter (Negative Binomial)
    log_r_init::Distribution = Normal(1.5, 0.5) 
    r_steps::Distribution = Normal(0,1)
    σ_r::Distribution = Truncated(Normal(0,0.05), 0, Inf )


    # --- Dynamic Hyperparameters (Process Noise) ---
    # Adjusted to 0.05 to prevent excessive volatility (factor of ~1.05 per week)
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
                    time_indices, model::GRWNegativeBinomialPhi,
                    ::Type{T} = Float64 ) where {T} 

    # --- 1. Hyperparameters ---
    # Global Intercept
    μ ~ model.μ
    
    # Home Advantage
    γ ~ model.γ
    
    # Process Noise
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k

    # Initial Spread (t=0)
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # --- 2. Latent Variables ---
    # Initial State
    z_att_init ~ filldist(model.z_init, n_teams)
    z_def_init ~ filldist(model.z_init, n_teams)

    # Steps
    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # --- 3. Trajectory Reconstruction (NCP) ---
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0

    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def

    # Integrate (Random Walk)
    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # --- 4. Centering (Robust Formulation) ---
    # Strictly Zero-Mean deviations. Global rate is handled by μ_global.
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- 5. GRW-Phi (Dispersion) ---
    σ_r ~ model.σ_r 
    log_r_init ~ model.log_r_init 

    # samples steps NCP for GRW phi 
    r_steps ~ filldist(model.r_steps, n_rounds -1)
    
    # reconstruct vector 
    scaled_steps_r = r_steps .* σ_r 
    ϕ_round = exp.(cumsum(vcat( log_r_init, scaled_steps_r)))
    # mapping
    ϕ = ϕ_round[time_indices]

    # --- 5. Likelihood ---
    # Extract specific match strengths
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Calculate Log-Rates
    # λ = exp(μ_global + γ + att + def)
    λₕ = exp.(μ .+ γ .+ att_h_flat .+ def_a_flat)
    λₐ = exp.(μ .+      att_a_flat .+ def_h_flat)

    flat_goals_pairs ~ arraydist(DoubleNegativeBinomial.(λₕ, λₐ, ϕ, ϕ ))
    
    return nothing
end


function build_turing_model(model::GRWNegativeBinomialPhi, feature_set::FeatureSet) 
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

# ==============================================================================
# 2. EXTRACT PARAMETERS (Prediction) for GRW-Phi
# ==============================================================================


function extract_parameters(
    model::GRWNegativeBinomialPhi, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # --- A. Setup ---
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer n_rounds from the chain dimensions
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # --- B. Reconstruct Full History (Teams) ---
    # Uses your common.jl helper: returns (Teams x Rounds x Samples)
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)

    # --- C. Reconstruct Full History (Dispersion r) ---
    # 1. Extract raw chains
    log_r_init = vec(Array(chain[:log_r_init])) # Vector (Samples)
    σ_r        = vec(Array(chain[:σ_r]))        # Vector (Samples)
    
    # 2. Extract steps: Matrix (Samples x (N_rounds-1))
    # Note: Turing groups are usually flattened, but Array(group) on a 1D vector of parameters usually returns (Samples x Dim)
    r_steps_mat = Array(group(chain, :r_steps)) 

    # 3. Integrate Random Walk for r (Vectorized over samples)
    # Scale steps: elementwise mult for broadcasting (Samples x Time) .* (Samples)
    scaled_steps_r = r_steps_mat .* σ_r

    # Combine: [Init, Steps] -> (Samples x Rounds)
    raw_log_r = hcat(log_r_init, scaled_steps_r)
    
    # Cumsum along time dimension (dim 2)
    log_r_traj = cumsum(raw_log_r, dims=2)
    
    # Convert to natural scale
    r_traj = exp.(log_r_traj) # Result: (Samples x Rounds)

    # --- D. Global Intercepts ---
    γ_vec = vec(Array(chain[:γ]))
    μ_vec = vec(Array(chain[:μ]))

    # --- E. Prediction Loop ---
    # Define return type
    ExtractionValue = NamedTuple{(:λ_h, :λ_a, :r), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        # Time Indexing
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        # Extract Team Views: (Samples)
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Extract Time-Specific Dispersion: (Samples)
        # Access the column corresponding to the current time index
        r_current = view(r_traj, :, t_idx) 

        # Calculate Rates
        λ_h = exp.(μ_vec .+ γ_vec .+ att_h .+ def_a)
        λ_a = exp.(μ_vec .+ att_a .+ def_h)

        # Store results (r is now a vector of samples for this specific week)
        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=Vector(r_current))
    end

    return extraction_dict
end


function extract_trends(
    model::GRWNegativeBinomialPhi, 
    feature_set::FeatureSet, 
    chain::Chains
)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer rounds
    step_names = names(group(chain, :z_att_steps))
    n_rounds = (length(step_names) ÷ n_teams) + 1

    # 1. Reconstruct via common helper
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    
    # 2. Summarize (Mean across samples, dim 3)
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


"""
    extract_dispersion_trends(model, feature_set, chain)

Extracts the time-varying dispersion parameter `r` (phi) from the GRWNegativeBinomialPhi model.
Returns a DataFrame with [round, r_mean, r_lower, r_upper].
"""
function extract_dispersion_trends(
    model::GRWNegativeBinomialPhi, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # Infer n_rounds
    n_teams = feature_set[:n_teams]
    step_names = names(group(chain, :z_att_steps))
    n_rounds = (length(step_names) ÷ n_teams) + 1

    # 1. Reconstruct r Trajectory (Same logic as extract_parameters)
    log_r_init = vec(Array(chain[:log_r_init]))
    σ_r        = vec(Array(chain[:σ_r]))
    r_steps_mat = Array(group(chain, :r_steps)) 

    scaled_steps = r_steps_mat .* σ_r
    raw_log_r = hcat(log_r_init, scaled_steps)
    log_r_traj = cumsum(raw_log_r, dims=2)
    r_traj = exp.(log_r_traj) # (Samples x Rounds)

    # 2. Summarize per round
    rounds = Int[]
    r_mean = Float64[]
    r_lower = Float64[]
    r_upper = Float64[]

    for t in 1:n_rounds
        push!(rounds, t)
        
        # Get all samples for time t
        vals = view(r_traj, :, t)
        
        push!(r_mean, mean(vals))
        push!(r_lower, quantile(vals, 0.05))
        push!(r_upper, quantile(vals, 0.95))
    end

    return DataFrame(
        round = rounds,
        r_mean = r_mean,
        r_lower = r_lower,
        r_upper = r_upper
    )
end
