# src/models/pregame/implementations/grw_double_neg_bin.jl

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics

export GRWNegativeBinomial

Base.@kwdef struct GRWNegativeBinomial <: AbstractDynamicNegBinModel
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.5)

    # Standard priors for team strength
    γ::Distribution   = Normal(log(1.3), 0.2)
    
    # Dispersion parameter (Negative Binomial)
    log_r_prior::Distribution = Normal(1.5, 1.0) 

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
                    time_indices, model::GRWNegativeBinomial,
                    ::Type{T} = Float64 ) where {T} 

    # --- 1. Hyperparameters ---
    # Global Intercept
    μ ~ model.μ
    
    # Home Advantage
    γ ~ model.γ
    
    # Dispersion (r)
    log_r ~ model.log_r_prior 
    r = exp(log_r)

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

    flat_goals_pairs ~ arraydist(DoubleNegativeBinomial.(λₕ, λₐ, r, r))
    
    return nothing
end


function build_turing_model(model::GRWNegativeBinomial, feature_set::FeatureSet) 
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
# 2. EXTRACT PARAMETERS (Prediction)
# ==============================================================================
function extract_parameters(
    model::GRWNegativeBinomial, 
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
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)

    # Extract Globals
    γ_vec = vec(Array(chain[:γ]))
    μ_vec = vec(Array(chain[:μ])) # New global intercept
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

        # Extract views
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Calculate Rates (Standard Formulation)
        λ_h = exp.(μ_vec .+ γ_vec .+ att_h .+ def_a)
        λ_a = exp.(μ_vec .+ att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=r_vec)
    end

    return extraction_dict
end

# ==============================================================================
# 3. EXTRACT TRENDS (Analysis)
# ==============================================================================
function extract_trends(model::GRWNegativeBinomial, feature_set::FeatureSet, chain::Chains)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

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
