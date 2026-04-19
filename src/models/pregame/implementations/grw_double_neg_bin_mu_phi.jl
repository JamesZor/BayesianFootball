# src/models/pregame/implementations/grw_double_neg_bin_mu_phi.jl

using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics

export GRWNegativeBinomialMuPhi

Base.@kwdef struct GRWNegativeBinomialMuPhi <: AbstractDynamicNegBinModel
    # --- Dynamic Global Baseline (Intercept μ) ---
    μ_init::Distribution    = Normal(0.2, 0.2)
    σ_μ::Distribution       = Truncated(Normal(0, 0.05), 0, Inf)
    z_μ_steps::Distribution = Normal(0, 1)

    # --- Dynamic Dispersion (Phi / r) ---
    # Log-scale random walk for positivity
    log_r_init::Distribution = Normal(1.5, 0.5) 
    σ_r::Distribution        = Truncated(Normal(0, 0.05), 0, Inf)
    r_steps::Distribution    = Normal(0, 1)

    # --- Static Parameters ---
    γ::Distribution = Normal(log(1.3), 0.2) # Home Advantage

    # --- Team Dynamic Hyperparameters ---
    σ_k::Distribution     = Truncated(Normal(0, 0.05), 0, Inf)
    σ_0::Distribution     = Truncated(Normal(0.5, 0.2), 0, Inf)
    z_init::Distribution  = Normal(0, 1)
    z_steps::Distribution = Normal(0, 1)
end

@model function grw_neg_bin_mu_phi_train(
                    n_teams, n_rounds, 
                    flat_home_ids, flat_away_ids, 
                    flat_goals_pairs,
                    time_indices, model::GRWNegativeBinomialMuPhi,
                    ::Type{T} = Float64 ) where {T} 

    # --- 1. Parameters & Hyperparameters ---
    γ ~ model.γ
    
    # Process Noises
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k
    σ_μ   ~ model.σ_μ
    σ_r   ~ model.σ_r
    
    # Initial Spreads
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # --- 2. Latent Variables ---
    # Teams
    z_att_init ~ filldist(model.z_init, n_teams)
    z_def_init ~ filldist(model.z_init, n_teams)
    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # Global Intercept μ
    μ_init ~ model.μ_init
    z_μ_steps ~ filldist(model.z_μ_steps, n_rounds - 1)

    # Dispersion r
    log_r_init ~ model.log_r_init
    z_r_steps ~ filldist(model.r_steps, n_rounds - 1)

    # --- 3. Reconstruction (NCP) ---
    
    # A. Teams
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0
    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def

    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # Centering (Identifiability)
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # B. Global Intercept μ
    scaled_steps_μ = z_μ_steps .* σ_μ
    μ_traj = cumsum(vcat(μ_init, scaled_steps_μ)) # Vector Length: n_rounds

    # C. Global Dispersion r (Log-Space Walk)
    scaled_steps_r = z_r_steps .* σ_r
    log_r_traj = cumsum(vcat(log_r_init, scaled_steps_r)) 
    r_traj = exp.(log_r_traj) # Vector Length: n_rounds

    # --- 4. Likelihood ---
    
    # Map time indices to specific values
    # These become vectors matching the length of the match data
    μ_flat = μ_traj[time_indices]
    r_flat = r_traj[time_indices]

    # Map team strengths
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Calculate Rates
    λ_h = exp.(μ_flat .+ γ .+ att_h_flat .+ def_a_flat)
    λ_a = exp.(μ_flat .+      att_a_flat .+ def_h_flat)

    # Broadcast likelihood
    # Note: r_flat is vector, so DoubleNegBin is broadcasted per match
    flat_goals_pairs ~ arraydist(DoubleNegativeBinomial.(λ_h, λ_a, r_flat, r_flat))
    
    return nothing
end

function build_turing_model(model::GRWNegativeBinomialMuPhi, feature_set::FeatureSet) 
    data_matrix = permutedims(hcat(feature_set[:flat_home_goals], feature_set[:flat_away_goals]))

    return grw_neg_bin_mu_phi_train(
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
# EXTRACTION: Parameters for Prediction
# ==============================================================================
function extract_parameters(
    model::GRWNegativeBinomialMuPhi, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # Infer rounds
    step_names = names(group(chain, :z_att_steps))
    n_rounds = (length(step_names) ÷ n_teams) + 1

    # 1. Reconstruct Teams
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)

    # 2. Reconstruct μ Trajectory
    μ_init = vec(Array(chain[:μ_init]))
    σ_μ    = vec(Array(chain[:σ_μ]))
    z_μ_steps = Array(group(chain, :z_μ_steps))
    
    raw_μ = hcat(μ_init, z_μ_steps .* σ_μ)
    μ_traj = cumsum(raw_μ, dims=2) # (Samples x Rounds)

    # 3. Reconstruct r Trajectory
    log_r_init = vec(Array(chain[:log_r_init]))
    σ_r        = vec(Array(chain[:σ_r]))
    z_r_steps  = Array(group(chain, :z_r_steps)) # Note name match with struct

    raw_log_r = hcat(log_r_init, z_r_steps .* σ_r)
    r_traj = exp.(cumsum(raw_log_r, dims=2)) # (Samples x Rounds)

    # 4. Globals
    γ_vec = vec(Array(chain[:γ]))

    # 5. Prediction Dictionary
    ExtractionValue = NamedTuple{(:λ_h, :λ_a, :r), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        t_idx = clamp(row.match_week, 1, n_rounds)

        # Views (Samples)
        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Dynamic Globals (Samples)
        μ_current = view(μ_traj, :, t_idx)
        r_current = view(r_traj, :, t_idx)

        λ_h = exp.(μ_current .+ γ_vec .+ att_h .+ def_a)
        λ_a = exp.(μ_current .+          att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a, r=Vector(r_current))
    end

    return extraction_dict
end

# ==============================================================================
# EXTRACTION: Trends (Teams + Mu)
# ==============================================================================
function extract_trends(
    model::GRWNegativeBinomialMuPhi, 
    feature_set::FeatureSet, 
    chain::Chains
)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    step_names = names(group(chain, :z_att_steps))
    n_rounds = (length(step_names) ÷ n_teams) + 1

    # Reconstruct Teams
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    att_means = dropdims(mean(att_cube, dims=3), dims=3)
    def_means = dropdims(mean(def_cube, dims=3), dims=3)

    # Reconstruct μ
    μ_init = vec(Array(chain[:μ_init]))
    σ_μ    = vec(Array(chain[:σ_μ]))
    z_μ_steps = Array(group(chain, :z_μ_steps))
    μ_traj = cumsum(hcat(μ_init, z_μ_steps .* σ_μ), dims=2)
    μ_means = vec(mean(μ_traj, dims=1))

    # Build DataFrame
    id_to_team = Dict(v => k for (k, v) in team_map)
    
    teams = String[]
    rounds = Int[]
    att_vals, def_vals, mu_vals, total_att_vals = Float64[], Float64[], Float64[], Float64[]

    for i in 1:n_teams
        t_name = id_to_team[i]
        for t in 1:n_rounds
            push!(teams, t_name)
            push!(rounds, t)
            
            a_val, d_val, m_val = att_means[i, t], def_means[i, t], μ_means[t]
            
            push!(att_vals, a_val)
            push!(def_vals, d_val)
            push!(mu_vals, m_val)
            push!(total_att_vals, a_val + m_val)
        end
    end

    return DataFrame(team=teams, round=rounds, att=att_vals, def=def_vals, 
                     mu_global=mu_vals, total_att=total_att_vals)
end

# ==============================================================================
# EXTRACTION: Trends (Dispersion r)
# ==============================================================================
function extract_dispersion_trends(
    model::GRWNegativeBinomialMuPhi, 
    feature_set::FeatureSet, 
    chain::Chains
)
    # Infer Rounds
    step_names = names(group(chain, :z_μ_steps)) # Use Mu steps as proxy for time
    n_rounds = length(step_names) + 1

    # Reconstruct r
    log_r_init = vec(Array(chain[:log_r_init]))
    σ_r        = vec(Array(chain[:σ_r]))
    z_r_steps  = Array(group(chain, :z_r_steps))
    
    raw_log_r = hcat(log_r_init, z_r_steps .* σ_r)
    r_traj = exp.(cumsum(raw_log_r, dims=2))

    # Summarize
    rounds = Int[]
    r_mean, r_lower, r_upper = Float64[], Float64[], Float64[]

    for t in 1:n_rounds
        push!(rounds, t)
        vals = view(r_traj, :, t)
        push!(r_mean, mean(vals))
        push!(r_lower, quantile(vals, 0.05))
        push!(r_upper, quantile(vals, 0.95))
    end

    return DataFrame(round=rounds, r_mean=r_mean, r_lower=r_lower, r_upper=r_upper)
end
