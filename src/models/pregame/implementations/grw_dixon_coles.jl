# src/models/pregame/implementations/grw_dixon_coles.jl

using Turing, Distributions, LinearAlgebra, Statistics
using ..MyDistributions 

export GRWDixonColes

# ==============================================================================
# 1. THE STRUCT
# ==============================================================================
Base.@kwdef struct GRWDixonColes <: AbstractDynamicDixonColesModel 
      # --- Global Baseline (Intercept) ---
      # Represents the average log-goal rate for an away team.
      μ::Distribution = Normal(0.2, 0.5)

      # --- Static Parameters ---
      γ::Distribution = Normal(log(1.3), 0.2) # Home Advantage

      # --- Dynamic Hyperparameters (Process Noise) ---
      # Standard deviation of the weekly random walk step
      σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf) 
      
      # --- Hierarchical Priors (t=0) ---
      # Initial Spread of abilities
      σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

      # --- Dependence Parameter (Dixon-Coles specific) ---
      # We use a raw Normal prior which is tanh-transformed later
      ρ_raw::Distribution = Normal(0, 1.0) 

      # --- Latent Variables (Random Walk) ---
      z_init::Distribution = Normal(0,1)
      z_steps::Distribution = Normal(0,1)
end

function Base.show(io::IO, ::MIME"text/plain", m::GRWDixonColes)
    printstyled(io, "Dynamic Dixon-Coles (GRW)\n", color=:magenta, bold=true)
    println(io, "  ├── Process Noise: $(m.σ_k)")
    println(io, "  ├── Initial Spread: $(m.σ_0)")
    println(io, "  └── Dependence (ρ): $(m.ρ_raw)")
end

# ==============================================================================
# 2. THE TURING MODEL
# ==============================================================================
@model function grw_dixon_coles_model_train(
    n_teams, n_rounds, 
    flat_home_ids, flat_away_ids, 
    time_indices,
    # Dixon-Coles Grouping Indices
    idx_00, idx_10, idx_01, idx_11, idx_else,
    scores_else_x, scores_else_y,
    model::GRWDixonColes,
    ::Type{T} = Float64
) where {T} 
    
    # --- 1. Hyperparameters ---
    # Global Intercept
    μ ~ model.μ
    
    # Home Advantage
    γ ~ model.γ
    
    # Process Noise & Initial Spread
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

    # Dependence (Rho)
    ρ_raw ~ model.ρ_raw
    ρ = 0.3 * tanh(ρ_raw)

    # --- 2. Latent Variables (Random Walk) ---
    z_att_init ~ filldist(model.z_init, n_teams)
    z_def_init ~ filldist(model.z_init, n_teams)

    z_att_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)
    z_def_steps ~ filldist(model.z_steps, n_teams, n_rounds - 1)

    # --- 3. Trajectory Reconstruction (NCP) ---
    # Scale Steps
    scaled_steps_att = z_att_steps .* σ_att
    scaled_steps_def = z_def_steps .* σ_def
    
    # Scale Init
    scaled_init_att = z_att_init .* σ_att_0
    scaled_init_def = z_def_init .* σ_def_0

    # Integrate (Cumulative Sum)
    att_raw = cumsum(hcat(scaled_init_att, scaled_steps_att), dims=2)
    def_raw = cumsum(hcat(scaled_init_def, scaled_steps_def), dims=2)

    # --- 4. Centering (Robust Formulation) ---
    # Enforce strictly Zero-Mean deviations. 
    # The global rate is handled entirely by μ_global.
    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- 5. Log-Rate Calculation (Vectorized) ---
    att_h_flat = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a_flat = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a_flat = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h_flat = view(def, CartesianIndex.(flat_home_ids, time_indices))

    # Calculate Theta (Log-Rates)
    # θ = μ_global + components
    θ_home_all = μ .+ att_h_flat .+ def_a_flat .+ γ
    θ_away_all = μ .+ att_a_flat .+ def_h_flat

    # --- 6. Likelihood (Dixon-Coles Grouped) ---
    
    # Group 0-0
    if !isempty(idx_00)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_00], θ_away_all[idx_00], ρ, :s00),
            zeros(2, length(idx_00))
        )
    end

    # Group 1-0
    if !isempty(idx_10)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_10], θ_away_all[idx_10], ρ, :s10),
            zeros(2, length(idx_10))
        )
    end

    # Group 0-1
    if !isempty(idx_01)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_01], θ_away_all[idx_01], ρ, :s01),
            zeros(2, length(idx_01))
        )
    end
    
    # Group 1-1
    if !isempty(idx_11)
        Turing.@addlogprob! logpdf(
            DixonColesLogGroup(θ_home_all[idx_11], θ_away_all[idx_11], ρ, :s11),
            zeros(2, length(idx_11))
        )
    end

    # Group Else (Standard Independent Poisson)
    if !isempty(idx_else)
        θ_h_else = θ_home_all[idx_else]
        θ_a_else = θ_away_all[idx_else]
        
        Turing.@addlogprob! sum(logpdf.(LogPoisson.(θ_h_else), scores_else_x))
        Turing.@addlogprob! sum(logpdf.(LogPoisson.(θ_a_else), scores_else_y))
    end
    
    return nothing
end


# ==============================================================================
# 3. BUILDER
# ==============================================================================
function build_turing_model(model::GRWDixonColes, feature_set::FeatureSet)
    data = feature_set.data
    
    flat_home = data[:flat_home_goals]
    flat_away = data[:flat_away_goals]
    
    # --- Pre-processing: Group Matches by Score ---
    idx_00 = Int[]
    idx_10 = Int[]
    idx_01 = Int[]
    idx_11 = Int[]
    idx_else = Int[]

    for i in eachindex(flat_home)
        h, a = flat_home[i], flat_away[i]
        
        if h == 0 && a == 0
            push!(idx_00, i)
        elseif h == 1 && a == 0
            push!(idx_10, i)
        elseif h == 0 && a == 1
            push!(idx_01, i)
        elseif h == 1 && a == 1
            push!(idx_11, i)
        else
            push!(idx_else, i)
        end
    end

    scores_else_x = flat_home[idx_else]
    scores_else_y = flat_away[idx_else]

    return grw_dixon_coles_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:flat_home_ids],
        data[:flat_away_ids],
        data[:time_indices],
        # Groups
        idx_00, idx_10, idx_01, idx_11, idx_else,
        scores_else_x, scores_else_y,
        model
    )
end

# ==============================================================================
# 4. EXTRACT PARAMETERS (Prediction)
# ==============================================================================
"""
    extract_parameters(...)
Extracts θ (Log-Rates) and ρ (Dependence) for prediction.
"""
function extract_parameters(
    model::GRWDixonColes, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    # 1. Infer Rounds
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    # 2. Reconstruct History
    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    
    # 3. Extract Global Params
    γ_vec = vec(Array(chain[:γ]))
    μ_vec = vec(Array(chain[:μ])) 

    # Handle Rho
    if :ρ in names(chain)
        ρ_vec = vec(Array(chain[:ρ]))
    else
        ρ_raw_vec = vec(Array(chain[:ρ_raw]))
        ρ_vec = 0.3 .* tanh.(ρ_raw_vec)
    end

    # 4. Prediction Loop
    ExtractionValue = NamedTuple{(:θ_1, :θ_2, :θ_3), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]
        
        t = row.match_week 
        t_idx = clamp(t, 1, n_rounds)

        att_h = view(att_cube, h_id, t_idx, :)
        def_a = view(def_cube, a_id, t_idx, :)
        att_a = view(att_cube, a_id, t_idx, :)
        def_h = view(def_cube, h_id, t_idx, :)

        # Calculate Log-Rates (Theta)
        # θ = μ_global + γ + att + def
        θ_1 = μ_vec .+ att_h .+ def_a .+ γ_vec
        θ_2 = μ_vec .+ att_a .+ def_h
        θ_3 = ρ_vec 

        extraction_dict[Int(row.match_id)] = (; θ_1, θ_2, θ_3)
    end

    return extraction_dict
end

"""
    extract_trends(...)
"""
function extract_trends(model::GRWDixonColes, feature_set::FeatureSet, chain::Chains)
    n_teams = feature_set[:n_teams]
    team_map = feature_set[:team_map]
    
    step_names = names(group(chain, :z_att_steps))
    n_steps_raw = length(step_names)
    n_rounds = (n_steps_raw ÷ n_teams) + 1

    att_cube, def_cube = reconstruct_states(chain, n_teams, n_rounds)
    
    att_means = dropdims(mean(att_cube, dims=3), dims=3)
    def_means = dropdims(mean(def_cube, dims=3), dims=3)

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
