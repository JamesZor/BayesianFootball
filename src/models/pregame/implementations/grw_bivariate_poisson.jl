# src/models/pregame/implementations/grw_bivariate_poisson.jl

using Turing, Distributions, LinearAlgebra, Statistics
using ..MyDistributions 

export GRWBivariatePoisson

# ==============================================================================
# 1. THE STRUCT
# ==============================================================================
Base.@kwdef struct GRWBivariatePoisson <: AbstractDynamicBivariatePoissonModel 
      # --- Global Baseline (Intercept) ---
      # Represents the average log-goal rate for an away team.
      μ::Distribution = Normal(0.2, 0.5)

      # --- Static Parameters ---
      γ::Distribution = Normal(log(1.3), 0.2) # Home Advantage
      
      # Covariance (Static)
      # We model this as a single static parameter for the whole history.
      # Normal(-2, 1) implies median covariance rate ≈ 0.135
      ρ::Distribution = Normal(-2, 1.0) 

      # --- Dynamic Hyperparameters (Process Noise) ---
      # Standard deviation of the weekly random walk step
      σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf) 
      
      # --- Hierarchical Priors (t=0) ---
      # Initial Spread of abilities
      σ_0::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

      # --- Latent Variables (Random Walk) ---
      z_init::Distribution = Normal(0,1)
      z_steps::Distribution = Normal(0,1)
end

function Base.show(io::IO, ::MIME"text/plain", m::GRWBivariatePoisson)
    printstyled(io, "Dynamic Bivariate Poisson (GRW)\n", color=:magenta, bold=true)
    println(io, "  ├── Process Noise: $(m.σ_k)")
    println(io, "  ├── Initial Spread: $(m.σ_0)")
    println(io, "  └── Log-Covariance (Static): $(m.ρ)")
end

# ==============================================================================
# 2. THE TURING MODEL
# ==============================================================================
@model function grw_bivariate_poisson_train(
    n_teams, n_rounds, 
    flat_home_ids, flat_away_ids, 
    data_pairs,       # 2xN Matrix of goals
    time_indices,
    model::GRWBivariatePoisson,
    ::Type{T} = Float64
) where {T} 
    
    # --- 1. Hyperparameters ---
    # Global Intercept
    μ ~ model.μ
    
    # Home Advantage
    γ ~ model.γ
    
    # Static Log-Covariance (θ_3)
    ρ ~ model.ρ

    # Process Noise & Initial Spread
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k
    σ_att_0 ~ model.σ_0
    σ_def_0 ~ model.σ_0

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
    # θ_1 = μ_global + γ + att_h + def_a
    # θ_2 = μ_global + att_a + def_h
    # θ_3 = ρ (Static scalar)
    
    θ_1 = μ .+ γ .+ att_h_flat .+ def_a_flat 
    θ_2 = μ .+      att_a_flat .+ def_h_flat

    # --- 6. Likelihood ---
    # Pass Log-Rates directly to BivariateLogPoisson
    # ρ is broadcasted automatically
    data_pairs ~ arraydist(BivariateLogPoisson.(θ_1, θ_2, ρ))
    
    return nothing
end


# ==============================================================================
# 3. BUILDER
# ==============================================================================
function build_turing_model(model::GRWBivariatePoisson, feature_set::FeatureSet)
    data = feature_set.data
    
    # Prepare Data Matrix (2 x N_matches)
    # Row 1: Home Goals, Row 2: Away Goals
    flat_home = data[:flat_home_goals]
    flat_away = data[:flat_away_goals]
    data_matrix = permutedims(hcat(flat_home, flat_away))

    return grw_bivariate_poisson_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:flat_home_ids],
        data[:flat_away_ids],
        data_matrix,       # <-- Passed here
        data[:time_indices],
        model
    )
end

# ==============================================================================
# 4. EXTRACT PARAMETERS (Prediction)
# ==============================================================================
function extract_parameters(
    model::GRWBivariatePoisson, 
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
    ρ_vec = vec(Array(chain[:ρ])) # Static Log-Covariance

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
        # θ_1 = Home
        # θ_2 = Away
        # θ_3 = Static Covariance
        θ_1 = μ_vec .+ att_h .+ def_a .+ γ_vec
        θ_2 = μ_vec .+ att_a .+ def_h
        θ_3 = ρ_vec 

        extraction_dict[Int(row.match_id)] = (; θ_1, θ_2, θ_3)
    end

    return extraction_dict
end

# ==============================================================================
# 5. EXTRACT TRENDS (Analysis)
# ==============================================================================
function extract_trends(model::GRWBivariatePoisson, feature_set::FeatureSet, chain::Chains)
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
