using Statistics # Required for the mean() function

#=
Ablation Study: Baseline + Home/Away Dispersion.
Tests if separating the Negative Binomial dispersion parameter (r) 
for home and away teams adds predictive power over the vanilla baseline.
=#

export AblationStudy_NB_baseline_HA_r

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================
Base.@kwdef struct AblationStudy_NB_baseline_HA_r <: AbstractMultiScaledNegBinModel
    # --- Global Baseline ---
    μ::Distribution         = Normal(0.2, 0.2)
    γ::Distribution         = Normal(0.12, 0.5)
    
    # --- Dispersion Parameters ---
    log_r::Distribution    = Normal(2.5, 0.5)      # Base dispersion (Away)
    δ_r_home::Distribution = Normal(0.0, 0.5)      # Delta shift for Home dispersion

    # --- Time Dynamics (Latent States) ---
    σ₀::Distribution = Gamma(2, 0.15)
    σₛ::Distribution = Gamma(2, 0.04)
    σₖ::Distribution = Gamma(2, 0.015)

    z₀::Distribution = Normal(0, 1)
    zₛ::Distribution = Normal(0, 1)
    zₖ::Distribution = Normal(0, 1)
end

# ==============================================================================
# 2. MAIN TURING MODEL
# ==============================================================================
@model function model_train(
          n_teams, n_rounds, n_history, n_target,
          home_ids_flat, away_ids_flat, home_goals_flat, away_goals_flat,
          time_indices, 
          model::AblationStudy_NB_baseline_HA_r,
          ::Type{T} = Float64 ) where {T} 

    μ ~ model.μ 
    γ ~ model.γ 
    
    # Dispersion priors
    log_r ~ model.log_r 
    δ_r_home ~ model.δ_r_home

    # Calculate separated r values
    r_a = exp(log_r)
    r_h = exp(log_r + δ_r_home)

    # Latent team strengths
    α ~ to_submodel(grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.σ₀, model.σₛ, model.σₖ))
    β ~ to_submodel(grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.σ₀, model.σₛ, model.σₖ))

    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))

    # Standard Baseline Lambdas
    λₕ = exp.(μ .+ γ .+ αₕ .+ βₐ)
    λₐ = exp.(μ .+      αₐ .+ βₕ)

    # Apply the specific r values to home and away targets
    home_goals_flat ~ arraydist(RobustNegativeBinomial.(r_h, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(r_a, λₐ))
end

# ==============================================================================
# 3. INTERFACE FUNCTIONS
# ==============================================================================
function build_turing_model(model::AblationStudy_NB_baseline_HA_r, feature_set::FeatureSet)
    data = feature_set.data
    return model_train(
        data[:n_teams]::Int, data[:n_rounds]::Int, data[:n_history_steps]::Int,
        data[:n_target_steps]::Int,
        data[:flat_home_ids], data[:flat_away_ids], data[:flat_home_goals],
        data[:flat_away_goals], data[:time_indices], model
    )
end

function extract_parameters(model::AblationStudy_NB_baseline_HA_r, df::AbstractDataFrame, feature_set::FeatureSet, chain::Chains)
    n_teams, n_rounds, n_history, n_target = feature_set.data[:n_teams], feature_set.data[:n_rounds], feature_set.data[:n_history_steps], feature_set.data[:n_target_steps]
    team_map = feature_set.data[:team_map]
    
    α = reconstruct_multiscale_submodel(chain, "α", n_teams, n_history, n_target)
    β = reconstruct_multiscale_submodel(chain, "β", n_teams, n_history, n_target)
    
    μ_v        = vec(Array(chain[:μ]))
    γ_v        = vec(Array(chain[:γ]))
    log_r_v    = vec(Array(chain[:log_r]))
    δ_r_home_v = vec(Array(chain[:δ_r_home]))
    
    # Calculate global separated r vectors for the chains
    r_a_v = exp.(log_r_v)
    r_h_v = exp.(log_r_v .+ δ_r_home_v)

    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        t_idx = clamp(hasproperty(row, :time_index) ? row.time_index : n_rounds, 1, n_rounds)
        h_id, a_id = team_map[row.home_team], team_map[row.away_team]
        
        # Standard Lambdas
        λ_h = exp.(μ_v .+ γ_v .+ α[h_id, t_idx, :] .+ β[a_id, t_idx, :])
        λ_a = exp.(μ_v .+        α[a_id, t_idx, :] .+ β[h_id, t_idx, :])

        home_alpha = mean(α[h_id, t_idx, :])
        away_alpha = mean(α[a_id, t_idx, :])
        home_beta  = mean(β[h_id, t_idx, :])
        away_beta  = mean(β[a_id, t_idx, :])

        # Return both r_h and r_a individually for downstream PMF building
        results[mid] = (; λ_h, λ_a, r_h = r_h_v, r_a = r_a_v, home_alpha, away_alpha, home_beta, away_beta)
    end
    return results
end
