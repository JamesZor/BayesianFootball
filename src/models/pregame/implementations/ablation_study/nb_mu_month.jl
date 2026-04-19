# src/models/pregame/implementations/ablation_study/nb_baseline_month_mu.jl

#=
Ablation Study: Baseline + Monthly Expectations.
Tests if a hierarchical monthly shift on Expected Goals (μ) adds predictive power
over the vanilla baseline.
=#

export AblationStudy_NB_baseline_month_mu

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================
Base.@kwdef struct AblationStudy_NB_baseline_month_mu <: AbstractMultiScaledNegBinModel
    # --- Global Baseline ---
    μ::Distribution     = Normal(0.2, 0.2)
    γ::Distribution     = Normal(0.12, 0.5)
    log_r::Distribution = Normal(2.5, 0.5)

    # --- TIME EFFECTS (Expectation ONLY) ---
    σ_δₘ::Distribution = Gamma(2, 0.05)         # Month Expectation variance
    z_δₘ::Distribution = Normal(0, 1)

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
          n_teams, n_rounds, n_history, n_target, n_months,
          home_ids_flat, away_ids_flat, home_goals_flat, away_goals_flat,
          months_flat, time_indices, 
          model::AblationStudy_NB_baseline_month_mu,
          ::Type{T} = Float64 ) where {T} 

    μ ~ model.μ 
    γ ~ model.γ 
    log_r ~ model.log_r 
    r = exp(log_r)

    # Monthly Expectation Shift
    δₘ ~ to_submodel(hierarchical_zero_centered_component(n_months, model.σ_δₘ, model.z_δₘ)) 

    α ~ to_submodel(grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.σ₀, model.σₛ, model.σₖ))
    β ~ to_submodel(grw_two_step_component(n_teams, n_rounds, n_history, n_target, model.z₀, model.zₛ, model.zₖ, model.σ₀, model.σₛ, model.σₖ))

    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))

    δₘᵛ = view(δₘ, months_flat)

    # Baseline + Monthly Expectation
    λₕ = exp.(μ .+ γ .+ αₕ .+ βₐ .+ δₘᵛ)
    λₐ = exp.(μ .+      αₐ .+ βₕ .+ δₘᵛ)

    home_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₐ))
end

# ==============================================================================
# 3. INTERFACE FUNCTIONS
# ==============================================================================
function build_turing_model(model::AblationStudy_NB_baseline_month_mu, feature_set::FeatureSet)
    data = feature_set.data
    return model_train(
        data[:n_teams]::Int, data[:n_rounds]::Int, data[:n_history_steps]::Int,
        data[:n_target_steps]::Int, data[:n_months]::Int,
        data[:flat_home_ids], data[:flat_away_ids], data[:flat_home_goals],
        data[:flat_away_goals], data[:flat_months], data[:time_indices], model
    )
end

function extract_parameters(model::AblationStudy_NB_baseline_month_mu, df::AbstractDataFrame, feature_set::FeatureSet, chain::Chains)
    n_teams, n_rounds, n_history, n_target = feature_set.data[:n_teams], feature_set.data[:n_rounds], feature_set.data[:n_history_steps], feature_set.data[:n_target_steps]
    team_map = feature_set.data[:team_map]
    
    α  = reconstruct_multiscale_submodel(chain, "α", n_teams, n_history, n_target)
    β  = reconstruct_multiscale_submodel(chain, "β", n_teams, n_history, n_target)
    δₘ = reconstruct_hierarchical_centered(chain, "δₘ")

    μ_v     = vec(Array(chain[:μ]))
    γ_v     = vec(Array(chain[:γ]))
    log_r_v = vec(Array(chain[:log_r]))
    r_v     = exp.(log_r_v)
    
    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        t_idx = clamp(hasproperty(row, :time_index) ? row.time_index : n_rounds, 1, n_rounds)
        h_id, a_id = team_map[row.home_team], team_map[row.away_team]
        month_idx = Features.get_feature(Val(:month), row)

        δₘᵛ = δₘ[:, month_idx]
        
        λ_h = exp.(μ_v .+ γ_v .+ α[h_id, t_idx, :] .+ β[a_id, t_idx, :] .+ δₘᵛ)
        λ_a = exp.(μ_v .+        α[a_id, t_idx, :] .+ β[h_id, t_idx, :] .+ δₘᵛ)

        results[mid] = (; λ_h, λ_a, r = r_v)
    end
    return results
end
