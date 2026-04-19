# src/models/pregame/implementations/ablation_study/nb_home_hierarchy.jl

#=
including Team-Specific Home Advantage
=#

export AblationStudy_NB_home_hierarchy

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================

Base.@kwdef struct AblationStudy_NB_home_hierarchy <: AbstractMultiScaledNegBinModel

    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.2)
    
    # Dispersion parameter (Negative Binomial)
    log_r::Distribution = Normal(2.5, 0.5)

    # --- HOME ADVANTAGE ---
    γ::Distribution        = Normal(0.2, 0.2)   # Global Baseline HA
    σ_γ_team::Distribution = Gamma(2, 0.05)     # Team-specific HA variance
    z_γ_team::Distribution = Normal(0, 1)
    
    # --- Time Dynamics ----
    σ₀::Distribution = Gamma(2, 0.15)   # Mean = 0.30 (Initial spread of teams)
    σₛ::Distribution = Gamma(2, 0.04)   # Mean = 0.08 (Macro season jump)
    σₖ::Distribution = Gamma(2, 0.015)   # Mean = 0.03 (Micro monthly jump)

    z₀::Distribution = Normal(0,1)
    zₛ::Distribution = Normal(0,1)
    zₖ::Distribution = Normal(0,1)

end

# ==============================================================================
# 2. MAIN TURING MODEL
# ==============================================================================

@model function model_train(
          n_teams, n_rounds, n_history, n_target, n_months,
          home_ids_flat, away_ids_flat, home_goals_flat, away_goals_flat,
          months_flat, is_midweek_flat, is_plastic_flat, time_indices, 
          model::AblationStudy_NB_home_hierarchy,
          ::Type{T} = Float64 ) where {T} 

    # --------------------------------------------------------------------------
    # A. HYPERPARAMETERS & POOLING
    # --------------------------------------------------------------------------
    μ ~ model.μ 
    
    # 1. Home Advantage
    γ_global ~ model.γ 
    γ_team ~ to_submodel(hierarchical_zero_centered_component(n_teams, model.σ_γ_team, model.z_γ_team))

    # 2. Dispersion (r)
    log_r ~ model.log_r 
    r = exp(log_r)

    # --------------------------------------------------------------------------
    # B. LATENT STATES (GRW)
    # --------------------------------------------------------------------------
    α ~ to_submodel(grw_two_step_component(
                        n_teams, n_rounds, n_history, n_target, 
                        model.z₀, model.zₛ, model.zₖ,
                        model.σ₀, model.σₛ, model.σₖ ) )
            
    β ~ to_submodel(grw_two_step_component(
                        n_teams, n_rounds, n_history, n_target, 
                        model.z₀, model.zₛ, model.zₖ,
                        model.σ₀, model.σₛ, model.σₖ ) )
            
    # --------------------------------------------------------------------------
    # C. LIKELIHOOD PIPELINE
    # --------------------------------------------------------------------------
    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))

    γ_team_v = view(γ_team, home_ids_flat)


    # Build final Expected Goals (λ) - Note the minus sign for defense!
    λₕ = exp.(μ .+ γ_global .+ γ_team_v .+ αₕ .+ βₐ )
    λₐ = exp.(μ .+                         αₐ .+ βₕ )

    home_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₐ))
end

# ==============================================================================
# 3. INTERFACE FUNCTIONS
# ==============================================================================

function build_turing_model(model::AblationStudy_NB_home_hierarchy, feature_set::FeatureSet)
    data = feature_set.data
    return model_train(
        data[:n_teams]::Int, data[:n_rounds]::Int, data[:n_history_steps]::Int,
        data[:n_target_steps]::Int, data[:n_months]::Int,
        data[:flat_home_ids], data[:flat_away_ids], data[:flat_home_goals],
        data[:flat_away_goals], data[:flat_months], data[:flat_is_midweek],
        data[:flat_is_plastic], # <-- Make sure this is in your feature set!
        data[:time_indices], model
    )
end

function extract_parameters(
    model::AblationStudy_NB_home_hierarchy, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    n_teams = feature_set.data[:n_teams]
    n_rounds = feature_set.data[:n_rounds]
    n_history = feature_set.data[:n_history_steps]
    n_target = feature_set.data[:n_target_steps]
    team_map = feature_set.data[:team_map]
    
    # Reconstruct Latent States
    α = reconstruct_multiscale_submodel(chain, "α", n_teams, n_history, n_target)
    β = reconstruct_multiscale_submodel(chain, "β", n_teams, n_history, n_target)
    
    # Reconstruct Hierarchical Components
    γ_team      = reconstruct_hierarchical_centered(chain, "γ_team")

    # Extract Globals
    μ_v = vec(Array(chain[:μ]))
    γ_global_v = vec(Array(chain[:γ_global]))
    log_r_v = vec(Array(chain[:log_r]))
    r_v = exp.(log_r_v)
    
    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        t = hasproperty(row, :time_index) ? row.time_index : n_rounds
        t_idx = clamp(t, 1, n_rounds)
        
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]


        # 1. Get specific slices for this match
        γ_team_v = γ_team[:, h_id]

        α_h = α[h_id, t_idx, :]
        α_a = α[a_id, t_idx, :]
        β_h = β[h_id, t_idx, :]
        β_a = β[a_id, t_idx, :]

        # 2. Calculate Final Expected Goals
        λ_h = exp.(μ_v .+ γ_global_v .+ γ_team_v .+ α_h .+ β_a )
        λ_a = exp.(μ_v .+                           α_a .+ β_h )

        results[mid] = (;
            λ_h = λ_h, 
            λ_a = λ_a,
            r = r_v,
        )
    end

    return results
end
