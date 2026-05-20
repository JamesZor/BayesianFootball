# src/models/pregame/implementations/ablation_study/nb_bloated.jl

#=
The "Kitchen Sink" / Bloated Model.
Includes EVERYTHING: Base + Home Hierarchy + Team Dispersion + 
Environmental Effects (Midweek/Plastic) + Monthly Dispersion & Expectations.
=#

export AblationStudy_NB_KitchenSink

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================

Base.@kwdef struct AblationStudy_NB_KitchenSink <: AbstractMultiScaledNegBinModel
    # --- Global Baseline (Intercept) ---
    μ::Distribution = Normal(0.2, 0.5)

    # --- HOME ADVANTAGE ---
    γ::Distribution        = Normal(0.2, 0.2)   # Global Baseline HA
    σ_γ_team::Distribution = Gamma(2, 0.05)     # Team-specific HA variance
    z_γ_team::Distribution = Normal(0, 1)

    # --- DISPERSION (Negative Binomial) ---
    log_r::Distribution     = Normal(2.5, 0.5)  # Global Baseline Dispersion
    σ_r_team::Distribution  = Gamma(2, 0.10)    # Team-specific variance
    z_r_team::Distribution  = Normal(0, 1)
    σ_r_month::Distribution = Gamma(2, 0.05)    # Month-specific variance
    z_r_month::Distribution = Normal(0, 1)

    # --- TIME EFFECTS (Expectation) ---
    σ_δₘ::Distribution = Gamma(2, 0.05)         # Month Expectation variance
    z_δₘ::Distribution = Normal(0, 1)
    
    # --- ENVIRONMENTAL EFFECTS ---
    δₙ::Distribution = Normal(0, 0.1)           # Midweek effect
    δₚ::Distribution = Normal(0, 0.1)           # Plastic pitch effect

    # --- LATENT STATES (Process Noise) ---
    σₖ::Distribution = Truncated(Normal(0, 0.05), 0, Inf) # micro (weeks)
    σₛ::Distribution = Truncated(Normal(0, 0.05), 0, Inf) # macro (seasons)
    σ₀::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf) # initial spread

    z₀::Distribution = Normal(0, 1)
    zₖ::Distribution = Normal(0, 1)
    zₛ::Distribution = Normal(0, 1)
end

# ==============================================================================
# 2. MAIN TURING MODEL
# ==============================================================================

@model function model_train(
          n_teams, n_rounds, n_history, n_target, n_months,
          home_ids_flat, away_ids_flat, home_goals_flat, away_goals_flat,
          months_flat, is_midweek_flat, is_plastic_flat, time_indices, 
          model::AblationStudy_NB_KitchenSink,
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
    log_r_team ~ to_submodel(hierarchical_zero_centered_component(n_teams, model.σ_r_team, model.z_r_team)) 
    log_r_month ~ to_submodel(hierarchical_zero_centered_component(n_months, model.σ_r_month, model.z_r_month)) 

    # 3. Fixed / Time Effects
    δₘ ~ to_submodel(hierarchical_zero_centered_component(n_months, model.σ_δₘ, model.z_δₘ)) 
    δₙ ~ model.δₙ  
    δₚ ~ model.δₚ 

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
    δₘᵛ = view(δₘ, months_flat)

    log_r_team_home = view(log_r_team, home_ids_flat)  
    log_r_team_away = view(log_r_team, away_ids_flat)  
    log_r_months = view(log_r_month, months_flat)

    # Build final r values
    rₕ = exp.(log_r .+ log_r_months .+ log_r_team_home)
    rₐ = exp.(log_r .+ log_r_months .+ log_r_team_away)

    # Build final Expected Goals (λ) 
    # FIXED: Standardized back to .+ β to match ablation baseline
    λₕ = exp.(μ .+ γ_global .+ γ_team_v .+ αₕ .+ βₐ .+ δₘᵛ .+ (δₙ .* is_midweek_flat) .+ (δₚ .* is_plastic_flat))
    λₐ = exp.(μ .+                         αₐ .+ βₕ .+ δₘᵛ .+ (δₙ .* is_midweek_flat) .+ (δₚ .* is_plastic_flat))

    home_goals_flat ~ arraydist(RobustNegativeBinomial.(rₕ, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(rₐ, λₐ))
end

# ==============================================================================
# 3. INTERFACE FUNCTIONS
# ==============================================================================

function build_turing_model(model::AblationStudy_NB_KitchenSink, feature_set::FeatureSet)
    data = feature_set.data
    return model_train(
        data[:n_teams]::Int, data[:n_rounds]::Int, data[:n_history_steps]::Int,
        data[:n_target_steps]::Int, data[:n_months]::Int,
        data[:flat_home_ids], data[:flat_away_ids], data[:flat_home_goals],
        data[:flat_away_goals], data[:flat_months], data[:flat_is_midweek],
        data[:flat_is_plastic], 
        data[:time_indices], model
    )
end

function extract_parameters(
    model::AblationStudy_NB_KitchenSink, 
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
    log_r_team  = reconstruct_hierarchical_centered(chain, "log_r_team")
    log_r_month = reconstruct_hierarchical_centered(chain, "log_r_month")
    δₘ          = reconstruct_hierarchical_centered(chain, "δₘ")

    # Extract Globals
    μ_v = vec(Array(chain[:μ]))
    γ_global_v = vec(Array(chain[:γ_global]))
    log_r_v = vec(Array(chain[:log_r]))
    δₙ = vec(Array(chain[:δₙ]))
    δₚ = vec(Array(chain[:δₚ]))
    
    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        t = hasproperty(row, :time_index) ? row.time_index : n_rounds
        t_idx = clamp(t, 1, n_rounds)
        
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        month_idx = Features.get_feature(Val(:month), row)
        is_mid    = Features.get_feature(Val(:midweek), row)
        is_plast  = Features.get_feature(Val(:is_plastic), row) 

        # 1. Get specific slices for this match
        δₘᵛ = δₘ[:, month_idx]
        γ_team_v = γ_team[:, h_id]

        log_r_team_h_v = log_r_team[:, h_id]
        log_r_team_a_v = log_r_team[:, a_id]
        log_r_month_v  = log_r_month[:, month_idx]

        α_h = α[h_id, t_idx, :]
        α_a = α[a_id, t_idx, :]
        β_h = β[h_id, t_idx, :]
        β_a = β[a_id, t_idx, :]

        # 2. Calculate Final Expected Goals (FIXED: .- β to .+ β)
        λ_h = exp.(μ_v .+ γ_global_v .+ γ_team_v .+ α_h .+ β_a .+ δₘᵛ .+ (δₙ .* is_mid) .+ (δₚ .* is_plast))
        λ_a = exp.(μ_v .+                           α_a .+ β_h .+ δₘᵛ .+ (δₙ .* is_mid) .+ (δₚ .* is_plast))

        # 3. Calculate Final Dispersion
        r_h = exp.(log_r_v .+ log_r_month_v .+ log_r_team_h_v)
        r_a = exp.(log_r_v .+ log_r_month_v .+ log_r_team_a_v)

        results[mid] = (;
            λ_h = λ_h, 
            λ_a = λ_a,
            r_h = r_h,
            r_a = r_a
        )
    end

    return results
end
