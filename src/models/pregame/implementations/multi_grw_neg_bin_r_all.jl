# src/models/pregame/implementations/multi_grw_neg_bin_r_all.jl

#=
Captures a midweek, month and 3g/plastic effect, 
Adds a dispersion parameters for Global, team, and month.

based on the multi grw negative binomial
=#


export MSNegativeBinomialRho

Base.@kwdef struct MSNegativeBinomialRho <: AbstractMultiScaledNegBinModel
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.5)

    # Standard priors for team strength
    γ::Distribution   = Normal(log(1.3), 0.2)
    
    # --- Global Dispersion ---
    log_r::Distribution = Normal(2.5, 0.5) 

    # --- Hierarchical Priors (Partial Pooling Standard Deviations) ---
    σ_r_team::Distribution = Truncated(Normal(0, 0.5), 0, Inf)
    σ_r_month::Distribution = Truncated(Normal(0, 0.5), 0, Inf)
    σ_δₘ::Distribution = Truncated(Normal(0, 0.1), 0, Inf)

    # --- NCP Standard Normal Priors ---
    z_r_team::Distribution = Normal(0, 1)
    z_r_month::Distribution = Normal(0, 1)
    z_δₘ::Distribution = Normal(0, 1)

    # --- Fixed Effects ---
    δₙ::Distribution = Normal(0, 0.1)  # \n week
    δₚ::Distribution = Normal(0, 0.1)  # \p plastic

    # --- Dynamic Hyperparameters (Process Noise) ---
    # Adjusted to 0.05 to prevent excessive volatility (factor of ~1.05 per week)
    σₖ::Distribution = Truncated(Normal(0, 0.05), 0, Inf) # micro  ( time Dynamic col - weeks / months)
    σₛ::Distribution = Truncated(Normal(0, 0.05), 0, Inf) # macro  ( seasons)
    
    # --- Initial State Hyperparameters (Hierarchical Prior t=0) ---
    σ₀::Distribution = Truncated(Normal(0.5, 0.2), 0, Inf)

    z₀::Distribution = Normal(0,1)
    zₖ::Distribution = Normal(0,1)
    zₛ::Distribution = Normal(0,1)
end


# ==============================================================================
# 2. MAIN TURING MODEL
# ==============================================================================

@model function multi_grw_neg_bin_model_train(
          n_teams,
          n_rounds,
          n_history,
          n_target,
          n_months,
          home_ids_flat,
          away_ids_flat,
          home_goals_flat,
          away_goals_flat,
          months_flat,
          is_midweek_flat,
          is_plastic_flat,
          time_indices, 
          model::MSNegativeBinomialRho,
          ::Type{T} = Float64 ) where {T} 

    # --------------------------------------------------------------------------
    # A. HYPERPARAMETERS
    # --------------------------------------------------------------------------
    μ ~ model.μ 
    γ ~ model.γ 

    # Global dispersion 
    log_r ~ model.log_r 

    # --- Team Dispersion (Partial Pooling + NCP + Sum-to-Zero) ---
    σ_r_team ~ model.σ_r_team
    z_r_team_raw ~ filldist(model.z_r_team, n_teams)
    raw_r_team = z_r_team_raw .* σ_r_team
    log_r_team = raw_r_team .- mean(raw_r_team)

    # --- Month Dispersion (Partial Pooling + NCP + Sum-to-Zero) ---
    σ_r_month ~ model.σ_r_month
    z_r_month_raw ~ filldist(model.z_r_month, n_months)
    raw_r_month = z_r_month_raw .* σ_r_month
    log_r_month = raw_r_month .- mean(raw_r_month)

    # --- Month Expectation Effect (Partial Pooling + NCP + Sum-to-Zero) ---
    σ_δₘ ~ model.σ_δₘ
    z_δₘ_raw ~ filldist(model.z_δₘ, n_months)
    raw_δₘ = z_δₘ_raw .* σ_δₘ
    δₘ = raw_δₘ .- mean(raw_δₘ)

    # --- 5. FIXED EFFECTS (No pooling needed for single parameters) ---
    δₙ ~ model.δₙ  # weeks 
    δₚ ~ model.δₚ # plastic effect

    # --------------------------------------------------------------------------
    # B. LATENT STATES (Using Submodels)
    # --------------------------------------------------------------------------
    
    # attacking strength 
    α ~ to_submodel(grw_two_step_component(
                        n_teams, n_rounds, n_history, n_target, 
                        model.z₀, model.zₛ, model.zₖ,
                        model.σ₀, model.σₛ, model.σₖ ) )
            
    # defence strength 
    β ~ to_submodel(grw_two_step_component(
                        n_teams, n_rounds, n_history, n_target, 
                        model.z₀, model.zₛ, model.zₖ,
                        model.σ₀, model.σₛ, model.σₖ ) )
            

    # --------------------------------------------------------------------------
    # C. LIKELIHOOD PIPELINE (Unchanged)
    # --------------------------------------------------------------------------
    

    αₕ = view(α, CartesianIndex.(home_ids_flat, time_indices))
    αₐ = view(α, CartesianIndex.(away_ids_flat, time_indices))
    βₐ = view(β, CartesianIndex.(away_ids_flat, time_indices))
    βₕ = view(β, CartesianIndex.(home_ids_flat, time_indices))

    δₘᵛ = view(δₘ, months_flat)

    log_r_team_home = view(log_r_team, home_ids_flat)  
    log_r_team_away = view(log_r_team, away_ids_flat)  
    log_r_months = view(log_r_month, months_flat)


    rₕ = exp.(log_r .+ log_r_months .+ log_r_team_home)
    rₐ = exp.(log_r .+ log_r_months .+ log_r_team_away)

    λₕ = exp.(μ .+ γ .+ αₕ .+ βₐ .+ δₘᵛ .+ ( δₙ .* is_midweek_flat) .+ ( δₚ .* is_plastic_flat ) ) 
    λₐ = exp.(μ .+      αₐ .+ βₕ .+ δₘᵛ .+ ( δₙ .* is_midweek_flat) .+ ( δₚ .* is_plastic_flat ) )


    home_goals_flat ~ arraydist(RobustNegativeBinomial.(rₕ, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(rₐ, λₐ))

end


function build_turing_model(model::MSNegativeBinomialRho, feature_set::FeatureSet)
    data = feature_set.data
    
    return multi_grw_neg_bin_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:n_history_steps]::Int,
        data[:n_target_steps]::Int,
        data[:n_months]::Int,
        data[:flat_home_ids],    
        data[:flat_away_ids],     
        data[:flat_home_goals],
        data[:flat_away_goals],
        data[:flat_months],
        data[:flat_is_midweek],
        data[:flat_is_plastic],
        data[:time_indices],
        model
    )
end



function extract_parameters(
    model::MSNegativeBinomialRho, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Get Context from the Feature Set
    n_teams = feature_set.data[:n_teams]
    n_rounds = feature_set.data[:n_rounds]
    n_history = feature_set.data[:n_history_steps]
    n_target = feature_set.data[:n_target_steps]
    team_map = feature_set.data[:team_map]
    
    # 2. Reconstruct the Latent States
    α = reconstruct_multiscale_submodel(chain, "α", n_teams, n_history, n_target)
    β = reconstruct_multiscale_submodel(chain, "β", n_teams, n_history, n_target)
    
    # --------------------------------------------------------------------------
    # 3. Extract Global Parameters & Reconstruct NCP Sum-to-Zero Parameters
    # --------------------------------------------------------------------------
    μ_v = vec(Array(chain[:μ]))
    γ_v = vec(Array(chain[:γ]))
    log_r_v = vec(Array(chain[:log_r]))
    δₙ = vec(Array(chain[:δₙ])) # weeks 
    δₚ = vec(Array(chain[:δₚ])) # plastic

    # Helper function to rebuild the Non-Centered, Sum-to-Zero parameters
    function reconstruct_ncp(param_z, param_σ, chain)
        z_raw = Array(group(chain, param_z)) # Shape: (n_samples, n_items)
        σ_vec = vec(Array(chain[param_σ]))   # Shape: (n_samples,)
        
        # Multiply z by σ (reshape σ to broadcast across columns)
        raw_val = z_raw .* reshape(σ_vec, :, 1)
        
        # Subtract the mean of each row to enforce sum-to-zero
        return raw_val .- mean(raw_val, dims=2)
    end

    # Reconstruct the dynamic arrays
    log_r_team  = reconstruct_ncp(:z_r_team_raw, :σ_r_team, chain)
    log_r_month = reconstruct_ncp(:z_r_month_raw, :σ_r_month, chain)
    δₘ          = reconstruct_ncp(:z_δₘ_raw, :σ_δₘ, chain)

    
    # --------------------------------------------------------------------------
    # 4. Predict
    # --------------------------------------------------------------------------
    results = Dict{Int64, NamedTuple}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = Int(row.match_id)
        
        # Resolve Time Index: 
        # For predicting unseen future matches, default to the last available round state.
        t = hasproperty(row, :time_index) ? row.time_index : n_rounds
        t_idx = clamp(t, 1, n_rounds)
        
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        month_idx  = Features.get_feature(Val(:month), row)
        is_mid     = Features.get_feature(Val(:midweek), row)
        is_plastic = Features.get_feature(Val(:is_plastic), row)

        # Get Team Strengths for this specific point in time
        α_h = α[h_id, t_idx, :]
        α_a = α[a_id, t_idx, :]
        β_h = β[h_id, t_idx, :]
        β_a = β[a_id, t_idx, :]

        # Extract month and team specific effects
        δₘᵛ = δₘ[:, month_idx]
        log_r_m = log_r_month[:, month_idx]
        log_r_h = log_r_team[:, h_id]
        log_r_a = log_r_team[:, a_id]

        # Calculate Lambda (Expected Goals)
        λ_h = exp.(μ_v .+ γ_v .+ α_h .+ β_a .+ δₘᵛ .+ (δₙ .* is_mid) .+ (δₚ .* is_plastic))
        λ_a = exp.(μ_v .+        α_a .+ β_h .+ δₘᵛ .+ (δₙ .* is_mid) .+ (δₚ .* is_plastic))

        # Calculate Match-Specific Dispersion (r)
        r_h = exp.(log_r_v .+ log_r_m .+ log_r_h)
        r_a = exp.(log_r_v .+ log_r_m .+ log_r_a)

        # Store the distributions needed to build the Negative Binomial PMF
        # Note: r is now separated into r_h and r_a
        results[mid] = (;
            λ_h = λ_h, 
            λ_a = λ_a,
            r_h = r_h,
            r_a = r_a
        )
    end

    return results
end
