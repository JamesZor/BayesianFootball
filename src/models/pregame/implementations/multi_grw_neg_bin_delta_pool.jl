# src/models/pregame/implementations/multi_grw_neg_bin_delta.jl

#=
Captures a midweek and month effect, based on the multi grw negative binomial
=#

export MSNegativeBinomialDeltaPool

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================

Base.@kwdef struct MSNegativeBinomialDeltaPool <: AbstractMultiScaledNegBinModel
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.5)

    # Standard priors for team strength
    γ::Distribution   = Normal(log(1.3), 0.2)
    
    # Dispersion parameter (Negative Binomial)
    log_r::Distribution = Normal(2.5, 0.5) 

    # Month m 
    z_δₘ::Distribution = Normal(0,1)
    σ_δₘ::Distribution = Truncated(Normal(0, 0.5), 0, Inf) 

    
    δₙ::Distribution = Normal(0, 0.1)  # \n week

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
          time_indices, 
          model::MSNegativeBinomialDeltaPool,
          ::Type{T} = Float64 ) where {T} 

    # --------------------------------------------------------------------------
    # A. HYPERPARAMETERS
    # --------------------------------------------------------------------------
    μ ~ model.μ 
    γ ~ model.γ 
    log_r ~ model.log_r 
    r = exp(log_r)


    # months pooling 
    δₘ ~ to_submodel(hierarchical_zero_centered_component(n_months, model.σ_δₘ, model.z_δₘ)) 
    δₙ ~ model.δₙ  # weeks 

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

    λₕ = exp.(μ .+ γ .+ αₕ .+ βₐ .+ δₘᵛ .+ ( δₙ .* is_midweek_flat) ) 
    λₐ = exp.(μ .+      αₐ .+ βₕ .+ δₘᵛ .+ ( δₙ .* is_midweek_flat) )


    home_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₐ))

end


function build_turing_model(model::MSNegativeBinomialDelta, feature_set::FeatureSet)
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
        data[:time_indices],
        model
    )
end

function extract_parameters(
    model::MSNegativeBinomialDelta, 
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
    
    # 3. Extract Global Parameters
    μ_v = vec(Array(chain[:μ]))
    γ_v = vec(Array(chain[:γ]))
    log_r_v = vec(Array(chain[:log_r]))
    r_v = exp.(log_r_v)  # Dispersion parameter for NegBin
    δₙ = vec(Array(chain[:δₙ]))
    δₘ = reconstruct_hierarchical_centered(chain, "δₘ")
    
    # 4. Predict
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
        δₘᵛ = δₘ[:, month_idx]
      

        # Get Team Strengths for this specific point in time
        α_h = α[h_id, t_idx, :]
        α_a = α[a_id, t_idx, :]
        β_h = β[h_id, t_idx, :]
        β_a = β[a_id, t_idx, :]


        # Calculate Lambda (Expected Goals)
        λ_h = exp.(μ_v .+ γ_v .+ α_h .+ β_a .+ δₘᵛ .+ (δₙ .* is_mid ))
        λ_a = exp.(μ_v .+        α_a .+ β_h .+ δₘᵛ .+ (δₙ .* is_mid ))

        # Store the distributions needed to build the Negative Binomial PMF
        results[mid] = (;
            λ_h = λ_h, 
            λ_a = λ_a,
            r = r_v
        )
    end

    return results
end
