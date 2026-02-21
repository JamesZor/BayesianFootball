# src/models/pregame/implementations/multi_grw_neg_bin.jl


using Turing, Distributions, DataFrames
using ..MyDistributions 
using LinearAlgebra
using Statistics


export MSNegativeBinomial 

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================

Base.@kwdef struct MSNegativeBinomial <: AbstractMultiScaledNegBinModel
    # --- Global Baseline (Intercept) ---
    # Represents the average log-goal rate for an away team.
    μ::Distribution = Normal(0.2, 0.5)

    # Standard priors for team strength
    γ::Distribution   = Normal(log(1.3), 0.2)
    
    # Dispersion parameter (Negative Binomial)
    log_r::Distribution = Normal(2.5, 0.5) 

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
# 2. HELPER MODEL (The Submodel)
# ==============================================================================

@model function grw_two_step_component(
        number_teams::Real,
        number_rounds::Real,
        number_history_steps::Real,
        number_target_steps::Real,
        z₀::Distribution,
        zₛ::Distribution,
        zₖ::Distribution,
        dist_σ₀::Distribution,
        dist_σₛ::Distribution,
        dist_σₖ::Distribution,
      )
    
    # These names (z_init, z_steps) will be prefixed by the caller
    σ₀ ~ dist_σ₀
    σₛ ~ dist_σₛ
    σₖ ~ dist_σₖ

    # e.g., att_create.z_init
    z_init  ~ filldist(z₀, number_teams)
    z_season_steps ~ filldist(zₛ, number_teams, number_history_steps )
    z_target_steps ~ filldist(zₖ, number_teams, number_target_steps - 1 )
    
    # Deterministic Transformation
    init   = z_init .* σ₀
    seaosn_steps  = z_season_steps .* σₛ
    target_steps  = z_target_steps .* σₖ
    
    raw    = cumsum(
                hcat(
                  init,
                  seaosn_steps,
                  target_steps),
              dims=2)
    
    # Center and Return
    centered = raw .- mean(raw, dims=1)
    return centered
end


# ==============================================================================
# 3. MAIN TURING MODEL
# ==============================================================================

@model function multi_grw_neg_bin_model_train(
          n_teams,
          n_rounds,
          n_history,
          n_target,
          home_ids_flat,
          away_ids_flat,
          home_goals_flat,
          away_goals_flat,
          time_indices, 
          model::MSNegativeBinomial,
          ::Type{T} = Float64 ) where {T} 

    # --------------------------------------------------------------------------
    # A. HYPERPARAMETERS
    # --------------------------------------------------------------------------
    μ ~ model.μ 
    γ ~ model.γ 
    log_r ~ model.log_r 
    r = exp(log_r)


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

    λₕ = exp.(μ .+ γ .+ αₕ .+ βₐ)
    λₐ = exp.(μ .+      αₐ .+ βₕ)


    home_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₕ))
    away_goals_flat ~ arraydist(RobustNegativeBinomial.(r, λₐ))

end


function build_turing_model(model::MSNegativeBinomial, feature_set::FeatureSet)
    data = feature_set.data
    
    return multi_grw_neg_bin_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:n_history_steps]::Int,
        data[:n_target_steps]::Int,
        data[:flat_home_ids],    
        data[:flat_away_ids],     
        data[:flat_home_goals],
        data[:flat_away_goals],
        data[:time_indices],
        model
    )
end





          
"""
    reconstruct_multiscale_submodel(chain, prefix, n_teams, n_history, n_target)

Reconstructs the two-speed GRW states (macro + micro steps) by iterating variables safely.
"""
function reconstruct_multiscale_submodel(chain, prefix::String, n_teams::Int, n_history::Int, n_target::Int)
    n_samples_per_chain, _, n_chains = size(chain)
    n_total = n_samples_per_chain * n_chains
    
    # 1. Extract the three Variances (Flattened and Reshaped for Broadcasting)
    S_0 = reshape(vec(Array(chain[Symbol("$prefix.σ₀")])), n_total, 1, 1)
    S_s = reshape(vec(Array(chain[Symbol("$prefix.σₛ")])), n_total, 1, 1)
    S_k = reshape(vec(Array(chain[Symbol("$prefix.σₖ")])), n_total, 1, 1)
    
    # Helper to safely grab symbols whether Turing used spaces or not
    function get_sym(base_name, idx...)
        str1 = "$base_name[$(join(idx, ", "))]"
        str2 = "$base_name[$(join(idx, ","))]"
        return Symbol(str1) in names(chain) ? Symbol(str1) : Symbol(str2)
    end
    
    # 2. Extract Initial States [Samples, Teams, 1]
    Z_init = zeros(Float64, n_total, n_teams, 1)
    for i in 1:n_teams
        Z_init[:, i, 1] = vec(Array(chain[get_sym("$prefix.z_init", i)]))
    end
    
    # 3. Extract Macro (Season) Steps [Samples, Teams, n_history]
    Z_season = zeros(Float64, n_total, n_teams, n_history)
    for t in 1:n_history
        for i in 1:n_teams
            Z_season[:, i, t] = vec(Array(chain[get_sym("$prefix.z_season_steps", i, t)]))
        end
    end
    
    # 4. Extract Micro (Target) Steps [Samples, Teams, n_target - 1]
    Z_target = zeros(Float64, n_total, n_teams, n_target - 1)
    for t in 1:(n_target - 1)
        for i in 1:n_teams
            Z_target[:, i, t] = vec(Array(chain[get_sym("$prefix.z_target_steps", i, t)]))
        end
    end
    
    # 5. Apply the Variances to the Standard Normals
    init_scaled   = Z_init .* S_0
    season_scaled = Z_season .* S_s
    target_scaled = Z_target .* S_k
    
    # 6. Stitch the timeline together along the Time axis (dim 3)
    raw_steps = cat(init_scaled, season_scaled, target_scaled, dims=3)
    
    # 7. Integrate the walk (Cumulative Sum over Time axis)
    full_raw = cumsum(raw_steps, dims=3)
    
    # 8. Center around 0 (Zero-Sum over Teams axis = 2)
    centered = full_raw .- mean(full_raw, dims=2)
    
    # Return in standard format: [Teams, Time, Samples]
    return permutedims(centered, (2, 3, 1))
end


function extract_parameters(
    model::MSNegativeBinomial, 
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

        # Get Team Strengths for this specific point in time
        α_h = α[h_id, t_idx, :]
        α_a = α[a_id, t_idx, :]
        β_h = β[h_id, t_idx, :]
        β_a = β[a_id, t_idx, :]

        # Calculate Lambda (Expected Goals)
        λ_h = exp.(μ_v .+ γ_v .+ α_h .+ β_a)
        λ_a = exp.(μ_v .+        α_a .+ β_h)

        # Store the distributions needed to build the Negative Binomial PMF
        results[mid] = (;
            λ_h = λ_h, 
            λ_a = λ_a,
            r = r_v
        )
    end

    return results
end
