# src/models/pregame/implementations/sequential_funnel.jl

using Turing
using LinearAlgebra
using Statistics
using Distributions
using ..MyDistributions 


export SequentialFunnelModel, funnel_model_train

# ==============================================================================
# 1. STRUCT DEFINITION (The Priors)
# ==============================================================================

Base.@kwdef struct SequentialFunnelModel <: AbstractFunnelModel
    
    # --- LAYER 1: CREATION (Shots) ---
    # Global Intercept for Shots (approx 2.5 on log scale is ~12 shots)
    creation_μ::Distribution = Normal(2.5, 0.3) 
    creation_home::Distribution = Normal(0.2, 0.1) # Home Advantage
    
    # Dynamics (GRW)
    # ( Gelman's Boundary Avoiding Priors 
    creation_σ_k::Distribution = Gamma(2, 0.05)
    creation_σ_0::Distribution = Gamma(2, 0.12)

    # Dispersion for NegBin (r) - Controls variance/clustering of shots
    log_r_create::Distribution = Normal(2.3, 0.5)


    # --- LAYER 2: PRECISION (Accuracy) ---
    # Logit Scale: 0.0 is 50%. -0.5 is ~38% accuracy.
    precision_μ::Distribution = Normal(-0.5, 0.2) 
    precision_home::Distribution = Normal(0.1, 0.1)

    # Dynamics
    precision_σ_k::Distribution = Gamma(2, 0.08)
    precision_σ_0::Distribution = Gamma(2, 0.08)


    # --- LAYER 3: CONVERSION (Finishing) ---
    # Logit Scale: -1.0 is ~27% conversion rate.
    conversion_μ::Distribution = Normal(-0.6, 0.3)
    conversion_home::Distribution = Normal(0.1, 0.1)

    # Dynamics
    conversion_σ_k::Distribution = Gamma(2, 0.08)
    conversion_σ_0::Distribution = Gamma(2, 0.2)
    
    # Latent Standard Normal
    z_dist::Distribution = Normal(0, 1)
end


# ==============================================================================
# 2. HELPER MODEL (The Submodel)
# ==============================================================================

@model function grw_component(n_t, n_r, σ_k, σ_0, z_dist)
    # These names (z_init, z_steps) will be prefixed by the caller
    # e.g., att_create.z_init
    z_init  ~ filldist(z_dist, n_t)
    z_steps ~ filldist(z_dist, n_t, n_r - 1)
    
    # Deterministic Transformation
    init   = z_init .* σ_0
    steps  = z_steps .* σ_k
    raw    = cumsum(hcat(init, steps), dims=2)
    
    # Center and Return
    centered = raw .- mean(raw, dims=1)
    return centered
end

# ==============================================================================
# 3. MAIN TURING MODEL
# ==============================================================================

@model function funnel_model_train(
    n_teams, n_rounds,
    home_ids, away_ids, time_indices,
    # Observed Data
    obs_shots_h, obs_shots_a,
    obs_sot_h,   obs_sot_a,
    obs_goals_h, obs_goals_a,
    # Config
    spec::SequentialFunnelModel
)
    # --------------------------------------------------------------------------
    # A. HYPERPARAMETERS
    # --------------------------------------------------------------------------
    
    # Layer 1: Creation
    μ_create    ~ spec.creation_μ
    γ_create    ~ spec.creation_home
    σ_create_k  ~ spec.creation_σ_k
    σ_create_0  ~ spec.creation_σ_0
    log_r_cr    ~ spec.log_r_create
    r_create    = exp(log_r_cr)
    
    # Layer 2: Precision
    μ_prec      ~ spec.precision_μ
    γ_prec      ~ spec.precision_home
    σ_prec_k    ~ spec.precision_σ_k
    σ_prec_0    ~ spec.precision_σ_0
    
    # Layer 3: Conversion
    μ_conv      ~ spec.conversion_μ
    γ_conv      ~ spec.conversion_home
    σ_conv_k    ~ spec.conversion_σ_k
    σ_conv_0    ~ spec.conversion_σ_0

    # --------------------------------------------------------------------------
    # B. LATENT STATES (Using Submodels)
    # --------------------------------------------------------------------------
    
    # Note: We pass spec.z_dist explicitly to the submodel
    
    # 1. Creation States
    # Traces will appear as: att_create.z_init, def_create.z_steps, etc.
    att_create ~ to_submodel(grw_component(n_teams, n_rounds, σ_create_k, σ_create_0, spec.z_dist))
    def_create ~ to_submodel(grw_component(n_teams, n_rounds, σ_create_k, σ_create_0, spec.z_dist))

    # 2. Precision States
    att_prec   ~ to_submodel(grw_component(n_teams, n_rounds, σ_prec_k,   σ_prec_0,   spec.z_dist))
    def_prec   ~ to_submodel(grw_component(n_teams, n_rounds, σ_prec_k,   σ_prec_0,   spec.z_dist))

    # 3. Conversion States
    att_conv   ~ to_submodel(grw_component(n_teams, n_rounds, σ_conv_k,   σ_conv_0,   spec.z_dist))
    def_conv   ~ to_submodel(grw_component(n_teams, n_rounds, σ_conv_k,   σ_conv_0,   spec.z_dist))

    # --------------------------------------------------------------------------
    # C. LIKELIHOOD PIPELINE (Unchanged)
    # --------------------------------------------------------------------------
    
    h_idx = CartesianIndex.(home_ids, time_indices)
    a_idx = CartesianIndex.(away_ids, time_indices)

    # --- LAYER 1: CREATION ---
    log_λ_shots_h = μ_create .+ γ_create .+ view(att_create, h_idx) .+ view(def_create, a_idx)
    log_λ_shots_a = μ_create             .+ view(att_create, a_idx) .+ view(def_create, h_idx)
    
    λ_shots_h = exp.(log_λ_shots_h)
    λ_shots_a = exp.(log_λ_shots_a)
    
    obs_shots_h ~ arraydist(RobustNegativeBinomial.(r_create, λ_shots_h))
    obs_shots_a ~ arraydist(RobustNegativeBinomial.(r_create, λ_shots_a))

    # --- LAYER 2: PRECISION ---
    logit_p_prec_h = μ_prec .+ γ_prec .+ view(att_prec, h_idx) .+ view(def_prec, a_idx)
    logit_p_prec_a = μ_prec           .+ view(att_prec, a_idx) .+ view(def_prec, h_idx)
    
    p_prec_h = logistic.(logit_p_prec_h)
    p_prec_a = logistic.(logit_p_prec_a)
    
    obs_sot_h ~ arraydist(Binomial.(obs_shots_h, p_prec_h))
    obs_sot_a ~ arraydist(Binomial.(obs_shots_a, p_prec_a))

    # --- LAYER 3: CONVERSION ---
    logit_p_conv_h = μ_conv .+ γ_conv .+ view(att_conv, h_idx) .+ view(def_conv, a_idx)
    logit_p_conv_a = μ_conv           .+ view(att_conv, a_idx) .+ view(def_conv, h_idx)
    
    p_conv_h = logistic.(logit_p_conv_h)
    p_conv_a = logistic.(logit_p_conv_a)
    
    obs_goals_h ~ arraydist(Binomial.(obs_sot_h, p_conv_h))
    obs_goals_a ~ arraydist(Binomial.(obs_sot_a, p_conv_a))
    
    return nothing
end

# ==============================================================================
# 3. BUILDER
# ==============================================================================

function build_turing_model(model::SequentialFunnelModel, feature_set::FeatureSet)
    data = feature_set.data
    
    return funnel_model_train(
        data[:n_teams]::Int,
        data[:n_rounds]::Int,
        data[:flat_home_ids],    
        data[:flat_away_ids],     
        data[:time_indices],
        
        # New Data Streams
        data[:flat_home_shots],
        data[:flat_away_shots],
        data[:flat_home_sot],
        data[:flat_away_sot],
        data[:flat_home_goals],
        data[:flat_away_goals],
        
        model
    )
end


# ==============================================================================
# 5. PARAMETER EXTRACTION (Robust Loop Version)
# ==============================================================================

"""
    reconstruct_submodel_robust(chain, prefix, σ_k_sym, σ_0_sym, n_teams, n_rounds)

Reconstructs GRW states by iterating variables one-by-one.
This avoids 'DimensionMismatch' and shape ambiguity with MCMCChains.Array().
"""
function reconstruct_submodel(chain, prefix, σ_k_sym, σ_0_sym, n_teams, n_rounds)
    # Calculate Total Samples (Samples * Chains)
    n_samples_per_chain, _, n_chains = size(chain)
    n_total = n_samples_per_chain * n_chains
    
    # 1. Extract Scalars (Flattened)
    σ_k_vec = vec(Array(chain[σ_k_sym]))
    σ_0_vec = vec(Array(chain[σ_0_sym]))
    
    # 2. Extract Init States [TotalSamples, Teams, 1]
    # We loop explicitly to handle 200x16 -> 3200 flattening consistently
    Z_init = zeros(Float64, n_total, n_teams, 1)
    
    for i in 1:n_teams
        sym = Symbol("$prefix.z_init[$i]")
        # vec(Array(...)) flattens [Samples, Chains] -> [TotalSamples]
        Z_init[:, i, 1] = vec(Array(chain[sym]))
    end
    
    # 3. Extract Steps [TotalSamples, Teams, Rounds-1]
    Z_steps = zeros(Float64, n_total, n_teams, n_rounds - 1)
    
    # Auto-detect naming format ([i,j] vs [i, j])
    sample_sym_space = Symbol("$prefix.z_steps[1, 1]")
    has_space = sample_sym_space in names(chain)
    
    for t in 1:(n_rounds - 1)
        for i in 1:n_teams
            # Construct symbol
            sym = has_space ? Symbol("$prefix.z_steps[$i, $t]") : Symbol("$prefix.z_steps[$i,$t]")
            # Flatten and Assign
            Z_steps[:, i, t] = vec(Array(chain[sym]))
        end
    end
    
    # 4. Apply GRW Scale
    # Reshape Sigmas for broadcasting: [TotalSamples, 1, 1]
    S_k = reshape(σ_k_vec, n_total, 1, 1)
    S_0 = reshape(σ_0_vec, n_total, 1, 1)
    
    init_scaled  = Z_init .* S_0
    steps_scaled = Z_steps .* S_k
    
    # 5. Integrate (Cumulative Sum over Time axis = 3)
    full_raw = cumsum(cat(init_scaled, steps_scaled, dims=3), dims=3)
    
    # 6. Center (Zero-Sum over Teams axis = 2)
    centered = full_raw .- mean(full_raw, dims=2)
    
    # Return: [Teams, Rounds, Samples] (Standard format for extraction loop)
    return permutedims(centered, (2, 3, 1))
end

function extract_parameters(
    model::SequentialFunnelModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Get Context
    n_teams = feature_set.data[:n_teams]
    n_rounds = feature_set.data[:n_rounds]
    team_map = feature_set.data[:team_map]
    
    # 2. Reconstruct Processes
    att_cr = reconstruct_submodel(chain, "att_create", :σ_create_k, :σ_create_0, n_teams, n_rounds)
    def_cr = reconstruct_submodel(chain, "def_create", :σ_create_k, :σ_create_0, n_teams, n_rounds)
    
    att_pr = reconstruct_submodel(chain, "att_prec", :σ_prec_k, :σ_prec_0, n_teams, n_rounds)
    def_pr = reconstruct_submodel(chain, "def_prec", :σ_prec_k, :σ_prec_0, n_teams, n_rounds)
    
    att_co = reconstruct_submodel(chain, "att_conv", :σ_conv_k, :σ_conv_0, n_teams, n_rounds)
    def_co = reconstruct_submodel(chain, "def_conv", :σ_conv_k, :σ_conv_0, n_teams, n_rounds)

    # 3. Extract Globals
    μ_cr_v = vec(Array(chain[:μ_create])); γ_cr_v = vec(Array(chain[:γ_create]))
    μ_pr_v = vec(Array(chain[:μ_prec]));   γ_pr_v = vec(Array(chain[:γ_prec]))
    μ_co_v = vec(Array(chain[:μ_conv]));   γ_co_v = vec(Array(chain[:γ_conv]))
    r_cre_v = exp.(vec(Array(chain[:log_r_cr])))

    # 4. Predict
    results = Dict{Int64, FunnelRates}()
    sizehint!(results, nrow(df))

    for row in eachrow(df)
        mid = row.match_id
        t = hasproperty(row, :time_index) ? row.time_index : row.match_week
        t_idx = clamp(t, 1, n_rounds)
        
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        # -- Layer 1: Creation --
        log_h_cr = μ_cr_v .+ γ_cr_v .+ att_cr[h_id, t_idx, :] .+ def_cr[a_id, t_idx, :]
        log_a_cr = μ_cr_v           .+ att_cr[a_id, t_idx, :] .+ def_cr[h_id, t_idx, :]
        λ_h = exp.(log_h_cr)
        λ_a = exp.(log_a_cr)

        # -- Layer 2: Precision --
        logit_h_pr = μ_pr_v .+ γ_pr_v .+ att_pr[h_id, t_idx, :] .+ def_pr[a_id, t_idx, :]
        logit_a_pr = μ_pr_v           .+ att_pr[a_id, t_idx, :] .+ def_pr[h_id, t_idx, :]
        θ_h = logistic.(logit_h_pr)
        θ_a = logistic.(logit_a_pr)

        # -- Layer 3: Conversion --
        logit_h_co = μ_co_v .+ γ_co_v .+ att_co[h_id, t_idx, :] .+ def_co[a_id, t_idx, :]
        logit_a_co = μ_co_v           .+ att_co[a_id, t_idx, :] .+ def_co[h_id, t_idx, :]
        ϕ_h = logistic.(logit_h_co)
        ϕ_a = logistic.(logit_a_co)

        # -- Expected Goals --
    #   FIX: do we need ?
        xg_h = λ_h .* θ_h .* ϕ_h 
        xg_a = λ_a .* θ_a .* ϕ_a

        results[mid] = (
            λ_shots_h = λ_h, λ_shots_a = λ_a, r_create = r_cre_v,
            θ_prec_h  = θ_h, θ_prec_a  = θ_a,
            ϕ_conv_h  = ϕ_h, ϕ_conv_a  = ϕ_a,
            exp_goals_h = xg_h, exp_goals_a = xg_a
        )
    end

    return results
end
