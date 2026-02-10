# src/models/pregame/implementations/sequential_funnel.jl

using Turing
using LinearAlgebra
using Statistics
using Distributions

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
    creation_σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf) 
    creation_σ_0::Distribution = Truncated(Normal(0.2, 0.1), 0, Inf)

    # Dispersion for NegBin (r) - Controls variance/clustering of shots
    creation_r::Distribution = Gamma(10, 1) 


    # --- LAYER 2: PRECISION (Accuracy) ---
    # Logit Scale: 0.0 is 50%. -0.5 is ~38% accuracy.
    precision_μ::Distribution = Normal(-0.5, 0.2) 
    precision_home::Distribution = Normal(0.1, 0.1)

    # Dynamics
    precision_σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
    precision_σ_0::Distribution = Truncated(Normal(0.1, 0.05), 0, Inf)


    # --- LAYER 3: CONVERSION (Finishing) ---
    # Logit Scale: -1.0 is ~27% conversion rate.
    conversion_μ::Distribution = Normal(-1.0, 0.2)
    conversion_home::Distribution = Normal(0.1, 0.1)

    # Dynamics
    conversion_σ_k::Distribution = Truncated(Normal(0, 0.05), 0, Inf)
    conversion_σ_0::Distribution = Truncated(Normal(0.1, 0.05), 0, Inf)
    
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
    r_create    ~ spec.creation_r
    
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
    
    obs_shots_h ~ arraydist(NegativeBinomial.(r_create, r_create ./ (r_create .+ λ_shots_h)))
    obs_shots_a ~ arraydist(NegativeBinomial.(r_create, r_create ./ (r_create .+ λ_shots_a)))

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
