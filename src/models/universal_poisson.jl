# src/models/universal_poisson.jl
using Turing

export UniversalPoisson

# 1. The Configuration Struct
Base.@kwdef struct UniversalPoisson{D<:AbstractDynamics, P<:AbstractParameterization}
    dynamics::D
    param_scheme::P
    
    # Global Priors
    μ_prior::Distribution = Normal(0.15, 0.5) 
    σ_k_prior::Distribution = Truncated(Normal(0, 1), 0, Inf)

    # Home Advantage Priors
    # For Standard: γ_prior is the prior for γ.
    # For Hierarchical: γ_prior is the prior for Mean(γ).
    γ_prior::Distribution = Normal(log(1.3), 0.2)
    
    # NEW: Prior for the variance of Home Advantage (Only used if Hierarchical)
    σ_γ_prior::Distribution = Truncated(Normal(0, 0.2), 0, Inf)
end

# 2. The Universal Model Block
@model function universal_poisson_model(
    n_teams, n_rounds, 
    home_ids, away_ids, time_indices, 
    goals_home, goals_away, 
    spec::UniversalPoisson
)
    # --- A. Global Parameters ---
    μ ~ spec.μ_prior
    σ_att ~ spec.σ_k_prior
    σ_def ~ spec.σ_k_prior

    # --- B. Home Advantage Logic (NEW) ---
    # 1. Sample the "Center" of home advantage
    γ_center ~ spec.γ_prior
    
    # 2. Sample the "Spread" (Even if unused by Standard, we sample it for simplicity)
    σ_γ ~ spec.σ_γ_prior 

    # 3. Sample Deviations (Dispatched!)

    dist_hfa = sample_hfa_deviations(spec.param_scheme, n_teams)
    if isnothing(dist_hfa)
        # If the model doesn't need deviations (Standard), set to 0.0 (Constant)
        γ_raw = 0.0 
    else
        # If the model needs them (Hierarchical), sample them!
        γ_raw ~ dist_hfa
    end


    # --- C. Team Strengths (Dispatched by Dynamics) ---
    att_raw ~ sample_latent(spec.dynamics, n_teams, n_rounds)
    def_raw ~ sample_latent(spec.dynamics, n_teams, n_rounds)

    att_seq = apply_dynamics(spec.dynamics, att_raw, σ_att)
    def_seq = apply_dynamics(spec.dynamics, def_raw, σ_def)

    # Center them
    att = att_seq .- mean(att_seq, dims=1)
    def = def_seq .- mean(def_seq, dims=1)

    # --- D. Likelihood Loop ---
    for i in 1:length(goals_home)
        h, a = home_ids[i], away_ids[i]
        t = time_indices[i]

        # 1. Get Team Strengths
        α_h = get_state_val(att, h, t)
        β_a = get_state_val(def, a, t)
        α_a = get_state_val(att, a, t)
        β_h = get_state_val(def, h, t)

        # 2. Get Home Advantage (Dispatched!)
        # Automatically handles Scalar vs Vector logic
        γ_h = get_hfa_val(spec.param_scheme, γ_center, σ_γ, γ_raw, h)

        # 3. Compute Rates
        λ_h = μ + γ_h + α_h + β_a
        λ_a = μ       + α_a + β_h

        goals_home[i] ~ LogPoisson(λ_h)
        goals_away[i] ~ LogPoisson(λ_a)
    end
end
