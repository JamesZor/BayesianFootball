# src/models/football_model.jl
using Turing

export FootballModel, football_model

struct FootballModel{D<:AbstractDynamics, P<:AbstractParameterization, O<:AbstractObservation}
    dynamics::D
    param::P
    obs::O
end

@model function football_model(
    n_teams, n_rounds, 
    home_ids, away_ids, time_indices, 
    goals_home, goals_away, 
    spec::FootballModel
)
    # --- 1. Parameters ---
    
    # A. Intercepts
    # Note: 'false' disables prefixing, so we get clean names like 'μ'
    lp_parts ~ to_submodel(sample_intercepts(spec.param, n_teams), false)
    μ = lp_parts.μ

    # B. Dynamics Scale (σ)
    # FIX: Use the accessor function instead of direct field access
    prior_σ = get_sigma_prior(spec.dynamics)
    σ_att ~ prior_σ
    σ_def ~ prior_σ

    # C. Dispersion (r)
    r ~ to_submodel(sample_dispersion(spec.obs), false)

    # --- 2. Latent States ---
    att_raw ~ sample_latent(spec.dynamics, n_teams, n_rounds)
    def_raw ~ sample_latent(spec.dynamics, n_teams, n_rounds)

    att_seq = apply_dynamics(spec.dynamics, att_raw, σ_att)
    def_seq = apply_dynamics(spec.dynamics, def_raw, σ_def)

    att = att_seq .- mean(att_seq, dims=1)
    def = def_seq .- mean(def_seq, dims=1)

    # --- 3. Likelihood Loop ---
    for i in 1:length(goals_home)
        h, a = home_ids[i], away_ids[i]
        t = time_indices[i]

        α_h = get_state_val(att, h, t)
        β_a = get_state_val(def, a, t)
        α_a = get_state_val(att, a, t)
        β_h = get_state_val(def, h, t)

        γ_h = get_hfa_val(spec.param, lp_parts, h)

        λ_h = exp(μ + γ_h + α_h + β_a)
        λ_a = exp(μ       + α_a + β_h)

        # FIX: Get the distribution object, then sample
        dist_h = make_observation_dist(spec.obs, λ_h, r)
        dist_a = make_observation_dist(spec.obs, λ_a, r)
        
        goals_home[i] ~ dist_h
        goals_away[i] ~ dist_a
    end
end
