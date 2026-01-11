# src/models/pregame/implementations/static_mixture_copula.jl

using Turing, Distributions, LinearAlgebra
include("../components/copula_densities.jl") # Load the densities

export StaticMixtureCopula

Base.@kwdef struct StaticMixtureCopula <: AbstractStaticMixCopulaModel 
    μ::Distribution   = Normal(0, 10)
    γ::Distribution   = Normal(log(1.3), 0.2)
    σ_k::Distribution = Truncated(Cauchy(0,5), 0, Inf) 
    σ_ϵ::Distribution = Truncated(Normal(0, 1), 0, Inf)
    Δₛ::Distribution  = Normal(0, 1)
end

@model function static_mixture_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticMixtureCopula)
    n_matches = length(home_goals)

    # --- A. Global Params ---
    μ ~ model.μ
    γ ~ model.γ
    σ_att ~ model.σ_k; σ_def ~ model.σ_k
    σ_h ~ model.σ_ϵ;   σ_a ~ model.σ_ϵ

    # --- B. Mixture Weights ---
    # We have 3 components: [1=Gaussian, 2=Clayton, 3=Frank]
    # Dirichlet(1,1,1) is uniform (agnostic) prior
    w ~ Dirichlet(2, 1.0) 

    # --- C. Component Parameters ---
    # 1. Gaussian Rho: (-1, 1)
    # ρ_raw ~ Normal(0, 1)
    # ρ_gauss = tanh(ρ_raw)
    #
    # 2. Clayton Theta: (0, Inf) - Reparameterized
    θ_clay_log ~ Normal(0.5, 0.5) 
    θ_clay = exp(θ_clay_log)

    # 3. Frank Theta: (-Inf, Inf) - Can be negative!
    θ_frank ~ Normal(0, 2) 

    # --- D. Team Skills ---
    αₛ ~ filldist(model.Δₛ, n_teams)
    βₛ ~ filldist(model.Δₛ, n_teams)
    α = (αₛ .* σ_att) .- mean(αₛ .* σ_att)
    β = (βₛ .* σ_def) .- mean(βₛ .* σ_def)

    # --- E. Latent Error "Mixture" ---
    # We treat epsilon as parameters to be inferred
    ϵ_h_raw ~ filldist(Normal(0, 1), n_matches)
    ϵ_a_raw ~ filldist(Normal(0, 1), n_matches)

    # Apply the Mixture Copula Prior to these errors manually
    # The "Raw" epsilons are technically independent Normals in the prior statement above.
    # We must RE-WEIGHT their probability based on the Copula Mixture.
    
    # Pre-compute transformed Uniforms (u, v)
    u_vec = cdf.(Normal(0,1), ϵ_h_raw)
    v_vec = cdf.(Normal(0,1), ϵ_a_raw)
    
    for i in 1:n_matches
        u, v = u_vec[i], v_vec[i]

        # Calculate log-density for each component
        # ld_1 = log_c_gaussian(u, v, ρ_gauss)
        ld_2 = log_c_clayton(u, v, θ_clay)
        ld_3 = log_c_frank(u, v, θ_frank)

        # Mixture Density: log( w1*exp(ld1) + w2*exp(ld2) + w3*exp(ld3) )
        # We use LogExpFunctions.logsumexp for stability
        # We assume the base density is Uniform (since we transformed via CDF), 
        # so the Copula Density IS the likelihood adjustment.
        
        log_mix_pdf = logsumexp([
            # log(w[1]) + ld_1,
            log(w[1]) + ld_2,
            log(w[2]) + ld_3
        ])
        
        # Add to the target log-probability
        Turing.@addlogprob! log_mix_pdf
    end

    # --- F. Likelihood (Rates) ---
    # Scale the errors
    λ_h = μ .+ γ .+ α[home_ids] .+ β[away_ids] .+ σ_h .* ϵ_h_raw
    λ_a = μ      .+ α[away_ids] .+ β[home_ids] .+ σ_a .* ϵ_a_raw

    home_goals ~ arraydist(LogPoisson.(λ_h))
    away_goals ~ arraydist(LogPoisson.(λ_a))
end



# --- 3. Builder ---
function build_turing_model(model::StaticMixtureCopula, feature_set::FeatureSet)
    return static_mixture_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        feature_set[:flat_home_goals],
        feature_set[:flat_away_goals],
        model
    )
end

