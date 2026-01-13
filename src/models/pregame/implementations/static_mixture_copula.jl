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

function extract_parameters(
    model::StaticMixtureCopula, 
    df_to_predict::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chains::Chains
)::Dict{Int, NamedTuple}

    extraction_dict = Dict{Int, NamedTuple}()
    team_map = feature_set[:team_map]

    # --- A. Extract Global Parameters ---
    μ = vec(chains[:μ])
    γ = vec(chains[:γ])
    
    # Noise Scales
    σ_h = vec(chains[:σ_h])
    σ_a = vec(chains[:σ_a])

    # --- B. Extract Mixture & Copula Parameters ---
    # Weights: w is a Dirichlet vector. In chains, typically stored as w[1], w[2].
    # We group them into a matrix (Samples x 2) and convert to vector of vectors.
    w_mat = Array(group(chains, :w)) 
    w = [w_mat[i, :] for i in 1:size(w_mat, 1)] # Vector of [w1, w2] vectors

    # Copula Params
    # 1. Clayton: Model samples θ_clay_log ~ Normal. We need θ_clay = exp(θ_clay_log)
    θ_clay_log = vec(chains[:θ_clay_log])
    θ_clay = exp.(θ_clay_log)

    # 2. Frank: Model samples θ_frank ~ Normal.
    θ_frank = vec(chains[:θ_frank])

    # --- C. Extract & Reconstruct Team Parameters ---
    σ_att = vec(chains[:σ_att])
    σ_def = vec(chains[:σ_def])
    αₛ = Array(group(chains, :αₛ))
    βₛ = Array(group(chains, :βₛ))

    # Scale
    α_mat = αₛ .* σ_att
    β_mat = βₛ .* σ_def
    
    # Center (Sum-to-zero constraint)
    α_c = α_mat .- mean(α_mat, dims=2)
    β_c = β_mat .- mean(β_mat, dims=2)

    # --- D. Prediction Loop ---
    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        α_h = α_c[:, h_id]
        β_a = β_c[:, a_id]
        α_a = α_c[:, a_id]
        β_h = β_c[:, h_id]

        # Deterministic Log-Location (Linear Predictor)
        # Note: These exclude the epsilon noise terms which are integrated out later
        loc_h = μ .+ γ .+ α_h .+ β_a
        loc_a = μ      .+ α_a .+ β_h
        
        # Pack everything needed for the score matrix computation
        extraction_dict[row.match_id] = (; 
            loc_h, 
            loc_a, 
            σ_h, 
            σ_a,
            w,       # Vector of Vectors
            θ_clay,  # Vector
            θ_frank  # Vector
        )
    end

    return extraction_dict
end
