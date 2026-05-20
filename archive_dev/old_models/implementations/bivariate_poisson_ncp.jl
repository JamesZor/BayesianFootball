# # src/models/pregame/implementations/bivariate_poisson_ncp.jl

using Turing, Distributions, LinearAlgebra, Statistics
# Ensure your custom distribution module is loaded
using ..MyDistributions 

export BivariatePoissonNCP

# --- 1. Struct ---
Base.@kwdef struct BivariatePoissonNCP <: AbstractStaticBivariatePoissonModel 
    μ::Distribution   = Normal(0, 10) 
    γ::Distribution   = Normal(log(1.3), 0.2) 
    σ_k::Distribution = Truncated(Cauchy(0,5), 0, Inf) 
    Δₛ::Distribution  = Normal(0, 1)    
    
    # NEW: Prior for the LOG-Covariance.
    # Normal(-2, 1) implies the median covariance rate is exp(-2) ≈ 0.135
    # This replaces the LogNormal prior since we are now in log-space.
    ρ::Distribution = Normal(-2, 1.0) 
end 

function Base.show(io::IO, ::MIME"text/plain", m::BivariatePoissonNCP)
    printstyled(io, "Static Bivariate Poisson (Log-Space Implementation)\n", color=:magenta, bold=true)
    println(io, "  ├── Heterogeneity: $(m.σ_k)")
    println(io, "  └── Log-Cov Prior: $(m.ρ)")
end

# --- 2. Model Definition ---
@model function bivariate_poisson_model_train(n_teams, home_ids, away_ids, data_pairs, model::BivariatePoissonNCP)

    # A. Global Parameters
    μ  ~ model.μ        
    γ  ~ model.γ        
    σₐ ~ model.σ_k  
    σᵦ ~ model.σ_k  
    
    # Sample Log-Covariance directly (Theta 3)
    ρ ~ model.ρ

    # B. Team Parameters (NCP)
    αₛ ~ filldist(model.Δₛ, n_teams) 
    βₛ ~ filldist(model.Δₛ, n_teams)

    # Scaling & Centering
    αᵣ = αₛ .* σₐ 
    βᵣ = βₛ .* σᵦ 
    α = αᵣ .- mean(αᵣ)
    β = βᵣ .- mean(βᵣ)

    # C. Log-Rate Calculation (Thetas)
    # We do NOT exponentiate here. We pass log-rates directly to the distribution.
    # This avoids overflow issues automatically.
    θ_home_vec = α[home_ids] .+ β[away_ids] .+ μ .+ γ
    θ_away_vec = α[away_ids] .+ β[home_ids] .+ μ      
    
    
    data_pairs ~ arraydist(BivariateLogPoisson.(θ_home_vec, θ_away_vec, ρ))
end

# --- 3. Builder ---
function build_turing_model(model::BivariatePoissonNCP, feature_set::FeatureSet)
    flat_home = feature_set[:flat_home_goals]
    flat_away = feature_set[:flat_away_goals]
    
    data_matrix = permutedims(hcat(flat_home, flat_away))

    return bivariate_poisson_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        data_matrix, # <-- Passing a 2xN Matrix
        model
    )
end

# --- 4. Extractor ---
function extract_parameters(
    model::BivariatePoissonNCP, 
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet, 
    chains::Chains
)::Dict{Int, NamedTuple}

    extraction_dict = Dict{Int, NamedTuple}()
    team_map = feature_set[:team_map]

    # Extract Global Params
    μ = vec(chains[:μ])
    γ = vec(chains[:γ])
    ρ = vec(chains[:ρ]) # This is log(λ3)
    
    # Extract Variance & Team Params
    σₐ_vec = vec(chains[:σₐ])
    σᵦ_vec = vec(chains[:σᵦ])
    αₛ_mat = Array(group(chains, :αₛ))
    βₛ_mat = Array(group(chains, :βₛ))

    # Reconstruct NCP
    α_mat = αₛ_mat .* σₐ_vec
    β_mat = βₛ_mat .* σᵦ_vec
    α_c = α_mat .- mean(α_mat, dims=2)
    β_c = β_mat .- mean(β_mat, dims=2)

    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        α_h = α_c[:, h_id]
        β_a = β_c[:, a_id]
        α_a = α_c[:, a_id]
        β_h = β_c[:, h_id]

        # We STOP here. No exp().
        θ_1 = μ .+ γ .+ α_h .+ β_a
        θ_2 = μ      .+ α_a .+ β_h
        θ_3 = ρ # Shared log-covariance

        # Return Thetas directly
        extraction_dict[row.match_id] = (; θ_1, θ_2, θ_3)
    end
    return extraction_dict
end
