# src/models/pregame/implementations/static_mvpln.jl

using Turing, Distributions, LinearAlgebra, Statistics

export StaticMVPLN

# --- 1. Struct ---
Base.@kwdef struct StaticMVPLN <: AbstractStaticMVPLNModel 
    μ::Distribution   = Normal(0, 10)       # Global Intercept
    γ::Distribution   = Normal(log(1.3), 0.2) # Home Advantage
    
    # Team Strength Heterogeneity (for α and β)
    σ_k::Distribution = Truncated(Cauchy(0, 5), 0, Inf) 
    
    # Residual Noise Heterogeneity (for the Log-Normal error terms)
    # This controls the diagonal of the covariance matrix Σ
    σ_ϵ::Distribution = Truncated(Normal(0, 1), 0, Inf)

    # Correlation between Home/Away performance (Latent correlation)
    # We sample in unconstrained space (atanh)
    ρ_raw::Distribution = Normal(0, 1) 

    # NCP Unit Normals
    Δₛ::Distribution  = Normal(0, 1) 
end

function Base.show(io::IO, ::MIME"text/plain", m::StaticMVPLN)
    printstyled(io, "Static Multivariate Poisson Log-Normal (MVPLN)\n", color=:cyan, bold=true)
    println(io, "  ├── Intercept:     $(m.μ)")
    println(io, "  ├── Home Adv:      $(m.γ)")
    println(io, "  ├── Team Var:      $(m.σ_k)")
    println(io, "  ├── Noise Scale:   $(m.σ_ϵ)")
    println(io, "  └── NCP Prior:     $(m.Δₛ)")
end


# --- 2. Model Definition ---
@model function static_mvpln_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticMVPLN)
    # Dimensions
    n_matches = length(home_goals)

    # --- A. Priors ---
    μ ~ model.μ
    γ ~ model.γ
    
    # Team Strength Variances
    σ_att ~ model.σ_k
    σ_def ~ model.σ_k

    # Log-Normal Noise Scale (Diagonal of Σ)
    # We can assume distinct scales for Home/Away errors, or shared. 
    # Here we assume shared scale for simplicity, or sample two:
    σ_h ~ model.σ_ϵ 
    σ_a ~ model.σ_ϵ

    # Correlation (Off-diagonal)
    ρ_raw ~ model.ρ_raw
    ρ = tanh(ρ_raw) # Constrain to (-1, 1)

    # --- B. Team Parameters (NCP) ---
    αₛ ~ filldist(model.Δₛ, n_teams)
    βₛ ~ filldist(model.Δₛ, n_teams)

    # Scaling & Centering (Sum-to-zero)
    α_raw = αₛ .* σ_att
    β_raw = βₛ .* σ_def
    α = α_raw .- mean(α_raw)
    β = β_raw .- mean(β_raw)

    # --- C. Latent Log-Normal Errors (The Copula Part) ---
    # We need a 2xN matrix of errors.
    # We use NCP: Sample Standard Normal, then transform via Cholesky L
    
    # 1. Sample standard normals for every match
    ϵ_raw ~ filldist(Normal(0, 1), 2, n_matches)

    # 2. Construct Cholesky Factor L such that Σ = L * L'
    # Σ = [σ_h²      ρ*σ_h*σ_a]
    #     [ρ*σ_h*σ_a      σ_a²]
    #
    # L = [σ_h             0          ]
    #     [ρ*σ_a     σ_a * sqrt(1-ρ²)]
    
    L11 = σ_h
    L21 = ρ * σ_a
    L22 = σ_a * sqrt(1 - ρ^2 + 1e-6) # jitter for stability

    # 3. Correlate the noise
    # We avoid constructing the full matrix L to keep it fast
    ϵ_home = L11 .* ϵ_raw[1, :] 
    ϵ_away = L21 .* ϵ_raw[1, :] .+ L22 .* ϵ_raw[2, :]

    # --- D. Likelihood ---
    # Latent Rate = Linear Predictor + Error
    # Note: We work in log-space.
    
    λ_home = exp.(μ .+ γ .+ α[home_ids] .+ β[away_ids] .+ ϵ_home)
    λ_away = exp.(μ      .+ α[away_ids] .+ β[home_ids] .+ ϵ_away)

    home_goals ~ arraydist(Poisson.(λ_home))
    away_goals ~ arraydist(Poisson.(λ_away))
end


# --- 3. Builder ---
function build_turing_model(model::StaticMVPLN, feature_set::FeatureSet)
    return static_mvpln_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        feature_set[:flat_home_goals],
        feature_set[:flat_away_goals],
        model
    )
end


# --- 4. Extractor ---
function extract_parameters(
    model::StaticMVPLN, 
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet, 
    chains::Chains
)::Dict{Int, NamedTuple}

    extraction_dict = Dict{Int, NamedTuple}()
    team_map = feature_set[:team_map]

    # --- A. Extract Global Parameters ---
    μ = vec(chains[:μ])
    γ = vec(chains[:γ])
    
    # Covariance Parameters
    σ_h = vec(chains[:σ_h])
    σ_a = vec(chains[:σ_a])
    ρ_raw = vec(chains[:ρ_raw])
    ρ = tanh.(ρ_raw)

    # --- B. Extract & Reconstruct Team Parameters ---
    σ_att = vec(chains[:σ_att])
    σ_def = vec(chains[:σ_def])
    αₛ = Array(group(chains, :αₛ))
    βₛ = Array(group(chains, :βₛ))

    α_mat = αₛ .* σ_att
    β_mat = βₛ .* σ_def
    
    # Centering
    α_c = α_mat .- mean(α_mat, dims=2)
    β_c = β_mat .- mean(β_mat, dims=2)

    # --- C. Prediction Loop ---
    # For MVPLN, we cannot return a single "Rate" λ because the rate is stochastic.
    # We return the "Linear Predictor" (Determininstic part) and the "Covariance" params.
    # The Prediction Module should use these to sample:
    #   z ~ N(0, Σ)
    #   λ = exp(LinearPred + z)
    
    for row in eachrow(df_to_predict)
        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        α_h = α_c[:, h_id]
        β_a = β_c[:, a_id]
        α_a = α_c[:, a_id]
        β_h = β_c[:, h_id]

        # Deterministic Log-Location (Linear Predictor)
        loc_h = μ .+ γ .+ α_h .+ β_a
        loc_a = μ      .+ α_a .+ β_h
        
        # We pack all parameters needed to construct the bivariate normal for this match
        extraction_dict[row.match_id] = (; 
            loc_h, # Vector of samples
            loc_a, # Vector of samples
            σ_h,   # Vector of samples
            σ_a,   # Vector of samples
            ρ      # Vector of samples
        )
    end

    return extraction_dict
end


#
# # src/models/pregame/implementations/static_mvpln.jl
# #=
# The Multivariate Poisson Log normal 
# =#
#
# export StaticMVPLN
#
# # --- 1. Struct ---
# Base.@kwdef struct StaticMVPLN <: AbstractStaticMVPLNModel 
#   κ::Distribution = Normal(0,10) # intercept 
#   γ::Distribution = Normal(log(1.3), 0.2) # home advantage
#   σₖ::Distribution = Gamma(2, 1/3) # for team strengths
#   σₜ::Distribution = Gamma(2, 1/3) # for correlations
#   Δₛ::Distribution = Normal(0,1) # Unit normal for NCP
#   ρᵣ::Distribution = Normal(0,1) # Unit normal for NCP ρ = tanh(ρᵣ) ∈ (-1,1)
#   ϵₛ::Distribution = MvNormal(zeros(2), I) # z score unit normal dim 2  - I is identity matrix for I₂
#
# end
#
# function Base.show(io::IO, ::MIME"text/plain", m::StaticMVPLN)
#     printstyled(io, "Static StaticMVPLN (NCP)\n", color=:green, bold=true)
#     println(io, "  ├── Intercept:     $(m.κ)")
#     println(io, "  ├── Home Adv:      $(m.γ)")
#     println(io, "  ├── Heterogeneity: $(m.σₖ)")
#     println(io, "  └── NCP Prior:     $(m.Δₛ)")
# end
#
#
# # --- 2. Model Definition ---
# @model function static_mvpln_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticMVPLN)
#
#     κ ~ model.κ        # intercept prior 
#     γ ~ model.γ        # home advantage
#     σₐ ~ model.σₖ  # attack parameters standard deviation 
#     σᵦ ~ model.σₖ  # defence parameters standard deviation 
#
#     # Non centered - z scores -"s-scores here"
#     αₛ ~ filldist(model.Δₛ, n_teams) 
#     βₛ ~ filldist(model.Δₛ, n_teams)
#
#     # Deterministic Transformation - Scaling 
#     αᵣ = αₛ .* σₐ 
#     βᵣ = βₛ .* σᵦ 
#
#     # sum-to-zero (STZ)
#     α = αᵣ .- mean(αᵣ)
#     β = βᵣ .- mean(βᵣ)
#
#     # 
#     σₕ ~ model.σₜ # home 
#     σᵥ ~ model.σₜ # away( v = visting)
#
#     ρᵣ ~ model.ρᵣ  # raw correlations coefficient 
#     ρ = tanh(ρᵣ)
#
#     ϵₛ ~ model.ϵₛ
#   MvNormal(zeros(2), I) # z score unit normal dim 2 
#
#     # manual Cholesky matrix  for Σ = LLᵀ
#     L = [ σₕ    ,        0         ;
#           ρ * σᵥ, σᵥ * √(1 - ρ^2 ) ]
#
#     ϵ = L * ϵₛ # matrix operation
#
#
#
#   home_goals ~ arraydist(LogPoisson.(α[home_ids] .+ β[away_ids] .+ μ .+ γ .+ ϵ[1]))
#   away_goals ~ arraydist(LogPoisson.(α[away_ids] .+ β[home_ids] .+ μ      .+ ϵ[2]))
#
# end
#
#
# # --- 3. Builder ---
# function build_turing_model(model::StaticMVPLN, feature_set::FeatureSet)
#     # Using dictionary syntax feature_set[:key]
#     return static_mvpln_model_train(
#         feature_set[:n_teams]::Int,
#         feature_set[:flat_home_ids],
#         feature_set[:flat_away_ids],
#         feature_set[:flat_home_goals],
#         feature_set[:flat_away_goals],
#         model
#     )
# end
#
#
#
# function extract_parameters(
#     model::StaticMVPLN, 
#     df_to_predict::AbstractDataFrame,
#     feature_set::FeatureSet, 
#     chains::Chains
# )::Dict{Int, MVPLNRates}
#
#
#
#
#     extraction_dict = Dict{Int64, PoissonRates}()
#     team_map = feature_set[:team_map]
#     n_teams = feature_set[:n_teams]
#
#     # --- A. Pre-Process Chains (The Fix) ---
#     # 1. Extract Global Parameters as vectors
#     κ = vec(chains[:κ])
#     γ = vec(chains[:γ])
#
#     σₐ = vec(chains[:σₐ])
#     σᵦ = vec(chains[:σᵦ])
#
#     αₛ = Array(group(chains, :αₛ))
#     βₛ = Array(group(chains, :βₛ))
#
#     # reconstruct 
#     α_matrix = αₛ .* σₐ   
#     β_matrix = βₛ .* σᵦ   
#
#
#     # Broadcast subtract to get the valid parameters
#     α_centered = α_matrix .- mean(α_matrix, dims=2)
#     β_centered = β_matrix .- mean(β_matrix, dims=2)
#
#     # --- B. Prediction Loop ---
#     for row in eachrow(df_to_predict)
#
#         h_id = team_map[row.home_team]
#         a_id = team_map[row.away_team]
#
#         # Extract the specific columns for these teams from our centered matrices
#         # Note: matrices are (Samples × Teams), so we grab column [:, h_id]
#         α_h = α_centered[:, h_id]
#         β_a = β_centered[:, a_id]
#         α_a = α_centered[:, a_id]
#         β_h = β_centered[:, h_id]
#
#         # Calculate Rates
#         λ_h = exp.(κ .+ γ .+ α_h .+ β_a)
#         λ_a = exp.(κ      .+ α_a .+ β_h)
#
#         extraction_dict[row.match_id] = (; λ_h, λ_a)
#     end
#     return extraction_dict
# end
#
