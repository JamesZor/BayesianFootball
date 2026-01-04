# # src/models/pregame/implementations/bivariate_poisson_ncp.jl
#
# using Turing, Copulas, Distributions, LinearAlgebra
#
# export BivariatePoissonNCP
#
# # --- 1. Struct ---
# Base.@kwdef struct BivariatePoissonNCP <: AbstractStaticPoissonModel 
#     μ::Distribution   = Normal(0, 10) 
#     γ::Distribution   = Normal(log(1.3), 0.2) 
#     σ_k::Distribution = Truncated(Cauchy(0,5), 0, Inf) 
#     Δₛ::Distribution  = Normal(0, 1)    # Unit Normal for NCP 
#     ρ::Distribution   = Uniform(-0.9, 0.9) # Correlation (Must be < 1.0)
# end 
#
# function Base.show(io::IO, ::MIME"text/plain", m::BivariatePoissonNCP)
#     printstyled(io, "Static Bivariate Hierarchical Poisson (NCP)\n", color=:magenta, bold=true)
#     println(io, "  ├── Heterogeneity: $(m.σ_k)")
#     println(io, "  ├── Correlation:   $(m.ρ)")
#     println(io, "  └── Structure:     Gaussian Copula w/ Poisson Marginals")
# end
#
# # Helper if you don't have it defined elsewhere
# # LogPoisson(x) = Poisson(exp(x)) 
# # (But Turing prefers explicit distributions usually, so we use Poisson(exp(.)))
#
# # --- 2. Model Definition ---
# @model function bivariate_poisson_model_train(n_teams, home_ids, away_ids, data_pairs, model::BivariatePoissonNCP)
#
#     # A. Global Parameters
#     μ  ~ model.μ        
#     γ  ~ model.γ        
#     σₐ ~ model.σ_k  
#     σᵦ ~ model.σ_k  
#     ρ  ~ model.ρ  
#
#     # B. Team Parameters (NCP)
#     αₛ ~ filldist(model.Δₛ, n_teams) 
#     βₛ ~ filldist(model.Δₛ, n_teams)
#
#     # Scaling (Deterministic)
#     αᵣ = αₛ .* σₐ 
#     βᵣ = βₛ .* σᵦ 
#
#     # Sum-to-zero
#     α = αᵣ .- mean(αᵣ)
#     β = βᵣ .- mean(βᵣ)
#
#     # C. Rate Calculation (Vectorized)
#     # Calculate log-rates for all matches at once
#     log_λₕ = α[home_ids] .+ β[away_ids] .+ μ .+ γ 
#     log_λₐ = α[away_ids] .+ β[home_ids] .+ μ      
#
#     # D. Copula Likelihood
#     # We construct the correlation matrix ONCE per sample to save time
#     # Note: Cholesky stability can be an issue if ρ is exactly 1 or -1
#     Σ = [1.0 ρ; ρ 1.0]
#
#     # We use a Gaussian Copula
#     # Note: Copulas.jl + Discrete Marginals + HMC is computationally very heavy.
#     my_copula = GaussianCopula(Σ)
#
#     # We must map over the data points. 
#     # Since each match has unique lambda parameters, each match is a unique distribution.
#     # We assume 'data_pairs' is a Vector{Vector{Float64}} or Vector{Vector{Int}}
#     match_dists = map(1:length(home_ids)) do i
#         # Define the marginals for this specific match
#         # We use Poisson(exp(...)) directly
#         m_h = Poisson(exp(log_λₕ[i]))
#         m_a = Poisson(exp(log_λₐ[i]))
#
#         # Combine via Sklar
#         SklarDist(my_copula, (m_h, m_a))
#     end
#
#     # Observe the joint pairs
#     data_pairs ~ arraydist(match_dists)
# end
#
# # --- 3. Builder ---
# function build_turing_model(model::BivariatePoissonNCP, feature_set::FeatureSet)
#     # We need to ZIP the goals into pairs [ [h1, a1], [h2, a2]... ]
#     flat_home = feature_set[:flat_home_goals]
#     flat_away = feature_set[:flat_away_goals]
#
#     # Turing requires the data to be in the format the distribution expects.
#     # For multivariate distributions, this is usually a Vector of Vectors.
#     data_pairs = [ [flat_home[i], flat_away[i]] for i in 1:length(flat_home) ]
#
#     return bivariate_poisson_model_train(
#         feature_set[:n_teams]::Int,
#         feature_set[:flat_home_ids],
#         feature_set[:flat_away_ids],
#         data_pairs, # <--- NEW INPUT FORMAT
#         model
#     )
# end
#
# # --- 4. Extractor ---
# # The extractor logic remains mostly the same, but now you implicitly have correlation.
# # The `λ_h` and `λ_a` are the MARGINAL rates.
# # If you want to simulate matches properly later, you must sample using the Copula and Rho.
# function extract_parameters(
#     model::BivariatePoissonNCP, 
#     df_to_predict::AbstractDataFrame,
#     feature_set::FeatureSet, 
#     chains::Chains
# )::Dict{Int, NamedTuple} # Updated return type to generic NamedTuple
#
#     extraction_dict = Dict{Int, NamedTuple}()
#     team_map = feature_set[:team_map]
#
#     # Extract Parameters
#     μ = vec(chains[:μ])
#     γ = vec(chains[:γ])
#     ρ = vec(chains[:ρ]) # <--- Capture Correlation
#
#     σₐ_vec = vec(chains[:σₐ])
#     σᵦ_vec = vec(chains[:σᵦ])
#
#     αₛ_mat = Array(group(chains, :αₛ))
#     βₛ_mat = Array(group(chains, :βₛ))
#
#     # NCP Reconstruction
#     α_mat = αₛ_mat .* σₐ_vec
#     β_mat = βₛ_mat .* σᵦ_vec
#
#     α_c = α_mat .- mean(α_mat, dims=2)
#     β_c = β_mat .- mean(β_mat, dims=2)
#
#     for row in eachrow(df_to_predict)
#         h_id = team_map[row.home_team]
#         a_id = team_map[row.away_team]
#
#         α_h = α_c[:, h_id]
#         β_a = β_c[:, a_id]
#         α_a = α_c[:, a_id]
#         β_h = β_c[:, h_id]
#
#         # Calculate MARGINAL rates
#         λ_h = exp.(μ .+ γ .+ α_h .+ β_a)
#         λ_a = exp.(μ      .+ α_a .+ β_h)
#
#         # We return ρ as well so the simulation engine knows how to couple them
#         extraction_dict[row.match_id] = (; λ_h, λ_a, ρ)
#     end
#     return extraction_dict
# end

# src/models/pregame/implementations/bivariate_poisson_ncp.jl

using Turing, Distributions, LinearAlgebra, Statistics

export BivariatePoissonNCP

# --- 1. Struct ---
Base.@kwdef struct BivariatePoissonNCP <: AbstractStaticPoissonModel 
    μ::Distribution   = Normal(0, 10) 
    γ::Distribution   = Normal(log(1.3), 0.2) 
    σ_k::Distribution = Truncated(Cauchy(0,5), 0, Inf) 
    Δₛ::Distribution  = Normal(0, 1)    
    λ₃_prior::Distribution = LogNormal(-2, 1.0) # Covariance prior
end 

function Base.show(io::IO, ::MIME"text/plain", m::BivariatePoissonNCP)
    printstyled(io, "Static Bivariate Poisson (Karlis-Ntzoufras)\n", color=:magenta, bold=true)
    println(io, "  ├── Heterogeneity: $(m.σ_k)")
    println(io, "  └── Covariance:    $(m.λ₃_prior)")
end

# --- 2. Model Definition ---
@model function bivariate_poisson_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::BivariatePoissonNCP)

    # A. Global Parameters
    μ  ~ model.μ        
    γ  ~ model.γ        
    σₐ ~ model.σ_k  
    σᵦ ~ model.σ_k  
    λ₃_raw ~ model.λ₃_prior

    # B. Team Parameters (NCP)
    αₛ ~ filldist(model.Δₛ, n_teams) 
    βₛ ~ filldist(model.Δₛ, n_teams)

    # Scaling 
    αᵣ = αₛ .* σₐ 
    βᵣ = βₛ .* σᵦ 

    # Sum-to-zero
    α = αᵣ .- mean(αᵣ)
    β = βᵣ .- mean(βᵣ)

    # C. Rate Calculation
    # We clamp the exponent to prevent Infinity/NaN if the model explores wild areas
    log_λ1 = clamp.(α[home_ids] .+ β[away_ids] .+ μ .+ γ, -20, 20)
    log_λ2 = clamp.(α[away_ids] .+ β[home_ids] .+ μ,      -20, 20)

    λ₁ = exp.(log_λ1)
    λ₂ = exp.(log_λ2)
    
    # D. Likelihood (Bivariate Poisson)
    # We use a custom logpdf that relies ONLY on Distributions.jl
    Turing.@addlogprob! bivariate_poisson_logpdf(home_goals, away_goals, λ₁, λ₂, λ₃_raw)
end

# --- 3. The Custom Log Likelihood Function (Robust Version) ---
function bivariate_poisson_logpdf(x, y, λ₁, λ₂, λ₃)
    log_prob = 0.0
    
    # We pre-allocate the Poisson distribution for the covariance term
    d3 = Poisson(λ₃)

    for i in 1:length(x)
        # SAFELY convert goals to Integers (handles 2.0 -> 2)
        xi = round(Int, x[i]) 
        yi = round(Int, y[i])
        
        # Create distributions for the independent parts
        d1 = Poisson(λ₁[i])
        d2 = Poisson(λ₂[i])

        # Karlis-Ntzoufras: P(x,y) = Sum_{k=0...min(x,y)} P(X=x-k)P(Y=y-k)P(Z=k)
        k_max = min(xi, yi)
        
        # Accumulate log-sum-exp terms
        # Initialize with the k=0 term
        current_max = logpdf(d1, xi) + logpdf(d2, yi) + logpdf(d3, 0)
        total_sum_exp = 0.0 # We will do log-sum-exp logic manually

        # If k_max is 0 (one team scored 0), we don't need the loop
        if k_max == 0
            log_prob += current_max
            continue
        end

        # Vector to store log terms for LSE
        log_terms = Vector{Float64}(undef, k_max + 1)
        log_terms[1] = current_max

        for k in 1:k_max
            val = logpdf(d1, xi - k) + 
                  logpdf(d2, yi - k) + 
                  logpdf(d3, k)
            log_terms[k+1] = val
        end
        
        # Perform LogSumExp
        max_val = maximum(log_terms)
        log_prob += max_val + log(sum(exp.(v - max_val) for v in log_terms))
    end
    
    return log_prob
end

# --- 4. Builder ---
function build_turing_model(model::BivariatePoissonNCP, feature_set::FeatureSet)
    return bivariate_poisson_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        feature_set[:flat_home_goals],
        feature_set[:flat_away_goals],
        model
    )
end

# --- 5. Extractor ---
function extract_parameters(
    model::BivariatePoissonNCP, 
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet, 
    chains::Chains
)::Dict{Int, NamedTuple}

    extraction_dict = Dict{Int, NamedTuple}()
    team_map = feature_set[:team_map]

    μ = vec(chains[:μ])
    γ = vec(chains[:γ])
    λ₃ = vec(chains[:λ₃_raw])
    
    σₐ_vec = vec(chains[:σₐ])
    σᵦ_vec = vec(chains[:σᵦ])
    αₛ_mat = Array(group(chains, :αₛ))
    βₛ_mat = Array(group(chains, :βₛ))

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

        # Calculate INDEPENDENT rates
        λ_1 = exp.(μ .+ γ .+ α_h .+ β_a)
        λ_2 = exp.(μ      .+ α_a .+ β_h)
        
        # Return components so we can simulate later (Sample X~Pois(λ1), Y~Pois(λ2), Z~Pois(λ3))
        extraction_dict[row.match_id] = (; λ_1, λ_2, λ_3=λ₃)
    end
    return extraction_dict
end
