# src/models/pregame/implementations/static_hierarchical_poisson_NCP.jl


export StaticHierarchicalPoissonNCP

# --- 1. Struct ---
Base.@kwdef struct StaticHierarchicalPoissonNCP <: AbstractStaticPoissonModel 
    μ::Distribution   = Normal(0, 10) 
    γ::Distribution   = Normal(log(1.3), 0.2) 
    σ_k::Distribution = Truncated(Cauchy(0,5), 0, Inf) 
    Δₛ::Distribution  = Normal(0, 1) # Unit Normal for NCP 
end

function Base.show(io::IO, ::MIME"text/plain", m::StaticHierarchicalPoissonNCP)
    printstyled(io, "Static Hierarchical Poisson (NCP)\n", color=:green, bold=true)
    println(io, "  ├── Intercept:     $(m.μ)")
    println(io, "  ├── Home Adv:      $(m.γ)")
    println(io, "  ├── Heterogeneity: $(m.σ_k)")
    println(io, "  └── NCP Prior:     $(m.Δₛ)")
end


# --- 2. Model Definition ---
@model function static_hierarchical_poisson_NCP_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticHierarchicalPoissonNCP)

    μ ~ model.μ        # intercept prior 
    γ ~ model.γ        # home advantage
    σₐ ~ model.σ_k  # attack parameters standard deviation 
    σᵦ ~ model.σ_k  # defence parameters standard deviation 

  
    # Non centered - z scores -"s-scores here"
    αₛ ~ filldist(model.Δₛ, n_teams) 
    βₛ ~ filldist(model.Δₛ, n_teams)

    # Deterministic Transformation - Scaling 
    αᵣ = αₛ .* σₐ 
    βᵣ = βₛ .* σᵦ 


    # sum-to-zero (STZ)
    α = αᵣ .- mean(αᵣ)
    β = βᵣ .- mean(βᵣ)

    home_goals ~ arraydist(LogPoisson.(α[home_ids] .+ β[away_ids] .+ μ .+ γ ))
    away_goals ~ arraydist(LogPoisson.(α[away_ids] .+ β[home_ids] .+ μ     ))
end

# --- 3. Builder ---
function build_turing_model(model::StaticHierarchicalPoissonNCP, feature_set::FeatureSet)
    # Using dictionary syntax feature_set[:key]
    return static_hierarchical_poisson_NCP_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        feature_set[:flat_home_goals],
        feature_set[:flat_away_goals],
        model
    )
end


# --- 4. The Worker ---
function extract_parameters(
    model::StaticHierarchicalPoissonNCP, 
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet, 
    chains::Chains
)::Dict{Int, PoissonRates}

    extraction_dict = Dict{Int64, PoissonRates}()
    team_map = feature_set[:team_map]
    n_teams = feature_set[:n_teams]

    # --- A. Pre-Process Chains (The Fix) ---
    # 1. Extract Global Parameters as vectors
    μ = vec(chains[:μ])
    γ = vec(chains[:γ])

    σₐ = vec(chains[:σₐ])
    σᵦ = vec(chains[:σᵦ])

    αₛ = Array(group(chains, :αₛ))
    βₛ = Array(group(chains, :βₛ))

    # reconstruct 
    α_matrix = αₛ .* σₐ   
    β_matrix = βₛ .* σᵦ   


    # Broadcast subtract to get the valid parameters
    α_centered = α_matrix .- mean(α_matrix, dims=2)
    β_centered = β_matrix .- mean(β_matrix, dims=2)

    # --- B. Prediction Loop ---
    for row in eachrow(df_to_predict)

        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        # Extract the specific columns for these teams from our centered matrices
        # Note: matrices are (Samples × Teams), so we grab column [:, h_id]
        α_h = α_centered[:, h_id]
        β_a = β_centered[:, a_id]
        α_a = α_centered[:, a_id]
        β_h = β_centered[:, h_id]

        # Calculate Rates
        λ_h = exp.(μ .+ γ .+ α_h .+ β_a)
        λ_a = exp.(μ      .+ α_a .+ β_h)
        
        extraction_dict[row.match_id] = (; λ_h, λ_a)
    end
    return extraction_dict
end

