# src/models/pregame/implementations/static_hierarchical_poisson.jl

export StaticHierarchicalPoisson

# --- 1. Struct ---
Base.@kwdef struct StaticHierarchicalPoisson{D<:Distribution, S<:Distribution, F<:Function} <: AbstractStaticPoissonModel 
  μ::D = Normal(0, 10) # intercept 
  γ::D = Normal(log(1.3), 0.2) # home advantage 
  σ_k::S = Truncated(Cauchy(0,5), 0, Inf) # standard deviation for the team parameters raw ( α, β ) 
  α_β_raw::F=  ( σ -> Normal(0, σ ) ) # function to define the team strength parameters - as we need the standard deviation.
end

function Base.show(io::IO, ::MIME"text/plain", m::StaticHierarchicalPoisson)
    printstyled(io, "Static Hierarchical Poisson (Flexible)\n", color=:cyan, bold=true)
    println(io, "  ├── Heterogeneity:   $(m.σ_k)")
    println(io, "  └── Team Dist:       σ -> Distribution") # Hard to print the lambda source
end

# --- 2. Model Definition ---
@model function static_hierarchical_poisson_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticHierarchicalPoisson)

    μ ~ model.μ        # intercept prior 
    γ ~ model.γ        # home advantage
    σ_att ~ model.σ_k  # attack parameters standard deviation 
    σ_def ~ model.σ_k  # defence parameters standard deviation 

    α_raw ~ filldist( model.α_β_raw(σ_att), n_teams) 
    β_raw ~ filldist( model.α_β_raw(σ_def), n_teams)

    # sum-to-zero (STZ)
    α = α_raw .- mean(α_raw)
    β = β_raw .- mean(β_raw)

    home_goals ~ arraydist(LogPoisson.(α[home_ids] .+ β[away_ids] .+ μ .+ γ ))
    away_goals ~ arraydist(LogPoisson.(α[away_ids] .+ β[home_ids] .+ μ     ))
end

# --- 3. Builder ---
function build_turing_model(model::StaticHierarchicalPoisson, feature_set::FeatureSet)
    # Using dictionary syntax feature_set[:key]
    return static_hierarchical_poisson_model_train(
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
    model::StaticHierarchicalPoisson, 
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

    # 2. Extract ALL raw team parameters into a Matrix (Samples × Teams)
    # This is much faster than querying chains["..."] inside a loop
    α_matrix = Array(group(chains, :α_raw)) 
    β_matrix = Array(group(chains, :β_raw))

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

