# src/models/pregame/implementations/static_mvpln.jl
#=
The Multivariate Poisson Log normal 
=#

export StaticMVPLN

# --- 1. Struct ---
Base.@kwdef struct StaticMVPLN <: AbstractStaticMVPLNModel 
  κ::Distribution = Normal(0,10) # intercept 
  γ::Distribution = Normal(log(1.3), 0.2) # home advantage
  σₖ::Distribution = Gamma(2, 1/3) # for team strengths
  σₜ::Distribution = Gamma(2, 1/3) # for correlations
  Δₛ::Distribution = Normal(0,1) # Unit normal for NCP
  ρᵣ::Distribution = Normal(0,1) # Unit normal for NCP ρ = tanh(ρᵣ) ∈ (-1,1)
  ϵₛ::Distribution = MvNormal(zeros(2), I) # z score unit normal dim 2  - I is identity matrix for I₂

end

function Base.show(io::IO, ::MIME"text/plain", m::StaticMVPLN)
    printstyled(io, "Static StaticMVPLN (NCP)\n", color=:green, bold=true)
    println(io, "  ├── Intercept:     $(m.κ)")
    println(io, "  ├── Home Adv:      $(m.γ)")
    println(io, "  ├── Heterogeneity: $(m.σₖ)")
    println(io, "  └── NCP Prior:     $(m.Δₛ)")
end


# --- 2. Model Definition ---
@model function static_mvpln_model_train(n_teams, home_ids, away_ids, home_goals, away_goals, model::StaticMVPLN)

    κ ~ model.κ        # intercept prior 
    γ ~ model.γ        # home advantage
    σₐ ~ model.σₖ  # attack parameters standard deviation 
    σᵦ ~ model.σₖ  # defence parameters standard deviation 

    # Non centered - z scores -"s-scores here"
    αₛ ~ filldist(model.Δₛ, n_teams) 
    βₛ ~ filldist(model.Δₛ, n_teams)

    # Deterministic Transformation - Scaling 
    αᵣ = αₛ .* σₐ 
    βᵣ = βₛ .* σᵦ 

    # sum-to-zero (STZ)
    α = αᵣ .- mean(αᵣ)
    β = βᵣ .- mean(βᵣ)

    # 
    σₕ ~ model.σₜ # home 
    σᵥ ~ model.σₜ # away( v = visting)

    ρᵣ ~ model.ρᵣ  # raw correlations coefficient 
    ρ = tanh(ρᵣ)

    ϵₛ ~ model.ϵₛ
  MvNormal(zeros(2), I) # z score unit normal dim 2 
    
    # manual Cholesky matrix  for Σ = LLᵀ
    L = [ σₕ    ,        0         ;
          ρ * σᵥ, σᵥ * √(1 - ρ^2 ) ]

    ϵ = L * ϵₛ # matrix operation



  home_goals ~ arraydist(LogPoisson.(α[home_ids] .+ β[away_ids] .+ μ .+ γ .+ ϵ[1]))
  away_goals ~ arraydist(LogPoisson.(α[away_ids] .+ β[home_ids] .+ μ      .+ ϵ[2]))

end


# --- 3. Builder ---
function build_turing_model(model::StaticMVPLN, feature_set::FeatureSet)
    # Using dictionary syntax feature_set[:key]
    return static_mvpln_model_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        feature_set[:flat_home_goals],
        feature_set[:flat_away_goals],
        model
    )
end


