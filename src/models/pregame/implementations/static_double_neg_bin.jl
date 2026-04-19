# src/models/pregame/implementations/static_double_neg_bin.jl

using Turing, Distributions 
using ..MyDistributions 


export StaticDoubleNegBin

Base.@kwdef struct StaticDoubleNegBin <: AbstractStaticNegBinModel
    # Standard priors for team strength
    μ::Distribution   = Normal(0.0, 10.0)
    γ::Distribution   = Normal(log(1.3), 0.2)
    σ_k::Distribution = Truncated(Normal(0, 1), 0, Inf)
    
    # Dispersion Priors (on the Log Scale)
    # r ≈ 10 implies close to Poisson. r ≈ 1 implies high variance.
    # Normal(1.5, 1) covers a reasonable range of r in [0.5, 20]
    log_r_prior::Distribution = Normal(1.5, 1.0) 
end

@model function static_double_neg_bin_train(n_teams, home_ids, away_ids, goal_pairs, model::StaticDoubleNegBin)

    # --- Global Params ---
    μ ~ model.μ
    γ ~ model.γ
    
    # --- Dispersion Parameters (Log Space) ---
    # We model separate dispersion for home and away, or shared?
    # Let's do shared for simplicity, or separate if data supports it.
    log_r ~ model.log_r_prior
    r = exp(log_r) # Transform back to positive space
    
    # --- Team Skills ---
    σ_a ~ model.σ_k
    σ_d ~ model.σ_k
    
    # Non-centered parameterization for team skills
    att_raw ~ filldist(Normal(0,1), n_teams)
    def_raw ~ filldist(Normal(0,1), n_teams)
    
    att = att_raw .* σ_a
    def = def_raw .* σ_d
    
    # Sum-to-zero constraint
    att = att .- mean(att)
    def = def .- mean(def)

    # --- Likelihood ---
    # Calculate rates
    # λ_h = exp(μ + γ + att_h + def_a)
    λ_h = exp.(μ .+ γ .+ att[home_ids] .+ def[away_ids])
    λ_a = exp.(μ      .+ att[away_ids] .+ def[home_ids])

    # We use the arraydist pattern for efficiency
    # But since DoubleNegativeBinomial takes scalars, we map it.
    # Alternatively, you can vectorize the custom dist, but a loop is often fine in Turing.
    
    # Option A: Vectorized construction (Faster)
    # distributions = DoubleNegativeBinomial.(λ_h, λ_a, r, r)
    # data = [[h, a] for (h,a) in zip(home_goals, away_goals)]
    goal_pairs ~ arraydist(DoubleNegativeBinomial.(λ_h, λ_a, r, r))
    
end




function build_turing_model(model::StaticDoubleNegBin, feature_set::FeatureSet)
    flat_home = feature_set[:flat_home_goals]
    flat_away = feature_set[:flat_away_goals]
    
    data_matrix = permutedims(hcat(flat_home, flat_away))

    return static_double_neg_bin_train(
        feature_set[:n_teams]::Int,
        feature_set[:flat_home_ids],
        feature_set[:flat_away_ids],
        data_matrix, # <-- Passing a 2xN Matrix
        model
    )
end



function extract_parameters(
    model::StaticDoubleNegBin,
    df_to_predict::AbstractDataFrame,
    feature_set::FeatureSet,
    chains::Chains 
)::Dict{Int, NamedTuple}
  extraction_dict =Dict{Int, NamedTuple}()
  team_map = feature_set[:team_map]


  μ = vec(chains[:μ])
  γ = vec(chains[:γ])
  r = exp.(vec(chains[:log_r]))
  
  σ_a = vec(chains[:σ_a])
  σ_d = vec(chains[:σ_d])
  α_s = Array(group(chains, :att_raw))
  β_s = Array(group(chains, :def_raw))

  # ncp reconstruct
  α  = ( α_s .* σ_a) .- mean( (α_s .* σ_a), dims=2)
  β  = ( β_s .* σ_d) .- mean( (β_s .* σ_d), dims=2)


  for row in eachrow(df_to_predict) 
    h_id = team_map[row.home_team]
    a_id = team_map[row.away_team]

    α_h = α[:, h_id]
    β_a = β[:, a_id]
    α_a = α[:, a_id]
    β_h = β[:, h_id]


    λ_h = exp.( μ .+ γ .+ α_h .+ β_a )
    λ_a = exp.( μ      .+ α_a .+ β_h )

    extraction_dict[row.match_id] = (; λ_h, λ_a, r) 
  end 

  return extraction_dict 

end

