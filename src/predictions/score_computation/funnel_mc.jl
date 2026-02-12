
# src/prediction/score_computation/funnel_mc.jl

using Distributions
using ..Models
using ..MyDistributions

function extract_params(model::Models.PreGame.AbstractFunnelModel, row)
    return (
        λ_shots_h   = row.λ_shots_h,
        λ_shots_a   = row.λ_shots_a,
        r_create    = row.r_create,
        θ_prec_h    = row.θ_prec_h,
        θ_prec_a    = row.θ_prec_a,
        ϕ_conv_h    = row.ϕ_conv_h,
        ϕ_conv_a    = row.ϕ_conv_a,
        exp_goals_h = row.exp_goals_h,
        exp_goals_a = row.exp_goals_a
    )
end


function compute_score_matrix(
  model::Models.PreGame.AbstractFunnelModel,
  params;
  max_goals::Int=12
)
    r = params.r_create
    xGₕ = params.exp_goals_h
    xGₐ = params.exp_goals_a
    n_samples = length(r)

    # Note: Your logpdf expects a Vector, so we create vectors, not tuples.
    goals_range = 0:(max_goals - 1)

    S = zeros(Float64, max_goals, max_goals, n_samples)


    @inbounds for k in 1:n_samples 
        dist_h = RobustNegativeBinomial(r[k], xGₕ[k])
        dist_a = RobustNegativeBinomial(r[k], xGₐ[k])

        s_k_h = exp.(logpdf.(Ref(dist_h), goals_range) )
        s_k_a = exp.(logpdf.(Ref(dist_a), goals_range) )
    
        S[:, :, k] .= s_k_h .* s_k_a' 
    end
  return ScoreMatrix(S)
end 


  
  
