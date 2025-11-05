# src/prediction/pregame/predict-abstractpoisson.jl


function predict_market(
    model::TypesInterfaces.AbstractPoissonModel,
    market::Markets.Market1X2, 
    λ_h, λ_a
)

          """
          Helper function:
          Calculates 1X2 probabilities for a SINGLE pair of goal-rate parameters.
          """
          function _calculate_1x2_from_params(λ::Float64, μ::Float64, max_goals::Int=10)
              
              # Create distributions for this one MCMC sample
              home_dist = Poisson(λ)
              away_dist = Poisson(μ)
              
              p_home_win = 0.0
              p_draw = 0.0
              p_away_win = 0.0

              for h in 0:max_goals
                  for a in 0:max_goals
                      # P(H=h, A=a) = P(H=h) * P(A=a)
                      p_score = pdf(home_dist, h) * pdf(away_dist, a)
                      
                      if h > a
                          p_home_win += p_score
                      elseif h == a
                          p_draw += p_score
                      else # a > h
                          p_away_win += p_score
                      end
                  end
              end
              
              # Note: These will sum to < 1.0 due to max_goals truncation,
              # but we can re-normalize them for a cleaner 1X2 probability.
              total_p = p_home_win + p_draw + p_away_win
              
              return (
                  home = p_home_win / total_p,
                  draw = p_draw / total_p,
                  away = p_away_win / total_p
              )
          end

          function compute_1x2_distributions(λs::AbstractVector, μs::AbstractVector, max_goals::Int=10)
              
              n_samples = length(λs)
              
              # Pre-allocate thread-safe output vectors
              p_home_vec = zeros(n_samples)
              p_draw_vec = zeros(n_samples)
              p_away_vec = zeros(n_samples)
              
              # Use @threads to split the loop across your available cores
              for i in 1:n_samples
                  # Note: No need to index λs[i], the loop does it
                  λ_i = λs[i]
                  μ_i = μs[i]
                  
                  # Call the original (fast) inner-loop function
                  probs = _calculate_1x2_from_params(λ_i, μ_i, max_goals)
                  
                  # Write to the pre-allocated vectors
                  # This is safe because each thread writes to a different index `i`
                  p_home_vec[i] = probs.home
                  p_draw_vec[i] = probs.draw
                  p_away_vec[i] = probs.away
              end
              
              return (
                  p_home_dist = p_home_vec,
                  p_draw_dist = p_draw_vec,
                  p_away_dist = p_away_vec
              )
          end

  computed_1x2 = compute_1x2_distributions(λ_h, λ_a, 10)
  return NamedTuple{(:home, :draw, :away)}((computed_1x2.p_home_dist, computed_1x2.p_draw_dist, computed_1x2.p_away_dist))
end

function predict_market(
    model::TypesInterfaces.AbstractPoissonModel,
    market::Markets.MarketOverUnder, 
    λ_h, λ_a
)

    total_rates_chain = λ_h .+ λ_a
    total_goal_dists = Poisson.(total_rates_chain)
    
    threshold = floor(Int, market.line) # e.g., 2 for line=2.5
    
    p_under_chain = cdf.(total_goal_dists, threshold)
    p_over_chain = 1.0 .- p_under_chain


    line_str = replace(string(market.line), "." => "")
            
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
    return NamedTuple{(over_key, under_key)}((p_over_chain, p_under_chain))
end




"""
Calculates the full posterior chain for BTTS probabilities.
"""
function predict_market(
    model::TypesInterfaces.AbstractPoissonModel,
    market::Markets.MarketBTTS, 
    λ_h, λ_a
)
    home_dists = Poisson.(λ_h)
    away_dists = Poisson.(λ_a)
    
    # P(Home > 0)
    p_home_scores_chain = 1.0 .- pdf.(home_dists, 0)
    # P(Away > 0)
    p_away_scores_chain = 1.0 .- pdf.(away_dists, 0)
    
    p_btts_yes_chain = p_home_scores_chain .* p_away_scores_chain
    p_btts_no_chain = 1.0 .- p_btts_yes_chain
    
  return NamedTuple{(:btts_yes, :btts_no)}((p_btts_yes_chain, p_btts_no_chain))
end

function predict_market(
    model::TypesInterfaces.AbstractPoissonModel,
    predict_config::PredictionConfig,
    λ_h::AbstractVector{Float64},
    λ_a::AbstractVector{Float64}
    )

    market_results_generator = (
        predict_market(model, market, λ_h, λ_a) for market in predict_config.markets
    )
  match_predict = reduce(merge, market_results_generator; init = (;) );
  return match_predict

end 

