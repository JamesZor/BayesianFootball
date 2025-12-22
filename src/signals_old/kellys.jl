"""
Calculates the optimal Kelly criterion fraction to bet.

# Arguments
- `decimal_odds`: The decimal odds offered by the bookmaker (e.g., 2.5, 3.0).
- `probability`: Your estimated true probability of the event occurring (e.g., 0.45).

# Returns
- `f`: The fraction of your bankroll to bet (from 0.0 to 1.0).
  A value of 0.0 means the bet has no value (p < 1/decimal_odds).
"""
function kelly_fraction(decimal_odds::Number, probability::Number)
  return max(0.0, probability - ( (1 - probability) / (decimal_odds - 1.0)))
end 

function kelly_fraction(decimal_odds::Number, probability::AbstractVector)
  return kelly_fraction.(decimal_odds, probability)
end 

function kelly_fraction(odds::NamedTuple, probabilities::NamedTuple) 
  common_keys = keys(odds) ∩ keys(probabilities)
  return NamedTuple(
          k => kelly_fraction(odds[k], probabilities[k])
          for k in common_keys
      )
end

function kelly_positive( kelly_dist::AbstractVector, c::Number)
  return mean(kelly_dist .> c)
end 

function kelly_decision_rule( kelly_dist::AbstractVector, c::Number, b::Number)::Bool
  return kelly_positive(kelly_dist, c) >= b 
end 

function kellys_stake_precent(kelly_dist::AbstractVector, kellys_fraction::Number)::Float64 
  return kellys_fraction * median(kelly_dist)
end 

function kelly_strategy(kelly_dist::AbstractVector, c::Number, b::Number, f::Number)::Number 
  return kelly_decision_rule(kelly_dist, c, b) * kellys_stake_precent(kelly_dist, f) 
end 



"""
Calculates the Optimal Shrinkage Factor 'k' using the Baker & McHale (2013) 
Bootstrap/Resampling method (Eq. 2).

This simulates the penalty of acting on noisy probability estimates.
"""
function bayesian_kelly(chain_probs::AbstractVector, offered_odds::Number)
    b = offered_odds - 1.0
    
    # 1. We treat the Mean of the posterior as the "Ground Truth" for this simulation
    p_true = mean(chain_probs)
    
    # If the mean suggests no bet, we can't shrink what doesn't exist.
    s_mean = kelly_fraction(offered_odds, p_true)
    if s_mean <= 1e-6
        return 0.0
    end

    # 2. We generate the "Naive" bets we would have made for every sample in the chain.
    # Ideally, we calculate s*(q) for every q.
    # This represents the variability of our decision making process.
    naive_bets = [kelly_fraction(offered_odds, q) for q in chain_probs]

    # 3. Objective Function: 
    # Find k such that if we shrink ALL our naive bets by k, 
    # we maximize growth against the "p_true".
    function objective(k)
        utility_sum = 0.0
        n = length(naive_bets)
        
        for s_q in naive_bets
            # The bet we actually place is the Naive Bet * Shrinkage k
            actual_stake = k * s_q
            
            # Constraint check
            actual_stake = k * s_q
            if actual_stake >= 0.999 return Inf end
            if actual_stake < 1e-4 actual_stake = 0.0 end
            
            
            # Utility evaluated against the Mean (p_true)
            u = p_true * log(1.0 + b * actual_stake) + (1.0 - p_true) * log(1.0 - actual_stake)
            utility_sum += u
        end
        
        return -(utility_sum / n)
    end

    res = optimize(objective, 0.0, 1.0)
    best_k = Optim.minimizer(res)
    
    return s_mean * best_k
end


function bayesian_kelly(probabilities::NamedTuple, odds::NamedTuple) 
    common_keys = keys(probabilities) ∩ keys(odds) 
    return NamedTuple(
            k => bayesian_kelly(probabilities[k], odds[k]) 
            for k in common_keys 
    )
end


"""Baker-McHale Eq 5 (Analytical Approx)"""
function calc_analytical_shrinkage(chain_probs::AbstractVector, offered_odds::Number)
    p_mean = mean(chain_probs)
    p_var = var(chain_probs)
    b = offered_odds - 1.0
    s_star = ((b + 1) * p_mean - 1) / b
    if s_star <= 0 return 0.0 end
    term = ((b + 1) / b)^2
    k_factor = (s_star^2) / (s_star^2 + term * p_var)
    return s_star * k_factor
end

function calc_analytical_shrinkage(probabilities::NamedTuple, odds::NamedTuple)
    common_keys = keys(probabilities) ∩ keys(odds) 
    return NamedTuple(
            k => calc_analytical_shrinkage(probabilities[k], odds[k])
            for k in common_keys 
          )
end

