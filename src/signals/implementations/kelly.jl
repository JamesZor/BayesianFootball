# ------------------------------------------------------------------
# 1. Standard Kelly
# ------------------------------------------------------------------

"""
    KellyCriterion(fraction::Float64)

Standard Kelly Criterion using the mean of the posterior distribution.
Scaled by `fraction` (e.g., 0.5 for Half-Kelly).
"""
struct KellyCriterion <: AbstractSignal
    fraction::Float64 
end

function compute_stake(s::KellyCriterion, dist::AbstractVector, odds::Number)
    p_mean = mean(dist)
    b = odds - 1.0
    
    if b <= 0 return 0.0 end
    
    # f = p - (1-p)/b
    f_star = p_mean - ((1.0 - p_mean) / b)
    
    return max(0.0, f_star * s.fraction)
end

signal_description(::KellyCriterion) = "Classic Kelly: f = p - (1-p)/(odds-1), using mean probability."

# ------------------------------------------------------------------
# 2. Bayesian Kelly (Optimization)
# ------------------------------------------------------------------

"""
    BayesianKelly()

Calculates the Optimal Shrinkage Factor 'k' using the Baker & McHale (2013) 
numerical optimization method to maximize expected log-growth over the 
full posterior distribution.
"""
# struct BayesianKelly <: AbstractSignal end
#
# function compute_stake(s::BayesianKelly, dist::AbstractVector, odds::Number)
#     # Refactored from previous kellys.jl
#     b = odds - 1.0
#     if b <= 0 return 0.0 end
#
#     p_true = mean(dist)
#
#     # If the mean suggests no bet, we can't shrink what doesn't exist.
#     s_mean = max(0.0, p_true - ((1.0 - p_true) / b))
#     if s_mean <= 1e-6
#         return 0.0
#     end
#
#     # Generate "Naive" bets for every sample in the chain
#     naive_bets = [max(0.0, q - ((1.0 - q) / b)) for q in dist]
#
#     # Objective: Find shrinkage k to max growth against "p_true"
#     function objective(k)
#         utility_sum = 0.0
#         n = length(naive_bets)
#
#         for s_q in naive_bets
#             actual_stake = k * s_q
#
#             # Constraints
#             if actual_stake >= 0.999 return Inf end
#             actual_stake = max(0.0, actual_stake)
#
#             # Utility evaluated against Mean
#             u = p_true * log(1.0 + b * actual_stake) + (1.0 - p_true) * log(1.0 - actual_stake)
#             utility_sum += u
#         end
#         return -(utility_sum / n)
#     end
#
#     res = optimize(objective, 0.0, 1.0)
#     best_k = Optim.minimizer(res)
#
#     return s_mean * best_k
# end
#
struct BayesianKelly <: AbstractSignal 
    min_edge::Float64
end
# Default constructor: If you don't pass an edge, it assumes 0.0
BayesianKelly(; min_edge::Float64 = 0.0) = BayesianKelly(min_edge)

function compute_stake(s::BayesianKelly, dist::AbstractVector, odds::Number)
    b = odds - 1.0
    if b <= 0 return 0.0 end

    p_true = mean(dist)
    p_implied = 1.0 / odds
    
    # THE EDGE FILTER
    if (p_true - p_implied) < s.min_edge
        return 0.0
    end
    
    s_mean = max(0.0, p_true - ((1.0 - p_true) / b))
    if s_mean <= 1e-6 return 0.0 end

    naive_bets = [max(0.0, q - ((1.0 - q) / b)) for q in dist]

    function objective(k)
        utility_sum = 0.0
        n = length(naive_bets)
        for s_q in naive_bets
            actual_stake = k * s_q
            if actual_stake >= 0.999 return Inf end
            actual_stake = max(0.0, actual_stake)
            u = p_true * log(1.0 + b * actual_stake) + (1.0 - p_true) * log(1.0 - actual_stake)
            utility_sum += u
        end
        return -(utility_sum / n)
    end

    res = optimize(objective, 0.0, 1.0)
    best_k = Optim.minimizer(res)
    
    return s_mean * best_k
end

signal_description(s::BayesianKelly) = "BayesianKelly (Baker-McHale 2013) with $(s.min_edge * 100)% Min Edge Filter."

# signal_description(::BayesianKelly) = "Baker-McHale (2013) Numerical Optimization: Shrinks bet size to account for parameter uncertainty."

# ------------------------------------------------------------------
# 3. Analytical Shrinkage
# ------------------------------------------------------------------

"""
    AnalyticalShrinkageKelly()

Baker-McHale Eq 5: An analytical approximation for the shrinkage factor 
based on the variance of the posterior distribution.
"""
# struct AnalyticalShrinkageKelly <: AbstractSignal end
#
# function compute_stake(s::AnalyticalShrinkageKelly, dist::AbstractVector, odds::Number)
#     p_mean = mean(dist)
#     p_var = var(dist)
#     b = odds - 1.0
#
#     if b <= 0 return 0.0 end
#
#     s_star = ((b + 1) * p_mean - 1) / b
#     if s_star <= 0 return 0.0 end
#
#     # Calculate shrinkage factor k based on variance
#     term = ((b + 1) / b)^2
#     k_factor = (s_star^2) / (s_star^2 + term * p_var)
#
#     return s_star * k_factor
# end
#
# signal_description(::AnalyticalShrinkageKelly) = "Baker-McHale (2013) Eq.5: Analytical Approx using posterior variance."

struct AnalyticalShrinkageKelly <: AbstractSignal 
    min_edge::Float64
end
AnalyticalShrinkageKelly(; min_edge::Float64 = 0.0) = AnalyticalShrinkageKelly(min_edge)

function compute_stake(s::AnalyticalShrinkageKelly, dist::AbstractVector, odds::Number)
    p_mean = mean(dist)
    p_implied = 1.0 / odds
    b = odds - 1.0
    
    if b <= 0 return 0.0 end

    #THE EDGE FILTER
    if (p_mean - p_implied) < s.min_edge
        return 0.0
    end

    s_star = ((b + 1) * p_mean - 1) / b
    if s_star <= 0 return 0.0 end
    
    p_var = var(dist)
    term = ((b + 1) / b)^2
    k_factor = (s_star^2) / (s_star^2 + term * p_var)
    
    return s_star * k_factor
end

signal_description(s::AnalyticalShrinkageKelly) = "AnalyticalShrinkage (Baker-McHale Eq.5) with $(s.min_edge * 100)% Min Edge Filter."

"""
    ExactBayesianKelly()

Calculates the optimal bet size 'f' by numerically maximizing the expected 
log-growth directly over the full posterior distribution samples.
This is superior to shrinkage methods as it accounts for the exact 
higher-order moments (skew/kurtosis) of your specific posterior.
"""
struct ExactBayesianKelly <: AbstractSignal end

function compute_stake(s::ExactBayesianKelly, dist::AbstractVector, odds::Number)
    b = odds - 1.0
    if b <= 0 return 0.0 end
    
    # 1. Quick check: Does the mean even justify a bet?
    # If E[p] * b - (1 - E[p]) <= 0, no amount of variance analysis will make it a bet.
    p_mean = mean(dist)
    if (p_mean * b - (1.0 - p_mean)) <= 0
        return 0.0
    end

    # 2. Objective Function: Expected Log Utility
    # We want to find scalar 'f' that maximizes: 1/N * sum( log(1 + f*r_i) )
    # where r_i is the return for sample i: b if win, -1 if loss.
    # U(f) = q * log(1 + f*b) + (1-q) * log(1 - f)
    # We average this utility across all posterior samples q.
    
    function neg_expected_utility(f)
        # Boundary constraints for optimizer (prevent log(<=0))
        if f <= 1e-9 return 0.0 end
        if f >= 0.99 return Inf end # Soft barrier
        
        utility_sum = 0.0
        n = length(dist)
        
        @inbounds for q in dist
            # For a given probability q (a sample from posterior), 
            # the expected utility of stake f is:
            u_sample = q * log(1.0 + b * f) + (1.0 - q) * log(1.0 - f)
            utility_sum += u_sample
        end
        
        return -(utility_sum / n)
    end

    # 3. Optimization
    # We search for f in [0.0, 0.99]. 
    # The upper bound 0.99 is arbitrary safety; rarely optimal to bet > 50%.
    res = optimize(neg_expected_utility, 0.0, 0.5) 
    
    return Optim.minimizer(res)
end

signal_description(::ExactBayesianKelly) = "Exact maximization of log-utility over posterior samples."
