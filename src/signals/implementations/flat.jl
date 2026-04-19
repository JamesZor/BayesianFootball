"""
    FlatStake(stake::Float64)
    
Bet a fixed percentage of bankroll if the Expected Value (EV) is positive.
"""
struct FlatStake <: AbstractSignal
    stake::Float64
end

function compute_stake(s::FlatStake, dist::AbstractVector, odds::Number)
    p_mean = mean(dist)
    # EV = (Probability * DecimalOdds) - 1
    ev = (p_mean * odds) - 1.0
    
    return ev > 0 ? s.stake : 0.0
end

signal_description(::FlatStake) = "Fixed stake if Expected Value (E[p]*odds - 1) > 0."
