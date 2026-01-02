export SortinoRatio

"""
    SortinoRatio
Calculates the Sortino Ratio (Return / Downside Deviation).
"""
struct SortinoRatio <: AbstractWealthMetric end

function metric_description(m::SortinoRatio)::String
    return " S = r / σ_d
            where:
                r : average return, 
                σ_d : Standard deviation of negative returns (downside).
           "
end

function compute_metric(m::SortinoRatio, equity_curve::AbstractVector{<:Number})
    if length(equity_curve) < 2 return 0.0 end
    
    returns = diff(equity_curve)
    avg_ret = mean(returns)
    
    # Filter for negative returns only
    neg_returns = filter(x -> x < 0.0, returns)
    
    if isempty(neg_returns)
        # If no losing bets, Sortino is infinite/undefined. Return 0 or high number?
        # Returning 0.0 is safer for plotting, but technically it's a perfect strategy.
        return avg_ret > 0 ? 999.0 : 0.0 
    end

    # Downside Deviation (std of negative returns, often calculated with 0 as target)
    # Standard definition uses the root mean square of the underperformance
    downside_sq_sum = sum(abs2, neg_returns)
    downside_dev = sqrt(downside_sq_sum / length(returns))
    
    if downside_dev == 0.0 return 0.0 end
    
    return avg_ret / downside_dev
end
