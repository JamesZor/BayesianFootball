export CalmarRatio

"""
    CalmarRatio
Calculates the Calmar Ratio (Total Return / Max Drawdown).
"""
struct CalmarRatio <: AbstractWealthMetric end

function metric_description(m::CalmarRatio)::String
    return " C = Total Return / |Max Drawdown| "
end

function compute_metric(m::CalmarRatio, equity_curve::AbstractVector{<:Number})
    if isempty(equity_curve) return 0.0 end
    
    total_return = last(equity_curve)
    
    # Calculate Max Drawdown
    peak = -Inf
    max_dd = 0.0
    
    for val in equity_curve
        if val > peak
            peak = val
        end
        # Drawdown is Peak - Current (since equity_curve is absolute PnL)
        dd = peak - val
        if dd > max_dd
            max_dd = dd
        end
    end
    
    if max_dd == 0.0 return 0.0 end
    
    return total_return / max_dd
end
