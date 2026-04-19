export BurkeRatio

"""
    BurkeRatio
Calculates the Burke Ratio (Total Return / sqrt(Sum of Drawdowns^2)).
"""
struct BurkeRatio <: AbstractWealthMetric end

function metric_description(m::BurkeRatio)::String
    return " B = Total Return / √(Σ(Drawdowns²)) "
end

function compute_metric(m::BurkeRatio, equity_curve::AbstractVector{<:Number})
    if isempty(equity_curve) return 0.0 end
    
    total_return = last(equity_curve)
    
    # Calculate Drawdowns
    peak = -Inf
    sum_sq_drawdowns = 0.0
    
    for val in equity_curve
        if val > peak
            peak = val
        end
        dd = peak - val
        sum_sq_drawdowns += abs2(dd)
    end
    
    if sum_sq_drawdowns == 0.0 return 0.0 end
    
    return total_return / sqrt(sum_sq_drawdowns)
end
