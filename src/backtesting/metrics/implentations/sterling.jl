export SterlingRatio

"""
    SterlingRatio
Calculates a Sterling-style Ratio (Total Return / Average of Top 5 Drawdowns).
"""
struct SterlingRatio <: AbstractWealthMetric 
    top_n_drawdowns::Int
end
# Default constructor
SterlingRatio() = SterlingRatio(5)

function metric_description(m::SterlingRatio)::String
    return " S = Total Return / Average(Top $(m.top_n_drawdowns) Drawdowns) "
end

function compute_metric(m::SterlingRatio, equity_curve::AbstractVector{<:Number})
    if isempty(equity_curve) return 0.0 end
    
    total_return = last(equity_curve)
    
    # 1. Identify Drawdowns
    # We need to find distinct drawdown periods (Peak to Valley to Peak)
    drawdowns = Float64[]
    peak = -Inf
    current_dd = 0.0
    
    for val in equity_curve
        if val > peak
            peak = val
            # If we were in a drawdown, record it and reset
            if current_dd > 0
                push!(drawdowns, current_dd)
                current_dd = 0.0
            end
        else
            dd = peak - val
            # Update max dd for this specific valley
            if dd > current_dd
                current_dd = dd
            end
        end
    end
    # Push final drawdown if active
    if current_dd > 0 push!(drawdowns, current_dd) end
    
    if isempty(drawdowns) return 0.0 end
    
    # 2. Average Top N
    sort!(drawdowns, rev=true)
    top_n = min(length(drawdowns), m.top_n_drawdowns)
    avg_dd = mean(drawdowns[1:top_n])
    
    if avg_dd == 0.0 return 0.0 end
    
    return total_return / avg_dd
end
