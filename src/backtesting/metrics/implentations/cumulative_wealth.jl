export CumulativeWealth

"""
    CumulativeWealth
Simply returns the total accumulated profit/loss at the end of the period.
"""
struct CumulativeWealth <: AbstractWealthMetric end

function metric_description(m::CumulativeWealth)::String
    return " Total PnL = End Equity - Start Equity "
end

function compute_metric(m::CumulativeWealth, equity_curve::AbstractVector{<:Number})
    if isempty(equity_curve) return 0.0 end
    # Assuming equity_curve is cumsum(pnl), the last value is the total.
    return last(equity_curve)
end
