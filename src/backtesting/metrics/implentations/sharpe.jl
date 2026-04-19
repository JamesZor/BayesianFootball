# src/backtesting/metrics/implentations/sharpe.jl 

export SharpeRatio

"""
    SharpeRatio
Calculates the annualized Sharpe Ratio.
"""
struct SharpeRatio <: AbstractWealthMetric end


function metric_description(m::SharpeRatio)::String
    return " s = r /  σ . 
            where:
                s : Sharpe ratio,
                r : asset return, 
                σ : Standard deviation of the assest return.
          "
end


function compute_metric(m::SharpeRatio, equity_curve::AbstractVector{<:Number})
    if length(equity_curve) < 2 return 0.0 end
    
    # Calculate daily returns (diff)
    returns = diff(equity_curve)
    
    avg_ret = mean(returns)
    std_dev = std(returns)
    
    if std_dev == 0.0 return 0.0 end 
    
    return avg_ret / std_dev 
    
end
