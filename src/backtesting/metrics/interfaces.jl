# src/backtesting/metrics/interfaces.jl

export compute_metric, metric_name

"""
    compute_metric(metric::AbstractWealthMetric, equity_curve::AbstractVector{<:Number})

Calculates the single scalar value for the given metric.
Must be implemented by concrete types.
"""
function compute_metric(metric::AbstractWealthMetric, equity_curve::AbstractVector{<:Number})
    error("Implementation missing for metric: $(typeof(metric))")
end

"""
    metric_symbol(metric::AbstractWealthMetric)

Returns the symbol key to be used in the DataFrame (e.g., :sharpe_ratio).
Defaults to the type name.
"""
function metric_symbol(m::AbstractWealthMetric)::Symbol
  Symbol(nameof(typeof(m)))
end


function metric_name(m::AbstractWealthMetric)::String
    return string(nameof(typeof(m)))
end

"""
    metric_description(s::AbstractSignal)::String

Returns a short description of the mathematical logic used.
"""
function metric_description(m::AbstractWealthMetric)::String
    return "No description provided."
end


# --- TUI / Display ---

function Base.show(io::IO, s::AbstractWealthMetric)
    # Compact inline print: KellyCriterion(fraction=0.5)
  print(io, "$(metric_name(s))")
end

function Base.show(io::IO, ::MIME"text/plain", s::AbstractWealthMetric)
    # Multiline TUI print for detailed inspection
    print(io, "Metric: $(metric_name(s))\n")
    print(io, "└─ Logic: $(metric_description(s))")
end

