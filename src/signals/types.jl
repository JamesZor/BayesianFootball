# src/signals/types.jl

abstract type AbstractSignal end

"""
    SignalsResult
The container for the output of the signals processing pipeline.
It wraps the resulting DataFrame with the context used to generate it.
"""
struct SignalsResult
    df::DataFrame              # Columns: match_id, selection, signal_name, signal_params, stake, odds, etc.
    model::AbstractFootballModel
    signals::Vector{AbstractSignal}
end

# Forwarding for convenience so REPL acts like it's just the DataFrame
Base.show(io::IO, res::SignalsResult) = show(io, res.df)
Base.size(res::SignalsResult) = size(res.df)
Base.getindex(res::SignalsResult, args...) = getindex(res.df, args...)
