# src/data/markets/types

abstract type AbstractMarket end

"""
    MarketConfig
Wrapper for the specific set of markets we want to process.
"""
struct MarketConfig
    markets::Vector{AbstractMarket}
end

# Convenience constructor
MarketConfig(m::AbstractMarket...) = MarketConfig(collect(m))

"""
    MarketData
Composition wrapper for the processed DataFrame. 
Behaves like a DataFrame but carries the configuration context.
"""
struct MarketData
    df::DataFrame
    config::MarketConfig
end

# --- Forwarding Base Methods to underlying DataFrame ---
Base.getindex(md::MarketData, args...) = getindex(md.df, args...)
Base.setindex!(md::MarketData, val, args...) = setindex!(md.df, val, args...)
Base.size(md::MarketData) = size(md.df)
Base.show(io::IO, md::MarketData) = show(io, md.df)

import DataFrames: nrow, ncol, names
nrow(md::MarketData) = nrow(md.df)
ncol(md::MarketData) = ncol(md.df)
names(md::MarketData) = names(md.df)

