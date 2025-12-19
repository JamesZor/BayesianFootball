# src/predictions/types.jl

# We rely on the parent module (Predictions) to provide DataFrames, AbstractFootballModel, etc.

"""
    PredictionConfig
Defines the set of markets to be predicted and analyzed.
"""
struct PredictionConfig
    markets::Set{<:AbstractMarket}
end

# ------------------------------------------------------------------
# MarketData
# ------------------------------------------------------------------
struct MarketData
    df::DataFrame
    config::PredictionConfig
end

# Holy Trait Pattern (Forwarding to DataFrame)
Base.getindex(md::MarketData, args...) = getindex(md.df, args...)
Base.setindex!(md::MarketData, val, args...) = setindex!(md.df, val, args...)
Base.size(md::MarketData) = size(md.df)
Base.size(md::MarketData, i) = size(md.df, i)
Base.show(io::IO, md::MarketData) = show(io, md.df)
DataFrames.nrow(md::MarketData) = nrow(md.df)
DataFrames.ncol(md::MarketData) = ncol(md.df)
DataFrames.groupby(md::MarketData, args...) = groupby(md.df, args...)

# ------------------------------------------------------------------
# PPD
# ------------------------------------------------------------------
struct PPD
    df::DataFrame
    model::TypesInterfaces.AbstractFootballModel
    config::PredictionConfig
end

Base.getindex(ppd::PPD, args...) = getindex(ppd.df, args...)
Base.size(ppd::PPD) = size(ppd.df)
Base.size(ppd::PPD, i) = size(ppd.df, i)
Base.show(io::IO, ppd::PPD) = show(io, ppd.df)
DataFrames.nrow(ppd::PPD) = nrow(ppd.df)
DataFrames.ncol(ppd::PPD) = ncol(ppd.df)
DataFrames.groupby(ppd::PPD, args...) = groupby(ppd.df, args...)
