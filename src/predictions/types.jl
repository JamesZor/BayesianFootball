# src/predictions/types.jl
using ..Data: MarketConfig, MarketData, AbstractMarket
# We rely on the parent module (Predictions) to provide DataFrames, AbstractFootballModel, etc.
"""
    PredictionConfig
Alias for MarketConfig. Maintained for backward compatibility.
"""
const PredictionConfig = MarketConfig

# --- CHANGED: Removed struct MarketData (now defined in Data) ---

# ------------------------------------------------------------------
# PPD (Posterior Predictive Distribution)
# ------------------------------------------------------------------
struct PPD
    df::DataFrame
    model::TypesInterfaces.AbstractFootballModel
    config::PredictionConfig # This works because PredictionConfig is an alias
end

Base.getindex(ppd::PPD, args...) = getindex(ppd.df, args...)
Base.size(ppd::PPD) = size(ppd.df)
Base.size(ppd::PPD, i) = size(ppd.df, i)
Base.show(io::IO, ppd::PPD) = show(io, ppd.df)
DataFrames.nrow(ppd::PPD) = nrow(ppd.df)
DataFrames.ncol(ppd::PPD) = ncol(ppd.df)
DataFrames.groupby(ppd::PPD, args...) = groupby(ppd.df, args...)

