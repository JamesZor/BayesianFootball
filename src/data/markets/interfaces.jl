# src/data/markets/interfaces.jl

"""
    market_group(m::AbstractMarket) -> String
Returns the standardized string name for the market group (e.g. "1X2", "OverUnder").
"""
function market_group(m::AbstractMarket) 
    error("market_group not implemented for $(typeof(m))") 
end

"""
    outcomes(m::AbstractMarket) -> NamedTuple
Returns a NamedTuple mapping logical keys to symbols (e.g. (home=:home, away=:away)).
Used to standardize selection column.
"""
function outcomes(m::AbstractMarket) 
    error("outcomes not implemented for $(typeof(m))") 
end

"""
    market_line(m::AbstractMarket) -> Float64
Returns the handicap or line (e.g. 2.5). Defaults to 0.0.
"""
market_line(m::AbstractMarket) = 0.0

# Fallback show
Base.show(io::IO, m::AbstractMarket) = print(io, market_group(m))
