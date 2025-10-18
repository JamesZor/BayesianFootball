module Markets

export AbstractMarket, Market1X2, MarketOverUnder, MarketBTTS, get_standard_markets

abstract type AbstractMarket end

struct Market1X2 <: AbstractMarket end

struct MarketOverUnder <: AbstractMarket
    line::Float64
end

struct MarketBTTS <: AbstractMarket end

# add more...
# struct MarketAsianHandicap <: AbstractMarket
#     line::Float64
# end

"""
Returns a standard set of common markets for prediction.
"""
function get_standard_markets()
    # Use a Set to store the market types.
    # We specify Set{AbstractMarket} to ensure type stability.
    standard_set = Set{AbstractMarket}([
        Market1X2(),
        MarketBTTS(),
        MarketOverUnder(0.5),
        MarketOverUnder(1.5),
        MarketOverUnder(2.5),
        MarketOverUnder(3.5),
        MarketOverUnder(4.5)
    ])
    
    return standard_set
end

end
