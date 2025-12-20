# src/data/Markets/Markets-module.jl

module Markets

using DataFrames
using Statistics

# We assume DataStore is defined in the parent Data module or available via ..Types
# If not, we just rely on passing the 'ds' object which usually quacks like a struct with .odds

export 
    # Types
    AbstractMarket, 
    Market1X2, 
    MarketOverUnder, 
    MarketBTTS, 
    MarketConfig, 
    MarketData,

    # Constants
    DEFAULT_MARKET_CONFIG,

    # Functions
    prepare_market_data,
    market_group,
    market_line,
    outcomes

# 1. Interfaces & Types (The Contract)
include("types.jl")
include("interfaces.jl")

# 2. Helpers (Parsers & Utils)
include("utils.jl")

# 3. Implementations (The Concrete Logic)
#    These files rely on interfaces.jl and types.jl being loaded first
include("implementations/1x2.jl")
include("implementations/over_under.jl")
include("implementations/btts.jl")

# 4. The Processing Engine (ETL)
include("processing.jl")


"""
    get_standard_market_config()

Returns a default MarketConfig containing the most common markets:
- 1X2 (Full Time)
- Over/Under 2.5
- BTTS (Both Teams To Score)
"""
function get_standard_market_config()
    return MarketConfig( reduce(vcat, ( [Market1X2(), MarketBTTS()], [MarketOverUnder( (i +0.5) ) for i in 0:10 ] ) ))
end


const DEFAULT_MARKET_CONFIG = get_standard_market_config()

end
