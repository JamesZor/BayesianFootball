# src/data/data-module.jl
module Data

using DataFrames
using Dates
using LibPQ
using InlineStrings 

# 1. Globals & Utils
include("types.jl")
include("utils.jl")

# 2. Markets Sub-Module
include("Markets/Markets-module.jl")
using .Markets

# 3. Fetchers (SQL -> Pipeline)
include("fetchers/schemas.jl")
include("fetchers/segments.jl")
include("fetchers/interfaces.jl")
include("fetchers/sql/matches.jl")
include("fetchers/sql/statistics.jl")
include("fetchers/sql/lineups.jl")
include("fetchers/sql/incidents.jl")
include("fetchers/sql/odds.jl")
include("fetchers/datastore.jl")

# 4. Downstream Processing
include("preprocessing.jl")
include("splitting/types.jl")
include("splitting/methods.jl")
include("splitting/display.jl")

# 5. QoL
include("display.jl")

export 
    # Types
    DBConfig, DataStore, DataTournemantSegment,
    ScottishLower, Ireland, SouthKorea,
    
    # Functions
    load_datastore_sql,
    
    # Re-export Markets
    MarketConfig, Market1X2, MarketOverUnder, MarketBTTS
    
    # (Plus whatever preprocessing/splitting exports you need)

end
