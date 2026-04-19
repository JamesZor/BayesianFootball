struct Market1X2 <: AbstractMarket end

# --- Interface ---
Base.show(io::IO, ::Market1X2) = print(io, "Market[1X2]")
market_group(::Market1X2) = "1X2" 
outcomes(::Market1X2) = (home=:home, draw=:draw, away=:away)

# --- Logic ---
function _process_market_type(raw_odds::DataFrame, m::Market1X2)
    # Using subset for cleaner syntax
    # We filter for market_group == "1X2" AND market_name == "Full time"
    sub_df = subset(raw_odds, 
        :market_group => ByRow(==("1X2")),
        :market_name => ByRow(==("Full time"))
    )
    
    keys = outcomes(m)
    # DB Mapping: "1" -> :home, "X" -> :draw, "2" -> :away
    choice_map = Dict("1" => keys.home, "X" => keys.draw, "2" => keys.away)
    
    return _build_long_rows(sub_df, choice_map, market_group(m), market_line(m))
end
