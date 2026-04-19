struct MarketDC <: AbstractMarket end 

# --- Interface ---
Base.show(io::IO, ::MarketDC) = print(io, "Market[DC]")
market_group(::MarketDC) = "DoubleChance"
outcomes(::MarketDC) = (dc1x=:DC_1X, dcx2=:DC_X2, dc12=:DC_12)

# --- Logic ---

function _process_market_type(raw_odds::DataFrame, m::MarketDC)
    sub_df = subset(raw_odds, 
        :market_group => ByRow(==("Double chance")),
        :market_name => ByRow(==("Double chance"))
    )
    
    keys = outcomes(m)
    # DB Mapping: "1" -> :home, "X" -> :draw, "2" -> :away
    choice_map = Dict("1X" => keys.dc1x, "X2" => keys.dcx2, "12" => keys.dc12)
    
    return _build_long_rows(sub_df, choice_map, market_group(m), market_line(m))
end

