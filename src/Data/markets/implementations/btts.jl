struct MarketBTTS <: AbstractMarket end

# --- Interface ---
Base.show(io::IO, ::MarketBTTS) = print(io, "Market[BTTS]")
market_group(::MarketBTTS) = "BTTS"
outcomes(::MarketBTTS) = (yes=:btts_yes, no=:btts_no)

# --- Logic ---
function _process_market_type(raw_odds::DataFrame, m::MarketBTTS)
    sub_df = subset(raw_odds, 
        :market_name => ByRow(==("Both teams to score"))
    )

    keys = outcomes(m)
    choice_map = Dict("Yes" => keys.yes, "No" => keys.no)
    
    return _build_long_rows(sub_df, choice_map, market_group(m), market_line(m))
end
