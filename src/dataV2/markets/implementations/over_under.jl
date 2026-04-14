struct MarketOverUnder <: AbstractMarket
    line::Float64
end

# --- Interface ---
Base.show(io::IO, m::MarketOverUnder) = print(io, "Market[O/U $(m.line)]")
market_group(::MarketOverUnder) = "OverUnder"
market_line(m::MarketOverUnder) = m.line

function outcomes(m::MarketOverUnder)
    # Dynamic symbols: :over_25, :under_25
    l_str = replace(string(m.line), "." => "")
    (over=Symbol("over_", l_str), under=Symbol("under_", l_str))
end

# --- Logic ---
function _process_market_type(raw_odds::DataFrame, m::MarketOverUnder)
    # Filter for "Match goals" AND matching line
    # Note: We use a closure for the line comparison to handle string/float mix
    target_line = m.line
    
    sub_df = subset(raw_odds, 
        :market_name => ByRow(==("Match goals")),
        :choice_group => ByRow(x -> _safe_equals(x, target_line))
    )

    keys = outcomes(m)
    choice_map = Dict("Over" => keys.over, "Under" => keys.under)
    
    return _build_long_rows(sub_df, choice_map, market_group(m), market_line(m))
end
