# src/predictions/market_data.jl

"""
    prepare_market_data(ds::DataStore, config::PredictionConfig)

Extracts, cleans, and processes market data into a standardized 'Long' format.
Calculates implied probabilities and removes vig (using basic normalization).
"""
function prepare_market_data(ds::DataStore, config::PredictionConfig)
    
    # 1. Filter and Clean Raw Odds per Market
    # We process each market type separately to handle specific naming conventions (1X2 vs Over/Under)
    dfs = DataFrame[]
    
    for market in config.markets
        # Dispatch to specific logic for 1X2, OverUnder, etc.
        push!(dfs, _process_market_type(ds.odds, market))
    end
    
    # Combine all markets into one big long DataFrame
    long_df = vcat(dfs...)

    # 2. Calculate Implied Probabilities (Raw)
    # 1 / Decimal Odds
    long_df.prob_implied_open = 1.0 ./ long_df.odds_open
    long_df.prob_implied_close = 1.0 ./ long_df.odds_close

    # 3. Calculate Vig & Fair Probabilities
    # We group by the market instance (Match + Market Name + Line) to sum the probabilities
    gdf = groupby(long_df, [:match_id, :market_name, :market_line])
    
    # Calculate the total probability sum (the overround) for each market
    # e.g. Home(0.4) + Draw(0.3) + Away(0.35) = 1.05
    transform!(gdf, 
        :prob_implied_open => sum => :overround_open,
        :prob_implied_close => sum => :overround_close
    )

    # Normalize to get Fair Probabilities (Probs sum to 1.0)
    # prob_fair = prob_implied / overround
    long_df.prob_fair_open = long_df.prob_implied_open ./ long_df.overround_open
    long_df.prob_fair_close = long_df.prob_implied_close ./ long_df.overround_close
    
    # Optional: Calculate specific vig amount if needed later
    # long_df.vig_open = long_df.overround_open .- 1.0

    return MarketData(long_df, config)
end

# ==============================================================================
# Internal Dispatch Helpers
# ==============================================================================

# --- 1X2 Market ---
function _process_market_type(raw_odds::DataFrame, market::Markets.Market1X2)
    # Filter for standard 1X2 rows
    subset = filter(row -> 
        row.market_group == "1X2" && 
        row.market_name == "Full time", 
    raw_odds)
    
    # Map DB choices "1", "X", "2" to standard Symbols :home, :draw, :away
    keys = Markets.market_keys(market) # returns (home=:home, draw=:draw, away=:away)
    choice_map = Dict("1" => keys.home, "X" => keys.draw, "2" => keys.away)
    
    # Build the standardized rows
    return _build_long_rows(subset, choice_map, "1X2", 0.0)
end

# --- Over/Under Market ---
function _process_market_type(raw_odds::DataFrame, market::Markets.MarketOverUnder)
    # Filter for specific line (e.g., 2.5)
    # Note: Assumes choice_group is consistent with market.line (float vs string check might be needed depending on DB)
    subset = filter(row -> 
        row.market_name == "Match goals" && 
        _safe_equals(row.choice_group, market.line), 
    raw_odds)

    # Map DB choices "Over", "Under" to :over, :under
    keys = Markets.market_keys(market)
    choice_map = Dict("Over" => keys.over, "Under" => keys.under)
    
    return _build_long_rows(subset, choice_map, "OverUnder", market.line)
end

# --- BTTS Market ---
function _process_market_type(raw_odds::DataFrame, market::Markets.MarketBTTS)
    subset = filter(row -> 
        row.market_name == "Both teams to score",
    raw_odds)

    keys = Markets.market_keys(market)
    choice_map = Dict("Yes" => keys.yes, "No" => keys.no)
    
    return _build_long_rows(subset, choice_map, "BTTS", 0.0)
end

# ==============================================================================
# Builders & Utilities
# ==============================================================================

function _build_long_rows(df::DataFrame, choice_map::Dict, mkt_name::String, line::Float64)
    if isempty(df)
        return DataFrame(match_id=Int[], market_name=String[], market_line=Float64[], 
                         selection=Symbol[], odds_open=Float64[], odds_close=Float64[], is_winner=Bool[])
    end

    # Safe fractional parser
    parse_odds(x) = x isa Number ? Float64(x) : _frac_to_dec(string(x))
    
    out_df = DataFrame()
    out_df.match_id = df.match_id
    out_df.market_name = fill(mkt_name, nrow(df))
    out_df.market_line = fill(line, nrow(df))
    
    # Map choice names to standardized Symbols using the provided map
    # Fallback to Symbol(c) if not in map, but usually map covers all
    out_df.selection = [get(choice_map, c, Symbol(c)) for c in df.choice_name]
    
    # Parse odds
    out_df.odds_open = parse_odds.(df.initial_fractional_value)
    out_df.odds_close = parse_odds.(df.final_fractional_value)
    
    # Outcome
    out_df.is_winner = df.winning
    
    return out_df
end

"""
Helper to safely compare the choice_group (which might be mixed types) to the market line
"""
function _safe_equals(val, target::Float64)
    if val isa Number
        return val == target
    else
        # Try parsing if it's a string, or return false
        try
            return parse(Float64, string(val)) == target
        catch
            return false
        end
    end
end

"""
Converts fractional odds strings ("1/2", "evens") to decimal floats.
"""
function _frac_to_dec(s::String)
    s_clean = strip(lowercase(s))
    
    if s_clean == "evens" || s_clean == "even"
        return 2.0
    elseif occursin("/", s_clean)
        parts = split(s_clean, "/")
        if length(parts) == 2
            num = parse(Float64, parts[1])
            den = parse(Float64, parts[2])
            return 1.0 + (num / den)
        end
    end
    
    # Fallback: try parsing as direct decimal
    return parse(Float64, s_clean)
end
