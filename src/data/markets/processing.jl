# src/data/markets/processing.jl

"""
    prepare_market_data(ds, config::MarketConfig)
Main entry point. Orchestrates extraction, cleaning, and enrichment.
"""
function prepare_market_data(ds; config::MarketConfig = DEFAULT_MARKET_CONFIG)
    dfs = DataFrame[]
    
    # 1. Extract & Standardize per Market
    for market in config.markets
        # Dispatch to specific implementation
        raw_df = _process_market_type(ds.odds, market)
        
        if !isempty(raw_df)
            push!(dfs, raw_df)
        end
    end
    
    if isempty(dfs)
        @warn "No market data found for the provided configuration."
        return MarketData(DataFrame(), config)
    end
    
    long_df = vcat(dfs...)

    # 2. Enrich (Probabilities, Vig, Fair Odds)
    _enrich_market_data!(long_df)

    return MarketData(long_df, config)
end

"""
    _enrich_market_data!(df::DataFrame)
Calculates implied probabilities, removes vig (normalization), and adds CLM.
"""
function _enrich_market_data!(df::DataFrame)
    # --- A. Implied Probabilities (Raw) ---
    # 1 / Decimal Odds
    df.prob_implied_open = 1.0 ./ df.odds_open
    df.prob_implied_close = 1.0 ./ df.odds_close

    # --- B. Calculate Vig & Fair Probabilities ---
    # We group by the specific market instance to sum probabilities
    # Grouping keys: match_id + market_name + market_line
    gdf = groupby(df, [:match_id, :market_name, :market_line])
    
    # Calculate Overround (Sum of implied probs)
    # We use transform! to broadcast the sum back to every row in the group
    transform!(gdf, 
        :prob_implied_open => sum => :overround_open,
        :prob_implied_close => sum => :overround_close
    )

    # Calculate Fair Probabilities (Normalize to 1.0)
    df.prob_fair_open = df.prob_implied_open ./ df.overround_open
    df.prob_fair_close = df.prob_implied_close ./ df.overround_close

    df.fair_odds_open =  1 ./ df.prob_fair_open 
    df.fair_odds_close =  1 ./ df.prob_fair_close
    
    # Calculate Vig (Margin)
    df.vig_open = df.overround_open .- 1.0
    df.vig_close = df.overround_close .- 1.0

    # --- C. Closing Line Movement (CLM) ---
    # Metric: Difference in fair probability (Close - Open)
    df.clm_prob = df.prob_fair_close .- df.prob_fair_open
    
    # Metric: Raw odds movement
    df.clm_odds = df.odds_close .- df.odds_open

    return df
end

# --- Abstract Builder (Used by implementations) ---

function _build_long_rows(df::DataFrame, choice_map::Dict, mkt_name::String, line::Float64)
    if isempty(df)
        # Return empty schema
        return DataFrame(
            match_id=Int[], market_name=String[], market_line=Float64[], 
            selection=Symbol[], odds_open=Float64[], odds_close=Float64[], 
            is_winner=Bool[]
        )
    end

    parse_odds(x) = x isa Number ? Float64(x) : _frac_to_dec(string(x))
    
    # Construct standard columns
    out_df = DataFrame()
    out_df.match_id = df.match_id
    out_df.market_name = fill(mkt_name, nrow(df))
    out_df.market_line = fill(line, nrow(df))
    
    # Map choices
    out_df.selection = [get(choice_map, c, Symbol(c)) for c in df.choice_name]
    
    # Parse Odds
    out_df.odds_open = parse_odds.(df.initial_fractional_value)
    out_df.odds_close = parse_odds.(df.final_fractional_value)
    
    out_df.is_winner = df.winning
    
    return out_df
end

# Fallback
_process_market_type(df, m) = error("Implementation missing for $(typeof(m))")
