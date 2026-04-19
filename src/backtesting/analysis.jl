
using Statistics

export generate_tearsheet, summarize_models, summarize_signals, summarize_markets

# --- Core Metrics Helper ---

"""
    _calc_metrics(df::DataFrame)

Internal function to aggregate PnL, ROI, and Win Rate from a set of bets.
Only considers rows where stake > 0.
"""
function _calc_metrics(df::AbstractDataFrame)
    # Filter for active bets
    active_bets = filter(r -> r.stake > 1e-6, df)
    
    if nrow(active_bets) == 0
        return (bets=0, turnover=0.0, profit=0.0, roi=0.0, win_rate=0.0, avg_odds=0.0)
    end

    # Calculate PnL for each bet
    # Profit = Stake * (Odds - 1) if Won, else -Stake
    pnl_vec = map(eachrow(active_bets)) do r
        if ismissing(r.is_winner) return 0.0 end
        r.is_winner ? r.stake * (r.odds - 1.0) : -r.stake
    end

    turnover = sum(active_bets.stake)
    profit = sum(pnl_vec)
    roi = turnover > 0 ? (profit / turnover) * 100 : 0.0
    
    # Win Rate (ignoring voids/missing)
    outcomes = skipmissing(active_bets.is_winner)
    win_rate = isempty(outcomes) ? 0.0 : mean(outcomes) * 100
    
    avg_odds = mean(active_bets.odds)

    return (
        bets = nrow(active_bets),
        turnover = round(turnover, digits=2),
        profit = round(profit, digits=2),
        roi = round(roi, digits=2),
        win_rate = round(win_rate, digits=1),
        avg_odds = round(avg_odds, digits=2)
    )
end

# --- Tear Sheet Generators ---

"""
    generate_tearsheet(ledger::BacktestLedger)

Returns a comprehensive summary dataframe grouping by Model and Signal.
Useful for comparing "StaticPoisson + Kelly" vs "StaticPoisson_Cauchy + Flat".
"""
function generate_tearsheet(ledger::BacktestLedger)
    # Group by the combination of Model Config and Signal Strategy
    gdf = groupby(ledger.df, [:model_name, :model_parameters, :signal_name, :signal_params])
    
    # Apply metrics
    results = combine(gdf) do sub_df
        m = _calc_metrics(sub_df)
        DataFrame(
            bets = m.bets,
            turnover = m.turnover,
            profit = m.profit,
            roi_pct = m.roi,
            win_rate_pct = m.win_rate,
            avg_odds = m.avg_odds
        )
    end
    
    # Sort by Profit to see best performers on top
    sort!(results, :profit, rev=true)
    return results
end

"""
    summarize_models(ledger::BacktestLedger)

Aggregates performance purely by Model (ignoring which Signal was used).
Useful to see if the Cauchy Prior is generally better than Normal Prior.
"""
function summarize_models(ledger::BacktestLedger)
    gdf = groupby(ledger.df, [:model_name, :model_parameters])
    
    results = combine(gdf) do sub_df
        m = _calc_metrics(sub_df)
        DataFrame(
            bets = m.bets,
            profit = m.profit,
            roi_pct = m.roi,
            win_rate_pct = m.win_rate
        )
    end
    sort!(results, :profit, rev=true)
end

"""
    summarize_signals(ledger::BacktestLedger)

Aggregates performance purely by Signal Strategy.
Useful to check if Kelly is too volatile compared to Flat staking.
"""
function summarize_signals(ledger::BacktestLedger)
    gdf = groupby(ledger.df, [:signal_name, :signal_params])
    
    results = combine(gdf) do sub_df
        m = _calc_metrics(sub_df)
        DataFrame(
            bets = m.bets,
            profit = m.profit,
            roi_pct = m.roi,
            win_rate_pct = m.win_rate
        )
    end
    sort!(results, :roi_pct, rev=true)
end

"""
    summarize_markets(ledger::BacktestLedger; market_type=nothing)

Drill down into Market performance (e.g., 1x2 Home vs Away, Over vs Under).
Optional `market_type` filter (e.g., "1X2", "OverUnder").
"""
function summarize_markets(ledger::BacktestLedger; market_type=nothing)
    df = ledger.df
    
    if !isnothing(market_type)
        df = filter(r -> r.market_name == market_type, df)
    end

    # Group by Market Name and Selection (e.g., 1X2 -> Draw)
    gdf = groupby(df, [:market_name, :selection])
    
    results = combine(gdf) do sub_df
        m = _calc_metrics(sub_df)
        DataFrame(
            bets = m.bets,
            profit = m.profit,
            roi_pct = m.roi,
            win_rate_pct = m.win_rate,
            avg_odds = m.avg_odds
        )
    end
    
    sort!(results, :roi_pct, rev=true)
end



"""
    summarize_markets(ledger::BacktestLedger; 
                      market_type=nothing, 
                      compare_models::Bool=false)

Drill down into Market performance.

# Arguments
- `market_type`: Optional string to filter by specific market (e.g., "1X2").
- `compare_models`: If true, groups results by Model + Market. If false (default), aggregates all models together to show general market difficulty.
"""
function summarize_markets(ledger::BacktestLedger; 
                           market_type=nothing, 
                           compare_models::Bool=false)
    df = ledger.df
    
    # 1. Optional Filtering
    if !isnothing(market_type)
        df = filter(r -> r.market_name == market_type, df)
    end

    # 2. Define Grouping Columns
    # Always group by Market and Selection
    group_cols = [:market_name, :selection]
    
    # If requested, add Model context
    if compare_models
        pushfirst!(group_cols, :model_parameters)
        pushfirst!(group_cols, :model_name)
    end

    # 3. Aggregation
    gdf = groupby(df, group_cols)
    
    results = combine(gdf) do sub_df
        m = _calc_metrics(sub_df)
        DataFrame(
            bets = m.bets,
            profit = m.profit,
            roi_pct = m.roi,
            win_rate_pct = m.win_rate,
            avg_odds = m.avg_odds
        )
    end
    
    # 4. Sorting
    # If comparing models, we usually want to sort by Market first to compare side-by-side, 
    # then by Profit to see the winner.
    if compare_models
        sort!(results, [:market_name, :selection, :profit], rev=[false, false, true])
    else
        sort!(results, :roi_pct, rev=true)
    end
    
    return results
end



"""
    detailed_breakdown(ledger::BacktestLedger)

Returns the most granular summary possible:
Grouped by Model -> Signal -> Market -> Selection.
Useful for debugging exactly where a strategy is failing.
"""
function detailed_breakdown(ledger::BacktestLedger)
    df = ledger.df

    # Define the full hierarchy of grouping
    group_cols = [
        :model_name, :model_parameters, 
        :signal_name, :signal_params, 
        :market_name, :selection
    ]
    
    gdf = groupby(df, group_cols)
    
    results = combine(gdf) do sub_df
        m = _calc_metrics(sub_df)
        DataFrame(
            bets = m.bets,
            profit = m.profit,
            roi_pct = m.roi,
            win_rate_pct = m.win_rate,
            avg_odds = m.avg_odds
        )
    end
    
    # Sort by Profit (ascending) to see the biggest leaks immediately
    sort!(results, :profit)
    
    return results
end
