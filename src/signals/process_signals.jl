# src/signals/process_signals.jl

using DataFrames

"""
    process_signals(ppd::PPD, market_data::DataFrame, signals::Vector{<:AbstractSignal}; 
                    odds_column::Symbol=:close_odds)

Applies a vector of signal strategies to the PPD (Predictions) using the specified 
odds column from the market data.

# Arguments
- `ppd`: The Posterior Predictive Distribution struct.
- `market_data`: DataFrame containing market odds (must include match_id, market_name, selection).
- `signals`: Vector of strategies to apply.
- `odds_column`: The column in `market_data` to use for odds (e.g., :close_odds, :fair_open_odds).
"""
function process_signals(ppd::PPD, market_data::DataFrame, signals::Vector{<:AbstractSignal}; 
                         odds_column::Symbol=:close_odds)
    
    # 1. Validation
    if !hasproperty(market_data, odds_column)
        error("Market data does not contain requested odds column: :$odds_column")
    end

    # 2. Join PPD (Probs) with Market Data (Odds)
    # We select only necessary columns to keep the join clean
    cols_to_keep = unique([:match_id, :date, :market_name, :selection, :is_winner, odds_column])
    
    # Inner join: We can only bet if we have BOTH a prediction and a market line
    joined_df = innerjoin(
        ppd.df, 
        select(market_data, cols_to_keep), 
        on = [:match_id, :market_name, :selection]
    )

    results = DataFrame()

    # 3. Iterate over every signal strategy provided
    for signal in signals
        sig_name = signal_name(signal)
        sig_params = signal_parameters(signal)
        
        # Define a kernel to apply to each row
        function calculate_row_stake(row)
            odds = row[odds_column]
            
            # Guard clauses for bad data
            if ismissing(odds) || odds <= 1.0 
                return 0.0 
            end
            
            return compute_stake(signal, row.distribution, odds)
        end

        # Map the kernel over the rows
        stakes = map(calculate_row_stake, eachrow(joined_df))
        
        # 4. Construct Result Block
        temp_df = select(joined_df, [:match_id, :date, :market_name, :selection, :is_winner])
        
        # Metadata columns
        temp_df.signal_name = fill(sig_name, nrow(temp_df))
        temp_df.signal_params = fill(sig_params, nrow(temp_df))
        temp_df.odds_type = fill(string(odds_column), nrow(temp_df))
        temp_df.odds = joined_df[!, odds_column]
        
        # The calculated stake
        temp_df.stake = stakes
        
        append!(results, temp_df)
    end

    return SignalsResult(results, ppd.model, signals)
end
