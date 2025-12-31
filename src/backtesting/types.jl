struct BacktestLedger
    df::DataFrame
    # Columns: 
    # :match_id, :market_name, :selection, :is_winner
    # :signal_name, :signal_params, :odds_type, :odds, :stake
    # :model_name, :model_parameters
end

# Forward standard DataFrame methods for ease of use
Base.show(io::IO, bl::BacktestLedger) = show(io, bl.df)
Base.length(bl::BacktestLedger) = nrow(bl.df)
Base.getindex(bl::BacktestLedger, args...) = getindex(bl.df, args...)
