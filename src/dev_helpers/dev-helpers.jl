# src/dev_helper/dev-helpers.jl

"""
    load_scottish_data(preset::Symbol=:s2425; split_week::Int=14)

Helper function to load, filter, and clean Scottish football data for development experiments.
Applies specific cleaning to remove corrupted odds feeds where the open/close ratio is extreme.
"""
function load_scottish_data(season_str::String; split_week::Int=14)
    # 1. Load the default data store
    data_store = Data.load_default_datastore()

    # Determine season string from symbol

    # 2. Filter Matches for the specific season and add match week
    df_matches = filter(row -> row.season == season_str, data_store.matches)
    df_matches = Data.add_match_week_column(df_matches)

    # 3. Process Odds
    # Work on a copy of the odds to avoid mutating the cached store if re-used
    df_odds = copy(data_store.odds)
    Data.DataPreprocessing.add_inital_odds_from_fractions!(df_odds)

    # 4. Filter Corrupt Odds
    # Logic: Identify (Match, MarketGroup) pairs where the open/close ratio is outside bounds.
    # Ratio = (Final - Initial) / Initial
    
    # Calculate ratio (handling cases where initial might be 0 to avoid Inf)
    # Note: We use a safe division or check, assuming add_inital_odds_from_fractions! handles parsing.
    # We calculate the ratio column temporarily for filtering.
    df_odds[!, :open_close_ratio] = (df_odds.decimal_odds .- df_odds.initial_decimal) ./ df_odds.initial_decimal

    # Define corruption criteria: ratio < -1.0 or ratio > 0.6
    is_corrupt(r) = !ismissing(r) && (r < -1.0 || r > 0.6)

    # Find the "Bad Keys" (MatchID, MarketGroup) that contain ANY corrupt value
    # We explicitly filter for rows that satisfy the corrupt condition
    bad_rows = filter(row -> is_corrupt(row.open_close_ratio), df_odds)
    
    # Create a Set of (match_id, market_group) tuples for fast lookup
    bad_keys_set = Set{Tuple{Int, String}}()
    for row in eachrow(bad_rows)
        push!(bad_keys_set, (row.match_id, row.market_group))
    end

    # Filter the original odds dataframe: Keep rows NOT in the bad keys set
    odds_clean = filter(row -> !((row.match_id, row.market_group) in bad_keys_set), df_odds)

    # 5. Re-assemble DataStore with cleaned odds
    ds_clean = Data.DataStore(
        df_matches,
        odds_clean,
        data_store.incidents
    )

    # 6. Apply Split Column
    ds_final = Data.DataPreprocessing.add_split_col_match_week(ds_clean, split_week)

    return ds_final
end
