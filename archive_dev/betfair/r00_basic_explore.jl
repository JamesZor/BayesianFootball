using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


using LibPQ

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())




db_config = Data.DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int64})
    # Note: We join `markets` to get the `market_type` string
    query = """
        SELECT 
            m.match_id,
            m.start_timestamp,
            mk.market_type,
            o.odds_data
        FROM matches m
        INNER JOIN betfair.match_meta mm ON m.match_id = mm.match_id
        INNER JOIN betfair.odds_history o ON m.match_id = o.match_id
        INNER JOIN betfair.markets mk ON o.market_id = mk.market_id
        WHERE m.tournament_id = ANY(\$1)
        AND mm.status IN ('SUCCESS', 'PARTIAL_SUCCESS')
        ORDER BY m.match_id ASC
    """
    
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

conn = Data.connect_to_db(db_config)

df = fetch_data(conn, [79])


using JSON3
using DataFrames

"""
    unpack_odds(df::DataFrame) -> DataFrame
Takes the raw SQL dataframe and expands the `odds_data` JSON strings into 
a tidy time-series DataFrame.
"""
function unpack_odds(df::DataFrame)
    # We will collect all the expanded rows here
    all_rows = DataFrame()

    for row in eachrow(df)
        # 1. Parse the JSON string
        parsed_json = JSON3.read(row.odds_data)
        
        # 2. Extract timestamps (convert from milliseconds to DateTime if needed)
        timestamps = parsed_json["timestamps"]
        
        # 3. Build a temporary DataFrame for this specific market
        temp_df = DataFrame(
            match_id = fill(row.match_id, length(timestamps)),
            market_type = fill(row.market_type, length(timestamps)),
            start_timestamp = fill(row.start_timestamp, length(timestamps)),
            odds_timestamp = timestamps
        )
        
        # 4. Add the selection arrays as columns (e.g., "home", "draw", "away")
        for (key, values) in parsed_json
            if key != "timestamps"
                temp_df[!, Symbol(key)] = values
            end
        end
        
        # Append to our master DataFrame
        append!(all_rows, temp_df, cols=:union)
    end
    
    return all_rows
end


tidy_odds_df = unpack_odds(df)


ds.odds



using DataFrames
using JSON3
using Dates
using TimeZones

"""
    get_betfair_kickoff_odds(raw_betfair_df::DataFrame) -> DataFrame
Finds the Betfair odds closest to the kickoff time and stacks them to match `ds.odds`.
"""
function get_betfair_kickoff_odds(raw_betfair_df::DataFrame)
    all_rows = DataFrame()

    for row in eachrow(raw_betfair_df)
        # Skip empty rows or non-match-odds
        if ismissing(row.odds_data) || row.market_type != "MATCH_ODDS"
            continue
        end
        
        parsed_json = JSON3.read(row.odds_data)
        timestamps = parsed_json["timestamps"]
        
        # --- THE FIX ---
        # 1. Force the ZonedDateTime into pure UTC
        dt_utc = DateTime(row.start_timestamp, Dates.UTC)
        # 2. Convert to UNIX seconds, multiply by 1000, and cast to Int64
        unix_start_ms = round(Int64, datetime2unix(dt_utc) * 1000)
        
        temp_df = DataFrame(
            match_id = fill(row.match_id, length(timestamps)),
            start_timestamp = fill(unix_start_ms, length(timestamps)), # Now an integer!
            odds_timestamp = timestamps
        )
        
        for (key, values) in parsed_json
            if key != "timestamps"
                temp_df[!, Symbol(key)] = values
            end
        end
        append!(all_rows, temp_df, cols=:union)
    end

    # --- THE MATH (Corrected) ---
    # Now that both are milliseconds, we just subtract them directly
    all_rows.time_diff_ms = abs.(all_rows.odds_timestamp .- all_rows.start_timestamp)
    
    # Group by match_id and take the row with the minimum time difference
    gdf = groupby(all_rows, :match_id)
    kickoff_df = combine(gdf) do group
        min_idx = argmin(group.time_diff_ms)
        return group[min_idx:min_idx, :] # Keep it as a DataFrame row
    end

    # Stack (Melt) the DataFrame to match ds.odds format
    stacked_df = stack(kickoff_df, [:home, :draw, :away], variable_name=:selection, value_name=:betfair_odds)
    stacked_df.selection = Symbol.(stacked_df.selection)
    
    dropmissing!(stacked_df, :betfair_odds)

    return stacked_df[:, [:match_id, :selection, :betfair_odds, :odds_timestamp, :time_diff_ms]]
end

"""
    compare_closing_odds(ds_odds::DataFrame, betfair_kickoff_df::DataFrame) -> DataFrame
Joins the Sofascore odds with Betfair kickoff odds and calculates the difference.
"""
function compare_closing_odds(ds_odds::DataFrame, betfair_kickoff_df::DataFrame)
    # Filter Sofascore data to only the Match Winner market (1X2)
    ds_1x2 = filter(row -> row.market_name == "1X2", ds_odds)
    
    # Inner join on match_id AND selection (home, draw, away)
    comparison = innerjoin(ds_1x2, betfair_kickoff_df, on=[:match_id, :selection])
    
    # --- THE FIX ---
    # 1. Drop any rows where Betfair had a JSON null (`nothing`)
    filter!(row -> !isnothing(row.betfair_odds), comparison)
    
    # 2. Force the column to be strictly Float64 so math operators work
    comparison.betfair_odds = Float64.(comparison.betfair_odds)
    
    # Calculate the Delta (Betfair - Sofascore)
    comparison.odds_delta = comparison.betfair_odds .- comparison.odds_close
    
    # Calculate the Ratio (Betfair / Sofascore)
    comparison.odds_ratio = comparison.betfair_odds ./ comparison.odds_close
    
    # Select just the relevant columns for analysis
    cols_to_keep = [
        :match_id, :date, :selection, :is_winner,
        :odds_close, :betfair_odds, :odds_delta, :odds_ratio, :time_diff_ms
    ]
    
    sort!(comparison, :date)
    return comparison[:, cols_to_keep]
end
# 1. Get your raw SQL data (using the fixed query from earlier)
# raw_betfair = fetch_data(conn, [79])

# 2. Extract the Betfair odds exactly at kickoff
betfair_kickoff = get_betfair_kickoff_odds(df)

# 3. Compare them against your Bayesian DataStore!
analysis_df = compare_closing_odds(ds.odds, betfair_kickoff)


# 4. View the results
display(analysis_df)


combine(groupby(analysis_df, :selection), :odds_ratio => mean)



using DataFrames
using JSON3
using Dates
using TimeZones

"""
    get_all_kickoff_odds(raw_betfair_df::DataFrame) -> DataFrame
Finds the Betfair odds closest to kickoff for EVERY market type and selection.
"""
function get_all_kickoff_odds(raw_betfair_df::DataFrame)
    # Pre-allocate arrays for maximum performance
    match_ids = Int[]
    market_types = String[]
    selections = Symbol[]
    odds = Float64[]
    time_diffs = Int64[]

    for row in eachrow(raw_betfair_df)
        if ismissing(row.odds_data)
            continue
        end

        parsed_json = JSON3.read(row.odds_data)
        timestamps = parsed_json["timestamps"]
        
        # 1. Convert Start Time to UNIX milliseconds
        dt_utc = DateTime(row.start_timestamp, Dates.UTC)
        unix_start_ms = round(Int64, datetime2unix(dt_utc) * 1000)
        
        # 2. Find the exact index closest to kickoff
        time_diff_array = abs.(timestamps .- unix_start_ms)
        min_idx = argmin(time_diff_array)
        best_diff = time_diff_array[min_idx]

        # 3. Extract the odds for EVERY selection at that specific index
        for (key, values) in parsed_json
            if key != "timestamps"
                val = values[min_idx]
                
                # Only save if there was actual money traded at this minute
                if !isnothing(val) 
                    push!(match_ids, row.match_id)
                    push!(market_types, row.market_type)
                    push!(selections, Symbol(key))
                    push!(odds, Float64(val))
                    push!(time_diffs, best_diff)
                end
            end
        end
    end

    # Return a beautiful, tidy DataFrame
    return DataFrame(
        match_id = match_ids,
        market_type = market_types,
        selection = selections,
        betfair_odds = odds,
        time_diff_ms = time_diffs
    )
end

"""
    compare_all_closing_odds(ds_odds::DataFrame, betfair_kickoff_df::DataFrame) -> DataFrame
Joins Sofascore odds with Betfair kickoff odds across ALL market types (1X2, O/U, BTTS).
"""
function compare_all_closing_odds(ds_odds::DataFrame, betfair_kickoff_df::DataFrame)
    
    # 1. Create a clean copy of the Betfair data
    bf_clean = filter(row -> row.selection != :timestamps, betfair_kickoff_df)
    
    # Drop any rows where Betfair had a JSON null (`nothing`)
    filter!(row -> !isnothing(row.betfair_odds) && !ismissing(row.betfair_odds), bf_clean)
    bf_clean.betfair_odds = Float64.(bf_clean.betfair_odds)
    
    # 2. Standardize the Over/Under strings!
    # This turns ":over_2_5" into ":over_25" to perfectly match ds.odds
    bf_clean.selection = map(bf_clean.selection) do sel
        str_sel = string(sel)
        # Regex: find an underscore, a digit, an underscore, a digit (e.g. _2_5)
        # and replace it with underscore and the two digits (e.g. _25)
        clean_str = replace(str_sel, r"_(\d)_(\d)" => s"_\1\2")
        return Symbol(clean_str)
    end

    # 3. Inner join on match_id AND the newly matching selection symbol!
    # Because :home, :over_25, :under_25 are all globally unique per match, 
    # this safely joins all markets at once without needing a market_type column.
    comparison = innerjoin(ds_odds, bf_clean, on=[:match_id, :selection])
    
    # 4. Calculate the Delta and Ratio
    comparison.odds_delta = comparison.betfair_odds .- comparison.odds_close
    comparison.odds_ratio = comparison.betfair_odds ./ comparison.odds_close
    
    # 5. Select relevant columns to view
    cols_to_keep = [
        :match_id, :date, :market_name, :market_line, :selection, 
        :odds_close, :betfair_odds, :odds_delta, :odds_ratio, :time_diff_ms
    ]
    
    # Sort nicely by Date, then Match, then Market (1X2, OverUnder, etc.)
    sort!(comparison, [:date, :match_id, :market_name, :selection])
    
    return comparison[:, cols_to_keep]
end

kickoff_odds_df = get_all_kickoff_odds(df)



all_markets_analysis = compare_all_closing_odds(ds.odds, kickoff_odds_df)

display(all_markets_analysis)


combine(groupby(all_markets_analysis, :market_name), :odds_ratio => mean)


ou_only = filter(row -> row.market_name == "OverUnder", all_markets_analysis)
combine(groupby(ou_only, :selection), :odds_ratio => mean)


using DataFrames
using JSON3
using Dates
using TimeZones

"""
    get_all_kickoff_odds(raw_betfair_df::DataFrame) -> DataFrame
Finds the Betfair odds closest to kickoff for EVERY market type and selection.
Automatically filters out nulls and ignores timestamp keys.
"""
function get_all_kickoff_odds(raw_betfair_df::DataFrame)
    # Pre-allocate arrays for maximum performance
    match_ids = Int[]
    market_types = String[]
    selections = Symbol[]
    odds = Float64[]
    time_diffs = Int64[]

    for row in eachrow(raw_betfair_df)
        if ismissing(row.odds_data)
            continue
        end

        parsed_json = JSON3.read(row.odds_data)
        timestamps = parsed_json["timestamps"]
        
        # 1. Convert Start Time to UNIX milliseconds
        dt_utc = DateTime(row.start_timestamp, Dates.UTC)
        unix_start_ms = round(Int64, datetime2unix(dt_utc) * 1000)
        
        # 2. Find the exact index closest to kickoff
        time_diff_array = abs.(timestamps .- unix_start_ms)
        min_idx = argmin(time_diff_array)
        best_diff = time_diff_array[min_idx]

        # 3. Extract the odds for EVERY selection at that specific index
        for (key, values) in parsed_json
            # STRICTLY ignore the timestamps array leaking into selections
            if key != "timestamps"
                val = values[min_idx]
                
                # Only save if there was actual money traded at this minute
                if !isnothing(val) 
                    push!(match_ids, row.match_id)
                    push!(market_types, row.market_type)
                    push!(selections, Symbol(key))
                    push!(odds, Float64(val))
                    push!(time_diffs, best_diff)
                end
            end
        end
    end

    # Return a tidy DataFrame
    return DataFrame(
        match_id = match_ids,
        market_type = market_types,
        selection = selections,
        betfair_odds = odds,
        time_diff_ms = time_diffs
    )
end



"""
    compare_all_closing_odds(ds_odds::DataFrame, betfair_kickoff_df::DataFrame) -> DataFrame
Joins Sofascore odds with Betfair kickoff odds across ALL market types.
Standardizes selection strings to ensure perfect joins.
"""
function compare_all_closing_odds(ds_odds::DataFrame, betfair_kickoff_df::DataFrame)
    
    # 1. Create a working copy of the Betfair data
    bf_clean = copy(betfair_kickoff_df)
    
    # 2. Standardize the Over/Under strings!
    # This turns ":over_2_5" into ":over_25" to perfectly match ds.odds
    bf_clean.selection = map(bf_clean.selection) do sel
        str_sel = string(sel)
        # Regex: find an underscore, a digit, an underscore, a digit (e.g. _2_5)
        # and replace it with an underscore and the two digits (e.g. _25)
        clean_str = replace(str_sel, r"_(\d)_(\d)" => s"_\1\2")
        return Symbol(clean_str)
    end

    # 3. Inner join on match_id AND the matching selection symbol
    # This will naturally drop games where the bookmaker hasn't posted an O/U line yet
    comparison = innerjoin(ds_odds, bf_clean, on=[:match_id, :selection])
    
    # 4. Calculate the Delta and Ratio
    comparison.odds_delta = comparison.betfair_odds .- comparison.odds_close
    comparison.odds_ratio = comparison.betfair_odds ./ comparison.odds_close
    
    # 5. Select relevant columns to view
    cols_to_keep = [
        :match_id, :date, :market_name, :market_line, :selection, 
        :odds_close, :betfair_odds, :odds_delta, :odds_ratio, :time_diff_ms
    ]
    
    # Sort nicely by Date, then Match, then Market
    sort!(comparison, [:date, :match_id, :market_name, :selection])
    
    return comparison[:, cols_to_keep]
end



# 1. Expand the raw JSON
kickoff_odds_df = get_all_kickoff_odds(df)

# 2. Join it against your Bayseian DataStore
all_markets_analysis = compare_all_closing_odds(ds.odds, kickoff_odds_df)

# 3. View the vig calculation for every single market!
combine(groupby(all_markets_analysis, [:market_name, :selection]), :odds_ratio => mean)
