using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


using LibPQ

ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


## --------
# connect to the database and grab the data
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

function featch_betfair_market_data(ds::Data.DataStore)

  db_config = Data.DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
  conn = Data.connect_to_db(db_config)
  df = fetch_data(conn, Data.tournament_ids(ds.segment))
  return df

end

# example use 
df = featch_betfair_market_data(ds)



## ---- 
# processing betfair data 

using DataFrames
using JSON3
using Dates
using TimeZones


# dev area 
rowdata = first(df)
kickoff_time = DateTime(rowdata.start_timestamp, Dates.UTC)

odds_json = JSON3.read(rowdata.odds_data)



times = unix2datetime.(odds_json["timestamps"] / 1000)

diff = abs.(times .- kickoff_time)

kickoff_idx = argmin(diff)
best_time = times[kickoff_idx]

odds_json[:over_2_5][kickoff_idx]
odds_json[:under_2_5][kickoff_idx]
odds_json






#=
Phase 1: Unpacking to a Granular "Long" Format
=#

using DataFrames
using JSON3
using Dates
using TimeZones

"""
Helper to map Betfair market_type to ds.odds taxonomy
"""
function map_market_info(m_type::String)
    if startswith(m_type, "OVER_UNDER_")
        # Extract "25" from "OVER_UNDER_25" and convert to 2.5
        line_val = parse(Float64, replace(m_type, "OVER_UNDER_" => "")) / 10.0
        return "OverUnder", line_val
    elseif m_type == "MATCH_ODDS"
        return "1X2", 0.0
    elseif m_type == "BOTH_TEAMS_TO_SCORE"
        return "BTTS", 0.0
    elseif m_type == "CORRECT_SCORE"
        return "CorrectScore", 0.0
    end
    return m_type, 0.0
end

"""
Helper to map selection symbols to ds.odds taxonomy
"""
function map_selection_symbol(sel::Symbol)
    s = string(sel)
    
    # 1. Handle Over/Under: over_2_5 -> :over_25
    if startswith(s, "over_") || startswith(s, "under_")
        parts = split(s, "_")
        # parts[1] is "over", join(parts[2:end]) is "25"
        return Symbol(parts[1], "_", join(parts[2:end]))
    
    # 2. Handle Correct Scores: "0_0" -> :cs_00
    elseif occursin(r"^\d+_\d+$", s) 
        return Symbol("cs_", replace(s, "_" => ""))
    
    # 3. Handle Any Other: "any_other_home" -> :cs_any_other_home
    elseif startswith(s, "any_other")
        return Symbol("cs_", s)
    
    # 4. Standard lines (home, away, draw, btts_yes) remain as is
    else
        return sel
    end
end

function unpack_betfair_odds(raw_df::DataFrame)
    long_data = NamedTuple{
        (:match_id, :market_name, :market_line, :selection, :timestamp, :minutes_to_kickoff, :traded_price), 
        Tuple{Int32, String, Float64, Symbol, DateTime, Float64, Float64}
    }[]

    for row in eachrow(raw_df)
        ismissing(row.odds_data) && continue
        odds_json = JSON3.read(row.odds_data)
        !haskey(odds_json, :timestamps) && continue
        
        # Get standard taxonomy info
        market_name, market_line = map_market_info(row.market_type)
        
        kickoff_dt = DateTime(row.start_timestamp, Dates.UTC)
        kickoff_unix_ms = datetime2unix(kickoff_dt) * 1000.0
        raw_timestamps = odds_json[:timestamps]

        for (selection_key, odds_array) in pairs(odds_json)
            selection_key == :timestamps && continue
            
            # Map the selection symbol
            clean_selection = map_selection_symbol(selection_key)
            
            for (i, price) in enumerate(odds_array)
                price === nothing && continue
                
                ts_ms = raw_timestamps[i]
                mins_to_ko = (ts_ms - kickoff_unix_ms) / 60000.0
                
                push!(long_data, (
                    match_id = row.match_id,
                    market_name = market_name,
                    market_line = market_line,
                    selection = clean_selection,
                    timestamp = unix2datetime(ts_ms / 1000.0),
                    minutes_to_kickoff = mins_to_ko,
                    traded_price = Float64(price)
                ))
            end
        end
    end
    return DataFrame(long_data)
end

# Usage:
long_df = unpack_betfair_odds(df)

#=
Phase 3: Data Cleaning & Smoothing 
=#

# Define a trait-based system for price estimation
abstract type PriceEstimator end

struct MedianEstimator <: PriceEstimator end
struct MeanEstimator   <: PriceEstimator end
struct TWAEstimator    <: PriceEstimator end # Time-Weighted Average

# 1. Median: Robust to outliers
estimate_price(::MedianEstimator, prices, mins) = median(prices)

# 2. Mean: Simple average
estimate_price(::MeanEstimator, prices, mins) = mean(prices)

# 3. TWA: Weights prices by how long they stayed active
function estimate_price(::TWAEstimator, prices, mins)
    # Sort to ensure chronological order
    p = sortperm(mins)
    sorted_p = prices[p]
    sorted_m = mins[p]
    
    if length(sorted_m) < 2
        return sorted_p[1]
    end

    # Explicitly use Base.diff to avoid collision with your 'diff' variable
    durations = Base.diff(sorted_m)
    
    # We need durations to match the length of prices. 
    # We'll assume the last price was valid for the average duration of previous ticks.
    push!(durations, mean(durations)) 
    
    return sum(sorted_p .* durations) / sum(durations)
end


using Statistics

"""
Summarizes the long_df into a ds.odds-style structure using a specific window and estimator.
"""
function summarize_odds(
    df_long::DataFrame, 
    estimator::PriceEstimator; 
    window::Tuple{Float64, Float64} = (-10.0, 0.0), # e.g., 10 mins before KO to KO
    min_ticks::Int = 1,
    overround_limits::Tuple{Float64, Float64} = (0.9, 1.10) # Efficiency filter
)
    # 1. Filter for the time window
    df_window = filter(row -> window[1] <= row.minutes_to_kickoff <= window[2], df_long)

    # 2. Group by match and selection to estimate the price
    summary = combine(groupby(df_window, [:match_id, :market_name, :market_line, :selection])) do sdf
        if nrow(sdf) < min_ticks
            return (odds = (missing),)
        end
        
        # Dispatch to our abstract estimator
        price = estimate_price(estimator, sdf.traded_price, sdf.minutes_to_kickoff)
        return (odds = price,)
    end

    # 3. Efficiency Filter: Cross-selection correlation check
    # We check if the sum of implied probabilities (1/price) for a market is sane.
    valid_matches = combine(groupby(summary, [:match_id, :market_name, :market_line])) do sdf
        # Calculate overround (sum of 1/odds)
        overround = sum(1.0 ./ skipmissing(sdf.odds))
        is_sane = overround_limits[1] <= overround <= overround_limits[2]
        return (is_sane = is_sane, overround = overround)
    end

    # Join back and filter out the "insane" markets
    final_df = leftjoin(summary, valid_matches, on = [:match_id, :market_name, :market_line])
    return filter(row -> row.is_sane, final_df)
end



closing_odds = summarize_odds(long_df, MedianEstimator(), window=(-10.0, 0.0))
closing_odds1 = summarize_odds(long_df, TWAEstimator(), window=(-10.0,0.0))



# compare the odds 
using DataFrames, Statistics

function compare_market_closing(bf_odds::DataFrame, b365_odds::DataFrame; odds_col::Symbol= :odds_close)
    # 1. Join the datasets
    # We rename to keep track of the source
    comp_df = innerjoin(
        bf_odds[:, [:match_id, :market_name, :market_line, :selection, :odds, :overround]], 
        b365_odds[:, [:match_id, :market_name, :market_line, :selection, odds_col, :overround_close]],
        on = [:match_id, :market_name, :market_line, :selection],
        makeunique = true
    )

    # Rename for clarity
    rename!(comp_df, :odds => :odds_bf, odds_col => :odds_b365)

    # 2. Calculate Comparison Metrics
    transform!(comp_df, 
        [:odds_bf, :odds_b365] => ((bf, b3) -> bf ./ b3) => :odds_ratio,
        [:overround, :overround_close] => ((ov_bf, ov_b3) -> (ov_b3 .- 1.0) .- (ov_bf .- 1.0)) => :vig_saving,
        [:odds_bf, :odds_b365] => ((bf, b3) -> (1.0 ./ b3) .- (1.0 ./ bf)) => :prob_delta
    )

    return comp_df
end

# Generate the merged comparison dataframe
comparison_df = compare_market_closing(closing_odds, ds.odds)

# 3. Create the Statistical Summary grouped by Selection
summary_stats = combine(groupby(comparison_df, [:market_name, :selection])) do sdf
    (
        count = nrow(sdf),
        mean_odds_ratio = mean(sdf.odds_ratio),
        median_odds_ratio = median(sdf.odds_ratio),
        std_odds_ratio = std(sdf.odds_ratio),
        max_odds_ratio = maximum(sdf.odds_ratio), # Changed to maximum
        min_odds_ratio = minimum(sdf.odds_ratio), # Changed to minimum
        avg_vig_saving = mean(sdf.vig_saving),
        avg_prob_delta = mean(sdf.prob_delta)
    )
end

# Sort by count so you see the most liquid markets first
sort!(summary_stats, :count, rev=true)


describe(comparison_df.odds_ratio)




#### ----

using Statistics

# 1. Define the Efficiency Helper Function
function market_efficiency_index(prices, mins, closing_price)
    length(prices) < 2 && return 0.0
    
    # Explicitly use Base.diff!
    durations = abs.(Base.diff(mins))
    push!(durations, 1.0) # padding for last tick
    
    total_deviation = sum(abs.(prices .- closing_price) .* durations)
    return total_deviation / sum(durations)
end

# 2. Compute the Metrics per Match & Selection
market_profiles = combine(groupby(long_df, [:match_id, :market_name, :market_line, :selection])) do sdf
    # Ensure chronological order
    sdf_sorted = sort(sdf, :minutes_to_kickoff)
    prices = sdf_sorted.traded_price
    mins = sdf_sorted.minutes_to_kickoff
    
    n_ticks = nrow(sdf_sorted)
    
    # Handle low-liquidity markets gracefully
    if n_ticks < 2
        return (
            close_price = n_ticks == 1 ? prices[1] : missing,
            volatility = NaN, 
            velocity = 0.0, 
            efficiency_score = 0.0, 
            max_overshoot = 0.0, 
            max_drop = 0.0, 
            tick_count = n_ticks
        )
    end

    # The Final Price of the series
    closing_price = prices[end]

    # 1. Volatility (Noise) - Note the Base.diff
    log_returns = Base.diff(log.(prices))
    vol = std(log_returns)

    # 2. Velocity (Overall price drift per minute)
    duration = mins[end] - mins[1]
    vel = duration > 0 ? (closing_price - prices[1]) / duration : 0.0

    # 3. Efficiency (Time spent away from closing price)
    eff_score = market_efficiency_index(prices, mins, closing_price)

    # 4. Drawdown / Overshoot (Resistance)
    max_overshoot = maximum(prices) - closing_price
    max_drop = minimum(prices) - closing_price

    return (
        close_price = closing_price,
        volatility = vol,
        velocity = vel,
        efficiency_score = eff_score,
        max_overshoot = max_overshoot,
        max_drop = max_drop,
        tick_count = n_ticks
    )
end



# 3. Summarize the metrics across the whole league/dataset
market_macro_summary = combine(groupby(filter(row -> row.tick_count > 5, market_profiles), [:market_name, :selection])) do sdf
    (
        matches_analyzed = nrow(sdf),
        avg_volatility = mean(filter(!isnan, sdf.volatility)),
        avg_efficiency = mean(sdf.efficiency_score),
        avg_velocity = mean(abs.(sdf.velocity)), # Absolute to see magnitude of moves
        avg_overshoot = mean(sdf.max_overshoot)
    )
end

# Sort by volatility to see the noisiest markets
sort!(market_macro_summary, :avg_volatility, rev=true)



# ----
#
function finalize_betfair_market(df_long::DataFrame; 
    open_window=(-1440.0, -1380.0), # 24h to 23h before KO
    close_window=(-20.0, 0.0),      # 20 mins to KO
    estimator=MedianEstimator()
)
    # 1. Generate Open and Close summaries
    df_open = summarize_odds(df_long, estimator, window=open_window)
    rename!(df_open, :odds => :odds_open, :overround => :overround_open)
    
    df_close = summarize_odds(df_long, estimator, window=close_window)
    rename!(df_close, :odds => :odds_close, :overround => :overround_close)

    # 2. Join them (Inner join ensures we only keep selections that existed at both times)
    # Or Outer join if you want to keep lines that only appeared late
    final_df = innerjoin(
        df_open[:, [:match_id, :market_name, :market_line, :selection, :odds_open, :overround_open]],
        df_close[:, [:match_id, :market_name, :market_line, :selection, :odds_close, :overround_close]],
        on = [:match_id, :market_name, :market_line, :selection]
    )

    # 3. Calculate implied probabilities
    final_df.prob_implied_open  = 1.0 ./ final_df.odds_open
    final_df.prob_implied_close = 1.0 ./ final_df.odds_close

    # 4. Calculate Fair Probabilities (removing the vig)
    # Logic: fair_prob = implied_prob / overround
    final_df.prob_fair_open  = final_df.prob_implied_open ./ final_df.overround_open
    final_df.prob_fair_close = final_df.prob_implied_close ./ final_df.overround_close

    # 5. Calculate Fair Odds
    final_df.fair_odds_open  = 1.0 ./ final_df.prob_fair_open
    final_df.fair_odds_close = 1.0 ./ final_df.prob_fair_close

    # 6. Calculate Vig (Vig = Overround - 1)
    final_df.vig_open  = final_df.overround_open .- 1.0
    final_df.vig_close = final_df.overround_close .- 1.0

    return final_df
end


# Now you can easily swap estimators or windows
final_betfair_odds = finalize_betfair_market(
    long_df, 
    estimator = TWAEstimator(), 
    open_window = (-15.0, 0.0),
    close_window = (-30.0, 0.0)
)


ds1 = Data.DataStore(
        ds.segment,
        ds.matches,
        ds.statistics,
         final_betfair_odds,
         ds.lineups,
         ds.incidents
)


final_betfair_odds1 = finalize_betfair_market(
    long_df, 
    estimator = TWAEstimator(), 
    open_window = (-15.0, 0.0),
    close_window = (-100.0, -10.0)
)


ds2 = Data.DataStore(
        ds.segment,
        ds.matches,
        ds.statistics,
         final_betfair_odds1,
         ds.lineups,
         ds.incidents
)




first(ds1.odds, 10)
first(ds.odds, 10)

using DataFrames

"""
Evaluates a betting selection against the actual match result.
Returns true if the bet won, false if it lost, and missing if no result exists.
"""
function grade_selection(
    market_name::String, 
    market_line::Float64, 
    selection::Symbol, 
    home_score::Union{Missing, Integer}, # Changed to Integer
    away_score::Union{Missing, Integer}  # Changed to Integer
)
    # If the match hasn't happened yet or scores are missing, return missing
    if ismissing(home_score) || ismissing(away_score)
        return missing
    end

    total_goals = home_score + away_score
    sel_str = string(selection)

    # 1X2 Market
    if market_name == "1X2"
        if selection == :home
            return home_score > away_score
        elseif selection == :draw
            return home_score == away_score
        elseif selection == :away
            return home_score < away_score
        end

    # Over/Under Markets
    elseif market_name == "OverUnder"
        if startswith(sel_str, "over_")
            return total_goals > market_line
        elseif startswith(sel_str, "under_")
            return total_goals < market_line
        end

    # BTTS Market
    elseif market_name == "BTTS"
        if selection == :btts_yes
            return home_score > 0 && away_score > 0
        elseif selection == :btts_no
            return home_score == 0 || away_score == 0
        end
        
    # Correct Score Markets
    elseif market_name == "CorrectScore"
        if occursin(r"^cs_\d\d$", sel_str)
            h = parse(Int, sel_str[4])
            a = parse(Int, sel_str[5])
            return home_score == h && away_score == a
        elseif sel_str == "cs_any_other_home"
            return home_score > away_score && (home_score >= 4 || away_score >= 4)
        elseif sel_str == "cs_any_other_away"
            return home_score < away_score && (home_score >= 4 || away_score >= 4)
        elseif sel_str == "cs_any_other_draw"
            return home_score == away_score && home_score >= 4
        end
    end

    return missing
end

"""
Merges match results into the odds dataframe and calculates the `is_winner` flag.
"""
function append_match_results(odds_df::DataFrame, matches_df::DataFrame)
    # Join just the columns we need from ds.matches
    target_cols = [:match_id, :home_score, :away_score, :match_date]
    merged_df = leftjoin(odds_df, matches_df[:, target_cols], on = :match_id)

    # Apply the grading function row by row using broadcasting
    merged_df.is_winner = grade_selection.(
        merged_df.market_name, 
        merged_df.market_line, 
        merged_df.selection, 
        merged_df.home_score, 
        merged_df.away_score
    )

    # Rename match_date to date to align with ds.odds
    rename!(merged_df, :match_date => :date)

    # Drop the temporary score columns to keep the schema clean
    select!(merged_df, Not([:home_score, :away_score]))

    return merged_df
end


# 1. Generate the initial Betfair odds
raw_final_odds = finalize_betfair_market(
    long_df, 
    estimator = TWAEstimator(), 
    open_window = (-15.0, 0.0),
    close_window = (-30.0, 0.0)
)

# 2. Append the match results and calculate is_winner
final_betfair_odds = append_match_results(raw_final_odds, ds.matches)

# 3. Build the new DataStore
ds1 = Data.DataStore(
    ds.segment,
    ds.matches,
    ds.statistics,
    final_betfair_odds, # Now contains :is_winner and :date!
    ds.lineups,
    ds.incidents
)



raw_final_odds1 = finalize_betfair_market(
    long_df, 
    estimator = TWAEstimator(), 
    open_window = (-15.0, 0.0),
    close_window = (-100.0, -15.0)
)

# 2. Append the match results and calculate is_winner
final_betfair_odds1 = append_match_results(raw_final_odds1, ds.matches)

# 3. Build the new DataStore
ds2 = Data.DataStore(
    ds.segment,
    ds.matches,
    ds.statistics,
    final_betfair_odds1, # Now contains :is_winner and :date!
    ds.lineups,
    ds.incidents
)




markets_config= Data.MarketConfig( reduce(vcat, ( [Data.Market1X2(), Data.MarketBTTS()], [Data.MarketOverUnder( (i +0.5) ) for i in 0:4 ] ) ))

ledger = BayesianFootball.BackTesting.run_backtest(
    ds2, 
    loaded_results, 
  [BayesianFootball.Signals.BayesianKelly(), BayesianFootball.Signals.BayesianKelly(min_edge=0.02)]; 
    market_config = markets_config
)
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end


