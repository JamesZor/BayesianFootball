# note this is a test file - further dev is need 

using LibPQ
using DataFrames
using Dates



# Task 1:
#   Create the connection 
struct DBConfig 
  url::String
end 


"""
Establish a connection to the PostgreSQL database.
"""
function connect_to_db(db_config::DBConfig)::LibPQ.Connection
    return LibPQ.Connection(db_config.url)
end


# Task 2:
#   Enum DataSegments
#   The Singleton Type Approach (The "Julia" Way for Dispatch)
abstract type DataTournemantSegment end

struct ScottishLower <: DataTournemantSegment end 
struct Ireland <: DataTournemantSegment end 
struct SouthKorea <: DataTournemantSegment end 

# 3. Map the types to their IDs
tournament_ids(::ScottishLower) = [56, 57]
tournament_ids(::Ireland)       = [79]
tournament_ids(::SouthKorea)    = [3284, 6230]


# ---  structs 
abstract type FootballDataType end

struct MatchesData    <: FootballDataType end # done
struct StatisticsData <: FootballDataType end # done 
struct OddsData       <: FootballDataType end # done 
struct LineUpsData    <: FootballDataType end
struct IncidentsData  <: FootballDataType end # done

# ---------------------------------------------------------
# 2. THE ONE WRAPPER TO RULE THEM ALL
# ---------------------------------------------------------
"""
Convenience wrapper: Extracts tournament IDs from a Segment and delegates to the Workhorse.
"""
function fetch_data(conn::LibPQ.Connection, segment::DataTournemantSegment, data_type::FootballDataType)::DataFrame
  return fetch_data(conn, tournament_ids(segment), data_type)
end

# ---------------------------------------------------------
# 3. The Workhorse Methods (Where the SQL lives)
# ---------------------------------------------------------
function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::MatchesData)::DataFrame
    query = """
        SELECT 
            m.tournament_id,
            m.season_id,
            s.year AS season, 
            m.match_id,
            m.raw_data -> 'tournament' ->> 'slug' AS tournament_slug,
            m.home_team,
            m.away_team,
            m.home_score,
            m.away_score,
            m.home_score_ht,
            m.away_score_ht,
            m.winner_code,
            m.start_timestamp, 
            m.round,
            (m.raw_data ->> 'hasXg')::boolean AS has_xg,
            (m.raw_data ->> 'hasEventPlayerStatistics')::boolean AS has_stats
        FROM matches m
        JOIN seasons s ON m.season_id = s.season_id
        WHERE m.status_type = 'finished'
        AND m.tournament_id = ANY(\$1) 
    """
    
    # 3. Pass the t_ids vector as a parameter wrapped in an array [t_ids]
    result = LibPQ.execute(conn, query, [t_ids])
    df = DataFrame(result)
    
    if nrow(df) > 0
        df.match_hour = hour.(df.start_timestamp)
        df.match_month = month.(df.start_timestamp)
        df.match_dayofweek = dayofweek.(df.start_timestamp) .- 1
        df.match_date = Date.(df.start_timestamp)
        select!(df, Not(:start_timestamp))
    end
    
    return df
end

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::IncidentsData)::DataFrame
    # Notice we alias match_incidents as 'i' and matches as 'm'
    # We join them on match_id, which allows us to filter on m.tournament_id
    query = """
        SELECT 
            i.id,
            i.match_id,
            i.incident_type,
            i.time,
            i.is_home,
            i.added_time,

            -- Flattened JSONB fields from the `data` column
            i.data -> 'player' ->> 'slug' AS player_name,
            i.data -> 'playerIn' ->> 'slug' AS player_in_name,
            i.data -> 'playerOut' ->> 'slug' AS player_out_name,
            i.data -> 'assist1' ->> 'slug' AS assist1_name,
            i.data -> 'assist2' ->> 'name' AS assist2_name,
            i.data ->> 'incidentClass' AS incident_class,
            i.data ->> 'reason' AS reason,
            
            -- Casting strings to booleans safely
            (i.data ->> 'injury')::boolean AS is_injury,
            (i.data ->> 'rescinded')::boolean AS rescinded,
            
            -- Period specific
            i.data ->> 'text' AS period_text,
            (i.data ->> 'timeSeconds')::numeric AS time_seconds
            
        FROM match_incidents i
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
    """
    
    # Execute with the parameterized t_ids
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::StatisticsData)::DataFrame
      query = """
      SELECT DISTINCT 
          m.match_id,
          m.tournament_id,
          m.season_id,
          s.period,
          s.stat_key,
          s.home_value,
          s.away_value
      FROM match_statistics s
      JOIN matches m ON s.match_id = m.match_id
      WHERE m.tournament_id = ANY(\$1)
      """

      long_df = DataFrame(LibPQ.execute(conn, query, [t_ids]))

      # Early exit if empty
      if nrow(long_df) == 0
          return long_df
      end

      # 1. Unstack the HOME values
      home_wide = unstack(
          long_df,
          [:match_id, :tournament_id, :season_id, :period], # Grouping columns
          :stat_key,                                        # New headers
          :home_value,                                      # The single value column
        renamecols = x -> "$(x)_home"                 # Prefix with home_value_
      )

      # 2. Unstack the AWAY values
      away_wide = unstack(
          long_df,
          [:match_id, :tournament_id, :season_id, :period], # Grouping columns
          :stat_key,                                        # New headers
          :away_value,                                      # The single value column
        renamecols = x -> "$(x)_away"                 # Prefix with away_value_
      )

      # 3. Join them back together into one massive wide table
      wide_df = innerjoin(
          home_wide, 
          away_wide, 
          on = [:match_id, :tournament_id, :season_id, :period]
      )
    return wide_df
end


"""
Maps raw API market ids to internal market groups.
"""
function map_odds_market_id_to_market_group(market_id::Integer)::String 
  if market_id in [1,3] return  "1X2"
    elseif market_id == 2 return "Double chance" 
    elseif market_id == 4 return "Draw no bet" 
    elseif market_id == 5 return "Both teams to score" 
    elseif market_id == 6 return "First team to score" 
    elseif market_id == 9 return "Match goals" 
    elseif market_id == 17 return "Asian handicap" 
    elseif market_id == 21 return "Corners 2-way" 
    else  return "Other" # fallback 
  end 
end 

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::OddsData)::DataFrame
    # 1. Fetch the raw split data, joining with matches to filter by tournament
    query = """
        SELECT 
            m.tournament_id,
            m.season_id,
            o.match_id,
            o.market_id,
            o.market_name,
            o.choice_name,
            o.choice_group,
            o.initial_fraction_num,
            o.initial_fraction_den,
            o.fraction_num,
            o.fraction_den,
            o.winning
        FROM match_odds o
        JOIN matches m ON o.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
    """
    
    df = DataFrame(LibPQ.execute(conn, query, [t_ids]))
    
    if nrow(df) == 0
        return df
    end
    
    # 2. Map the market_group dynamically
    df.market_group = map_odds_market_id_to_market_group.(df.market_id)
    
    # 3. Reconstruct the Fractional Strings (e.g., "19/10") safely handling SQL nulls/missing
    df.initial_fractional_value = [
        ismissing(n) ? missing : "$n/$d" 
        for (n, d) in zip(df.initial_fraction_num, df.initial_fraction_den)
    ]
    
    df.final_fractional_value = [
        ismissing(n) ? missing : "$n/$d" 
        for (n, d) in zip(df.fraction_num, df.fraction_den)
    ]
    
    # 4. Calculate Decimal Odds: (Numerator / Denominator) + 1
    df.decimal_odds = [
        ismissing(n) ? missing : round((n / d) + 1.0, digits=2) 
        for (n, d) in zip(df.fraction_num, df.fraction_den)
    ]
    
    # 5. Drop the raw numerator/denominator columns to keep the DataFrame clean
    select!(df, Not([
        :initial_fraction_num, :initial_fraction_den, 
        :fraction_num, :fraction_den
    ]))
    
    # Reorder columns to exactly match your old CSV structure (Optional, but nice for visual parity)
    select!(df, :tournament_id, :season_id, :match_id, :market_id, :market_name, 
                :market_group, :choice_name, :choice_group, 
                :initial_fractional_value, :final_fractional_value, 
                :winning, :decimal_odds)
                
    return df
end

    
function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::LineUpsData)::DataFrame
    # -------------------------------------------------------------------
    # 1. Fetch the Base Player Details (Core columns)
    # -------------------------------------------------------------------
    base_query = """
        SELECT 
            m.tournament_id,
            m.season_id,
            l.match_id,
            CASE WHEN l.is_home_team THEN 'home' ELSE 'away' END AS team_side,
            l.player_id,
            l.player_name,
            l.position,
            l.shirt_number,
            l.substitute AS is_substitute,
            l.captain AS is_captain,
            l.minutes_played,
            l.rating,
            l.goals,
            l.expected_goals,
            l.expected_assists
        FROM match_player_lineups l
        JOIN matches m ON l.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
    """
    
    base_df = DataFrame(LibPQ.execute(conn, base_query, [t_ids]))
    
    if nrow(base_df) == 0
        return base_df
    end

    # -------------------------------------------------------------------
    # 2. Fetch and Unpack the JSON Statistics (Long Format)
    # -------------------------------------------------------------------
    # jsonb_each(statistics) turns {"touches": 54, "saves": 5} into:
    # match_id | player_id | stat_key | stat_value
    # 13250768 |    910395 | touches  | 54
    # 13250768 |    910395 | saves    | 5
    
    json_query = """
        SELECT 
            l.match_id,
            l.player_id,
            stats.key AS stat_key,
            (stats.value)::text AS stat_value -- Cast to text so Julia can parse it uniformly
        FROM match_player_lineups l
        JOIN matches m ON l.match_id = m.match_id,
        jsonb_each(l.statistics) AS stats
        WHERE m.tournament_id = ANY(\$1)
        -- Exclude the nested 'ratingVersions' object to prevent pivot errors
        AND stats.key != 'ratingVersions' 
    """
    
    stats_long_df = DataFrame(LibPQ.execute(conn, json_query, [t_ids]))

    # -------------------------------------------------------------------
    # 3. Pivot and Merge
    # -------------------------------------------------------------------
    # If the JSON was completely empty for all matches, just return the base_df
    if nrow(stats_long_df) == 0
        # Optional: ensure 'assists' column exists even if empty, to match CSV
        base_df.assists .= missing 
        return base_df
    end
    
    # Parse the text values back into Float64 (since JSON numbers are mostly floats/ints)
    stats_long_df.stat_value = passmissing(parse).(Float64, stats_long_df.stat_value)

    # Pivot the long format into wide format
    stats_wide_df = unstack(
        stats_long_df,
        [:match_id, :player_id], # The identifiers
        :stat_key,               # The new column headers (touches, saves, etc.)
        :stat_value,             # The numeric values
        combine = first          # Defensive combining, just like last time
    )

    # -------------------------------------------------------------------
    # 4. Clean up naming to match your old CSV exactly
    # -------------------------------------------------------------------
    desired_renames = Dict(
        "totalPass"    => "total_passes",
        "accuratePass" => "accurate_passes",
        "goalAssist"   => "assists",
        "duelWon"      => "duels_won",
        "duelLost"     => "duels_lost",
        "aerialWon"    => "aerials_won",
        "aerialLost"   => "aerials_lost"
    )
    
    actual_columns = names(stats_wide_df)
    valid_renames = [old_name => new_name for (old_name, new_name) in desired_renames if old_name in actual_columns]
    
    if !isempty(valid_renames)
        rename!(stats_wide_df, valid_renames)
    end

    # -------------------------------------------------------------------
    # 5. Strip Redundant Overlaps & Join
    # -------------------------------------------------------------------
    # Find any columns that exist in both tables (e.g., "rating", "goals")
    # We exclude the join keys ("match_id", "player_id") so we don't drop them!
    overlapping_cols = setdiff(
        intersect(names(base_df), names(stats_wide_df)), 
        ["match_id", "player_id"]
    )
    
    # Drop the redundant columns from the JSON stats table
    if !isempty(overlapping_cols)
        select!(stats_wide_df, Not(overlapping_cols))
    end

    # Safely perform the join now that the column names are strictly unique
    final_df = leftjoin(base_df, stats_wide_df, on = [:match_id, :player_id])
    
    return final_df
end


function get_datastore(conn::LibPQ.Connection, segment::DataTournemantSegment)::DataStore
    local matches, statistics, incidents, lineup, odds

    @info "Building DataStore for $(typeof(segment))..."

    @sync begin
        # The wrapper intercepts these, grabs the IDs, and delegates!
        @async matches    = fetch_data(conn, segment, MatchesData())
        @async statistics = fetch_data(conn, segment, StatisticsData())
        @async lineup     = fetch_data(conn, segment, LineUpsData())
        @async incidents  = fetch_data(conn, segment, IncidentsData())
        @async odds      = fetch_data(conn, segment, OddsData())
    end

    return DataStore(matches,statistics, odds, lineup, incidents)
end


function load_datastore_sql(segment::DataTournemantSegment)::DataStore 
    db_config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
    db_conn = connect_to_db(db_config)
    data_store = get_datastore(db_conn, segment)
    return data_store
end 


