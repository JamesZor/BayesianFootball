# dev_repl/data_module_sql_update/l01_get_datastore_from_sql.jl 


#=
--- Loader ----
____________________
=#



using Revise
using LibPQ
using DataFrames
using Dates
using BayesianFootball



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



# Task 3:
#   Get DataStore:
#   Sub-Tacks:
#     fetch:
#    1. Matches.
#    2. Statistics.
#    3. Odds. - Note we need to check winning - post processing ? 
#    4. LineUps
#    5. Incidents.


# 3.1 fetch - matches 
"""
Fetch matches for a specific tournament segment.
"""
function fetch_matches(conn::LibPQ.Connection, t_ids::Vector{Int})::DataFrame
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



function fetch_matches(conn::LibPQ.Connection, segment::DataTournemantSegment)::DataFrame
  return fetch_matches(conn, tournament_ids(segment))
end

# ---  structs 
abstract type FootballDataType end

struct MatchesData    <: FootballDataType end
struct StatisticsData <: FootballDataType end
struct OddsData       <: FootballDataType end
struct LineUpsData    <: FootballDataType end
struct IncidentsData  <: FootballDataType end

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
Maps raw API market names to internal market groups.
"""
function map_market_group(market_name::AbstractString)::String
    if market_name in ("Full time", "1st half", "2nd half")
        return "1X2"
    elseif market_name == "Both teams to score"
        return "BTTS"
    elseif market_name == "Double chance"
        return "Double chance"
    else
        return "Other" # Fallback for unmapped markets
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
    df.market_group = map_market_group.(df.market_name)
    
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

