
using LibPQ
# using DBInterface
using DataFrames
using Dates


"""
Establish a connection to the PostgreSQL database.
"""
function connect_to_db(url::String)
    return LibPQ.Connection(url)
end

"""
Fetch matches and format them to closely match your existing DataFrame structure.
"""
function fetch_matches(conn::LibPQ.Connection)
    query = """
        SELECT 
            m.tournament_id,
            m.season_id,
            s.year AS season, 
            m.match_id,
            m.home_team,
            m.away_team,
            m.home_score,
            m.away_score,
            m.home_score_ht,
            m.away_score_ht,
            m.winner_code,
            m.start_timestamp AS match_date,
            m.round,
            
            -- Extracting boolean flags directly from the JSONB raw_data
            (m.raw_data ->> 'hasXg')::boolean AS has_xg,
            (m.raw_data ->> 'hasEventPlayerStatistics')::boolean AS has_stats
            
        FROM matches m
        JOIN seasons s ON m.season_id = s.season_id
        WHERE m.status_type = 'finished'
    """
    
    df = DataFrame(LibPQ.execute(conn, query))
    
    # Cast timestamp to Date to match your CSV output
    if nrow(df) > 0 && !(typeof(df.match_date[1]) <: Date)
        df.match_date = Date.(df.match_date) 
    end
    
    return df
end

function fetch_matches_v2(conn)
    query = """
        SELECT 
            m.tournament_id,
            m.season_id,
            s.year AS season, 
            m.match_id,
            
            -- Extract tournament slug from JSON just like Python did
            m.raw_data -> 'tournament' ->> 'slug' AS tournament_slug,
            
            m.home_team,
            m.away_team,
            m.home_score,
            m.away_score,
            m.home_score_ht,
            m.away_score_ht,
            m.winner_code,
            m.start_timestamp,  -- We pull the full timestamp to extract hour/month below
            m.round,
            (m.raw_data ->> 'hasXg')::boolean AS has_xg,
            (m.raw_data ->> 'hasEventPlayerStatistics')::boolean AS has_stats
        FROM matches m
        JOIN seasons s ON m.season_id = s.season_id
        WHERE m.status_type = 'finished'
    """
    
    df = DataFrame(LibPQ.execute(conn, query))
    
    if nrow(df) > 0
        # 1. Replicate Python's dt.hour and dt.month
        df.match_hour = hour.(df.start_timestamp)
        df.match_month = month.(df.start_timestamp)
        
        # 2. Replicate Python's dt.weekday() (Monday=0, Sunday=6)
        # Julia's dayofweek is Monday=1, Sunday=7, so we just subtract 1
        df.match_dayofweek = dayofweek.(df.start_timestamp) .- 1
        
        # 3. Create the final match_date column (pure Date, no time)
        df.match_date = Date.(df.start_timestamp)
        
        # 4. Drop the raw start_timestamp so it matches your old dataframe exactly
        select!(df, Not(:start_timestamp))
    end
    
    return df
end

"""
Fetch incidents, flattening the JSON payload into columns exactly like the Python script did.
"""
function fetch_incidents(conn::LibPQ.Connection)
    query = """
        SELECT 
            match_id,
            tournament_id,
            season_id,
            incident_type,
            time,
            is_home,
            added_time,
            
            -- Flattened JSONB fields from the `data` column
            data -> 'player' ->> 'name' AS player_name,
            data -> 'playerIn' ->> 'name' AS player_in_name,
            data -> 'playerOut' ->> 'name' AS player_out_name,
            data -> 'assist1' ->> 'name' AS assist1_name,
            data -> 'assist2' ->> 'name' AS assist2_name,
            data ->> 'incidentClass' AS incident_class,
            data ->> 'reason' AS reason,
            
            -- Casting strings to booleans safely
            (data ->> 'injury')::boolean AS is_injury,
            (data ->> 'rescinded')::boolean AS rescinded,
            
            -- Period specific
            data ->> 'text' AS period_text,
            (data ->> 'timeSeconds')::numeric AS time_seconds
            
        FROM match_incidents
    """
    return DataFrame(LibPQ.execute(conn, query))
end

"""
Fetch odds and concatenate the fractions so they match your old CSV logic.
"""
function fetch_odds(conn::LibPQ.Connection)
    query = """
        SELECT 
            match_id,
            market_id,
            market_name,
            choice_group,
            choice_name,
            winning,
            
            -- Recreating your old CSV fractional formatting (e.g., "5/2")
            CONCAT(initial_fraction_num, '/', initial_fraction_den) AS initial_fractional_value,
            CONCAT(fraction_num, '/', fraction_den) AS final_fractional_value
            
        FROM match_odds
    """
    return DataFrame(LibPQ.execute(conn, query))
end


# stage two
#
# ---- struct 

using BayesianFootball

struct DBConfig
    url::String
end


function load_datastore(config::DBConfig)::BayesianFootball.Data.DataStore 
    conn = LibPQ.Connection(config.url)
    try
        matches = fetch_matches_v2(conn)
        odds = fetch_odds(conn)
        incidents =  DataFrame()
        return BayesianFootball.Data.DataStore(matches, odds, incidents)
    finally
        close(conn)
    end
end
