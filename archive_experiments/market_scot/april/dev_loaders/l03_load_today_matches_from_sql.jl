# exp/market_scot/april/dev_loaders/l03_load_today_matches_from_sql.jl


using Revise
using LibPQ
using DataFrames
using Dates
using BayesianFootball
# using DBInterface


# ---------------------------------
# Taken from 
# exp/market_scot/april/dev_loaders/l01_load_datastore_from_sql.jl

struct DBConfig
    url::String
end

"""
Establish a connection to the PostgreSQL database.
"""
function connect_to_db(url::String)
    return LibPQ.Connection(url)
end


# -------------------------------

function fetch_todays_matches(conn::LibPQ.Connection)
    query = """

SELECT 
    match_id,
    home_team,
    away_team,
    round,
    tournament_id,
    season_id
FROM 
    events
WHERE 
    status_type = 'notstarted'
    AND start_timestamp >= EXTRACT(EPOCH FROM CURRENT_DATE)
    AND start_timestamp < EXTRACT(EPOCH FROM CURRENT_DATE + INTERVAL '1 day');
    """
    df = DataFrame(LibPQ.execute(conn, query))

    df.match_week .= [ 999 for _ in nrow(df)]
    df.match_date .= [ today() for _ in nrow(df)]

    return df


end


