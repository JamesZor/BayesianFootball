# src/data/fetchers/sql/matches.jl

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::MatchesData)
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
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

function process_data(df::DataFrame, ::MatchesData)
    # Extract date parts from the timestamp
    df.match_hour = hour.(df.start_timestamp)
    df.match_month = month.(df.start_timestamp)
    df.match_dayofweek = dayofweek.(df.start_timestamp) .- 1
    df.match_date = Date.(df.start_timestamp)

    add_match_week_column(df.matches),

    transform!(df, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
    
    # Drop the raw timestamp as we have the parsed components
    select!(df, Not(:start_timestamp))
    return df
end

function validate_data(df::DataFrame, ::MatchesData)
    required_cols = ["match_id", "home_team", "away_team", "match_date"]
    missing_cols = setdiff(required_cols, names(df))
    
    if !isempty(missing_cols)
        @error "MatchesData QA Failed: Missing critical columns: $missing_cols"
        return false
    end
    return true
end
