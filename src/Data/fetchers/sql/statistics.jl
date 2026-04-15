# src/data/fetchers/sql/statistics.jl

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::StatisticsData)
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
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

function process_data(df::DataFrame, ::StatisticsData)
    # 1. Unstack the HOME values
    home_wide = unstack(
        df,
        [:match_id, :tournament_id, :season_id, :period], 
        :stat_key,                 
        :home_value,                                      
        renamecols = x -> "$(x)_home"           
    )

    # 2. Unstack the AWAY values
    away_wide = unstack(
        df,
        [:match_id, :tournament_id, :season_id, :period], 
        :stat_key,                              
        :away_value,                                      
        renamecols = x -> "$(x)_away"                 
    )

    # 3. Join them back together
    return innerjoin(
        home_wide, 
        away_wide, 
        on = [:match_id, :tournament_id, :season_id, :period]
    )
end

function validate_data(df::DataFrame, ::StatisticsData)
    if !("match_id" in names(df)) || !("period" in names(df))
        @error "StatisticsData QA Failed: Missing base identifiers."
        return false
    end
    return true
end
