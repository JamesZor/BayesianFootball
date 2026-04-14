# src/data/fetchers/sql/incidents.jl

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::IncidentsData)
    query = """
        SELECT 
            i.id, i.match_id, i.incident_type, i.time, i.is_home, i.added_time,
            i.data -> 'player' ->> 'slug' AS player_name,
            i.data -> 'playerIn' ->> 'slug' AS player_in_name,
            i.data -> 'playerOut' ->> 'slug' AS player_out_name,
            i.data -> 'assist1' ->> 'slug' AS assist1_name,
            i.data -> 'assist2' ->> 'name' AS assist2_name,
            i.data ->> 'incidentClass' AS incident_class,
            i.data ->> 'reason' AS reason,
            (i.data ->> 'injury')::boolean AS is_injury,
            (i.data ->> 'rescinded')::boolean AS rescinded,
            i.data ->> 'text' AS period_text,
            (i.data ->> 'timeSeconds')::numeric AS time_seconds
        FROM match_incidents i
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
    """
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

# process_data falls back to the default interface (returns raw df)

function validate_data(df::DataFrame, ::IncidentsData)
    if !("incident_type" in names(df))
        @error "IncidentsData QA Failed: Missing incident_type."
        return false
    end
    return true
end
