# src/data/fetchers/datastore.jl

"""
    connect_to_db(db_config::DBConfig)::LibPQ.Connection
Establish a connection to the PostgreSQL database.
"""
function connect_to_db(db_config::DBConfig)::LibPQ.Connection
    return LibPQ.Connection(db_config.url)
end

"""
    get_datastore(conn, segment) -> DataStore
Executes all data pipelines concurrently and aggregates them into the DataStore struct.
"""
function get_datastore(conn::LibPQ.Connection, segment::DataTournemantSegment; custom_config = Markets.DEFAULT_MARKET_CONFIG)::DataStore
    local matches, statistics, incidents, lineups, odds

    @info "Building DataStore for $(typeof(segment))..."

    # Concurrently fire the load_data pipeline for all 5 domains
    @sync begin
        @async matches    = load_data(conn, segment, MatchesData())
        @async statistics = load_data(conn, segment, StatisticsData())
        @async lineups    = load_data(conn, segment, LineUpsData())
        @async incidents  = load_data(conn, segment, IncidentsData())
        @async odds       = load_data(conn, segment, OddsData(); config = custom_config)
    end

    return DataStore(segment, matches, statistics, odds, lineups, incidents)
end


"""
    load_datastore_sql(segment) -> DataStore
Main entry point for external use. Manages the DB connection lifecycle and returns the data.
"""
function load_datastore_sql(segment::DataTournemantSegment)::DataStore 
    # In a production environment, this URL should be loaded from an ENV variable.
    db_config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
    
    db_conn = connect_to_db(db_config)
    
    try
        data_store = get_datastore(db_conn, segment)
        return data_store
    finally
        # Always close the connection, even if an error occurs during fetching
        close(db_conn) 
    end
end
