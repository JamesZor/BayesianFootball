using Revise
using DataFrames

# Include the dev methods we just created

# include("db_methods.jl")
using .DBMethods

# Define the connection using the Tailscale IP
const DB_URL = "postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db"

println("Establishing connection to the database...")
conn = connect_to_db(DB_URL)

try
    println("Fetching matches...")
    df_matches = fetch_matches_v2(conn)
    
    println("Fetching odds...")
    df_odds = fetch_odds(conn)
    
    # println("Fetching incidents...")
    # df_incidents = fetch_incidents(conn)
    
    println("\n--- Data Pipeline Success ---")
    println("Matches loaded:   ", nrow(df_matches))
    println("Odds loaded:      ", nrow(df_odds))
    # println("Incidents loaded: ", nrow(df_incidents))
    
    println("\n--- Matches Preview ---")
    println(first(df_matches, 3))
    
    
    println("\n--- Odds Preview ---")
    println(first(df_odds, 3))
    
catch e
    println("An error occurred while fetching data:")
    showerror(stdout, e)
finally
    # Always ensure the connection is closed when the script finishes or errors out
    close(conn)
    println("\nDatabase connection closed.")
end


# stage two 

using BayesianFootball
using DBInterface

BayesianFootball.data.add_inital_odds_from_fractions!(df_odds)

config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")


data_store = load_datastore(config)


data_store.matches



transform!(data_store.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

ref_ds = Data.load_extra_ds()




market_data = Data.prepare_market_data(data_store)
