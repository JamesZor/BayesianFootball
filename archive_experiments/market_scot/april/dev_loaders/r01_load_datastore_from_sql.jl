#************************************************** 
# ---- Runner -----
#************************************************** 


# exp/market_scot/april/dev_loaders/r01_load_datastore_from_sql.jl
# ----
# need to run  
#  exp/market_scot/april/dev_loaders/l01_load_datastore_from_sql.jl
include("./l01_load_datastore_from_sql.jl")
# ---
config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")


# Dev test 1 - sql function call 
conn = connect_to_db(config.url)

sql_matches = fetch_matches(conn)
sql_incidents = fetch_incidents(conn)
sql_odds = fetch_odds(conn)

data_store = load_datastore(config)

