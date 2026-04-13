# dev_repl/data_module_sql_update/r01_get_datastore_from_sql.jl 


#=
--- Runner ----
____________________
=#

include("./l01_get_datastore_from_sql.jl")

# Task 1:
#   Create the connection 

db_config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")

db_conn = connect_to_db(db_config)



# Task 2 
sl = ScottishLower()
tournament_ids(sl)
il = Ireland()
tournament_ids(il)


# Task 3 

# task 3.1 - fetch - matches 

df_matches = fetch_matches(
                  db_conn,
                  ScottishLower())

df_matches = fetch_data(db_conn, Ireland(), MatchesData())
df_inc = fetch_data(db_conn, ScottishLower(), IncidentsData())

df_stats = fetch_data(db_conn, SouthKorea(), StatisticsData())



df_odds = fetch_data(db_conn, SouthKorea(), OddsData())

