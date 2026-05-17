#************************************************** 
# ---- Runner -----
#************************************************** 


# ----
# need to run  
include("./l03_load_today_matches_from_sql.jl")
# ---
config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")


# Dev test 1 - sql function call 
conn = connect_to_db(config.url)

todays_matches = fetch_todays_matches(conn)


