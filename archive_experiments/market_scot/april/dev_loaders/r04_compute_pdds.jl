#************************************************** 
# ---- Runner -----
#************************************************** 


# ----
# need to run  
#  But no load this files 
include("./l01_load_datastore_from_sql.jl")
include("./l02_load_experiment_data.jl")
include("./l03_load_today_matches_from_sql.jl")
# ####################

include("./l04_compute_pdds.jl")
# ---

# Run past runners as we need this for r04

# from r01
config = DBConfig("postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db")
conn = connect_to_db(config.url)
data_store = load_datastore(config)

# from r02
exp_m1 = load_experiment_data_from_disk()

# from r03
todays_matches = fetch_todays_matches(conn)


# dev test area
todays_ppds = compute_todays_matches_pdds(data_store,  exp_m1, todays_matches)



