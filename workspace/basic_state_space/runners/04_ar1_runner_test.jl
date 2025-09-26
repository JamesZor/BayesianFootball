using BayesianFootball
using DataFrames
using Dates

const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"

data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)



sort(unique( data_store.matches.season))
sort(filter(row -> row.season=="25/26" && row.tournament_id==54, data_store.matches), :match_date, rev=true)





# --- Add the global round column right after loading ---
add_global_round_column!(data_store.matches)

names_test = [
 "tournament_id"
 "season"
 "tournament_slug"
 "home_team"
 "away_team"
 "global_round"
 "match_date"
 "round"
 "home_score"
 "away_score"
 "home_score_ht"
 "away_score_ht"
]

sort(filter(row -> row.season=="25/26", data_store.matches), :match_date, rev=false)[10:50, names_test]
