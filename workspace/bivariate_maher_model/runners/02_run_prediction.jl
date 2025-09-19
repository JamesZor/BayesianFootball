using BayesianFootball
using DataFrames
using Dates
using JLD2

# Include our experimental modules
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/prediction.jl")
using .BivariateMaher
using .BivariatePrediction

# --- 1. Setup and Load Model ---
const EXPERIMENT_NAME = "bivariate_maher_test"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"

file_path = "/home/james/bet_project/models_julia/experiments/bivariate_maher_test/bivariate_maher_verification_20250919-175239" 
println("Loading model from: $file_path")
loaded_model = load_model(file_path)

# --- 2. Load Data and Find Match ---
println("Loading data to find the match...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

team_name_home = "middlesbrough"
team_name_away = "west-bromwich-albion"

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=2,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)


# --- 3. Run Prediction ---
chains_for_prediction = loaded_model.result.chains_sequence[1]
features = BayesianFootball.create_master_features(match_to_predict, loaded_model.result.mapping)
mapping = loaded_model.result.mapping # Get the mapping object

println("\nPredicting match lines using bivariate model...")

# Call the NEW top-level function
match_predictions = predict_bivariate_match_lines(
    loaded_model.config.model_def,
    chains_for_prediction,
    features,
    mapping
)


# --- 4. Interpret and Display the Predictions ---
# You can now access the full distributions for both FT and HT markets
using StatsBase, Statistics

# Full-Time results
home_win_prob = mean(match_predictions.ft.home)
draw_prob = mean(match_predictions.ft.draw)
away_win_prob = mean(match_predictions.ft.away)
under_2_5_prob = mean(match_predictions.ft.under_25)

# Half-Time results
ht_home_win_prob = mean(match_predictions.ht.home)
ht_draw_prob = mean(match_predictions.ht.draw)

println("\n--- Predicted FT Probability (Posterior Mean) ---")
println("Home Win: ", round(home_win_prob * 100, digits=1), "%")
println("Draw:     ", round(draw_prob * 100, digits=1), "%")
println("Away Win: ", round(away_win_prob * 100, digits=1), "%")
println("Under 2.5: ", round(under_2_5_prob * 100, digits=1), "%")

println("\n--- Predicted HT Probability (Posterior Mean) ---")
println("HT Home Win: ", round(ht_home_win_prob * 100, digits=1), "%")
println("HT Draw:     ", round(ht_draw_prob * 100, digits=1), "%")


using Distributions, StatsPlots, Plots

density(1 ./ match_predictions.ft.home, label="home", title="1x2 odds")
density(1 ./ match_predictions.ft.away, label="away", title="1x2 odds")
density(1 ./ match_predictions.ft.draw, label="draw", title="1x2 odds")

density(1 ./ match_predictions.ft.under_05, label="05", title="under over")
density!(1 ./ match_predictions.ft.under_15, label="15", title="under over")
density!(1 ./ match_predictions.ft.under_25, label="25", title="under over")

mean(1 ./ match_predictions.ft.under_05 )
mean(1 ./ match_predictions.ft.under_15 )
mean(1 ./ match_predictions.ft.under_25 )
mean(1 ./ match_predictions.ft.under_35 )


mean( 1 ./ match_predictions.ft.btts)


p_cs = Dict( k => mean(v) for (k,v) in match_predictions.ft.correct_score)
sort(collect(p_cs), by = x -> x[2], rev=true)

cs = Dict( k => mean(1 ./ v) for (k,v) in match_predictions.ft.correct_score)
##### older version 

println("Found match: $(match_to_predict.home_team) vs $(match_to_predict.away_team).")

# --- 3. Run Prediction ---
chains_for_prediction = loaded_model.result.chains_sequence[1]
features = BayesianFootball.create_master_features(match_to_predict, loaded_model.result.mapping)

println("\nPredicting match using direct computation...")
# Call our NEW, refactored prediction function
match_predictions = predict_bivariate_match_ft(
    loaded_model.config.model_def,
    chains_for_prediction.ft,
    features
)

# --- 4. Interpret and Display the Predictions ---
home_win_prob = match_predictions.home
draw_prob = match_predictions.draw
away_win_prob = match_predictions.away

home_odds = 1 / home_win_prob
draw_odds = 1 / draw_prob
away_odds = 1 / away_win_prob

println("\n--- Predicted FT Odds (Direct Computation) ---")
println("$team_name_home Win: ", round(home_odds, digits=2), " (", round(home_win_prob*100, digits=1), "%)")
println("Draw:         ", round(draw_odds, digits=2), " (", round(draw_prob*100, digits=1), "%)")
println("$team_name_away Win:   ", round(away_odds, digits=2), " (", round(away_win_prob*100, digits=1), "%)")

"""
julia> home_win_prob = match_predictions.home
0.4381572636324575

julia> draw_prob = match_predictions.draw
0.28314645907338054

julia> away_win_prob = match_predictions.away
0.2786959033806963

julia> home_odds = 1 / home_win_prob
2.2822855695913717

julia> draw_odds = 1 / draw_prob
3.5317411465168242

julia> away_odds = 1 / away_win_prob
3.588140291513393

julia> println("\n--- Predicted FT Odds (Direct Computation) ---")

--- Predicted FT Odds (Direct Computation) ---

julia> println("$team_name_home Win: ", round(home_odds, digits=2), " (", round(home_win_prob*100, digits=1), "%)")
middlesbrough Win: 2.28 (43.8%)

julia> println("Draw:         ", round(draw_odds, digits=2), " (", round(draw_prob*100, digits=1), "%)")
Draw:         3.53 (28.3%)

julia> println("$team_name_away Win:   ", round(away_odds, digits=2), " (", round(away_win_prob*100, digits=1), "%)")
west-bromwich-albion Win:   3.59 (27.9%)

The market odds for the 1X2 (Home/Draw/Away) market hover around:

Middlesbrough to win: 6/5 (or 2.20) -> 45.5% implied probability

Draw: 9/4 (or 3.25) -> 30.8% implied probability

West Brom to win: 12/5 (or 3.40) -> 29.4% implied probability

"""



#### other game St Johnstone v Dunfermline
team_name_home = "st-johnstone"
team_name_away = "dunfermline-athletic"

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=55,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)

println("Found match: $(match_to_predict.home_team) vs $(match_to_predict.away_team).")

# --- 3. Run Prediction ---
chains_for_prediction = loaded_model.result.chains_sequence[1]
features = BayesianFootball.create_master_features(match_to_predict, loaded_model.result.mapping)

println("\nPredicting match using direct computation...")
# Call our NEW, refactored prediction function
match_predictions = predict_bivariate_match_ft(
    loaded_model.config.model_def,
    chains_for_prediction.ft,
    features
)

# --- 4. Interpret and Display the Predictions ---
home_win_prob = match_predictions.home
draw_prob = match_predictions.draw
away_win_prob = match_predictions.away

home_odds = 1 / home_win_prob
draw_odds = 1 / draw_prob
away_odds = 1 / away_win_prob

println("\n--- Predicted FT Odds (Direct Computation) ---")
println("$team_name_home Win: ", round(home_odds, digits=2), " (", round(home_win_prob*100, digits=1), "%)")
println("Draw:         ", round(draw_odds, digits=2), " (", round(draw_prob*100, digits=1), "%)")
println("$team_name_away Win:   ", round(away_odds, digits=2), " (", round(away_win_prob*100, digits=1), "%)")


"""
julia> home_win_prob = match_predictions.home
0.4180506896352648

julia> draw_prob = match_predictions.draw
0.2932691704423277

julia> away_win_prob = match_predictions.away
0.28867978220744744

julia> home_odds = 1 / home_win_prob
2.3920544201768124

julia> draw_odds = 1 / draw_prob
3.4098367669937306

julia> away_odds = 1 / away_win_prob
3.464045844684033

julia> println("\n--- Predicted FT Odds (Direct Computation) ---")

--- Predicted FT Odds (Direct Computation) ---

julia> println("$team_name_home Win: ", round(home_odds, digits=2), " (", round(home_win_prob*100, digits=1), "%)")
st-johnstone Win: 2.39 (41.8%)

julia> println("Draw:         ", round(draw_odds, digits=2), " (", round(draw_prob*100, digits=1), "%)")
Draw:         3.41 (29.3%)

julia> println("$team_name_away Win:   ", round(away_odds, digits=2), " (", round(away_win_prob*100, digits=1), "%)")
dunfermline-athletic Win:   3.46 (28.9%)

market: 

  home: 1.6 
  draw: 4.2 
  away: 3.46
"""


