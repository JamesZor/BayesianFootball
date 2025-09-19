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

run_manager = get_latest_run(EXPERIMENT_NAME, SAVE_PATH)
file_path = joinpath(run_manager.run_path, "result.jld2")

println("Loading model from: $file_path")
loaded_model = load_model(file_path)

# --- 2. Load Data and Find Match ---
println("Loading data to find the match...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

team_name_home = "Middlesbrough"
team_name_away = "West Brom"

match_to_predict = first(
    filter(
        row -> row.home_team == team_name_home && row.away_team == team_name_away,
        data_store.matches
    )
)
println("Found match: $(match_to_predict.home_team) vs $(match_to_predict.away_team) on $(match_to_predict.date)")

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
