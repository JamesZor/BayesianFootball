# scripts/evaluate.jl
using BayesianFootball, DataFrames

# --- 1. Setup ---
const EXPERIMENT_NAME = "scottish_league_initial_test"
const SAVE_PATH = "./experiments"
experiment_path = joinpath(SAVE_PATH, EXPERIMENT_NAME)

# --- 2. List and Select a Run ---
available_runs = list_runs(experiment_path)
if isempty(available_runs) error("No runs found!") end

println("Available models for experiment '$EXPERIMENT_NAME':")
println(available_runs)
print("\nEnter the ID of the model you want to use for prediction: ")
chosen_id = parse(Int, readline())

run_path = available_runs[available_runs.id .== chosen_id, :path][1]
# --- 3. Load the Model ---
println("\nLoading model from: $run_path")
model = load_model(run_path)
println("✅ Model '$(model.config.name)' loaded successfully.")


# --- 4. Load Target Data ---
# For this example, we'll create a dummy DataFrame of matches to predict
target_matches = DataFrame(
    match_id = [101, 102, 201],
    round = [1, 1, 2], # Two matches in round 1, one in round 2
    home_team = ["celtic", "motherwell", "rangers"],
    away_team = ["rangers", "hibernian", "celtic"],
    tournament_id = [54, 54, 54],
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)
println("\nPredicting for $(nrow(target_matches)) matches...")

# --- 5. Generate Predictions ---
all_predictions = predict_target_season(model, target_matches)

using Statistics
# --- 6. Display Results ---
println("\n--- Prediction Summary ---")
for (match_id, preds) in all_predictions
    match_info = target_matches[target_matches.match_id .== match_id, :]
    hw = round(mean(preds.ft.home) * 100, digits=1)
    dr = round(mean(preds.ft.draw) * 100, digits=1)
    aw = round(mean(preds.ft.away) * 100, digits=1)
    println("Match: $(match_info.home_team[1]) vs $(match_info.away_team[1]) -> Home: $hw%, Draw: $dr%, Away: $aw%")
end


###
c = BayesianFootball.get_chains_for_match(model, target_matches[1, :])

f = BayesianFootball.create_master_features(SubDataFrame(target_matches, 1:1, :), model.result.mapping)

m = BayesianFootball.predict_match_lines(model.config.model_def, c, f, model.result.mapping)
