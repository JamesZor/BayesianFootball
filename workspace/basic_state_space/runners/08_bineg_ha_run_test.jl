using BayesianFootball
using DataFrames
using Dates
using Turing
using ReverseDiff, Memoization
using Statistics
using Plots

# --- 1. SETUP & CONFIGURATION ---

# Configure Turing to use a performant backend
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Include the new modularized model and prediction logic
# Ensure these files are in the same directory or provide the correct path
include("./neg_bin_ar1_model.jl") 
using .AR1NegativeBinomial_ha

# include("./neg_bin_ar1_prediction.jl")
using .AR1NegBiPrediction_HA

# --- Configuration for the experiment ---
const EXPERIMENT_GROUP_NAME = "dynamic_negbin_models"
const SAVE_PATH = "./experiments"
# Make sure this path is correct for your system
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" 

# --- 2. MODEL TRAINING ---

println("--- Starting Model Training ---")

# Setup data access and add the global_round column needed for time-series models
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
# This function is crucial for any time-series model in the framework
add_global_round_column!(data_store.matches) 
# Note: Ensure your data in data_store.matches contains a 'league_id' column

# Define the model and sampling parameters
sample_config = BayesianFootball.ModelSampleConfig(500, true) # 500 warmup/adaptation steps, then 500 samples
model_def = AR1NegativeBinomialHAModel() # Use our new dynamic model
run_name = "dyn_negbin_2425_to_2526"

# Define the cross-validation split (here, just a single training period)
cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["24/25", "25/26"],
    [],
    :round
)

# Define how to map categorical features (like team names) to integer IDs
mapping_funcs = BayesianFootball.MappingFunctions(BayesianFootball.create_list_mapping)

# Combine everything into a single experiment configuration
config = ExperimentConfig(run_name, model_def, cv_config, sample_config, mapping_funcs)

# Create a global mapping from the entire dataset
global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
# Filter the data for the specified training seasons
train_df = filter(row -> row.season in config.cv_config.base_seasons, data_store.matches) 

# Compose the function that will execute the training
training_morphism = BayesianFootball.compose_training_morphism(
    config.model_def,
    config.sample_config,
    global_mapping
)

# Execute the training
start_time = now()
trained_chains = training_morphism(train_df, "Training on 24/25 and 25/26 seasons")
end_time = now()
run_duration_seconds = Dates.value(end_time - start_time) / 1000
println("Training completed in $(round(run_duration_seconds, digits=2)) seconds.")

# --- Save the results ---
result = ExperimentResult(
    [trained_chains], # Stored in a vector [cite: 63]
    global_mapping,
    hash(config),
    run_duration_seconds
)

run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
save(run_manager, result)
println("Model saved successfully for run: $(run_name) at $(run_manager.run_path)")
"""
"./experiments/dynamic_negbin_models/dyn_negbin_2425_to_2526_20250928-132059"
"""

# --- 3. PREDICTION & VISUALIZATION ---

println("\n--- Loading Model and Making Predictions ---")

# Load the model we just saved
# You can comment out the training part above and run from here if the model is already saved
file_path = run_manager.run_path 
loaded_model = load_model(file_path)
mapping = loaded_model.result.mapping
chain = loaded_model.result.chains_sequence[1]



# --- Predict a future match ---
team_name_home = "west-bromwich-albion"
team_name_away = "leicester-city"
# **IMPORTANT**: We must now specify the league_id for the match
league_name = "championship" 
league_id_to_predict = mapping.league["2"]

# Determine the next round for out-of-sample prediction
# We need to extract samples first to find the number of rounds in the training data
posterior_samples = BayesianFootball.extract_posterior_samples(
    loaded_model.config.model_def,
    chain.ft,
    mapping
)

last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1
println("Predicting for global round $(next_round) (first round after training data).")

# Create a DataFrame for the match to predict
match_to_predict = DataFrame(
    home_team = team_name_home,
    away_team = team_name_away,
    tournament_id = league_id_to_predict, # <-- KEY CHANGE: Use the required league_id
    global_round = next_round,
    # Dummy scores are required by the feature creation function
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 
)

# Create features and run prediction
features = BayesianFootball.create_master_features(match_to_predict, mapping)
predictions = predict_ar1_neg_bin_match_lines(
    loaded_model.config.model_def,
    chain,
    features,
    mapping
)

# --- Display predicted odds ---
println("\n--- Predicted Full-Time Odds for $(team_name_home) vs $(team_name_away) ---")
println("Home Win: ", round(mean(1 ./ predictions.ft.home), digits=2))
println("Away Win: ", round(mean(1 ./ predictions.ft.away_win_probs), digits=2))
println("Draw:     ", round(mean(1 ./ predictions.ft.draw_probs), digits=2))
println("Over 2.5: ", round(mean(1 ./ (1 .- predictions.ft.under_25)), digits=2))
println("BTTS:     ", round(mean(1 ./ predictions.ft.btts), digits=2))


# --- 4. VISUALIZE PARAMETER EVOLUTION ---

println("\n--- Generating plot of team strength evolution ---")

team1_id = mapping.team[team_name_home]
team2_id = mapping.team[team_name_away]

log_α_centered = posterior_samples.log_α_centered
log_β_centered = posterior_samples.log_β_centered

# [cite_start]Calculate posterior mean and standard deviation over time [cite: 72]
team1_attack_mean = vec(mean(log_α_centered[:, team1_id, :], dims=1))
team1_defense_mean = vec(mean(log_β_centered[:, team1_id, :], dims=1))
team2_attack_mean = vec(mean(log_α_centered[:, team2_id, :], dims=1))
team2_defense_mean = vec(mean(log_β_centered[:, team2_id, :], dims=1))

team1_attack_std = vec(std(log_α_centered[:, team1_id, :], dims=1))
team1_defense_std = vec(std(log_β_centered[:, team1_id, :], dims=1))
team2_attack_std = vec(std(log_α_centered[:, team2_id, :], dims=1))
team2_defense_std = vec(std(log_β_centered[:, team2_id, :], dims=1))

# Create the plot object
p = plot(
    layout=(1, 2),
    size=(1400, 500),
    legend=:outertopright,
    link=:y, # Link the y-axes of the two subplots
    xlabel="Global Round"
)

# Subplot 1: Attacking Strength
plot!(p[1], 1:n_rounds, team1_attack_mean,
    ribbon = team1_attack_std, # Show 1 standard deviation uncertainty
    fillalpha = 0.2, 
    label = team_name_home,
    title = "Attacking Strength (log α)",
    ylabel = "Parameter Value",
    lw = 2
)
plot!(p[1], 1:n_rounds, team2_attack_mean,
    ribbon = team2_attack_std,
    fillalpha = 0.2,
    label = team_name_away,
    lw = 2
)

# Subplot 2: Defensive Strength (lower is better)
plot!(p[2], 1:n_rounds, team1_defense_mean,
    ribbon = team1_defense_std,
    fillalpha = 0.2,
    label = team_name_home,
    title = "Defensive Strength (log β)",
    lw = 2
)
plot!(p[2], 1:n_rounds, team2_defense_mean,
    ribbon = team2_defense_std,
    fillalpha = 0.2,
    label = team_name_away,
    lw = 2
)

display(p)
println("\nScript finished.")
