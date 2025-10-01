# workspace/basic_state_space/runners/train_predict_poisson.jl

using BayesianFootball
using DataFrames
using Dates
using Turing
using Statistics

# For performance with this model
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# --- 1. SETUP AND INCLUDES ---

# Include our new, self-contained model module
include("../models/ar1_poisson.jl")
using .AR1Poisson

# Include the new utilities module
include("../analysis/utils.jl")
using .SSMUtils

# --- 2. CONFIGURE PATHS AND CONSTANTS ---

# Adjust these paths to match your system
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"
const SAVE_PATH = "./experiments" # Saves in the current directory

# --- 3. LOAD AND PREPARE DATA ---

println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# Add the essential :global_round column for our state-space model
add_global_round_column!(data_store.matches)

# --- 4. CONFIGURE AND RUN TRAINING ---

println("Configuring and running the training experiment...")

# Define the model we want to train
model_def = AR1PoissonModel()

# For this simple test, we'll train on one season and validate on nothing,
# just to get a trained model quickly.
cv_config = BayesianFootball.SingleSplitConfig(
    ["24/25"], # Use one season for training
    []         # No validation set
)

# Use a small number of samples for a quick test run
sample_config = BayesianFootball.ModelSampleConfig(250, true) # 250 steps, show progress

# Define the full experiment
run_name = "ar1_poisson_simple_test"
experiment_config = ExperimentConfig(run_name, model_def, cv_config, sample_config)

# Run the experiment! This will train the model and save the MCMC chain to a file.
run_results = run_experiment(data_store, experiment_config, SAVE_PATH)

# --- 5. LOAD THE TRAINED MODEL AND PREDICT ---

println("\n--- Starting Prediction Step ---")

# The path to the trained model is in the results object
# We'll use the 'full train' (ft) model, which is trained on all specified data
model_path = run_results.ft_path
println("Loading trained model from: $(model_path)")
trained_chains = load_model(model_path)

# Create a global mapping of all teams/leagues from the original data
# This is crucial so the model knows the ID for "Man City", "Arsenal", etc.
global_mapping = MappedData(data_store)

# Define the match we want to predict
# NOTE: The teams must exist in the training data
predict_df = DataFrame(
    home_team="Man City",
    away_team="Arsenal",
    league="Premier League",
    # We need to find the next available global_round after the training data ends
    global_round=maximum(data_store.matches.global_round) + 1
)
println("Predicting match: $(predict_df.home_team[1]) vs $(predict_df.away_team[1])")

# Prepare the features for the prediction function
# This function converts team names and leagues into the integer IDs the model needs
features, mapping = BayesianFootball.get_features_and_mapping(
    model_def,
    predict_df,
    global_mapping # Pass the global mapping to ensure IDs are consistent
)

# Call the generic `predict` function. Julia will dispatch to our model's implementation.
match_prediction = predict(model_def, trained_chains, features, mapping)

# --- 6. DISPLAY RESULTS ---

# The `match_prediction` object contains all the posterior distributions.
# Let's just look at the average expected goals for a simple check.
avg_lambda_home = mean(match_prediction.ft.λ_home)
avg_lambda_away = mean(match_prediction.ft.λ_away)

println("\n--- Prediction Results ---")
@info "Average Expected Goals (Home): $(round(avg_lambda_home, digits=2))"
@info "Average Expected Goals (Away): $(round(avg_lambda_away, digits=2))"

println("\n✅ Script finished successfully!")
