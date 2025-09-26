# workspace/basic_state_space/runner/)4_runner_test.jl

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



################################################################################
# Main train  script 
################################################################################

using BayesianFootball
using DataFrames
using Dates
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
# --- 1. SETUP AND INCLUDES ---

# Include the new AR(1) model definition from our workspace
include("setup.jl")
using .AR1StateSpace

# --- 2. CONFIGURATION ---

const EXPERIMENT_GROUP_NAME = "ar1_poisson_test"
const SAVE_PATH = "./experiments"
# Make sure this path is correct for your system
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" 

# --- 3. MAIN SCRIPT ---

println("▶️  Starting AR(1) Poisson Model Training Run...")

# --- Load Data ---
println("Loading data store...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# --- Pre-process Data (Phase 1) ---
add_global_round_column!(data_store.matches)
println("✅ Data loaded and preprocessed.")

# --- Define Model and Training Configurations ---
sample_config = BayesianFootball.ModelSampleConfig(500, true) # 1500 steps, show progress bar
model_def = AR1PoissonModel()
run_name = "ar1_poisson_2425_to_2526"

# Define which seasons to use for this training run
cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["24/25", "25/26"],
    [],
    :round
)

mapping_funcs = BayesianFootball.MappingFunctions(BayesianFootball.create_list_mapping)

# [cite_start]Create a single config object for saving results, mirroring your framework's pattern [cite: 85]
config = ExperimentConfig(run_name, model_def, cv_config, sample_config, mapping_funcs)

# --- Prepare for Training ---
println("Preparing training data and functions...")
global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
train_df = filter(row -> row.season in config.cv_config.base_seasons, data_store.matches) 

# Compose the training morphism (using the flexible version from Phase 3)
training_morphism = BayesianFootball.compose_training_morphism(
    config.model_def,
    config.sample_config,
    global_mapping
)

# --- Execute Training ---
println("🚀 Starting MCMC sampling on $(nrow(train_df)) matches for run: $(run_name)")
start_time = now()
trained_chains = training_morphism(train_df, "test sample")
end_time = now()
run_duration_seconds = Dates.value(end_time - start_time) / 1000
println("✅ Training complete in $(round(run_duration_seconds, digits=1)) seconds.")

# --- Package and Save Results ---
result = ExperimentResult(
    [trained_chains], # Stored in a vector to match the expected format [cite: 86]
    global_mapping,
    hash(config),
    run_duration_seconds
)

run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
save(run_manager, result)
println("💾 Model saved successfully for run: $(run_name)")
println("✔️  COMPLETED RUN.")

model = TrainedModel(config, result)


################################################################################
# Predict stuff
################################################################################
using BayesianFootball
using DataFrames
using Dates
include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")
include("/home/james/bet_project/models_julia/workspace/basic_state_space/prediction.jl")
using .AR1StateSpace
using .AR1Prediction
using StatsBase

const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" 

data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# --- Pre-process Data (Phase 1) ---
add_global_round_column!(data_store.matches)

cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["24/25", "25/26"],
    [],
    :round
)

train_df = filter(row -> row.season in cv_config.base_seasons, data_store.matches) 



file_path = "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_poisson_2425_to_2526_20250926-135921"
loaded_model = load_model(file_path)

# chains_for_prediction = loaded_model.result.chains_sequence[1]
# mapping = loaded_model.result.mapping # Get the mapping object


DataFrame(train_df[301, :])
# features = BayesianFootball.create_master_features(match_to_predict, loaded_model.result.mapping)
features = BayesianFootball.create_master_features(DataFrame(train_df[301, :]), loaded_model.result.mapping)


predictions = predict_ar1_match_lines(
    loaded_model.config.model_def,
    loaded_model.result.chains_sequence[1],
    features,
    loaded_model.result.mapping
)

id = 14025121
id = 12436574

home_win_prob = mean(predictions.ft.home)
draw_prob = mean(predictions.ft.draw)
away_win_prob = mean(predictions.ft.away)


println("\n--- Predicted FT Probability (Posterior Mean) ---")
println("Home Win: ", round(home_win_prob * 100, digits=1), "%")
println("Draw:     ", round(draw_prob * 100, digits=1), "%")
println("Away Win: ", round(away_win_prob * 100, digits=1), "%")

println("Home Win: ", round(mean( 1 ./ predictions.ft.home) , digits=2))
println("Draw:     ", round(draw_prob * 100, digits=1), "%")
println("Away Win: ", round(away_win_prob * 100, digits=1), "%")


filter(row -> row.match_id==id, data_store.odds)

team_name_home = "west-bromwich-albion"
team_name_away = "leicester-city"
global_round = 595



match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=2,
    global_round = 595,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)
features = BayesianFootball.create_master_features(match_to_predict, loaded_model.result.mapping)
predictions = predict_ar1_match_lines(
    loaded_model.config.model_def,
    loaded_model.result.chains_sequence[1],
    features,
    loaded_model.result.mapping
)

println("Home Win: ", round(mean( 1 ./ predictions.ft.home) , digits=2))
println("away Win: ", round(mean( 1 ./ predictions.ft.away) , digits=2))
println("draw Win: ", round(mean( 1 ./ predictions.ft.draw) , digits=2))

mean( 1 ./ predictions.ft.under_15)


"""
julia> println("Home Win: ", round(mean( 1 ./ predictions.ft.home) , digits=2))
Home Win: 2.31

julia> println("away Win: ", round(mean( 1 ./ predictions.ft.away) , digits=2))
away Win: 4.46

julia> println("draw Win: ", round(mean( 1 ./ predictions.ft.draw) , digits=2))
draw Win: 3.93

julia> mean( 1 ./ predictions.ft.under_15)
3.511803259293509
"""


match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=2,
    global_round = 530,
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0 # Dummy data
)
features = BayesianFootball.create_master_features(match_to_predict, loaded_model.result.mapping);
predictions = predict_ar1_match_lines(
    loaded_model.config.model_def,
    loaded_model.result.chains_sequence[1],
    features,
    loaded_model.result.mapping
);

"""

julia> println("Home Win: ", round(mean( 1 ./ predictions.ft.home) , digits=2))
Home Win: 2.33

julia> println("away Win: ", round(mean( 1 ./ predictions.ft.away) , digits=2))
away Win: 4.38

julia> println("draw Win: ", round(mean( 1 ./ predictions.ft.draw) , digits=2))
draw Win: 3.91

julia> mean( 1 ./ predictions.ft.under_15)
3.4724346278281297
"""
###

using Statistics # Required for mean()

# --- 1. Load your trained model ---
file_path = "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_poisson_2425_to_2526_20250926-135921"
loaded_model = load_model(file_path)
mapping = loaded_model.result.mapping
chains = loaded_model.result.chains_sequence[1]

# --- 2. Determine the number of rounds the model was trained on ---
# We can infer this from the dimensions of the model's parameters.
# This extracts all posterior samples, including the number of rounds.
posterior_samples = BayesianFootball.extract_posterior_samples(
    loaded_model.config.model_def,
    chains.ft,
    mapping
)
last_training_round = posterior_samples.n_rounds
next_round = last_training_round + 1

println("Model was trained on $(last_training_round) rounds. Predicting for round $(next_round)...")

# --- 3. Set up the match to predict for the next round ---
team_name_home = "west-bromwich-albion"
team_name_away = "leicester-city"

match_to_predict = DataFrame(
    home_team=team_name_home,
    away_team=team_name_away,
    tournament_id=2,
    global_round = next_round, # Use the calculated next_round
    home_score_ht=0, away_score_ht=0, home_score=0, away_score=0
)

# --- 4. Run the prediction ---
features = BayesianFootball.create_master_features(match_to_predict, mapping)

predictions = predict_ar1_match_lines(
    loaded_model.config.model_def,
    chains,
    features,
    mapping
)

println("\n--- Predicted Odds for Next Round ---")
println("Home Win: ", round(mean( 1 ./ predictions.ft.home), digits=2))
println("Away Win: ", round(mean( 1 ./ predictions.ft.away), digits=2))
println("Draw:     ", round(mean( 1 ./ predictions.ft.draw), digits=2))



# v2 
using Plots
using Statistics

# --- 1. Define the teams and find their IDs from the mapping ---
team1_name = "west-bromwich-albion"
team2_name = "leicester-city"


team1_id = loaded_model.result.mapping.team[team1_name]
team2_id = loaded_model.result.mapping.team[team2_name]

# --- 2. Get the full time-series of the parameters ---
log_α_centered = posterior_samples.log_α_centered
log_β_centered = posterior_samples.log_β_centered
n_rounds = posterior_samples.n_rounds

# --- 3. Calculate the posterior mean AND STANDARD DEVIATION over time ---
# Mean calculations
team1_attack_mean = vec(mean(log_α_centered[:, team1_id, :], dims=1))
team1_defense_mean = vec(mean(log_β_centered[:, team1_id, :], dims=1))
team2_attack_mean = vec(mean(log_α_centered[:, team2_id, :], dims=1))
team2_defense_mean = vec(mean(log_β_centered[:, team2_id, :], dims=1))

# Standard deviation calculations
team1_attack_std = vec(std(log_α_centered[:, team1_id, :], dims=1))
team1_defense_std = vec(std(log_β_centered[:, team1_id, :], dims=1))
team2_attack_std = vec(std(log_α_centered[:, team2_id, :], dims=1))
team2_defense_std = vec(std(log_β_centered[:, team2_id, :], dims=1))


# --- 4. Create the 1x2 plot with ribbons ---
p = plot(
    layout=(1, 2),
    size=(1400, 500),
    legend=:outertopright,
    link=:y,
    xlabel="Global Round"
)

# Subplot 1: Attacking Strength
plot!(p[1], 1:n_rounds, team1_attack_mean,
    ribbon = 1 .* team1_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,                # Make ribbon transparent
    label = team1_name,
    title = "Attacking Strength (log α)",
    ylabel = "Parameter Value",
    lw = 2
)
plot!(p[1], 1:n_rounds, team2_attack_mean,
    ribbon = 1 .* team2_attack_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)

# Subplot 2: Defensive Strength
plot!(p[2], 1:n_rounds, team1_defense_mean,
    ribbon = 1 .* team1_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team1_name,
    title = "Defensive Strength (log β)",
    lw = 2
)
plot!(p[2], 1:n_rounds, team2_defense_mean,
    ribbon = 1 .* team2_defense_std, # <-- ADDED RIBBON
    fillalpha = 0.2,
    label = team2_name,
    lw = 2
)

# Display the plot
display(p)
