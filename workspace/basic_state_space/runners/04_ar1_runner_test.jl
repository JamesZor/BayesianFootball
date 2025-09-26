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
sample_config = BayesianFootball.ModelSampleConfig(100, true) # 1500 steps, show progress bar
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
