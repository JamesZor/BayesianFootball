using BayesianFootball
using DataFrames
using Dates
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl") # Include the new model definition
using .BivariateMaher # Use the new module

# --- 1. General Setup ---
const EXPERIMENT_NAME = "bivariate_maher_test"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"

println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# --- 2. Configuration ---
cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    [ #"20/21", "21/22", "22/23", "23/24",
    "24/25", "25/26"],
    [],
    :round
)
# Use a small sample size for the initial test run
sample_config = BayesianFootball.ModelSampleConfig(100, true)
mapping_funcs = BayesianFootball.MappingFunctions(BayesianFootball.create_list_mapping)

# --- 3. Define the Bivariate Maher Model Configuration ---
println("Configuring for the Bivariate Maher model...")
config_bivariate = create_experiment_config(
    "bivariate_maher_verification",
    :maher,
    :bivariate,
    cv_config,
    sample_config,
    mapping_funcs
)

# --- 4. Training Set Preparation ---
println("Preparing a single training set...")
train_df = filter(row -> row.season in config_bivariate.cv_config.base_seasons, data_store.matches)

println("Composing the training function...")
mapping = BayesianFootball.MappedData(data_store, config_bivariate.mapping_funcs)

# The training morphism needs to be updated to use the bivariate model
training_morphism = BayesianFootball.compose_training_morphism(
    BivariateMaher.maher_bivariate_model, # Pass the bivariate model function
    config_bivariate.sample_config,
    mapping
)

# --- 5. Run the Training ---
println("Starting training on $(nrow(train_df)) matches...")
trained_chains = training_morphism(train_df, "Full History")
println("✅ Training complete.")

# --- 6. Package, Save, and Load the Model ---
result = ExperimentResult(
    [trained_chains],
    mapping,
    hash(config_bivariate),
    0.0
)

model = TrainedModel(config_bivariate, result)

run_manager = prepare_run(EXPERIMENT_NAME, config_bivariate, SAVE_PATH)
save(run_manager, result)
println("💾 Bivariate Maher model saved successfully.")
