# scripts/train_models.jl

using BayesianFootball
using DataFrames
using Dates
using Turing
using Statistics

# For performance with SSM models
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# --- 1. CONFIGURE PATHS ---
# Adjust these paths to match your system
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"
const SAVE_PATH = "./experiments"

# --- 2. DEFINE THE BATCH OF MODELS TO TRAIN ---
# This list makes it easy to add or remove models from a training run.
models_to_train = [
    (
        name="ar1_poisson",
        def=BayesianFootball.AR1Poisson.AR1PoissonModel()
    ),
    (
        name="ar1_neg_bin_ha",
        def=BayesianFootball.AR1NegativeBinomialHA.AR1NegativeBinomialHAModel()
    ),
    # You can add new model variations here later, e.g.:
    # (
    #     name="ar1_poisson_weak_priors",
    #     def=BayesianFootball.AR1PoissonWeakPriors.AR1PoissonWeakPriorsModel()
    # ),
]

# --- 3. CONFIGURE THE EXPERIMENT ---
# For this simple test, we'll train on one season
cv_config = BayesianFootball.SingleSplitConfig(
    ["24/25"], # Training seasons
    []         # Validation seasons (none for now)
)

# Use a small number of samples for a quick test run
sample_config = BayesianFootball.ModelSampleConfig(
    250,  # Number of sampling steps
    true  # Show progress bar
)

# --- 4. LOAD AND PREPARE DATA ---
println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# Add the essential :global_round column for our state-space models
add_global_round_column!(data_store.matches)
println("Data loaded and prepared.")

# --- 5. RUN THE TRAINING LOOP ---
println("\n--- Starting Batch Training Run ---")

for model in models_to_train
    println("\n>>> Training model: $(model.name)")

    # Create a unique name for this specific run
    run_name = "$(model.name)_" * Dates.format(now(), "yyyymmdd_HHMMSS")
    
    experiment_config = ExperimentConfig(run_name, model.def, cv_config, sample_config)

    # This is your existing function that handles the training and saving
    run_experiment(data_store, experiment_config, SAVE_PATH)

    println(">>> Finished training: $(model.name)")
end

println("\n✅ Batch training finished successfully!")
