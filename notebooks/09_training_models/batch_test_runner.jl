# scripts/run_batch.jl
using BayesianFootball

# --- 1. Setup ---
const EXPERIMENT_NAME = "scottish_league_initial_test"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football_data/scot_nostats_20_to_24" # UPDATE THIS PATH

println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# Define shared configs for this experiment batch
cv_config = BayesianFootball.TimeSeriesSplitsConfig(["20/21", "21/22"], ["22/23"], :round)
sample_config = ModelSampleConfig(100, true) # 1000 steps, no progress bar for batch runs
mapping_funcs = MappingFunctions(create_list_mapping)

# --- 2. Define Experiments to Run ---
# Format: (model_family::Symbol, model_variant::Symbol, unique_name_for_saving::String)
experiments_to_run = [
    (:maher, :basic, "maher_basic_2_base_1_target"),
    # Add more runs here as you define them in the registry
    # e.g., (:maher, :league_ha, "maher_league_ha_2b_1t"),
]
experiments_to_run = [
    (:maher, :basic, "maher_basic"),
    
    # ADD THIS LINE TO RUN YOUR NEW MODEL
    (:maher, :league_ha, "maher_league") 
]

# --- 3. Execute Workflow ---
println("Starting experiment batch: $EXPERIMENT_NAME")
for (family, variant, name) in experiments_to_run
    println("\n" * "="^50)
    println("🚀 Starting Run: $name")
    
    # A. Create the full experiment configuration from the registry
    config = create_experiment_config(
        name, family, variant, cv_config, sample_config, mapping_funcs
    )
    
    # B. Prepare the run directory and get the manager object
    run_manager = prepare_run(EXPERIMENT_NAME, config, SAVE_PATH)

    # C. Execute the core training pipeline
    result = train_all_splits(
        data_store,
        config.cv_config,
        config.model_config,
        config.sample_config,
        config.mapping_funcs;
        parallel=false # Set to true for speed, false for easier debugging
    )
    
    # D. Save the results
    save(run_manager, result)
end

println("\nBatch finished successfully!")


####
using DataFrames
using BayesianFootball

# --- 1. Standard Setup ---
const DATA_PATH = "/home/james/bet_project/football_data/scot_nostats_20_to_24"
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# --- 2. Create the necessary configs ---
sample_config = ModelSampleConfig(500, true) # Fewer steps, show progress bar
mapping_funcs = MappingFunctions(create_list_mapping)

# IMPORTANT: The mapping must be created from the full data_store
# to ensure all teams and leagues are included.
mapping = MappedData(data_store, mapping_funcs)

# Get a model config from the registry
model_config = BayesianFootball.ALL_MODEL_CONFIGS[:maher][:basic]

# --- 3. Manually Create Your Data Split ---
# For example, let's train only on the "20/21" season data.
training_df = filter(row -> row.season == "20/21", data_store.matches)
training_df = data_store.matches
println("Training on a manual split of $(nrow(training_df)) matches.")


# --- 4. Call the New Function ---
trained_chains = BayesianFootball.train_single_split(
    training_df,
    model_config,
    sample_config,
    mapping;
    info="Manual test on 20/21 season"
)

# --- 5. Inspect the Results ---
println("\nTraining finished!")
println("Chains for split: $(trained_chains.round_info)")
println("Number of samples in data: $(trained_chains.n_samples)")
println("\nFull-Time Model Summary:")
display(summarystats(trained_chains.ft))
