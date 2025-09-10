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
sample_config = ModelSampleConfig(100, false) # 1000 steps, no progress bar for batch runs
mapping_funcs = MappingFunctions(create_list_mapping)

# --- 2. Define Experiments to Run ---
# Format: (model_family::Symbol, model_variant::Symbol, unique_name_for_saving::String)
# experiments_to_run = [
#     (:maher, :basic, "maher_basic_2_base_1_target"),
#     # Add more runs here as you define them in the registry
#     # e.g., (:maher, :league_ha, "maher_league_ha_2b_1t"),
# ]
experiments_to_run = [
    (:maher, :basic, "maher_basic_2_base_1_target"),
    
    # ADD THIS LINE TO RUN YOUR NEW MODEL
    (:maher, :league_ha, "maher_league_ha_2b_1t") 
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
        parallel=true # Set to true for speed, false for easier debugging
    )
    
    # D. Save the results
    save(run_manager, result)
end

println("\nBatch finished successfully!")
