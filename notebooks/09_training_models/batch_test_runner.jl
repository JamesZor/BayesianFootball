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
sample_config = ModelSampleConfig(10, false) # Using 100 steps for a quick test
mapping_funcs = MappingFunctions(create_list_mapping)

# --- 2. Define Experiments to Run ---
# The format is the same as before, powered by the new registry.
experiments_to_run = [
    (:maher, :basic, "maher_basic"),
    (:maher, :league_ha, "maher_league_ha") # Running the league home advantage model
]

# --- 3. Execute Workflow ---
println("Starting experiment batch: $EXPERIMENT_NAME")
for (family, variant, name) in experiments_to_run
    println("\n" * "="^50)
    println("🚀 Starting Run: $name")
    
    # A. Create the full experiment configuration from the registry.
    # This function now returns the new ExperimentConfig with a .model_def field.
    config = create_experiment_config(
        name, family, variant, cv_config, sample_config, mapping_funcs
    )
    
    # B. Prepare the run directory (Unchanged).
    # NOTE: You'll need to have a `prepare_run` and `save` function defined,
    # presumably in `experiments/persistence.jl`.
    run_manager = prepare_run(EXPERIMENT_NAME, config, SAVE_PATH)

    # C. Execute the core training pipeline.
    # The main change is here: pass `config.model_def` to the pipeline.
    result = train_all_splits(
        data_store,
        config.cv_config,
        config.model_def,       
        config.sample_config,
        config.mapping_funcs;
        parallel=true         
    )
    
    # D. Save the results (Unchanged).
    save(run_manager, result)
    println("✅ Finished run: $name. Result has $(length(result.chains_sequence)) trained chains.")
end

println("\nBatch finished successfully!")

