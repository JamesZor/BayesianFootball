# ver under test TEST:
# scripts/run_batch.jl
using BayesianFootball
using Dates

# --- 1. Setup ---
const EXPERIMENT_NAME = "scottish_league_initial_test"
const SAVE_PATH = "./experiments"
# const DATA_PATH = "/home/james/bet_project/football_data/scot_nostats_20_to_24"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"
println("Loading data...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)

# Define shared configs for this experiment batch
cv_config = BayesianFootball.TimeSeriesSplitsConfig(["20/21", "21/22","22/23", "23/24", "24/25"], ["25/26"], :round)
sample_config = BayesianFootball.ModelSampleConfig(10, true)
mapping_funcs = MappingFunctions(create_list_mapping)

# --- 1.5 SAVE EXPERIMENT METADATA ---
# This is called once for the entire batch to save shared info.
save_experiment_metadata(
    EXPERIMENT_NAME, SAVE_PATH, DATA_PATH, cv_config, sample_config
)

# --- 2. Define Experiments to Run ---
experiments_to_run = [
    (:maher, :basic, "maher_basic"),
    # (:maher, :league_ha, "maher_league_ha")
]

# --- 3. Execute Workflow ---
println("\nStarting experiment batch: $EXPERIMENT_NAME")
for (family, variant, name) in experiments_to_run
    println("\n" * "="^50)
    println("🚀 Starting Run: $name")
    
    config = create_experiment_config(
        name, family, variant, cv_config, sample_config, mapping_funcs
    )
    
    run_manager = prepare_run(EXPERIMENT_NAME, config, SAVE_PATH)

    result = train_all_splits(
        data_store,
        config.cv_config,
        config.model_def,
        config.sample_config,
        config.mapping_funcs;
        parallel=false
    )
    
    save(run_manager, result)
    println("✅ Finished run: $name.")
end

println("\nBatch finished successfully!")


###
family = :maher 
variant = :basic 
name = "maher_basic"
config = create_experiment_config(
    name, family, variant, cv_config, sample_config, mapping_funcs
)

run_manager = prepare_run(EXPERIMENT_NAME, config, SAVE_PATH)

mapping = BayesianFootball.MappedData(data_store, config.mapping_funcs)

training_morphism = BayesianFootball.compose_training_morphism(
        config.model_def,
        config.sample_config,
        mapping
    )

a = training_morphism(data_store.matches,"")



c = BayesianFootball.get_chains_for_match(model, target_matches[1, :])

f = BayesianFootball.create_master_features(SubDataFrame(target_matches, 1:1, :), model.result.mapping)

m = BayesianFootball.predict_match_lines(model.config.model_def, c, f, model.result.mapping)
