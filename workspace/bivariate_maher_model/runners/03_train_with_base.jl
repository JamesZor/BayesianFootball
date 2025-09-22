using BayesianFootball
using DataFrames
using Dates
using Base.Threads

# --- 1. Setup and Includes ---
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
using .BivariateMaher

# --- Constants ---
const EXPERIMENT_GROUP_NAME = "model_comparison_all"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"

# ============================================================================
#  CORE TRAINING FUNCTION
# ============================================================================
"""
    run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)

Runs a single, complete MCMC training instance for a given model and data configuration.
"""
function run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)
    
    # --- A. Configure the specific run ---
    season_str = join(cv_config.base_seasons, "_")
    run_name = "$(model_spec.name)_seasons_$(replace(season_str, "/" => ""))"
    
    # Use threadid for clearer parallel logging
    println("[Thread $(threadid())] ▶️  STARTING RUN: $(run_name)")
    
    mapping_funcs = BayesianFootball.MappingFunctions(
        BayesianFootball.create_list_mapping
  )

    config = ExperimentConfig(
        run_name,
        model_spec.def,
        cv_config,
        sample_config,
        mapping_funcs
    )

    # --- B. Prepare data and training morphism ---
    train_df = filter(row -> row.season in config.cv_config.base_seasons, data_store.matches)

    training_morphism = BayesianFootball.compose_training_morphism(
        config.model_def,
        config.sample_config,
        global_mapping
    )

    # --- C. Execute the training ---
    println("[Thread $(threadid())]   Starting MCMC sampling on $(nrow(train_df)) matches for run: $(run_name)")
    start_time = now()
    trained_chains = training_morphism(train_df, "Full History")
    end_time = now()
    run_duration_seconds = Dates.value(end_time - start_time) / 1000
    
    println("[Thread $(threadid())]   ✅ Training complete for $(run_name) in $(round(run_duration_seconds, digits=1)) seconds.")

    # --- D. Package and save the results ---
    result = ExperimentResult(
        [trained_chains],
        global_mapping,
        hash(config),
        run_duration_seconds
    )
    
    run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
    save(run_manager, result)
    println("[Thread $(threadid())]   💾 Model saved for run: $(run_name)")
    println("[Thread $(threadid())] ✔️  COMPLETED RUN: $(run_name)")
end


# ============================================================================
#  MAIN SCRIPT EXECUTION
# ============================================================================

# --- 1. Load Data and Define Configurations ---
println("Loading data store...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
println("✅ Data loaded.")

sample_config = BayesianFootball.ModelSampleConfig(2000, true)

model_definitions = [
    (name="maher_basic", def=MaherBasic()),
    (name="maher_bivariate", def=MaherBivariate())
]

cv_splits = [
    BayesianFootball.TimeSeriesSplitsConfig(["20/21","21/22", "22/23","23/24","24/25", "25/26"], [], :round),
    BayesianFootball.TimeSeriesSplitsConfig(["21/22", "22/23","23/24","24/25", "25/26"], [], :round),
    BayesianFootball.TimeSeriesSplitsConfig(["22/23","23/24","24/25", "25/26"], [], :round),
    BayesianFootball.TimeSeriesSplitsConfig(["23/24","24/25", "25/26"], [], :round),
    BayesianFootball.TimeSeriesSplitsConfig(["24/25", "25/26"], [], :round),
    BayesianFootball.TimeSeriesSplitsConfig(["25/26"], [], :round)
]

# --- 2. Create Global Mapping (Using the correct constructor) ---
println("Creating global data mapping...")
mapping_funcs = BayesianFootball.MappingFunctions(
  BayesianFootball.create_list_mapping
)
# This is the correct call, which relies on the constructor defined in your BayesianFootball package
global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
println("✅ Mapping complete.")


# --- 3. Create Task List and Run in Parallel ---
tasks = [(m, c) for m in model_definitions for c in cv_splits]
num_tasks = length(tasks)

println("\n--- Starting Batch Training Run ---")
println("Found $(num_tasks) tasks to run on $(nthreads()) threads. 🚀")

@threads for (model_spec, cv_config) in tasks
    run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)
end

println("\n--- All training runs completed successfully! ---")
