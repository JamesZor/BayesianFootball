using BayesianFootball
using DataFrames
using Dates
using Base.Threads

# --- 1. Setup and Includes ---
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
using .BivariateMaher

# --- Constants ---
const EXPERIMENT_GROUP_NAME = "bivar_chain_length"
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
    run_name = "$(model_spec.name)_steps_$(sample_config.steps)"
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
    println("[Thread $(threadid())]Model saved for run: $(run_name)")
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


model_definitions = [
    (name="maher_bivariate", def=MaherBivariate())
]

cv_config = BayesianFootball.TimeSeriesSplitsConfig(["20/21"], [], :round)

sample_configs_list = [
        BayesianFootball.ModelSampleConfig(10, true),
        BayesianFootball.ModelSampleConfig(50, true),
        # BayesianFootball.ModelSampleConfig(500, true),
        # BayesianFootball.ModelSampleConfig(1000, true),
]

# --- 2. Create Global Mapping (Using the correct constructor) ---
println("Creating global data mapping...")
mapping_funcs = BayesianFootball.MappingFunctions(
  BayesianFootball.create_list_mapping
)
# This is the correct call, which relies on the constructor defined in your BayesianFootball package
global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
println("✅ Mapping complete.")

tasks = [(m, c) for m in model_definitions for c in sample_configs_list]
num_tasks = length(tasks)

println("\n--- Starting Batch Training Run ---")
println("Found $(num_tasks) tasks to run on $(nthreads()) threads. 🚀")

@threads for (model_spec, sample_config) in tasks
    run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)
end

println("\n--- All training runs completed successfully! ---")
