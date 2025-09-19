using BayesianFootball
using DataFrames
using Dates
using Base.Threads # Import the Threads module

# --- 1. Setup and Includes ---
# Include your experimental bivariate model definition
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/setup.jl")
using .BivariateMaher # Use the new module

# Constants for the experiment batch
const EXPERIMENT_GROUP_NAME = "model_comparison_runs_threaded"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"

# --- 2. Load Data Once ---
println("Loading data store...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
println("✅ Data loaded.")

# --- 3. Define Configurations for the Batch Run ---

# MCMC sampling configuration with 2,000 steps
sample_config = BayesianFootball.ModelSampleConfig(2000, true)

# Define the two model types to be tested
model_definitions = [
    (name="maher_basic", def=MaherBasic()),
    (name="maher_bivariate", def=MaherBivariate())
]

# Define the two CV splits to be used
cv_splits = [
    BayesianFootball.TimeSeriesSplitsConfig(["24/25", "25/26"], [], :round),
    BayesianFootball.TimeSeriesSplitsConfig(["25/26"], [], :round)
]

# Standard mapping functions
mapping_funcs = BayesianFootball.MappingFunctions(
    BayesianFootball.create_team_mapping_func,
    BayesianFootball.create_league_mapping_func
)
# Create a global mapping based on all available data
println("Creating global data mapping...")
global_mapping = BayesianFootball.MappedData(data_store.matches, mapping_funcs.team_mapping_func, mapping_funcs.league_mapping_func)
println("✅ Mapping complete.")

# --- 4. Main Training Loop (Parallelized) ---

# First, create a collection of all the tasks to run
tasks = [(m, c) for m in model_definitions for c in cv_splits]
num_tasks = length(tasks)

println("\n--- Starting Batch Training Run ---")
println("Found $(num_tasks) tasks to run on $(nthreads()) threads. 🚀")

# Use Threads.@threads to run the loop in parallel
@threads for (model_spec, cv_config) in tasks
    
    # --- A. Configure the specific run ---
    
    # Create a descriptive name for this specific experiment run
    season_str = join(cv_config.base_seasons, "_")
    run_name = "$(model_spec.name)_seasons_$(replace(season_str, "/" => ""))"
    
    # Thread-safe logging with threadid
    println("[Thread $(threadid())] ▶️  STARTING RUN: $(run_name)")
    
    config = ExperimentConfig(
        run_name,
        model_spec.def,
        cv_config,
        sample_config,
        mapping_funcs
    )

    # --- B. Prepare data and training function for this run ---
    
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

    model = TrainedModel(config, result)

    run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
    save(run_manager, result)
    println("[Thread $(threadid())]   💾 Model saved successfully for run: $(run_name)")
    println("[Thread $(threadid())] ✔️  COMPLETED RUN: $(run_name)")
end

println("\n--- All training runs completed successfully! ---")
