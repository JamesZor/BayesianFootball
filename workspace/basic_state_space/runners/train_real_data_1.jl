# runners/train_ar1_batch.jl
using BayesianFootball
using DataFrames
using Dates
using Turing
using Base.Threads

# --- Performance Libraries ---
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# --- 1. Includes for Model Definitions ---
# Make sure these paths are correct for your project structure
include("../models/ar1_poisson_ha.jl")
using .AR1PoissonHA
include("../models/ar1_negative_binomial_ha.jl")
using .AR1NegativeBinomialHA

# --- 2. Constants and Configuration ---
const EXPERIMENT_GROUP_NAME = "ar1_model_comparison"
const SAVE_PATH = "./experiments"
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26"

# --- 3. Utility Functions ---
"""
    add_global_round_column!(matches_df::DataFrame)

Adds a `:global_round` column in-place to the DataFrame.
"""
function add_global_round_column!(matches_df::DataFrame)
    sort!(matches_df, :match_date)
    num_matches = nrow(matches_df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(matches_df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end
    
    matches_df.global_round = global_rounds
    println("✅ Successfully added `:global_round` column. Found $(global_round_counter) unique time steps.")
    return matches_df
end

# ============================================================================
#  CORE TRAINING FUNCTION (ADAPTED FOR PARALLEL EXECUTION)
# ============================================================================
"""
    run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)

Runs a single, complete MCMC training instance for a given model.
"""
function run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)
    run_name = model_spec.name # Use the simple name for the run
    
    println("[Thread $(threadid())] ▶️  STARTING RUN: $(run_name)")
    
    config = ExperimentConfig(run_name, model_spec.def, cv_config, sample_config)

    mapping_funcs = BayesianFootball.MappingFunctions(
        BayesianFootball.create_list_mapping
  )

    # Prepare data and training morphism
    train_df = filter(row -> row.season in config.cv_config.base_seasons, data_store.matches)
    training_morphism = BayesianFootball.compose_training_morphism(
        config.model_def,
        config.sample_config,
        mapping_funcs
    )

    # Execute the training
    println("[Thread $(threadid())]    Starting MCMC sampling on $(nrow(train_df)) matches for run: $(run_name)")
    start_time = now()
    trained_chains = training_morphism(train_df, "Full History")
    end_time = now()
    run_duration_seconds = Dates.value(end_time - start_time) / 1000
    
    println("[Thread $(threadid())]    ✅ Training complete for $(run_name) in $(round(run_duration_seconds, digits=1)) seconds.")

    # Package and save the results
    result = ExperimentResult([trained_chains], global_mapping, hash(config), run_duration_seconds)
    run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
    save(run_manager, result)
    println("[Thread $(threadid())]    💾 Model saved for run: $(run_name)")
    println("[Thread $(threadid())] ✔️  COMPLETED RUN: $(run_name)")
end


# ============================================================================
#  MAIN SCRIPT EXECUTION
# ============================================================================

# --- 1. Load Data and Define Configurations ---
println("Loading data store...")
data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
add_global_round_column!(data_store.matches)
println("✅ Data loaded and prepared.")

# Define the models you want to train
model_definitions = [
    (name="ar1_poisson_ha", def=AR1PoissonHAModel()),
    (name="ar1_neg_bin_ha", def=AR1NegativeBinomialHAModel())
]

# Define the single data configuration for this experiment
# cv_config = BayesianFootball.TimeSeriesSplitsConfig(
#     ["20/21", "21/22", "22/23", "23/24", "24/25", "25/26"],
#     [],
#     :round
# )

cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["24/25", "25/26"],
    [],
    :round
)
sample_config = BayesianFootball.ModelSampleConfig(10, true) # Using 500 samples for a quicker run

# --- 2. Create Global Mapping ---
println("Creating global data mapping...")

mapping_funcs = BayesianFootball.MappingFunctions(
  BayesianFootball.create_list_mapping
)
# This is the correct call, which relies on the constructor defined in your BayesianFootball package
global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
println("✅ Mapping complete.")

# --- 3. Run Training in Parallel ---
num_tasks = length(model_definitions)
println("\n--- Starting Batch Training Run ---")
println("Found $(num_tasks) models to train on $(nthreads()) threads. 🚀")

@threads for model_spec in model_definitions
    run_training_instance(model_spec, cv_config, sample_config, data_store, global_mapping)
end

println("\n--- All training runs completed successfully! ---")
