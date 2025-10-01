using BayesianFootball
using DataFrames
using Dates
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)


using .AR1NegBinVectorized

# using .TestModel


const EXPERIMENT_GROUP_NAME = "ar1_neg_bi"
const SAVE_PATH = "./experiments"
# Make sure this path is correct for your system
const DATA_PATH = "/home/james/bet_project/football/uk_football_data_20_26" 



data_files = DataFiles(DATA_PATH)
data_store = DataStore(data_files)
add_global_round_column!(data_store.matches)



sample_config = BayesianFootball.ModelSampleConfig(10, true) # 1500 steps, show progress bar
model_def = AR1NegBinVectorizedModel()
# model_def = SimplePoissonModel()
run_name = "ar1_negbi_vec_test_2425_to_2526"



cv_config = BayesianFootball.TimeSeriesSplitsConfig(
    ["24/25", "25/26"],
    [],
    :round
)

mapping_funcs = BayesianFootball.MappingFunctions(BayesianFootball.create_list_mapping)

config = ExperimentConfig(run_name, model_def, cv_config, sample_config, mapping_funcs)

global_mapping = BayesianFootball.MappedData(data_store, mapping_funcs)
train_df = filter(row -> row.season in config.cv_config.base_seasons, data_store.matches) 

training_morphism = BayesianFootball.compose_training_morphism(
    config.model_def,
    config.sample_config,
    global_mapping
)


# train the model
start_time = now()
trained_chains = training_morphism(train_df, "test sample")
end_time = now()
run_duration_seconds = Dates.value(end_time - start_time) / 1000

# save 
result = ExperimentResult(
    [trained_chains], # Stored in a vector to match the expected format [cite: 86]
    global_mapping,
    hash(config),
    run_duration_seconds
)

run_manager = prepare_run(EXPERIMENT_GROUP_NAME, config, SAVE_PATH)
save(run_manager, result)
println("Model saved successfully for run: $(run_name)")

