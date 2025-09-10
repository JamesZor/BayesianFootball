using BayesianFootball
using DataFrames

data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)
mapping_functions = MappingFunctions(create_list_mapping) 
mapping = MappedData(data_store, mapping_functions)



cv_config = TimeSeriesSplitsConfig(["20/21", "21/22", "22/23", "23/24"], "24/25", :round)

splits = time_series_splits(data_store, cv_config)

for (i, (train_data, round_info)) in enumerate(splits)
    println("Split $i ($round_info): $(nrow(train_data)) rows")
end



####
include("/home/james/bet_project/models_julia/notebooks/07_data_utils/improved_window_split_setup.jl")


cv_config = Splits.TimeSeriesSplitsConfig(
    ["20/21", "21/22"],           # base seasons
    ["22/23", "23/24", "24/25"],  # multiple target seasons
    :round
)

splits = Splits.time_series_splits(data_store, cv_config)
splits_seq = Splits.time_series_splits(data_store, cv_config, :sequential)

Splits.summarize_splits(splits)
Splits.summarize_splits(splits_seq)

# Iterate as before
for (i, (train_data, round_info)) in enumerate(splits_seq)
    println("Split $i ($round_info): $(nrow(train_data)) rows")
    # Your training code here
    # if i >= 38
    #     break  # Just show first few for testing
    # end
end
