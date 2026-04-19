using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics



data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

# create ds for 24/25 
df = filter(row -> row.season=="24/25", data_store.matches)
df = BayesianFootball.Data.add_match_week_column(df)

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)
# large v2 
ds.matches.split_col = max.(0, ds.matches.match_week .- 14);


### dev 
#
# splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #
# data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
# feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


# 1. Expanding Window (Standard)
expanding_cfg = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential)
splits_exp = BayesianFootball.Data.create_data_splits(ds, expanding_cfg)
features_exp = BayesianFootball.Features.create_features(splits_exp, vocabulary, model, expanding_cfg)

# 2. Sliding Window (Last 5 rounds)
# This will iterate through 24/25, but at each step, 
# it only keeps rows where :split_col > (current_round - 5)
sliding_cfg = BayesianFootball.Data.WindowCV([], ["24/25"], :split_col, 10, :sequential)
splits_slide = BayesianFootball.Data.create_data_splits(ds, sliding_cfg)
features_slide = BayesianFootball.Features.create_features(splits_slide, vocabulary, model, sliding_cfg)


splits_exp[25]
splits_slide[25]

features_slide[25]
features_exp[25]


### train sliding Window
dd = BayesianFootball.load_scottish_data("24/25")
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(dd, model)


sliding_cfg = BayesianFootball.Data.WindowCV([], ["24/25"], :split_col, 10, :sequential)
splits_slide = BayesianFootball.Data.create_data_splits(dd, sliding_cfg)
features_slide = BayesianFootball.Features.create_features(splits_slide, vocabulary, model, sliding_cfg)




sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct

# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

# training_config_limited = TrainingConfig(sampler_conf, strategy_parallel_limited)
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# Then run:
results = BayesianFootball.Training.train(model, training_config_custom, features_slide)
# save and load 
JLD2.save_object("training_sliding_results_large.jld2", results)


###

split_col_name = :split_col

# 2. Get all unique split keys (e.g., [0, 1, 2, 3])
all_splits = sort(unique(ds.matches[!, split_col_name]))

# 3. Define the splits you want to *predict* (e.g., [1, 2, 3])
#    We skip the first key (0), as it was for training the first model
prediction_split_keys = all_splits[2:end] 

# 4. Group the data ONCE
grouped_matches = groupby(ds.matches, split_col_name)

# 5. Create the vector of DataFrames (as efficient SubDataFrame views)
#    This is the new argument for your function
dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]


oos_slide = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results
)

# get expanding results 
results_exp = JLD2.load_object("training_results_large.jld2")


data_store = BayesianFootball.Data.load_default_datastore()
vocabulary_exp = BayesianFootball.Features.create_vocabulary(data_store, model)

oos_exp = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,
    vocabulary_exp,
    results_exp
)



##############################
# ---  predict 
##############################
using Statistics, Distributions

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )



match_id = rand(keys(oos_exp))
subset( ds.matches, :match_id => ByRow(isequal(match_id)))

r_exp=  oos_exp[match_id]
r_slide=  oos_slide[match_id]



match_predict_exp = BayesianFootball.Predictions.predict_market(model, predict_config, r_exp...);
match_predict_slide = BayesianFootball.Predictions.predict_market(model, predict_config, r_slide...);


model_odds_exp = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_exp));
model_odds_slide = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_slide));
model_odds_exp
model_odds_slide

BayesianFootball.Predictions.get_market(match_id, predict_config, data_store.odds)


using StatsPlots

sym = :away
density(match_predict_exp[sym], label="exp")
density!(match_predict_slide[sym], label="slide")

sym = :over_25
density(match_predict_exp[sym], label="exp")
density!(match_predict_slide[sym], label="slide")


