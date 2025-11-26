using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics


## 
model = BayesianFootball.Models.PreGame.StaticDixonColes()

data_store = BayesianFootball.Data.load_default_datastore()

vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

splitter_config = BayesianFootball.Data.StaticSplit(train_seasons =["24/25"]) #
data_splits = BayesianFootball.Data.create_data_splits(data_store, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #



sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=2, n_warmup=100) # Use renamed struct

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

training_config  = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, feature_sets)

####
# filter for one season for quick training
df = filter(row -> row.season=="24/25", data_store.matches)
# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)

# 1. Define your "special cases" mapping
split_map = Dict(37 => 1, 38 => 2, 39 => 3)

# 2. Use get() with a default value of 0
#    We use Ref(split_map) to tell Julia to treat the Dict as a single object
#    and not try to broadcast over its elements.
ds.matches.split_col = get.(Ref(split_map), ds.matches.match_week, 0);

# large v2 
ds.matches.split_col = max.(0, ds.matches.match_week .- 14);



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


# --- 6. Call your new function ---
# It's now much cleaner and more flexible
is_dixon = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results
)

###
r = results[1][1]

mp = filter( row -> row.split_col == 1, ds.matches)

r[Symbol("log_α[18]")]
names(r)

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )
rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, r)

match_id = rand(keys(rr))

r1 =  rr[match_id]



match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

close = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)

subset( ds.matches, :match_id => ByRow(isequal(match_id)))

using StatsPlots
density(match_predict[:home])



results_pos = JLD2.load_object("training_results_large.jld2")

model_pos = BayesianFootball.Models.PreGame.StaticPoisson()



split_col_name = :split_col
all_splits = sort(unique(ds.matches[!, split_col_name]))
prediction_split_keys = all_splits[2:end] 
grouped_matches = groupby(ds.matches, split_col_name)
dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]


# --- 6. Call your new function ---
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model_pos,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results_pos
)


BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)




match_id = rand(keys(rr))

subset( ds.matches, :match_id => ByRow(isequal(match_id)))

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, rr[match_id]...);
match_predict_pos = BayesianFootball.Predictions.predict_market(model_pos, predict_config, all_oos_results[match_id]...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds


model_odds_pos = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_pos));
model_odds_pos


open,close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)


open

subset( ds.matches, :match_id => ByRow(isequal(match_id)))

sym = :away
density( match_predict[sym], label="dixon")
density!( match_predict_pos[sym], label="poisson")
