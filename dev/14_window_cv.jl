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
