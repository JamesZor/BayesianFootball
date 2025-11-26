using Revise
using BayesianFootball
using DataFrames
using Statistics

data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticDixonColes()



ds = BayesianFootball.load_scottish_data("24/25", split_week=0)

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)



splitter_config = BayesianFootball.Data.StaticSplit(train_seasons =["24/25"]) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

    # splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    # data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    # feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #



feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


model_grw = BayesianFootball.Models.PreGame.GRWPoisson()
fs_grw = BayesianFootball.Features.create_features(data_splits, vocabulary, model_grw, splitter_config) #

