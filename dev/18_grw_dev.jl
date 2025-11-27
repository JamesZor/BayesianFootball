using Revise
using BayesianFootball
using DataFrames
using Statistics

data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticDixonColes()



ds = BayesianFootball.load_scottish_data("24/25", split_week=0)

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

v2 = BayesianFootball.Features.create_vocabulary(data_store, model)

splitter_config = BayesianFootball.Data.StaticSplit(train_seasons =["24/25"]) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

    # splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    # data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    # feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #



# feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


model_grw = BayesianFootball.Models.PreGame.GRWPoisson()
fs_grw = BayesianFootball.Features.create_features(data_splits, vocabulary, model_grw, splitter_config) #

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=50, n_chains=4, n_warmup=10) # Use renamed struct
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

training_config  = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model_grw, training_config, fs_grw)

r = results[1][1]


mp = filter( row -> row.split_col == 25, ds.matches)

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model_grw, mp, vocabulary, r)


match_id = rand(keys(rr))
r1 =  rr[match_id]

model = BayesianFootball.Models.PreGame.StaticPoisson()

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds


open, close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds )


subset( ds.matches, :match_id => ByRow(isequal(match_id)))

sym = :away
density( match_predict[sym], label="dixon")

using JLD2
results_pos = JLD2.load_object("training_results_large.jld2")


model_pos = BayesianFootball.Models.PreGame.StaticPoisson()

ds.matches.split_col = max.(0, ds.matches.match_week .- 14);
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
    v2,
    results_pos
)


match_predict_pos = BayesianFootball.Predictions.predict_market(model_pos, predict_config, all_oos_results[match_id]...);


model_odds_pos = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_pos));
model_odds_pos




kelly_grw_res   = BayesianFootball.Signals.bayesian_kelly(match_predict, open)
kelly_poisson_res = BayesianFootball.Signals.bayesian_kelly(match_predict_pos, open)

# using funcitons from dev.17
compare_all_markets(
    match_id, 
    match_predict, 
    match_predict_pos, 
    open, 
    close, 
    outcome, 
    kelly_grw_res, 
    kelly_poisson_res;
    markets=[:home, :draw, :away, :over_25, :under_25, :btts_yes, :btts_no]
)




###


mp = filter( row -> row.split_col == 25, ds.matches)

rr = BayesianFootball.Models.PreGame.extract_parameters(model_grw, mp, vocabulary, r)

match_id = rand(keys(rr))
r1 =  rr[match_id]

open, close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds )




match_predict = BayesianFootball.Predictions.predict_market(model_grw, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds


subset( ds.matches, :match_id => ByRow(isequal(match_id)))
# subset( ds.odds, :match_id => ByRow(isequal(match_id)))


match_predict_pos = BayesianFootball.Predictions.predict_market(model_pos, predict_config, all_oos_results[match_id]...);


model_odds_pos = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_pos));
model_odds_pos




kelly_grw_res   = BayesianFootball.Signals.bayesian_kelly(match_predict, open)
kelly_poisson_res = BayesianFootball.Signals.bayesian_kelly(match_predict_pos, open)


compare_all_markets(
    match_id, 
    match_predict, 
    match_predict_pos, 
    open, 
    close, 
    outcome, 
    kelly_grw_res, 
    kelly_poisson_res;
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :btts_yes, :btts_no]
)




### extract trends 

using Plots

# 1. Extract the data
trends_df = BayesianFootball.Models.PreGame.extract_trends(model_grw, vocabulary, r)

# 2. Filter for a few specific teams (plotting 20 teams is messy)
teams_of_interest = ["celtic", "rangers", "aberdeen", "hearts"]
subset_df = filter(row -> row.team in teams_of_interest, trends_df)

# 3. Plot Attack Strength Over Time
plot(
    subset_df.round, 
    subset_df.att, 
    group = subset_df.team, 
    title = "Team Attack Strength (Gaussian Random Walk)",
    xlabel = "Round / Matchweek",
    ylabel = "Attack Rating (Log Scale)",
    lw = 2,           # Line width
    legend = :outertopright
)
