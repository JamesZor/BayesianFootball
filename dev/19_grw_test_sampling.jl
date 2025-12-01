using Revise
using BayesianFootball
using DataFrames
using Statistics


# --- HPC OPTIMIZATION START ---
using ThreadPinning
using LinearAlgebra

# 1. Pin Julia threads to physical cores (8 cores on your 5800X)
# This prevents threads from jumping between cores and thrashing the cache.
pinthreads(:cores)

# 2. Force BLAS (Matrix Math) to be single-threaded
# Since NUTS runs 4 chains in parallel, we don't want each chain 
# trying to spawn 8 more threads for matrix multiplication.
BLAS.set_num_threads(1) 

# Verify the setup (Optional, prints to console)
println("Threads pinned to: ", threadinfo())
println("BLAS Threads: ", BLAS.get_num_threads())

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.load_scottish_data("24/25", split_week=14)


"""
Here limit to one league to improve the sampling speed and reduce the compleity of the data 
"""

dss = BayesianFootball.Data.DataStore( 
    subset( ds.matches, :tournament_id => ByRow(isequal(54)) ),
    ds.odds,
    ds.incidents
)
  


"""
    compress_time_col!(df, split_week; col=:split_col)

Modifies the dataframe in-place.
- Matches with match_week <= split_week are mapped to index 1.
- Matches with match_week > split_week are mapped to match_week - split_week + 1.
"""
function compress_time_col!(df::DataFrame, split_week::Int; col::Symbol=:split_col)
    # Ensure the column exists and is Int
    if !hasproperty(df, col)
        df[!, col] = zeros(Int, nrow(df))
    end

    # Logic:
    # If week 14 is the split:
    # Week 1-14 -> 1 (The Baseline)
    # Week 15   -> 2 (The first Step)
    # Week 16   -> 3
    for row in eachrow(df)
        if row.match_week <= split_week
            row[col] = 1
        else
            row[col] = row.match_week - split_week + 1
        end
    end
    return df
end

compress_time_col!(dss.matches, 14, col=:split_col)


"""
Check the group of the split cols 
""" 
combine( 
  groupby(dss.matches, :split_col), 
  nrow => :n_matchs
)

combine( 
  groupby(dss.matches, :match_week), 
  nrow => :n_matchs
)


subset( dss.matches, :split_col => ByRow(isequal(17)))[:, [:home_team, :away_team, :match_date, :split_col, :match_week, :match_dayofweek]]


model= BayesianFootball.Models.PreGame.GRWPoisson()

vocabulary = BayesianFootball.Features.create_vocabulary(dss, model)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :split_col
)

data_splits = BayesianFootball.Data.create_data_splits(dss, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 


sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=1000, n_chains=4, n_warmup=1000) # Use renamed struct


training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)


results = BayesianFootball.Training.train(model, training_config, feature_sets)


""" 
chain diag 
"""

using MCMCChains, Plots, StatsPlots

r = results[1][1]

hyperparams = ["σ_att", "σ_def", "home_adv"]

plot(r[hyperparams])
meanplot(r[hyperparams])
autocorplot(r[hyperparams])
### 
# 


all_names = string.(names(r))

# 2. Filter for a specific team (e.g., Team ID 1) at specific time steps
# Adjust '1' to a valid team ID and '10', '20' to valid round numbers in your data
target_vars = filter(n -> n in ["att_hist[1, 1]", "att_hist[1, 10]", "att_hist[1, 20]"], all_names)

# 3. Plot only these specific latent variables
plot(r[target_vars])

""" 
julia> vocabulary.mappings[:team_map]
Dict{InlineStrings.String31, Int64} with 12 entries:
  "motherwell"          => 2
  "st-johnstone"        => 6
  "hibernian"           => 10
  "dundee-united"       => 3
  "st-mirren"           => 5
  "heart-of-midlothian" => 1
  "dundee-fc"           => 7
  "rangers"             => 9
  "celtic"              => 4
  "aberdeen"            => 12
  "ross-county"         => 8
  "kilmarnock"          => 11

"""

team_id = 6
team_attact_list_str = [ "att_hist[$team_id, $round]" for round in 1:5:20] 
team_def_list_str = [ "att_hist[$team_id, $round]" for round in 1:5:20] 
target_vars_att = filter( n -> n in team_attact_list_str, all_names)
target_vars_def = filter( n -> n in team_def_list_str, all_names)

plot(r[target_vars_att])
meanplot(r[target_vars_att])
autocorplot(r[target_vars_att])


println("\n")
plot(r[target_vars_def])
autocorplot(r[target_vars_def])



team_id = 6
team_attact_list_str = [ "att_hist[$team_id, $round]" for round in 1:5:20] 
team_def_list_str = [ "att_hist[$team_id, $round]" for round in 1:20] 
target_vars_att = filter( n -> n in team_attact_list_str, all_names)
target_vars_def = filter( n -> n in team_def_list_str, all_names)

forestplot(r[target_vars_att], hpd_val = [0.05, 0.15, 0.25], ordered = true)




# Convert your existing string vector to symbols
hyperparams_sym = Symbol.(hyperparams)
# Now run the plot
ridgelineplot(r, hyperparams_sym)
forestplot(r, hyperparams_sym)


team_id = 6
team_def_list_str = [ "att_hist[$team_id,$round]" for round in 1:20] 
target_vars_att = filter( n -> n in sym, all_names)
target_vars_def = filter( n -> n in team_def_list_str, all_names)

team_attact_sym = Symbol.([ "att_hist[$team_id, $round]" for round in 1:20])
forestplot(r, team_attact_sym)


###

stats_df = DataFrame(summarystats(r))

# 2. Find parameters that haven't converged (Rhat > 1.05)
# Rhat should be close to 1.0. Values > 1.05 indicate the chain is stuck or not mixing.
bad_rhat = filter(row -> row.rhat > 1.02, stats_df)

println("--- Parameters with Poor Convergence (Rhat > 1.05) ---")
if isempty(bad_rhat)
    println("All parameters have converged (Rhat <= 1.05)!")
else
    # Sort by worst Rhat first
    sort!(bad_rhat, :rhat, rev=true)
    display(bad_rhat[:, [:parameters, :rhat, :ess_bulk]])
end


low_ess_threshold = 100  # Adjust based on your needs
low_ess = filter(row -> row.ess_bulk < low_ess_threshold, stats_df)

println("\n--- Parameters with Low ESS (< $low_ess_threshold) ---")
if isempty(low_ess)
    println("All parameters have sufficient ESS!")
else
    sort!(low_ess, :ess_bulk)
    display(low_ess[:, [:parameters, :ess_bulk, :rhat]])
end





"""

dealing with the results
"""

r = results[1][1]

unique(dss.matches.split_col)


mp = filter( row -> row.split_col == 23 , dss.matches)

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, r)


match_id = rand(keys(rr))
subset(ds.matches, :match_id => ByRow(isequal(match_id)))

r1 =  rr[match_id]
match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

open, close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds )



using StatsPlots
sym = :under_05
density( match_predict[sym], label="grw")
density!( match_predict_pos[sym], label="poisson")




####
results_pos = JLD2.load_object("training_results_large.jld2")

# 

data_store = BayesianFootball.Data.load_default_datastore()
model_pos = BayesianFootball.Models.PreGame.StaticPoisson()

vocabulary_l = BayesianFootball.Features.create_vocabulary(data_store, model)

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
    vocabulary_l,
    results_pos
)


match_predict_pos = BayesianFootball.Predictions.predict_market(model_pos, predict_config, all_oos_results[match_id]...);


model_odds_pos = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_pos));
model_odds_pos


#### dev.17 for help functions 

match_id = rand(keys(rr))
r1 =  rr[match_id]
open, close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds )



match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds


subset( ds.matches, :match_id => ByRow(isequal(match_id)))

# a = subset( ds.odds, :match_idhome => ByRow(isequal(match_id)))
# a[:, [:choice_name, :choice_group, :winning, :decimal_odds, :initial_decimal]]
#

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
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no]
)




