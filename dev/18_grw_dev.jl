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
model = BayesianFootball.Models.PreGame.StaticDixonColes()



ds = BayesianFootball.load_scottish_data("24/25", split_week=0)

using Dates
ds.matches.match_month = [month(d) >= 8 ? month(d) - 7 : month(d) + 5 for d in ds.matches.match_date] ;


vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

vocabulary_l = BayesianFootball.Features.create_vocabulary(data_store, model)


splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :match_week  # <--- THIS IS THE MISSING KEY
)

# splitter_config = BayesianFootball.Data.StaticSplit(train_seasons =["24/25"]) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

    # splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    # data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    # feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

# feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


model_grw = BayesianFootball.Models.PreGame.GRWPoisson()
fs_grw = BayesianFootball.Features.create_features(data_splits, vocabulary, model_grw, splitter_config) #


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
# nuts
# sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=4, n_warmup=50) # Use renamed struct

# SGLD



sampler_conf = BayesianFootball.Samplers.SGLDConfig(
    step_size = 0.005,      # Tuning knob (lower if it explodes)
    n_samples = 20000,      # SGLD needs more samples than NUTS
    n_chains  = 4,          # Run 4 parallel simulations
    ad_backend = :reversediff # We found this was 4ms (super fast)
)

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)
results = BayesianFootball.Training.train(model_grw, training_config, fs_grw)

### check 
using BenchmarkTools
using DynamicPPL
using Turing
using ReverseDiff
using Zygote
using LogDensityProblems

# 1. Instantiate the model (You already did this successfully)
turing_instance = BayesianFootball.Models.PreGame.build_turing_model(model_grw, fs_grw[1][1])

# 2. Extract the current parameter values
# FIX: Use DynamicPPL explicitly
vi = DynamicPPL.VarInfo(turing_instance)
initial_theta = vi[DynamicPPL.SampleFromPrior()]

ldf = Turing.LogDensityFunction(turing_instance)
dim = LogDensityProblems.dimension(ldf)
initial_theta = randn(dim);

# 3. Define the "Log Density" function
log_density_function = Turing.LogDensityFunction(turing_instance)

# 4. Benchmark
println("--- Benchmarking Gradient Calculation ---")
println("Parameter Count: $(length(initial_theta))")

# Test A: ReverseDiff (Compiled)
println("\nTesting ReverseDiff (Compiled)...")
try
    # Compile the tape
    tape = ReverseDiff.GradientTape(q -> Turing.LogDensityProblems.logdensity(log_density_function, q), initial_theta)
    compiled_tape = ReverseDiff.compile(tape)
    
    # Benchmark
    # We allocate the result buffer once to be fair
    result_buffer = similar(initial_theta)
    b_rev = @benchmark ReverseDiff.gradient!($result_buffer, $compiled_tape, $initial_theta)
    display(b_rev)
catch e
    println("ReverseDiff Failed: ", e)
end

# Test B: Zygote
println("\nTesting Zygote...")
try
    b_zyg = @benchmark Zygote.gradient(q -> Turing.LogDensityProblems.logdensity($log_density_function, q), $initial_theta)
    display(b_zyg)
catch e
    println("Zygote Failed: ", e)
end



####


# results = BayesianFootball.Training.train(model_grw, training_config, fs_grw)


using JLD2

save_dir = "dev_exp/grw_poisson/"
save_file = save_dir * "s_24_25_week.jld2"
JLD2.save_object(save_file, results)

# done
save_file = save_dir * "s_24_25_month.jld2"

results = JLD2.load_object(save_file)

r = results[1][1]


mp = filter( row -> row.split_col == 25, ds.matches)

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model_grw, mp, vocabulary, r)


match_id = rand(keys(rr))
r1 =  rr[match_id]

model = BayesianFootball.Models.PreGame.StaticPoisson()

match_predict = BayesianFootball.Predictions.predict_market(model_grw, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds


open, close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds )


subset( ds.matches, :match_id => ByRow(isequal(match_id)))
a = subset( ds.odds, :match_id => ByRow(isequal(match_id)))
a[:, [:choice_name, :choice_group, :winning, :decimal_odds, :initial_decimal]]


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
    vocabulary_l,
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


mp = filter( row -> row.split_col == 24, ds.matches)

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
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no]
)



using StatsPlots
sym = :home
density( match_predict[sym], label="grw")
density!( match_predict_pos[sym], label="poisson")


### extract trends 

using Plots

# 1. Extract the data
trends_df = BayesianFootball.Models.PreGame.extract_trends(model_grw, vocabulary, r)

# 2. Filter for a few specific teams (plotting 20 teams is messy)
teams_of_interest = [ "airdrieonians", "livingston", "hamilton-academical", "falkirk-fc"]


teams_of_interest = ["stranraer", "bonnyrigg-rose"]
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

plot!(
    subset_df.round, 
    subset_df.def, 
    group = subset_df.team, 
    title = "Team  deff Strength (Gaussian Random Walk)",
    xlabel = "Round / Matchweek",
    ylabel = "def Rating (Log Scale)",
    lw = 2,           # Line width
    legend = :outertopright
)


### 

r[Symbol("att_hist[12, 1]")]

