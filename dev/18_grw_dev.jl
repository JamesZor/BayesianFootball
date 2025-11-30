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



ds = BayesianFootball.load_scottish_data("24/25", split_week=14)

using Dates
ds.matches.match_month = [month(d) >= 8 ? month(d) - 7 : month(d) + 5 for d in ds.matches.match_date] ;


dss = BayesianFootball.Data.DataStore( 
    subset( ds.matches, :tournament_id => ByRow(isequal(54)) ),
    ds.odds,
    ds.incidents
)
  

# vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)
vocabulary = BayesianFootball.Features.create_vocabulary(dss, model)

vocabulary_l = BayesianFootball.Features.create_vocabulary(data_store, model)


splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :split_col  # <--- THIS IS THE MISSING KEY
)

# splitter_config = BayesianFootball.Data.StaticSplit(train_seasons =["24/25"]) #
data_splits = BayesianFootball.Data.create_data_splits(dss, splitter_config)

    # splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    # data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    # feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

# feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


model_grw = BayesianFootball.Models.PreGame.GRWPoisson()
fs_grw = BayesianFootball.Features.create_features(data_splits, vocabulary, model_grw, splitter_config) #


train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
# nuts
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=6, n_warmup=100) # Use renamed struct

# SGLD
# sampler_conf = BayesianFootball.Samplers.SGLDConfig(
#     step_size = 0.005,      # Tuning knob (lower if it explodes)
#     n_samples = 20000,      # SGLD needs more samples than NUTS
#     n_chains  = 4,          # Run 4 parallel simulations
#     ad_backend = :reversediff # We found this was 4ms (super fast)
# )
#
training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model_grw, training_config, fs_grw)

r = results[1][1]
#### chain  diag 

using MCMCChains, Plots, StatsPlots

# Select only the scalar hyperparameters
# These keys come from your @model definition
hyperparams = ["σ_att", "σ_def", "home_adv"]

# Plot trace and density for just these 3
plot(r[hyperparams])


#
# 1. Get all parameter names as strings
all_names = string.(names(r))

# 2. Filter for a specific team (e.g., Team ID 1) at specific time steps
# Adjust '1' to a valid team ID and '10', '20' to valid round numbers in your data
target_vars = filter(n -> n in ["att_hist[1, 1]", "att_hist[1, 10]", "att_hist[1, 20]"], all_names)

# 3. Plot only these specific latent variables
plot(r[target_vars])

#
# 1. Get the summary statistics directly as a DataFrame
# This replaces 'describe(r)[1]' which caused the error
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

# 3. Find parameters with low Effective Sample Size (ESS)
# Low ESS means the samples are highly correlated (less information than it seems).
low_ess_threshold = 100  # Adjust based on your needs
low_ess = filter(row -> row.ess_bulk < low_ess_threshold, stats_df)

println("\n--- Parameters with Low ESS (< $low_ess_threshold) ---")
if isempty(low_ess)
    println("All parameters have sufficient ESS!")
else
    sort!(low_ess, :ess_bulk)
    display(low_ess[:, [:parameters, :ess_bulk, :rhat]])
end

#

using Statistics, Plots


"""
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
# --- Settings ---
team_id = 7         # Change this to the ID of the team you want to check
n_rounds = 20       # Adjust to your actual number of rounds
# ----------------

# Vectors to hold the summary data
means = Float64[]
lower = Float64[]
upper = Float64[]
rounds = 1:n_rounds

for t in rounds
    # Construct the symbol for Team i at Time t
    # Note: Ensure this matches your model's internal naming (att_hist vs att)
    sym = Symbol("att_hist[$team_id, $t]")
    
    # Extract the full chain for this specific parameter
    s = r[sym]
    
    # Calculate Mean and 95% Credible Interval
    m = mean(s)
    q = quantile(vec(s), [0.2, 0.8])
    
    push!(means, m)
    push!(lower, m - q[1]) # Ribbon expects distance from mean, or absolute values depending on backend
    push!(upper, q[2] - m) # Storing distances for error bars is safer usually, but let's use simple ribbon logic
end

# Calculate ribbon width (distance from mean)
ribbon_lower = means .- (means .- lower) # re-calculating for clarity: value - lower_bound
ribbon_upper = (means .+ upper) .- means

# PLOT
plot(rounds, means, 
    ribbon = (ribbon_lower, ribbon_upper), 
    fillalpha = 0.3, 
    lw = 3, 
    label = "Team $team_id Attack",
    xlabel = "Round",
    ylabel = "Attack Strength (Centered)",
    title = "Evolution of Attack Strength (Gaussian Random Walk)",
    legend = :topleft
)



# --- Settings ---
team_id = 4      # "dundee-fc" based on your dict
n_rounds = 20     # Adjust to your max rounds
# ----------------

# 1. Flatten the data for plotting
# We need two long vectors: one for the x-coordinates (time) and one for y (strength)
all_times = Int[]
all_values = Float64[]

for t in 1:n_rounds
    # Construct the symbol for this time step
    sym = Symbol("att_hist[$team_id, $t]")
    
    # Extract the samples (flattened)
    s = vec(r[sym])
    
    # Append to our long vectors
    # If we have 1000 samples for round 1, we add 1000 '1's to all_times
    append!(all_times, fill(t, length(s)))
    append!(all_values, s)
end

# 2. Plot the Heatmap
# bins=(x_bins, y_bins): 
#   - x_bins=n_rounds ensures we have roughly one column per round
#   - y_bins=100 gives us high resolution for the strength density
histogram2d(
    all_times, 
    all_values, 
    bins = (n_rounds, 100), 
    
    # Formatting
  title = "Attack Strength Heatmap: celtic (finished 1)",
    xlabel = "Round",
    ylabel = "Attack Strength (Centered)",
    seriescolor = :viridis,  # Options: :plasma, :inferno, :turbo
    fill = true,             # Fills the bins
    
    # Optional: Smooth the visual if you have lots of samples
    # normalize = :pdf 
)

#
using Plots, Measures

# 1. Setup the comparison
# Replace these IDs with the actual ones from your `vocabulary.mappings`
winner_id = vocabulary.mappings[:team_map]["celtic"] 
loser_id  = vocabulary.mappings[:team_map]["st-johnstone"] # or "livingston" if they finished last

teams_to_plot = [
    (winner_id, "Celtic (Winner)"), 
    (loser_id,  "St. Johnstone (Relegated)")
]

plots_list = []

for (tid, tname) in teams_to_plot
    # --- Data Extraction ---
    times = Int[]
    values = Float64[]
    
    for t in 1:20 # Your n_rounds
        sym = Symbol("att_hist[$tid, $t]")
        s = vec(r[sym])
        append!(times, fill(t, length(s)))
        append!(values, s)
    end
    
    # --- Plotting ---
    p = histogram2d(
        times, values, 
        bins=(20, 100), 
        seriescolor=:viridis, 
        title=tname,
        xlabel="Round", 
        ylabel="Att Strength",
        ylims=(-1.0, 1.5), # Fixed Y-limits allow direct comparison!
        legend=false
    )
    # Add a horizontal line at 0 (Average)
    hline!(p, [0], color=:white, linestyle=:dash, label="League Avg")
    
    push!(plots_list, p)
end

# Display side-by-side
plot(plots_list..., layout=(1,2), size=(1000, 400), margin=5mm)


using Plots, StatsPlots, Distributions

# 1. Define the Prior you used in the model
prior_dist = Truncated(Normal(0, 0.05), 0, Inf)

# 2. Extract the Posterior (what the model learned)
posterior_att = vec(r["σ_att"])
posterior_def = vec(r["σ_def"])

# 3. Plot Comparison
density(posterior_att, label="Posterior (Att)", lw=3, fill=true, alpha=0.3)
density!(posterior_def, label="Posterior (Def)", lw=3, fill=true, alpha=0.3)

# Overlay the Prior
plot!(x -> pdf(prior_dist, x), 0, 0.15, 
      label="Prior (Your Assumption)", 
      linestyle=:dash, color=:black, lw=2,
      title="Did the model learn the Variance?",
      xlabel="Step Size (Sigma)", ylabel="Density")

# Get the mean sigma from your chain
mean_sigma = mean(r["σ_att"])

# Calculate the expected weekly swing in performance
# exp(sigma) - 1  = % change in goal expectancy
pct_change = (exp(mean_sigma) - 1) * 100

println("Average Sigma: ", round(mean_sigma, digits=4))
println("Typical weekly form swing: ±", round(pct_change, digits=2), "%")



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


mp = filter( row -> row.split_col == 22 , dss.matches)

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

rr = BayesianFootball.Models.PreGame.extract_parameters(model_grw, mp, vocabulary, r)


match_id = rand(keys(rr))

match_id = keys(rr)[1]
r1 =  rr[match_id]


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


mp = filter( row -> row.split_col == 21, dss.matches)

rr = BayesianFootball.Models.PreGame.extract_parameters(model_grw, mp, vocabulary, r)

match_id = rand(keys(rr))

match_id = collect(keys(rr))[2]


r1 =  rr[match_id]
open, close, outcome = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds )



match_predict = BayesianFootball.Predictions.predict_market(model_grw, predict_config, r1...);


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



using StatsPlots
sym = :under_25
density( match_predict[sym], label="grw")
density!( match_predict_pos[sym], label="poisson")


### extract trends 

using Plots

# 1. Extract the data
trends_df = BayesianFootball.Models.PreGame.extract_trends(model_grw, vocabulary, r)

# 2. Filter for a few specific teams (plotting 20 teams is messy)
teams_of_interest = [ "airdrieonians", "livingston", "hamilton-academical", "falkirk-fc"]

teams_of_interest = unique(trends_df.team)


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

plot(
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

