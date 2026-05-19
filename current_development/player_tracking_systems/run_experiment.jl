# current_development/player_tracking_systems/run_experiment.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Statistics


# Include source files
#
include("src/types.jl")
include("src/preprocessing.jl")
include("src/trackers/last_value.jl")
include("src/trackers/window_average.jl")
include("src/trackers/ewma.jl")
include("src/trackers/bayesian.jl")
include("src/experiment_engine.jl")
include("src/results_handler.jl")

println("--- Player Tracking System Experiment ---")

# 1. Setup Data and CV Boundaries
# Using a subset for faster prototyping (e.g. Scottish Premier League)
ds = Data.load_datastore_sql(Data.Ireland()) 

cv_config = Data.GroupedCVConfig(
    # Pass a list of lists. Each inner list is processed as a single group.
    # To combine leagues 56 and 57, use: tournament_groups = [[56, 57]]
    tournament_groups = [Data.tournament_ids(ds.segment)], 
    target_seasons = ["2023","2024","2025","2026"],  # We want to test on the 25/26 season
    history_seasons = 2,        # Use 24/25 as history
    dynamics_col = :match_biweek,# Step forward month-by-month
    warmup_period = 0, 
    stop_early = false
)
boundaries = Data.create_id_boundaries(ds, cv_config);
println("[INFO] Created $(length(boundaries)) CV boundaries.")

#=
julia> boundaries = Data.create_id_boundaries(ds, cv_config)
8-element Vector{Tuple{SplitBoundary, GroupedSplitMetaData}}:
 (SplitBoundary(1, 0, [11907472, 11907470, 11907469, 11907471, 11907468, 11907474, 11907475, 11907476, 11907473, 11907477  …  13242864, 13242865, 13242859, 13242861, 14773611, 13242835, 13242877, 13242866, 13242850, 13242851], Int64[]), GroupedSplit(Tourns: [79], Season: 2026, Week: 0, Hist: 2))
 (SplitBoundary(2, 1, [11907472, 11907470, 11907469, 11907471, 11907468, 11907474, 11907475, 11907476, 11907473, 11907477  …  13242864, 13242865, 13242859, 13242861, 14773611, 13242835, 13242877, 13242866, 13242850, 13242851], [15238009, 15238008, 15238007, 15238011, 15238058, 15238012, 15238016]), GroupedSplit(Tourns: [79], Season: 2026, Week: 1, Hist: 2))
 (SplitBoundary(3, 2, [11907472, 11907470, 11907469, 11907471, 11907468, 11907474, 11907475, 11907476, 11907473, 11907477  …  13242864, 13242865, 13242859, 13242861, 14773611, 13242835, 13242877, 13242866, 13242850, 13242851], [15238009, 15238008, 15238007, 15238011, 15238058, 15238012, 15238016, 15238019, 15238020,
=#


#=
julia> println("[INFO] Created $(length(boundaries)) CV boundaries.")
[INFO] Created 8 CV boundaries.
=#



# 2. Define Experiment Grid
configs = AbstractRatingTracker[]

# Baseline: Last Value
push!(configs, LastValueTracker())

# Window Averages
for w in [5, 10, 20]
    push!(configs, WindowAverageTracker(w))
end

# EWMA
for a in [0.05, 0.1, 0.2, 0.3]
    push!(configs, EWMATracker(a))
end

# Bayesian
# assuming ratings are roughly around 6-7 with some variance
for noise in [0.01, 0.05, 0.1]
    push!(configs, BayesianTracker(6.5, 1.0, 0.5, noise))
end

println("[INFO] Running experiment grid with $(length(configs)) configurations...")

# 3. Run Experiment
results = run_experiment_grid(configs, ds, boundaries)

# 4. Process and Display Results
df_results = compile_results(results)

println("\n--- Top 5 Results (Sorted by LogLoss) ---")
display(first(df_results, 5))

# Optional: Save results
# CSV.write("player_tracking_results.csv", df_results)
#
#
#


#=
julia> show(df_results; truncate = 0)
11×5 DataFrame
 Row │ tracker_type          edge_pvalue  log_loss  parameters                                                      edge_coef 
     │ String                Float64      Float64   String                                                          Float64   
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ BayesianTracker        1.80993e-6  0.657599  prior_mean=6.5, prior_var=1.0, obs_var=0.5, process_noise=0.01   0.332952
   2 │ BayesianTracker        5.46335e-6  0.662139  prior_mean=6.5, prior_var=1.0, obs_var=0.5, process_noise=0.05   0.28244
   3 │ BayesianTracker        1.06783e-5  0.664155  prior_mean=6.5, prior_var=1.0, obs_var=0.5, process_noise=0.1    0.255877
   4 │ WindowAverageTracker   5.23264e-6  0.665024  window_size=20, agg_func=mean                                    0.341149
   5 │ EWMATracker            5.56435e-6  0.666876  alpha=0.05                                                       0.341102
   6 │ EWMATracker            4.9119e-6   0.667286  alpha=0.1                                                        0.33773
   7 │ EWMATracker            7.94263e-6  0.668201  alpha=0.2                                                        0.30436
   8 │ EWMATracker            1.5769e-5   0.669801  alpha=0.3                                                        0.270591
   9 │ WindowAverageTracker   1.51535e-5  0.671564  window_size=10, agg_func=mean                                    0.313932
  10 │ WindowAverageTracker   3.78131e-5  0.671676  window_size=5, agg_func=mean                                     0.250873
  11 │ LastValueTracker       0.00159966  0.675488                                                                   0.120819
=#


# ==========================================
# BAYESIAN GRID GENERATION
# ==========================================
println("[INFO] Generating Bayesian Hyperparameter Grid...")

# Drop the square brackets entirely
prior_means    = 6.1:0.1:6.8              # 8 values
prior_vars     = 0.5:0.1:2.0              # 16 values
obs_vars       = 0.1:0.1:2.0              # 20 values
process_noises = 0.01:0.05:0.5            # 10 values

bayesian_grid = AbstractRatingTracker[
    BayesianTracker(pm, pv, ov, pn)
    for pm in prior_means
    for pv in prior_vars
    for ov in obs_vars
    for pn in process_noises
];

# Create the full Cartesian product (2 * 3 * 4 * 4 = 96 configurations)
bayesian_grid = AbstractRatingTracker[
    BayesianTracker(pm, pv, ov, pn)
    for pm in prior_means
    for pv in prior_vars
    for ov in obs_vars
    for pn in process_noises
];

println("[INFO] Generated $(length(bayesian_grid)) Bayesian configurations.")

# Combine with your naive baseline to ensure the Bayesian models are actually winning
experiment_configs = vcat(
    [LastValueTracker()], 
    bayesian_grid
);

# Pass this directly into your threaded engine!
results_df = run_experiment_grid(experiment_configs, ds, boundaries)
df_results_Bayesian_grid = compile_results(results_df)


show(sort(df_results_Bayesian_grid, :log_loss); truncate = 0)

#=
julia> show(sort(df_results_Bayesian_grid, :log_loss); truncate = 0)                                                                                                                                                                         
97×5 DataFrame                                                                                                                                                                                                                               
 Row │ tracker_type      edge_pvalue  log_loss  parameters                                                       edge_coef 
     │ String            Float64      Float64   String                                                           Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ BayesianTracker   1.00202e-6   0.649764  prior_mean=6.5, prior_var=0.5, obs_var=1.0, process_noise=0.01    0.360679
   2 │ BayesianTracker   1.15839e-6   0.653002  prior_mean=6.5, prior_var=1.0, obs_var=1.0, process_noise=0.01    0.353534
   3 │ BayesianTracker   1.49767e-6   0.655101  prior_mean=6.5, prior_var=0.5, obs_var=0.5, process_noise=0.01    0.337947
   4 │ BayesianTracker   2.30945e-6   0.655643  prior_mean=6.5, prior_var=0.5, obs_var=1.0, process_noise=0.05    0.3129  
   5 │ BayesianTracker   1.3922e-6    0.655733  prior_mean=6.5, prior_var=2.0, obs_var=1.0, process_noise=0.01    0.346987
   6 │ BayesianTracker   2.78498e-6   0.657012  prior_mean=6.8, prior_var=0.5, obs_var=1.0, process_noise=0.01    0.36165 
   7 │ BayesianTracker   1.80993e-6   0.657599  prior_mean=6.5, prior_var=1.0, obs_var=0.5, process_noise=0.01    0.332952
   8 │ BayesianTracker   2.6391e-6    0.658054  prior_mean=6.5, prior_var=1.0, obs_var=1.0, process_noise=0.05    0.310058
   9 │ BayesianTracker   4.48061e-6   0.658605  prior_mean=6.5, prior_var=0.5, obs_var=1.0, process_noise=0.1     0.28602
  10 │ BayesianTracker   2.14776e-6   0.65937   prior_mean=6.5, prior_var=2.0, obs_var=0.5, process_noise=0.01    0.329004
  11 │ BayesianTracker   3.03372e-6   0.659439  prior_mean=6.8, prior_var=1.0, obs_var=1.0, process_noise=0.01    0.353105
  12 │ BayesianTracker   2.63953e-6   0.659519  prior_mean=6.5, prior_var=0.5, obs_var=0.25, process_noise=0.01   0.314063
  13 │ BayesianTracker   3.07691e-6   0.66015   prior_mean=6.5, prior_var=2.0, obs_var=1.0, process_noise=0.05    0.306965
  14 │ BayesianTracker   4.89617e-6   0.660453  prior_mean=6.5, prior_var=0.5, obs_var=0.5, process_noise=0.05    0.284371
  15 │ BayesianTracker   4.89617e-6   0.660453  prior_mean=6.5, prior_var=1.0, obs_var=1.0, process_noise=0.1     0.284371
  16 │ BayesianTracker   3.05884e-6   0.661106  prior_mean=6.5, prior_var=1.0, obs_var=0.25, process_noise=0.01   0.311121
  17 │ BayesianTracker   3.75392e-6   0.661146  prior_mean=6.8, prior_var=0.5, obs_var=0.5, process_noise=0.01    0.337499
  18 │ BayesianTracker   3.45449e-6   0.661512  prior_mean=6.8, prior_var=2.0, obs_var=1.0, process_noise=0.01    0.345581
  19 │ BayesianTracker   9.42187e-6   0.661746  prior_mean=6.5, prior_var=0.5, obs_var=1.0, process_noise=0.2     0.257801
  20 │ BayesianTracker   5.27387e-6   0.661795  prior_mean=6.8, prior_var=0.5, obs_var=1.0, process_noise=0.05    0.312784
  21 │ BayesianTracker   3.39812e-6   0.662093  prior_mean=6.5, prior_var=2.0, obs_var=0.25, process_noise=0.01   0.309083
  22 │ BayesianTracker   5.46335e-6   0.662139  prior_mean=6.5, prior_var=1.0, obs_var=0.5, process_noise=0.05    0.28244 
  23 │ BayesianTracker   5.46335e-6   0.662139  prior_mean=6.5, prior_var=2.0, obs_var=1.0, process_noise=0.1     0.28244 
  24 │ BayesianTracker   9.94445e-6   0.662958  prior_mean=6.5, prior_var=0.5, obs_var=0.5, process_noise=0.1     0.25695
  ⋮  │        ⋮               ⋮          ⋮                                     ⋮                                     ⋮                                                                                                                       
  74 │ BayesianTracker   5.49216e-5   0.669523  prior_mean=6.5, prior_var=2.0, obs_var=0.1, process_noise=0.1     0.196061
  75 │ BayesianTracker   3.27056e-5   0.669554  prior_mean=6.8, prior_var=0.5, obs_var=0.5, process_noise=0.2     0.227977
  76 │ BayesianTracker   2.00888e-5   0.669918  prior_mean=6.8, prior_var=2.0, obs_var=0.25, process_noise=0.05   0.252002
  77 │ BayesianTracker   3.40425e-5   0.670199  prior_mean=6.8, prior_var=0.5, obs_var=0.25, process_noise=0.1    0.227381
  78 │ BayesianTracker   3.40425e-5   0.670199  prior_mean=6.8, prior_var=1.0, obs_var=0.5, process_noise=0.2     0.227381
  79 │ BayesianTracker   9.75959e-5   0.670439  prior_mean=6.5, prior_var=0.5, obs_var=0.1, process_noise=0.2     0.175105
  80 │ BayesianTracker   9.93917e-5   0.670571  prior_mean=6.5, prior_var=1.0, obs_var=0.1, process_noise=0.2     0.175006
  81 │ BayesianTracker   0.000100747  0.670669  prior_mean=6.5, prior_var=2.0, obs_var=0.1, process_noise=0.2     0.174926
  82 │ BayesianTracker   3.54177e-5   0.670771  prior_mean=6.8, prior_var=1.0, obs_var=0.25, process_noise=0.1    0.226768
  83 │ BayesianTracker   3.54177e-5   0.670771  prior_mean=6.8, prior_var=2.0, obs_var=0.5, process_noise=0.2     0.226768
  84 │ BayesianTracker   3.65238e-5   0.671185  prior_mean=6.8, prior_var=2.0, obs_var=0.25, process_noise=0.1    0.226251
  85 │ BayesianTracker   4.36697e-5   0.671355  prior_mean=6.8, prior_var=0.5, obs_var=0.1, process_noise=0.05    0.218428
  86 │ BayesianTracker   4.47817e-5   0.671681  prior_mean=6.8, prior_var=1.0, obs_var=0.1, process_noise=0.05    0.21804
  87 │ BayesianTracker   6.3963e-5    0.67176   prior_mean=6.8, prior_var=0.5, obs_var=0.25, process_noise=0.2    0.20222
  88 │ BayesianTracker   4.54802e-5   0.671878  prior_mean=6.8, prior_var=2.0, obs_var=0.1, process_noise=0.05    0.21777
  89 │ BayesianTracker   6.5872e-5    0.672133  prior_mean=6.8, prior_var=1.0, obs_var=0.25, process_noise=0.2    0.201932
  90 │ BayesianTracker   6.74767e-5   0.672425  prior_mean=6.8, prior_var=2.0, obs_var=0.25, process_noise=0.2    0.201668
  91 │ BayesianTracker   8.10408e-5   0.672629  prior_mean=6.8, prior_var=0.5, obs_var=0.1, process_noise=0.1     0.194335
  92 │ BayesianTracker   8.26813e-5   0.672859  prior_mean=6.8, prior_var=1.0, obs_var=0.1, process_noise=0.1     0.194139
  93 │ BayesianTracker   8.37654e-5   0.673007  prior_mean=6.8, prior_var=2.0, obs_var=0.1, process_noise=0.1     0.19399
  94 │ BayesianTracker   0.000147261  0.673638  prior_mean=6.8, prior_var=0.5, obs_var=0.1, process_noise=0.2     0.17309
  95 │ BayesianTracker   0.0001495    0.673792  prior_mean=6.8, prior_var=1.0, obs_var=0.1, process_noise=0.2     0.173001
  96 │ BayesianTracker   0.000151096  0.673899  prior_mean=6.8, prior_var=2.0, obs_var=0.1, process_noise=0.2     0.172928
  97 │ LastValueTracker  0.00159966   0.675488                                                                    0.120819
=#


