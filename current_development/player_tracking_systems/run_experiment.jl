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
    target_seasons = ["2026"],  # We want to test on the 25/26 season
    history_seasons = 2,        # Use 24/25 as history
    dynamics_col = :match_biweek,# Step forward month-by-month
    warmup_period = 0, 
    stop_early = false
)
boundaries = Data.create_id_boundaries(ds, cv_config)
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
