# current_development/player_tracking_systems/run_experiment.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Statistics

# Include source files
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
ds = Data.load_datastore_sql(Data.ScottishLower()) 

cv_config = Data.GroupedCVConfig(
    group_col = :season,
    min_train_groups = 2,
    test_groups_size = 1
)

boundaries = Data.create_id_boundaries(ds, cv_config)
println("[INFO] Created $(length(boundaries)) CV boundaries.")

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
