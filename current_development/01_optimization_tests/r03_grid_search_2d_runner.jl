# current_development/01_optimization_tests/r03_grid_search_2d_runner.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores)
using DataFrames

# Include the 2D grid search loaders
include("l03_grid_search_2d_loaders.jl")

# 1. Load Data
println("\n[1] Loading Ireland Data...")
ds = Data.load_datastore_cached(Data.Ireland())

# 2. Define 2D Parameter Grid
# Customize these arrays to change the granularity of the search
half_lives_grid = [30, 90.0, 180.0, 270.0, 300, 600]
market_weights_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

# 3. Run the 2D Grid Search (Defaults to MAP for speed)
summary_df, results_dict = run_grid_search_2d(
    ds, 
    half_lives_grid, 
    market_weights_grid; 
    use_map=true, 
    use_mle=false
)

# 4. Print Results Table (Sorted by best logloss by default)
println("\n============================================================")
println(" 🎉 2D Grid Search Completed!")
println("============================================================")
println(summary_df)

# 5. Identify Best Parameter Combination
best_row = summary_df[1, :]
println("\nOptimal Parameter Choice:")
println("  Best days_half_life: ", best_row.days_half_life)
println("  Best market_weight:  ", best_row.market_weight)
println("  Model LogLoss:       ", best_row.logloss_model)
println("  Observations:        ", best_row.n_obs)

println("\nNote: The full experiment results are preserved in the `results_dict` dictionary.")
println("For example, to inspect the parameters of the best run:")
println("  best_results = results_dict[($(best_row.days_half_life), $(best_row.market_weight))]")
println("  extract_chains(ds, best_results)")
