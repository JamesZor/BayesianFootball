# current_development/player_tracking_systems/run_optimization.jl

# ==============================================================================
# REMINDER: Start Julia with multiple threads to utilize the parallel pipeline.
# Example: `julia -t auto --project` or `julia -t 16 --project`
# ==============================================================================

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Statistics
using Optim

# Include source files
include("src/types.jl")
include("src/preprocessing.jl")
include("src/trackers/last_value.jl")
include("src/trackers/window_average.jl")
include("src/trackers/ewma.jl")
include("src/trackers/bayesian.jl")
include("src/experiment_engine.jl")
include("src/results_handler.jl")
include("src/hyper_optimizer.jl")

println("--- Multithreaded Hyperparameter Optimization Pipeline ---")
println("[INFO] Julia running with $(Threads.nthreads()) threads.")

# 1. Setup Data and CV Boundaries
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

# 2. Run Optimization
result = optimize_bayesian_tracker(ds, boundaries)

# 3. Print Results
println("\n=== OPTIMIZATION COMPLETE ===")
println("Convergence: ", Optim.converged(result))
println("Minimum LogLoss Achieved: ", Optim.minimum(result))

opt_params = Optim.minimizer(result)
println("\nOptimal Parameters Discovered:")
println("- prior_mean    = ", opt_params[1])
println("- prior_var     = ", opt_params[2])
println("- obs_var       = ", opt_params[3])
println("- process_noise = ", opt_params[4])

# Optional: Run a final validation with the best config
best_config = BayesianTracker(opt_params[1], opt_params[2], opt_params[3], opt_params[4])
best_metrics = evaluate_tracker_on_boundaries(best_config, ds, boundaries)

println("\nValidation LogLoss: ", best_metrics.log_loss)
println("Validation Edge Coef: ", best_metrics.glm_edge_coef)
println("Validation Edge P-Value: ", best_metrics.glm_edge_pvalue)
