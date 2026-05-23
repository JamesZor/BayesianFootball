# current_development/01_optimization_tests/r01_runner.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores)
using BenchmarkTools

include("l01_loaders.jl")

# 1. Load Data
println("Loading Data...")
ds = Data.load_datastore_cached(Data.Ireland())
# Removed subsetting as dataset is already small enough

# 2. Create MAP Experiment
println("--- Running MAP Experiment ---")
map_task = create_optim_test_task(ds, use_map=true)

# Time the MAP execution
map_time = @elapsed begin
    map_results = Experiments.run_experiment(map_task)
end
println("MAP Experiment completed in $(round(map_time, digits=2)) seconds.")

# 3. Predict & Check Diagnostics for MAP
println("Extracting MAP Diagnostics...")
map_chains = Experiments.Diagnostics.extract_chains(ds, map_results)
println(map_chains)

println("Running MAP Inference...")
map_preds = Predictions.model_inference(ds, map_results)
println("Total MAP Predictions: ", nrow(map_preds.df))

# 4. Create NUTS Experiment (for comparison)
println("\n--- Running NUTS Experiment ---")
nuts_task = create_optim_test_task(ds, use_map=false)

# Time the NUTS execution
nuts_time = @elapsed begin
    nuts_results = Experiments.run_experiment(nuts_task)
end
println("NUTS Experiment completed in $(round(nuts_time, digits=2)) seconds.")

println("\n--- Speedup Summary ---")
println("MAP Time:  $(round(map_time, digits=2))s")
println("NUTS Time: $(round(nuts_time, digits=2))s")
println("Speedup:   $(round(nuts_time / map_time, digits=2))x")
