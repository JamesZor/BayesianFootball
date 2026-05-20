# current_development/time_decayed_models/r06_investigate_chains.jl

using Pkg; Pkg.activate(".")
using BayesianFootball
using MCMCChains
using DataFrames
using Statistics

# 1. Setup paths (Matching user's previous context)
save_dir = "./data/test_src_player_models/"
saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = []
for folder in saved_folders
    try
        push!(loaded_results, Experiments.load_experiment(folder))
    catch e
        @warn "Could not load $folder"
    end
end

if isempty(loaded_results)
    error("No results found. Run the experiment first.")
end

# Assuming we are looking at the first experiment (e.g., Bayesian Tracker)
expr = loaded_results[1]
results = expr.training_results.items

"""
    check_fold_diagnostics(results, fold_idx)

Computes R-hat and ESS for a specific fold to check convergence.
"""
function check_fold_diagnostics(results, fold_idx)
    # results is 1-indexed, but user's 'fold' col might be 0-indexed or match indices
    # We'll use the index directly from the results array.
    # Note: If user says "Fold 5", and they start at 0, that's results[6].
    chain, meta = results[fold_idx + 1]
    
    println("\n" * "="^50)
    println("DIAGNOSTICS FOR FOLD: $(fold_idx) ($(meta.target_season) Week $(meta.time_step))")
    println("="^50)
    
    # Filter for the specific parameters we observed instability in
    target_params = [:w_G_att, :w_D_att, :w_M_att, :w_F_att, :w_G_def, :w_D_def, :w_M_def, :w_F_def]
    # Turing prefixes these with "p_dyn."
    prefixed_params = [Symbol("p_dyn.", p) for p in target_params]
    
    # Also check log-density (lp)
    push!(prefixed_params, :lp)

    # Get summary with R-hat and ESS
    # We only take params that actually exist in the chain
    available_params = [p for p in prefixed_params if p in keys(chain)]
    summ = summarize(chain[available_params])
    
    # Show the summary
    show(summ)
    println()
    
    # Check for R-hat > 1.05
    bad_rhat = subset(DataFrame(summ), :rhat => r -> r .> 1.05)
    if !isempty(bad_rhat)
      @warn "High R-hat detected in Fold $(fold_idx)!"
        show(bad_rhat)
    else
        println("✅ All R-hat values are within acceptable bounds (< 1.05).")
    end
    
    return summ
end

# 2. Run Diagnostics
println("Comparing Stable vs Unstable Folds...")

# Fold 0 (Stable)
check_fold_diagnostics(results, 0)

# Folds 5, 6, 11 (Unstable)
check_fold_diagnostics(results, 5)
check_fold_diagnostics(results, 6)
check_fold_diagnostics(results, 11)
