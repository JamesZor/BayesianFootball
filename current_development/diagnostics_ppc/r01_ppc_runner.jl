# current_development/diagnostics_ppc/r01_ppc_runner.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball

# Include the loader script
include("l01_ppc_diagnostics.jl")

# 1. Load Data
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# 2. Load Completed Experiment
save_dir = "./data/ab_test_hierarchical_player/"
println("[INFO] Loading Experiment from $save_dir...")
results = BayesianFootball.Experiments.load_experiments(save_dir)[1]

# 3. Generate PPC Data
# We select fold_idx = 6 as a representative fold (as used in previous scripts)
fold_index = 6
println("[INFO] Generating PPC for fold $fold_index...")
pp_h, pp_a, obs_h, obs_a = generate_pp_goals(results, ds; fold_idx=fold_index)

# 4. Generate & Save HTML Plots
# Saving directly to the root of the project so they are easy to serve via python -m http.server
home_filename = "ppc_home_goals_fold$(fold_index).html"
away_filename = "ppc_away_goals_fold$(fold_index).html"

println("[INFO] Plotting overlays...")
plot_ppc(pp_h, obs_h, "Posterior Predictive Check: Home Goals", home_filename)
plot_ppc(pp_a, obs_a, "Posterior Predictive Check: Away Goals", away_filename)

println("\n✅ Done! You can view the plots at:")
println("   http://localhost:8080/$home_filename")
println("   http://localhost:8080/$away_filename")
