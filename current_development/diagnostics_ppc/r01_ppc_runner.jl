# current_development/diagnostics_ppc/r01_ppc_runner.jl

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
saved_files = Experiments.list_experiments(save_dir, data_dir="")
results = BayesianFootball.Experiments.load_experiments(saved_files)[2]

# 3. Generate PPC Data
# We select fold_idx = 6 as a representative fold (as used in previous scripts)
fold_index = 10
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




println("\n" * "="^50)
println("1. Extracting Chains...")
println("="^50)
chains_df = BayesianFootball.Experiments.Diagnostics.extract_chains(ds, results)
display(chains_df)

println("\n" * "="^50)
println("2. Convergence Diagnostics...")
println("="^50)
conv_diag = BayesianFootball.Experiments.Diagnostics.check_convergence(chains_df)
display(conv_diag)

println("\n" * "="^50)
println("3. Stability Diagnostics...")
println("="^50)
stab_diag = BayesianFootball.Experiments.Diagnostics.check_stability(chains_df)
display(stab_diag)

println("\nDemo Completed Successfully!")



metrics = [
    Evaluation.RQR(),
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]
master_eval_df = Evaluation.evaluate_experiments(metrics, [results], ds)

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)





ledger = BackTesting.run_backtest(
    ds, 
    [results], 
    [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

