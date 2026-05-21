# current_development/ab_test_outfield_player/r01_ab_test_runner.jl

# using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates
using Distributions
using Turing
using Statistics

# Short-hands
const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Signals = BayesianFootball.Signals

# --- Experiment Execution ---

# 1. Load Data (Ireland as requested)
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
# save_dir::String = "./data/ab_test_outfield_player/" # using the other files saved since that were the experiment is

save_dir::String = "./data/ab_test_hierarchical_player/"

# 2. Shared Component Configs
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()


# Bayesian Tracker for player ratings
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# 3. Model Definitions

# Model A: is in the save dir 
# Model A: Standard Player Time-Decay (G, D, M, F)
# model_std = PreGame.DynamicMarketXGPlayerTimeDecayModel(
#     interception_config  = inter_cfg,
#     player_dynamics_config = PreGame.PositionalPlayerDynamics(days_half_life=180.0),
#     dispersion_config    = disp_cfg,
#     homeadvantage_config = ha_cfg,
#     kappa_config         = kap_cfg,
#     player_ratings_feature = feature_cfg_bayes,
#     market_weight        = 1.0
# )
#
# Model B: Simplified Outfield Time-Decay (G, Outfield)
model_outfield = PreGame.DynamicMarketXGOutfieldPlayerTimeDecayModel(
    interception_config  = inter_cfg,
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=180.0),
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)

# 4. Execute Runs
println("\n[INFO] Starting A/B Test Execution...")
# task_std = Experiments.create_experiment_task(ds, model_std, "ab_std_player_", save_dir; target_seasons=["2025", "2026"], dynamics_col=:match_month)
task_outfield = Experiments.create_experiment_task(ds, model_outfield, "ab_outfield_player_", save_dir; target_seasons=["2025", "2026"], dynamics_col=:match_month)

# Print out the task to verify the new Base.show works!
display(task_outfield)

println("\n" * "="^60)
println(">>> RUNNING EXPERIMENT: $(task_outfield.config.name)")
println("="^60)

results_outfield = Experiments.run_experiment(task_outfield)
Experiments.save_experiment(results_outfield)
println("✅ Success: $(task_outfield.config.name)")

# --- Analysis & Evaluation ---

# all_results = [results_std, results_outfield];

saved_folders = Experiments.list_experiments(save_dir; data_dir="")

function loaded_experiment_files(saved_folders::Vector{String})
  loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
  for folder in saved_folders
      try
          res = Experiments.load_experiment(folder)
          push!(loaded_results, res)
      catch e
          @warn "Could not load $folder: $e"
      end
  end

  if isempty(loaded_results)
      error("No results loaded! Did you run runner.jl?")
  end

  return loaded_results

end

all_results = loaded_experiment_files(saved_folders[[1,3]]);



all_results = [results_outfield];

# 1. Predictive Metrics
println("\n>>> Evaluating Predictive Performance...")
metrics = [
    Evaluation.RQR(),
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]
master_eval_df = Evaluation.evaluate_experiments(metrics, all_results, ds)

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)

# 2. Backtesting (Bayesian Kelly Staking)
println("\n>>> Running Backtest Staking Analysis...")
ledger = BackTesting.run_backtest(
    ds, 
    all_results, 
    [Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

println("\n\n✅ A/B Test Complete. Results saved to $save_dir")

# 3. Check Convergence (R-hat) of the Outfield model
c_outfield = results_outfield.training_results[6][1] # Fold 6, Model 1
println("\n>>> Summary Statistics for Outfield Model (Fold 6):")
display(describe(c_outfield))
