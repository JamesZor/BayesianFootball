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
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Signals = BayesianFootball.Signals

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
save_dir::String = "./data/ab_test_hierarchical_player/"

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
# Shared Component Configs
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

# Bayesian Tracker for player ratings
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

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

model_all = PreGame.DynamicMarketXGPlayerTimeDecayModel(
    interception_config  = inter_cfg,
    player_dynamics_config = PreGame.PositionalPlayerDynamics(days_half_life=180.0),
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)


# ==========================================
# 3. EXPERIMENT TASK CREATION
# ==========================================
println("\n[INFO] Creating Experiment Task...")
task_outfield = Experiments.create_experiment_task(
    ds, 
    model_outfield, 
    "ab_outfield_player", 
    save_dir; 
    target_seasons=["2025", "2026"], 
    dynamics_col=:match_month,
    samples=1000,   # Increased samples
    warmup=1000,    # Increased warmup
    chains=4        # Ensure 4 chains for robust R-hat checking
)

task_all = Experiments.create_experiment_task(
    ds, 
    model_all, 
    "ab_all_player", 
    save_dir; 
    target_seasons=["2025", "2026"], 
    dynamics_col=:match_month,
    samples=1000,   # Increased samples
    warmup=1000,    # Increased warmup
    chains=4        # Ensure 4 chains for robust R-hat checking
)


display(task_outfield)

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
println("\n" * "="^60)
println(">>> RUNNING EXPERIMENT: $(task_outfield.config.name)")
println("="^60)

results_all = Experiments.run_experiment(task_all)
results_outfield = Experiments.run_experiment(task_outfield)

println("\n[INFO] Saving Experiment...")
Experiments.save_experiment(results_outfield)
Experiments.save_experiment(results_all)

println("✅ Success: $(task_outfield.config.name)")

# ==========================================
# 5. DIAGNOSTICS (NEW STANDARD WORKFLOW)
# ==========================================
println("\n" * "="^50)
println(">>> EXPERIMENT DIAGNOSTICS")
println("="^50)

# Extract MCMC chains into long-format dataframe
#
chains_df_all = Diagnostics.extract_chains(ds, results_all)
chains_df_outfield = Diagnostics.extract_chains(ds, results_outfield)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Diagnostics.check_convergence(chains_df_all)
conv_diag_outfield = Diagnostics.check_convergence(chains_df_outfield)
display(conv_diag)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Diagnostics.check_stability(chains_df_all)
stab_diag_outfield = Diagnostics.check_stability(chains_df_outfield)
display(stab_diag)

# ==========================================
# 6. EVALUATION
# ==========================================
println("\n" * "="^50)
println(">>> PREDICTIVE EVALUATION")
println("="^50)

metrics = [
    Evaluation.RQR(),
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]
master_eval_df = Evaluation.evaluate_experiments(metrics, [results_all,results_outfield], ds)

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)

# ==========================================
# 7. BACKTESTING
# ==========================================
println("\n" * "="^50)
println(">>> BACKTEST STAKING ANALYSIS")
println("="^50)

ledger = BackTesting.run_backtest(
    ds, 
    [results_all, results_outfield], 
    [Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows=true)

println("\n\n✅ Complete pipeline executed successfully! Results saved to $save_dir")



odds =Data.summarize_betfair_market(
    ds, 
    open_window=(-100000.0, -10.0), 
    close_window=(-10.0, 0.0)
)

ds1 = Data.DataStore(
  ds.segment,
  ds.matches,
  ds.statistics,
  odds,
  ds.lineups,
  ds.incidents,
  ds.betfair_odds
  )

ledger1 = BackTesting.run_backtest(
    ds1, 
    [results_all, results_outfield], 
    [Signals.BayesianKelly()]; 
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet1 = BackTesting.generate_tearsheet(ledger1)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet1[:, cols_to_show], allrows=true)

