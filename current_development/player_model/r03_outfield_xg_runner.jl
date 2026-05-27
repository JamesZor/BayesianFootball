# current_development/player_model/r03_outfield_xg_runner.jl

using BayesianFootball
using DataFrames
using Turing
using MCMCChains

using ThreadPinning
pinthreads(:cores)

# Short-hands
const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation

println("--- Testing: Outfield xG Model (No Market Data) ---")

# 1. Setup Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

save_dir = "./data/meta_model_layer1/ireland"

# 2. Shared Base Configurations
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

# Bayesian Tracker for player ratings
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# 3. Model Instances

# Outfield Time-Decay xG Model (No Market)
model_outfield_xg = PreGame.DynamicXGOutfieldPlayerTimeDecayModel(
    interception_config  = inter_cfg,
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=180.0),
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes
)

task = Experiments.create_experiment_task(
    ds, 
    model_outfield_xg, 
    "outfield_xg_player_test", 
    save_dir; 
    target_seasons = [ "2024", "2025". "2026"],
    dynamics_col = :match_biweek,
    samples=800,    # Reduced for quick testing
    warmup=200,     # Reduced for quick testing
    chains=4,       # Standard 4 chains
    use_queue=true, # Triggers the new blazing fast MCMC queue
    max_concurrent_tasks = 16
)

results = Experiments.run_experiment(task)
Experiments.save_experiment(results)


# ==========================================
# DIAGNOSTICS
# ==========================================
chains_df_all = Diagnostics.extract_chains(ds, results)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Diagnostics.check_convergence(chains_df_all)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Diagnostics.check_stability(chains_df_all)


# ==========================================
# EVALUATION
# ==========================================
metrics = [
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.RQR(),
    Evaluation.GLMEdge()
]

master_eval_df = Evaluation.evaluate_experiments(metrics, [results], ds)

println("\n>>> Evaluation Results (Sorted by LogLoss):")
display(sort(master_eval_df, :logloss_overall_diff_ll))

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)
