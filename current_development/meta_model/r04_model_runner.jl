using BayesianFootball
using DataFrames
using Turing
using MCMCChains

using ThreadPinning

pinthreads(:cores)


println("--- A/B Testing: Goals vs Global Copula vs Hierarchical Copula ---")

# 1. Setup Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
save_dir = "./data/meta_model_layer1/"
# 2. Shared Base Configurations
inter_cfg = BayesianFootball.Models.PreGame.GlobalInterception()
disp_cfg  = BayesianFootball.Models.PreGame.HomeAwayDispersion()
ha_cfg    = BayesianFootball.Models.PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = BayesianFootball.Models.PreGame.TimeDecayDynamics(days_half_life=60.0)

# 3. Model Instances

# Model A: Standard Poisson/NegBin Goals Model (No Copula)
model_goals = BayesianFootball.Models.PreGame.DynamicGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg
)

task = Experiments.create_experiment_task(
    ds, 
    model_goals, 
    "queued_preset_test", 
    save_dir; 
    target_seasons = ["22/23","23/24","24/25", "25/26"],
    dynamics_col = :match_biweek,
    samples=800,    # Reduced for quick testing
    warmup=200,     # Reduced for quick testing
    chains=4,       # Standard 4 chains
    use_queue=true,  # <--- This triggers the new blazing fast MCMC queue (it defaults to true anyway!)
    max_concurrent_tasks = 16
)

results = Experiments.run_experiment(task)
Experiments.save_experiment(results)


#  Diagnostics 
chains_df_all = Experiments.Diagnostics.extract_chains(ds, results)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Experiments.Diagnostics.check_stability(chains_df_all)


metrics = [
    BayesianFootball.Evaluation.LogLoss(), 
    BayesianFootball.Evaluation.CRPS(), 
    BayesianFootball.Evaluation.RQR(),
    BayesianFootball.Evaluation.GLMEdge()
]

master_eval_df = BayesianFootball.Evaluation.evaluate_experiments(metrics, [results], ds)

println("\n>>> Evaluation Results (Sorted by LogLoss):")
display(sort(master_eval_df, :logloss_overall_diff_ll))

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)

