# current_development/ab_test_outfield_player/r02_ab_test_dixon_coles.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning

# Lock OS threads to hardware cores for optimal MCMC queue performance
pinthreads(:cores) 

# Aliases
const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
# Standard cached datastore load for the target tournament
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
save_dir = "./mcmc_checkpoints/ab_test_dixon_coles/"

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=180.0)

# Bayesian Tracker for positional player ratings
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
println("[INFO] Initializing Models...")

# Model A: Original Outfield XG Market Model
model_market = PreGame.DynamicMarketXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 1.0
)

# Model B: New Outfield XG Dixon-Coles Model
model_dixon = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 1.0
)

# ==========================================
# 4. EXPERIMENT TASK CREATION
# ==========================================
println("[INFO] Creating Queued MCMC Tasks...")

# Setting `use_queue=true` automatically initializes `QueuedNUTSConfig` internally
# and dynamically scales independent training folds across `Threads.nthreads()`.
task_market = Experiments.create_experiment_task(
    ds, 
    model_market, 
    "ab_outfield_xg_market", 
    save_dir; 
    target_seasons=["2025", "2026"], 
    dynamics_col=:match_month,
    samples=1000,   
    warmup=1000,    
    chains=4,       
    use_queue=true  # <--- Triggers the new high-performance QueuedNUTSConfig
)

task_dixon = Experiments.create_experiment_task(
    ds, 
    model_dixon, 
    "ab_outfield_xg_dixon_coles", 
    save_dir; 
    target_seasons=["2025", "2026"], 
    dynamics_col=:match_month,
    samples=1000,   
    warmup=1000,    
    chains=4,       
    use_queue=true  # <--- Triggers the new high-performance QueuedNUTSConfig
)

# ==========================================
# 5. CONCURRENT EXECUTION
# ==========================================
println("\n" * "="^60)
println(">>> RUNNING QUEUED EXPERIMENTS CONCURRENTLY")
println("="^60)

# Kick off both queuing jobs asynchronously. The internal Queue logic inside 
# `Experiments.run_experiment` ensures hardware is saturated via `Threads.nthreads()`
job_market = Threads.@spawn Experiments.run_experiment(task_market)
job_dixon  = Threads.@spawn Experiments.run_experiment(task_dixon)

# Wait for both queued MCMC batches to fully synthesize
results_market = fetch(job_market)
results_dixon  = fetch(job_dixon)

# ==========================================
# 6. SAVE
# ==========================================
println("\n[INFO] Saving Experiment Results...")
Experiments.save_experiment(results_market)
Experiments.save_experiment(results_dixon)

println("✅ A/B Test Execution Complete! Models persisted to: \$save_dir")
