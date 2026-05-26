# current_development/08_queued_nuts_test/r08_queued_runner.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

using ThreadPinning
pinthreads(:cores)

# Include loader if needed
# include("l08_queued_loader.jl")

# --- Setup ---
println("--- Testing Queued NUTS Execution ---")
# Adjust to whatever segment you typically test with
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

# 1. NEW SAMPLER CONFIG: QueuedNUTSConfig
# We set show_progress=false to disable Turing's inner bar
sampler_conf = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_warmup = 100,
    n_chains = 6,   # 4 chains per split
    n_samples = 100, 
    accept_rate = 0.65,
    max_depth = 10,
    show_progress = false, 
    initialisation = BayesianFootball.Samplers.UniformInit(-1, 1)
)

# Common Components
inter_cfg = BayesianFootball.Models.PreGame.GlobalInterception()
disp_cfg  = BayesianFootball.Models.PreGame.HomeAwayDispersion()
ha_cfg    = BayesianFootball.Models.PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = BayesianFootball.Models.PreGame.TimeDecayDynamics(days_half_life=60.0)

# 3. Model Instances

# Model A: Standard Poisson/NegBin Goals Model (No Copula)
test_model = BayesianFootball.Models.PreGame.DynamicGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg
)


# --- CV Configuration ---
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 1,
    dynamics_col = :match_month,
    warmup_period = 4,
    stop_early = false # Run all folds to test the queue!
)

# --- Execution ---
try
    # Create Split Boundaries
    boundaries = Data.create_id_boundaries(ds, cv_config)
    
    # Create Features for all splits
    feature_sets = Features.create_features(boundaries, ds, test_model, cv_config.dynamics_col)
    
    # 2. NEW TRAINING CONFIG: Independent with max_concurrent_tasks
    train_cfg = BayesianFootball.Training.Independent(
        parallel = true, 
        max_concurrent_tasks = Threads.nthreads() # Set to your server's core count
    )
    
    training_config = BayesianFootball.Training.TrainingConfig(
        sampler = sampler_conf, 
        strategy = train_cfg, 
        checkpoint_dir = "./mcmc_checkpoints", 
        cleanup_checkpoints = false
    )
    
    # 3. Run the queue!
    results = BayesianFootball.Training.train(test_model, training_config, feature_sets)
    
    println("\n✅ Queued MCMC Run Successful!")
    println("Number of splits processed: ", length(results))
    
    # Example check of first result
    first_chain, _ = results[1]
    println("First chain dimensions: ", size(first_chain))
    
catch e
    @error "❌ FAILED" exception=(e, catch_backtrace())
end
