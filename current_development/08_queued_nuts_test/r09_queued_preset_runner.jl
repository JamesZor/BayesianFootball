# current_development/08_queued_nuts_test/r09_queued_preset_runner.jl

# using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames

using ThreadPinning
pinthreads(:cores) # Lock our Julia threads to physical hardware cores!

# Short-hands
const PreGame = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./mcmc_checkpoints/queued_preset_test/"

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

inter_cfg = BayesianFootball.Models.PreGame.GlobalInterception()
disp_cfg  = BayesianFootball.Models.PreGame.HomeAwayDispersion()
ha_cfg    = BayesianFootball.Models.PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = BayesianFootball.Models.PreGame.TimeDecayDynamics(days_half_life=60.0)

model = BayesianFootball.Models.PreGame.DynamicGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg
)


# ==========================================
# 3. EXPERIMENT TASK CREATION
# ==========================================
println("\n[INFO] Creating Experiment Task...")

# Thanks to our updates in `presets.jl`, this will now automatically:
# 1. Use `QueuedNUTSConfig` internally.
# 2. Use `Independent` strategy and auto-detect your max cores via `Threads.nthreads()`.
task = Experiments.create_experiment_task(
    ds, 
    model, 
    "queued_preset_test", 
    save_dir; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    samples=800,    # Reduced for quick testing
    warmup=200,     # Reduced for quick testing
    chains=4,       # Standard 4 chains
    use_queue=true  # <--- This triggers the new blazing fast MCMC queue (it defaults to true anyway!)
)

display(task)

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
println("\n" * "="^60)
println(">>> RUNNING QUEUED EXPERIMENT: $(task.config.name)")
println("="^60)

# This will spawn a massive queue of `splits × 4 chains` tasks
# and process them concurrently without any idle CPU cores!
results = Experiments.run_experiment(task)

println("\n[INFO] Saving Experiment...")
Experiments.save_experiment(results)

println("✅ Success! The new preset pipeline is working perfectly!")


chains_df_all = Experiments.Diagnostics.extract_chains(ds, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)
display(conv_diag)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Diagnostics.check_stability(chains_df_all)
stab_diag_outfield = Diagnostics.check_stability(chains_df_outfield)
display(stab_diag)

