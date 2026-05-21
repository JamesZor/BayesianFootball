# current_development/demo_diagnostics.jl

using Pkg; Pkg.activate(".")
using BayesianFootball

println("[INFO] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

println("[INFO] Setting up model...")
model = BayesianFootball.Models.PreGame.DynamicGoalsModel(
    interception_config  = BayesianFootball.Models.PreGame.GlobalInterception(),
    dynamics_config      = BayesianFootball.Models.PreGame.MultiScaleGRW(),
    dispersion_config    = BayesianFootball.Models.PreGame.HomeAwayDispersion(),
    homeadvantage_config = BayesianFootball.Models.PreGame.GlobalHomeAdvantage()
)

println("[INFO] Creating Experiment Task...")
task = BayesianFootball.Experiments.create_experiment_task(
    ds, 
    model, 
    "demo_diagnostics", 
    "./data/demo_diag"; 
    target_seasons=["24/25"], 

    history_seasons=1,
    samples=100, # Very fast
    warmup=100,
    chains=2,
    max_concurrent_splits=2
)

println("[INFO] Running Experiment (Fast)...")
results = BayesianFootball.Experiments.run_experiment(task)

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
