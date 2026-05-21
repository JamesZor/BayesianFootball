# current_development/demo_experiment_ux.jl
# 
# A quick script to demonstrate the refactored Experiments UX,
# specifically the beautiful new display methods!
#
# Run this from the REPL via:
# julia> include("current_development/demo_experiment_ux.jl")

using Revise
using BayesianFootball

const PreGame = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments

println("========================================")
println(" 🚀 EXPERIMENT UX REFACTOR DEMO")
println("========================================\n")

# 1. Load some Data
println("[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# 2. Setup a quick Model
println("\n[2] Setting up Model Components...")
model = PreGame.DynamicMarketXGOutfieldPlayerTimeDecayModel(
    interception_config  = PreGame.GlobalInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=180.0),
    dispersion_config    = PreGame.HomeAwayDispersion(),
    homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config         = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = BayesianFootball.Features.PlayerRatingsFeature(
        BayesianFootball.Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    ),
    market_weight        = 1.0
)

# 3. Create the Task using the new Factory
println("\n[3] Creating Experiment Task...")
task = Experiments.create_experiment_task(
    ds, 
    model, 
    "demo_experiment_run", 
    "./data/demo"; 
    target_seasons=["2025", "2026"],
    history_seasons=2,
    samples=1000,
    warmup=300,
    chains=4
)

# 4. Show off the Display!
println("\n[4] Calling display(task)...\n")
display(task)

println("\n✅ Demo Complete! To run this experiment, you would simply execute:")
println("   Experiments.run_experiment(task)")
