# current_development/copula_negbin/r01_frank_copula_src.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

println("--- Testing Hierarchical Frank Copula NegBin Model from SRC ---")

# 1. Setup Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# 2. Configure the Model
inter_cfg = BayesianFootball.Models.PreGame.GlobalInterception()
disp_cfg  = BayesianFootball.Models.PreGame.HomeAwayDispersion()
ha_cfg    = BayesianFootball.Models.PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = BayesianFootball.Models.PreGame.TimeDecayDynamics(days_half_life=90.0)
cop_cfg   = BayesianFootball.Models.PreGame.HierarchicalFrankCopulaConfig()

copula_model = BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg,
    copula_config=cop_cfg
)

# 3. CV Configuration
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["2026"],
    history_seasons = 2,
    dynamics_col = :match_month,
    warmup_period = 0,
    stop_early = true # Just 1 fold for smoke testing
)

# 4. Sampler Config (Using NUTS to properly explore the Hierarchical Copula)
sampler_config = BayesianFootball.Samplers.NUTSConfig(
    samples=800,
    warmup=400,
    chains=4
)

train_cfg = BayesianFootball.Training.Independent(
    parallel = true,
    max_concurrent_splits = 4
)

training_config = BayesianFootball.Training.TrainingConfig(sampler_config, train_cfg, nothing, false)

# 5. Experiment Setup
config = BayesianFootball.Experiments.ExperimentConfig(
    name = "hierarchical_copula_model_test",
    model = copula_model, 
    splitter = cv_config,
    training_config = training_config,
    save_dir = "./data/hierarchical_copula_test/"
)

task = BayesianFootball.Experiments.ExperimentTask(ds, config)

# 6. Run Experiment
println(">>> Running Hierarchical Copula Experiment...")
results = BayesianFootball.Experiments.run_experiment(task)

# 7. Extract Results
println(">>> Extracting Chains...")
chains = BayesianFootball.Experiments.Diagnostics.extract_chains(ds, results)

println(">>> Chain Summary:")
display(describe(chains.df))

println("\nAll done! Model works dynamically from src!")
