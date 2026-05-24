# current_development/copula_negbin/r02_ab_test.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

println("--- A/B Testing: Goals vs Global Copula vs Hierarchical Copula ---")

# 1. Setup Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

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

# Model B: Global Copula Model
global_cop_cfg = BayesianFootball.Models.PreGame.GlobalFrankCopulaConfig()
model_global_copula = BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg,
    copula_config=global_cop_cfg
)

# Model C: Hierarchical Copula Model
hierarchical_cop_cfg = BayesianFootball.Models.PreGame.HierarchicalFrankCopulaConfig()
model_hierarchical_copula = BayesianFootball.Models.PreGame.DynamicCopulaGoalsTimeDecayModel(
    interception_config=inter_cfg, 
    dynamics_config=dyn_cfg, 
    dispersion_config=disp_cfg, 
    homeadvantage_config=ha_cfg,
    copula_config=hierarchical_cop_cfg
)

# 4. CV Configuration
cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
    target_seasons = ["22/23","23/24","24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    stop_early = true # Just 1 fold for speed
)

# 5. Sampler Config (NUTS)
sampler_config = BayesianFootball.Samplers.NUTSConfig(
            500,  # samples
            4,    # chains
            200,  # warmup
            0.65, # accept_rate
            10,   # max_depth
            Samplers.UniformInit(-2, 2),
            false  # show_progress
        )


train_cfg = BayesianFootball.Training.Independent(
    parallel = true,
    max_concurrent_splits = 4
)
training_config = BayesianFootball.Training.TrainingConfig(sampler_config, train_cfg, nothing, false)

# 6. Run Experiments Loop
models_to_test = [
    ("A_Standard_Goals", model_goals),
    ("B_Global_Copula", model_global_copula),
    ("C_Hierarchical_Copula", model_hierarchical_copula)
]

all_results = BayesianFootball.Experiments.ExperimentResults[]

for (name, model) in models_to_test
    println("\n==============================================")
    println(">>> Running Model: $name")
    println("==============================================")
    
    config = BayesianFootball.Experiments.ExperimentConfig(
        name = name,
        model = model, 
        splitter = cv_config,
        training_config = training_config,
        save_dir = "./data/copula_ab_test/"
    )

    task = BayesianFootball.Experiments.ExperimentTask(ds, config)
    results = BayesianFootball.Experiments.run_experiment(task)
    Experiments.save_experiment(results)
    push!(all_results, results)
end

# 7. Batch Evaluation
println("\n>>> Running Batch Evaluation...")
metrics = [
    BayesianFootball.Evaluation.LogLoss(), 
    BayesianFootball.Evaluation.CRPS(), 
    BayesianFootball.Evaluation.RQR(),
    BayesianFootball.Evaluation.GLMEdge()
]

master_eval_df = BayesianFootball.Evaluation.evaluate_experiments(metrics, all_results, ds)

println("\n>>> Evaluation Results (Sorted by LogLoss):")
display(sort(master_eval_df, :logloss_overall_diff_ll))

println("\nAll done! A/B test complete.")
