# current_development/time_decayed_models/r04_test_player_models.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features

# --- Helper Functions ---

struct ExperimentTask
    ds::BayesianFootball.Data.DataStore
    config::BayesianFootball.Experiments.ExperimentConfig
end

function create_experiment_tasks(ds::BayesianFootball.Data.DataStore, model, label::String, save_dir::String, target_seasons::Vector{<:String} )
    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 3,
        dynamics_col = :match_week,
        warmup_period = 0,
        stop_early = true # Just run one fold for quick validation
    )

    sampler_conf = BayesianFootball.Samplers.NUTSConfig(
        500, # Reduced for quick testing
        2,   
        200,  
        0.65,
        10,  
        BayesianFootball.Samplers.UniformInit(-1, 1),
        false,
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=4
    )
    training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        BayesianFootball.Experiments.ExperimentConfig(
            name = "$(label)_",
            model = model, 
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
    ]

    return ExperimentTask.(Ref(ds), configs)
end

function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("\n>>> Running: $(conf.name)")

    try
        results = BayesianFootball.Experiments.run_experiment(task.ds, conf)
        BayesianFootball.Experiments.save_experiment(results)
        println("✅ Success: $(conf.name)")
        return true
    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        return false
    end
end

# --- Runner Logic ---

println("--- Validating DynamicMarketXGPlayerModel ---")

# 1. Load Data
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_src_player_models/"

# 2. Shared Config Components
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
# Player model requires PositionalPlayerDynamics
dyn_cfg   = PreGame.PositionalPlayerDynamics() 
kap_cfg   = PreGame.GlobalKappa()

# 3. Model A: Last Value Tracker
tracker_lv = Features.LastValueTracker()
feature_cfg_lv = Features.PlayerRatingsFeature(tracker_lv)

model_lv = PreGame.DynamicMarketXGPlayerModel(
    interception_config  = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_lv,
    market_weight        = 1.0
)

# 4. Model B: Bayesian Tracker
# Using reasonable parameters from the previous grid search
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

model_bayes = PreGame.DynamicMarketXGPlayerModel(
    interception_config  = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)

# 5. Create Tasks (Testing on 2026 season)
task_lv = create_experiment_tasks(ds, model_lv, "player_model_last_value", save_dir, ["2026"])
task_bayes = create_experiment_tasks(ds, model_bayes, "player_model_bayesian", save_dir, ["2026"])

# 6. Execute 
println("\n[INFO] Starting execution. This will run 1 fold per model for validation.")
run_experiment_task.(task_lv)
run_experiment_task.(task_bayes)

println("\nValidation complete. Check $save_dir for results.")
