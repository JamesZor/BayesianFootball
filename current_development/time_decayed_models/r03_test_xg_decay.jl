# current_development/time_decayed_models/r03_test_xg_decay.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates
using Distributions

const PreGame = BayesianFootball.Models.PreGame

# --- Helper Functions (Copied for runner logic) ---

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end

function create_experiment_tasks(ds::Data.DataStore, model, label::String, save_dir::String, target_seasons::Vector{<:String} )
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 3,
        dynamics_col = :match_week,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = Samplers.NUTSConfig(
        500, # Reduced for quick testing
        2,   
        200,  
        0.65,
        10,  
        Samplers.UniformInit(-1, 1),
        false,
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=8
    )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        Experiments.ExperimentConfig(
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
        results = Experiments.run_experiment(task.ds, conf)
        Experiments.save_experiment(results)
        println("✅ Success: $(conf.name)")
        return true
    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        return false
    end
end

# --- Runner Logic ---

# 1. Load Data
# Ireland typically has both xG and Market data
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_src_xg_decay/"

# 2. Shared Config Components
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life = 180)
kap_cfg   = PreGame.GlobalKappa()

# 3. Model A: DynamicXGTimeDecayModel
model_xg = PreGame.DynamicXGTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)

# 4. Model B: DynamicMarketXGTimeDecayModel
model_market_xg = PreGame.DynamicMarketXGTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    market_weight        = 1.0
)

# 5. Create Tasks
task_xg = create_experiment_tasks(ds, model_xg, "src_xg_decay_test", save_dir, ["2026"])
task_market_xg = create_experiment_tasks(ds, model_market_xg, "src_market_xg_decay_test", save_dir, ["2026"])

# To execute the test runs:
run_experiment_task.([task_xg[1], task_market_xg[1]])
# run_experiment_task(task_xg[1])
# run_experiment_task(task_market_xg[1])

println("\nxG Decay test runners initialized.")
println("1. DynamicXGTimeDecayModel")
println("2. DynamicMarketXGTimeDecayModel (Market Weight: 0.5)")
