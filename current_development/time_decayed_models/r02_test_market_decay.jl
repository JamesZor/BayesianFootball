# current_development/time_decayed_models/r02_test_market_decay.jl

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
        1000, # Number of samples for each chain
        4,   # Number of chains
        200, # Number of warm up steps 
        0.65,# Accept rate  [0,1]
        10,  # Max tree depth
        Samplers.UniformInit(-1, 1), # Interval for starting a chain 
        false,   # show_progress
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=4
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
    println("Running: $(conf.name)")

    try
        results = Experiments.run_experiment(task.ds, conf)
        Experiments.save_experiment(results)
        return true
    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        return false
    end
end

# --- Runner Logic ---

# 1. Load Data
# Note: Ireland segment has market data available
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_src_market_time_decay/"

# 2. Instantiate Model
# Testing with market integration and custom market_weight
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life = 180)

model = PreGame.DynamicMarketGoalsTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    market_weight        = 0.5 # Testing with reduced market influence
)

# 3. Create and Run Tasks
# Testing on 2026 season
training_tasks = create_experiment_tasks(ds, model, "src_market_decay_test_w05", save_dir, ["2026"])

# To execute the test run:
results = run_experiment_task.(training_tasks)

println("Test runner for DynamicMarketGoalsTimeDecayModel initialized.")
println("Label: src_market_decay_test_w05")
println("Market Weight: $(model.market_weight)")
