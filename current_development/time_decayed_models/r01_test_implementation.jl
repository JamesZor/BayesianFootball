# current_development/time_decayed_models/r01_test_implementation.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates
using Distributions

const PreGame = BayesianFootball.Models.PreGame

# --- Helper Functions (Copied from l00 for runner logic) ---

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
        500, # Number of samples for each chain
        4,   # Number of chains
        150, # Number of warm up steps 
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

function loaded_experiment_files(saved_folders::Vector{String})
    loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
    for folder in saved_folders
        try
            res = Experiments.load_experiment(folder)
            push!(loaded_results, res)
        catch e
            @warn "Could not load $folder: $e"
        end
    end
    return loaded_results
end

# --- Runner Logic ---

# 1. Load Data
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_src_time_decay/"

# 2. Instantiate Model (using src implementation)
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life = 180)

model = PreGame.DynamicGoalsTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg
)

# 3. Create and Run Tasks
training_tasks = create_experiment_tasks(ds, model, "src_time_decay_test", save_dir, ["2026"])

results = run_experiment_task.(training_tasks)

# 4. (Optional) Load results and run backtest summary
# saved_folders = Experiments.list_experiments(save_dir; data_dir="")
# loaded_results = loaded_experiment_files(saved_folders)
# println("Loaded $(length(loaded_results)) results.")
