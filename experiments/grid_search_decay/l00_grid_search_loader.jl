using BayesianFootball
using Revise
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Turing, Distributions, Dates

const PreGame = BayesianFootball.Models.PreGame

# --- 1. Structs for Experiment Orchestration ---

struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
  target_seasons::Vector{<:String}
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end

# --- 2. Task Creation Logic ---

"""
    create_experiment_tasks(ds, model, label, save_dir, target_seasons)

Creates an ExperimentTask for a specific model configuration.
"""
function create_experiment_tasks(
    ds::Data.DataStore, 
    model::PreGame.AbstractFootballModel, 
    label::String, 
    save_dir::String, 
    target_seasons::Vector{<:String}
)
    # 1. Define the CV and Training Configs
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 4,
        dynamics_col = :match_week,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = Samplers.NUTSConfig(
        1000, 
        4,   
        200,  
        0.65,
        10,  
        Samplers.UniformInit(-1, 1),
        false,
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel = true,
        max_concurrent_splits = 4
    )

    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


    # 2. Build the ExperimentConfig
    config = Experiments.ExperimentConfig(
        name = "$(label)",
        model = model, 
        splitter = cv_config,
        training_config = training_config,
        save_dir = save_dir
    )

    return ExperimentTask(ds, config)
end

# --- 3. Execution Logic ---

"""
    run_experiment_task(task::ExperimentTask)

Executes a single experiment task and saves the results.
"""
function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("\n>>> Starting Experiment: $(conf.name)")

    try
        # 1. Execute the experiment
        results = Experiments.run_experiment(task.ds, conf)

        # 2. Persist the results
        Experiments.save_experiment(results)
        
        println("✅ Completed: $(conf.name)")
        return true 

    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        return false
    end
end

# --- 4. Result Loading Helpers ---

"""
    loaded_experiment_files(folders)

Loads all experiment results from a list of directory paths.
"""
function loaded_experiment_files(folders::Vector{String})
    return [Experiments.load_experiment(f) for f in folders]
end
