using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)


function get_datastore_local_ip()

db_config = Data.DBConfig("postgresql://admin:supersecretpassword@192.168.1.88:5432/sofascrape_db")
db_conn =   Data.connect_to_db(db_config)
segment =   Data.ScottishLower()
  try
    data_store = Data.get_datastore(db_conn, segment)
    return data_store
  finally
    close(db_conn) 
  end 
end


function get_datastore_legacy()
  ds = BayesianFootball.DataLegacy.load_extra_ds()
# Generate the month index required by the time-varying models
  transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
  return ds 
end

struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end

function create_experiment_tasks(es::DSExperimentSettings)
    return create_experiment_tasks(es.ds, es.label, es.save_dir)
end

function create_list_experiment_tasks(es_list::Vector{DSExperimentSettings}) 
    # Map the creator over the list and flatten the nested results
    return reduce(vcat, create_experiment_tasks.(es_list); init=ExperimentTask[])
end

function create_list_experiment_configs(es_list::Vector{DSExperimentSettings})
    return reduce(vcat, create_experiment_configs_for_ds.(es_list))
end


function create_experiment_tasks(ds::Data.DataStore, label::String, save_dir::String)

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons = ["22/23", "23/24", "24/25", "25/26"],
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = Samplers.NUTSConfig(200, 2, 100, 0.65, 10, Samplers.UniformInit(-1, 1), false)
    train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8)
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # 2. Build the list of Configs
    configs = [
        Experiments.ExperimentConfig(
            name = "$(label)_01_baseline",
            model = Models.PreGame.AblationStudy_NB_baseLine(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
        Experiments.ExperimentConfig(
            name = "$(label)_02_monthlyR",
            model = Models.PreGame.AblationStudy_NB_baseline_month_r(),
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        )
    ]

    # 3. THE "SMART" BIT: 
    # Wrap every config with the DS into an ExperimentTask
    # We use Ref(ds) so it doesn't try to "iterate" the DataStore
    return ExperimentTask.(Ref(ds), configs)
end

function create_list_experiment_tasks(es_list::Vector{DSExperimentSettings}) 
    # Map the creator over the list and flatten the nested results
    return reduce(vcat, create_experiment_tasks.(es_list); init=ExperimentTask[])
end

function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("Running: $(conf.name)")

    # 1. Muzzle the logs for the noisy sampling phase
    Logging.disable_logging(Logging.Info)

    try
        # 2. Execute
        results = Experiments.run_experiment(task.ds, conf)

        # 3. Re-enable logging to save and confirm
        Logging.disable_logging(Logging.BelowMinLevel) 
        Experiments.save_experiment(results)
        
        println("✅ Completed: $(conf.name)")
        return true # Success flag

    catch e
        Logging.disable_logging(Logging.BelowMinLevel)
        @error "❌ Failed [$(conf.name)]: $e"
        # If you want to see the stacktrace for debugging:
        # Base.showerror(stdout, e, catch_backtrace())
        return false # Failure flag
    end
end



function run_all_experiments(ds::Data.DataStore, configs::Vector{Experiments.ExperimentConfig})
    println("============================================================")
    println(" Starting Experiment Suite")
    println(" > Total Models: $(length(configs))")
    println("============================================================")

    # 1. Create the tasks (Pairing the same DS with each config)
    tasks = ExperimentTask.(Ref(ds), configs)

    # 2. Execute them one by one
    results = run_experiment_task.(tasks)

    # 3. Summary
    success_count = sum(results)
    println("\n============================================================")
    println(" Suite Finished: $success_count / $(length(configs)) successful.")
    println("============================================================")
end
