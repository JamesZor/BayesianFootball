
using Revise
using BayesianFootball

using DataFrames
using ThreadPinning
pinthreads(:cores)



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

function create_experiment_tasks(ds::Data.DataStore, label::String, save_dir::String)

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = ["2023", "2024", "2025", "2026"],
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

  if isempty(loaded_results)
      error("No results loaded! Did you run runner.jl?")
  end

  return loaded_results

end




# ---- 3 back test ----


function run_simple_backtest(loaded_results::Vector{BayesianFootball.Experiments.ExperimentResults}, ds::Data.DataStore) 

    baker = BayesianFootball.Signals.BayesianKelly()
    # flat_strat = BayesianFootball.Signals.FlatStake(0.05)
    # my_signals = [baker, flat_strat]
    my_signals = [baker]

    ledger = BayesianFootball.BackTesting.run_backtest(
        ds, 
        loaded_results, 
        my_signals; 
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG
    )

    return BayesianFootball.BackTesting.generate_tearsheet(ledger), ledger

end


function display_tearsheet_by_market(tearsheet::AbstractDataFrame) 
    model_names = unique(tearsheet.selection)
    model_names = model_names[1:15]
    for m_name in model_names
        println("\nStats for: $m_name")
        sub= subset(tearsheet, :selection => ByRow(isequal(m_name)))
        show(sub)
    end
end
