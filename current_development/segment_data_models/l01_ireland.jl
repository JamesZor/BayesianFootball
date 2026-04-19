
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



# ----------------------------------------------------
# --- metric scoring for the models 
# ----------------------------------------------------

function evaluate_batch(metric::Evaluation.AbstractScoringRule, results_array, ds; label="Batch Evaluation")
    # Dynamically grab the metric name (e.g., "crps", "rqr", "miq") and uppercase it for display
    metric_name = uppercase(Evaluation.get_metric_method_name(metric))
    
    println("\n============================================================")
    println(" 🚀 Running Batch $metric_name Evaluation: $label")
    println("============================================================")

    flat_rows = []

    # Loop through all provided experiments
    for (i, exp) in enumerate(results_array)
        model_name = exp.config.name
        print("[$i/$(length(results_array))] Evaluating: $(model_name) ... ")
        
        try
            # Computes whatever metric was passed in
            metric_data = Evaluation.compute_metric(metric, exp, ds)
            
            # Flattens the nested structs into a single row using your unroller
            flat_row = Evaluation.to_dataframe_row(exp, metric_data)
            
            push!(flat_rows, flat_row)
            println("✅ Done")
        catch e
            println("❌ Failed")
            @warn "Error evaluating $model_name on $metric_name: $e"
        end
    end

    # Build the Master DataFrame
    master_df = DataFrame(flat_rows)

    if nrow(master_df) > 0
        # Sort by model name to keep it organized
        if hasproperty(master_df, :model)
            sort!(master_df, :model)
        end

        println("\n============================================================")
        println(" 📊 MASTER $metric_name COMPARISON: $label")
        if metric_name == "CRPS"
             println(" Note: LOWER is BETTER")
        elseif metric_name == "MIQ"
             println(" Note: Look for Positive mean_gaps and low p_values for edge.")
        end
        println("============================================================")
        display(master_df)
    else
        println("⚠️ No results successfully evaluated.")
    end
    
    return master_df
end

miq_df = evaluate_batch(Evaluation.MIQ(), loaded_results_, ds, label="Baseline Models")
