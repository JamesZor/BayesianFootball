# current_development/bench_models/l00_main_utils.jl

using Revise
using BayesianFootball

using LibPQ

using DataFrames
using ThreadPinning
pinthreads(:cores)



# ========================================
#  Stage 1 - Training the model
# ========================================

# --- 1. find the lasts month
#  Find the cv_configs that get the current split 
struct CVParameters 
  target_season::String
  warmup_period::Integer
end


"""
  helper function for the Data.segment type, to get the target_season 
"""
function get_target_seasons_string(segment::Data.DataTournemantSegment) 
    # none - place holder 
    println("Placeholder for the type: $(segment)")
    return 
end  

get_target_seasons_string(::Data.Ireland)=[
 # "2021", # skipped as we need the history
 # "2022",
 "2023",
 "2024",
 "2025",
 "2026",
]

# --- 2. Running the experiment
struct DSExperimentSettings 
  ds::Data.DataStore
  label::String
  save_dir::String
  target_season::Vector{<:String}
end

struct ExperimentTask
    ds::Data.DataStore
    config::Experiments.ExperimentConfig
end


"""
  Wrapper verison of the func to allow for the DSExperimentSettings type to 
  be used as a parameter.
"""
function create_experiment_tasks(es::DSExperimentSettings)
    return create_experiment_tasks(es.ds, es.label, es.save_dir, es.target_season)
end



function create_experiment_tasks(ds::Data.DataStore, label::String, save_dir::String, target_seasons::Vector{<:String} )

    # 1. Define the shared parts (CV and Training)
        cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 2,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = Samplers.NUTSConfig(
    500, # Number of samples for each chain
    2,   # Number of chains
    150, # Number of warm up steps 
    0.65,# Accept rate  [0,1]
    10,  # Max tree depth
    Samplers.UniformInit(-1, 1), # Interval for starting a chain 
    false,   # show_progress (We use the Global Logger instead)
    # false, # Display progress bar setting
  )
    train_cfg = BayesianFootball.Training.Independent(
    parallel=true,
    max_concurrent_splits=8
  )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    # 2. Build the list of Configs
    configs = [
        # Experiments.ExperimentConfig(
        #     name = "$(label)_01_baseline",
        #     model = Models.PreGame.AblationStudy_NB_baseLine(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        #
        # Experiments.ExperimentConfig(
        #     name = "$(label)_02_home_hierarchy",
        #     model = Models.PreGame.AblationStudy_NB_home_hierarchy(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        #
        # # --- Model 4: Lean Target (Team HA + Team Dispersion) ---
        # Experiments.ExperimentConfig(
        #     name = "$(label)_03_team_dispersion",
        #     model = Models.PreGame.AblationStudy_NB_team_dispersion(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        #
        # # --- Model 5: Baseline + Monthly Expectations ---
        # Experiments.ExperimentConfig(
        #     name = "$(label)_04_monthly_mu",
        #     model = Models.PreGame.AblationStudy_NB_baseline_month_mu(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        # Experiments.ExperimentConfig(
        #     name = "$(label)_05_monthly_r",
        #     model = Models.PreGame.AblationStudy_NB_baseline_month_r(),
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        # Experiments.ExperimentConfig(
        #     name = "$(label)_06_baseline_HA_r",
        #     model = Models.PreGame.AblationStudy_NB_baseline_HA_r(), 
        #     splitter = cv_config,
        #     training_config = training_config,
        #     save_dir = save_dir
        # ),
        Experiments.ExperimentConfig(
            name = "$(label)_06_HA_r_H_hierarchy",
            model = Models.PreGame.AblationStudy_NB_HA_r_home_hierarchy(), 
            splitter = cv_config,
            training_config = training_config,
            save_dir = save_dir
        ),
    ]

    # 3. THE "SMART" BIT: 
    # Wrap every config with the DS into an ExperimentTask
    # We use Ref(ds) so it doesn't try to "iterate" the DataStore
    return ExperimentTask.(Ref(ds), configs)
end


function run_experiment_task(task::ExperimentTask)
    conf = task.config
    println("Running: $(conf.name)")

    try
        # 2. Execute
        results = Experiments.run_experiment(task.ds, conf)

        # 3. Re-enable logging to save and confirm
        Experiments.save_experiment(results)
        
        return true # Success flag

    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        # If you want to see the stacktrace for debugging:
        # Base.showerror(stdout, e, catch_backtrace())
        return false # Failure flag
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

  if isempty(loaded_results)
      error("No results loaded! Did you run runner.jl?")
  end

  return loaded_results

end




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




