using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


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


get_target_seasons_string(::Data.Ireland)       = ["2026"]

function create_CVsplit_training_config(ds::Data.DataStore, target_seasons::Vector{<:String})

    # 1. Define the shared parts (CV and Training)
    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 1,
        dynamics_col = :match_biweek,
        warmup_period = 0,
        stop_early = false
    )

    sampler_conf = Samplers.NUTSConfig(
    1000, # n steps
    2,    # n chains
    300,  # warm up steps
    0.65, # acceptance rate
    10,   # Max depth
    Samplers.UniformInit(-1, 1), # init step up 
    :false # show progress bar
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true, max_concurrent_splits=8
    )
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


    return (; cv_cfg=cv_config, training_cfg=training_config)

end



# ==========================================
#  1: Combine Model + Cfgs into an ExperimentTask
# ==========================================
function build_experiment_task(ds::BayesianFootball.Data.DataStore, model, label, save_dir::String, cfgs::NamedTuple)
    # 1. Define where this specific model will save its chains/metrics
    
    # 2. Build the master config
    exp_config = BayesianFootball.Experiments.ExperimentConfig(
        name = label,
        model = model,
        splitter = cfgs.cv_cfg,
        training_config = cfgs.training_cfg,
        save_dir = save_dir
    )
    
    # 3. Return the task ready for the execution pipeline
    return ExperimentTask(ds, exp_config)
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

############
using Distributions

# for running the models
const PreGame = BayesianFootball.Models.PreGame


# inter_cfg = PreGame.GlobalInterception()


inter_season_cfg = PreGame.SeasonalInterception()

# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()

dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)
kap_cfg   = PreGame.HierarchicalTeamKappa() 

## models 

model_gm = PreGame.DynamicMarketGoalsModel(
    interception_config  = inter_season_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_g = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

model_gxg = PreGame.DynamicXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)


model_gxgm = PreGame.DynamicMarketXGModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg
)


save_dir::String = "./data/dev_regime_intercept/ireland/"


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())


cfgs = create_CVsplit_training_config(ds,get_target_seasons_string(ds.segment))

task_g = build_experiment_task(ds, model_g, "goals_biweek_season_mu", save_dir, cfgs)
# task_gm = build_experiment_task(ds, model_gm, "goals_market_biweek", save_dir, cfgs)
# task_gxg = build_experiment_task(ds, model_gxg, "goals_xg_biweek", save_dir, cfgs)
# task_gxgm = build_experiment_task(ds, model_gxgm, "goals_xg_market_biweek", save_dir, cfgs)
#
all_task = [task_g]

run_experiment_task.(all_task)

