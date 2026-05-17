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
    interception_config  = inter_season_cfg,
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


# -----

saved_folders = Experiments.list_experiments(save_dir; data_dir="")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results[[2]], 
  [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end

expr1 = loaded_results[1]
expr2 = loaded_results[2]


# -----

using DataFrames
using Statistics

function check_parameter_stability(chains::Vector, target_params::Vector{Symbol})
    # Initialize an empty DataFrame
    df = DataFrame(Fold = Int[])
    
    # FIX: Explicitly tell Julia these columns can contain missing values
    for p in target_params
        df[!, Symbol(string(p), "_mean")] = Union{Missing, Float64}[]
        df[!, Symbol(string(p), "_std")]  = Union{Missing, Float64}[]
    end
    
    # Iterate through each fold's MCMCChain
    for (fold_idx, chain) in enumerate(chains)
        row_dict = Dict{Symbol, Any}(:Fold => fold_idx)
        
        for p in target_params
            # Check if the parameter exists in the chain
            if p in keys(chain)
                samples = vec(chain[p]) 
                row_dict[Symbol(string(p), "_mean")] = mean(samples)
                row_dict[Symbol(string(p), "_std")]  = std(samples)
            else
                row_dict[Symbol(string(p), "_mean")] = missing
                row_dict[Symbol(string(p), "_std")]  = missing
            end
        end
        
        push!(df, row_dict) # This will now safely accept the missing values!
    end
    
    return df
end



params_to_track_xg = [
    Symbol("inter.μ"), 
    Symbol("σ_market"), # NEW: Variance/spread of team conversion abilities
    Symbol("disp.log_r"), 
    Symbol("ha.γ_global"),
    :ν_xg,       # NEW: xG Gamma shape parameter
    Symbol("kap.κ_base"),
    Symbol("kap.σ_κ"),
    Symbol("dyn.α.σ₀"), 
    Symbol("dyn.α.σₛ"), 
    Symbol("dyn.α.σₖ"),
    Symbol("dyn.β.σ₀"), 
    Symbol("dyn.β.σₛ"), 
    Symbol("dyn.β.σₖ")
]

