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
run_experiment_task.(task_xg)
# run_experiment_task.([task_xg[1], task_market_xg[1]])


inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
dyn_cfg   = PreGame.TimeDecayDynamics(days_half_life = 180)
kaph_cfg   = PreGame.HierarchicalTeamKappa()

# 3. Model A: DynamicXGTimeDecayModel
model_xg = PreGame.DynamicXGTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kaph_cfg
)

# 4. Model B: DynamicMarketXGTimeDecayModel
model_market_xg = PreGame.DynamicMarketXGTimeDecayModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kaph_cfg,
    market_weight        = 1.0
)

# 5. Create Tasks
task_xg = create_experiment_tasks(ds, model_xg, "src_h_xg_decay_test", save_dir, ["2026"])
task_market_xg = create_experiment_tasks(ds, model_market_xg, "src_h_market_xg_decay_test", save_dir, ["2026"])


run_experiment_task.([task_xg[1], task_market_xg[1]])



# run_experiment_task(task_xg[1])
# run_experiment_task(task_market_xg[1])

println("\nxG Decay test runners initialized.")
println("1. DynamicXGTimeDecayModel")
println("2. DynamicMarketXGTimeDecayModel (Market Weight: 0.5)")



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


saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders)


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

μ

params_to_track_xg = [
Symbol("inter.μ"),
Symbol("disp.log_r"), 
Symbol("disp.δ_r_home"),
Symbol("ha.γ_base"),
Symbol("ha.σ_γ"),
Symbol("ha.γ_team_raw[1]"),
Symbol("ha.γ_team_raw[2]"),
Symbol("ha.γ_team_raw[3]")
    ]

parameters        = ν_xg, σ_market, inter.μ, disp.log_r, disp.δ_r_home, ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], ha.γ_team_raw[12], kap.κ_
base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_team_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], kap.κ_team_raw[12], dyn.σ_a, dyn.σ_d, dyn.raw_a[1], dyn.raw_a[2], dyn.raw_a[3], dyn.raw_a[
4], dyn.raw_a[5], dyn.raw_a[6], dyn.raw_a[7], dyn.raw_a[8], dyn.raw_a[9], dyn.raw_a[10], dyn.raw_a[11], dyn.raw_a[12], dyn.raw_d[1], dyn.raw_d[2], dyn.raw_d[3], dyn.raw_d[4], dyn.raw_d[5], dyn.raw_d[6], dyn.raw_d[7], dyn.raw_d[8], dyn.raw_d[9], dyn.raw_d[10], dyn.raw_d[11], dyn.raw_d[12]                            

expr = loaded_results[1]

chain_fold_3 = expr.training_results[3][1]
describe(chain_fold_3)

all_chains = [res[1] for res in expr.training_results] 
# 3. Generate the Stability Report
stability_df_xg = check_parameter_stability(all_chains, params_to_track_xg)

display(stability_df_xg)


