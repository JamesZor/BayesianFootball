# current_development/time_decayed_models/r04_test_player_models.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features

# --- Helper Functions ---

struct ExperimentTask
    ds::BayesianFootball.Data.DataStore
    config::BayesianFootball.Experiments.ExperimentConfig
end

function create_experiment_tasks(ds::BayesianFootball.Data.DataStore, model, label::String, save_dir::String, target_seasons::Vector{<:String} )
    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = 3,
        dynamics_col = :match_week,
        warmup_period = 0,
        stop_early = true # Just run one fold for quick validation
    )

    sampler_conf = BayesianFootball.Samplers.NUTSConfig(
        500, # Reduced for quick testing
        2,   
        200,  
        0.65,
        10,  
        BayesianFootball.Samplers.UniformInit(-1, 1),
        false,
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=4
    )
    training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    configs = [
        BayesianFootball.Experiments.ExperimentConfig(
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
        results = BayesianFootball.Experiments.run_experiment(task.ds, conf)
        BayesianFootball.Experiments.save_experiment(results)
        println("✅ Success: $(conf.name)")
        return true
    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        return false
    end
end

# --- Runner Logic ---

println("--- Validating DynamicMarketXGPlayerModel ---")

# 1. Load Data
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_src_player_models/"

# 2. Shared Config Components
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
# Player model requires PositionalPlayerDynamics
dyn_cfg   = PreGame.PositionalPlayerDynamics() 
kap_cfg   = PreGame.GlobalKappa()

# 3. Model A: Last Value Tracker
tracker_lv = Features.LastValueTracker()
feature_cfg_lv = Features.PlayerRatingsFeature(tracker_lv)

model_lv = PreGame.DynamicMarketXGPlayerModel(
    interception_config  = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_lv,
    market_weight        = 1.0
)

# 4. Model B: Bayesian Tracker
# Using reasonable parameters from the previous grid search
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

model_bayes = PreGame.DynamicMarketXGPlayerModel(
    interception_config  = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)

# 5. Create Tasks (Testing on 2026 season)
task_lv = create_experiment_tasks(ds, model_lv, "player_model_last_value", save_dir, ["2026"])
task_bayes = create_experiment_tasks(ds, model_bayes, "player_model_bayesian", save_dir, ["2026"])

# 6. Execute 
println("\n[INFO] Starting execution. This will run 1 fold per model for validation.")
run_experiment_task.(task_lv)
run_experiment_task.(task_bayes)

println("\nValidation complete. Check $save_dir for results.")




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


saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);

expr = loaded_results[1]
chain_fold_1 = expr.training_results[1][1]
chain_fold_2 = expr.training_results[2][1]
chain_fold_3 = expr.training_results[3][1]
expr


config = expr.config
# 1. Reconstruct Context using the NEW Relational Pipeline
boundaries_with_meta = Data.create_id_boundaries(ds, config.splitter)
feature_sets = Features.create_features(
    boundaries_with_meta, 
    ds, 
    config.model, 
    config.splitter.dynamics_col
)


#=
julia> expr.training_results[2][2]
GroupedSplit(Tourns: [79], Season: 2026, Week: 1, Hist: 3)
=#

#=
julia> chain_fold_2 = expr.training_results[2][1]
Chains MCMC chain (500×42×2 Array{Float64, 3}):

Iterations        = 201:1:700
Number of chains  = 2
Samples per chain = 500
Wall duration     = 591.92 seconds
Compute duration  = 1147.62 seconds
parameters        = ν_xg, σ_market, inter.μ, disp.log_r, disp.δ_r_home, ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], ha.γ_team_raw[12], kap.κ_global, p_dyn.w_G_att, p_dyn.w_D_att, p_dyn.w_M_att, p_dyn.w_F_att, p_dyn.w_G_def, p_dyn.w_D_def, p_dyn.w_M_def, p_dyn.w_F_def
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood
=#

team_mapping = feature_sets[1][1][:team_map]

#=
julia> team_mapping = feature_sets[1][1][:team_map]
Dict{String, Int64} with 12 entries:
  "sligo-rovers"              => 9
  "dundalk-fc"                => 5
  "drogheda-united"           => 4
  "shamrock-rovers"           => 7
  "derry-city"                => 3
  "st-patricks-athletic"      => 10
  "cork-city"                 => 2
  "bohemian"                  => 1
  "university-college-dublin" => 11
  "waterford-fc"              => 12
  "galway-united"             => 6
  "shelbourne"                => 8
=#



# 1. Load the logic
# include("current_development/time_decayed_models/l05_stability_analysis.jl")

# 2. Get your results (assuming you have 'ds' and 'expr' from your runner)
stability_df = extract_stability_dataframe(ds, expr)

# 3. Analyze Stability
# Example: Check how 'w_G_att' (Goalkeeper Attacking Weight) evolves over weeks
using DataFrames, Statistics
att_stability_D = subset(stability_df, :parameter => p -> p .== "w_D_att")
att_stability_M = subset(stability_df, :parameter => p -> p .== "w_M_att")
att_stability_F = subset(stability_df, :parameter => p -> p .== "w_F_att")



def_stability_D = subset(stability_df, :parameter => p -> p .== "w_D_def")
def_stability_M = subset(stability_df, :parameter => p -> p .== "w_M_def")
def_stability_F = subset(stability_df, :parameter => p -> p .== "w_F_def")

# Example: Check Home Advantage for a specific team
shamrock_ha = subset(stability_df, 
    :parameter => p -> p .== "home_advantage",
    :entity => e -> e .== "shamrock-rovers"
)


using BayesianFootball.Evaluation

# 1. Define the metrics you want to run
metrics = [
    Evaluation.RQR(), 
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]

# 2. Run everything in one go
master_eval_df = Evaluation.evaluate_experiments(metrics, loaded_results, ds)

# 3. View clean summaries for specific families
Evaluation.display_summary_metric(master_eval_df, :rqr)
Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)


ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results, 
  [BayesianFootball.Signals.BayesianKelly()]; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

model_names = unique(tearsheet.selection)

model_names = model_names

for m_name in model_names[1:18]
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    # show(sub)
    show(sub[:, [:model_name, :selection, :opportunities, :bets_placed, :activity_pct, :turnover, :profit, :roi_pct, :win_rate_pct]])
end

