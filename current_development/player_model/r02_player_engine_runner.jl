# current_development/player_model/r02_player_engine_runner.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Dates
using Distributions
using Turing

const PreGame = BayesianFootball.Models.PreGame

# Include our prototype loaders
include("l00_player_features.jl")
include("l01_player_dynamics.jl")
include("l02_player_engine.jl")

# --- Helper Functions ---

struct ExperimentTask
    ds::BayesianFootball.Data.DataStore
    config::BayesianFootball.Experiments.ExperimentConfig
end

function create_experiment_tasks(ds::BayesianFootball.Data.DataStore, model, label::String, save_dir::String)
    cv_config = BayesianFootball.Data.GroupedCVConfig(
        tournament_groups = [BayesianFootball.Data.tournament_ids(ds.segment)],
        target_seasons = ["2026"],
        history_seasons = 1,
        dynamics_col = :match_month,
        warmup_period = 0,
        stop_early = true
    )

    sampler_conf = BayesianFootball.Samplers.NUTSConfig(
        400, # samples
        4,   # chains
        200, # adapt
        0.65,
        10,  
        BayesianFootball.Samplers.UniformInit(-1, 1),
        false,
    )

    train_cfg = BayesianFootball.Training.Independent(
        parallel=true,
        max_concurrent_splits=2
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
        return results
    catch e
        @error "❌ Failed [$(conf.name)]: $e"
        rethrow(e)
    end
end

# --- Runner Logic ---

# 1. Load Data
println("Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
save_dir::String = "./data/test_player_model/"

# 2. Instantiate Model with PositionalPlayerDynamics
println("Initializing Model Configs...")
inter_cfg = PreGame.GlobalInterception()
dyn_cfg   = PositionalPlayerDynamics() # Our NEW component
disp_cfg  = PreGame.GlobalDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

model = DynamicMarketXGPlayerModel(
    interception_config  = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    market_weight        = 1.0
)

# 3. Create and Run Tasks
println("\nCreating Experiment Tasks...")
tasks = create_experiment_tasks(ds, model, "player_model_test", save_dir)

println("\nExecuting Test Run...")
results = run_experiment_task(tasks[1])


# 4. Verification: Extract Latents
println("\n>>> Verifying Parameter Extraction...")
try
    # We use the results directly
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, results)
    println("✅ Latents extracted successfully.")
    println("Number of predicted matches: ", nrow(latents.df))
    
    # Check a sample match
    sample_match = latents.df[1, :]
    println("\nSample Match ID: ", sample_match.match_id)
    println("λ_h (mean): ", round(mean(sample_match.λ_h), digits=3))
    println("λ_a (mean): ", round(mean(sample_match.λ_a), digits=3))
    
catch e
    @error "❌ Parameter extraction failed: $e"
    rethrow(e)
end


latents = BayesianFootball.Experiments.extract_oos_predictions(ds, ll[1]);
println("✅ Latents extracted successfully.")
println("Number of predicted matches: ", nrow(latents.df));

# Check a sample match
sample_match = latents.df[1, :]
println("\nSample Match ID: ", sample_match.match_id)
println("λ_h (mean): ", round(mean(sample_match.λ_h), digits=3))
println("λ_a (mean): ", round(mean(sample_match.λ_a), digits=3))


println("\n✅ Phase 3 complete. Model engine is fully functional.")

ll = [ loaded_results_[1], loaded_results_[2], loaded_results[1]]
using BayesianFootball.Evaluation

# 1. Define the metrics you want to run
metrics = [
    Evaluation.RQR(), 
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.GLMEdge()
]

# 2. Run everything in one go
master_eval_df = Evaluation.evaluate_experiments(metrics, ll, ds)

# 3. View clean summaries for specific families
Evaluation.display_summary_metric(master_eval_df, :rqr)
Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)



ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  ll, 
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

