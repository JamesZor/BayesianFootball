# current_development/layer_two/r01_basic_explore.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

# Include our new Layer 2 files
include("l01_calib_utils.jl")
include("data_pipeline.jl")
include("models/glm_shift.jl")
include("runner.jl")

# 1. Load DataStore & L1 Experiment (Using your existing code structure)
println("Loading DataStore & Layer 1 Experiment...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())

exp_dir = "exp/ablation_study"
saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])

# 2. Build the Layer 2 Dataset
# This takes the latents, runs inference, joins odds, and resolves targets
println("Building L2 Data Pipeline...")
l2_data = build_l2_training_df(exp, ds)

# Display a preview
println(first(l2_data, 5))

# 3. Configure the Layer 2 Recalibration
config = CalibrationConfig(
    name = "GLM_Time_Weighted_Shift",
    model = TimeWeightedGLM(use_implied_odds = false), # Try True to use market info!
    # target_markets = [:over_25, :under_25, :home, :away, :draw],
    target_markets = [:over_25],
    min_history_splits = 4,   # Wait for 4 months of data before starting L2
    max_history_splits = 0,   # 0 = expanding window (use all available history)
    time_decay_half_life = 90.0 # Older matches matter less (half-life of 90 days)
)

# 4. Run the Backtest
println("Starting L2 Backtest...")
results = run_calibration_backtest(l2_data, config)

# 5. Quick Analysis: Did Calibration Help?
# Let's compare the LogLoss of Raw PPD vs Calibrated PPD
using MLJBase: log_loss # Or write a quick logloss function

y_true = results.oos_predictions.outcome_hit
p_raw = results.oos_predictions.raw_prob
p_calib = Float64.(results.oos_predictions.calib_prob)

function custom_logloss(y, p)
    p = clamp.(p, 1e-15, 1 - 1e-15)
    return -mean(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
end

ll_raw = custom_logloss(y_true, p_raw)
ll_calib = custom_logloss(y_true, p_calib)

println("\n=== Calibration Results ===")
println("Raw L1 LogLoss:    ", round(ll_raw, digits=4))
println("Calib L2 LogLoss:  ", round(ll_calib, digits=4))
println("Improvement:       ", round(ll_raw - ll_calib, digits=4))
