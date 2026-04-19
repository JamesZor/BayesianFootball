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
    target_markets = [:over_15],
    min_history_splits = 10,   # Wait for 4 months of data before starting L2
    max_history_splits = 0,   # 0 = expanding window (use all available history)
    # time_decay_half_life = 90.0 # Older matches matter less (half-life of 90 days)
    time_decay_half_life = 1000000 # Older matches matter less (half-life of 90 days)
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


# ==========================================
# 6. Native Signal Processing & ROI Analysis
# ==========================================
println("\n=== Generating Native PPDs ===")


# 1. Get the Raw PPD 
latents = Experiments.extract_oos_predictions(ds, exp)
raw_ppd = Predictions.model_inference(latents) 

# 2. Convert L2 Output back into a native PPD
calib_ppd = create_calibrated_ppd(raw_ppd, results)

# 3. Align the Raw PPD to exactly match the scope of the Calibrated PPD
# semijoin keeps only the rows in raw_ppd.df that perfectly match the keys in calib_ppd.df
aligned_raw_df = semijoin(
    raw_ppd.df, 
    calib_ppd.df, 
    on=[:match_id, :market_name, :market_line, :selection]
)

# 4. Overwrite raw_ppd with the aligned version
raw_ppd = BayesianFootball.Predictions.PPD(
    aligned_raw_df, 
    raw_ppd.model, 
    raw_ppd.config 
)

println("Raw PPD rows:   ", nrow(raw_ppd.df))
println("Calib PPD rows: ", nrow(calib_ppd.df))
    

# 3. Define your native Signal Strategy
# (Assumes you have this imported from BayesianFootball.Signals)
min_edge = 0.00
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

println("Running Signals on Raw PPD...")
raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close)

println("Running Signals on Calibrated PPD...")
calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close)

# 4. Helper to Compute PnL from your signal ledger
function summarize_ledger(ledger_df)
    # Compute PnL for placed bets
    ledger_df.pnl = map(eachrow(ledger_df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner == 1.0 # Or true, depending on your type
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    total_stake = sum(ledger_df.stake)
    total_pnl = sum(ledger_df.pnl)
    roi = total_stake > 0 ? (total_pnl / total_stake) * 100 : 0.0
    bets = count(x -> x > 0, ledger_df.stake)
    
    return bets, total_stake, total_pnl, roi
end

# 5. Print out the final comparison

function display_result(raw_sig_result, calib_sig_result)
    raw_bets, raw_staked, raw_pnl, raw_roi = summarize_ledger(raw_sig_result.df)
    calib_bets, calib_staked, calib_pnl, calib_roi = summarize_ledger(calib_sig_result.df)

    @printf("\n=== L3 Strategy: Bayesian Kelly (Edge > %.2f) ===\n", min_edge)

    @printf("\n[RAW L1 MODEL]\n")
    @printf("  Bets Placed: %d\n", raw_bets)
    @printf("  Total Stake: %.2f units\n", raw_staked)
    @printf("  Total PnL:   %+.2f units\n", raw_pnl)
    @printf("  ROI:         %+.2f%%\n", raw_roi)

    @printf("\n[CALIBRATED L2 MODEL]\n")
    @printf("  Bets Placed: %d\n", calib_bets)
    @printf("  Total Stake: %.2f units\n", calib_staked)
    @printf("  Total PnL:   %+.2f units\n", calib_pnl)
    @printf("  ROI:         %+.2f%%\n", calib_roi)
end



display_result(raw_sig_result, calib_sig_result)



# ---- verison two


l2_data = build_l2_training_df(exp, ds)
# Display a preview
println(first(l2_data, 5))

# 3. Configure the Layer 2 Recalibration
config = CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = PureLogitShift(), 
    target_markets = [:over_15], # Or whatever you are calibrating
    min_history_splits = 20,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)

# 4. Run the Backtest
println("Starting L2 Backtest...")
results = run_calibration_backtest(l2_data, config)

# 5. Quick Analysis: Did Calibration Help?
# Let's compare the LogLoss of Raw PPD vs Calibrated PPD
using MLJBase: log_loss # Or write a quick logloss function

y_true = results.oos_predictions.outcome_hit;
p_raw = results.oos_predictions.raw_prob;
p_calib = Float64.(results.oos_predictions.calib_prob);

ll_raw = custom_logloss(y_true, p_raw)
ll_calib = custom_logloss(y_true, p_calib)

println("\n=== Calibration Results ===")
println("Raw L1 LogLoss:    ", round(ll_raw, digits=4))
println("Calib L2 LogLoss:  ", round(ll_calib, digits=4))
println("Improvement:       ", round(ll_raw - ll_calib, digits=4))


# ==========================================
# 6. Native Signal Processing & ROI Analysis
# ==========================================
println("\n=== Generating Native PPDs ===")


# 1. Get the Raw PPD 
latents = Experiments.extract_oos_predictions(ds, exp)
raw_ppd = Predictions.model_inference(latents) 

# 2. Convert L2 Output back into a native PPD
calib_ppd = create_calibrated_ppd(raw_ppd, results)

# 3. Align the Raw PPD to exactly match the scope of the Calibrated PPD
# semijoin keeps only the rows in raw_ppd.df that perfectly match the keys in calib_ppd.df
aligned_raw_df = semijoin(
    raw_ppd.df, 
    calib_ppd.df, 
    on=[:match_id, :market_name, :market_line, :selection]
)

# 4. Overwrite raw_ppd with the aligned version
raw_ppd = BayesianFootball.Predictions.PPD(
    aligned_raw_df, 
    raw_ppd.model, 
    raw_ppd.config 
)

println("Raw PPD rows:   ", nrow(raw_ppd.df))
println("Calib PPD rows: ", nrow(calib_ppd.df))
    

# 3. Define your native Signal Strategy
# (Assumes you have this imported from BayesianFootball.Signals)
min_edge = 0.015
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

println("Running Signals on Raw PPD...")
raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close)

println("Running Signals on Calibrated PPD...")
calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close)

# 5. Print out the final comparison
# 4. Helper to Compute PnL from your signal ledger
function summarize_ledger(ledger_df)
    # Compute PnL for placed bets
    ledger_df.pnl = map(eachrow(ledger_df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner == 1.0 # Or true, depending on your type
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    # Financial Metrics
    total_stake = sum(ledger_df.stake)
    total_pnl = sum(ledger_df.pnl)
    roi = total_stake > 0 ? (total_pnl / total_stake) * 100 : 0.0
    
    # Volume & Hit Metrics
    seen = nrow(ledger_df)
    bets = count(x -> x > 0, ledger_df.stake)
    active_rate = seen > 0 ? (bets / seen) * 100 : 0.0
    
    # Win Rate (only counting bets we actually placed)
    won_bets = count(r -> r.stake > 0.0 && r.is_winner == 1.0, eachrow(ledger_df))
    win_rate = bets > 0 ? (won_bets / bets) * 100 : 0.0
    
    return (
        bets = bets, 
        total_stake = total_stake, 
        total_pnl = total_pnl, 
        roi = roi, 
        seen = seen, 
        active_rate = active_rate, 
        win_rate = win_rate
    )
end

# 5. Print out the final comparison
function display_result(raw_sig_result, calib_sig_result; min_edge=min_edge)
    raw = summarize_ledger(raw_sig_result.df)
    calib = summarize_ledger(calib_sig_result.df)

    @printf("\n=== L3 Strategy: Bayesian Kelly (Edge > %.2f) ===\n", min_edge)

    @printf("\n[RAW L1 MODEL]\n")
    @printf("  Seen Markets: %d\n", raw.seen)
    @printf("  Bets Placed:  %d (Active Rate: %.2f%%)\n", raw.bets, raw.active_rate)
    @printf("  Win Rate:     %.2f%%\n", raw.win_rate)
    @printf("  Total Stake:  %.2f units\n", raw.total_stake)
    @printf("  Total PnL:    %+.2f units\n", raw.total_pnl)
    @printf("  ROI:          %+.2f%%\n", raw.roi)

    @printf("\n[CALIBRATED L2 MODEL]\n")
    @printf("  Seen Markets: %d\n", calib.seen)
    @printf("  Bets Placed:  %d (Active Rate: %.2f%%)\n", calib.bets, calib.active_rate)
    @printf("  Win Rate:     %.2f%%\n", calib.win_rate)
    @printf("  Total Stake:  %.2f units\n", calib.total_stake)
    @printf("  Total PnL:    %+.2f units\n", calib.total_pnl)
    @printf("  ROI:          %+.2f%%\n", calib.roi)
end

display_result(raw_sig_result, calib_sig_result)



function dis()
println("\n=== Calibrated Model: Edge Threshold Analysis ===")
println("Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%")
println("---------------------------------------------------------")

for edge_pct in [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
    
    # Create temporary signals with the new edge
    temp_signals = [BayesianFootball.Signals.BayesianKelly(min_edge=edge_pct)]
    
    # Process just the calibrated PPD
    temp_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, temp_signals; odds_column=:odds_close)
    
    # Summarize
    metrics = summarize_ledger(temp_result.df)
    
    @printf("%5.1f | %4d | %7.2f | %6.2f | %6.2f | %+.2f | %+.2f%%\n", 
            edge_pct * 100, metrics.bets, metrics.active_rate, metrics.win_rate, 
            metrics.total_stake, metrics.total_pnl, metrics.roi)
end

end 


dis()
