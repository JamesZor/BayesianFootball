# Include the new model
include("models/xgboost_shift.jl")

# Rebuild the dataset so it has the team names
l2_data = build_l2_training_df(exp, ds)

# Configure for XGBoost
config = CalibrationConfig(
    name = "XGBoost_Basic_Features",
    model = XGBoostCalibrator(num_rounds=15, max_depth=2, eta=0.05), 
    target_markets = [:btts_yes], 
    min_history_splits = 8,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)

# Run the backtest!
results = run_calibration_backtest(l2_data, config);


quick_analysis(results)

# Check the PnL exactly as before
raw_ppd, calib_ppd = get_ppd_for_raw_and_calib(ds, exp, results);

min_edge = 0.04
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close);

calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close);

display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

display_edge_threshold_analysis(calib_ppd, ds)
display_edge_threshold_analysis(raw_ppd, ds)


# ------
using XGBoost

# 1. Grab the latest XGBoost model from the history
latest_split = sort(collect(keys(results.fitted_models_history[:over_25])))[end]
latest_model = results.fitted_models_history[:over_25][latest_split]

println("\n=== Feature Importance for Split: $latest_split ===")
# Generate the importance report
importance_report = XGBoost.importance(latest_model.booster)
display(importance_report)


# -------
using Statistics
using StatsFuns: logit

df = results.oos_predictions

# Calculate the Logit Shift XGBoost applied to every single match
eps = 1e-6
raw_logits = logit.(clamp.(df.raw_prob, eps, 1.0 - eps));
calib_logits = logit.(clamp.(Float64.(df.calib_prob), eps, 1.0 - eps));
implied_shifts = calib_logits .- raw_logits;

println("\n=== XGBoost Implied Logit Shifts ===")
println("Mean Shift:   ", round(mean(implied_shifts), digits=4))
println("Median Shift: ", round(median(implied_shifts), digits=4))
println("Min Shift:    ", round(minimum(implied_shifts), digits=4))
println("Max Shift:    ", round(maximum(implied_shifts), digits=4))
println("Std Dev:      ", round(std(implied_shifts), digits=4))

# Let's look at the most extreme shifts!
df.shift = implied_shifts;
extreme_shifts = sort(df, :shift, rev=true);
println("\nTop 5 Most Aggressive Positive Shifts (Model LOVES the Over here):")
display(first(extreme_shifts[:, [:match_date, :home_team, :away_team, :market_line, :raw_prob, :calib_prob, :shift, :outcome_hit]], 5))
