# current_development/layer_two/r01_basic_explore.jl

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

# Include our new Layer 2 files
include("l01_calib_utils.jl")
include("data_pipeline.jl")
# include("models/glm_shift.jl")
include("runner.jl")

# 1. Load DataStore & L1 Experiment (Using your existing code structure)
println("Loading DataStore & Layer 1 Experiment...")
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"



ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
exp_dir = "exp/dev_ireland"


saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])


# ---- verison two
l2_data = build_l2_training_df(exp, ds);
# Display a preview
println(first(l2_data, 5))

# 3. Configure the Layer 2 Recalibration
config = CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = PureLogitShift(), 
    target_markets = [:over_15, :over_25, :over_35], # Or whatever you are calibrating
    min_history_splits = 8,   
    max_history_splits = 0,   
    time_decay_half_life = nothing 
)

# 4. Run the Backtest
results = run_calibration_backtest(l2_data, config);

# 5. Quick Analysis: Did Calibration Help?
quick_analysis(results)



#=
target_markets = [:over_15, :over_25, :over_35], # Or whatever you are calibrating
julia> # 5. Quick Analysis: Did Calibration Help?
       quick_analysis(results)

=== Calibration Results ===
Raw L1 LogLoss:    0.6134
Calib L2 LogLoss:  0.612
Improvement:       0.0014

=== Brier Score Results ===
Raw L1 Brier:    0.2119
Calib L2 Brier:  0.2114
Improvement:     0.0005
=#


#=
target_markets = [:over_05, :over_15, :over_25, :over_35], # Or whatever you are calibrating

julia> quick_analysis(results)

=== Calibration Results ===
Raw L1 LogLoss:    0.52
Calib L2 LogLoss:  0.5196
Improvement:       0.0004

=== Brier Score Results ===
Raw L1 Brier:    0.1742
Calib L2 Brier:  0.1738
Improvement:     0.0003
=#



history_dict = results.fitted_models_history[:btts_yes];

println("--- Logit Shift Drift Over Time ---")
# Sort by split to view chronologically
for split in sort(collect(keys(history_dict)))
    model_for_split = history_dict[split]
    shift_val = round(model_for_split.C_shift, digits=4)
    
    println("Split $split  |  C_shift: $shift_val")
end


# ---- check fit 
# 1. Grab the specific GLM model from your history
# (Change the split string to match your latest split, e.g., "25/26-08")
latest_split_key = sort(collect(keys(results.fitted_models_history[:over_25])))[end]
latest_fitted_shift = results.fitted_models_history[:over_25][latest_split_key]
glm_obj = latest_fitted_shift.model

# 2. Print the beautiful GLM Coefficient Table (includes Coef, Std. Error, z, Pr(>|z|), and 95% CI)
using StatsBase
println("\n=== Fit Statistics for Split: $latest_split_key ===")
display(coeftable(glm_obj))

# 3. Or programmatically extract specific metrics if you want to plot them!
p_val = coeftable(glm_obj).cols[4][1] # Column 4 is usually the p-value
z_stat = coeftable(glm_obj).cols[3][1] # Column 3 is usually the z-statistic

println("\nP-Value: ", round(p_val, digits=5))
if p_val < 0.05
    println("Verdict: The Layer 1 bias is statistically significant (p < 0.05).")
else
    println("Verdict: The Layer 1 bias is NOT statistically significant. Shift might be noise.")
end

# ==========================================
# 6. Native Signal Processing & ROI Analysis
# ==========================================
println("\n=== Generating Native PPDs ===")
raw_ppd, calib_ppd = get_ppd_for_raw_and_calib(ds, exp, results);

# 3. Define your native Signal Strategy
# (Assumes you have this imported from BayesianFootball.Signals)
min_edge = 0.00

signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]
println("Running Signals on Raw PPD...")
raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd, ds.odds, signals; odds_column=:odds_close);
println("Running Signals on Calibrated PPD...")
calib_sig_result = BayesianFootball.Signals.process_signals(calib_ppd, ds.odds, signals; odds_column=:odds_close);

# 5. Print out the final comparison
# 5. Print out the final comparison
display_result(raw_sig_result, calib_sig_result)



#=
target_markets = [:over_15, :over_25, :over_35], # Or whatever you are calibrating
=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 2703
  Bets Placed:  505 (Active Rate: 18.68%)
  Win Rate:     47.72%
  Total Stake:  13.91 units
  Total PnL:    +1.46 units
  ROI:          +10.48%

[CALIBRATED L2 MODEL]
  Seen Markets: 2703
  Bets Placed:  1060 (Active Rate: 39.22%)
  Win Rate:     52.92%
  Total Stake:  46.99 units
  Total PnL:    +1.75 units
  ROI:          +3.73%
=#


#=
display_result(raw_sig_result, calib_sig_result)

=== L3 Strategy: Bayesian Kelly (Edge > 0.04) ===

[RAW L1 MODEL]
  Seen Markets: 3618
  Bets Placed:  238 (Active Rate: 6.58%)
  Win Rate:     51.26%
  Total Stake:  16.21 units
  Total PnL:    +2.83 units
  ROI:          +17.46%

[CALIBRATED L2 MODEL]
  Seen Markets: 3618
  Bets Placed:  560 (Active Rate: 15.48%)
  Win Rate:     51.96%
  Total Stake:  48.48 units
  Total PnL:    +3.30 units
  ROI:          +6.81%

=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 3618
  Bets Placed:  637 (Active Rate: 17.61%)
  Win Rate:     48.67%
  Total Stake:  19.39 units
  Total PnL:    +2.54 units
  ROI:          +13.12%

[CALIBRATED L2 MODEL]
  Seen Markets: 3618
  Bets Placed:  1312 (Active Rate: 36.26%)
  Win Rate:     53.05%
  Total Stake:  58.11 units
  Total PnL:    +2.55 units
  ROI:          +4.39%


=#





#=
=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 1524
  Bets Placed:  331 (Active Rate: 21.72%)
  Win Rate:     67.67%
  Total Stake:  15.31 units
  Total PnL:    +0.74 units
  ROI:          +4.83%

[CALIBRATED L2 MODEL]
  Seen Markets: 1524
  Bets Placed:  506 (Active Rate: 33.20%)
  Win Rate:     68.18%
  Total Stake:  25.68 units
  Total PnL:    +1.74 units
  ROI:          +6.78%
=#

#=
=== L3 Strategy: Bayesian Kelly (Edge > 0.02) ===

[RAW L1 MODEL]
  Seen Markets: 508
  Bets Placed:  15 (Active Rate: 2.95%)
  Win Rate:     86.67%
  Total Stake:  1.27 units
  Total PnL:    +0.22 units
  ROI:          +17.12%

[CALIBRATED L2 MODEL]
  Seen Markets: 508
  Bets Placed:  105 (Active Rate: 20.67%)
  Win Rate:     83.81%
  Total Stake:  10.82 units
  Total PnL:    +1.02 units
  ROI:          +9.43%
=#


#=
=== L3 Strategy: Bayesian Kelly (Edge > 0.00) ===

[RAW L1 MODEL]
  Seen Markets: 508
  Bets Placed:  51 (Active Rate: 10.04%)
  Win Rate:     88.24%
  Total Stake:  1.67 units
  Total PnL:    +0.30 units
  ROI:          +17.76%

[CALIBRATED L2 MODEL]
  Seen Markets: 508
  Bets Placed:  194 (Active Rate: 38.19%)
  Win Rate:     79.90%
  Total Stake:  11.78 units
  Total PnL:    +0.90 units
  ROI:          +7.67%
=#



display_edge_threshold_analysis(calib_ppd, ds)
display_edge_threshold_analysis(raw_ppd, ds)




# ---- Looking into the l2 data 

  matches_df = ds.matches[:, [:match_id, :match_date, :season, :match_month, :home_score, :away_score]]

    matches_df.split_id = [string(r.season, "-", lpad(r.match_month, 2, "0")) for r in eachrow(matches_df)]

matches_df


#=
julia> matches_df
1950×7 DataFrame
  Row │ match_id  match_date  season   match_month  home_score  away_score  split_id 
      │ Int32?    Date        String?  Int64        Int32?      Int32?      String   
──────┼──────────────────────────────────────────────────────────────────────────────
    1 │  8824033  2020-10-17  20/21              1           3           1  20/21-01
    2 │  8824035  2020-10-17  20/21              1           2           0  20/21-01
    3 │  8824032  2020-10-17  20/21              1           1           3  20/21-01
    4 │  8824034  2020-10-17  20/21              1           0           0  20/21-01
    5 │  8824045  2020-10-17  20/21              1           1           0  20/21-01
    6 │  8824073  2020-10-24  20/21              1           1           1  20/21-01
    7 │  8824070  2020-10-24  20/21              1           0           2  20/21-01
=#

