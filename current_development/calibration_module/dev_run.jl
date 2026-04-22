using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)

using Statistics
using Dates
using Printf


using GLM
using StatsFuns: logit, logistic
using StatsModels


# files to include:
include("./experiment_utils.jl")
include("./data_l2_prep.jl")
include("./types.jl")
incude("./shift_models/basic_glm.jl")
include("./trainer.jl")



# i. Load DataStore & L1 Experiment (Using your existing code structure)
ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishLower())
exp_dir = "exp/ablation_study"

# ii: load the experiment results
saved_folders = BayesianFootball.Experiments.list_experiments(exp_dir)
exp = BayesianFootball.Experiments.load_experiment(saved_folders[2])



# 1. experiment_utils  

# latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp)
# ppd = BayesianFootball.Predictions.model_inference(latents)

ppd_raw= model_inference(ds, exp)

ppd_raw= Predictions.model_inference(ds, exp)

# 2. data_l2_prep
# @btime training_data_l2 = build_l2_training_df(ds, ppd_raw)
# Profile.clear()
# @profile build_l2_training_df(ds, ppd_raw)
#
# Profile.print(maxdepth=15)




training_data_l2 = build_l2_training_df(ds, ppd_raw)
training_data_l2 = BayesianFootball.Calibration.build_l2_training_df(ds, ppd_raw)
training_data_l2 = Calibration.build_l2_training_df(ds, ppd_raw)



# 3. Configs
shift_model_config = CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = BasicLogitShift(), 
    min_history_splits = 8,   
    max_history_splits = 0,   
)

shift_model_config = BayesianFootball.Calibration.CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = BayesianFootball.Calibration.BasicLogitShift(), 
    min_history_splits = 8,   
    max_history_splits = 0,   
)




# 4. Training 
# --- calibration_trainer 
#  - inputs 
training_data_l2 
config = shift_model_config 


fitted_model_history = train_calibrators(training_data_l2, shift_model_config);
fitted_model_history = BayesianFootball.Calibration.train_calibrators(training_data_l2, shift_model_config);


# 5. applying the calibration to ppd

ppd_cali = apply_calibrators(ppd_raw, ds, fitted_model_history)

ppd_cali = BayesianFootball.Calibration.apply_calibrators(ppd_raw, ds, fitted_model_history)
#
# id = rand(calib_preds.df.match_id)
# target_select = :under_25
#
# d_raw = subset(ppd_raw.df, 
#                 :match_id => ByRow(isequal(id)),
#                :selection => ByRow(isequal(target_select))).distribution[1];
# d_cali = subset(calib_preds.df, 
#                 :match_id => ByRow(isequal(id)),
#                :selection => ByRow(isequal(target_select))).distribution[1];
# mean(d_raw) 
# mean(d_cali)
#

# --- 6. basic metrics

# 1. Build the massive evaluation frame
df_eval = build_evaluation_df(ppd_raw, calib_preds, ds)

df_eval = Calibration.build_evaluation_df(ppd_raw, calib_preds, ds)

# 2. View the overall performance grouped by selection (home, away, over_15, etc.)
summary_by_selection = summarize_metrics(df_eval, groupby_cols=[:selection])
display(summary_by_selection)

# 3. Want to see if the calibration degraded over time? Group by split_id!
summary_by_time = summarize_metrics(df_eval, groupby_cols=[:split_id])
display(sort(summary_by_time, :split_id))

# 4. Want to isolate a specific market to debug? Just use subset!
over_25_only = subset(df_eval, :selection => ByRow(isequal(:over_25)))

eval_raw = build_evaluation_df(ppd_raw, ds)
eval_cali = build_evaluation_df(ppd_cali, ds)

summarize_metrics(eval_raw)
summarize_metrics(eval_cali)
# Get the direct comparison!
df_comparison = compare_models(eval_raw, eval_cali)

display(df_comparison)

# --- 7. signals

min_edge =0.0
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

raw_sig_result = BayesianFootball.Signals.process_signals(ppd_raw, ds.odds, signals; odds_column=:odds_close);
calib_sig_result = BayesianFootball.Signals.process_signals(ppd_cali, ds.odds, signals; odds_column=:odds_close);




using Printf
display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)




println("calib_ppd")
display_edge_threshold_analysis(calib_preds, ds)

println("raw_ppd")
display_edge_threshold_analysis(ppd_raw, ds)


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




function display_edge_threshold_analysis(ppd, ds)
println("\n=== Model: Edge Threshold Analysis ===")
println("Edge% | Bets | Active% | Win%   | Staked | PnL   | ROI%")
println("---------------------------------------------------------")

# for edge_pct in [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
  for edge_pct in 0.0:0.01:0.1

    # Create temporary signals with the new edge
    temp_signals = [BayesianFootball.Signals.BayesianKelly(min_edge=edge_pct)]
    
    # Process just the calibrated PPD
    temp_result = BayesianFootball.Signals.process_signals(ppd, ds.odds, temp_signals; odds_column=:odds_close)
    
    # Summarize
    metrics = summarize_ledger(temp_result.df)
    
    @printf("%5.1f | %4d | %7.2f | %6.2f | %6.2f | %+.2f | %+.2f%%\n", 
            edge_pct * 100, metrics.bets, metrics.active_rate, metrics.win_rate, 
            metrics.total_stake, metrics.total_pnl, metrics.roi)
end

end



function summarize_roi_by_market(sig_result)
    ledger_df = sig_result.df
    
    # Group by the selection column and calculate metrics
    summary = combine(groupby(ledger_df, :selection)) do group
        seen = nrow(group)
        bets = count(>(0.0), group.stake)
        active_rate = seen > 0 ? (bets / seen) * 100 : 0.0
        
        total_stake = sum(group.stake)
        total_pnl = sum(group.pnl)
        roi = total_stake > 0.0 ? (total_pnl / total_stake) * 100 : 0.0
        
        # Calculate wins (handling missing or boolean is_winner formats safely)
        won_bets = count(r -> r.stake > 0.0 && r.is_winner == 1.0, eachrow(group))
        win_rate = bets > 0 ? (won_bets / bets) * 100 : 0.0
        
        return (
            seen = seen,
            bets = bets,
            active_rate = round(active_rate, digits=2),
            win_rate = round(win_rate, digits=2),
            staked = round(total_stake, digits=2),
            pnl = round(total_pnl, digits=2),
            roi = round(roi, digits=2)
        )
    end
    
    # Sort by the markets where we bet the most
    # sort!(summary, :bets, rev=true)
    
    return summary
end



min_edge =0.025
raw_sig_result = BayesianFootball.Signals.process_signals(ppd_raw, ds.odds, signals; odds_column=:odds_close);
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]
calib_sig_result = BayesianFootball.Signals.process_signals(ppd_cali, ds.odds, signals; odds_column=:odds_close);

display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

market_roi_raw = summarize_roi_by_market(raw_sig_result)
market_roi_calib = summarize_roi_by_market(calib_sig_result)


display(market_roi_raw)
display(market_roi_calib)



# -----
# 1. Setup your Configs
shift_median = CalibrationConfig(
    name = "Affine_Median",
    model = BasicLogitShift(), 
    prob_col = :prob_median,
    min_history_splits = 10,   
    max_history_splits = 0,   
)

shift_mean = CalibrationConfig(
    name = "Affine_Mean",
    model = BasicLogitShift(), 
    prob_col = :prob_mean,
    min_history_splits = 10,   
    max_history_splits = 0,   
)

# 2. Train them all
list_configs = [shift_median, shift_mean]
fitted_histories = train_calibrators(training_data_l2, list_configs)

# 3. Apply them all
calib_ppds = apply_calibrators(ppd_raw, ds, fitted_histories)

# Now calib_ppds[1] is your Median model, and calib_ppds[2] is your Mean model!


# -----
shift_team_bias = CalibrationConfig(
    name = "Team_Bias_Logit_Shift",
    model = TeamBiasLogitShift(), 
    prob_col = :prob_mean,
    min_history_splits = 10,   
)

fitted_model_history = train_calibrators(training_data_l2, shift_team_bias);

ppd_cali = apply_calibrators(ppd_raw, ds, fitted_model_history)

# df_eval = build_evaluation_df(ppd_raw, ppd_cali, ds)
# summary_by_selection = summarize_metrics(df_eval, groupby_cols=[:selection])

eval_raw = build_evaluation_df(ppd_raw, ds)
eval_cali = build_evaluation_df(ppd_cali, ds)

summarize_metrics(eval_raw)
summarize_metrics(eval_cali)


df_comparison = compare_models(eval_raw, eval_cali)


aligned_raw_df = semijoin(
    ppd_raw.df, 
    ppd_cali.df, 
    on=[:match_id, :market_name, :selection]
)

# 4. Overwrite raw_ppd with the aligned version
raw_ppd_1 = BayesianFootball.Predictions.PPD(
    aligned_raw_df, 
    ppd_raw.model, 
    ppd_raw.config 
)


min_edge =0.0
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd_1, ds.odds, signals; odds_column=:odds_close);
calib_sig_result = BayesianFootball.Signals.process_signals(ppd_cali, ds.odds, signals; odds_column=:odds_close);



display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

min_edge =0.00
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]
raw_sig_result = BayesianFootball.Signals.process_signals(raw_ppd_1, ds.odds, signals; odds_column=:odds_close);

min_edge =0.05
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]
calib_sig_result = BayesianFootball.Signals.process_signals(ppd_cali, ds.odds, signals; odds_column=:odds_close);

display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)

market_roi_raw = summarize_roi_by_market(raw_sig_result)
market_roi_calib = summarize_roi_by_market(calib_sig_result)


println("calib_ppd")
display_edge_threshold_analysis(ppd_cali, ds)

println("raw_ppd")
display_edge_threshold_analysis(raw_ppd_1, ds)

