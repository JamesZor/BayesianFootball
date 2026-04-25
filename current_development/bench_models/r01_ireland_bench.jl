using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


include("./l00_main_utils.jl")


ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())



save_dir::String = "./data/bench_models/ireland/"

es = DSExperimentSettings(
  ds,
  "test_batch_1_",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)



saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);


tearsheet, ledger = run_simple_backtest(loaded_results, ds );



display_tearsheet_by_market(tearsheet)



# ----

println("============================================================")
println(" 🚀 Running Batch RQR Evaluation...")
println("============================================================")

# 1. Initialize an empty array to hold our NamedTuple rows
flat_rows = []

# 2. Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results))] Evaluating: $(model_name) ... ")
    
    try
        # Compute the nested RQR struct
        rqr_data = Evaluation.compute_metric(Evaluation.RQR(), exp, ds)
        
        # Flatten it using the magic unroller
        flat_row = Evaluation.to_dataframe_row(exp, rqr_data)
        
        # Save to our list
        push!(flat_rows, flat_row)
        println("✅ Done")
    catch e
        println("❌ Failed")
        @warn "Error evaluating $model_name: $e"
    end
end

# 3. Build the Master DataFrame
master_rqr_df = DataFrame(flat_rows)

# Sort by model name to keep it organized (01 to 07)
sort!(master_rqr_df, :model)

summary_df = select(master_rqr_df, 
    :model, 
    :rqr_all_mean, 
    :rqr_all_std, 
    :rqr_all_skewness, 
    :rqr_all_kurtosis, 
    :rqr_all_shapiro_w,
    :rqr_all_shapiro_p
)


# --- cprs --- 
println("============================================================")
println(" 🚀 Running Batch CRPS Evaluation...")
println("============================================================")

flat_rows_crps = []

# Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results))] Evaluating: $(model_name) ... ")
    
    try
        # FIXED TYPO: CRPS() instead of CPRS()
        crps_data = Evaluation.compute_metric(Evaluation.CRPS(), exp, ds)
        
        # Flatten it using the magic unroller
        flat_row = Evaluation.to_dataframe_row(exp, crps_data)
        
        # Save to our list
        push!(flat_rows_crps, flat_row)
        println("✅ Done")
    catch e
        println("❌ Failed")
        @warn "Error evaluating $model_name: $e"
    end
end

# Build the Master DataFrame
master_crps_df = DataFrame(flat_rows_crps)

# Sort by model name to keep it organized
sort!(master_crps_df, :model)



# ------

flat_rows_glm = []

for (i, exp) in enumerate(loaded_results)
    print("Evaluating GLM Edge for $(exp.config.name)... ")
    
    glm_data = Evaluation.compute_metric(Evaluation.GLMEdge(), exp, ds)
    flat_row = Evaluation.to_dataframe_row(exp, glm_data)
    
    push!(flat_rows_glm, flat_row)
    println("Done")
end

master_glm_df = DataFrame(flat_rows_glm)
sort!(master_glm_df, :model)

# Let's just view the most important columns: The Spread Coef and its P-Value
display(select(master_glm_df, 
    :model, 
    :glmedge_intercept_coef,
    :glmedge_spread_fair_coef, 
    :glmedge_spread_fair_p_value,
    :glmedge_n_obs
))



println("============================================================")
println(" 🚀 Running Batch LogLoss Evaluation...")
println("============================================================")

flat_rows_ll = []

for (i, exp) in enumerate(loaded_results)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results))] Evaluating LogLoss for: $(model_name) ... ")
    
    try
        # Compute the LogLoss struct
        ll_data = Evaluation.compute_metric(Evaluation.LogLoss(), exp, ds)
        
        # Flatten it
        flat_row = Evaluation.to_dataframe_row(exp, ll_data)
        push!(flat_rows_ll, flat_row)
        println("✅ Done")
    catch e
        println("❌ Failed")
        @warn "Error evaluating $model_name: $e"
    end
end

# Build DataFrame
master_ll_df = DataFrame(flat_rows_ll)
sort!(master_ll_df, :model)

println("\n============================================================")
println(" 📉 MASTER LOGLOSS COMPARISON (LOWER IS BETTER)")
println(" Note: A negative 'diff_ll' means your model beat the bookmaker!")
println("============================================================")

display(select(master_ll_df, 
    :model, 
    :logloss_overall_model_ll, 
    :logloss_overall_market_ll, 
    :logloss_overall_diff_ll
))

# ----- calibration 

saved_folders = Experiments.list_experiments(save_dir; data_dir="")
exp = BayesianFootball.Experiments.load_experiment(saved_folders[1])
ppd_raw= Predictions.model_inference(ds, exp)
training_data_l2 = Calibration.build_l2_training_df(ds, ppd_raw)



shift_model_config = BayesianFootball.Calibration.CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = BayesianFootball.Calibration.BasicLogitShift(), 
    min_history_splits = 8,   
    max_history_splits = 0,   
)


fitted_model_history = BayesianFootball.Calibration.train_calibrators(training_data_l2, shift_model_config);
ppd_cali = BayesianFootball.Calibration.apply_calibrators(ppd_raw, ds, fitted_model_history)

aligned_raw_df = semijoin(
    ppd_raw.df, 
    ppd_cali.df, 
    on=[:match_id, :market_name, :selection]
)

# 4. Overwrite raw_ppd with the aligned version
ppd_raw1 = BayesianFootball.Predictions.PPD(
    aligned_raw_df, 
    ppd_raw.model, 
    ppd_raw.config 
)

eval_raw = Calibration.build_evaluation_df(ppd_raw1, ds)
eval_cali = Calibration.build_evaluation_df(ppd_cali, ds)

# 2. If you just want to look at one model:
summary_raw = Calibration.summarize_metrics(eval_raw, groupby_cols=[:selection])

# 3. If you want to compare them side-by-side:
comparison_df = Calibration.compare_models(eval_raw, eval_cali, groupby_cols=[:selection])


min_edge =0.0
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]
raw_sig_result = BayesianFootball.Signals.process_signals(ppd_raw1, ds.odds, signals; odds_column=:odds_close);
calib_sig_result = BayesianFootball.Signals.process_signals(ppd_cali, ds.odds, signals; odds_column=:odds_close);


# current_development/calibration_module/dev_run 
using Printf
display_result(raw_sig_result, calib_sig_result, min_edge=min_edge)



market_roi_raw = summarize_roi_by_market(raw_sig_result)
market_roi_calib = summarize_roi_by_market(calib_sig_result)


display(market_roi_raw)
display(market_roi_calib)




#=
julia> display(market_roi_raw)
21×8 DataFrame
 Row │ selection  seen   bets   active_rate  win_rate  staked   pnl      roi     
     │ Symbol     Int64  Int64  Float64      Float64   Float64  Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────────
   1 │ home         353    131        37.11     33.59     6.82    -2.19   -32.06
   2 │ draw         353     92        26.06     27.17     1.49     0.9     60.9
   3 │ away         353    133        37.68     14.29     4.4     -0.69   -15.6
   4 │ btts_yes     353     22         6.23     59.09     0.34    -0.01    -1.97
   5 │ btts_no      353    208        58.92     49.04    13.88    -1.68   -12.1
   6 │ over_05      353      7         1.98     71.43     0.15    -0.01    -3.95
   7 │ under_05     353    194        54.96      7.73     3.05    -0.68   -22.21
   8 │ over_15      353     22         6.23     72.73     0.56    -0.03    -5.94
   9 │ under_15     353    229        64.87     31.0     11.43    -1.3    -11.4
  10 │ over_25      353     58        16.43     46.55     1.29    -0.17   -12.88
  11 │ under_25     353    215        60.91     57.67    17.86     1.91    10.72
  12 │ over_35      353     64        18.13     23.44     0.86    -0.01    -1.32
  13 │ under_35     353    170        48.16     78.82    16.76     1.67     9.97
  14 │ over_45      353     74        20.96     13.51     0.7      0.23    33.34
  15 │ under_45     353    107        30.31     91.59    12.13     0.57     4.66
  16 │ over_55      353     60        17.0       5.0      0.29    -0.08   -26.21
  17 │ under_55     353     62        17.56     96.77     6.58     0.2      2.98
  18 │ over_65      326     29         8.9       0.0      0.05    -0.05  -100.0
  19 │ under_65     326     21         6.44    100.0      1.76     0.02     1.32
  20 │ over_75       96      1         1.04      0.0      0.0     -0.0   -100.0
  21 │ under_75      96      5         5.21    100.0      0.17     0.0      0.34

julia> display(market_roi_calib)
21×8 DataFrame
 Row │ selection  seen   bets   active_rate  win_rate  staked   pnl      roi     
     │ Symbol     Int64  Int64  Float64      Float64   Float64  Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────────
   1 │ home         353    147        41.64     34.01     8.75    -2.75   -31.45
   2 │ draw         353    109        30.88     23.85     2.13     0.82    38.38
   3 │ away         353    106        30.03     14.15     2.81    -0.22    -7.78
   4 │ btts_yes     353     51        14.45     50.98     1.35     0.22    15.93
   5 │ btts_no      353    158        44.76     49.37     9.62    -0.87    -9.03
   6 │ over_05      353      9         2.55     77.78     0.21    -0.03   -13.58
   7 │ under_05     353    202        57.22      8.42     3.93    -0.99   -25.27
   8 │ over_15      353     37        10.48     70.27     1.55    -0.03    -1.99
   9 │ under_15     353    196        55.52     31.63     8.53    -1.54   -18.09
  10 │ over_25      353     81        22.95     44.44     2.63    -0.4    -15.37
  11 │ under_25     353    178        50.42     56.74    12.01     1.56    13.03
  12 │ over_35      353     58        16.43     22.41     0.82    -0.11   -12.92
  13 │ under_35     353    173        49.01     79.19    17.4      1.5      8.59
  14 │ over_45      353     32         9.07     15.62     0.21     0.03    16.8
  15 │ under_45     353    173        49.01     92.49    23.96     0.65     2.71
  16 │ over_55      353      0         0.0       0.0      0.0      0.0      0.0
  17 │ under_55     353    250        70.82     97.2     71.78     0.38     0.53
  18 │ over_65      326      0         0.0       0.0      0.0      0.0      0.0
  19 │ under_65     326    180        55.21     97.78    56.68     0.26     0.46
  20 │ over_75       96      0         0.0       0.0      0.0      0.0      0.0
  21 │ under_75      96     96       100.0     100.0     95.9      0.25     0.26
=#


