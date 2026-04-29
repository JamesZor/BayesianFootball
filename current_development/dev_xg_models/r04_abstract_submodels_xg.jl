using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)



include("./l04_abstract_submodels_xg.jl")

# ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

save_dir::String = "./data/dev_xg_models/"

es = DSExperimentSettings(
  ds,
  "xg_grid_2",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks_grid(es)


results = run_experiment_task.(training_task)




saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);

expr = loaded_results[1]

expr.training_results[2][1]



loaded_results_ = loaded_results[1:5]

# ----

println("============================================================")
println(" 🚀 Running Batch RQR Evaluation...")
println("============================================================")

# 1. Initialize an empty array to hold our NamedTuple rows
flat_rows = []

# 2. Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results_)
    model_name = exp.config.name
  print("[$i/$(length(loaded_results_))] Evaluating: $(model_name) ... ")
    
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



# ----
flat_rows_glm = []

for (i, exp) in enumerate(loaded_results_)
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

for (i, exp) in enumerate(loaded_results_)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results_))] Evaluating LogLoss for: $(model_name) ... ")
    
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



# 1. SETUP GLOBAL CONFIGS
min_edge = 0.00
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]

# We'll store Ledger objects here for the final big comparison table
all_ledger_results = Pair{String, Any}[]

# Tables for metrics
glm_rows = []
rqr_rows = []
ll_rows  = []

println("============================================================")
println(" 🏁 STARTING GLOBAL BATCH EVALUATION")
println("============================================================")

for (i, exp) in enumerate(loaded_results_)
    model_name = exp.config.name
    println("\n[$i/$(length(loaded_results_))] processing: $model_name")
    
    try
        # --- A. INFERENCE & RAW METRICS ---
        ppd_raw = Predictions.model_inference(ds, exp)
        
        # 1. LogLoss
        ll_data = Evaluation.compute_metric(Evaluation.LogLoss(), exp, ds)
        push!(ll_rows, Evaluation.to_dataframe_row(exp, ll_data))

        # 2. GLM Edge (Statistical edge test)
        glm_data = Evaluation.compute_metric(Evaluation.GLMEdge(), exp, ds)
        push!(glm_rows, Evaluation.to_dataframe_row(exp, glm_data))

        # 3. RQR (Distribution calibration)
        rqr_data = Evaluation.compute_metric(Evaluation.RQR(), exp, ds)
        push!(rqr_rows, Evaluation.to_dataframe_row(exp, rqr_data))

        # --- B. CALIBRATION (The ML Band-Aid) ---
        training_data_l2 = Calibration.build_l2_training_df(ds, ppd_raw)
        shift_model_config = Calibration.CalibrationConfig(
            name = "Affine_Logit_Shift",
            model = Calibration.BasicLogitShift(), 
            min_history_splits = 2,   
            max_history_splits = 0,   
        )
        calibrators = Calibration.train_calibrators(training_data_l2, shift_model_config)
        ppd_cali = Calibration.apply_calibrators(ppd_raw, ds, calibrators)

        # --- C. SIGNAL PROCESSING (PnL/ROI) ---
        # 1. Raw ROI
        ledger_raw = BayesianFootball.Signals.process_signals(
            ppd_raw, ds.odds, signals; odds_column=:odds_close
        )
        push!(all_ledger_results, "$model_name (RAW)" => ledger_raw)

        # 2. Calibrated ROI
        ledger_cali = BayesianFootball.Signals.process_signals(
            ppd_cali, ds.odds, signals; odds_column=:odds_close
        )
        push!(all_ledger_results, "$model_name (CALI)" => ledger_cali)

        println("   ✅ Metrics & Calibration completed.")

    catch e
        println("   ❌ Failed to evaluate $model_name: $e")
    end
end

# 2. AGGREGATE RESULTS INTO DATAFRAMES
master_ll_df  = sort!(DataFrame(ll_rows), :logloss_overall_model_ll)
master_glm_df = sort!(DataFrame(glm_rows), :glmedge_spread_fair_p_value)
master_rqr_df = DataFrame(rqr_rows)

println("\n============================================================")
println(" 📊 FINAL BATCH SUMMARY")
println("============================================================")

# Display LogLoss (Predictive Accuracy)
println("\n--- Predictive Power (LogLoss) ---")
display(select(master_ll_df, :model, :logloss_overall_model_ll, :logloss_overall_diff_ll))

# Display GLM Edge (Statistical validity of value)
println("\n--- Betting Edge Validity (GLM) ---")
display(select(master_glm_df, :model, :glmedge_spread_fair_p_value, :glmedge_spread_fair_coef))

# The "Big Board" (ROI and PnL Comparison)
println("\n--- Backtest Ledger Comparison ---")
display_results(all_ledger_results...; min_edge = min_edge)





#=
[XG_GRID_HIERARCHICALTEAMKAPPA_HOMEAWAYDISPERSION (RAW)]                                                                                                                                                                                     
  Overall -> Bets: 763 | Win Rate: 38.53% | Stake: 24.01 | PnL: -0.10 | ROI: -0.41%                                                                                                                                                          
  ----------------------------------------------------------------------------------                                                                                                                                                         
  Selection    | Bets   | Active(%)  | Win Rate(%) | Staked   | PnL      | ROI(%)                                                                                                                                                            
  ----------------------------------------------------------------------------------                                                                                                                                                         
  away         | 69     | 38.33      | 14.49       | 2.02     | +0.22    | +11.13                                                                                                                                                            
  btts_no      | 76     | 42.22      | 50.00       | 3.14     | -0.20    | -6.27                                                                                                                                                             
  btts_yes     | 9      | 5.00       | 33.33       | 0.16     | -0.02    | -10.18                                                                                                                                                            
  draw         | 35     | 19.44      | 37.14       | 0.48     | +0.29    | +61.17                                                                                                                                                            
  home         | 76     | 42.22      | 34.21       | 4.55     | -0.89    | -19.64                                                                                                                                                            
  over_05      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00                                                                                                                                                             
  over_15      | 3      | 1.67       | 66.67       | 0.02     | +0.01    | +46.90                                                                                                                                                            
  over_25      | 23     | 12.78      | 47.83       | 0.18     | +0.10    | +56.51                                                                                                                                                            
  over_35      | 27     | 15.00      | 22.22       | 0.16     | -0.05    | -30.13                                                                                                                                                            
  over_45      | 28     | 15.56      | 10.71       | 0.12     | -0.05    | -42.31                                                                                                                                                            
  over_55      | 17     | 9.44       | 0.00        | 0.03     | -0.03    | -100.00                                                                                                                                                           
  over_65      | 5      | 3.11       | 0.00        | 0.00     | -0.00    | -100.00                                                                                                                                                           
  over_75      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00                                                                                                                                                             
  under_05     | 80     | 44.44      | 12.50       | 0.51     | +0.63    | +124.18                                                                                                                                                           
  under_15     | 108    | 60.00      | 28.70       | 2.76     | +0.47    | +16.94                                                                                                                                                            
  under_25     | 94     | 52.22      | 47.87       | 4.10     | -0.48    | -11.67                                                                                                                                                            
  under_35     | 64     | 35.56      | 78.12       | 3.29     | -0.24    | -7.27                                                                                                                                                             
  under_45     | 37     | 20.56      | 91.89       | 1.67     | +0.10    | +5.75                                                                                                                                                             
  under_55     | 9      | 5.00       | 100.00      | 0.64     | +0.03    | +4.39                                                                                                                                                             
  under_65     | 3      | 1.86       | 100.00      | 0.18     | +0.00    | +1.44                                                                                                                                                             
  under_75     | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00                                                                                                                                                             
  ----------------------------------------------------------------------------------
=#



#=
[XG_GRID_GLOBALKAPPA_HOMEAWAYDISPERSION (RAW)]
  Overall -> Bets: 782 | Win Rate: 39.39% | Stake: 26.13 | PnL: +0.07 | ROI: +0.25%
  ----------------------------------------------------------------------------------
  Selection    | Bets   | Active(%)  | Win Rate(%) | Staked   | PnL      | ROI(%)  
  ----------------------------------------------------------------------------------
  away         | 70     | 38.89      | 14.29       | 2.05     | +0.15    | +7.47   
  btts_no      | 80     | 44.44      | 52.50       | 3.30     | -0.17    | -5.11   
  btts_yes     | 9      | 5.00       | 33.33       | 0.15     | -0.01    | -5.72   
  draw         | 35     | 19.44      | 37.14       | 0.54     | +0.36    | +66.66  
  home         | 76     | 42.22      | 32.89       | 4.61     | -0.93    | -20.05  
  over_05      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  over_15      | 6      | 3.33       | 66.67       | 0.02     | +0.01    | +48.21  
  over_25      | 23     | 12.78      | 43.48       | 0.19     | +0.12    | +63.04  
  over_35      | 23     | 12.78      | 21.74       | 0.17     | -0.05    | -29.63  
  over_45      | 26     | 14.44      | 11.54       | 0.13     | -0.05    | -42.26  
  over_55      | 15     | 8.33       | 0.00        | 0.03     | -0.03    | -100.00 
  over_65      | 5      | 3.11       | 0.00        | 0.00     | -0.00    | -100.00 
  over_75      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  under_05     | 85     | 47.22      | 12.94       | 0.57     | +0.69    | +122.31 
  under_15     | 108    | 60.00      | 27.78       | 3.01     | +0.58    | +19.28  
  under_25     | 97     | 53.89      | 49.48       | 4.51     | -0.51    | -11.37  
  under_35     | 66     | 36.67      | 75.76       | 3.75     | -0.30    | -7.99   
  under_45     | 44     | 24.44      | 90.91       | 2.04     | +0.16    | +7.89   
  under_55     | 10     | 5.56       | 100.00      | 0.80     | +0.04    | +4.51   
  under_65     | 4      | 2.48       | 100.00      | 0.27     | +0.00    | +1.55   
  under_75     | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  ----------------------------------------------------------------------------------
=#

