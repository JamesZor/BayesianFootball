using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())

save_dir::String = "./data/dev_xg_models/"

es = DSExperimentSettings(
  ds,
  "test_featureset_v2",
  save_dir,
  get_target_seasons_string(ds.segment)
)

training_task = create_experiment_tasks2(es)


# results = run_experiment_task.(training_task)



saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);
expr = loaded_results[1]

expr.training_results[2][1]


params_to_track_xg = [
    :μ, 
    :γ, 
    :log_r,
    :κ_global,       # UPDATED: Global goal conversion baseline
    Symbol("δ_κ.σ"), # NEW: Variance/spread of team conversion abilities
    :ν_xg,           
    Symbol("α.σ₀"), 
    Symbol("α.σₛ"), 
    Symbol("α.σₖ"),
    Symbol("β.σ₀"), 
    Symbol("β.σₛ"), 
    Symbol("β.σₖ")
]

all_chains = [res[1] for res in expr.training_results] 
# 3. Generate the Stability Report
stability_df_xg = check_parameter_stability(all_chains, params_to_track_xg)

display(stability_df_xg)


# 
chain_fold_1 = expr.training_results[1][1]
chain_fold_2 = expr.training_results[2][1]
chain_fold_3 = expr.training_results[3][1]

chain_fold_6 = exp.training_results[6][1]


describe(chain_fold_1)
describe(chain_fold_2)



loaded_results_ = loaded_results[1:4]

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



#=
4×7 DataFrame
 Row │ model                              rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p 
     │ String                             Float64       Float64      Float64           Float64           Float64            Float64           
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_featureset_v2_02_home_hiera…    -0.0193521     0.923346        -0.152235        -0.00947393           0.995062          0.30703
   2 │ test_featureset_v2__01_baseline       0.0622615     0.891257        -0.0937457       -0.0135283            0.995557          0.39986
   3 │ test_featureset_v2_xg_basic_runn…     0.0538114     0.932605        -0.180415         0.142568             0.993696          0.139592
   4 │ test_featureset_v2_xg_kappa_team      0.0448429     0.921703        -0.21173         -0.065107             0.99133           0.0331162
=#


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


#=
4×5 DataFrame
 Row │ model                              glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value  glmedge_n_obs 
     │ String                             Float64                 Float64                   Float64                      Int64         
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_featureset_v2_02_home_hiera…                -2.93799                   2.21916                   0.00739276           3466
   2 │ test_featureset_v2__01_baseline                  -2.9156                    1.6354                    0.0364451            3466
   3 │ test_featureset_v2_xg_basic_runn…                -2.90681                   1.05544                   0.224042             3466
   4 │ test_featureset_v2_xg_kappa_team                 -2.90442                   1.24091                   0.150123             3466
=#

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


#=
julia> display(select(master_ll_df, 
           :model, 
           :logloss_overall_model_ll, 
           :logloss_overall_market_ll, 
           :logloss_overall_diff_ll
       ))
4×4 DataFrame
 Row │ model                              logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                             Float64                   Float64                    Float64                 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_featureset_v2_02_home_hiera…                  0.444557                    18.1667                 -17.7221
   2 │ test_featureset_v2__01_baseline                    0.44417                     18.1667                 -17.7225
   3 │ test_featureset_v2_xg_basic_runn…                  0.443433                    18.1667                 -17.7232
   4 │ test_featureset_v2_xg_kappa_team                   0.442996                    18.1667                 -17.7237
=#


min_edge =0.00
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]



signal_result_baseline = BayesianFootball.Signals.process_signals(
  Predictions.model_inference(ds, loaded_results[4]), ds.odds, signals; odds_column=:odds_close);

signal_result_ha = BayesianFootball.Signals.process_signals(
  Predictions.model_inference(ds, loaded_results[3]), ds.odds, signals; odds_column=:odds_close);

signal_result_xg = BayesianFootball.Signals.process_signals(
  Predictions.model_inference(ds, loaded_results[2]), ds.odds, signals; odds_column=:odds_close);

signal_result_xg_kappa = BayesianFootball.Signals.process_signals(
  Predictions.model_inference(ds, loaded_results[1]), ds.odds, signals; odds_column=:odds_close);

signal_result_xg_kappa_cl = BayesianFootball.Signals.process_signals(
  ppd_cali, ds.odds, signals; odds_column=:odds_close);



display_results(
    "Baseline Model" => signal_result_baseline,
    "Hierarchical Home Adv" => signal_result_ha,
    "Joint xG Empirical" => signal_result_xg,
    "Joint xG kappa" => signal_result_xg_kappa,
    "Joint xG kappa cali" => signal_result_xg_kappa_cl;
    min_edge = min_edge
)

# calib
ppd_raw= Predictions.model_inference(ds, expr)
training_data_l2 = Calibration.build_l2_training_df(ds, ppd_raw)
shift_model_config = BayesianFootball.Calibration.CalibrationConfig(
    name = "Pure_Affine_Logit_Shift",
    model = BayesianFootball.Calibration.BasicLogitShift(), 
    min_history_splits = 2,   
    max_history_splits = 0,   
)

fitted_model_history = BayesianFootball.Calibration.train_calibrators(training_data_l2, shift_model_config);
ppd_cali = BayesianFootball.Calibration.apply_calibrators(ppd_raw, ds, fitted_model_history)


xg_res_df = dev_compute_xg_residuals(ds, loaded_results[2]);
xg_res_df_kappa = dev_compute_xg_residuals(ds, expr); 

id = 15238043


subset(ds.odds, :match_id => ByRow( isequal(id)))


# corr xG 

using DataFrames, Statistics, StatsPlots

# 1. Filter for full match stats and isolate the xG columns
xg_eda_df = filter(row -> row.period == "ALL", ds.statistics)

# 2. Keep only the columns we care about and drop any rows with missing xG data
xg_clean = dropmissing(xg_eda_df[!, [:expectedGoals_home, :expectedGoals_away]])

# Extract the vectors
home_xg = Float64.(xg_clean.expectedGoals_home)
away_xg = Float64.(xg_clean.expectedGoals_away)

# 3. Calculate Pearson Correlation
xg_cor = cor(home_xg, away_xg)
println("=== xG Correlation Report ===")
println("Sample Size: ", nrow(xg_clean), " matches")
println("Correlation (r): ", round(xg_cor, digits=4))
println("=============================")

# 4. Visualize the relationship (Hexbin is usually better than scatter for overlapping match data)
@df xg_clean hexbin(:expectedGoals_home, :expectedGoals_away, 
    xlabel="Home xG", 
    ylabel="Away xG", 
    title="Home xG vs Away xG (r = $(round(xg_cor, digits=3)))",
    color=:viridis,
    bins=20
)
