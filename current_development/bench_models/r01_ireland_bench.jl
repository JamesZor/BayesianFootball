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


