using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
#
ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/ablation_study"

println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/ablation_study"; data_dir="./data")
# saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end

# extract the loaded files and check that they are the right ones by looking at times.
loaded_results_ = loaded_results[1:7]

for (i, lr) in enumerate(loaded_results_)
  println("file index: $i : name :$(lr.config.name) - $(lr.config.tags[1])")
end



# 2. Run Backtest
# ===============
println("\nRunning Backtest on $(length(loaded_results_)) models...")

baker = BayesianKelly()
# as = AnalyticalShrinkageKelly()
# kelly = KellyCriterion(1)
# kelly25 = KellyCriterion(1/4)
flat_strat = FlatStake(0.05)
my_signals = [baker]
my_signals = [baker, flat_strat]

# my_signals = [baker, as, , kelly, kelly25, flat_strat]



# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results_, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

# 3. Analyze
# ==========
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

println("\n=== TEARSHEET SUMMARY ===")
println(tearsheet)

# Breakdown by Model (Selection)
println("\n=== BREAKDOWN BY MODEL ===")
model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end

for m in loaded_results 
println(m.config.model)
println("\n")
end



# ---- evaluation module testing / dev 
using BayesianFootball.Evaluation

exp = loaded_results_[1]



# 1. Compute the strict DTO Result
rqr_data = Evaluation.compute_metric(Evaluation.RQR(), exp, ds)

# 2. Flatten it into a NamedTuple using the Recursive Unroller
flat_row = Evaluation.to_dataframe_row(exp, rqr_data)

# 3. Create DataFrame
df = DataFrame([flat_row])
display(df)

###

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

# Sort by model name to keep it organized (01 to 07)
sort!(master_rqr_df, :model)

println("\n============================================================")
println(" 📊 MASTER RQR COMPARISON (ALL COLUMNS)")
println("============================================================")
display(master_rqr_df)

println("\n============================================================")
println(" 🎯 EXECUTIVE SUMMARY (Total Match Calibration)")
println(" Target: Mean ≈ 0.00 | StdDev ≈ 1.00 | Shapiro_p > 0.05")
println("============================================================")

# Filter down to just the "pooled/all" stats so it reads beautifully in the REPL
summary_df = select(master_rqr_df, 
    :model, 
    :rqr_all_mean, 
    :rqr_all_std, 
    :rqr_all_skewness, 
    :rqr_all_kurtosis, 
    :rqr_all_shapiro_w,
    :rqr_all_shapiro_p
)

display(summary_df)



# --- cprs --- 
println("============================================================")
println(" 🚀 Running Batch CRPS Evaluation...")
println("============================================================")

flat_rows_crps = []

# Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results_)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results_))] Evaluating: $(model_name) ... ")
    
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

println("\n============================================================")
println(" 📊 MASTER CRPS COMPARISON (LOWER is BETTER)")
println("============================================================")
display(master_crps_df)



### 

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


