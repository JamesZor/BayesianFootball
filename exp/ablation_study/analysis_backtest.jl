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
loaded_results_ = loaded_results[1:6]
loaded_results__ = loaded_results[7:13]

loaded_results_ = loaded_results[[1:6; 8:14]]

for (i, lr) in enumerate(loaded_results__)
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

ledger_ = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results__, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)


# 3. Analyze
# ==========
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)
tearsheet_ = BayesianFootball.BackTesting.generate_tearsheet(ledger_)

println("\n=== TEARSHEET SUMMARY ===")
println(tearsheet)

# Breakdown by Model (Selection)
println("\n=== BREAKDOWN BY MODEL ===")
model_names = unique(tearsheet.selection)

model_names = model_names[1:12]

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end





for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet_, :selection => ByRow(isequal(m_name)))
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



flat_rows_ = []

# 2. Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results__)
    model_name = exp.config.name
    print("[$i/$(length(loaded_results__))] Evaluating: $(model_name) ... ")
    
    try
        # Compute the nested RQR struct
        rqr_data = Evaluation.compute_metric(Evaluation.RQR(), exp, ds)
        
        # Flatten it using the magic unroller
        flat_row_ = Evaluation.to_dataframe_row(exp, rqr_data)
        
        # Save to our list
        push!(flat_rows_, flat_row_)
        println("✅ Done")
    catch e
        println("❌ Failed")
        @warn "Error evaluating $model_name: $e"
    end
end

# 3. Build the Master DataFrame
master_rqr_df_ = DataFrame(flat_rows_)

# Sort by model name to keep it organized (01 to 07)
sort!(master_rqr_df_, :model)


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


summary_df_ = select(master_rqr_df_, 
    :model, 
    :rqr_all_mean, 
    :rqr_all_std, 
    :rqr_all_skewness, 
    :rqr_all_kurtosis, 
    :rqr_all_shapiro_w,
    :rqr_all_shapiro_p
)


display(summary_df)
display(summary_df_)



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


# --- funcitons ---
# to compare the mcmc size 
#
function evaluate_batch_crps(results_array, ds; label="CRPS Evaluation")
    println("\n============================================================")
    println(" 🚀 Running Batch CRPS Evaluation: $label")
    println("============================================================")

    flat_rows_crps = []

    # Loop through all provided experiments
    for (i, exp) in enumerate(results_array)
        model_name = exp.config.name
        print("[$i/$(length(results_array))] Evaluating: $(model_name) ... ")
        
        try
            crps_data = Evaluation.compute_metric(Evaluation.CRPS(), exp, ds)
            flat_row = Evaluation.to_dataframe_row(exp, crps_data)
            
            push!(flat_rows_crps, flat_row)
            println("✅ Done")
        catch e
            println("❌ Failed")
            @warn "Error evaluating $model_name: $e"
        end
    end

    # Build the Master DataFrame
    master_crps_df = DataFrame(flat_rows_crps)

    if nrow(master_crps_df) > 0
        # Sort by model name to keep it organized
        sort!(master_crps_df, :model)

        println("\n============================================================")
        println(" 📊 MASTER CRPS COMPARISON: $label")
        println(" Note: LOWER is BETTER")
        println("============================================================")
        display(master_crps_df)
    else
        println("⚠️ No results successfully evaluated.")
    end
    
    return master_crps_df
end


# Run the 300-sample models (_02_)
df_long_run = evaluate_batch_crps(loaded_results_, ds, label="300 Samples (Long Run)")

# Run the 120-sample models
df_short_run = evaluate_batch_crps(loaded_results__, ds, label="120 Samples (Short Run)")

# 1. Make a copy of the long run and clean the names
df_long_clean = copy(df_long_run)
df_long_clean.model = replace.(df_long_clean.model, "_02_" => "_")

# 2. Now join them (the names will match perfectly)
comparison_crps_df = innerjoin(
    select(df_short_run, :model, :crps_all_mean => :crps_120),
    select(df_long_clean, :model, :crps_all_mean => :crps_300),
    on = :model
)

# 3. Calculate the difference (Negative means 300 is better)
comparison_crps_df.improvement = comparison_crps_df.crps_300 .- comparison_crps_df.crps_120

println("\n🔍 MCMC SAMPLE SIZE IMPACT (300 vs 120)")
display(comparison_crps_df)


# ----
function evaluate_batch_glm(results_array, ds; label="GLM Edge Evaluation")
    println("\n============================================================")
    println(" 🚀 Running Batch GLM Edge Evaluation: $label")
    println("============================================================")

    flat_rows_glm = []

    # Loop through all provided experiments
    for (i, exp) in enumerate(results_array)
        model_name = exp.config.name
        print("[$i/$(length(results_array))] Evaluating: $(model_name) ... ")
        
        try
            glm_data = Evaluation.compute_metric(Evaluation.GLMEdge(), exp, ds)
            flat_row = Evaluation.to_dataframe_row(exp, glm_data)
            
            push!(flat_rows_glm, flat_row)
            println("✅ Done")
        catch e
            println("❌ Failed")
            @warn "Error evaluating $model_name: $e"
        end
    end

    # Build the Master DataFrame
    master_glm_df = DataFrame(flat_rows_glm)

    if nrow(master_glm_df) > 0
        # Sort by model name to keep it organized
        sort!(master_glm_df, :model)

        println("\n============================================================")
        println(" 📈 MASTER GLM EDGE COMPARISON: $label")
        println(" Note: HIGHER 'spread_fair_coef' and LOWER 'p_value' is BETTER")
        println("============================================================")
        
        # Display only the critical columns for readability
        display(select(master_glm_df, 
            :model, 
            :glmedge_intercept_coef,
            :glmedge_spread_fair_coef, 
            :glmedge_spread_fair_p_value,
            :glmedge_n_obs
        ))
    else
        println("⚠️ No results successfully evaluated.")
    end
    
    return master_glm_df
end



# 1. Evaluate both runs
df_glm_long = evaluate_batch_glm(loaded_results_, ds, label="300 Samples (Long Run)")
df_glm_short = evaluate_batch_glm(loaded_results__, ds, label="120 Samples (Short Run)")


df_long_clean = copy(df_glm_short)
df_long_clean.model = replace.(df_glm_long.model, "_02_" => "_")


# 2. Join them to compare the Spread Coefficient
comparison_glm_df = innerjoin(
    select(df_glm_short, :model, :glmedge_spread_fair_coef => :coef_120, :glmedge_spread_fair_p_value => :pval_120),
    select(df_glm_long, :model, :glmedge_spread_fair_coef => :coef_300, :glmedge_spread_fair_p_value => :pval_300),
    on = :model,
    makeunique=true
)

# 3. Calculate the difference (Positive means the 300 sample run is BETTER)
comparison_glm_df.coef_improvement = comparison_glm_df.coef_300 .- comparison_glm_df.coef_120

println("\n🔍 MCMC SAMPLE SIZE IMPACT ON GLM EDGE (300 vs 120)")
display(select(comparison_glm_df, :model, :coef_120, :coef_300, :coef_improvement))
