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

training_task = create_experiment_tasks(es)

results = run_experiment_task.(training_task)

saved_folders = Experiments.list_experiments(save_dir; data_dir="")
loaded_results = loaded_experiment_files(saved_folders);
expr = loaded_results[1]




chain_fold_1 = expr.training_results[1][1]
chain_fold_2 = expr.training_results[2][1]
chain_fold_3 = expr.training_results[3][1]

chain_fold_6 = exp.training_results[6][1]


describe(chain_fold_1)
describe(chain_fold_2)


params_to_track_xg = [
    :μ, 
    :γ, 
    :log_r,
    :κ,          # NEW: Goal scaling factor
    :ν_xg,       # NEW: xG Gamma shape parameter
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


#=
julia> display(stability_df_xg)
9×23 DataFrame
 Row │ Fold   μ_mean    μ_std      γ_mean    γ_std      log_r_mean  log_r_std  κ_mean    κ_std      ν_xg_mean  ν_xg_std  α.σ₀_mean  α.σ₀_std   α.σₛ_mean  α.σₛ_std   α.σₖ_mean  α.σₖ_std   β.σ₀_mean  β.σ₀_std   β.σₛ_mean  β.σₛ_std   β.σₖ_mean  β.σₖ_std  
     │ Int64  Float64?  Float64?   Float64?  Float64?   Float64?    Float64?   Float64?  Float64?   Float64?   Float64?  Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?   Float64?  
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1  0.124275  0.192675   0.227173  0.100949      2.81486   0.420345  0.883508  0.182324    10.425    4.63367    0.156404  0.0803867  0.0800811  0.0560835  0.030392   0.0213596   0.215342  0.0864149  0.0788119  0.0572069  0.0305208  0.0231636
   2 │     2  0.153538  0.0952939  0.289451  0.0831265     2.88063   0.414137  0.833608  0.0835335    2.48172  0.49505    0.135601  0.0697949  0.0644452  0.0411537  0.0300132  0.0202667   0.17412   0.0709391  0.0878831  0.0571618  0.0302158  0.0217475
   3 │     3  0.129658  0.0761726  0.283034  0.0720097     2.96151   0.391368  0.855437  0.0697389    2.75293  0.387372   0.119377  0.0630737  0.054582   0.0351659  0.02911    0.0202704   0.213171  0.0750036  0.108193   0.0658634  0.0303579  0.0211584
   4 │     4  0.190049  0.0588143  0.260546  0.0627007     3.05964   0.383846  0.840702  0.0550689    2.96966  0.31745    0.106814  0.0564821  0.0578707  0.0359935  0.0370336  0.0270462   0.201775  0.0690778  0.0907426  0.0552198  0.0276952  0.0195503
   5 │     5  0.197176  0.0534187  0.242074  0.05561       3.08326   0.391778  0.835424  0.051279     3.09783  0.298718   0.117058  0.0623157  0.0556722  0.0357954  0.0346851  0.0230472   0.197417  0.0672085  0.0942795  0.0543073  0.0275758  0.01814
   6 │     6  0.178906  0.0511412  0.274539  0.0555036     3.07642   0.371544  0.837511  0.0461426    3.08574  0.270797   0.110005  0.059124   0.0579961  0.036677   0.0373181  0.0237864   0.206886  0.0670791  0.0952504  0.0544903  0.0280304  0.0183142
   7 │     7  0.172876  0.0476263  0.271594  0.0507423     3.09452   0.38783   0.834902  0.0431688    2.98744  0.247512   0.115322  0.0622279  0.0628165  0.038424   0.032887   0.0212132   0.218799  0.0697947  0.114783   0.0605989  0.0295791  0.0196521
   8 │     8  0.174944  0.0448677  0.260051  0.0498894     3.10425   0.378087  0.843454  0.0432883    3.04903  0.252125   0.116201  0.0611877  0.0614584  0.0387209  0.0305504  0.0194088   0.216462  0.0690233  0.11412    0.0576362  0.0263901  0.0173789
   9 │     9  0.168033  0.0434578  0.273618  0.0468812     3.14798   0.376929  0.839503  0.0410575    3.10498  0.237303   0.115308  0.0591646  0.0607106  0.03705    0.0296024  0.018846    0.2053    0.0624474  0.110048   0.0543838  0.023761   0.0156078
=#

# THe model - extract_parameters
latents = BayesianFootball.Experiments.extract_oos_predictions(ds, expr)


ppd = BayesianFootball.Predictions.model_inference(ds, expr)



# ----

println("============================================================")
println(" 🚀 Running Batch RQR Evaluation...")
println("============================================================")

# 1. Initialize an empty array to hold our NamedTuple rows
flat_rows = []

# 2. Loop through all loaded experiments
for (i, exp) in enumerate(loaded_results[1:3])
    model_name = exp.config.name
  print("[$i/$(length(loaded_results[1:3]))] Evaluating: $(model_name) ... ")
    
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

loaded_results_ = loaded_results[1:3];

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



#=
julia> summary_df = select(master_rqr_df,                                                                                                                                                                                                                                      
           :model,                                                                                                                                                                                                                                                             
           :rqr_all_mean,                                                                                                                                                                                                                                                      
           :rqr_all_std,                                                                                                                                                                                                                                                       
           :rqr_all_skewness,                                                                                                                                                                                                                                                  
           :rqr_all_kurtosis,                                                                                                                                                                                                                                                  
           :rqr_all_shapiro_w,                                                                                                                                                                                                                                                 
           :rqr_all_shapiro_p                                                                                                                                                                                                                                                  
       )                                                                                                                                                                                                                                                                       
3×7 DataFrame                                                                                                                                                                                                                                                                  
 Row │ model                              rqr_all_mean  rqr_all_std  rqr_all_skewness  rqr_all_kurtosis  rqr_all_shapiro_w  rqr_all_shapiro_p                                                                                                                                  
     │ String                             Float64       Float64      Float64           Float64           Float64            Float64                                                                                                                                            
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                 
   1 │ test_featureset_v2_02_home_hiera…     0.0106862     0.864823         -0.103295          0.239074           0.99556           0.40034                                                                                                                                    
   2 │ test_featureset_v2__01_baseline       0.0575883     0.9531           -0.292573          0.156746           0.990432          0.0192178                                                                                                                                  
   3 │ test_featureset_v2_xg_basic_runn…     0.0490165     0.930042         -0.161075          0.39582            0.995454          0.378896
=#


#=
julia> # Let's just view the most important columns: The Spread Coef and its P-Value                                                                                                                                                                                           
       display(select(master_glm_df,                                                                                                                                                                                                                                           
           :model,                                                                                                                                                                                                                                                             
           :glmedge_intercept_coef,                                                                                                                                                                                                                                            
           :glmedge_spread_fair_coef,                                                                                                                                                                                                                                          
           :glmedge_spread_fair_p_value,                                                                                                                                                                                                                                       
           :glmedge_n_obs                                                                                                                                                                                                                                                      
       ))                                                                                                                                                                                                                                                                      
3×5 DataFrame                                                                                                                                                                                                                                                                  
 Row │ model                              glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value  glmedge_n_obs                                                                                                                                         
     │ String                             Float64                 Float64                   Float64                      Int64                                                                                                                                                 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                        
   1 │ test_featureset_v2_02_home_hiera…                -2.93799                   2.21916                   0.00739276           3466                                                                                                                                         
   2 │ test_featureset_v2__01_baseline                  -2.9156                    1.6354                    0.0364451            3466                                                                                                                                         
   3 │ test_featureset_v2_xg_basic_runn…                -2.90681                   1.05544                   0.224042             3466
=#

#=
julia> display(select(master_ll_df, 
           :model, 
           :logloss_overall_model_ll, 
           :logloss_overall_market_ll, 
           :logloss_overall_diff_ll
       ))
3×4 DataFrame
 Row │ model                              logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                             Float64                   Float64                    Float64                 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test_featureset_v2_02_home_hiera…                  0.444557                    18.1667                 -17.7221
   2 │ test_featureset_v2__01_baseline                    0.44417                     18.1667                 -17.7225
   3 │ test_featureset_v2_xg_basic_runn…                  0.443433                    18.1667                 -17.7232
=#


ppd_baseline = Predictions.model_inference(ds, loaded_results[3])
ppd_HA = Predictions.model_inference(ds, loaded_results[2])
ppd_xg = ppd

min_edge =0.03
signals = [BayesianFootball.Signals.BayesianKelly(min_edge=min_edge)]



signal_result_baseline = BayesianFootball.Signals.process_signals(ppd_baseline, ds.odds, signals; odds_column=:odds_close);
signal_result_HA = BayesianFootball.Signals.process_signals(ppd_HA, ds.odds, signals; odds_column=:odds_close);
signal_results_xg = BayesianFootball.Signals.process_signals(ppd_xg, ds.odds, signals; odds_column=:odds_close);




display_results(
    "Baseline Model" => signal_result_baseline,
    "Hierarchical Home Adv" => signal_result_HA,
    "Joint xG Empirical" => signal_results_xg;
    min_edge = 0.05
)


#=
=== L3 Strategy: Bayesian Kelly Ledger (Edge > 0.05) ===

[BASELINE MODEL]
  Seen Markets: 3466
  Bets Placed:  822 (Active Rate: 23.72%)
  Win Rate:     36.74%
  Total Stake:  29.75 units
  Total PnL:    -1.08 units
  ROI:          -3.64%

[HIERARCHICAL HOME ADV]
  Seen Markets: 3466
  Bets Placed:  845 (Active Rate: 24.38%)
  Win Rate:     30.41%
  Total Stake:  24.60 units
  Total PnL:    -1.20 units
  ROI:          -4.88%

[JOINT XG EMPIRICAL]
  Seen Markets: 3466
  Bets Placed:  786 (Active Rate: 22.68%)
  Win Rate:     37.53%
  Total Stake:  24.11 units
  Total PnL:    -0.17 units
  ROI:          -0.71%

========================================================
=#


using Printf


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


function display_results(results::Pair{String, <:Any}...; min_edge=0.05)
    @printf("\n=== L3 Strategy: Bayesian Kelly Ledger (Edge > %.2f) ===\n", min_edge)

    for (model_name, sig_result) in results
        # Compute the metrics for this specific model
        metrics = summarize_ledger(sig_result.df)

        # Print the block
        @printf("\n[%s]\n", uppercase(model_name))
        @printf("  Seen Markets: %d\n", metrics.seen)
        @printf("  Bets Placed:  %d (Active Rate: %.2f%%)\n", metrics.bets, metrics.active_rate)
        @printf("  Win Rate:     %.2f%%\n", metrics.win_rate)
        @printf("  Total Stake:  %.2f units\n", metrics.total_stake)
        @printf("  Total PnL:    %+.2f units\n", metrics.total_pnl)
        @printf("  ROI:          %+.2f%%\n", metrics.roi)
    end
    
    @printf("\n========================================================\n")
end

# -------

display_results(
    "Baseline Model" => signal_result_baseline,
    "Hierarchical Home Adv" => signal_result_HA,
    "Joint xG Empirical" => signal_results_xg;
    min_edge = min_edge
)



using Printf
using DataFrames

function summarize_roi_by_market(sig_result)
    ledger_df = sig_result.df
    
    # SAFEGUARD: Ensure PnL exists before grouping
    if !("pnl" in names(ledger_df))
        ledger_df.pnl = map(eachrow(ledger_df)) do r
            if ismissing(r.is_winner) || r.stake == 0.0
                0.0
            elseif r.is_winner == 1.0 
                r.stake * (r.odds - 1.0)
            else
                -r.stake
            end
        end
    end
    
    # Group by the selection column and calculate metrics
    summary = combine(groupby(ledger_df, :selection)) do group
        seen = nrow(group)
        bets = count(>(0.0), group.stake)
        active_rate = seen > 0 ? (bets / seen) * 100 : 0.0
        
        total_stake = sum(group.stake)
        total_pnl = sum(group.pnl)
        roi = total_stake > 0.0 ? (total_pnl / total_stake) * 100 : 0.0
        
        # Calculate wins 
        won_bets = count(r -> r.stake > 0.0 && r.is_winner == 1.0, eachrow(group))
        win_rate = bets > 0 ? (won_bets / bets) * 100 : 0.0
        
        return (;
            seen = seen,
            bets = bets,
            active_rate = round(active_rate, digits=2),
            win_rate = round(win_rate, digits=2),
            staked = round(total_stake, digits=2),
            pnl = round(total_pnl, digits=2),
            roi = round(roi, digits=2)
        )
    end
    
    # Sort alphabetically by selection for a cleaner read
    sort!(summary, :selection)
    
    return summary
end

function display_results(results::Pair{String, <:Any}...; min_edge=0.00)
    @printf("\n=== L3 Strategy: Bayesian Kelly Ledger (Edge > %.2f) ===\n", min_edge)

    for (model_name, sig_result) in results
        # 1. Get Overall Metrics (this also ensures pnl is calculated)
        metrics = summarize_ledger(sig_result.df)

        # 2. Print Overall Header
        @printf("\n[%s]\n", uppercase(model_name))
        @printf("  Overall -> Bets: %d | Win Rate: %.2f%% | Stake: %.2f | PnL: %+.2f | ROI: %+.2f%%\n", 
            metrics.bets, metrics.win_rate, metrics.total_stake, metrics.total_pnl, metrics.roi)
        
        # 3. Get Market Breakdown
        market_summary = summarize_roi_by_market(sig_result)
        
        # 4. Print Market Breakdown Table
        @printf("  ----------------------------------------------------------------------------------\n")
        @printf("  %-12s | %-6s | %-10s | %-11s | %-8s | %-8s | %-8s\n", 
            "Selection", "Bets", "Active(%)", "Win Rate(%)", "Staked", "PnL", "ROI(%)")
        @printf("  ----------------------------------------------------------------------------------\n")
        
        for row in eachrow(market_summary)
            @printf("  %-12s | %-6d | %-10.2f | %-11.2f | %-8.2f | %-+8.2f | %-+8.2f\n", 
                string(row.selection), row.bets, row.active_rate, row.win_rate, row.staked, row.pnl, row.roi)
        end
        @printf("  ----------------------------------------------------------------------------------\n")
    end
    
    @printf("\n====================================================================================\n")
end






#=
julia> display_results(                                                                                                                
           "Baseline Model" => signal_result_baseline,                                                                                 
           "Hierarchical Home Adv" => signal_result_HA,                                                                                
           "Joint xG Empirical" => signal_results_xg;                                                                                  
           min_edge = 0.00                                                                                                             
       )                                                                                                                               
                                                                                                                                       
=== L3 Strategy: Bayesian Kelly Ledger (Edge > 0.00) ===                                                                               
                                                                                                                                       
[BASELINE MODEL]                                                                                                                       
  Overall -> Bets: 822 | Win Rate: 36.74% | Stake: 29.75 | PnL: -1.08 | ROI: -3.64%
  ----------------------------------------------------------------------------------
  Selection    | Bets   | Active(%)  | Win Rate(%) | Staked   | PnL      | ROI(%)   
  ----------------------------------------------------------------------------------                                                                          
  away         | 81     | 45.00      | 17.28       | 3.67     | -0.57    | -15.49                                                                             
  btts_no      | 77     | 42.78      | 49.35       | 3.38     | -0.25    | -7.46   
  btts_yes     | 18     | 10.00      | 55.56       | 0.23     | -0.09    | -38.64   
  draw         | 39     | 21.67      | 33.33       | 0.69     | +0.39    | +56.94  
  home         | 70     | 38.89      | 28.57       | 4.41     | -1.66    | -37.57   
  over_05      | 3      | 1.67       | 100.00      | 0.01     | +0.00    | +12.00  
  over_15      | 10     | 5.56       | 90.00       | 0.15     | +0.07    | +46.68  
  over_25      | 25     | 13.89      | 48.00       | 0.40     | +0.04    | +9.72   
  over_35      | 30     | 16.67      | 20.00       | 0.32     | +0.06    | +17.46  
  over_45      | 39     | 21.67      | 7.69        | 0.25     | -0.20    | -78.19  
  over_55      | 31     | 17.22      | 0.00        | 0.07     | -0.07    | -100.00 
  over_65      | 11     | 6.83       | 0.00        | 0.01     | -0.01    | -100.00 
  over_75      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  under_05     | 85     | 47.22      | 7.06        | 0.72     | +0.28    | +38.26   
  under_15     | 104    | 57.78      | 30.77       | 3.12     | -0.06    | -1.87                                                       
  under_25     | 95     | 52.78      | 52.63       | 4.51     | +0.16    | +3.58                                                       
  under_35     | 56     | 31.11      | 73.21       | 3.80     | +0.50    | +13.12  
  under_45     | 27     | 15.00      | 88.89       | 2.51     | +0.25    | +10.12   
  under_55     | 15     | 8.33       | 100.00      | 1.30     | +0.07    | +5.25   
  under_65     | 6      | 3.73       | 100.00      | 0.21     | +0.00    | +1.30    
  under_75     | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  ----------------------------------------------------------------------------------
                                                                                                                                       
[HIERARCHICAL HOME ADV]                                                                                                                
  Overall -> Bets: 845 | Win Rate: 30.41% | Stake: 24.60 | PnL: -1.20 | ROI: -4.88%
  ----------------------------------------------------------------------------------
  Selection    | Bets   | Active(%)  | Win Rate(%) | Staked   | PnL      | ROI(%)   
  ----------------------------------------------------------------------------------                                                                          
  away         | 99     | 55.00      | 19.19       | 5.82     | -0.61    | -10.47   
  btts_no      | 50     | 27.78      | 46.00       | 1.45     | +0.03    | +2.31                                                                              
  btts_yes     | 42     | 23.33      | 50.00       | 1.05     | -0.03    | -3.24                                                                              
  draw         | 32     | 17.78      | 34.38       | 0.55     | +0.32    | +58.10                                                                             
  home         | 62     | 34.44      | 22.58       | 3.17     | -1.52    | -48.01                                                                             
  over_05      | 10     | 5.56       | 100.00      | 0.18     | +0.02    | +10.42                                                                             
  over_15      | 26     | 14.44      | 76.92       | 0.81     | +0.32    | +39.73                                                                             
  over_25      | 50     | 27.78      | 46.00       | 1.48     | +0.18    | +11.82                                                                             
  over_35      | 61     | 33.89      | 16.39       | 1.15     | -0.09    | -8.11                                                                              
  over_45      | 82     | 45.56      | 9.76        | 0.87     | -0.42    | -47.83                                                                             
  over_55      | 71     | 39.44      | 1.41        | 0.34     | -0.34    | -99.99                                                                             
  over_65      | 32     | 19.88      | 0.00        | 0.06     | -0.06    | -100.00                                                                            
  over_75      | 5      | 11.90      | 0.00        | 0.00     | -0.00    | -100.00                                                                            
  under_05     | 50     | 27.78      | 10.00       | 0.38     | +0.20    | +51.65                                                                             
  under_15     | 63     | 35.00      | 30.16       | 1.61     | +0.04    | +2.46   
  under_25     | 54     | 30.00      | 48.15       | 2.24     | +0.18    | +8.19                                                                              
  under_35     | 31     | 17.22      | 77.42       | 1.82     | +0.39    | +21.63  
  under_45     | 16     | 8.89       | 87.50       | 1.11     | +0.16    | +14.46  
  under_55     | 7      | 3.89       | 100.00      | 0.52     | +0.03    | +5.97   
  under_65     | 2      | 1.24       | 100.00      | 0.01     | +0.00    | +1.03   
  under_75     | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  ----------------------------------------------------------------------------------

[JOINT XG EMPIRICAL]                                                                                                  
  Overall -> Bets: 786 | Win Rate: 37.53% | Stake: 24.11 | PnL: -0.17 | ROI: -0.71%
  ----------------------------------------------------------------------------------
  Selection    | Bets   | Active(%)  | Win Rate(%) | Staked   | PnL      | ROI(%)  
  ----------------------------------------------------------------------------------
  away         | 73     | 40.56      | 13.70       | 2.06     | +0.06    | +2.70   
  btts_no      | 83     | 46.11      | 50.60       | 3.51     | -0.29    | -8.21   
  btts_yes     | 7      | 3.89       | 28.57       | 0.12     | -0.01    | -11.63  
  draw         | 34     | 18.89      | 38.24       | 0.50     | +0.37    | +74.61  
  home         | 73     | 40.56      | 32.88       | 4.28     | -1.06    | -24.69  
  over_05      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  over_15      | 3      | 1.67       | 66.67       | 0.01     | +0.01    | +48.87  
  over_25      | 24     | 13.33      | 50.00       | 0.15     | +0.08    | +55.50  
  over_35      | 27     | 15.00      | 18.52       | 0.16     | -0.04    | -24.92  
  over_45      | 32     | 17.78      | 12.50       | 0.14     | -0.09    | -62.67  
  over_55      | 19     | 10.56      | 0.00        | 0.04     | -0.04    | -100.00 
  over_65      | 7      | 4.35       | 0.00        | 0.00     | -0.00    | -100.00 
  over_75      | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  under_05     | 88     | 48.89      | 12.50       | 0.62     | +0.61    | +98.07  
  under_15     | 112    | 62.22      | 28.57       | 3.00     | +0.62    | +20.80  
  under_25     | 99     | 55.00      | 50.51       | 4.16     | -0.33    | -7.92   
  under_35     | 62     | 34.44      | 75.81       | 3.13     | -0.24    | -7.57   
  under_45     | 33     | 18.33      | 93.94       | 1.52     | +0.16    | +10.25  
  under_55     | 7      | 3.89       | 100.00      | 0.57     | +0.03    | +4.50   
  under_65     | 3      | 1.86       | 100.00      | 0.13     | +0.00    | +1.61   
  under_75     | 0      | 0.00       | 0.00        | 0.00     | +0.00    | +0.00   
  ----------------------------------------------------------------------------------

====================================================================================
=#




xg_res_df = dev_compute_xg_residuals(ds, expr); # Assuming results[3] is the xG model;


latents
