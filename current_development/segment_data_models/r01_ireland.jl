
include("./l01_ireland.jl")




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Ireland())
#=
julia> unique(ds.matches.season)
6-element Vector{Union{Missing, String}}:
 "2021"
 "2022"
 "2023"
 "2024"
 "2025"
 "2026"

=#

save_dir::String = "./data/exp/dev_ireland"
es = DSExperimentSettings(ds, "ireland", save_dir)

tasks = create_experiment_tasks(es)


results = run_experiment_task.(tasks)




# --- loaded and back test 

saved_folders = Experiments.list_experiments(save_dir; data_dir="")

loaded_results = loaded_experiment_files(saved_folders);

# ------ 4. Simple Backtesting -----

tearsheet, ledger = run_simple_backtest(loaded_results, ds );


display_tearsheet_by_market(tearsheet)

#=
                                                                                                                                                                                                                                                                                                                            
display_tearsheet_by_market(tearsheet)
Stats for: home                                                                                                                                                                                                                                                                                                             
2×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  home                 503          179          35.6      8.95    -2.56   -28.56          29.6       -0.092            -2.556       -0.959      -0.066         -0.959        -0.127 
   2 │ AblationStudy_NB_baseLine          home                 503          181          36.0      8.89    -2.66   -29.88          29.3       -0.098            -2.656       -0.952      -0.065         -0.952        -0.133 
Stats for: draw                                                                                                                                                                                                                                            
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  draw                 503          124          24.7      2.19     1.22    55.59          23.4        0.057             1.217        3.017       0.435          5.482         0.202 
   2 │ AblationStudy_NB_baseLine          draw                 503          124          24.7      2.19     1.28    58.31          23.4        0.059             1.279        3.187       0.458          5.766         0.213 
Stats for: away                                                                                                                                                                                                                                            
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  away                 503          193          38.4      5.88    -1.0    -17.09          14.5       -0.033            -1.004       -0.622      -0.051          -1.79        -0.08  
   2 │ AblationStudy_NB_baseLine          away                 503          195          38.8      5.77    -0.98   -17.02          14.9       -0.032            -0.981       -0.578      -0.047          -1.73        -0.079 
Stats for: btts_yes                                                                                                                                                                                                                                        
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  btts_yes             502           48           9.6      0.93    -0.05    -5.17          47.9       -0.01             -0.048       -0.127      -0.013         -0.254        -0.013 
   2 │ AblationStudy_NB_baseLine          btts_yes             502           49           9.8      0.94    -0.12   -12.89          46.9       -0.023            -0.121       -0.278      -0.028         -0.554        -0.028 
Stats for: btts_no                                                                                                                                                                                                                                         
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  btts_no              502          270          53.8     15.99    -1.31    -8.19          48.1       -0.044            -1.31        -0.819      -0.055         -2.353        -0.056 
   2 │ AblationStudy_NB_baseLine          btts_no              502          272          54.2     16.06    -1.36    -8.45          48.5       -0.045            -1.358       -0.836      -0.055         -2.394        -0.057 
Stats for: over_05                                                                                                                                                                                                                                         
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  over_05              485           13           2.7      0.38     0.03     8.28         100.0        0.092             0.031          0.0         0.0            0.0         999.0                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          over_05              485           10           2.1      0.34     0.03     8.31         100.0        0.089             0.028          0.0         0.0            0.0         999.0                                                                                                                                                                                              
Stats for: under_05                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                            
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  under_05             485          241          49.7      3.34    -0.23    -6.92           7.5       -0.01             -0.231       -0.19       -0.016         -0.33         -0.033                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          under_05             485          240          49.5      3.38    -0.36   -10.69           7.9       -0.016            -0.361       -0.292      -0.025         -0.506        -0.05                                                                                                                                                                                               
Stats for: over_15                                                                                                                                                                                                                                                                                                                                                                                                        
2×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                            
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  over_15              485           33           6.8      1.35     0.13     9.33          72.7        0.029             0.126        0.564       0.071          1.481         0.042                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          over_15              485           31           6.4      1.31     0.04     3.11          67.7        0.009             0.041        0.143       0.017          0.387         0.011                                                                                                                                                                                              
Stats for: under_15                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                            
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  under_15             485          299          61.6     13.08    -0.51    -3.89          30.1       -0.014            -0.509       -0.242      -0.022         -0.621        -0.025                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          under_15             485          302          62.3     13.18    -0.7     -5.33          29.5       -0.019            -0.702       -0.321      -0.029         -0.826        -0.033                                                                                                                                                                                              
Stats for: over_25                                                                                                                                                                                                                                                                                                                                                                                                        
2×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                            
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  over_25              486           85          17.5      2.38     0.24    10.21          42.4        0.021             0.243        0.486       0.058          1.378         0.041                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          over_25              486           84          17.3      2.28     0.12     5.15          45.2        0.011             0.117        0.202       0.025          0.617         0.019                                                                                                                                                                                              
Stats for: under_25                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                                                                                                                                                                                                                                                                                                                                                                                            
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  under_25             486          277          57.0     20.29     1.89     9.33          56.0        0.056             1.892        1.789       0.226          3.023         0.084
   2 │ AblationStudy_NB_baseLine          under_25             486          279          57.4     20.53     1.9      9.24          55.6        0.055             1.897        1.796       0.225          2.944         0.083
Stats for: over_35                                                                           
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  over_35              485           98          20.2      1.73     0.1      6.05          21.4        0.009             0.105        0.24        0.02           0.527         0.018
   2 │ AblationStudy_NB_baseLine          over_35              485           96          19.8      1.65     0.13     7.79          24.0        0.011             0.129        0.356       0.027          0.613         0.023
Stats for: under_35                                                                          
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  under_35             485          206          42.5     18.73     1.1      5.88          78.2        0.053             1.102        1.418       0.189          2.547         0.067
   2 │ AblationStudy_NB_baseLine          under_35             485          208          42.9     19.12     0.96     5.01          77.9        0.045             0.959        1.2         0.153          2.044         0.056
Stats for: over_45                                                                           
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  over_45              488          118          24.2      1.46     0.56    38.08          11.0        0.032             0.556        1.536       0.147          3.246         0.115
   2 │ AblationStudy_NB_baseLine          over_45              488          118          24.2      1.4      0.46    33.09          11.9        0.029             0.462        1.43        0.125          2.765         0.097
Stats for: under_45                                                                          
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  under_45             488          137          28.1     12.94     0.13     1.03          90.5        0.011             0.134        0.33        0.04           0.565         0.012
   2 │ AblationStudy_NB_baseLine          under_45             488          134          27.5     13.46     0.19     1.38          90.3        0.015             0.186        0.466       0.057          0.779         0.017

=#

