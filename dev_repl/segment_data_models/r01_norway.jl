include("./l01_ireland.jl")




ds = BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.Norway())
#=
unique(ds.matches.season)
6-element Vector{Union{Missing, String}}:
 "2021"
 "2022"
 "2023"
 "2024"
 "2025"
 "2026"

count(ismissing(ds.matches.home_score))
count(ismissing(ds.matches.away_score))


=#

save_dir::String = "./data/exp/dev_norway"
es = DSExperimentSettings(ds, "norway", save_dir)

tasks = create_experiment_tasks(es)


# results = run_experiment_task.(tasks)




# --- loaded and back test 

saved_folders = Experiments.list_experiments(save_dir; data_dir="")

loaded_results = loaded_experiment_files(saved_folders);

# ------ 4. Simple Backtesting -----

tearsheet, ledger = run_simple_backtest(loaded_results, ds );


display_tearsheet_by_market(tearsheet)




#=
julia> display_tearsheet_by_market(tearsheet)                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
Stats for: home                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
2×18 DataFrame                                                                                                                                                                                                                                                                                                              
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio                               
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  home                1215          483          39.8     31.78   -11.98   -37.7           28.4       -0.122           -11.98        -1.013      -0.051         -2.026        -0.175 
   2 │ AblationStudy_NB_baseLine          home                1215          485          39.9     31.65   -11.76   -37.16          28.7       -0.119           -11.759       -1.013      -0.052         -2.026        -0.173                                                                                                                                                                                              
Stats for: draw                                                                                                                                                                                                                                                                                                                                                                                                           
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio                               
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  draw                1215          234          19.3      5.6     -1.59   -28.34          14.1       -0.043            -1.588       -0.897      -0.036         -1.747        -0.088 
   2 │ AblationStudy_NB_baseLine          draw                1215          226          18.6      5.63    -1.65   -29.36          14.2       -0.045            -1.654       -0.913      -0.036         -1.78         -0.09                                                                                                                                                                                               
Stats for: away                                                                                                                                                                                                                                                                                                                                                                                                           
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio                               
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  away                1215          613          50.5     30.36    -5.85   -19.28          21.5       -0.051            -5.854       -0.737      -0.032         -2.042        -0.104 
   2 │ AblationStudy_NB_baseLine          away                1215          609          50.1     30.2     -5.81   -19.25          21.5       -0.051            -5.814       -0.736      -0.032         -2.046        -0.103                                                                                                                                                                                              
Stats for: btts_yes                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio                               
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  btts_yes            1207          187          15.5      5.15     0.59    11.46          57.2        0.03              0.59         2.268       0.143          3.461         0.047 
   2 │ AblationStudy_NB_baseLine          btts_yes            1207          185          15.3      4.93     0.56    11.26          56.8        0.029             0.556        2.104       0.127          3.776         0.045                                                                                                                                                                                              
Stats for: btts_no                                                                                                                                                                                                                                                                                                                                                                                                        
2×18 DataFrame                                                                                                                                                                                                                                             
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio                               
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  btts_no             1207          435          36.0     16.26    -2.26   -13.88          38.2       -0.045            -2.257       -0.694      -0.033         -2.908        -0.066 
   2 │ AblationStudy_NB_baseLine          btts_no             1207          436          36.1     16.06    -2.2    -13.72          38.3       -0.045            -2.204       -0.675      -0.032         -2.854        -0.066                                                                                                                                                                                              
Stats for: over_05                                                                                                                                                                                                                                                                                                                                                                                                        
2×18 DataFrame                                       
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64                                                                                                                                                                                                   
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  over_05             1207           29           2.4      1.49     0.02     1.24          89.7        0.009             0.018        0.343       0.021          1.157          0.01                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          over_05             1207           30           2.5      1.27     0.02     1.91          90.0        0.017             0.024        0.664       0.042          0.664          0.02                                                                                                                                                                                              
Stats for: under_05                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                       
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64                                                                                                                                                                                                   
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  under_05            1207          382          31.6      2.54    -0.31   -12.15           4.7       -0.012            -0.309       -0.51       -0.028         -1.188        -0.04                                                                                                                                                                                               
   2 │ AblationStudy_NB_baseLine          under_05            1207          383          31.7      2.53    -0.25    -9.85           4.4       -0.009            -0.25        -0.417      -0.024         -0.967        -0.032                                                                                                                                                                                              
Stats for: over_15                                                                                                                                                                                                                                                                                                                                                                                                        
2×18 DataFrame                                       
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64                                                                                                                                                                                                   
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  over_15             1207          110           9.1      4.67     0.08     1.81          78.2        0.007             0.085        0.255       0.016          0.659         0.009                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          over_15             1207          107           8.9      4.38     0.1      2.29          78.5        0.009             0.1          0.313       0.02           0.747         0.011                                                                                                                                                                                              
Stats for: under_15                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                       
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64                                                                                                                                                                                                   
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  under_15            1207          572          47.4     15.4     -2.99   -19.4           18.2       -0.042            -2.988       -0.695      -0.029         -1.69         -0.088                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          under_15            1207          568          47.1     15.33    -2.99   -19.48          18.3       -0.041            -2.987       -0.692      -0.029         -1.686        -0.088                                                                                                                                                                                              
Stats for: over_25                                                                                                                                                                                                                                                                                                                                                                                                        
2×18 DataFrame                                       
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64                                                                                                                                                                                                   
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                             
   1 │ AblationStudy_NB_baseline_month_r  over_25             1207          274          22.7     11.41     1.48    13.02          62.0        0.048             1.485        2.819       0.179          4.59          0.075                                                                                                                                                                                              
   2 │ AblationStudy_NB_baseLine          over_25             1207          280          23.2     11.11     1.36    12.28          62.1        0.045             1.364        2.836       0.172          4.319         0.07                                                                                                                                                                                               
Stats for: under_25                                                                                                                                                                                                                                                                                                                                                                                                       
2×18 DataFrame                                       
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  under_25            1207          532          44.1     27.96    -4.84   -17.32          36.5       -0.054            -4.842       -0.768      -0.034         -2.128        -0.088
   2 │ AblationStudy_NB_baseLine          under_25            1207          524          43.4     27.87    -4.7    -16.86          35.9       -0.052            -4.699       -0.763      -0.034         -2.078        -0.085
Stats for: over_35                                                                           
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  over_35             1207          285          23.6      8.79     1.53    17.46          39.6        0.039             1.535        2.24        0.127          4.261         0.078
   2 │ AblationStudy_NB_baseLine          over_35             1207          293          24.3      8.57     1.54    18.0           39.2        0.04              1.543        2.296       0.129          4.352         0.082
Stats for: under_35                                                                          
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  under_35            1207          434          36.0     33.18    -2.7     -8.13          59.9       -0.035            -2.698       -0.686      -0.031         -1.288        -0.047
   2 │ AblationStudy_NB_baseLine          under_35            1207          432          35.8     33.12    -2.62    -7.91          59.5       -0.034            -2.622       -0.67       -0.03          -1.257        -0.045
Stats for: over_45                                                                           
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  over_45             1207          364          30.2      7.79    -0.66    -8.44          19.0       -0.014            -0.657       -0.427      -0.022         -1.216         -0.03
   2 │ AblationStudy_NB_baseLine          over_45             1207          362          30.0      7.68    -0.63    -8.25          19.3       -0.014            -0.633       -0.456      -0.023         -1.128         -0.03
Stats for: under_45                                                                          
2×18 DataFrame                                                                               
 Row │ model_name                         selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio  SterlingRatio  SortinoRatio 
     │ String                             Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64     Float64        Float64      
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_baseline_month_r  under_45            1207          315          26.1     31.5     -3.68   -11.68          71.7       -0.06             -3.68        -0.925      -0.038         -1.782        -0.069
   2 │ AblationStudy_NB_baseLine          under_45            1207          317          26.3     31.56    -3.63   -11.51          71.6       -0.059            -3.633       -0.923      -0.037         -1.776        -0.067
=#


