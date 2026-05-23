# current_development/01_optimization_tests/r03_grid_search_2d_runner.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores)
using DataFrames

# Include the 2D grid search loaders
include("l03_grid_search_2d_loaders.jl")

# 1. Load Data
println("\n[1] Loading Ireland Data...")
ds = Data.load_datastore_cached(Data.Ireland())

# 2. Define 2D Parameter Grid
# Customize these arrays to change the granularity of the search
half_lives_grid = [30, 90.0, 180.0, 270.0, 300, 600]
market_weights_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

# 3. Run the 2D Grid Search (Defaults to MAP for speed)
summary_df, results_dict = run_grid_search_2d(
    ds, 
    half_lives_grid, 
    market_weights_grid; 
    use_map=true, 
    use_mle=false
)

# 4. Print Results Table (Sorted by best logloss by default)
println("\n============================================================")
println(" 🎉 2D Grid Search Completed!")
println("============================================================")
println(summary_df)

# 5. Identify Best Parameter Combination
best_row = summary_df[1, :]
println("\nOptimal Parameter Choice:")
println("  Best days_half_life: ", best_row.days_half_life)
println("  Best market_weight:  ", best_row.market_weight)
println("  Model LogLoss:       ", best_row.logloss_model)
println("  Observations:        ", best_row.n_obs)

println("\nNote: The full experiment results are preserved in the `results_dict` dictionary.")
println("For example, to inspect the parameters of the best run:")
println("  best_results = results_dict[($(best_row.days_half_life), $(best_row.market_weight))]")
println("  extract_chains(ds, best_results)")




#=
(30×9 DataFrame                                                                                                                                                                                                                                                                                                             
 Row │ days_half_life  market_weight  logloss_model  n_obs  ν_xg     σ_market   ha_σ_γ       kap_σ_κ     lp                                                                                                                                                                                                                 
     │ Float64         Float64        Float64        Int64  Float64  Float64    Float64      Float64     Float64                                                                                                                                                                                                            
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                                                                                                         
   1 │           90.0           0.5        0.507615   5920  2.98818  0.195573   0.0199701    0.219109    -210.024                                                                                                                                                                                                           
   2 │           90.0           1.0        0.507926   5920  3.15323  0.200136   0.0017741    0.297801    -199.473                                                                                                                                                                                                           
   3 │          270.0           1.0        0.507973   5920  1.08682  0.209172   0.00178703   0.334787    -523.85                                                                                                                                                                                                            
   4 │          180.0           0.5        0.507981   5920  3.36732  0.219426   0.00272489   0.164681    -399.1                                                                                                                                                                                                             
   5 │          270.0           0.75       0.50804    5920  3.00233  0.209681   0.048042     0.299346    -526.323                                                                                                                                                                                                           
   6 │          270.0           0.25       0.508151   5920  2.99547  0.231937   0.18522      0.00604411  -558.59                                                                                                                                                                                                            
   7 │          270.0           0.5        0.508206   5920  3.432    0.205446   0.000183252  0.289585    -543.239                                                                                                                                                                                                           
   8 │          300.0           0.75       0.508209   5920  3.21945  0.231836   0.216277     0.00618839  -595.859                                                                                                                                                                                                           
   9 │          180.0           0.75       0.508252   5920  3.15714  0.205992   0.0305664    0.271854    -376.015                                                                                                                                                                                                           
  10 │           90.0           0.75       0.508312   5920  2.96936  0.192147   0.0744411    0.238511    -201.177                                                                                                                                                                                                           
  11 │          180.0           1.0        0.508678   5920  2.92597  0.198707   0.00134706   0.289567    -360.065                                                                                                                                                                                                           
  12 │          180.0           0.25       0.508999   5920  3.00011  0.232847   0.000449694  0.00476012  -403.197                                                                                                                                                                                                           
  13 │          300.0           0.25       0.509363   5920  3.0051   0.213898   0.0793802    0.258405    -598.287                                                                                                                                                                                                           
  14 │           90.0           0.25       0.509418   5920  2.99362  0.209842   0.119337     0.0471123   -217.831                                                                                                                                                                                                           
  15 │           30.0           0.5        0.509434   5920  2.99769  0.226786   0.130334     0.0386077    -87.8821                                                                                                                                                                                                          
  16 │          600.0           0.25       0.509454   5920  3.02385  0.221454   0.00151384   0.289589    -900.406                                                                                                                                                                                                           
  17 │          600.0           0.75       0.509456   5920  2.97423  0.219937   0.0812598    0.300548    -871.11                                                                                                                                                                                                            
  18 │           30.0           1.0        0.509587   5920  3.01197  0.198582   0.153529     0.23405      -83.1489                                                                                                                                                                                                          
  19 │          270.0           0.0        0.509625   5920  3.01086  0.0831317  0.0690233    0.0110724   -558.558                                                                                                                                                                                                           
  20 │          300.0           0.5        0.509761   5920  3.01188  0.21041    0.00804443   0.302652    -583.749                                                                                                                                                                                                           
  21 │          600.0           1.0        0.509852   5920  3.0125   0.219944   1.71301e-5   0.273892    -868.068                                                                                                                                                                                                           
  22 │          300.0           1.0        0.509896   5920  2.99013  0.209179   0.0318022    0.310191    -554.647                                                                                                                                                                                                           
  23 │          600.0           0.0        0.510162   5920  2.99474  0.0294002  0.17651      0.00434569  -905.782                                                                                                                                                                                                           
  24 │           30.0           0.75       0.510182   5920  2.99852  0.210113   0.00764525   0.218269     -86.2496                                                                                                                                                                                                          
  25 │          300.0           0.0        0.510373   5920  2.99269  0.0770615  0.000813777  0.0165295   -603.333                                                                                                                                                                                                           
  26 │          180.0           0.0        0.511245   5920  2.99859  0.100807   0.0174674    0.00817784  -403.575                                                                                                                                                                                                           
  27 │           30.0           0.25       0.512909   5920  2.99824  0.240864   0.0022049    0.00605238   -88.0703                                                                                                                                                                                                          
  28 │           90.0           0.0        0.513923   5920  3.0005   0.100215   0.00397778   0.00386192  -220.704                                                                                                                                                                                                           
  29 │           30.0           0.0        0.527718   5920  2.99993  0.100302   0.000297116  2.53294e-5   -86.4393                                                                                                                                                                                                          
  30 │          600.0           0.5        0.599413   5920  0.68038  0.228758   0.00787877   0.455459    -922.409,
=#

