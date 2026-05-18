using Revise
using BayesianFootball
using DataFrames
using ThreadPinning

# Pin threads for efficiency
pinthreads(:cores)

# Include the logic file
include("./l00_validation_logic.jl")

println("==========================================================")
println(" LEAGUE OF IRELAND PREMIER DIVISION: MODEL VALIDATION EDA ")
println("==========================================================\n")

# 1. Load Data
println("[INFO] Loading Ireland Premier Division data...")
ds = Data.load_datastore_sql(Data.Ireland())
df = ds.matches

# 2. Extract Goals
home_goals = collect(skipmissing(df.home_score))
away_goals = collect(skipmissing(df.away_score))
all_goals = vcat(home_goals, away_goals)

# --- TEST 1: Base Distribution (Poisson vs NB) ---
test_overdispersion(home_goals, "Home Goals")
test_overdispersion(away_goals, "Away Goals")
test_overdispersion(all_goals, "All Goals")




#=
julia> test_overdispersion(home_goals, "Home Goals")

══════════════════════════════════════════════════
 DISTRIBUTION TEST: HOME GOALS 
══════════════════════════════════════════════════
Mean: 1.3988 | Variance: 1.4969 | Index of Dispersion: 1.0702
Metric          | Poisson      | NegBinomial 
---------------------------------------------
Log-Likelihood  | -1480.11     | -1479.08    
AIC             | 2962.21      | 2962.16     

Result: Overdispersion detected (AIC_NB < AIC_P). Justifies Negative Binomial.
(mean = 1.3987730061349692, var = 1.4969073977557448, di = 1.0701574817288877, aic_p = 2962.2131259256953, aic_nb = 2962.1639294448355)

julia> test_overdispersion(away_goals, "Away Goals")

══════════════════════════════════════════════════
 DISTRIBUTION TEST: AWAY GOALS 
══════════════════════════════════════════════════
Mean: 1.0890 | Variance: 1.0494 | Index of Dispersion: 0.9637
Metric          | Poisson      | NegBinomial 
---------------------------------------------
Log-Likelihood  | -1311.68     | -1311.68    
AIC             | 2625.35      | 2627.35     

Result: Poisson is sufficient (AIC_P <= AIC_NB).
(mean = 1.0889570552147239, var = 1.0493968640699274, di = 0.9636714864416799, aic_p = 2625.3520261803687, aic_nb = 2627.35200242891)

julia> test_overdispersion(all_goals, "All Goals")

══════════════════════════════════════════════════
 DISTRIBUTION TEST: ALL GOALS 
══════════════════════════════════════════════════
Mean: 1.2439 | Variance: 1.2965 | Index of Dispersion: 1.0423
Metric          | Poisson      | NegBinomial 
---------------------------------------------
Log-Likelihood  | -2810.70     | -2809.94    
AIC             | 5623.40      | 5623.89     

Result: Poisson is sufficient (AIC_P <= AIC_NB).
(mean = 1.2438650306748467, var = 1.2965096574772874, di = 1.0423234237671903, aic_p = 5623.398199799196, aic_nb = 5623.885833479515)
=#

test_distributions(home_goals, "Home Goals")
test_distributions(away_goals, "Away Goals")
test_distributions(all_goals, "All Goals")


#=
julia> test_distributions(home_goals, "Home Goals")

═════════════════════════════════════════════════════════════════
 DISTRIBUTION TEST: HOME GOALS 
═════════════════════════════════════════════════════════════════
Mean: 1.3988 | Variance: 1.4969 | Index of Dispersion: 1.0702
Metric          | Poisson      | NegBinomial  | WeibullCount
-----------------------------------------------------------------
Log-Likelihood  | -1480.11     | -1479.08     | -1479.64    
AIC             | 2962.21      | 2962.16      | 2963.28     

Result:
Winner: Negative Binomial. Massive overdispersion detected; variance heavily outweighs the mean.
(mean = 1.3987730061349692, var = 1.4969073977557448, di = 1.0701574817288877, aic_p = 2962.2131259256953, aic_nb = 2962.1639294448355, aic_w = 2963.2842233474553)

julia> test_distributions(away_goals, "Away Goals")

═════════════════════════════════════════════════════════════════
 DISTRIBUTION TEST: AWAY GOALS 
═════════════════════════════════════════════════════════════════
Mean: 1.0890 | Variance: 1.0494 | Index of Dispersion: 0.9637
Metric          | Poisson      | NegBinomial  | WeibullCount
-----------------------------------------------------------------
Log-Likelihood  | -1311.68     | -1311.68     | -1311.01    
AIC             | 2625.35      | 2627.35      | 2626.01     

Result:
Winner: Poisson. The data lacks significant variance or time-varying hazards. Simpler is better.
(mean = 1.0889570552147239, var = 1.0493968640699274, di = 0.9636714864416799, aic_p = 2625.3520261803687, aic_nb = 2627.35200242891, aic_w = 2626.012231646384)

julia> test_distributions(all_goals, "All Goals")

═════════════════════════════════════════════════════════════════
 DISTRIBUTION TEST: ALL GOALS 
═════════════════════════════════════════════════════════════════
Mean: 1.2439 | Variance: 1.2965 | Index of Dispersion: 1.0423
Metric          | Poisson      | NegBinomial  | WeibullCount
-----------------------------------------------------------------
Log-Likelihood  | -2810.70     | -2809.94     | -2810.54    
AIC             | 5623.40      | 5623.89      | 5625.09     

Result:
Winner: Poisson. The data lacks significant variance or time-varying hazards. Simpler is better.
(mean = 1.2438650306748467, var = 1.2965096574772874, di = 1.0423234237671903, aic_p = 5623.398199799196, aic_nb = 5623.885833479515, aic_w = 5625.0890372202775)
=#



# --- TEST 2: Home Advantage on Mean ---
test_home_advantage_mean(df)
#=
julia> test_home_advantage_mean(df)

--- Home Advantage Test (Mean) ---
Home Mean: 1.3988 | Away Mean: 1.0890 | Difference: 0.3098
Mann-Whitney U p-value: 3.9078e-08
Result: Statistically significant home advantage on goals.
=#


# --- TEST 3: Home Advantage on Variance ---
test_home_advantage_variance(df)


#=
julia> test_home_advantage_variance(df)

--- Home Advantage Test (Variance/Chaos) ---
Home Variance: 1.4969 | Away Variance: 1.0494 | Ratio: 1.4264
F-test (Variance) p-value: 3.1102e-08
Result: Statistically significant difference in variance (Home vs Away).
=#


# --- TEST 4: Team-Specific Volatility ---
test_team_volatility(df)


#=
julia> test_team_volatility(df)

--- Team-Specific Volatility (Goals Conceded) ---
5×5 DataFrame
 Row │ team_id                    mean_conceded  var_conceded  n      dispersion_index 
     │ String                     Float64        Float64       Int64  Float64          
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ drogheda-united                  1.41026       1.67618    195           1.18857
   2 │ sligo-rovers                     1.28061       1.37214    196           1.07147
   3 │ university-college-dublin        2.26389       2.366       72           1.0451
   4 │ st-patricks-athletic             1.04592       1.09019    196           1.04233
   5 │ dundalk-fc                       1.225         1.23208    160           1.00578

Average Team-Level Dispersion Index: 0.9283
14×5 DataFrame
 Row │ team_id                    mean_conceded  var_conceded  n      dispersion_index 
     │ String                     Float64        Float64       Int64  Float64          
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ drogheda-united                 1.41026       1.67618     195          1.18857
   2 │ sligo-rovers                    1.28061       1.37214     196          1.07147
   3 │ university-college-dublin       2.26389       2.366        72          1.0451
   4 │ st-patricks-athletic            1.04592       1.09019     196          1.04233
   5 │ dundalk-fc                      1.225         1.23208     160          1.00578
   6 │ waterford-fc                    1.58537       1.58896     123          1.00227
   7 │ cork-city                       1.73611       1.63361      72          0.940958
   8 │ shelbourne                      1.03145       0.954701    159          0.925594
   9 │ shamrock-rovers                 0.806122      0.690424    196          0.856475
  10 │ longford-town                   1.97222       1.68492      36          0.854326
  11 │ bohemian                        1.16837       0.961251    196          0.82273
  12 │ derry-city                      0.928571      0.753846    196          0.811834
  13 │ finn-harps                      1.70833       1.25176      72          0.732738
  14 │ galway-united                   1.10345       0.768244     87          0.696221
=#


# --- TEST 5: Match-Level Chaos ---
test_match_level_chaos(df)


#=
9×4 DataFrame
 Row │ matchup           variance_total_goals  mean_total_goals  n     
     │ String            Float64               Float64           Int64 
─────┼─────────────────────────────────────────────────────────────────
   1 │ Mid vs Mid                     2.79647           2.44776     67
   2 │ Top vs Bottom                  2.76148           3.12712    118
   3 │ Bottom vs Mid                  2.68654           2.43077     65
   4 │ Top vs Top                     2.42828           2.32178    202
   5 │ Mid vs Bottom                  2.34038           2.81538     65
   6 │ Top vs Mid                     2.26211           2.55263    152
   7 │ Bottom vs Bottom               2.24615           2.6         40
   8 │ Bottom vs Top                  1.92708           2.26891    119
   9 │ Mid vs Top                     1.62935           2.18667    150
=#


# --- TEST 6: Temporal Stability (Monthly) ---
test_temporal_stability(df)


#=
--- Temporal Stability: Monthly Goal Statistics ---
10×4 DataFrame
 Row │ month  mean_goals  var_goals  n     
     │ Int64  Float64     Float64    Int64 
─────┼─────────────────────────────────────
   1 │     2     2.43077    2.46779     65
   2 │     3     2.31624    2.39051    117
   3 │     4     2.55952    2.24793    168
   4 │     5     2.51534    2.20192    163
   5 │     6     2.46392    2.23046     97
   6 │     7     2.71875    2.74504     64
   7 │     8     2.3375     2.22642     80
   8 │     9     2.54255    2.37989     94
   9 │    10     2.40659    2.53284     91
  10 │    11     2.71795    2.78677     39

Kruskal-Wallis (Mean Drift) p-value: 0.7854542540522376
Result: Scoring mean is relatively stable across months.
Variance Range: Max 2.787 vs Min 2.202 (Ratio: 1.27)
=#


# --- TEST 7: Form Autocorrelation ---
test_form_autocorrelation(df)


#=
--- Form Autocorrelation (Goal Difference) ---
Lag      | Avg ACF   
----------------------
1        | -0.0294   
2        | -0.0003   
3        | 0.0127    
4        | -0.0270   
5        | 0.0038    
6        | -0.0200   
7        | 0.0277    
8        | 0.0213    
9        | 0.0011    
10       | -0.0098   
11       | 0.0224    
12       | 0.0279    
13       | 0.0458    
14       | -0.0136   
15       | 0.0176    

Result: Form decays below 0.1 after approximately 1 matches.
Note: If 1 is small (e.g. < 5), form is transient. If large (e.g. > 10), form is stable.
16-element Vector{Float64}:
  1.0
 -0.029426135804603553
 -0.00029204242817723006
  0.012674512876231664
 -0.02702919404854422
  0.003847487902688037
 -0.02001857434392144
  0.02769977840916597
  0.021294832285417155
  0.0010682308196031307
 -0.009803658503555945
  0.022395633977757987
  0.027905392458468275
  0.04581942158806729
 -0.013632917718913626
  0.017576148036998702
=#


println("\n[INFO] EDA Complete.")
