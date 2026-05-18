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



using DataFrames, Dates, Statistics, GLM, ShiftedArrays, Printf

println("==========================================================")
println(" PLAYER-LEVEL SOFASCORE VALIDATION EDA ")
println("==========================================================\n")

# ==========================================
# 1. PREPARE THE DATA (No Look-Ahead EWMA)
# ==========================================
println("[INFO] Calculating Pre-Match Player Form (EWMA)...")

# Join lineups with dates and sort strictly by time
lineups_time = innerjoin(
    dropmissing(ds.lineups, :rating), 
    select(ds.matches, :match_id, :match_date, :home_score, :away_score, :winner_code), 
    on = :match_id
)
sort!(lineups_time, :match_date)

# EWMA Function (Strictly Pre-Match)
function calc_pre_match_ewma(ratings::AbstractVector; alpha=0.15)
    n = length(ratings)
    baselines = zeros(Float64, n)
    baselines[1] = NaN # Cold start
    
    if n > 1
        current_ewma = Float64(ratings[1])
        for i in 2:n
            baselines[i] = current_ewma # Log the form BEFORE the match
            # Update the EWMA AFTER the match for the next iteration
            current_ewma = (alpha * ratings[i]) + ((1.0 - alpha) * current_ewma)
        end
    end
    return baselines
end

# Apply to every player
transform!(groupby(lineups_time, :player_id), 
    :rating => (r -> calc_pre_match_ewma(r, alpha=0.15)) => :pre_match_ewma
)

# Fill NaNs (debuts) with the global average to prevent data loss
global_avg = mean(skipmissing(lineups_time.rating))
lineups_time.pre_match_ewma = coalesce.(replace(lineups_time.pre_match_ewma, NaN => global_avg), global_avg)

# ==========================================
# 2. AGGREGATE TO TEAM LEVEL (Minutes Weighted)
# ==========================================
println("[INFO] Aggregating to Match Level...")

# Weight by minutes played (max 90)
lineups_time.mins = coalesce.(lineups_time.minutes_played, 0.0)
lineups_time.weighted_rating = lineups_time.pre_match_ewma .* (clamp.(lineups_time.mins, 0.0, 90.0) ./ 90.0)

# Sum ratings per team per match
team_ratings = combine(groupby(lineups_time, [:match_id, :team_side]), :weighted_rating => sum => :total_rating)

home_ratings = rename!(filter(r -> r.team_side == "home", team_ratings), :total_rating => :home_rating)[!, [:match_id, :home_rating]]
away_ratings = rename!(filter(r -> r.team_side == "away", team_ratings), :total_rating => :away_rating)[!, [:match_id, :away_rating]]

# Join back to match results
df_model = innerjoin(ds.matches[!, [:match_id, :winner_code, :home_score, :away_score]], home_ratings, on=:match_id)
df_model = innerjoin(df_model, away_ratings, on=:match_id)
dropmissing!(df_model)

# ==========================================
# TEST 1: The "Wisdom of the Ratings" Test
# ==========================================
# Does the team with the higher aggregated pre-match rating actually win?
df_model.higher_rated_won = [
    (r.home_rating > r.away_rating && r.winner_code == 1) || 
    (r.away_rating > r.home_rating && r.winner_code == 2) 
    for r in eachrow(df_model)
]

# Note: In football, the better team wins roughly 45-50% of the time due to draws.
win_pct = mean(df_model.higher_rated_won)
@printf("TEST 1: The higher-rated team won %.2f%% of the time.\n", win_pct * 100)
if win_pct > 0.45
    println(" -> SUCCESS: The aggregated ratings hold predictive baseline power.\n")
else
    println(" -> WARNING: The ratings are struggling to predict outright winners. They might be too noisy.\n")
end


#=
julia> win_pct = mean(df_model.higher_rated_won)
0.44227642276422763

julia> @printf("TEST 1: The higher-rated team won %.2f%% of the time.\n", win_pct * 100)
TEST 1: The higher-rated team won 44.23% of the time.

julia> if win_pct > 0.45
           println(" -> SUCCESS: The aggregated ratings hold predictive baseline power.\n")
       else
           println(" -> WARNING: The ratings are struggling to predict outright winners. They might be too noisy.\n")
       end
 -> WARNING: The ratings are struggling to predict outright winners. They might be too noisy.
=#


# ==========================================
# TEST 2: Logistic Regression (Replicating Table 5.1)
# ==========================================
# Does the rating gap statistically explain a Home Win?
df_model.is_home_win = df_model.winner_code .== 1

logit_model = glm(@formula(is_home_win ~ home_rating + away_rating), df_model, Binomial(), LogitLink())

#=
julia> logit_model = glm(@formula(is_home_win ~ home_rating + away_rating), df_model, Binomial(), LogitLink())
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

is_home_win ~ 1 + home_rating + away_rating

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   2.8022     5.16693     0.54    0.5876  -7.32478   12.9292
home_rating   0.240923   0.056926    4.23    <1e-04   0.129351   0.352496
away_rating  -0.282177   0.0532555  -5.30    <1e-06  -0.386556  -0.177798
─────────────────────────────────────────────────────────────────────────
=#


println("TEST 2: Logistic Regression (Predicting Home Win)")
println(coeftable(logit_model))

p_vals = coeftable(logit_model).cols[4] # P-value column
if p_vals[2] < 0.05 && p_vals[3] < 0.05
    println("\n -> SUCCESS: Both Home and Away ratings are statistically significant predictors of the match outcome.")
else
    println("\n -> WARNING: The ratings lack statistical significance. The EWMA alpha might need tuning or the raw ratings are flawed.")
end




#=
julia> println("TEST 2: Logistic Regression (Predicting Home Win)")
TEST 2: Logistic Regression (Predicting Home Win)

julia> println(coeftable(logit_model))
CoefTable(Any[[2.8022030463999346, 0.2409234896579257, -0.2821769042085349], [5.166925045950265, 0.05692597292576697, 0.053255527074824], [0.5423347583871468, 4.232224365705555, -5.298546830868399], [0.5875879150282083, 2.3139140382831126e-5, 1.1672794033833592e-7], [-7.324783954480566, 0.12935063293852003, -0.3865558192528879], [12.929190047280436, 0.3524963463773314, -0.17779798916418194]], ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"], ["(Intercept)", "home_rating", "away_rating"], 4, 3)

julia> p_vals = coeftable(logit_model).cols[4] # P-value column
3-element Vector{Float64}:
 0.5875879150282083
 2.3139140382831126e-5
 1.1672794033833592e-7

julia> if p_vals[2] < 0.05 && p_vals[3] < 0.05
           println("\n -> SUCCESS: Both Home and Away ratings are statistically significant predictors of the match outcome.")
       else
           println("\n -> WARNING: The ratings lack statistical significance. The EWMA alpha might need tuning or the raw ratings are flawed.")
       end

 -> SUCCESS: Both Home and Away ratings are statistically significant predictors of the match outcome.
=#



using DataFrames, Dates, Statistics, GLM, ShiftedArrays, Printf

println("==========================================================")
println(" EWMA ALPHA HYPERPARAMETER TUNING ")
println("==========================================================\n")

# 1. Base Setup (Run once outside the loop to save time)
lineups_time = innerjoin(
    dropmissing(ds.lineups, :rating), 
    select(ds.matches, :match_id, :match_date, :home_score, :away_score, :winner_code), 
    on = :match_id
)
sort!(lineups_time, :match_date)
global_avg = mean(skipmissing(lineups_time.rating))
lineups_time.mins = coalesce.(lineups_time.minutes_played, 0.0)

# The EWMA Function
function calc_pre_match_ewma(ratings::AbstractVector, alpha::Float64)
    n = length(ratings)
    baselines = zeros(Float64, n)
    baselines[1] = NaN 
    
    if n > 1
        current_ewma = Float64(ratings[1])
        for i in 2:n
            baselines[i] = current_ewma
            current_ewma = (alpha * ratings[i]) + ((1.0 - alpha) * current_ewma)
        end
    end
    return baselines
end

# 2. Define the Alphas to test (from very slow 0.02 to highly reactive 0.50)
alphas_to_test = 0.02:0.02:0.50

# 3. Create an empty DataFrame to store the results
alpha_results = DataFrame(
    Alpha = Float64[],
    AIC = Float64[],
    LogLikelihood = Float64[],
    Home_Coef = Float64[],
    Away_Coef = Float64[],
    Home_PValue = Float64[],
    Away_PValue = Float64[]
)

println("[INFO] Running Alpha Grid Search...")

# 4. The Experiment Loop
for test_alpha in alphas_to_test
    # A. Calculate EWMA for this specific alpha
    temp_lineups = copy(lineups_time)
    transform!(groupby(temp_lineups, :player_id), 
        :rating => (r -> calc_pre_match_ewma(r, test_alpha)) => :pre_match_ewma
    )
    temp_lineups.pre_match_ewma = coalesce.(replace(temp_lineups.pre_match_ewma, NaN => global_avg), global_avg)

    # B. Minute-weighting and aggregation
    temp_lineups.weighted_rating = temp_lineups.pre_match_ewma .* (clamp.(temp_lineups.mins, 0.0, 90.0) ./ 90.0)
    team_ratings = combine(groupby(temp_lineups, [:match_id, :team_side]), :weighted_rating => sum => :total_rating)

    home_ratings = rename!(filter(r -> r.team_side == "home", team_ratings), :total_rating => :home_rating)[!, [:match_id, :home_rating]]
    away_ratings = rename!(filter(r -> r.team_side == "away", team_ratings), :total_rating => :away_rating)[!, [:match_id, :away_rating]]

    # C. Join to match data
    df_model = innerjoin(ds.matches[!, [:match_id, :winner_code]], home_ratings, on=:match_id)
    df_model = innerjoin(df_model, away_ratings, on=:match_id)
    dropmissing!(df_model)
    df_model.is_home_win = df_model.winner_code .== 1

    # D. Fit the GLM
    logit_model = glm(@formula(is_home_win ~ home_rating + away_rating), df_model, Binomial(), LogitLink())
    
    # E. Extract Metrics
    ct = coeftable(logit_model)
    coefs = ct.cols[1]   # Column 1 is the Estimates
    pvals = ct.cols[4]   # Column 4 is the P-values
    
    ll = loglikelihood(logit_model)
    model_aic = aic(logit_model) # GLM package provides this natively

    # F. Save to Results DataFrame
    push!(alpha_results, (
        test_alpha, 
        model_aic, 
        ll, 
        coefs[2], # Home_Coef
        coefs[3], # Away_Coef
        pvals[2], # Home_PValue
        pvals[3]  # Away_PValue
    ))
end

# 5. Find the optimal Alpha (Lowest AIC)
sort!(alpha_results, :AIC)

println("\n[SUCCESS] Grid Search Complete. Top 5 Best Alphas by AIC:")
display(first(alpha_results, 5))

best_alpha = alpha_results[1, :Alpha]
println("\n-> Recommendation: Use alpha = $best_alpha for your Turing Model parameters.")



#=
julia> sort!(alpha_results, :AIC)
25×7 DataFrame
 Row │ Alpha    AIC      LogLikelihood  Home_Coef  Away_Coef  Home_PValue  Away_PValue 
     │ Float64  Float64  Float64        Float64    Float64    Float64      Float64     
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │    0.06  800.289       -397.144   0.251475  -0.3018    2.40235e-5    4.22049e-8
   2 │    0.04  800.357       -397.179   0.243418  -0.300846  4.13937e-5    3.78675e-8
   3 │    0.08  800.48        -397.24    0.253399  -0.299461  1.90063e-5    5.13094e-8
   4 │    0.1   800.864       -397.432   0.251977  -0.295409  1.78504e-5    6.44316e-8
   5 │    0.02  801.347       -397.673   0.219226  -0.291321  0.000160577   4.80113e-8
   6 │    0.12  801.404       -397.702   0.248479  -0.290425  1.87578e-5    8.18388e-8
   7 │    0.14  802.053       -398.027   0.243663  -0.284978  2.12802e-5    1.03884e-7
   8 │    0.16  802.769       -398.384   0.238028  -0.279361  2.54243e-5    1.3081e-7
   9 │    0.18  803.517       -398.758   0.231917  -0.27376   3.14192e-5    1.62687e-7
  10 │    0.2   804.273       -399.136   0.225565  -0.268289  3.96367e-5    1.99402e-7
  11 │    0.22  805.019       -399.51    0.219139  -0.263011  5.0555e-5     2.40691e-7
  12 │    0.24  805.745       -399.873   0.212754  -0.257959  6.47344e-5    2.86204e-7
  13 │    0.26  806.443       -400.222   0.20649   -0.253142  8.27928e-5    3.35585e-7
  14 │    0.28  807.111       -400.555   0.200399  -0.248554  0.000105379   3.88549e-7
  15 │    0.3   807.745       -400.873   0.194518  -0.244183  0.000133142   4.44948e-7
  16 │    0.32  808.348       -401.174   0.188867  -0.240008  0.000166698   5.04823e-7
  17 │    0.34  808.919       -401.46    0.183458  -0.23601   0.000206598   5.68426e-7
  18 │    0.36  809.462       -401.731   0.178294  -0.232166  0.000253301   6.36239e-7
  19 │    0.38  809.979       -401.989   0.173373  -0.228456  0.000307143   7.08978e-7
  20 │    0.4   810.471       -402.236   0.168691  -0.224859  0.000368323   7.87601e-7
  21 │    0.42  810.943       -402.471   0.16424   -0.221358  0.00043689    8.73311e-7
  22 │    0.44  811.396       -402.698   0.16001   -0.217936  0.000512737   9.67577e-7
  23 │    0.46  811.832       -402.916   0.155991  -0.21458   0.000595606   1.07216e-6
  24 │    0.48  812.255       -403.127   0.152172  -0.211276  0.000685093   1.18914e-6
  25 │    0.5   812.666       -403.333   0.148542  -0.208014  0.000780667   1.32101e-6

julia> best_alpha = alpha_results[1, :Alpha]
0.06
=#

