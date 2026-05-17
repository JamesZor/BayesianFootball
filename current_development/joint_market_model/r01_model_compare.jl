# current_development/joint_market_model/r01_model_compare.jl
include("./l00_inverse_problem.jl")

using Revise
using BayesianFootball
using DataFrames
using ThreadPinning
pinthreads(:cores)


# for running the models
const PreGame = BayesianFootball.Models.PreGame


# ==========================================
# 1. LOAD THE ROBUST COMPONENTS
# ==========================================
inter_cfg = PreGame.GlobalInterception()
# Note: Using HomeAwayDispersion based on your previous grid (unless you built a custom TeamDispersion!)
disp_cfg  = PreGame.HomeAwayDispersion() 
# "ha global"
ha_cfg    = PreGame.GlobalHomeAdvantage() 

dyn_cfg   = PreGame.MultiScaleGRW(
    z₀ = Normal(0, 1), 
      zₛ = Normal(0,1),   # The fat-tailed robust winner
    zₖ = Normal(0, 1)
)

model = PreGame.DynamicGoalsModel(
    interception_config  = inter_cfg,
    dynamics_config      = dyn_cfg,
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
)

save_dir::String = "./data/dev_inverse_model/"



cfgs = create_CVsplit_training_config(ds, get_target_seasons_string(ds.segment))
task_base    = build_experiment_task(ds, model,    "basic",    save_dir, cfgs)

# res_base    = Experiments.run_experiment(task_base.ds,    task_base.config)

Experiments.save_experiment(res_base)
###############

# xG model 
kap_base = PreGame.HierarchicalTeamKappa(
    κ_base = Normal(1.0, 0.2),
    σ_κ    = truncated(Normal(0, 0.1), lower=0.0) 
)

model_xg      = PreGame.DynamicXGModel(
  interception_config=inter_cfg,
  dynamics_config=dyn_cfg,
  dispersion_config=disp_cfg,
  homeadvantage_config=ha_cfg,
  kappa_config=kap_base
)

cfgs = create_CVsplit_training_config(ds, get_target_seasons_string(ds.segment))
task_base_xg    = build_experiment_task(ds, model_xg,    "basic_xg",    save_dir, cfgs)

res_base_xg    = Experiments.run_experiment(task_base_xg.ds,    task_base_xg.config)
Experiments.save_experiment(res_base_xg)


##### 



latents = Experiments.extract_oos_predictions(ds, res_base)

latents = Experiments.extract_oos_predictions(ds, res_base_xg)


matches = subset(ds.matches, :season => ByRow(isequal("2025")))



using DataFrames
using Statistics
using StatsPlots # Used for the EDA visualisations

# 1. Isolate the 
matches_2025 = subset(ds.matches, :season => ByRow(isequal("2025")))
odds_2025 = subset(ds.odds, :match_id => ByRow(in(matches_2025.match_id)))

# 2. Extract Market Parameters for all 2025 Matches
# Group the odds dataframe by match_id
grouped_odds = groupby(odds_2025, :match_id)

# Apply your fit_market_implied_parameters function to each group 
# and collect the NamedTuples directly into a new DataFrame
println("Fitting market parameters for $(length(grouped_odds)) matches...")
market_df = DataFrame([fit_market_implied_parameters(sub_df) for sub_df in grouped_odds])

# 3. Merge Market Data with Model Latents
analysis_df = innerjoin(market_df, latents.df, on=:match_id)

# 4. Calculate the Log Differences
# We calculate the mean of your posterior arrays first, then calculate the log difference
analysis_df = transform(analysis_df,
    :λ_h => ByRow(mean) => :model_λ_h_mean,
    :λ_a => ByRow(mean) => :model_λ_a_mean
)

analysis_df = transform(analysis_df,
    [:λ_home, :model_λ_h_mean] => ByRow((mkt, mod) -> log(mkt) - log(mod)) => :log_diff_h,
    [:λ_away, :model_λ_a_mean] => ByRow((mkt, mod) -> log(mkt) - log(mod)) => :log_diff_a
)

# Optional: View a summary of the residuals
println("\n--- Summary of Home Log Differences ---")
describe(analysis_df[:, [:log_diff_h]])
println("\n--- Summary of Away Log Differences ---")
describe(analysis_df[:, [:log_diff_a]])



#=

market_goals (month)
--- Summary of Home Log Differences ---

julia> describe(analysis_df[:, [:log_diff_h]])
1×7 DataFrame
 Row │ variable    mean       min        median     max       nmissing  eltype   
     │ Symbol      Float64    Float64    Float64    Float64   Int64     DataType 
─────┼───────────────────────────────────────────────────────────────────────────
   1 │ log_diff_h  0.0164489  -0.439191  0.0226689  0.327871         0  Float64

julia> println("\n--- Summary of Away Log Differences ---")

--- Summary of Away Log Differences ---

julia> describe(analysis_df[:, [:log_diff_a]])
1×7 DataFrame
 Row │ variable    mean        min        median      max      nmissing  eltype   
     │ Symbol      Float64     Float64    Float64     Float64  Int64     DataType 
─────┼────────────────────────────────────────────────────────────────────────────
   1 │ log_diff_a  0.00893412  -0.247067  0.00813164  0.54612         0  Float64


xg model 
--- Summary of Home Log Differences ---

julia> describe(analysis_df[:, [:log_diff_h]])
1×7 DataFrame
 Row │ variable    mean         min        median       max       nmissing  eltype   
     │ Symbol      Float64      Float64    Float64      Float64   Int64     DataType 
─────┼───────────────────────────────────────────────────────────────────────────────
   1 │ log_diff_h  -0.00618748  -0.425699  -0.00737084  0.283747         0  Float64

julia> println("\n--- Summary of Away Log Differences ---")

--- Summary of Away Log Differences ---

julia> describe(analysis_df[:, [:log_diff_a]])
1×7 DataFrame
 Row │ variable    mean       min        median     max       nmissing  eltype   
     │ Symbol      Float64    Float64    Float64    Float64   Int64     DataType 
─────┼───────────────────────────────────────────────────────────────────────────
   1 │ log_diff_a  0.0403007  -0.367895  0.0350981  0.612526         0  Float64
=#


# 5. Exploratory Data Analysis (EDA) Visualisation
# Plot histograms with overlaid Kernel Density Estimates (KDE) to check for Normality
#
# p1 = histogram(analysis_df.log_diff_h, 
#                normalize=:pdf, 
#                bins=20, 
#                title="Home λ Log Difference\n(Market vs Model)", 
#                label="Empirical", 
#                color=:steelblue, 
#                alpha=0.7)
# density!(p1, analysis_df.log_diff_h, label="KDE", color=:red, linewidth=2)
#
# p2 = histogram(analysis_df.log_diff_a, 
#                normalize=:pdf, 
#                bins=20, 
#                title="Away λ Log Difference\n(Market vs Model)", 
#                label="Empirical", 
#                color=:seagreen, 
#                alpha=0.7)
# density!(p2, analysis_df.log_diff_a, label="KDE", color=:red, linewidth=2)
#
# # Display side-by-side
# display(plot(p1, p2, layout=(1,2), size=(900, 400), margin=5Plots.mm))
#


using HypothesisTests
using Distributions

# Extract the vectors, dropping any potential missings
res_h = dropmissing(analysis_df).log_diff_h;
res_a = dropmissing(analysis_df).log_diff_a;

println("==========================================")
println(" 1. JARQUE-BERA TEST (Skewness & Kurtosis)")
println(" Null Hypothesis: Data is Normally Distributed")
println("==========================================")

jb_h = JarqueBeraTest(res_h)
jb_a = JarqueBeraTest(res_a)

println("\n--- HOME Log Differences ---")
println("p-value: ", round(pvalue(jb_h), digits=4))
if pvalue(jb_h) < 0.05
    println("Result: REJECT Null (Home is likely NOT perfectly Normal)")
else
    println("Result: FAIL TO REJECT (Home looks Normal)")
end

println("\n--- AWAY Log Differences ---")
println("p-value: ", round(pvalue(jb_a), digits=4))
if pvalue(jb_a) < 0.05
    println("Result: REJECT Null (Away is likely NOT perfectly Normal)")
else
    println("Result: FAIL TO REJECT (Away looks Normal)")
end




#=
xg model 
julia> jb_h = JarqueBeraTest(res_h)
Jarque-Bera normality test
--------------------------
Population details:
    parameter of interest:   skewness and kurtosis
    value under h_0:         "0 and 3"
    point estimate:          "-0.34207781533482734 and 2.976532440206826"

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.1725

Details:
    number of observations:         180
    JB statistic:                   3.51465


julia> jb_a = JarqueBeraTest(res_a)
Jarque-Bera normality test
--------------------------
Population details:
    parameter of interest:   skewness and kurtosis
    value under h_0:         "0 and 3"
    point estimate:          "0.013255432048962642 and 3.697880101248363"

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.1606

Details:
    number of observations:         180
    JB statistic:                   3.65805


julia> println("\n--- HOME Log Differences ---")

--- HOME Log Differences ---

julia> println("p-value: ", round(pvalue(jb_h), digits=4))
p-value: 0.1725

julia> if pvalue(jb_h) < 0.05
           println("Result: REJECT Null (Home is likely NOT perfectly Normal)")
       else
           println("Result: FAIL TO REJECT (Home looks Normal)")
       end
Result: FAIL TO REJECT (Home looks Normal)

julia> println("\n--- AWAY Log Differences ---")

--- AWAY Log Differences ---

julia> println("p-value: ", round(pvalue(jb_a), digits=4))
p-value: 0.1606

julia> if pvalue(jb_a) < 0.05
           println("Result: REJECT Null (Away is likely NOT perfectly Normal)")
       else
           println("Result: FAIL TO REJECT (Away looks Normal)")
       end
Result: FAIL TO REJECT (Away looks Normal)
=#



#=
# basic model
julia> jb_h = JarqueBeraTest(res_h)
Jarque-Bera normality test
--------------------------
Population details:
    parameter of interest:   skewness and kurtosis
    value under h_0:         "0 and 3"
    point estimate:          "-0.19118893660760014 and 2.9096907834447903"

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.5605

Details:
    number of observations:         180
    JB statistic:                   1.15776


julia> jb_a = JarqueBeraTest(res_a)
Jarque-Bera normality test
--------------------------
Population details:
    parameter of interest:   skewness and kurtosis
    value under h_0:         "0 and 3"
    point estimate:          "-0.12072929582022261 and 2.8042169503067242"

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.6960

Details:
    number of observations:         180
    JB statistic:                   0.724749
=#


#=
julia> println("\n--- HOME Log Differences ---")

--- HOME Log Differences ---

julia> println("p-value: ", round(pvalue(jb_h), digits=4))
p-value: 0.5605

julia> if pvalue(jb_h) < 0.05
           println("Result: REJECT Null (Home is likely NOT perfectly Normal)")
       else
           println("Result: FAIL TO REJECT (Home looks Normal)")
       end
Result: FAIL TO REJECT (Home looks Normal)

julia> println("\n--- AWAY Log Differences ---")

--- AWAY Log Differences ---

julia> println("p-value: ", round(pvalue(jb_a), digits=4))
p-value: 0.696

julia> if pvalue(jb_a) < 0.05
           println("Result: REJECT Null (Away is likely NOT perfectly Normal)")
       else
           println("Result: FAIL TO REJECT (Away looks Normal)")
       end
Result: FAIL TO REJECT (Away looks Normal)
=#



println("\n==========================================")
println(" 2. KOLMOGOROV-SMIRNOV TEST (CDF Comparison)")
println(" Null Hypothesis: Data matches a fitted Normal")
println("==========================================")

# Fit perfectly theoretical Normal distributions to your data's mean and std
dist_h = Normal(mean(res_h), std(res_h))
dist_a = Normal(mean(res_a), std(res_a))

ks_h = ExactOneSampleKSTest(res_h, dist_h)
ks_a = ExactOneSampleKSTest(res_a, dist_a)

println("\n--- HOME Log Differences ---")
println("p-value: ", round(pvalue(ks_h), digits=4))

println("\n--- AWAY Log Differences ---")
println("p-value: ", round(pvalue(ks_a), digits=4))




#=
# xg model 
julia> ks_h = ExactOneSampleKSTest(res_h, dist_h)
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0589969

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.5383

Details:
    number of observations:   180


julia> ks_a = ExactOneSampleKSTest(res_a, dist_a)
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0510828

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.7157

Details:
    number of observations:   180


julia> println("\n--- HOME Log Differences ---")

--- HOME Log Differences ---

julia> println("p-value: ", round(pvalue(ks_h), digits=4))
p-value: 0.5383

julia> println("\n--- AWAY Log Differences ---")

--- AWAY Log Differences ---

julia> println("p-value: ", round(pvalue(ks_a), digits=4))
p-value: 0.7157
=#


#=
julia> println("\n==========================================")

==========================================

julia> println(" 2. KOLMOGOROV-SMIRNOV TEST (CDF Comparison)")
 2. KOLMOGOROV-SMIRNOV TEST (CDF Comparison)

julia> println(" Null Hypothesis: Data matches a fitted Normal")
 Null Hypothesis: Data matches a fitted Normal

julia> println("==========================================")
==========================================

julia> # Fit perfectly theoretical Normal distributions to your data's mean and std
       dist_h = Normal(mean(res_h), std(res_h))
Normal{Float64}(μ=0.00784247530047668, σ=0.18010330602559418)

julia> dist_a = Normal(mean(res_a), std(res_a))
Normal{Float64}(μ=-0.017431194805143057, σ=0.1823545524224973)

julia> ks_h = ExactOneSampleKSTest(res_h, dist_h)
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0428741

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.8810

Details:
    number of observations:   180


julia> ks_a = ExactOneSampleKSTest(res_a, dist_a)
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0297839

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.9959

Details:
    number of observations:   180


julia> println("\n--- HOME Log Differences ---")

--- HOME Log Differences ---

julia> println("p-value: ", round(pvalue(ks_h), digits=4))
p-value: 0.881

julia> println("\n--- AWAY Log Differences ---")

--- AWAY Log Differences ---

julia> println("p-value: ", round(pvalue(ks_a), digits=4))
p-value: 0.9959
=#


# test for grouping of the winner_code



using DataFrames
using Statistics
using HypothesisTests

# Abstract function to print a clean console table of stats per group
function print_group_stats(df::DataFrame, target_col::Symbol, group_col::Symbol)
    outcomes = sort(unique(df[!, group_col]))
    
    # Print Headers
    println(rpad("Outcome", 12), rpad("Mean", 9), rpad("Median", 9), rpad("StdDev", 9), rpad("T-Test(p)", 12), "Normality_JB(p)")
    println("-"^65)
    
    for g in outcomes
        # Extract subset data
        subset_data = df[df[!, group_col] .== g, target_col]
        
        m = round(mean(subset_data), digits=4)
        med = round(median(subset_data), digits=4)
        sd = round(std(subset_data), digits=4)
        
        # T-Test: Is the mean residual statistically different from 0? 
        # (p < 0.05 means YES, there is a distinct market bias for this outcome)
        tt_p = round(pvalue(OneSampleTTest(subset_data)), digits=4)
        
        # Jarque-Bera Test: Is the distribution normal?
        # (p < 0.05 means NO, there are extreme fat tails/outliers)
        jb_p = round(pvalue(JarqueBeraTest(subset_data)), digits=4)
        
        # Map outcome codes for readability
        label = g == 1 ? "1: Home" : (g == 2 ? "2: Away" : "3: Draw")
        
        println(rpad(label, 12), rpad(m, 9), rpad(med, 9), rpad(sd, 9), rpad(tt_p, 12), jb_p)
    end
end

function run_headless_eda(eda_df::DataFrame)
    # 1. ENGINEER THE TOTAL GOALS (PACE) FEATURES
    # We use the raw lambdas here, not log-lambdas, for absolute goals difference
    eda_df.model_total = eda_df.model_λ_h_mean .+ eda_df.model_λ_a_mean
    eda_df.market_total = eda_df.λ_home .+ eda_df.λ_away
    eda_df.diff_total = eda_df.market_total .- eda_df.model_total
    
    println("\n=================================================================")
    println(" 1. TOTAL GOALS (PACE) BIAS CHECK")
    println("=================================================================")
    display(describe(eda_df[:, [:model_total, :market_total, :diff_total]]))
    
    # Test if the Market expects significantly more/fewer goals than the Model overall
    tt_total = OneSampleTTest(eda_df.diff_total)
    jb_total = JarqueBeraTest(eda_df.diff_total)
    
    println("\n-> Is the Market biased on Total Goals? (T-Test p-value): ", round(pvalue(tt_total), digits=4))
    if pvalue(tt_total) < 0.05
        println("   RESULT: REJECT NULL. The Market systematically expects a different pace than the Model.")
    else
        println("   RESULT: FAIL TO REJECT. Market and Model agree on overall league pace.")
    end
    
    println("-> Are the Pace differences Normal? (JB p-value):       ", round(pvalue(jb_total), digits=4))

    println("\n=================================================================")
    println(" 2. HOME LOG-RESIDUALS BY MATCH OUTCOME")
    println("=================================================================")
    print_group_stats(eda_df, :log_diff_h, :winner_code)

    println("\n=================================================================")
    println(" 3. AWAY LOG-RESIDUALS BY MATCH OUTCOME")
    println("=================================================================")
    print_group_stats(eda_df, :log_diff_a, :winner_code)
end



analysis_df_matches= innerjoin(analysis_df, matches , on=:match_id)

# Execute the script on your dataframe
run_headless_eda(analysis_df_matches)


#=
julia> run_headless_eda(analysis_df_matches)                                                                                          
                                                                                                                                      
=================================================================                                                                     
 1. TOTAL GOALS (PACE) BIAS CHECK                                                                                                     
=================================================================                                                                     
3×7 DataFrame                                                                                                                         
 Row │ variable      mean       min        median     max       nmissing  eltype                                                      
     │ Symbol        Float64    Float64    Float64    Float64   Int64     DataType                                                    
─────┼─────────────────────────────────────────────────────────────────────────────                                                   
   1 │ model_total   2.42085     1.88752   2.42095    3.22739          0  Float64                                                     
   2 │ market_total  2.4967      1.92836   2.47972    3.20496          0  Float64                                                     
   3 │ diff_total    0.0758463  -0.463504  0.0767566  0.598324         0  Float64                                                     
                                                                                                                                      
-> Is the Market biased on Total Goals? (T-Test p-value): 0.0                                                                         
   RESULT: REJECT NULL. The Market systematically expects a different pace than the Model.                                            
-> Are the Pace differences Normal? (JB p-value):       0.8926                                                                        
                                                                                                                                      
=================================================================                                                                     
 2. HOME LOG-RESIDUALS BY MATCH OUTCOME                                                                                               
=================================================================                                                                     
Outcome     Mean     Median   StdDev   T-Test(p)   Normality_JB(p)                                                                    
-----------------------------------------------------------------                                                                     
1: Home     0.0217   0.0125   0.124    0.1292      0.7845                                                                             
2: Away     -0.0669  -0.0757  0.1482   0.0022      0.3236                                                                             
3: Draw     0.0121   0.0331   0.1452   0.5492      0.329                                                                              
                                                                                                                                      
=================================================================                                                                     
 3. AWAY LOG-RESIDUALS BY MATCH OUTCOME                                                                                               
=================================================================
Outcome     Mean     Median   StdDev   T-Test(p)   Normality_JB(p)
-----------------------------------------------------------------
1: Home     0.0031   0.006    0.1492   0.8564      0.8637
2: Away     0.0807   0.0655   0.1631   0.0009      0.0817
3: Draw     0.0558   0.0427   0.1472   0.0086      0.8727
=#


function check_raw_lambda_bias(df::DataFrame)
    # Calculate the raw differences (Market minus Model)
    df.diff_home  = df.λ_home .- df.model_λ_h_mean
    df.diff_away  = df.λ_away .- df.model_λ_a_mean
    df.diff_total = (df.λ_home .+ df.λ_away) .- (df.model_λ_h_mean .+ df.model_λ_a_mean)
    
    metrics = [
        ("HOME GOALS", :λ_home, :model_λ_h_mean, :diff_home),
        ("AWAY GOALS", :λ_away, :model_λ_a_mean, :diff_away),
        ("TOTAL GOALS", :market_total, :model_total, :diff_total) # Assuming market_total/model_total exist from earlier
    ]
    
    # Just in case they don't exist yet
    df.market_total = df.λ_home .+ df.λ_away
    df.model_total = df.model_λ_h_mean .+ df.model_λ_a_mean
    
    println("\n=================================================================")
    println(" RAW LAMBDA BIAS: MARKET vs MODEL (OVERALL)")
    println("=================================================================")
    println(rpad("Metric", 15), rpad("Market Mean", 15), rpad("Model Mean", 15), rpad("Diff (Bias)", 15), "T-Test (p-value)")
    println("-"^75)
    
    for (name, mkt_col, mod_col, diff_col) in metrics
        mkt_mean = round(mean(df[!, mkt_col]), digits=4)
        mod_mean = round(mean(df[!, mod_col]), digits=4)
        diff_mean = round(mean(df[!, diff_col]), digits=4)
        
        # T-Test to see if the difference is statistically significant from 0
        tt_p = round(pvalue(OneSampleTTest(df[!, diff_col])), digits=4)
        
        println(rpad(name, 15), rpad(mkt_mean, 15), rpad(mod_mean, 15), rpad(diff_mean, 15), tt_p)
    end
    
    println("-"^75)
    println("* Note: A positive Diff means the Market expects MORE goals than your Model.")
    println("* Note: A p-value < 0.05 proves the bias is structural, not random noise.\n")
end

check_raw_lambda_bias(eda_df)





# Test for the rho 
using GLM

# 1. Create the fundamental features from the market lambdas
# Note: We use the market's lambda, because rho is priced against the market's expectation
analysis_df.sum_lambda = analysis_df.λ_home .+ analysis_df.λ_away
analysis_df.diff_lambda = abs.(analysis_df.λ_home .- analysis_df.λ_away)

# 2. Fit a Linear Regression Model
# We are asking: How much does Total Goals and Competitiveness predict ρ?
rho_formula = @formula(ρ ~ sum_lambda + diff_lambda)
rho_model = lm(rho_formula, analysis_df)

# 3. View the Bookmaker's "Secret" Weights
println("==========================================")
println(" BOOKMAKER RHO CALCULATION FORMULA")
println("==========================================")
println(coeftable(rho_model))




#=
julia> rho_model = lm(rho_formula, analysis_df)
StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

ρ ~ 1 + sum_lambda + diff_lambda

Coefficients:
─────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)    Lower 95%   Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.0918432  0.0332705   -2.76    0.0064  -0.157501    -0.0261852
sum_lambda    0.0150537  0.0142864    1.05    0.2935  -0.0131398    0.0432472
diff_lambda   0.0104554  0.00890131   1.17    0.2417  -0.00711093   0.0280218
─────────────────────────────────────────────────────────────────────────────

julia>
=#


# We formulate rho as a function of the categorical team names
# 1. Grab just the match_id and team names from the main matches dataset
matches_teams = ds.matches[:, [:match_id, :home_team, :away_team]]

# 2. Join them into your analysis_df
analysis_df_with_teams = innerjoin(analysis_df, matches_teams, on=:match_id)

# 3. Now run the One-Hot Encoded Regression
using GLM

team_rho_formula = @formula(ρ ~ home_team + away_team)
team_rho_model = lm(team_rho_formula, analysis_df_with_teams)

println("==========================================")
println(" TEAM-DRIVEN RHO REGRESSION")
println("==========================================")
println(coeftable(team_rho_model))






# --- rho 
using DataFrames
using Statistics
using StatsPlots
using LinearAlgebra

# 1. Merge the Market Rho with the Match Metadata and Odds Metadata
# (Assuming analysis_df from earlier and your raw ds.matches / ds.odds)

# First, get the overround from the 1X2 home odds to represent market margin
odds_margin = subset(odds_2025, :selection => ByRow(isequal(:home)))[:, [:match_id, :overround_close]]

# Join everything together
eda_df = innerjoin(analysis_df, ds.matches[:, [:match_id, :match_week]], on=:match_id)
eda_df = innerjoin(eda_df, odds_margin, on=:match_id)

# 2. Engineer the numeric features
eda_df.sum_lambda = eda_df.λ_home .+ eda_df.λ_away
eda_df.diff_lambda = abs.(eda_df.λ_home .- eda_df.λ_away)

# 3. Select only the numeric columns for the correlation matrix
features = [:ρ, :sum_lambda, :diff_lambda, :match_week, :overround_close]
corr_data = Matrix(eda_df[:, features])

# 4. Calculate the Pearson Correlation Matrix
C = cor(corr_data)

# 5. Print a clean console table
function display_corr()
  println("==========================================")
  println(" RHO FEATURE CORRELATION MATRIX")
  println("==========================================")
  for (i, feat) in enumerate(features)
      # Print the correlation of Rho against all other features
      println(rpad(feat, 20), " | ", round(C[1, i], digits=4))
  end
end
display_corr()

# 6. Plot the Heatmap for visual EDA
heatmap(string.(features), string.(features), C, 
        aspect_ratio=1, 
        color=:RdBu, 
        clim=(-1, 1), 
        title="Market Feature Correlations",
        xrotation=45, 
        fontsize=10,
        size=(600, 500))
        
# Add the correlation numbers directly onto the heatmap
for i in 1:length(features), j in 1:length(features)
    annotate!(j, i, text(round(C[i,j], digits=2), 8, :black, :center))
end


# ---
using DataFrames
using Statistics

# 1. Ensure we are ONLY working with the 2025 season and sort it chronologically
# (matches_2025 was created in our very first step)
sort!(matches_2025, :match_date)

# 2. Initialize a fresh dictionary for 2025
team_stats = Dict{String, Tuple{Int, Int, Int}}()
unique_teams = unique(vcat(matches_2025.home_team, matches_2025.away_team))
for team in unique_teams
    team_stats[team] = (0, 0, 0) # (Points, GF, GA)
end

# 3. Prepare fresh arrays
home_pts = Int[]; away_pts = Int[]
home_gd = Int[]; away_gd = Int[]

# 4. Iterate through 2025 matches to build Rolling Stats
for row in eachrow(matches_2025)
    h_team = row.home_team
    a_team = row.away_team
    
    # Extract current stats BEFORE the game
    hp, hgf, hga = team_stats[h_team]
    ap, agf, aga = team_stats[a_team]
    
    push!(home_pts, hp)
    push!(home_gd, hgf - hga)
    push!(away_pts, ap)
    push!(away_gd, agf - aga)
    
    # Update the dictionary WITH this game's results
    if !ismissing(row.home_score) && !ismissing(row.away_score)
        h_score = row.home_score
        a_score = row.away_score
        
        h_points_earned = h_score > a_score ? 3 : (h_score == a_score ? 1 : 0)
        a_points_earned = a_score > h_score ? 3 : (h_score == a_score ? 1 : 0)
        
        team_stats[h_team] = (hp + h_points_earned, hgf + h_score, hga + a_score)
        team_stats[a_team] = (ap + a_points_earned, agf + a_score, aga + h_score)
    end
end

# 5. Attach corrected features to the 2025 dataframe
matches_2025.home_points_entering = home_pts
matches_2025.away_points_entering = away_pts
matches_2025.points_diff = home_pts .- away_pts
matches_2025.gd_diff = home_gd .- away_gd

# 6. Re-Merge with eda_df
# Select only the columns we need to avoid clutter
context_df = matches_2025[:, [:match_id, :points_diff, :gd_diff]]
eda_df_advanced = innerjoin(eda_df, context_df, on=:match_id)

# 7. Run the Corrected Correlation Matrix
advanced_features = [:ρ, :sum_lambda, :diff_lambda, :match_week, :points_diff, :gd_diff, :overround_close]
corr_data_adv = Matrix(eda_df_advanced[:, advanced_features])

C_adv = cor(corr_data_adv)

function displau_corr()
  println("==========================================")
  println(" ADVANCED CONTEXTUAL CORRELATION MATRIX")
  println("==========================================")
  for (i, feat) in enumerate(advanced_features)
      println(rpad(feat, 18), " | ", round(C_adv[1, i], digits=4))
  end
end 


displau_corr()

