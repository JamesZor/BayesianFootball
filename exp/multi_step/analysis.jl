
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
exp_dir = "./data/exp/multi_step"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/multi_step"; data_dir="./data")
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







exp_res = loaded_results[3]
exp_res2 = loaded_results[1]



latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
latents_raw2 = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res2)



joined = innerjoin(
    select(latents_raw.df, :match_id, :λ_h, :λ_a, :r),
    select(ds.matches, :match_id,:match_month, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
    on = :match_id
)



joined2 = innerjoin(
    select(latents_raw2.df, :match_id, :λ_h, :λ_a, :r_h, :r_a),
    select(ds.matches, :match_id,:match_month, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
    on = :match_id
)

###
using HypothesisTests
using DataFrames
using Statistics
using Printf

"""
    prepare_metrics!(df)

Calculates expectations, residuals, and cumulative sums in-place.
"""
function prepare_metrics!(df)
    # Calculate means of latent distributions
    df.exp_home = mean.(df.λ_h)
    df.exp_away = mean.(df.λ_a)
    df.exp_all  = df.exp_home .+ df.exp_away

    # Calculate individual residuals
    df.res_home = df.home_score .- df.exp_home
    df.res_away = df.away_score .- df.exp_away
    df.res_all  = (df.home_score .+ df.away_score) .- df.exp_all

    # Cumulative residuals (useful for plotting drift)
    df.cum_res_home = cumsum(df.res_home)
    df.cum_res_away = cumsum(df.res_away)
    df.cum_res_all  = cumsum(df.res_all)
    
    return df
end

"""
    run_diagnostic_test(df, label="Experiment")

Prints a structured summary of residuals and T-test results for a given experiment.
"""
function run_diagnostic_test(df, label="Experiment")
    prepare_metrics!(df)
    
    # Perform the T-test on total goal residuals
    t_test = OneSampleTTest(df.res_all)
    pv = pvalue(t_test)
    conf = confint(t_test)

    println("="^40)
    println(" DIAGNOSTIC REPORT: $label")
    println("="^40)
    
    # Summary Table Style
    @printf("%-15s | %-10s | %-10s\n", "Metric", "Sum Res", "Mean Res")
    println("-"^40)
    @printf("%-15s | %10.3f | %10.4f\n", "Home Goals", sum(df.res_home), mean(df.res_home))
    @printf("%-15s | %10.3f | %10.4f\n", "Away Goals", sum(df.res_away), mean(df.res_away))
    @printf("%-15s | %10.3f | %10.4f\n", "Total Goals", sum(df.res_all), mean(df.res_all))
    
    println("-"^40)
    println("One-Sample T-Test (Residuals == 0):")
    @printf("  t-statistic: %.4f\n", t_test.t)
    @printf("  p-value:     %.4e %s\n", pv, pv < 0.05 ? "(* Significant)" : "(Not Significant)")
    @printf("  95%% CI:      [%.4f, %.4f]\n", conf[1], conf[2])
    println("="^40, "\n")
    
    return (t_test = t_test, summary = describe(df[!, [:res_home, :res_away, :res_all]]))
end

# --- Execution ---

# Process experiment 1
res1 = run_diagnostic_test(joined, "Model A (baseline)")


# Process experiment 2
res2 = run_diagnostic_test(joined2, "Model B (Delta)")


#=


RHO model 
julia> res2 = run_diagnostic_test(joined2, "Model B (Delta)")
========================================
 DIAGNOSTIC REPORT: Model B (Delta)
========================================
Metric          | Sum Res    | Mean Res  
----------------------------------------
Home Goals      |     68.328 |     0.0595
Away Goals      |      4.739 |     0.0041
Total Goals     |     73.067 |     0.0636
----------------------------------------
One-Sample T-Test (Residuals == 0):
  t-statistic: 1.3165
  p-value:     1.8828e-01 (Not Significant)
  95% CI:      [-0.0312, 0.1585]
========================================

(t_test = One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          0.0636468
    95% confidence interval: (-0.03121, 0.1585)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.1883

Details:
    number of observations:   1148
    t-statistic:              1.3164823631217177
    degrees of freedom:       1147
    empirical standard error: 0.04834613857191845
, summary = 3×7 DataFrame
 Row │ variable  mean        min       median      max      nmissing  eltype   
     │ Symbol    Float64     Float64   Float64     Float64  Int64     DataType 
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ res_home  0.0595192   -2.18727  -0.185309   6.64915         0  Float64
   2 │ res_away  0.00412766  -2.1642   -0.222698   5.24093         0  Float64
   3 │ res_all   0.0636468   -3.32947  -0.0577664  5.55459         0  Float64)



julia> res1 = run_diagnostic_test(joined, "Model A (baseline)")
========================================
 DIAGNOSTIC REPORT: Model A (baseline)
========================================
Metric          | Sum Res    | Mean Res  
----------------------------------------
Home Goals      |     59.765 |     0.0521
Away Goals      |      2.832 |     0.0025
Total Goals     |     62.597 |     0.0545
----------------------------------------
One-Sample T-Test (Residuals == 0):
  t-statistic: 1.1273
  p-value:     2.5987e-01 (Not Significant)
  95% CI:      [-0.0404, 0.1494]
========================================

(t_test = One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          0.054527
    95% confidence interval: (-0.04038, 0.1494)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.2599

Details:
    number of observations:   1148
    t-statistic:              1.1272658650511744
    degrees of freedom:       1147
    empirical standard error: 0.048370984144518854
, summary = 3×7 DataFrame
 Row │ variable  mean        min       median      max      nmissing  eltype   
     │ Symbol    Float64     Float64   Float64     Float64  Int64     DataType 
─────┼─────────────────────────────────────────────────────────────────────────
   1 │ res_home  0.0520598   -2.1751   -0.228163   6.58571         0  Float64
   2 │ res_away  0.00246718  -2.16013  -0.22394    5.18943         0  Float64
   3 │ res_all   0.054527    -3.29679  -0.0942759  5.74913         0  Float64)

julia> # Process experiment 2
       res2 = run_diagnostic_test(joined2, "Model B (Delta)")
========================================
 DIAGNOSTIC REPORT: Model B (Delta)
========================================
Metric          | Sum Res    | Mean Res  
----------------------------------------
Home Goals      |     71.340 |     0.0621
Away Goals      |     13.180 |     0.0115
Total Goals     |     84.520 |     0.0736
----------------------------------------
One-Sample T-Test (Residuals == 0):
  t-statistic: 1.5191
  p-value:     1.2900e-01 (Not Significant)
  95% CI:      [-0.0215, 0.1687]
========================================

(t_test = One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          0.0736237
    95% confidence interval: (-0.02146, 0.1687)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.1290

Details:
    number of observations:   1148
    t-statistic:              1.5191419449628543
    degrees of freedom:       1147
    empirical standard error: 0.04846397493167676
, summary = 3×7 DataFrame
 Row │ variable  mean       min       median      max      nmissing  eltype   
     │ Symbol    Float64    Float64   Float64     Float64  Int64     DataType 
─────┼────────────────────────────────────────────────────────────────────────
   1 │ res_home  0.0621426  -2.17789  -0.190557   6.6334          0  Float64
   2 │ res_away  0.0114811  -2.14954  -0.21056    5.26032         0  Float64
   3 │ res_all   0.0736237  -3.31055  -0.0839622  5.56908         0  Float64)



# ----------------------------
#  small one season version 
# ----------------------------
julia> res1 = run_diagnostic_test(joined, "Model A (baseline)")
========================================
 DIAGNOSTIC REPORT: Model A (baseline)
========================================
Metric          | Sum Res    | Mean Res  
----------------------------------------
Home Goals      |     -0.093 |    -0.0003
Away Goals      |    -19.479 |    -0.0607
Total Goals     |    -19.572 |    -0.0610
----------------------------------------
One-Sample T-Test (Residuals == 0):
  t-statistic: -0.6792
  p-value:     4.9752e-01 (Not Significant)
  95% CI:      [-0.2376, 0.1156]
========================================

(t_test = One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          -0.0609708
    95% confidence interval: (-0.2376, 0.1156)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.4975

Details:
    number of observations:   321
    t-statistic:              -0.6791654992103433
    degrees of freedom:       320
    empirical standard error: 0.08977311328112551
, summary = 3×7 DataFrame
 Row │ variable  mean          min       median     max      nmissing  eltype   
     │ Symbol    Float64       Float64   Float64    Float64  Int64     DataType 
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ res_home  -0.000289001  -2.21798  -0.207881  4.74252         0  Float64
   2 │ res_away  -0.0606818    -1.884    -0.215105  4.94383         0  Float64
   3 │ res_all   -0.0609708    -3.36935  -0.285378  4.55148         0  Float64)


julia> res2 = run_diagnostic_test(joined2, "Model B (Delta)")
========================================
 DIAGNOSTIC REPORT: Model B (Delta)
========================================
Metric          | Sum Res    | Mean Res  
----------------------------------------
Home Goals      |      5.101 |     0.0159
Away Goals      |    -14.536 |    -0.0453
Total Goals     |     -9.435 |    -0.0294
----------------------------------------
One-Sample T-Test (Residuals == 0):
  t-statistic: -0.3324
  p-value:     7.3980e-01 (Not Significant)
  95% CI:      [-0.2033, 0.1446]
========================================

(t_test = One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          -0.0293916
    95% confidence interval: (-0.2033, 0.1446)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.7398

Details:
    number of observations:   321
    t-statistic:              -0.33240870077245727
    degrees of freedom:       320
    empirical standard error: 0.0884200524460494
, summary = 3×7 DataFrame
 Row │ variable  mean        min       median     max      nmissing  eltype   
     │ Symbol    Float64     Float64   Float64    Float64  Int64     DataType 
─────┼────────────────────────────────────────────────────────────────────────
   1 │ res_home   0.0158913  -2.03798  -0.214964  4.79851         0  Float64
   2 │ res_away  -0.0452829  -1.76834  -0.227907  4.86754         0  Float64
   3 │ res_all   -0.0293916  -3.10329  -0.296464  4.39839         0  Float64)



=#


using Distributions, Random, DataFrames, Statistics, HypothesisTests, Plots, StatsPlots, Printf

# 1. Core RQR Calculation (Unchanged)
function compute_rqr(y::Int, λ::Float64, r_disp::Float64)
    p = r_disp / (r_disp + λ)
    dist = NegativeBinomial(r_disp, p)
    
    cdf_lower = y > 0 ? cdf(dist, y - 1) : 0.0
    cdf_upper = cdf(dist, y)
    
    u = rand(Uniform(cdf_lower, cdf_upper))
    
    # Clamp u slightly to avoid Inf/-Inf in extreme edge cases
    u = clamp(u, 1e-7, 1.0 - 1e-7) 
    return quantile(Normal(0, 1), u)
end

function get_r(df)
  if hasproperty(df, :r)
    return mean.(df.r), mean.(df.r)
  elseif hasproperty(df, :r_h)
    return mean.(df.r_h), mean.(df.r_a) 
  else
      throw(ArgumentError("Row does not contain expected shape parameters (:r or :rₕ)"))
  end 
end
    


# 2. Append RQR to DataFrame
function append_rqr_metrics!(df)
    # Ensure expected values are present
    if !("exp_home" in names(df))
        df.exp_home = mean.(df.λ_h)
        df.exp_away = mean.(df.λ_a)
    end

    df.exp_r_h, df.exp_r_a = get_r(df)
    
    df.rqr_home = compute_rqr.(df.home_score, df.exp_home, df.exp_r_h)
    df.rqr_away = compute_rqr.(df.away_score, df.exp_away, df.exp_r_a)
    return df
end

# 3. Normality Diagnostics Printout
function print_rqr_diagnostics(df, label="Model")
    println("="^50)
    println(" RQR NORMALITY DIAGNOSTICS: $label")
    println("="^50)
    
    @printf("%-15s | %-12s | %-12s\n", "Metric", "Home RQR", "Away RQR")
    println("-"^50)
    @printf("%-15s | %12.4f | %12.4f\n", "Mean (Target 0)", mean(df.rqr_home), mean(df.rqr_away))
    @printf("%-15s | %12.4f | %12.4f\n", "StdDev(Target 1)", std(df.rqr_home), std(df.rqr_away))
    @printf("%-15s | %12.4f | %12.4f\n", "Skewness", skewness(df.rqr_home), skewness(df.rqr_away))
    @printf("%-15s | %12.4f | %12.4f\n", "Exc. Kurtosis", kurtosis(df.rqr_home), kurtosis(df.rqr_away))
    println("-"^50)
    
    # Shapiro-Wilk Tests
    sw_home = ShapiroWilkTest(df.rqr_home)
    sw_away = ShapiroWilkTest(df.rqr_away)
    
    @printf("Shapiro-Wilk W: Home = %.4f (p=%.3f)\n", sw_home.W, pvalue(sw_home))
    @printf("Shapiro-Wilk W: Away = %.4f (p=%.3f)\n", sw_away.W, pvalue(sw_away))
    println("="^50, "\n")
end

# 4. Monthly Aggregation
function aggregate_monthly_rqr(df, month_col=:match_month)
    df_home = DataFrame(month = df[!, month_col], rqr = df.rqr_home)
    df_away = DataFrame(month = df[!, month_col], rqr = df.rqr_away)
    df_pooled = vcat(df_home, df_away)
    
    monthly = combine(
        groupby(df_pooled, :month),
        :rqr => mean => :mean_rqr,
        :rqr => std => :std_rqr,
        :rqr => skewness => :skewness, 
        :rqr => kurtosis => :kurtosis, 
        :rqr => (x -> ShapiroWilkTest(x).W) => :SW_W,
        nrow => :n_obs
    )
    
    sort!(monthly, :month)
    monthly.margin_of_error = 2 .* (monthly.std_rqr ./ sqrt.(monthly.n_obs))
    
    return monthly
end

# 5. Comparative Plotting
function plot_model_comparison(monthly_base, monthly_delta)
    p = plot(
        title = "RQR Temporal Bias: Baseline vs Delta Model",
        xlabel = "Month of Season",
        ylabel = "Mean RQR (Target: 0.0)",
        size = (1400, 800),
        legend = :topright
    );
    
    # Add Baseline
    plot!(p, monthly_base.month, monthly_base.mean_rqr, 
        ribbon = monthly_base.margin_of_error, fillalpha = 0.2, 
        color = :red, linewidth = 2, marker = :circle, 
        label = "Baseline (Model A)");
        
    # Add Delta Model
    plot!(p, monthly_delta.month, monthly_delta.mean_rqr, 
        ribbon = monthly_delta.margin_of_error, fillalpha = 0.3, 
        color = :blue, linewidth = 3, marker = :diamond, 
        label = "Delta w/ Month Params (Model B)");

    # Zero-bias line
    hline!(p, [0.0], line=(:black, 2, :dash), label="Ideal Bias = 0.0");
    
    return p
end


# 1. Calculate RQRs
append_rqr_metrics!(joined)
append_rqr_metrics!(joined2)

# 2. Print Normal Diagnostics
print_rqr_diagnostics(joined, "Model A (Baseline)")
print_rqr_diagnostics(joined2, "Model B (Delta Midweek/Month)")


#=
julia> print_rqr_diagnostics(joined, "Model A (Baseline)")
==================================================
 RQR NORMALITY DIAGNOSTICS: Model A (Baseline)
==================================================
Metric          | Home RQR     | Away RQR    
--------------------------------------------------
Mean (Target 0) |       0.0371 |      -0.0182
StdDev(Target 1) |       0.9994 |       1.0329
Skewness        |       0.0164 |       0.0575
Exc. Kurtosis   |       0.2422 |       0.1388
--------------------------------------------------
Shapiro-Wilk W: Home = 0.9978 (p=0.132)
Shapiro-Wilk W: Away = 0.9989 (p=0.712)
==================================================

# Model RHO
julia> print_rqr_diagnostics(joined2, "Model B (Delta Midweek/Month)")
==================================================
 RQR NORMALITY DIAGNOSTICS: Model B (Delta Midweek/Month)
==================================================
Metric          | Home RQR     | Away RQR    
--------------------------------------------------
Mean (Target 0) |       0.0520 |       0.0126
StdDev(Target 1) |       0.9969 |       1.0099
Skewness        |       0.0250 |       0.1229
Exc. Kurtosis   |       0.1291 |       0.2248
--------------------------------------------------
Shapiro-Wilk W: Home = 0.9985 (p=0.453)
Shapiro-Wilk W: Away = 0.9978 (p=0.138)
==================================================







#-----------------------------
# full model 
#  ------------------------
julia> print_rqr_diagnostics(joined, "Model A (Baseline)")
==================================================
 RQR NORMALITY DIAGNOSTICS: Model A (Baseline)
==================================================
Metric          | Home RQR     | Away RQR    
--------------------------------------------------
Mean (Target 0) |       0.0586 |      -0.0063
StdDev(Target 1) |       0.9887 |       1.0230
Skewness        |       0.0360 |       0.0744
Exc. Kurtosis   |       0.0999 |       0.1382
--------------------------------------------------
Shapiro-Wilk W: Home = 0.9991 (p=0.880)
Shapiro-Wilk W: Away = 0.9986 (p=0.464)
==================================================


julia> print_rqr_diagnostics(joined2, "Model B (Delta Midweek/Month)")
==================================================
 RQR NORMALITY DIAGNOSTICS: Model B (Delta Midweek/Month)
==================================================
Metric          | Home RQR     | Away RQR    
--------------------------------------------------
Mean (Target 0) |       0.0637 |       0.0109
StdDev(Target 1) |       0.9701 |       1.0267
Skewness        |       0.1695 |       0.1763
Exc. Kurtosis   |       0.0258 |      -0.0486
--------------------------------------------------
Shapiro-Wilk W: Home = 0.9976 (p=0.092)
Shapiro-Wilk W: Away = 0.9969 (p=0.023)
==================================================


#-----------------------------
# small data set - one season 
#  ------------------------

julia> print_rqr_diagnostics(joined, "Model A (Baseline)")
==================================================
 RQR NORMALITY DIAGNOSTICS: Model A (Baseline)
==================================================
Metric          | Home RQR     | Away RQR    
--------------------------------------------------
Mean (Target 0) |       0.0277 |      -0.0487
StdDev (Target 1) |       0.9931 |       1.0521
Skewness        |       0.1890 |       0.1202
Exc. Kurtosis   |       0.0555 |      -0.2863
--------------------------------------------------
Shapiro-Wilk W: Home = 0.9947 (p=0.332)
Shapiro-Wilk W: Away = 0.9954 (p=0.453)
==================================================


julia> print_rqr_diagnostics(joined2, "Model B (Delta Midweek/Month)")
==================================================
 RQR NORMALITY DIAGNOSTICS: Model B (Delta Midweek/Month)
==================================================
Metric          | Home RQR     | Away RQR    
--------------------------------------------------
Mean (Target 0) |      -0.0005 |      -0.0510
StdDev (Target 1) |       0.9910 |       1.0534
Skewness        |       0.1242 |       0.0393
Exc. Kurtosis   |      -0.0052 |      -0.2230
--------------------------------------------------
Shapiro-Wilk W: Home = 0.9974 (p=0.893)
Shapiro-Wilk W: Away = 0.9969 (p=0.792)
==================================================

=#

# 3. Aggregate Monthly Data 
# (Note: Using :match_month based on your innerjoin from earlier)
monthly_base = aggregate_monthly_rqr(joined, :match_month)
monthly_delta = aggregate_monthly_rqr(joined2, :match_month)

#=
julia> monthly_base = aggregate_monthly_rqr(joined, :match_month)
9×8 DataFrame
 Row │ month  mean_rqr    std_rqr   skewness      kurtosis    SW_W      n_obs  margin_of_error 
     │ Int64  Float64     Float64   Float64       Float64     Float64   Int64  Float64         
─────┼─────────────────────────────────────────────────────────────────────────────────────────
   1 │     2   0.0495176  1.03047   -0.0320057     0.59821    0.993481    316         0.115937
   2 │     3  -0.012666   0.981722   0.219933     -0.244273   0.994085    300         0.113359
   3 │     4   0.0578872  1.00937   -0.000467946  -0.187963   0.992745    280         0.120643
   4 │     5   0.0888295  1.03647   -0.0161793    -0.0317986  0.994507    272         0.12569
   5 │     6  -0.0213203  0.977597   0.208205     -0.237638   0.992035    258         0.121725
   6 │     7  -0.0215345  1.02818   -0.161619      0.41082    0.993128    292         0.120339
   7 │     8  -0.0205412  1.04505    0.144548      0.477291   0.993398    254         0.131145
   8 │     9  -0.064044   1.05008   -0.00432984    0.366361   0.994919    254         0.131776
   9 │    10   0.0401032  0.927577   0.00110081    0.67835    0.98922      70         0.221733

julia> monthly_delta = aggregate_monthly_rqr(joined2, :match_month)
9×8 DataFrame
 Row │ month  mean_rqr     std_rqr   skewness    kurtosis    SW_W      n_obs  margin_of_error 
     │ Int64  Float64      Float64   Float64     Float64     Float64   Int64  Float64         
─────┼────────────────────────────────────────────────────────────────────────────────────────
   1 │     2   0.0485733   1.02181    0.0765134   0.360681   0.995385    316         0.114962
   2 │     3   0.00782515  0.996202   0.0745814  -0.0735586  0.997611    300         0.115032
   3 │     4   0.0444849   0.991339   0.0642992  -0.169075   0.99185     280         0.118488
   4 │     5   0.0313715   1.00274   -0.0410125   0.216518   0.995708    272         0.1216
   5 │     6   0.021762    0.996496   0.0675379  -0.139615   0.996043    258         0.124078
   6 │     7   0.0694964   0.980874   0.08789    -0.115014   0.995427    292         0.114803
   7 │     8   0.0287518   1.06037    0.0778772   0.572325   0.992672    254         0.133067
   8 │     9  -0.0249709   1.02088    0.154914    0.508754   0.993506    254         0.128111
   9 │    10   0.1243      0.875062   0.409698    0.187924   0.979137     70         0.20918

julia> 





julia> monthly_base = aggregate_monthly_rqr(joined, :match_month)
9×8 DataFrame
 Row │ month  mean_rqr      std_rqr   skewness     kurtosis     SW_W      n_obs  margin_of_error 
     │ Int64  Float64       Float64   Float64      Float64      Float64   Int64  Float64         
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │     2   0.065449     1.00732   -0.0273338    0.539295    0.994898    316         0.113332
   2 │     3   0.0319175    0.986205   0.00469404  -0.00909444  0.997763    300         0.113877
   3 │     4   0.0916436    0.976919  -0.0856584    0.0307577   0.99288     280         0.116764
   4 │     5   0.0843406    1.01052    0.160254    -0.224257    0.993309    272         0.122544
   5 │     6  -0.0409086    1.04577    0.00480127  -0.277852    0.995347    258         0.130213
   6 │     7   0.000807059  0.986065  -0.0681621    0.0777423   0.99779     292         0.11541
   7 │     8  -0.00533456   1.03384    0.22664      0.218433    0.99351     254         0.129738
   8 │     9  -0.0299293    1.02334    0.280571     0.389252    0.991574    254         0.12842
   9 │    10   0.00662938   0.980222  -0.0683008    1.05222     0.976743     70         0.234318

julia> monthly_delta = aggregate_monthly_rqr(joined2, :match_month)
9×8 DataFrame
 Row │ month  mean_rqr     std_rqr   skewness   kurtosis    SW_W      n_obs  margin_of_error 
     │ Int64  Float64      Float64   Float64    Float64     Float64   Int64  Float64         
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │     2   0.0617421   0.945738  0.334521    0.523597   0.99083     316         0.106404
   2 │     3  -0.0230498   1.02666   0.0359481  -0.207517   0.997125    300         0.118548
   3 │     4   0.0352163   0.988078  0.145383   -0.390444   0.991721    280         0.118098
   4 │     5   0.0817035   0.982467  0.288654   -0.55242    0.983897    272         0.119142
   5 │     6  -0.0269309   1.03921   0.0364716  -0.236053   0.996542    258         0.129396
   6 │     7   0.113276    0.951057  0.154643   -0.0408376  0.995924    292         0.111313
   7 │     8   0.0517056   1.05743   0.265202    0.144444   0.991989    254         0.132698
   8 │     9   0.00361961  1.02202   0.199358    0.428654   0.992816    254         0.128255
   9 │    10   0.0112247   0.980395  0.102002    0.22816    0.990393     70         0.234359




# - ----------------
# smaller data set - one season 
# --------------------
julia> monthly_base = aggregate_monthly_rqr(joined, :match_month)
9×8 DataFrame
 Row │ month  mean_rqr     std_rqr   skewness    kurtosis    SW_W      n_obs  margin_of_error 
     │ Int64  Float64      Float64   Float64     Float64     Float64   Int64  Float64         
─────┼────────────────────────────────────────────────────────────────────────────────────────
   1 │     2  -0.144403    0.962166   0.093753   -0.168256   0.990866     80         0.215147
   2 │     3  -0.237015    1.04712    0.105067    0.0129146  0.984149     72         0.246809
   3 │     4   0.00175356  0.981145  -0.426287    0.129899   0.979789     66         0.241541
   4 │     5  -0.0597866   1.12401    0.168765   -0.396764   0.991664     82         0.248252
   5 │     6   0.171069    0.987832   0.0033913  -0.987875   0.969649     66         0.243188
   6 │     7   0.0362431   0.959903  -0.216186   -0.24744    0.990586     86         0.207018
   7 │     8  -0.0138571   1.02789    0.252808   -0.437944   0.987818     90         0.216699
   8 │     9   0.129951    1.07413    0.615578   -0.108818   0.961213     80         0.240182
   9 │    10   0.153922    0.977723   1.51564     2.62086    0.861682     20         0.437251

julia> monthly_delta = aggregate_monthly_rqr(joined2, :match_month)
9×8 DataFrame
 Row │ month  mean_rqr     std_rqr   skewness    kurtosis   SW_W      n_obs  margin_of_error 
     │ Int64  Float64      Float64   Float64     Float64    Float64   Int64  Float64         
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │     2  -0.200053    0.986919  -0.0650203  -0.391171  0.987554     80         0.220682
   2 │     3  -0.234227    1.05976    0.140392    0.103143  0.991404     72         0.249788
   3 │     4   0.15504     0.904082  -0.125281   -0.129614  0.983007     66         0.22257
   4 │     5   0.0397588   1.08199    0.434975   -0.254283  0.979442     82         0.238971
   5 │     6   0.0698766   1.00836   -0.200213   -0.614539  0.981303     66         0.248241
   6 │     7  -0.0063955   0.958959  -0.258687   -0.219267  0.98682      86         0.206814
   7 │     8  -0.0149811   1.02727    0.0193059  -0.001282  0.99075      90         0.216568
   8 │     9  -0.00851818  1.06452    0.46453    -0.317263  0.97081      80         0.238034
   9 │    10   0.0400582   1.22652   -0.0571977   0.876876  0.948709     20         0.548516


=#

# 4. Generate and Save Comparison Plot
comparison_plot = plot_model_comparison(monthly_base, monthly_delta);
Plots.savefig(comparison_plot, "figures_all/rqr_model_comparison.html")


function plot_std_comparison(monthly_base, monthly_delta)
    p = plot(
        title = "RQR Dispersion over Season: Baseline vs Delta Model",
        xlabel = "Month of Season",
        ylabel = "StdDev of RQR (Target: 1.0)",
        size = (1600, 800),
        legend = :bottomright # Placed at bottom so it doesn't overlap the lines around 1.0
    );
    
    # Add Baseline
    plot!(p, monthly_base.month, monthly_base.std_rqr, 
        color = :red, linewidth = 2, marker = :circle, 
        label = "Baseline (Model A)");
        
    # Add Delta Model
    plot!(p, monthly_delta.month, monthly_delta.std_rqr, 
        color = :blue, linewidth = 3, marker = :diamond, 
        label = "Delta w/ Month Params (Model B)");

    # Ideal target line for Standard Deviation
    hline!(p, [1.0], line=(:black, 2, :dash), label="Ideal StdDev = 1.0");
    
    return p
end

# Generate and display the plot
std_plot = plot_std_comparison(monthly_base, monthly_delta);
Plots.savefig(std_plot, "figures_all/rqr_std_comparison.html")


using Distributions, Statistics, DataFrames, Printf

# 1. Function to compute CRPS for a single Negative Binomial prediction
function compute_crps(y::Int, λ::Float64, r_disp::Float64; max_goals=30)
    # Convert mean (λ) and dispersion (r) to NegBinomial (r, p)
    p = r_disp / (r_disp + λ)
    dist = NegativeBinomial(r_disp, p)
    
    crps_value = 0.0
    # Sum over possible goal counts
    for x in 0:max_goals
        F_x = cdf(dist, x)           # Model's cumulative probability up to x goals
        indicator = x >= y ? 1.0 : 0.0 # 1.0 if the actual score was less than or equal to x
        
        crps_value += (F_x - indicator)^2
    end
    
    return crps_value
end

# 2. Append CRPS to the DataFrames
function append_crps!(df)
    # Calculate CRPS for Home and Away
    df.crps_home = compute_crps.(df.home_score, df.exp_home, df.exp_r)
    df.crps_away = compute_crps.(df.away_score, df.exp_away, df.exp_r)
    
    # Total match CRPS (Average of Home and Away)
    df.crps_match = (df.crps_home .+ df.crps_away) ./ 2.0
    return df
end

# 3. Print the Comparison
function compare_crps(df_base, df_delta)
    # Ensure metrics are calculated
    append_crps!(df_base)
    append_crps!(df_delta)
    
    base_home = mean(df_base.crps_home)
    base_away = mean(df_base.crps_away)
    base_total = mean(df_base.crps_match)
    
    delta_home = mean(df_delta.crps_home)
    delta_away = mean(df_delta.crps_away)
    delta_total = mean(df_delta.crps_match)
    
    println("==================================================")
    println(" CRPS MODEL COMPARISON (Lower is Better)")
    println("==================================================")
    @printf("%-15s | %-12s | %-12s | %-12s\n", "Model", "Home CRPS", "Away CRPS", "Match CRPS")
    println("-"^50)
    @printf("%-15s | %12.5f | %12.5f | %12.5f\n", "A (Baseline)", base_home, base_away, base_total)
    @printf("%-15s | %12.5f | %12.5f | %12.5f\n", "B (Delta)", delta_home, delta_away, delta_total)
    println("-"^50)
    
    # Calculate percentage improvement
    diff = ((delta_total - base_total) / base_total) * 100
    if diff < 0
        @printf("Conclusion: Model B improves accuracy by %.3f%%\n", abs(diff))
    else
        @printf("Conclusion: Model A is more accurate by %.3f%%\n", diff)
    end
    println("==================================================")
end

# --- Execution ---
compare_crps(joined, joined2)


#=

julia> # --- Execution ---
       compare_crps(joined, joined2)
==================================================
 CRPS MODEL COMPARISON (Lower is Better)
==================================================
Model           | Home CRPS    | Away CRPS    | Match CRPS  
--------------------------------------------------
A (Baseline)    |      0.66616 |      0.63585 |      0.65101
B (Delta)       |      0.66777 |      0.63622 |      0.65200
--------------------------------------------------
Conclusion: Model A is more accurate by 0.152%
==================================================


# --------------------
# small data set one season 
# --------------------
==================================================
 CRPS MODEL COMPARISON (Lower is Better)
==================================================
Model           | Home CRPS    | Away CRPS    | Match CRPS  
--------------------------------------------------
A (Baseline)    |      0.66848 |      0.63383 |      0.65115
B (Delta)       |      0.66261 |      0.62962 |      0.64611
--------------------------------------------------
Conclusion: Model B improves accuracy by 0.774%
==================================================


=#


using DataFrames, Statistics, GLM, Plots, StatsPlots, Printf

# 1. Prepare Market Data & Calculate Spreads
"""
    prepare_betting_df(market_data_df, latents)

Takes raw latents, runs model inference, merges with market odds, and calculates the edge/spread.
"""
function prepare_betting_df(market_data_df, latents)
    # Generate predictive distributions from latents
    ppd = BayesianFootball.Predictions.model_inference(latents)
    
    # Extract the mean probability for the specific bet
    model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
    select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)

    # Merge with market data
    analysis_df = innerjoin(
        market_data_df,
        model_features,
        on = [:match_id, :market_name, :market_line, :selection]
    )
    
    # Clean up missing odds and calculate targets
    dropmissing!(analysis_df, [:odds_close, :is_winner])
    
    analysis_df.spread = analysis_df.prob_model .- analysis_df.prob_implied_close
    analysis_df.spread_fair = analysis_df.prob_model .- analysis_df.prob_fair_close
    analysis_df.Y = Float64.(analysis_df.is_winner)
    
    return analysis_df
end

# 2. Run GLM Regression
"""
    evaluate_betting_edge(df, label="Model"; target_markets=nothing, target_selections=nothing)

Runs a logistic regression to see if the predicted spread translates to actual wins.
Optionally filters by specific markets (e.g., "OverUnder") and selections (e.g., :over_25).
"""
function evaluate_betting_edge(df, label="Model"; target_markets=nothing, target_selections=nothing)
    # Filter for portfolio subsets if requested
    if target_markets !== nothing && target_selections !== nothing
        sub_df = filter(row -> row.market_name in target_markets && row.selection in target_selections, df)
        strat_name = "Filtered Strategy"
    else
        sub_df = df
        strat_name = "All Markets"
    end

    println("="^60)
    println(" GLM EDGE ANALYSIS: $label | $strat_name")
    println("="^60)
    
    # Run the logistic regression using the fair spread
    reg_model = glm(@formula(Y ~ prob_fair_close + spread_fair), sub_df, Binomial(), LogitLink())
    
    println(coeftable(reg_model))
    println("  Observations: ", nrow(sub_df))
    println("="^60, "\n")
    
    return reg_model, sub_df
end

# 3. Plot Realized Alpha
"""
    plot_realized_alpha(df_base, df_delta)

Creates a bubble chart grouping bets by their predicted edge, comparing realized alpha.
"""
function plot_realized_alpha(df_base, df_delta)
    # Helper to bucket edges and calculate win rates
    function get_grouped(df)
        df_copy = copy(df)
        df_copy.edge_bucket = round.(df_copy.spread_fair, digits=2)
        
        grouped = combine(groupby(df_copy, :edge_bucket), 
            :Y => mean => :actual_win_rate,
            :prob_fair_close => mean => :market_implied,
            nrow => :count
        )
        
        # Filter out tiny sample sizes to remove noise
        filter!(r -> r.count > 10, grouped)
        grouped.excess_return = grouped.actual_win_rate .- grouped.market_implied
        return grouped
    end

    grp_base = get_grouped(df_base)
    grp_delta = get_grouped(df_delta)

    p = scatter(
        title="Realized Alpha vs Predicted Edge",
        xlabel="Predicted Edge vs Fair Line (spread_fair)",
        ylabel="Actual Excess Return (Realized - Implied)",
        legend=:topleft,
        size=(1600, 800)
    );

    # Plot Baseline Bubbles
    scatter!(p, grp_base.edge_bucket, grp_base.excess_return,
        markersize = sqrt.(grp_base.count) ./ 1.5, # Bubble size based on count
        color = :red, markeralpha=0.6, label="Baseline (Model A)");

    # Plot Delta Bubbles
    scatter!(p, grp_delta.edge_bucket, grp_delta.excess_return,
        markersize = sqrt.(grp_delta.count) ./ 1.5, 
        color = :blue, markeralpha=0.6, label="Delta (Model B)");

    # Zero Alpha Line (Break-even against implied probability)
    hline!(p, [0.0], line=(:black, 2, :dash), label="Zero Excess Return");
    
    # Ideal Alpha Line (y = x)
    # If your model predicts a 5% edge, you should see a 5% excess return
    min_x = min(minimum(grp_base.edge_bucket), minimum(grp_delta.edge_bucket))
    max_x = max(maximum(grp_base.edge_bucket), maximum(grp_delta.edge_bucket))
    plot!(p, [min_x, max_x], [min_x, max_x], line=(:green, 2, :dot), label="Ideal Alpha (y=x)");

    return p
end



# 1. Prepare DataFrames for both models
# (I am assuming market_data is the loaded struct, so passing market_data.df)
df_model_a = prepare_betting_df(market_data.df, latents_raw)
df_model_b = prepare_betting_df(market_data.df, latents_raw2)

# 2. Run Global Regression (All Markets)
reg_a_all, _ = evaluate_betting_edge(df_model_a, "Model A (Baseline)");
reg_b_all, _ = evaluate_betting_edge(df_model_b, "Model B (Delta)");
reg_a_all
reg_b_all

#=

julia> reg_a_all
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

Y ~ 1 + prob_fair_close + spread_fair

Coefficients:
─────────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)      -2.99391   0.0745515  -40.16    <1e-99   -3.14003   -2.8478
prob_fair_close   6.10604   0.144175    42.35    <1e-99    5.82347    6.38862
spread_fair       2.95363   0.563593     5.24    <1e-06    1.84901    4.05826
─────────────────────────────────────────────────────────────────────────────


julia> reg_b_all
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

Y ~ 1 + prob_fair_close + spread_fair

Coefficients:
─────────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)      -2.96776   0.0749889  -39.58    <1e-99   -3.11474   -2.82079
prob_fair_close   6.05159   0.145337    41.64    <1e-99    5.76673    6.33644
spread_fair       4.81464   0.627316     7.67    <1e-13    3.58512    6.04415
=#

# 3. Run Strategy-Specific Regressions (e.g., Just Overs)
target_markets_overs = ["OverUnder"]
target_lines_overs = [:over_15, :over_25, :over_35]

reg_a_overs, sub_a_overs = evaluate_betting_edge(df_model_a, "Model A", 
    target_markets=target_markets_overs, target_selections=target_lines_overs)
    
reg_b_overs, sub_b_overs = evaluate_betting_edge(df_model_b, "Model B", 
    target_markets=target_markets_overs, target_selections=target_lines_overs)

reg_a_overs
reg_b_overs

#=

julia> reg_a_overs
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

Y ~ 1 + prob_fair_close + spread_fair

Coefficients:
──────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────────
(Intercept)      -2.31104     0.215355  -10.73    <1e-26   -2.73313   -1.88895
prob_fair_close   4.71723     0.405523   11.63    <1e-30    3.92242    5.51204
spread_fair       0.534872    1.16238     0.46    0.6454   -1.74334    2.81309
──────────────────────────────────────────────────────────────────────────────

julia> reg_b_overs
StatsModels.TableRegressionModel{GeneralizedLinearModel{GLM.GlmResp{Vector{Float64}, Binomial{Float64}, LogitLink}, GLM.DensePredChol{Float64, CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

Y ~ 1 + prob_fair_close + spread_fair

Coefficients:
─────────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)      -2.32365    0.215702  -10.77    <1e-26  -2.74642    -1.90088
prob_fair_close   4.74183    0.406574   11.66    <1e-30   3.94496     5.5387
spread_fair       3.21652    1.38738     2.32    0.0204   0.497299    5.93574
─────────────────────────────────────────────────────────────────────────────

=#
plotlyjs()

# 4. Generate the Alpha Plot
alpha_plot = plot_realized_alpha(df_model_a, df_model_b);
Plots.savefig(alpha_plot, "figures/realized_alpha_comparison.html")
display(alpha_plot)
