
exp_res = loaded_results[1]
exp_res2 = loaded_results[2]



latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
latents_raw2 = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res2)



joined = innerjoin(
    select(latents_raw.df, :match_id, :λ_h, :λ_a, :r),
    select(ds.matches, :match_id,:match_month, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
    on = :match_id
)



joined2 = innerjoin(
    select(latents_raw2.df, :match_id, :λ_h, :λ_a, :r),
    select(ds.matches, :match_id,:match_month, :match_date, :home_score, :away_score, :tournament_id, :season, :home_team, :away_team),
    on = :match_id
)

using HypothesisTests

function test_one(joined)

    joined.exp_home = [ mean(λ) for λ in joined.λ_h]
    joined.exp_away = [ mean(λ) for λ in joined.λ_a]
    joined.exp_all = joined.exp_home .+ joined.exp_away


    joined.res_home = joined.home_score .- joined.exp_home
    joined.res_away = joined.away_score .- joined.exp_away
    joined.res_all = ( joined.home_score .+ joined.away_score ) .- joined.exp_all
  println(sum(joined.res_home), " res home") # -60.415
    sum(joined.res_away) # -38.07
    sum(joined.res_all)  # -98.488

  println(describe(joined.res_home))
  println(describe(joined.res_away))
  println(describe(joined.res_all))


    #
    joined.cum_res_home = cumsum(joined.home_score) .- cumsum(joined.exp_home)
    joined.cum_res_away = cumsum(joined.away_score) .- cumsum(joined.exp_away)
    joined.cum_res_all = cumsum(joined.away_score .+ joined.home_score) .- cumsum(joined.exp_all)

    test =  OneSampleTTest(joined.res_all)

    println(test)

end



test_one(joined)


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

# 2. Append RQR to DataFrame
function append_rqr_metrics!(df)
    # Ensure expected values are present
    if !("exp_home" in names(df))
        df.exp_home = mean.(df.λ_h)
        df.exp_away = mean.(df.λ_a)
    end
    
    df.exp_r = mean.(df.r)
    
    df.rqr_home = compute_rqr.(df.home_score, df.exp_home, df.exp_r)
    df.rqr_away = compute_rqr.(df.away_score, df.exp_away, df.exp_r)
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
Plots.savefig(comparison_plot, "figures/rqr_model_comparison.html")


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
Plots.savefig(std_plot, "figures/rqr_std_comparison.html")


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
