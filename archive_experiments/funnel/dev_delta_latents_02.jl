using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

using Plots
plotlyjs()  # Switch the backend to PlotlyJS

# Load DataStore again (Data is lightweight, models are heavy)
#
ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


# 1. Load Experiments from Disk
# =============================
# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/funnel_basics"; data_dir="./data")
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

exp_res = loaded_results[1]

market_data = Data.prepare_market_data(ds)
#=
julia> market_data.df
73361×20 DataFrame
   Row │ match_id  market_name  market_line  selection  odds_open  odds_close  is_winner  prob_implied_open  prob_implied_close  overround_open  overround ⋯
       │ Int64     String       Float64      Symbol     Float64    Float64     Bool?      Float64            Float64             Float64         Float64   ⋯
───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     1 │ 14035482  1X2                  0.0  home         2.5         2.38         false          0.4                 0.420168          1.09412          1 ⋯
     2 │ 14035482  1X2                  0.0  draw         3.4         3.3          false          0.294118            0.30303           1.09412          1
     3 │ 14035482  1X2                  0.0  away         2.5         2.7           true          0.4                 0.37037           1.09412          1
     4 │ 14035484  1X2                  0.0  home         1.67        1.57         false          0.598802            0.636943          1.09273          1


julia> names(market_data.df)
20-element Vector{String}:
 "match_id"
 "market_name"
 "market_line"
 "selection"
 "odds_open"
 "odds_close"
 "is_winner"
 "prob_implied_open"
 "prob_implied_close"
 "overround_open"
 "overround_close"
 "prob_fair_open"
 "prob_fair_close"
 "fair_odds_open"
 "fair_odds_close"
 "vig_open"
 "vig_close"
 "clm_prob"
 "clm_odds"
 "date"
=#

latents = BayesianFootball.BackTesting.extract_oos_predictions(ds, exp_res)
#=
julia> latents.df 
708×4 DataFrame
 Row │ match_id  r                                  λ_a                                λ_h                               
     │ Any       Any                                Any                                Any                               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 11395624  [23.6678, 3.60027, 50.7031, 4.15…  [1.26134, 1.96007, 1.10567, 1.00…  [2.41723, 1.90512, 0.830434, 2.4…
   2 │ 11395610  [23.6678, 3.60027, 50.7031, 4.15…  [1.43671, 1.51957, 1.56802, 2.00…  [1.3092, 1.60782, 1.07593, 2.471…
   3 │ 11395618  [23.6678, 3.60027, 50.7031, 4.15…  [1.83271, 1.63113, 1.2208, 1.360…  [1.12939, 1.43397, 1.3722, 0.912…
   4 │ 11395621  [23.6678, 3.60027, 50.7031, 4.15…  [0.789443, 1.15163, 1.06453, 1.1…  [1.61686, 1.74078, 2.34622, 1.31

julia> names(latents.df)
4-element Vector{String}:
 "match_id"
 "r"
 "λ_a"
 "λ_h"
=#

ppd = BayesianFootball.BackTesting.model_inference(latents)
#=
julia> ppd.df
19116×5 DataFrame
   Row │ match_id  market_name  market_line  selection  distribution                      
       │ Int64     String       Float64      Symbol     Array…                            
───────┼──────────────────────────────────────────────────────────────────────────────────
     1 │ 11395624  1X2                  0.0  away       [0.194172, 0.412206, 0.416887, 0…
     2 │ 11395624  1X2                  0.0  home       [0.621307, 0.395342, 0.272958, 0…
     3 │ 11395624  1X2                  0.0  draw       [0.184482, 0.191857, 0.310156, 0…
     4 │ 11395624  BTTS                 0.0  btts_yes   [0.636544, 0.619006, 0.373228, 0…
     5 │ 11395624  BTTS                 0.0  btts_no    [0.363417, 0.380399, 0.626772, 0
=#


over_15_ppd = subset( ppd.df, :selection => ByRow(isequal(:over_15)))
market_data_over_15 = subset( market_data.df, :selection => ByRow(isequal(:over_15)))



using Roots, Statistics, DataFrames

function solve_negbin_lambda_from_under15(prob_under, r_val)
    # Objective: Find Lambda such that NegBinCDF(1 | r, lambda) = prob_under
    
    function objective(lam)
        if lam <= 0 return 1.0 end # Penalty for invalid lambda
        
        # Calculate p parameter for NegBin
        # p = r / (r + lambda)
        p = r_val / (r_val + lam)
        
        # P(Y=0) = p^r
        p0 = p^r_val
        
        # P(Y=1) = r * (1-p) * p^r
        p1 = r_val * (1.0 - p) * p0
        
        # CDF(1) = P(0) + P(1)
        cdf_1 = p0 + p1
        
        return cdf_1 - prob_under
    end
    
    try
        # Search for Lambda between 0.01 and 10.0
        return find_zero(objective, (0.001, 10.0))
    catch
        return NaN
    end
end



function compute_negbin_delta(latents_df, market_df)
    # 1. Summarize Latents (Vectors -> Means)
    # We need scalar values to solve the inverse problem efficiently
    
    # Create a clean summary dataframe
    model_summary = select(latents_df, :match_id, 
        # Calculate Total Lambda (Home + Away) per sample, then take mean
        [:λ_h, :λ_a] => ((h, a) -> mean.(h .+ a)) => :model_lambda_mean,
        
        # Calculate Mean Dispersion (r)
        :r => (r -> mean.(r)) => :model_r_mean
    )
    
    # 2. Join with Market Data
    joined = innerjoin(model_summary, market_df, on = :match_id, makeunique = true)
    
    joined.lambda_mkt = fill(NaN, nrow(joined))
    joined.lambda_delta = fill(NaN, nrow(joined))
    
    # 3. Solve Row by Row
    for i in 1:nrow(joined)
        # Market Probability of Under 1.5
        fair_odds_over = joined.fair_odds_close[i]
        prob_over = 1.0 / fair_odds_over
        prob_under = 1.0 - prob_over
        
        # Model Dispersion (r)
        r_val = joined.model_r_mean[i]
        
        # Solve for Market Lambda (using Model's r)
        lam_mkt = solve_negbin_lambda_from_under15(prob_under, r_val)
        joined.lambda_mkt[i] = lam_mkt
        
        # Calculate Delta
        lam_model = joined.model_lambda_mean[i]
        
        if !isnan(lam_mkt) && lam_mkt > 0
            joined.lambda_delta[i] = log(lam_model) - log(lam_mkt)
        end
    end
    
    return joined
end


final_signal_df = compute_negbin_delta(latents.df, market_data_over_15)

valid_rows = filter(row -> !isnan(row.lambda_delta), final_signal_df)
cor(valid_rows.lambda_delta, valid_rows.is_winner)

#=
 this all the 
julia> cor(valid_rows.lambda_delta, valid_rows.is_winner)
0.003935007438210145
=#

# here we want to apply the Signals and find the correlation between when our model thinks theres 
# is an edge - namely when stake > 0. 


baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    [exp_res], 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

over_15_ledger = subset(ledger.df, :selection => ByRow(isequal(:over_15)))

ids_model_yes = subset(over_15_ledger, :stake => ByRow(>(1e-6))).match_id
ids_model_no = subset(over_15_ledger, :stake => ByRow(<(1e-6))).match_id

model_yes = subset(final_signal_df, :match_id => ByRow(in(ids_model_yes)))
model_no = subset(final_signal_df, :match_id => ByRow(in(ids_model_no)))
cor(model_yes.lambda_delta, model_yes.is_winner) 
cor(model_no.lambda_delta, model_no.is_winner)

describe(final_signal_df.lambda_delta)
describe(model_yes.lambda_delta)
describe(model_no.lambda_delta)

#= 
 this all the matches. 
julia> cor(valid_rows.lambda_delta, valid_rows.is_winner)
0.003935007438210145

julia> cor(model_yes.lambda_delta, model_yes.is_winner) 
0.183968750211225

julia> cor(model_no.lambda_delta, model_no.is_winner)
0.011890624166742816

julia> describe(final_signal_df.lambda_delta)
Summary Stats:
Length:         684
Missing Count:  0
Mean:           -0.022324
Std. Deviation: 0.109522
Minimum:        -0.492550
1st Quartile:   -0.097096
Median:         -0.033112
3rd Quartile:   0.041573
Maximum:        0.418314
Type:           Float64

julia> describe(model_yes.lambda_delta)
Summary Stats:
Length:         51
Missing Count:  0
Mean:           0.196083
Std. Deviation: 0.070244
Minimum:        0.105940
1st Quartile:   0.146430
Median:         0.181822
3rd Quartile:   0.227501
Maximum:        0.418314
Type:           Float64

julia> describe(model_no.lambda_delta)
Summary Stats:
Length:         633
Missing Count:  0
Mean:           -0.039921
Std. Deviation: 0.091724
Minimum:        -0.492550
1st Quartile:   -0.100991
Median:         -0.040209
3rd Quartile:   0.021883
Maximum:        0.217730
Type:           Float64

=#







# z score 
using Statistics, DataFrames, Plots

function compute_z_scores(latents_df, market_df)
    # Join Model Latents with Market Data
    # We need the full vector samples here, not just means
    joined = innerjoin(
        select(latents_df, :match_id, :λ_h, :λ_a), 
        select(market_df, :match_id, :fair_odds_close), 
        on = :match_id
    )

    joined.z_score = fill(NaN, nrow(joined))
    joined.raw_delta = fill(NaN, nrow(joined))
    joined.model_std = fill(NaN, nrow(joined))

    for i in 1:nrow(joined)
        # 1. Market Implied Lambda (Point Estimate)
        # -----------------------------------------
        prob_under = 1.0 - (1.0 / joined.fair_odds_close[i])
        
        # We need a 'rough' r to invert the market. 
        # Using the model's mean r or a fixed r=1.0 (Poisson) is standard for finding market lambda.
        # Let's use Poisson (r -> Inf) assumption for Market to keep it pure, 
        # OR use the code you already wrote to solve for lambda_mkt.
        # Assuming you have 'lambda_mkt' from your previous step:
        # (I will re-calculate a simple Poisson lambda_mkt here for the example)
        # Poisson CDF(1) = (1+lambda)*exp(-lambda) = prob_under. 
        # Approx inversion or use your existing solver.
        
        # *Re-using your previous solver logic for consistency*
        # (Assuming you added 'lambda_mkt' to the dataframe in the previous step,
        #  if not, we calculate a quick proxy here: -log(prob_0_and_1) approx)
        
        # Let's assume you pass the dataframe 'final_signal_df' which already has 'lambda_mkt'
        # But since we need vectors, let's re-calculate quickly:
        lambda_mkt = -log(prob_under) # Crude Poisson approx for illustration if solver not available
        # Better: Use the lambda_mkt you calculated in the previous snippet.
    end
    
    # Let's use the 'final_signal_df' you created before, which has 'lambda_mkt', 
    # and join it with 'latents' to get the vectors back.
    
    return joined
end

# --- BETTER WORKFLOW ---

# 1. Take your existing 'final_signal_df' (which has lambda_mkt)
# 2. Join it with 'latents.df' to get the distributions back
z_analysis = innerjoin(
    select(final_signal_df, :match_id, :lambda_mkt, :is_winner),
    select(latents.df, :match_id, :λ_h, :λ_a),
    on = :match_id
)

# 3. Compute Z-Scores
z_scores = []
model_stds = []

for row in eachrow(z_analysis)
    # Total Lambda Distribution (Vector)
    λ_total_dist = row.λ_h .+ row.λ_a
    
    # Log-Transform (to normalize skew)
    log_λ_dist = log.(λ_total_dist)
    
    # Model Moments
    mu_model = mean(log_λ_dist)
    sigma_model = std(log_λ_dist)
    
    # Market Value (in Log Space)
    log_mkt = log(row.lambda_mkt)
    
    # Z-Score: (Model - Market) / Uncertainty
    # Positive Z = Model Higher = Over Value
    z = (mu_model - log_mkt) / sigma_model
    
    push!(z_scores, z)
    push!(model_stds, sigma_model)
end

z_analysis.z_score = z_scores
z_analysis.model_std = model_stds

# --- DIAGNOSTICS ---

# 1. Distribution Check
histogram(z_analysis.z_score, normalize=true, label="Model Z-Scores", alpha=0.6, bins=30)
plot!(x -> pdf(Normal(0,1), x), -4, 4, label="Standard Normal", linewidth=3)
title!("Meta-Calibration: Model vs Market")
xlabel!("Z-Score (Standard deviations from Market)")

# 2. Correlation of Z vs Winner
# High Z should correlate with winning 'Over' bets
cor(z_analysis.z_score, z_analysis.is_winner)

#-- 
using Plots, Distributions, Statistics, DataFrames, StatsPlots, Plots
plotlyjs()  # Switch backend to PlotlyJS

# 1. Prepare Data: Join Latents (Distributions) with Market Signal (Point Estimate)
# ==============================================================================
# We need 'final_signal_df' (contains lambda_mkt) and 'latents.df' (contains vectors)
z_df = innerjoin(
    select(final_signal_df, :match_id, :lambda_mkt, :is_winner),
    select(latents.df, :match_id, :λ_h, :λ_a),
    on = :match_id
)

# 2. Calculate Z-Scores
# ==============================================================================
z_scores = Float64[]

for row in eachrow(z_df)
    # Get the model's total goals distribution (Home + Away)
    # We use samples to get the true moments of the Log-Space
    total_goals_dist = row.λ_h .+ row.λ_a
    
    # Transform to Log-Space (since Lambda is strictly positive and skewed)
    log_dist = log.(total_goals_dist)
    
    # Calculate Model Moments
    mu_model = mean(log_dist)
    sigma_model = std(log_dist)
    
    # Market Value in Log-Space
    log_mkt = log(row.lambda_mkt)
    
    # Calculate Z-Score
    # (Model Mean - Market Mean) / Model Uncertainty
    z = (mu_model - log_mkt) / sigma_model
    
    push!(z_scores, z)
end

z_df.z_score = z_scores

# 3. Generate Histogram Plot
# ==============================================================================
# Create the histogram normalized to probability density (normalize=true)
#histogram
p = StatsPlots.histogram(z_df.z_score, 
    normalize = :pdf,          # Normalize area to 1 so it matches the curve
    bins = 40,                 # Granularity
    label = "Model Z-Scores", 
    alpha = 0.6,               # Transparency
    color = :blue,
    title = "Meta-Calibration: Model Residuals vs Market",
    xlabel = "Z-Score (Std Devs from Market Price)",
    ylabel = "Density",
    legend = :topright
);

# Overlay Standard Normal Distribution (N(0,1))
# This represents "Perfect Calibration"
x_grid = range(minimum(z_df.z_score), maximum(z_df.z_score), length=200);
y_normal = pdf.(Normal(0, 1), x_grid);

plot!(p, x_grid, y_normal, 
    label = "Theoretical N(0,1)", 
    color = :red, 
    linewidth = 3,
    linestyle = :dash
);

# Add a vertical line for the mean of your Z-scores (Bias check)
mean_z = mean(z_df.z_score)
StatsPlots.vline!(p, [mean_z], 
    label = "Your Mean Z ($(round(mean_z, digits=2)))", 
    color = :black, 
    linewidth = 2
);

# 4. Save to HTML
# ==============================================================================
output_filename = "z_score_hist.html"
Plots.savefig(p, output_filename)

println("Plot saved to $output_filename")
println("View at: http://localhost:8080/$output_filename (via your SSH tunnel)")

using Plots, Distributions, Statistics, HypothesisTests, StatsBase
plotlyjs()

# 1. Prepare Data
# ==============================================================================
# Filter out any NaNs just in case
clean_z = filter(!isnan, z_df.z_score)

# 2. Formal Statistical Tests
# ==============================================================================
# A. Shapiro-Wilk Test
# Null Hypothesis: Data is drawn from a Normal distribution
sw_test = ShapiroWilkTest(clean_z)
println("--- Normality Test Results ---")
println(sw_test)
#= 
Shapiro-Wilk normality test
---------------------------
Population details:
    parameter of interest:   Squared correlation of data and expected order statistics of N(0,1) (W)
    value under h_0:         1.0
    point estimate:          0.990194

Test summary:
    outcome with 95% confidence: reject h_0
    one-sided p-value:           0.0002

Details:
    number of observations: 684
    censored ratio:         0.0
    W-statistic:            0.990194


=#

# B. Moment Analysis
# Skewness (0 = Symmetric)
# Kurtosis (0 = Normal, >0 = Leptokurtic/Peaked, <0 = Platykurtic/Flat)
# Note: StatsBase uses "excess kurtosis" (Normal = 0.0)
sk = skewness(clean_z)
ku = kurtosis(clean_z) 

println("\n--- Moment Analysis ---")
println("Mean (Bias):      ", round(mean(clean_z), digits=4))
println("Std Dev:          ", round(std(clean_z), digits=4))
println("Skewness:         ", round(sk, digits=4), " (Target: 0.0)")
println("Excess Kurtosis:  ", round(ku, digits=4), " (Target: 0.0)")

#=
Mean (Bias):      -0.2073
Std Dev:          0.519
Skewness:         0.2926 (Target: 0.0)
Excess Kurtosis:  0.1573 (Target: 0.0)
=#

# 3. Generate Q-Q Plot
# ==============================================================================
# We create this manually to ensure full control over the reference line
qq_p = plot(
    title = "Q-Q Plot: Model Z-Scores vs Standard Normal",
    xlabel = "Theoretical Quantiles (Normal)",
    ylabel = "Sample Quantiles (Model Z)",
    legend = :topleft
);

# A. The Reference Line (y=x)
# If perfectly normal, points fall exactly on this line
ref_x = range(-4, 4, length=100);
plot!(qq_p, ref_x, ref_x, 
    label = "Perfect Normality", 
    color = :red, 
    linewidth = 2, 
    linestyle = :dash
);

# B. The Data Points
# Sort data to compute quantiles
sorted_z = sort(clean_z);
n = length(sorted_z);
# Compute theoretical quantiles for these positions
probs = (1:n) ./ (n + 1)
theoretical_q = quantile.(Normal(0,1), probs)

scatter!(qq_p, theoretical_q, sorted_z, 
    label = "Model Residuals", 
    color = :blue, 
    alpha = 0.6,
    markersize = 3
);

# 4. Save
Plots.savefig(qq_p, "qq_plot.html")
println("\nQ-Q Plot saved to qq_plot.html")









# 1. Split the Z-Scores into "Action" vs "No Action"
# ==============================================================================
# Ensure we have the IDs (re-running this logic from your previous snippet to be safe)
ids_model_yes = subset(over_15_ledger, :stake => ByRow(>(1e-6))).match_id
ids_model_no = subset(over_15_ledger, :stake => ByRow(<=(1e-6))).match_id

# Filter the Z-dataframe
z_yes = filter(row -> row.match_id in ids_model_yes, z_df)
z_no  = filter(row -> row.match_id in ids_model_no,  z_df)

# 2. Comparative Statistics
# ==============================================================================
function print_regime_stats(name, data_vec)
    clean_data = filter(!isnan, data_vec)
    n = length(clean_data)
    
    if n < 3
        println("--- $name (Insufficient Data: N=$n) ---")
        return
    end

    sw = ShapiroWilkTest(clean_data)
    mu = mean(clean_data)
    sig = std(clean_data)
    sk = skewness(clean_data)
    ku = kurtosis(clean_data) # Excess kurtosis

    println("\n--- $name (N=$n) ---")
    println("Mean (Signal):    ", round(mu, digits=4))
    println("Std Dev (Risk):   ", round(sig, digits=4))
    println("Skewness:         ", round(sk, digits=4))
    println("Ex. Kurtosis:     ", round(ku, digits=4))
    println("Shapiro-Wilk P:   ", round(pvalue(sw), digits=5))
end

print_regime_stats("Global (All Matches)", z_df.z_score)
print_regime_stats("Bets Placed (Model Yes)", z_yes.z_score)
print_regime_stats("No Bet (Model No)", z_no.z_score)

#=

julia> print_regime_stats("Global (All Matches)", z_df.z_score)

--- Global (All Matches) (N=684) ---
Mean (Signal):    -0.2073
Std Dev (Risk):   0.519
Skewness:         0.2926
Ex. Kurtosis:     0.1573
Shapiro-Wilk P:   0.00016

julia> print_regime_stats("Bets Placed (Model Yes)", z_yes.z_score)

--- Bets Placed (Model Yes) (N=51) ---
Mean (Signal):    0.824
Std Dev (Risk):   0.2363
Skewness:         0.5408
Ex. Kurtosis:     -0.7744
Shapiro-Wilk P:   0.00398

julia> print_regime_stats("No Bet (Model No)", z_no.z_score)

--- No Bet (Model No) (N=633) ---
Mean (Signal):    -0.2903
Std Dev (Risk):   0.4404
Skewness:         -0.1131
Ex. Kurtosis:     0.0248
Shapiro-Wilk P:   0.00713

=#

# 3. Visualization: The "Separation" Plot
# ==============================================================================
# We want to see WHERE the strategy is betting relative to the global distribution

p_hist = StatsPlots.histogram(z_no.z_score, 
    label="No Bet (Noise)", 
    normalize=:pdf, 
    color=:gray, 
    alpha=0.5, 
    bins=30,
    title="Strategy Activation: Signal vs Noise",
    xlabel="Z-Score (Std Devs from Market)",
    ylabel="Density"
);

StatsPlots.histogram!(p_hist, z_yes.z_score, 
    label="Bet Placed (Signal)", 
    normalize=:pdf, 
    color=:green, 
    alpha=0.7, 
    bins=15
);

# Add the Theoretical Normal for reference
x_grid = range(-4, 4, length=200);
plot!(p_hist, x_grid, pdf.(Normal(0,1), x_grid), 
    label="Theoretical N(0,1)", color=:red, linestyle=:dash, linewidth=2
);

Plots.savefig(p_hist, "strategy_z_separation.html")

# 4. Q-Q Plot for the "Bets" Only
# ==============================================================================
# Does the *selected* portfolio follow a specific tail distribution?

# 1. Prepare the data vectors first
sorted_z_yes = sort(filter(!isnan, z_yes.z_score))
n = length(sorted_z_yes)
probs = (1:n) ./ (n + 1)
quantiles_theo = quantile.(Normal(0,1), probs)

# 2. Plot in one go (Note the semicolons!)
#    We create the plot and add the scatter layer in a single chain if we want,
#    or just use the `!` notation.

p = plot(title="Q-Q Plot: Active Bets Only", legend=:topleft);
plot!(p, range(-4, 4, length=100), range(-4, 4, length=100), label="Normal Reference", color=:red, linestyle=:dash);
scatter!(p, quantiles_theo, sorted_z_yes, label="Active Bets", color=:green);

# 3. Save directly
Plots.savefig(p, "strategy_qq.html")
println("Saved strategy_qq.html")


### Rescale the λ model for a better z score / test against the market

function calibrate_latents(original_df::DataFrame; bias_shift=0.0, temperature=1.0)
    # Create a deep copy so we don't mess up the original data
    df = deepcopy(original_df)
    
    # We transform the vectors for Home and Away independently
    # Assuming symmetry in error, we apply the same correction to both.
    
    for row in eachrow(df)
        for col in [:λ_h, :λ_a]
            # 1. Get the raw samples
            raw_samples = row[col]
            
            # 2. Move to Log-Space (Lambda is naturally Log-Normal)
            log_samples = log.(raw_samples)
            
            # 3. Calculate Moments
            mu = mean(log_samples)
            
            # 4. Apply Variance Scaling (Temperature)
            # T < 1.0 shrinks variance (makes model more confident/aggressive)
            # T > 1.0 expands variance (makes model more humble)
            centered = log_samples .- mu
            scaled_log = mu .+ (centered .* temperature)
            
            # 5. Apply Bias Shift
            # Positive shift = Boost goal expectation
            shifted_log = scaled_log .+ (bias_shift / 2.0) # Split shift between H/A
            
            # 6. Transform back to Real Space
            row[col] = exp.(shifted_log)
        end
    end
    
    return df
end


# ------------------------------------------------------------------
# The Hacked Backtest Runner
# ------------------------------------------------------------------
function run_hacked_backtest(latents_df, market_df, signal_algo; bias=0.21, temp=0.6)
    
    println("Running Hack: Bias=$bias, Temp=$temp ...")
    
    # 1. Calibrate Latents
    hacked_latents = calibrate_latents(latents_df, bias_shift=bias, temperature=temp)
    
    # 2. Join with Market Data (Over 1.5 Only)
    # Ensure we only get the closing odds for Over 1.5
    mkt_subset = filter(row -> row.market_name == "Over/Under" && row.selection == :over_15, market_df)
    
    # Join on Match ID
    joined = innerjoin(hacked_latents, mkt_subset, on=:match_id, makeunique=true)
    
    ledger_entries = []
    
    for row in eachrow(joined)
        # --- A. Generate Probabilities from Hacked Model ---
        # We assume Total Goals = λ_h + λ_a
        # r is usually fixed per match or vector. If vector, we broadcast.
        
        # Total Lambda Vector
        total_lambda = row.λ_h .+ row.λ_a
        r_vec = row.r # Assuming r is a vector of samples too
        
        # Calculate P(Over 1.5) for every sample in the posterior
        # This creates the distribution 'dist' for the Kelly Algo
        dist_probs = get_over15_prob.(r_vec, total_lambda)
        
        # --- B. Get Market Odds ---
        odds = row.odds_close
        
        # --- C. Ask Kelly for Stake ---
        stake = BayesianFootball.Signals.compute_stake(signal_algo, dist_probs, odds)
        
        # --- D. Record Result ---
        pnl = 0.0
        if stake > 0
            if row.is_winner
                pnl = stake * (odds - 1.0)
            else
                pnl = -stake
            end
        end
        
        push!(ledger_entries, (
            match_id = row.match_id,
            date = row.date,
            stake = stake,
            pnl = pnl,
            odds = odds,
            model_prob_mean = mean(dist_probs)
        ))
    end
    
    return DataFrame(ledger_entries)
end

# from the package
function run_hacked_backtest(
    exp_res,
    ds,
    signals, 
    market_df,
    market_config;
    bias=0.21,
    temp=0.6,
    odds_column::Symbol=:odds_close
)
    # A. Experiment Bridge: Get Latents
    # (This might trigger heavy computation/allocations)
    latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
    hacked_latents = calibrate_latents(latents_raw.df, bias_shift=bias, temperature=temp)

    latents = BayesianFootball.Experiments.LatentStates(
                      hacked_latents,
                      latents_raw.model 
              )
    

    # B. Prediction View: Get Probabilities
    # (This likely uses Threads inside to compute scores for chains)
    ppd = BayesianFootball.Predictions.model_inference(latents; market_config=market_config)

    # C. Signal Agent: Get Decisions
    # (This might use Threads to process thousands of rows)
    sig_result = BayesianFootball.Signals.process_signals(ppd, market_df.df, signals; odds_column=odds_column)

    # D. Enrichment
    # Add metadata so we can distinguish this model's bets in the big ledger
    df = sig_result.df

    # Calculate PnL immediately ---
    df.pnl = map(eachrow(df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    m_name = BayesianFootball.Models.model_name(exp_res.config.model)
    m_params = BayesianFootball.Models.model_parameters(exp_res.config.model)

    df.model_name = fill(m_name, nrow(df))
    df.model_parameters = fill(m_params, nrow(df))
    
    return df
end


# Define your Kelly Signal
baker = BayesianKelly()

# 1. The "Control" (No Hacks)
ledger_control = run_hacked_backtest(latents.df, market_data.df, baker, bias=0.0, temp=1.0)
println("Control Total PnL: ", sum(ledger_control.pnl))

# 2. The "Bias Fix" (Shift only)
ledger_bias = run_hacked_backtest(latents.df, market_data.df, baker, bias=0.21, temp=1.0)
println("Bias Fix Total PnL: ", sum(ledger_bias.pnl))

# 3. The "Full Hack" (Shift + Sharpen)
# Temp 0.5 effectively doubles your Z-scores (since 0.5 * 2 = 1.0)
ledger_full = run_hacked_backtest(latents.df, market_data.df, baker, bias=0.11, temp=0.5)
println("Full Hack Total PnL: ", sum(ledger_full.pnl))

# Quick Diagnostics
println("\n--- Bets Placed ---")
println("Control: ", count(>(1e-6), ledger_control.stake))
println("Bias Fix: ", count(>(1e-6), ledger_bias.stake))
println("Full Hack: ", count(>(1e-6), ledger_full.stake))





ledger_hacked = run_hacked_backtest(
    exp_res,
    ds, 
    my_signals,
    market_data, 
    BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG;
    bias=0.21,
    temp=0.6,
)
 
ledger_over15 = subset(ledger_hacked, :selection => ByRow(isequal(:over_15)))
println("Control Total PnL: ", sum(ledger_over15.pnl))




ledger_hacked_c = run_hacked_backtest(
    exp_res,
    ds, 
    my_signals,
    market_data, 
    BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG;
    bias=0,
    temp=1,
)
 
ledger_over15_c = subset(ledger_hacked_c, :selection => ByRow(isequal(:over_15)))
println("Control Total PnL: ", sum(ledger_over15_c.pnl))


#=
julia> println("Control Total PnL: ", sum(ledger_over15_c.pnl))
Control Total PnL: 0.35901922446332185

julia> println("Control Total PnL: ", sum(ledger_over15.pnl))
Control Total PnL: 0.36112857568971135

=#


# z test 

# Filter for active bets only
bets_control = subset(ledger_over15_c, :stake => ByRow(>(1e-6)))
bets_hacked  = subset(ledger_over15,   :stake => ByRow(>(1e-6)))

println("--- Vital Signs ---")
println("1. Volume (Number of Bets):")
println("   Control: ", nrow(bets_control))
println("   Hacked:  ", nrow(bets_hacked))

#=
1. Volume (Number of Bets):
julia> println("   Control: ", nrow(bets_control))
   Control: 51
julia> println("   Hacked:  ", nrow(bets_hacked))
   Hacked:  193
=#

println("\n2. Average Stake:")
println("   Control: ", round(mean(bets_control.stake), digits=4))
println("   Hacked:  ", round(mean(bets_hacked.stake), digits=4))

#=
2. Average Stake:
julia> println("   Control: ", round(mean(bets_control.stake), digits=4))
   Control: 0.0361
julia> println("   Hacked:  ", round(mean(bets_hacked.stake), digits=4))
   Hacked:  0.0732
=#

println("\n3. ROI (Yield):")
dropmissing!(bets_control, [:pnl, :stake])
dropmissing!(bets_hacked, [:pnl, :stake])
roi_c = sum(bets_control.pnl) / sum(bets_control.stake)
roi_h = sum(bets_hacked.pnl)  / sum(bets_hacked.stake)
println("   Control: ", round(roi_c * 100, digits=2), "%")
println("   Hacked:  ", round(roi_h * 100, digits=2), "%")

#=
julia> println("   Control: ", round(roi_c * 100, digits=2), "%")
   Control: 19.53%

julia> println("   Hacked:  ", round(roi_h * 100, digits=2), "%")
   Hacked:  2.55%
=#



using Plots, Distributions, Statistics, HypothesisTests, StatsBase, DataFrames, StatsPlots

# ------------------------------------------------------------------
# 1. Compute Z-Scores for a Specific Scenario (Bias/Temp)
# ------------------------------------------------------------------
function compute_prob_z_scores(ds, exp_res, market_df; bias=0.0, temp=1.0)
    # A. Get Latents & Hack
    latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
    hacked_latents = calibrate_latents(latents_raw.df, bias_shift=bias, temperature=temp)
    
    # B. Market Data
    mkt_subset = filter(row -> row.market_name == "Over/Under" && row.selection == :over_15, market_df)
    joined = innerjoin(hacked_latents, select(mkt_subset, :match_id, :prob_implied_close), on=:match_id)
    
    z_scores = Float64[]
    
    for row in eachrow(joined)
        # 1. Generate Model Probability Distribution for Over 1.5
        # For each sample in the posterior (h_vec, a_vec, r_vec):
        lambdas = row.λ_h .+ row.λ_a
        rs = row.r
        
        # Calculate P(Over 1.5) for every posterior sample
        # P(Over) = 1 - CDF(NegBin, 1)
        # Note: We use the NegBin CDF logic here
        
        # Vectorized calculation over the samples
        probs_over_samples = map((lam, r_val) -> begin
            p = r_val / (r_val + lam)
            p0 = p^r_val
            p1 = r_val * (1-p) * p0
            return 1.0 - (p0 + p1)
        end, lambdas, rs)
        
        # 2. Get Moments of the PROBABILITY Distribution
        mu_prob = mean(probs_over_samples)
        std_prob = std(probs_over_samples)
        
        # 3. Market Probability
        mkt_prob = row.prob_implied_close
        
        # 4. Z-Score (Signal / Noise)
        # Positive Z = Model sees HIGHER prob than Market
        if std_prob > 1e-6
            z = (mu_prob - mkt_prob) / std_prob
            push!(z_scores, z)
        else
            push!(z_scores, NaN)
        end
    end
    
    joined.z_score = z_scores
    return joined
end
function compute_scenario_z_scores(ds, exp_res, market_df; bias=0.0, temp=1.0)
    # A. Get OOS Predictions
    latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
    
    # B. Apply Calibration (The "Hack")
    # (Assuming you have your calibrate_latents function defined from previous steps)
    hacked_latents = calibrate_latents(latents_raw.df, bias_shift=bias, temperature=temp)
    
    # C. Join with Market Data (Over 1.5 Close Odds)
    # We filter for Over 1.5 to get the relevant odds
    mkt_subset = filter(row -> row.selection == :over_15, market_df)
    
    # Join on Match ID
    z_df = innerjoin(
        select(hacked_latents, :match_id, :λ_h, :λ_a),
        select(mkt_subset, :match_id, :fair_odds_close, :prob_fair_close, :is_winner),
        on = :match_id
    )
    
    # D. Calculate Z-Scores
    z_scores = Float64[]
    
    for row in eachrow(z_df)
        # 1. Market Implied Lambda (Approximate Inverse Poisson)
        # Prob(Under 1.5) = 1 - (1/Odds)
        # Poisson(0)+Poisson(1) ≈ exp(-lambda)*(1+lambda)
        # Quick approx for Z-score purpose: lambda ≈ -log(Prob_Under) 
        # (Or use the solver you wrote previously if you want exactness)
        lambda_mkt = -log(row.prob_fair_close) # Poisson Approximation
        
        # 2. Model Distribution (Total Goals)
        total_goals_dist = row.λ_h .+ row.λ_a
        log_dist = log.(total_goals_dist)
        
        # 3. Moments
        mu_model = mean(log_dist)
        sigma_model = std(log_dist)
        
        # 4. Z-Score
        # (Model - Market) / Uncertainty
        z = (mu_model - log(lambda_mkt)) / sigma_model
        push!(z_scores, z)
    end
    
    z_df.z_score = z_scores
    return z_df
end

# Ensure you have your solver defined (from your earlier snippet)
# function solve_negbin_lambda_from_under15(prob_under, r_val) ... end

function compute_scenario_z_scores(ds, exp_res, market_df; bias=0.0, temp=1.0)
    # A. Get OOS Predictions & Calibrate
    latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
    hacked_latents = calibrate_latents(latents_raw.df, bias_shift=bias, temperature=temp)
    
    # B. Filter Market Data (Over 1.5 Only)
    # We need 'odds_close' or 'prob_implied_close' for the Over 1.5 selection
    mkt_subset = filter(row -> row.selection == :over_15, market_df)
    
    # Join on Match ID
    z_df = innerjoin(
        select(hacked_latents, :match_id, :λ_h, :λ_a, :r), # Include 'r'
        select(mkt_subset, :match_id, :fair_odds_close, :is_winner),
        on = :match_id
    )
    
    z_scores = Float64[]
    
    for row in eachrow(z_df)
        # 1. Get Market Probability for UNDER 1.5
        # The odds are for OVER 1.5, so we invert.
        prob_over = 1.0 / row.fair_odds_close
        prob_under = 1.0 - prob_over
        
        # 2. Get Model Dispersion (r)
        # We use the mean 'r' of the model to interpret the market price
        r_mean = mean(row.r)
        
        # 3. Solve Inverse Problem: What Lambda implies this Market Price?
        # (Using your rigorous solver instead of the Poisson approx)
        lambda_mkt = solve_negbin_lambda_from_under15(prob_under, r_mean)
        
        if isnan(lambda_mkt) || lambda_mkt <= 0
            push!(z_scores, NaN)
            continue
        end
        
        # 4. Model Distribution (Total Goals Lambda)
        total_goals_dist = row.λ_h .+ row.λ_a
        
        # Transform to Log-Space (Lambda is Log-Normal distributed)
        log_dist = log.(total_goals_dist)
        
        # 5. Moments & Z-Score
        mu_model = mean(log_dist)
        sigma_model = std(log_dist)
        
        # Calculate Z: (Model Log-Lambda - Market Log-Lambda) / Uncertainty
        z = (mu_model - log(lambda_mkt)) / sigma_model
        push!(z_scores, z)
    end
    
    z_df.z_score = z_scores
    return z_df
end

# ------------------------------------------------------------------
# 2. Run Diagnostics & Plotting
# ------------------------------------------------------------------
function run_z_diagnostics(z_df, scenario_name; output_prefix="z_diag")
    
    clean_z = filter(!isnan, z_df.z_score)
    n = length(clean_z)
    
    println("\n========================================")
    println("DIAGNOSTICS: $scenario_name (N=$n)")
    println("========================================")

    # A. Statistics
    sw = ShapiroWilkTest(clean_z)
    mu = mean(clean_z)
    sig = std(clean_z)
    sk = skewness(clean_z)
    ku = kurtosis(clean_z) # Excess kurtosis

    println("Mean (Bias):      ", round(mu, digits=4))
    println("Std Dev (Risk):   ", round(sig, digits=4), " (Target: 1.0)")
    println("Skewness:         ", round(sk, digits=4))
    println("Ex. Kurtosis:     ", round(ku, digits=4))
    println("Shapiro-Wilk P:   ", round(pvalue(sw), digits=5))
    
    # B. Histogram Plot
    p_hist = StatsPlots.histogram(clean_z, 
        normalize = :pdf, bins = 40, label = "$scenario_name Z", 
        alpha = 0.6, color = :blue,
        title = "Calibration: $scenario_name",
        xlabel = "Z-Score", ylabel = "Density"
    );
    
    # Overlay Normal
    x_grid = range(minimum(clean_z), maximum(clean_z), length=200);
    plot!(p_hist, x_grid, pdf.(Normal(0, 1), x_grid), 
        label="N(0,1)", color=:red, linewidth=3, linestyle=:dash
    );
    
    # Mean Line
    StatsPlots.vline!(p_hist, [mu], label="Mean Z", color=:black, linewidth=2);
    
    # Save Histogram
    hist_filename = "$(output_prefix)_hist.html"
    Plots.savefig(p_hist, hist_filename)
    println("Saved Histogram:  $hist_filename")

    # C. Q-Q Plot
    qq_p = plot(title="Q-Q Plot: $scenario_name", legend=:topleft);
    ref_x = range(-4, 4, length=100);
    plot!(qq_p, ref_x, ref_x, label="Normal Reference", color=:red, linestyle=:dash);
    
    sorted_z = sort(clean_z)
    probs = (1:n) ./ (n + 1)
    quantiles_theo = quantile.(Normal(0,1), probs)
    scatter!(qq_p, quantiles_theo, sorted_z, label="Residuals", color=:blue, alpha=0.6);
    
    # Save Q-Q
    qq_filename = "$(output_prefix)_qq.html"
    Plots.savefig(qq_p, qq_filename)
    println("Saved Q-Q Plot:   $qq_filename")
end


# 1. Analyze Control (No Bias, Temp 1.0)
# ==========================================================
z_control = compute_scenario_z_scores(ds, exp_res, market_data.df; bias=0.0, temp=1.0);
run_z_diagnostics(z_control, "Control_Model", output_prefix="control")

#=
========================================
DIAGNOSTICS: Control_Model (N=684)
========================================
Mean (Bias):      -0.2073
Std Dev (Risk):   0.519 (Target: 1.0)
Skewness:         0.2926
Ex. Kurtosis:     0.1573
Shapiro-Wilk P:   0.00016
Saved Histogram:  control_hist.html
Saved Q-Q Plot:   control_qq.html

=#


# 2. Analyze Hacked (Bias 0.21, Temp 0.6)
# ==========================================================
z_hacked = compute_scenario_z_scores(ds, exp_res, market_data.df; bias=0.11, temp=0.6);
run_z_diagnostics(z_hacked, "Hacked_Model", output_prefix="hacked")

#=


julia> z_hacked = compute_scenario_z_scores(ds, exp_res, market_data.df; bias=0.21, temp=0.6);
julia> run_z_diagnostics(z_hacked, "Hacked_Model", output_prefix="hacked")

========================================
DIAGNOSTICS: Hacked_Model (N=684)
========================================
Mean (Bias):      0.435
Std Dev (Risk):   0.8887 (Target: 1.0)
Skewness:         0.1899
Ex. Kurtosis:     0.0671
Shapiro-Wilk P:   0.00261
Saved Histogram:  hacked_hist.html
Saved Q-Q Plot:   hacked_qq.html

julia> z_hacked = compute_scenario_z_scores(ds, exp_res, market_data.df; bias=0.0, temp=0.6);
julia> run_z_diagnostics(z_hacked, "Hacked_Model", output_prefix="hacked")

========================================
DIAGNOSTICS: Hacked_Model (N=684)
========================================
Mean (Bias):      -0.4519
Std Dev (Risk):   0.8707 (Target: 1.0)
Skewness:         0.2921
Ex. Kurtosis:     0.1597
Shapiro-Wilk P:   0.00015
Saved Histogram:  hacked_hist.html
Saved Q-Q Plot:   hacked_qq.html


julia> z_hacked = compute_scenario_z_scores(ds, exp_res, market_data.df; bias=0.11, temp=0.6);
julia> run_z_diagnostics(z_hacked, "Hacked_Model", output_prefix="hacked")

========================================
DIAGNOSTICS: Hacked_Model (N=684)
========================================
Mean (Bias):      0.0127
Std Dev (Risk):   0.8784 (Target: 1.0)
Skewness:         0.2388
Ex. Kurtosis:     0.1062
Shapiro-Wilk P:   0.00065
Saved Histogram:  hacked_hist.html
Saved Q-Q Plot:   hacked_qq.html

=#


# 1. Run the Optimized Backtest
ledger_opt = run_hacked_backtest(
    exp_res,
    ds, 
    my_signals,
    market_data, 
    BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG;
    bias=0.11,   # The Optimized Bias
    temp=0.6,    # The Optimized Temp
)

# 2. Filter for Over 1.5
ledger_opt_over = subset(ledger_opt, :selection => ByRow(isequal(:over_15)))

# 3. Check Vital Signs (Did we unlock volume?)
active_bets = subset(ledger_opt_over, :stake => ByRow(>(1e-6)))

println("--- OPTIMIZED MODEL RESULTS ---")
println("Total PnL:      ", round(sum(active_bets.pnl), digits=2))
println("Total Volume:   ", nrow(active_bets))
println("Avg Stake:      ", round(mean(active_bets.stake), digits=4))
println("ROI (Yield):    ", round((sum(active_bets.pnl) / sum(active_bets.stake)) * 100, digits=2), "%")

# 4. Compare with Control (Raw Model)
# (Assuming 'ledger_over15_c' is still in memory from previous steps)
println("Control PnL:    ", round(sum(ledger_over15_c.pnl), digits=2))


function display_hacked_backtest(exp_res, bais, temp) 

    ledger_opt = run_hacked_backtest(
        exp_res,
        ds, 
        my_signals,
        market_data, 
        BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG;
        bias=bais,   # The Optimized Bias
        temp=temp,    # The Optimized Temp
    )

    # 2. Filter for Over 1.5
    ledger_opt_over = subset(ledger_opt, :selection => ByRow(isequal(:over_15)))

    # 3. Check Vital Signs (Did we unlock volume?)
    active_bets = subset(ledger_opt_over, :stake => ByRow(>(1e-6)))

    println("--- OPTIMIZED MODEL RESULTS ---")
    println("Total PnL:      ", round(sum(active_bets.pnl), digits=2))
    println("Total Volume:   ", nrow(active_bets))
    println("Avg Stake:      ", round(mean(active_bets.stake), digits=4))
    println("ROI (Yield):    ", round((sum(active_bets.pnl) / sum(active_bets.stake)) * 100, digits=2), "%")
    println("Win precentage: ", round(  (sum(active_bets.is_winner) / nrow(active_bets))*100, digits=3), "%")


end 


display_hacked_backtest(exp_res, 0.21, 0.6)

#= 
julia> display_hacked_backtest(exp_res, 0.11, 0.6)
Running Inference on 708 matches...
--- OPTIMIZED MODEL RESULTS ---
Total PnL:      0.47
Total Volume:   109
Avg Stake:      0.059
ROI (Yield):    7.26%
Win precentage: 76.147%


julia> display_hacked_backtest(exp_res, 0.0, 1)
Running Inference on 708 matches...
--- OPTIMIZED MODEL RESULTS ---
Total PnL:      0.36
Total Volume:   51
Avg Stake:      0.0361
ROI (Yield):    19.53%
Win precentage: 74.51%

julia> display_hacked_backtest(exp_res, 0.11, 0.4)
Running Inference on 708 matches...
--- OPTIMIZED MODEL RESULTS ---
Total PnL:      0.49
Total Volume:   109
Avg Stake:      0.0723
ROI (Yield):    6.22%
Win precentage: 76.147%


julia> display_hacked_backtest(exp_res, 0.21, 0.6)
Running Inference on 708 matches...
--- OPTIMIZED MODEL RESULTS ---
Total PnL:      0.36
Total Volume:   193
Avg Stake:      0.0732
ROI (Yield):    2.55%
Win precentage: 75.648%

=#


using Plots, DataFrames, Dates

# 1. Run the two best backtests to get the full ledgers
ledger_control = run_hacked_backtest(exp_res, ds, my_signals, market_data, BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG; bias=0.0, temp=1.0)
ledger_opt     = run_hacked_backtest(exp_res, ds, my_signals, market_data, BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG; bias=0.025, temp=0.4)
ledger_vol     = run_hacked_backtest(exp_res, ds, my_signals, market_data, BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG; bias=0.21, temp=1)

# 2. Filter for Over 1.5
lc = subset(ledger_control, :selection => ByRow(isequal(:over_15)))
lo = subset(ledger_opt,     :selection => ByRow(isequal(:over_15)))
lv = subset(ledger_vol,     :selection => ByRow(isequal(:over_15)))

# 3. Sort by Date
sort!(lc, :date)
sort!(lo, :date)
sort!(lv, :date)

# 4. Calculate Cumulative PnL
lc.cum_pnl = cumsum(lc.pnl)
lo.cum_pnl = cumsum(lo.pnl)
lv.cum_pnl = cumsum(lv.pnl)

# 5. Plot
p = plot(lc.date, lc.cum_pnl, label="Control (Bias 0, Temp 1)", linewidth=2, title="Equity Curve Comparison", legend = :outertopright, size=(1200, 600));
plot!(p, lo.date, lo.cum_pnl, label="Optimized (Bias 0.025, Temp 0.4)", linewidth=2, color=:green);
plot!(p, lv.date, lv.cum_pnl, label="Optimized (Bias 0.1, Temp 0.4)", linewidth=2, color=:red);

# Add a zero line
StatsPlots.hline!(p, [0], color=:black, linestyle=:dash, label="");

Plots.savefig(p, "equity_curve_comparison.html")
println("Saved equity curve to equity_curve_comparison.html")



# ----- 
using DataFrames, Statistics, Plots, ProgressMeter

function optimize_calibration(
    ds, exp_res, signals, market_df;
    bias_range = -0.1:0.02:0.3,   # Search from -0.1 to +0.3 in steps of 0.02
    temp_range = 0.3:0.1:1.2,     # Search Temp from 0.3 to 1.2
    min_volume = 50,              # Ignore settings with too few bets
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)
    results = DataFrame(
        bias = Float64[], 
        temp = Float64[], 
        pnl = Float64[], 
        roi = Float64[], 
        volume = Int[], 
        win_rate = Float64[],
        avg_stake = Float64[]
    )
    
    # Total iterations for progress bar
    total_iter = length(bias_range) * length(temp_range)
    p = Progress(total_iter, 1, "Optimizing Strategy...")

    # Grid Search Loop
    for b in bias_range
        for t in temp_range
            # Run the Backtest
            ledger = run_hacked_backtest(
                exp_res, ds, signals, market_df, market_config;
                bias = b, temp = t
            )
            
            # Filter for Over 1.5 Active Bets
            active = subset(ledger, 
                :selection => ByRow(isequal(:over_15)), 
                :stake => ByRow(>(1e-6))
            )
            
            vol = nrow(active)
            
            # Calculate Metrics (if volume is sufficient)
            if vol >= min_volume
                total_pnl = sum(active.pnl)
                total_stake = sum(active.stake)
                roi = (total_pnl / total_stake) * 100
                wr = (sum(active.is_winner) / vol) * 100
                avg_s = mean(active.stake)
                
                push!(results, (b, t, total_pnl, roi, vol, wr, avg_s))
            else
                # Push zeros/NaN for low volume to keep grid shape (optional, but skipping is cleaner)
            end
            
            next!(p)
        end
    end
    
    # Sort by Total PnL (Descending)
    sort!(results, :pnl, rev=true)
    
    return results
end


# 1. Run Optimization
# ==========================================================
# (This might take 1-2 minutes depending on grid size)
opt_results = optimize_calibration(
    ds, exp_res, my_signals, market_data;
    bias_range = -0.05:0.025:0.25,  # Fine-tune around your 0.11 finding
    temp_range = 0.4:0.1:1.1
)

# 2. Display Top 10 Settings
# ==========================================================
println("--- TOP 10 PARAMETER SETTINGS ---")
display(first(opt_results, 20))

#=
julia> display(first(opt_results, 20))
20×7 DataFrame
 Row │ bias     temp     pnl       roi       volume  win_rate  avg_stake 
     │ Float64  Float64  Float64   Float64   Int64   Float64   Float64   
─────┼───────────────────────────────────────────────────────────────────
   1 │   0.025      0.4  0.538381  13.8321       59   76.2712  0.0659703
   2 │   0.05       0.4  0.533909  11.1386       72   76.3889  0.066574
   3 │   0.075      0.4  0.52622    8.91329      85   75.2941  0.0694561
   4 │   0.05       0.5  0.508056  11.8098       72   76.3889  0.0597497
   5 │   0.075      0.5  0.506469   9.54897      85   75.2941  0.0623989
   6 │   0.025      0.5  0.505713  14.4951       59   76.2712  0.0591332
   7 │   0.1        0.4  0.503962   6.94203     101   75.2475  0.071877
   8 │   0.1        0.5  0.491645   7.52985     101   75.2475  0.0646464
   9 │   0.075      0.6  0.484014  10.0711       85   75.2941  0.0565408
  10 │   0.05       0.6  0.481175  12.3538       72   76.3889  0.0540967
  11 │   0.1        0.6  0.474997   8.02023     101   75.2475  0.0586385
  12 │   0.025      0.6  0.474059  15.0207       59   76.2712  0.0534923
  13 │   0.125      0.4  0.467395   5.25237     118   77.9661  0.0754131
  14 │   0.125      0.5  0.463371   5.78224     118   77.9661  0.0679127
  15 │   0.075      0.7  0.461906  10.5237       85   75.2941  0.0516377
  16 │   0.1        0.7  0.45689    8.44148     101   75.2475  0.0535886
  17 │   0.05       0.7  0.456053  12.823        72   76.3889  0.0493962
  18 │   0.125      0.6  0.453531   6.23354     116   77.5862  0.0627212
  19 │   0.025      0.7  0.445685  15.4735       59   76.2712  0.0488189
  20 │   0.075      0.8  0.441425  10.9359       85   75.2941  0.0474881


=#

# 3. Visualization: The PnL Heatmap
# ==========================================================
# We pivot the data to create a matrix for plotting
# (We need a dense grid, so we might have gaps if min_volume triggered)

# Helper to plot heatmap
try
    # extract vectors
    x = sort(unique(opt_results.bias))
    y = sort(unique(opt_results.temp))
    z = zeros(length(y), length(x)) # Temp is rows (Y), Bias is cols (X)
    
    for (i, t_val) in enumerate(y)
        for (j, b_val) in enumerate(x)
            # Find row in results
            match = filter(r -> r.bias ≈ b_val && r.temp ≈ t_val, opt_results)
            if nrow(match) > 0
                z[i, j] = match.pnl[1]
            else
                z[i, j] = NaN # No data (low volume)
            end
        end
    end

    hm = heatmap(
        x, y, z,
        title = "Strategy PnL Surface",
        xlabel = "Bias (Shift)",
        ylabel = "Temperature (Variance Scaling)",
        color = :viridis,
        clims = (0, maximum(filter(!isnan, z))) # Set color limits
    )
    
    # Mark the Max Point
    best = opt_results[1, :]
    scatter!(hm, [best.bias], [best.temp], 
        marker=:star, color=:red, label="Max PnL", markersize=8
    )

    savefig(hm, "optimization_heatmap.html")
    println("\nSaved Heatmap to optimization_heatmap.html")
catch e
    println("Could not generate heatmap: $e")
end




#--- regimes shifts 
using DataFrames, Statistics, Dates, PrettyTables

function diagnose_regime_drift(ds, exp_res, market_df; bias=0.025, temp=0.4)
    # 1. Get the calibrated Z-scores for the "Sniper" settings
    # (Re-using your compute_prob_z_scores or equivalent)
    # Let's assume you use the lambda-space one we fixed earlier
    z_df = compute_scenario_z_scores(ds, exp_res, market_df; bias=bias, temp=temp)
    
    # 2. Add Date information (Join back with match data if needed)
    # z_df usually has match_id. We need dates.
    # We can join with 'market_df' or 'ds.matches' to get dates
    with_dates = innerjoin(z_df, select(ds.matches, :match_id, :match_date), on=:match_id)
    
    # 3. Create "Season" column (Approximation)
    # Assuming standard Aug-May season. 
    with_dates.season = map(d -> month(d) > 7 ? year(d) : year(d)-1, with_dates.match_date)
    
    # 4. Group by Season and Analyze
    gdf = groupby(with_dates, :season)
    
    stats = combine(gdf, 
        :z_score => mean => :mean_z_signal,
        :z_score => std => :z_volatility,
        :is_winner => (w -> mean(skipmissing(w))) => :win_rate,
        nrow => :count
    )
    
    # 5. Add Implied "P&L" Proxy (Did the signal work?)
    # Correlation between Z and Winner per season
    stats.correlation = [cor(g.z_score, g.is_winner) for g in gdf]
    
    sort!(stats, :season)
    
    println("--- REGIME DRIFT DIAGNOSTIC (Bias=$bias, Temp=$temp) ---")
    pretty_table(stats)
    
    return with_dates
end

# Run the diagnostic
drift_data = diagnose_regime_drift(ds, exp_res, market_data.df; bias=0.21, temp=1);


#=
julia> # Run the diagnostic                                                                                                                                                                                                                                                    
       drift_data = diagnose_regime_drift(ds, exp_res, market_data.df; bias=0.025, temp=0.4)                                                                                                                                                                                   
--- REGIME DRIFT DIAGNOSTIC (Bias=0.025, Temp=0.4) ---                                                                                                                                                                                                                         
┌────────┬───────────────┬──────────────┬──────────┬───────┬─────────────┐                                                                                                                                                                                                     
│ season │ mean_z_signal │ z_volatility │ win_rate │ count │ correlation │                                                                                                                                                                                                     
│  Int64 │       Float64 │      Float64 │  Float64 │ Int64 │     Float64 │                                                                                                                                                                                                     
├────────┼───────────────┼──────────────┼──────────┼───────┼─────────────┤                                                                                                                                                                                                     
│   2023 │     -0.289845 │      1.28279 │ 0.773077 │   260 │  -0.0298142 │                                                                                                                                                                                                     
│   2024 │     -0.718258 │      1.26152 │ 0.777778 │   279 │ -0.00489194 │                                                                                                                                                                                                     
│   2025 │     -0.787029 │      1.37314 │ 0.806897 │   145 │   0.0478953 │                                                                                                                                                                                                     
└────────┴───────────────┴──────────────┴──────────┴───────┴─────────────┘      

--- REGIME DRIFT DIAGNOSTIC (Bias=0.0, Temp=1) ---
┌────────┬───────────────┬──────────────┬──────────┬───────┬─────────────┐
│ season │ mean_z_signal │ z_volatility │ win_rate │ count │ correlation │
│  Int64 │       Float64 │      Float64 │  Float64 │ Int64 │     Float64 │
├────────┼───────────────┼──────────────┼──────────┼───────┼─────────────┤
│   2023 │    -0.0949363 │     0.511008 │ 0.773077 │   260 │  -0.0284912 │
│   2024 │     -0.269968 │     0.497273 │ 0.777778 │   279 │ -0.00432358 │
│   2025 │     -0.287986 │     0.541773 │ 0.806897 │   145 │   0.0447576 │
└────────┴───────────────┴──────────────┴──────────┴───────┴─────────────┘
--- REGIME DRIFT DIAGNOSTIC (Bias=0.21, Temp=1) ---
┌────────┬───────────────┬──────────────┬──────────┬───────┬─────────────┐
│ season │ mean_z_signal │ z_volatility │ win_rate │ count │ correlation │
│  Int64 │       Float64 │      Float64 │  Float64 │ Int64 │     Float64 │
├────────┼───────────────┼──────────────┼──────────┼───────┼─────────────┤
│   2023 │      0.418236 │     0.519446 │ 0.773077 │   260 │  -0.0331375 │
│   2024 │      0.280898 │      0.51152 │ 0.777778 │   279 │ -0.00154471 │
│   2025 │      0.230693 │     0.555043 │ 0.806897 │   145 │   0.0587114 │
└────────┴───────────────┴──────────────┴──────────┴───────┴─────────────┘
=#



# Adaptive Backtest code 
using DataFrames, Statistics, Dates, Distributions, ProgressMeter

# ------------------------------------------------------------------
# 1. Helper: Solvers & Updates
# ------------------------------------------------------------------

"""
Exponential Smoothing Update
state: Previous value
observation: New value
alpha: Decay factor (Higher = Faster adaptation, Lower = Smoother)
"""
update_ema(state, obs, alpha) = isnan(state) ? obs : (alpha * obs + (1.0 - alpha) * state)

function get_market_lambda(prob_under, r_val)
    # Re-using your robust solver logic
    # If prob_under is extreme, clamp it to avoid Inf
    p_safe = clamp(prob_under, 0.001, 0.999)
    try
        # Simple solver backup if the root finder is slow/unavailable in this scope
        # (Using the previous Approx or Solver function is fine. 
        #  Here using a robust approximation for speed in the loop)
        return -log(p_safe) # Poisson proxy is fast; use your NegBin solver for precision
    catch
        return 2.5
    end
end

# ------------------------------------------------------------------
# 2. The Adaptive Runner
# ------------------------------------------------------------------

function run_adaptive_backtest(
    ds, exp_res, signals, market_df;
    alpha_bias = 0.05,      # Learning rate for Bias (e.g., 0.05 ~ 20 matches)
    alpha_temp = 0.02,      # Learning rate for Temp (Slower, e.g., 0.02 ~ 50 matches)
    initial_bias = 0.075,   # Start with your "Robust" finding
    initial_temp = 0.4,     # Start with your "Robust" finding
    min_temp = 0.2,         # Safety clamp (don't get too confident)
    max_temp = 1.5,         # Safety clamp (don't get too scared)
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)
    # A. Setup
    # --------------------------------------------------------------
    latents_raw = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
    
    # Filter Market Data for Over 1.5 Close
    mkt_subset = filter(row -> row.selection == :over_15, market_df)
    
    # Join Data (Sort by Date strictly!)
    joined = innerjoin(
        select(latents_raw.df, :match_id, :λ_h, :λ_a, :r),
        select(mkt_subset, :match_id, :odds_close, :fair_odds_close, :is_winner, :date),
        on = :match_id
    )
    sort!(joined, :date)

    # B. State Initialization
    # --------------------------------------------------------------
    # The "Control State"
    current_bias_correction = initial_bias
    current_temp_factor = initial_temp
    
    # The "EMAs" (Trackers)
    # We track the 'Error Mean' directly. 
    # Bias Correction is roughly -ErrorMean.
    ema_error_mean = -initial_bias 
    ema_var_ratio = initial_temp^2
    
    ledger_entries = []
    history_log = DataFrame(
        date=Date[], 
        bias_used=Float64[], 
        temp_used=Float64[], 
        raw_error=Float64[],
        z_score=Float64[]
    )

    # C. The Loop (Chronological)
    # --------------------------------------------------------------
    
    for row in eachrow(joined)
        
        # --- 1. PREDICT (Using YESTERDAY'S Calibration) ---
        
        # Apply Current Bias/Temp to Model Distributions
        # (We do this "Just in Time" for the single match)
        raw_lambda_dist = row.λ_h .+ row.λ_a
        log_samples = log.(raw_lambda_dist)
        
        # Calibrate: Scale Variance, then Shift Mean
        mu_raw = mean(log_samples)
        centered = log_samples .- mu_raw
        
        # Apply Temp & Bias
        calibrated_log = mu_raw .+ (centered .* current_temp_factor) .+ current_bias_correction
        calibrated_lambda_vec = exp.(calibrated_log)
        
        # Calculate Probability of Over 1.5
        # (Vectorized NegBin calculation)
        r_vec = row.r
        probs_over = map((lam, r) -> begin
            p = r / (r + lam)
            1.0 - (p^r + r*(1-p)*p^r) 
        end, calibrated_lambda_vec, r_vec)
        
        # --- 2. TRADE (Kelly Decision) ---
        odds = row.odds_close
        stake = BayesianFootball.Signals.compute_stake(BayesianFootball.Signals.BayesianKelly(), probs_over, odds)
        
        # Record Result
        pnl = 0.0
        if stake > 0
            pnl = row.is_winner ? stake * (odds - 1.0) : -stake
        end
        
        # --- 3. MEASURE (Feedback Loop) ---
        
        prob_over = 1.0 / row.fair_odds_close
        prob_under = 1.0 - prob_over
        # (Inverse Problem)
        lambda_mkt = solve_negbin_lambda_from_under15(prob_under, mean(r_vec))
        
        # Calculate the "Raw Residual" (Log Space)
        # Error = Model_View (Uncalibrated) - Market_View
        # We track the RAW error to decide how much bias to add next time
        raw_model_log = mu_raw # The center of the raw model
        error_t = raw_model_log - log(lambda_mkt)
        
        # Calculate the "Variance Discrepancy"
        # Did the market fall inside our predicted range?
        sigma_model = std(log_samples)
        if sigma_model < 1e-6 sigma_model = 1.0 end # Safety
        
        # Normalized Squared Error
        # If this is > 1, the market is moving more than we predicted -> Increase Temp
        sq_error_ratio = ((error_t - ema_error_mean)^2) / (sigma_model^2)
        
        # --- 4. ADAPT (Update State) ---
        
        # Update Estimators
        ema_error_mean = update_ema(ema_error_mean, error_t, alpha_bias)
        ema_var_ratio  = update_ema(ema_var_ratio, sq_error_ratio, alpha_temp)
        
        # Update Controls for NEXT Match
        # Bias: Counter-act the mean error
        # If Error is Positive (Model > Market), we must Subtract (Negative Bias)
        current_bias_correction = -ema_error_mean
        
        # Temp: Scale to match volatility
        # If Ratio > 1, we are under-confident. Wait...
        # Let's check logic: Temp scales the MODEL variance.
        # If Market is far away (High Sq Error), we want WIDER distributions (Higher Temp) to be safe?
        # OR do we want SHARPER distributions?
        # Actually: High Error = High Uncertainty = High Temp.
        # But earlier we found Low Temp (0.4) was good (Sharpening).
        # Let's assume the EMA tracks the "Optimal Scaling".
        # We will clamp it to be safe.
        target_temp = sqrt(ema_var_ratio)
        current_temp_factor = clamp(target_temp, min_temp, max_temp)
        
        # Save History
        push!(ledger_entries, (
            match_id = row.match_id, date = row.date,
            selection = :over_15, odds = odds, closing_odds = row.odds_close,
            stake = stake, pnl = pnl, is_winner = row.is_winner,
            model_prob = mean(probs_over)
        ))
        
        push!(history_log, (row.date, current_bias_correction, current_temp_factor, error_t, 0.0))
        
    end
    
    return DataFrame(ledger_entries), history_log
end


# 1. Run Adaptive Backtest
# Alpha 0.05 approx last 20 games memory
ledger_adapt, log_adapt = run_adaptive_backtest(
    ds, exp_res, my_signals, market_data.df;
    alpha_bias = 0.1, 
    alpha_temp = 0.02, # Keep temp adaptation slow
    initial_bias = 0.0,
    initial_temp = 0.1
)

# 2. Plot the Adaptation Curve
p1 = plot(log_adapt.date, log_adapt.bias_used, 
    label="Dynamic Bias", title="Adaptive Calibration State",
    ylabel="Bias Correction", linewidth=2, color=:blue);

p2 = plot(log_adapt.date, log_adapt.temp_used, 
    label="Dynamic Temp", ylabel="Temp Factor", 
    linewidth=2, color=:orange, ylim=(0, 1.5));

p3 = plot(p1, p2, layout=(2,1), size=(1200, 600), legend=:outertopright);
Plots.savefig(p3, "adaptive_state.html")

# 3. Check PnL
active = subset(ledger_adapt, :stake => ByRow(>(1e-6)))
println("--- ADAPTIVE MODEL RESULTS ---")
println("Total PnL:    ", round(sum(active.pnl), digits=2))
println("ROI:          ", round((sum(active.pnl)/sum(active.stake))*100, digits=2), "%")
println("Volume:       ", nrow(active))


100 * sum(ledger_adapt.is_winner) / nrow(ledger_adapt)

not_active = subset(ledger_adapt, :stake => ByRow(<(1e-6)))

100* (sum(not_active.is_winner) / nrow(not_active))


function display_run_adaptive_backtest(
    ds, exp_res, signals, market_df;
    alpha_bias = 0.05,      # Learning rate for Bias (e.g., 0.05 ~ 20 matches)
    alpha_temp = 0.02,      # Learning rate for Temp (Slower, e.g., 0.02 ~ 50 matches)
    initial_bias = 0.075,   # Start with your "Robust" finding
    initial_temp = 0.4,     # Start with your "Robust" finding
    min_temp = 0.2,         # Safety clamp (don't get too confident)
    max_temp = 1.5,         # Safety clamp (don't get too scared)
)

    ledger_adapt, log_adapt = run_adaptive_backtest(
        ds, exp_res, my_signals, market_data.df;
        alpha_bias = alpha_bias, 
        alpha_temp = alpha_temp,
        initial_bias = initial_bias,
        initial_temp = initial_temp
    )

    # 2. Plot the Adaptation Curve
    p1 = plot(log_adapt.date, log_adapt.bias_used, 
        label="Dynamic Bias", title="Adaptive Calibration State",
        ylabel="Bias Correction", linewidth=2, color=:blue);

    p2 = plot(log_adapt.date, log_adapt.temp_used, 
        label="Dynamic Temp", ylabel="Temp Factor", 
        linewidth=2, color=:orange, ylim=(0, 1.5));

    p3 = plot(p1, p2, layout=(2,1), size=(1200, 600), legend=:outertopright);
    Plots.savefig(p3, "adaptive_state.html")

    # 3. Check PnL
    active = subset(ledger_adapt, :stake => ByRow(>(1e-6)))
    println("--- ADAPTIVE MODEL RESULTS ---")
    println("Total PnL:    ", round(sum(active.pnl), digits=2))
    println("ROI:          ", round((sum(active.pnl)/sum(active.stake))*100, digits=2), "%")
    println("Volume:       ", nrow(active))
    println("Volume Precent: ", round( 100*(nrow(active) /nrow(ledger_adapt)), digits=2), "%")
    println("Win Rate: ", round( 100*(sum(active.is_winner) / nrow(active)), digits=2), "%")

end


display_run_adaptive_backtest(
    ds, exp_res, my_signals, market_data.df;
     alpha_bias = 0.01, 
     alpha_temp = 0.1,
     initial_bias = 0.075,
     initial_temp = 0.1
)

#=

# cold start 
# Here is with the fair odds - namely with the vig removed.
julia> display_run_adaptive_backtest(
           ds, exp_res, my_signals, market_data.df;
           alpha_bias = 0.05, 
           alpha_temp = 0.02, # Keep temp adaptation slow
           initial_bias = 0.0,
           initial_temp = 0.1
       )
--- ADAPTIVE MODEL RESULTS ---
Total PnL:    1.93
ROI:          10.14%
Volume:       244
Volume Precent: 35.67%
Win Rate: 78.69%


# with the vig 
julia> display_run_adaptive_backtest(
           ds, exp_res, my_signals, market_data.df;
           alpha_bias = 0.05, 
           alpha_temp = 0.01,
           initial_bias = 0.075,
           initial_temp = 0.4
       )
--- ADAPTIVE MODEL RESULTS ---
Total PnL:    0.45
ROI:          11.96%
Volume:       64
Volume Precent: 9.36%
Win Rate: 79.69%

# with the vig 
julia> display_run_adaptive_backtest(
           ds, exp_res, my_signals, market_data.df;
            alpha_bias = 0.05, 
            alpha_temp = 0.1, # Keep temp adaptation slow
            initial_bias = 0.075,
            initial_temp = 0.4
       )
--- ADAPTIVE MODEL RESULTS ---
Total PnL:    0.41
ROI:          11.91%
Volume:       63
Volume Precent: 9.21%
Win Rate: 79.37%


# with vig 
julia> display_run_adaptive_backtest(
           ds, exp_res, my_signals, market_data.df;
            alpha_bias = 0.05, 
            alpha_temp = 0.01,
            initial_bias = 0.075,
            initial_temp = 0.4
       )
--- ADAPTIVE MODEL RESULTS ---
Total PnL:    0.45
ROI:          11.96%
Volume:       64
Volume Precent: 9.36%
Win Rate: 79.69%

 ++++++++++++++++++++++++++++

julia> display_run_adaptive_backtest(
           ds, exp_res, my_signals, market_data.df;
           alpha_bias = 0.05, 
           alpha_temp = 0.1, # Keep temp adaptation slow
           initial_bias = 0.075,
           initial_temp = 0.4
       )
--- ADAPTIVE MODEL RESULTS ---
Total PnL:    1.83
ROI:          9.81%
Volume:       257
Volume Precent: 37.57%
Win Rate: 78.6%


 ++++++++++++++++++++++++++++


julia> display_run_adaptive_backtest(
           ds, exp_res, my_signals, market_data.df;
           alpha_bias = 0.05, 
           alpha_temp = 0.01,
           initial_bias = 0.075,
           initial_temp = 0.4
       )
--- ADAPTIVE MODEL RESULTS ---
Total PnL:    1.92
ROI:          9.91%
Volume:       253
Volume Precent: 36.99%
Win Rate: 78.26%



=#
using Statistics, DataFrames

# 1. The "Blind Bettor" (Bet 1.0 on every single match)
blind_pnl = map(eachrow(ledger_adapt)) do r
    if r.is_winner
        return 1.0 * (r.closing_odds - 1.0) # Profit
    else
        return -1.0 # Loss
    end
end

# 2. The "Model Bettor" (Your Adaptive Strategy)
# We use the 'pnl' column already calculated in your ledger
model_pnl = ledger_adapt.pnl 

# 3. Compare Results
println("--- BLIND BETTING (Bet Everything) ---")
println("Total PnL:      ", round(sum(blind_pnl), digits=2))
println("ROI:            ", round((sum(blind_pnl) / nrow(ledger_adapt)) * 100, digits=2), "%")
println("Win Rate:       ", round((count(r -> r.is_winner, eachrow(ledger_adapt)) / nrow(ledger_adapt)) * 100, digits=1), "%")




println("\n--- MODEL BETTING (Selective) ---")
# Filter for actual bets placed
active = subset(ledger_adapt, :stake => ByRow(>(1e-6)))
println("Total PnL:      ", round(sum(active.pnl), digits=2))
println("ROI:            ", round((sum(active.pnl) / sum(active.stake)) * 100, digits=2), "%")
println("Win Rate:       ", round((count(r -> r.is_winner, eachrow(active)) / nrow(active)) * 100, digits=1), "%")
println("Volume Skipped: ", nrow(ledger_adapt) - nrow(active))



#=
# not correct was using fair odds - so the vig was removed : 
--- BLIND BETTING (Bet Everything) ---
Total PnL:      32.38
ROI:            4.73%
Win Rate:       78.2%

# with the closing market odds with vig 
Total PnL:      -12.35
ROI:            -1.81%
Win Rate:       78.2%



--- MODEL BETTING (Selective) ---
Total PnL:      1.93
ROI:            10.14%
Win Rate:       78.7%
Volume Skipped: 440


=#


using DataFrames, Statistics, ProgressMeter, Base.Threads

function optimize_adaptive_parameters_threaded(
    ds, exp_res, signals, market_df;
    alpha_bias_range = [0.01, 0.05, 0.1, 0.2],
    alpha_temp_range = [0.01, 0.05],
    init_bias_range  = [0.0, 0.05],
    init_temp_range  = [0.1, 0.4, 1.0],
    min_volume       = 30
)
    # 1. Flatten the Grid
    # We create a vector of all parameter combinations first.
    # This makes threading much easier than trying to thread nested loops.
    combos = []
    for ab in alpha_bias_range
        for at in alpha_temp_range
            for ib in init_bias_range
                for it in init_temp_range
                    push!(combos, (ab, at, ib, it))
                end
            end
        end
    end

    total_iter = length(combos)
    
    # 2. Prepare Storage for Results
    # We use a Vector of NamedTuples (thread-safe for assignment) 
    # instead of pushing to a DataFrame inside the loop (which causes race conditions).
    results_storage = Vector{Any}(undef, total_iter)
    
    # Progress Bar (SpinLock needed for thread safety)
    p = Progress(total_iter, 1, "Optimizing (16 Cores)...")
    progress_lock = ReentrantLock()

    # 3. The Threaded Loop
    @threads for i in 1:total_iter
        (ab, at, ib, it) = combos[i]
        
        # Run Backtest
        # (Assuming run_adaptive_backtest is defined and pure)
        ledger, _ = run_adaptive_backtest(
            ds, exp_res, signals, market_df;
            alpha_bias = ab,
            alpha_temp = at,
            initial_bias = ib,
            initial_temp = it
        )
        
        # Calculate Metrics
        active = subset(ledger, :stake => ByRow(>(1e-6)))
        vol = nrow(active)
        
        if vol >= min_volume
            total_pnl = sum(active.pnl)
            total_stake = sum(active.stake)
            roi = total_stake > 0 ? (total_pnl / total_stake) * 100 : 0.0
            wr = (sum(active.is_winner) / vol) * 100
            
            # Store result in the pre-allocated vector at index i
            results_storage[i] = (
                alpha_b = ab, 
                alpha_t = at, 
                init_b  = ib, 
                init_t  = it, 
                pnl     = total_pnl, 
                roi     = roi, 
                volume  = vol, 
                win_rate = wr
            )
        else
            results_storage[i] = nothing
        end
        
        # Safe Progress Update
        lock(progress_lock) do 
            next!(p)
        end
    end
    
    # 4. Convert to DataFrame
    # Filter out 'nothing' results (low volume)
    valid_results = filter(!isnothing, results_storage)
    
    if isempty(valid_results)
        return DataFrame()
    end

    # Construct DataFrame from the vector of tuples
    df = DataFrame(valid_results)
    
    # Sort by PnL
    sort!(df, :pnl, rev=true)
    
    return df
end



opt_threaded = optimize_adaptive_parameters_threaded(
    ds, exp_res, my_signals, market_data.df;
    alpha_bias_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],  # Expanded range since it's fast now
    alpha_temp_range = [0.01, 0.02, 0.05, 0.1],
    init_bias_range  = [0.0, 0.05, 0.075],
    init_temp_range  = [0.1, 0.2, 0.4, 0.6]
)

println("--- TOP 10 SETTINGS (THREADED) ---")
display(first(opt_threaded, 10))

#=

julia> display(first(opt_threaded, 10))
10×8 DataFrame
 Row │ alpha_b  alpha_t  init_b   init_t   pnl       roi      volume  win_rate 
     │ Float64  Float64  Float64  Float64  Float64   Float64  Int64   Float64  
─────┼─────────────────────────────────────────────────────────────────────────
   1 │    0.01     0.01    0.075      0.1  0.603035  13.2871      76   77.6316
   2 │    0.01     0.01    0.075      0.2  0.595331  13.2756      76   77.6316
   3 │    0.01     0.01    0.05       0.1  0.572941  13.5652      73   79.4521
   4 │    0.01     0.01    0.075      0.4  0.568306  13.193       74   78.3784
   5 │    0.01     0.01    0.05       0.2  0.565837  13.5449      73   79.4521
   6 │    0.01     0.02    0.075      0.1  0.563251  13.1445      74   78.3784
   7 │    0.01     0.02    0.075      0.2  0.559539  13.1294      74   78.3784
   8 │    0.02     0.01    0.075      0.1  0.556724  13.0338      69   78.2609
   9 │    0.02     0.01    0.075      0.2  0.548867  13.0043      69   78.2609
  10 │    0.01     0.02    0.075      0.4  0.54612   13.0639      74   78.3784



julia> display(first(opt_threaded, 10))
10×8 DataFrame
 Row │ alpha_b  alpha_t  init_b   init_t   pnl       roi      volume  win_rate 
     │ Float64  Float64  Float64  Float64  Float64   Float64  Int64   Float64  
─────┼─────────────────────────────────────────────────────────────────────────
   1 │    0.01     0.1     0.075      0.1  1.01591   11.0998     157   82.8025
   2 │    0.01     0.1     0.075      0.2  1.0154    11.0962     157   82.8025
   3 │    0.01     0.1     0.075      0.4  1.01347   11.0829     157   82.8025
   4 │    0.01     0.05    0.075      0.1  1.01263   10.948      159   83.0189
   5 │    0.01     0.1     0.075      0.6  1.01062   11.0634     157   82.8025
   6 │    0.01     0.05    0.075      0.2  1.01044   10.9356     159   83.0189
   7 │    0.01     0.05    0.075      0.4  1.00308   10.8957     159   83.0189
   8 │    0.01     0.02    0.075      0.1  0.998594  10.4153     160   83.125
   9 │    0.01     0.01    0.075      0.1  0.994973  10.0074     160   83.125
  10 │    0.01     0.05    0.075      0.6  0.993866  10.8501     159   83.0189



=#

sort(opt_threaded, :roi, rev=true)





using Base.Threads, ProgressMeter, DataFrames

function optimize_adaptive_parameters_threaded(
    ds, exp_res, market_data;
    alpha_bias_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    alpha_temp_range = [0.01, 0.02, 0.05, 0.1],
    init_bias_range  = [0.0, 0.05, 0.075],
    init_temp_range  = [0.1, 0.2, 0.4, 0.6],
    min_volume       = 30
)
    # 1. Flatten the Grid (using a clean list comprehension)
    combos = [(ab, at, ib, it) for ab in alpha_bias_range 
                               for at in alpha_temp_range 
                               for ib in init_bias_range 
                               for it in init_temp_range]
    
    total_iter = length(combos)
    println("Starting grid search over $total_iter combinations...")
    
    # 2. Prepare Storage
    # Vector of Any allows us to store NamedTuples or 'nothing' safely
    results_storage = Vector{Any}(undef, total_iter)
    
    # Progress Bar setup
    p = Progress(total_iter, 1, "Optimizing ($(Threads.nthreads()) Threads)...")
    progress_lock = ReentrantLock()

    # 3. The Threaded Loop
    @threads for i in 1:total_iter
        (ab, at, ib, it) = combos[i]
        
        # Run Backtest
        # (Make sure to silence the print statements inside the backtest 
        #  so your console doesn't get flooded!)
        ledger = run_adaptive_backtest_under(
            ds, exp_res, market_data;
            alpha_bias = ab,
            alpha_temp = at,
            initial_bias = ib,
            initial_temp = it
        )
        
        # Calculate Metrics
        active = subset(ledger, :stake => ByRow(>(1e-6)))
        vol = nrow(active)
        
        if vol >= min_volume
            total_pnl = sum(active.pnl)
            total_stake = sum(active.stake)
            roi = total_stake > 0 ? (total_pnl / total_stake) * 100 : 0.0
            wr = (sum(active.is_winner) / vol) * 100
            
            # Store result
            results_storage[i] = (
                alpha_bias = ab, 
                alpha_temp = at, 
                init_bias  = ib, 
                init_temp  = it, 
                pnl        = total_pnl, 
                roi        = roi, 
                volume     = vol, 
                win_rate   = wr
            )
        else
            results_storage[i] = nothing
        end
        
        # Safe Progress Update
        lock(progress_lock) do 
            next!(p)
        end
    end
    
    # 4. Convert to DataFrame
    valid_results = filter(!isnothing, results_storage)
    
    if isempty(valid_results)
        println("Warning: No parameter combinations met the minimum volume threshold.")
        return DataFrame()
    end

    df = DataFrame(valid_results)
    
    # Sort by PnL (Highest first)
    sort!(df, :pnl, rev=true)
    
    return df
end


# Run the optimizer
opt_results = optimize_adaptive_parameters_threaded(
    ds, exp_res, market_data;
    alpha_bias_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2], 
    alpha_temp_range = [0.01, 0.02, 0.05, 0.1],
    init_bias_range  = [0.0, 0.05, 0.075],
    init_temp_range  = [0.1, 0.2, 0.4, 0.6]
)

# View the top 5 parameter combinations
first(opt_results, 5)
