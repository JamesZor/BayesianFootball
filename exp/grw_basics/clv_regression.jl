
using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/grw_basics"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/grw_basics"; data_dir="./data")
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


# backtesting deconstructed 
exp_1 = loaded_results[1]

market_data = Data.prepare_market_data(ds)

latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_1)

ppd = BayesianFootball.Predictions.model_inference(latents)

using DataFrames, Statistics, GLM

model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)


analysis_df = innerjoin(
    market_data.df,
    model_features,
    on = [:match_id, :market_name, :market_line, :selection]
)

dropmissing!(analysis_df, [:odds_close, :is_winner])

#= 
julia> names(analysis_df)
21-element Vector{String}:
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
 "prob_model"

=#

analysis_df.spread = analysis_df.prob_model .- analysis_df.prob_implied_close
analysis_df.spread_fair = analysis_df.prob_model .- analysis_df.prob_fair_close
analysis_df.Y = Float64.(analysis_df.is_winner)

reg_model = glm(@formula(Y ~ prob_implied_close + spread), analysis_df, Binomial(), LogitLink())
reg_model = glm(@formula(Y ~ prob_fair_close + spread_fair), analysis_df, Binomial(), LogitLink())


using Plots

# Group your bets into "Edge Buckets" (e.g., 0-2%, 2-5%, 5%+)
analysis_df.edge_bucket = round.(analysis_df.spread, digits=2)

# Calculate actual win rate vs implied probability per bucket
# (You can use DataFrames aggregation for this)
grouped = combine(groupby(df_overs_early, :edge_bucket), 
    :Y => mean => :actual_win_rate,
    :prob_implied_close => mean => :market_implied,
    nrow => :count
)

# Filter for buckets with enough sample size
filter!(r -> r.count > 10, grouped)

scatter(grouped.edge_bucket, grouped.actual_win_rate .- grouped.market_implied,
    title="Realized Alpha vs Predicted Edge",
    xlabel="Your Predicted Edge (Spread)",
    ylabel="Actual Excess Return (Realized - Implied)",
    legend=false,
    markersize=sqrt.(grouped.count)./2 # Size bubbles by sample count
)

# If the dots slope UP to the right, you have Alpha.

# 1. Define the winning markets (The "Overs" Portfolio)
# Adjust these strings to match your exact selection names in the dataframe
target_markets = ["OverUnder"] # Filter by market name first
target_selections = [:over_15, :over_25, :over_35] # The specific lines

# 2. Create the "Winning Portfolio" DataFrame
df_overs = filter(row -> 
    row.market_name in target_markets && 
    row.selection in target_selections, 
    analysis_df
)

# 3. Create the "Losing Portfolio" DataFrame (Unders + 1X2 + BTTS)
df_rest = filter(row -> 
    !(row.selection in target_selections), # Everything NOT in the list above
    analysis_df
)

# 4. Run the Regressions Side-by-Side

println("--- REGRESSION: OVERS STRATEGY ---")
model_overs = glm(@formula(Y ~ prob_implied_close + spread), df_overs, Binomial(), LogitLink())

println("\n--- REGRESSION: THE REST (Unders/1x2) ---")
model_rest = glm(@formula(Y ~ prob_implied_close + spread), df_rest, Binomial(), LogitLink())

using Dates
#
# Filter for the "Good Times" (Before the crash)
df_overs_early = filter(r -> r.date < Date("2024-01-01"), df_overs)

# Filter for the "Crash" (2024 onwards)
df_overs_crash = filter(r -> r.date >= Date("2024-01-01"), df_overs)

println("--- OVERS: PRE-2024 ---")
glm(@formula(Y ~ prob_implied_close + spread), df_overs_early, Binomial(), LogitLink())

println("--- OVERS: POST-2024 ---")
glm(@formula(Y ~ prob_implied_close + spread), df_overs_crash, Binomial(), LogitLink())

using Plots

# 1. Define the "Clean" Portfolio
# Filter OUT Over 4.5/5.5/Unders (Toxic)
# Filter OUT Spread < 0.03 (Vig Trap)
clean_df = filter(row -> 
    (row.selection in [:over_15, :over_25, :over_35]) && 
    (row.spread > 0.03), 
    analysis_df
)

# 2. Recalculate Buckets
clean_df.edge_bucket = round.(clean_df.spread, digits=2)

grouped_clean = combine(groupby(clean_df, :edge_bucket), 
    :Y => mean => :actual_win_rate,
    :prob_implied_close => mean => :market_implied,
    nrow => :count
)

# 3. Plot the Difference
scatter(grouped_clean.edge_bucket, grouped_clean.actual_win_rate .- grouped_clean.market_implied,
    title="Purified Alpha: The 'Goldilocks' Zone",
    xlabel="Predicted Edge",
    ylabel="Excess Return",
    legend=false,
    markersize=sqrt.(grouped_clean.count)./2,
    color=:green
)
# Goal: A straight line sloping UP.



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Automating the Mincer-Zarnowitz (Encompassing) Test across all models will give you a "League Table of Alpha"—showing exactly which models have intrinsic skill and which are just noise.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



using DataFrames, Statistics, GLM, Printf

"""
    benchmark_models(experiments, dataset, market_data)

Iterates through a vector of experiment results, extracts predictions,
and performs the Encompassing Test (Mincer-Zarnowitz) regression.

Returns a DataFrame with the Alpha (Spread) coefficients for comparison.
"""
function benchmark_models(experiments::Vector, ds, market_data; target_selections=[:over_15])
    # Initialize the results container
    results_df = DataFrame(
        Model_Name = String[],
        Subset = String[],          # "Overall" vs "Overs_Selected"
        N_Bets = Int[],
        Intercept_Vig = Float64[],  # The hurdle rate
        Coef_Market = Float64[],    # Should be ~1.0 if market is efficient
        Coef_Alpha = Float64[],     # The "Spread" coefficient (YOUR EDGE)
        P_Value = Float64[],        # Is the edge real?
        Significant = Bool[]        # p < 0.05
    )

    # Define the target selection for the "Selected" subset
    target_markets = ["OverUnder"]

    for (i, exp) in enumerate(experiments)
        model_name = exp.config.name
        println("Processing Model $i: $model_name ...")

        # --- 1. REPLICATE THE ETL PIPELINE ---
        # Extract OOS Latents
        latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp)
        
        # Run Inference (This might take time depending on sample count)
        ppd = BayesianFootball.Predictions.model_inference(latents)
        
        # Flatten PPD to Mean Probability
        model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
        select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)

        # Join with Market Data
        analysis_df = innerjoin(
            market_data.df,
            model_features,
            on = [:match_id, :market_name, :market_line, :selection]
        )
        
        # Clean Data
        dropmissing!(analysis_df, [:odds_close, :is_winner])
        
        # Calculate Regression Features
        analysis_df.P_market = 1.0 ./ analysis_df.odds_close
        analysis_df.spread = analysis_df.prob_model .- analysis_df.P_market
        analysis_df.Y = Float64.(analysis_df.is_winner)

        # --- 2. RUN REGRESSION: OVERALL ---
        # (Includes 1X2, BTTS, Unders - The "Toxic" mix)
        try
            stats_overall = run_glm(analysis_df)
            push!(results_df, (model_name, "Overall", stats_overall...))
        catch e
            println("  -> Failed Overall Regression for $model_name")
        end

        # --- 3. RUN REGRESSION: SELECTED OVERS ---
        # (The "Goldilocks" Zone)
        mask = (analysis_df.market_name .== "OverUnder") .& 
               (in.(analysis_df.selection, Ref(target_selections)))
        
        df_selected = analysis_df[mask, :]
        
        if nrow(df_selected) > 0
            try
                stats_selected = run_glm(df_selected)
                push!(results_df, (model_name, "Overs_Selected", stats_selected...))
            catch e
                println("  -> Failed Selected Regression for $model_name")
            end
        else
            println("  -> No data for Selected Overs")
        end
    end

    # Sort by Alpha Coefficient (Descending) to see the best model on top
    sort!(results_df, [:Subset, :Coef_Alpha], rev=true)
    
    return results_df
end

"""
Helper to fit the GLM and extract the specific Mincer-Zarnowitz params
"""
function run_glm(df)
    # Fit: Y ~ Market_Prob + Spread
    model = glm(@formula(Y ~ P_market + spread), df, Binomial(), LogitLink())
    
    # Extract the CoefTable object
    ct = coeftable(model)
    
    # 1. Find the index for the "spread" row safely
    # ct.rownms is a vector of strings like ["(Intercept)", "P_market", "spread"]
    spread_idx = findfirst(==("spread"), ct.rownms)
    
    if isnothing(spread_idx)
        # Fallback if spread gets dropped (e.g. perfect collinearity)
        return (nrow(df), NaN, NaN, NaN, NaN, false)
    end

    # 2. Extract Coefficients
    # coef(model) returns the vector of estimates. 
    # We can use our index to find the specific ones.
    # Note: We need to find the indices for Intercept/P_market too if we want them specifically,
    # or just assume standard ordering. Let's trust the model object for raw values.
    val_int = coef(model)[1]
    val_mkt = coef(model)[2]
    val_spr = coef(model)[spread_idx] # Use the specific index for spread
    
    # 3. Extract P-Value
    # ct.cols is a Vector of Vectors. ct.pvalcol is the index of the p-value column.
    p_values = ct.cols[ct.pvalcol]
    p_val_spr = p_values[spread_idx]
    
    return (nrow(df), val_int, val_mkt, val_spr, p_val_spr, p_val_spr < 0.05)
end

# 1. Prepare Market Data (if not already done)
market_data = Data.prepare_market_data(ds)

# 2. Run the Benchmark
alpha_table = benchmark_models(loaded_results, ds, market_data)
alpha_table = benchmark_models(loaded_results, ds, market_data; target_selections =[:over_15])
alpha_table = benchmark_models(loaded_results, ds, market_data; target_selections =[:over_25])
alpha_table = benchmark_models(loaded_results, ds, market_data; target_selections =[:over_35])

# 3. View the Leaderboard
display(alpha_table)



###
using Distributions, Random, Plots, StatsBase, DataFrames, Statistics
using BayesianFootball # Assuming this is your package name

"""
    run_residual_diagnostics(experiments, ds, model_name_pattern="grw_neg_bin_v2")

Performs a full forensic audit on the residuals of a specific model.
Generates Q-Q Plots, Time Series Drift, and Autocorrelation charts.
"""
function run_residual_diagnostics(experiments, ds, model_name_pattern="grw_neg_bin_v2")
    
    # 1. Select the Target Model
    # Find the experiment that matches your "Stiff" model name
    exp_idx = findfirst(e -> occursin(model_name_pattern, e.config.name), experiments)
    if isnothing(exp_idx)
        error("Model matching '$model_name_pattern' not found!")
    end
    exp_target = experiments[exp_idx]
    println("--- Analyzing Residuals for: $(exp_target.config.name) ---")

    # 2. Extract Out-of-Sample Predictions (Latents)
    # This gives us the predicted Mu (Mean) and Phi (Shape) for every match
    println("Extracting OOS predictions...")
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_target)
    
    # 3. Get Actual Goal Counts
    # We need to map predictions back to the ground truth in the dataset 'ds'
    # Assuming 'ds' has a dataframe or we can lookup matches by ID.
    # Let's assume latents has 'match_id' and we can join with market_data or ds.df
    market_data = Data.prepare_market_data(ds) # Helper from your previous snippets
    
    # Join Latents (Preds) with Realities (Goals)
    # We need a DataFrame that has: match_id, date, home_goals, away_goals, pred_home_dist, pred_away_dist
    
    # A. Extract predicted parameters (Mu, Phi)
    # Note: Adjust column names if your 'latents' structure is different.
    # Usually latents will have: home_mu, away_mu, home_phi, away_phi (or shared phi)
    
    # Let's iterate and build the residual vectors
    home_residuals = Float64[]
    away_residuals = Float64[]
    dates = Date[]
    
    println("Computing Dunn-Smyth Residuals...")
    
    # Map match_ids to actual results for fast lookup
    # Creating a Dict: match_id -> (home_goals, away_goals, date)
    actuals_map = Dict(
        r.match_id => (r.home_score, r.away_score, r.match_date) 
        for r in eachrow(ds.matches) # Assuming ds.df holds the raw match data
    )

    for i in 1:nrow(latents.df)
        row = latents.df[i, :]
        mid = row.match_id
        
        if !haskey(actuals_map, mid) continue end
        (h_score, a_score, match_date) = actuals_map[mid]
# The 'vec' ensures matrices become vectors, and 'mean' collapses them to a scalar.
        # This gives us the Point Estimate for the match.
        h_mu = mean(vec(row.λ_h))
        h_phi = mean(vec(row.r)) # Dispersion
        
        a_mu = mean(vec(row.λ_a))
        a_phi = mean(vec(row.r))
        
        # --- Parameter Conversion ---
        # Convert Mean(μ) + Dispersion(ϕ) to Success(p) + Failures(r)
        # Formula: p = ϕ / (ϕ + μ), r = ϕ
        
        # Home Dist
        h_p = h_phi / (h_phi + h_mu)
        # Ensure 0 < p < 1 to prevent errors
        h_p = clamp(h_p, 1e-6, 1.0 - 1e-6) 
        dist_home = NegativeBinomial(h_phi, h_p)
        
        # Away Dist
        a_p = a_phi / (a_phi + a_mu)
        a_p = clamp(a_p, 1e-6, 1.0 - 1e-6)
        dist_away = NegativeBinomial(a_phi, a_p)
        
        # --- Calculate RQR ---
        push!(home_residuals, calculate_rqr(dist_home, h_score))
        push!(away_residuals, calculate_rqr(dist_away, a_score))
        push!(dates, match_date)


    end
    
    # 4. Generate Diagnostic Plots
    generate_plots(home_residuals, dates, "Home Goals")
    generate_plots(away_residuals, dates, "Away Goals")
    
    return (home_residuals, away_residuals, dates)
end

"""
Helper: Calculate single Dunn-Smyth Residual
"""
function calculate_rqr(dist, y_obs)
    # CDF at y (Upper bound)
    p_high = cdf(dist, y_obs)
    
    # CDF at y-1 (Lower bound)
    p_low = cdf(dist, y_obs - 1)
    
    # Randomize uniform between bounds
    u = rand(Uniform(p_low, p_high))
    
    # Clamp for numerical safety
    u = clamp(u, 1e-9, 1.0-1e-9)
    
    # Inverse Normal Transform
    return quantile(Normal(0,1), u)
end

"""
Helper: Plotting Suite
"""
function generate_plots(residuals, dates, label)
    # Sort by date for the time series
    perm = sortperm(dates)
    sorted_res = residuals[perm]
    sorted_dates = dates[perm]
    
    # 1. Q-Q Plot (Normality/Tail Check)
    p1 = qqplot(Normal(0,1), sorted_res, 
        title="$label: Q-Q Plot", xlabel="Theoretical", ylabel="Observed",
        legend=false, color=:blue, markerstrokewidth=0, markersize=3)
    
    # 2. Residuals vs Time (Regime Check)
    # Calculate Rolling Mean (Window = 50 games)
    roll_mean = [mean(sorted_res[max(1, i-30):i]) for i in 1:length(sorted_res)]
    
    p2 = scatter(sorted_dates, sorted_res, 
        title="$label: Regime Stability", alpha=0.2, color=:black, label="", markersize=2)
    plot!(p2, sorted_dates, roll_mean, 
        color=:red, linewidth=2, label="Rolling Mean (Bias)")
    hline!(p2, [0.0], color=:blue, linestyle=:dash, label="")
    
    # 3. Rolling Variance (The Volatility Check)
    # This detects if your 'Phi' is failing (e.g., Variance explodes in 2024)
    roll_var = [var(sorted_res[max(1, i-30):i]) for i in 1:length(sorted_res)]
    
    p3 = plot(sorted_dates, roll_var, 
        title="$label: Rolling Variance (Target = 1.0)", 
        color=:purple, linewidth=2, legend=false)
    hline!(p3, [1.0], color=:black, linestyle=:dash)
    
    # Combine
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000))
    display(final_plot)
end

using Dates
(home_res, away_res, dates) = run_residual_diagnostics(loaded_results, ds, "grw_neg_bin_v2")

(home_res, away_res, dates) = run_residual_diagnostics(loaded_results, ds, "grw_neg_bin_phi")

using Plots, StatsPlots

# 1. Re-generate the plot object using the data you just calculated
# (We sort by date first to make the Time Series look right)
perm = sortperm(dates)
sorted_res = home_res[perm]
sorted_dates = dates[perm]

# A. Q-Q Plot
p1 = qqplot(Normal(0,1), sorted_res, 
    title="Home Goals: Q-Q Plot", xlabel="Theoretical", ylabel="Observed",
    legend=false, color=:blue, markerstrokewidth=0, markersize=3)

# B. Regime Stability (Rolling Mean)
roll_mean = [mean(sorted_res[max(1, i-50):i]) for i in 1:length(sorted_res)]
p2 = scatter(sorted_dates, sorted_res, 
    title="Regime Stability (Bias)", alpha=0.2, color=:black, label="", markersize=2, markerstrokewidth=0)
plot!(p2, sorted_dates, roll_mean, color=:red, linewidth=2, label="Rolling Mean")
hline!(p2, [0.0], color=:blue, linestyle=:dash)

# C. Rolling Variance (The Crash Detector)
roll_var = [var(sorted_res[max(1, i-50):i]) for i in 1:length(sorted_res)]
p3 = plot(sorted_dates, roll_var, 
    title="Rolling Variance (Target = 1.0)", 
    color=:purple, linewidth=2, legend=false)
hline!(p3, [1.0], color=:black, linestyle=:dash)

# 2. Combine and Save
final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000))

# Save to your current directory
savefig(final_plot, "rqr_diagnosis_phi.png")
println("Saved plot to: $(pwd())/rqr_diagnosis.png")



function compare_stability(dates, res_model_A, label_A, res_model_B, label_B; window=30)
    # Helper to calculate the metrics
    function calc_metrics(dates, residuals)
        # Sort by date first
        perm = sortperm(dates)
        sorted_res = residuals[perm]
        
        # Calculate Rolling Series
        n = length(sorted_res)
        roll_mean = [mean(sorted_res[max(1, i-window):i]) for i in 1:n]
        roll_var  = [var(sorted_res[max(1, i-window):i]) for i in 1:n]
        
        # 1. Bias Score (MSE of Mean from 0.0)
        # Penalizes periods where the model systematically over/under predicts
        bias_score = mean((roll_mean .- 0.0).^2)
        
        # 2. Variance Stability Score (MSE of Variance from 1.0)
        # Penalizes "Swinging" and "Fat Tails"
        stability_score = mean((roll_var .- 1.0).^2)
        
        # 3. "Wiggle" (Standard Deviation of the Rolling Variance)
        # Pure measure of how jerky the line is, regardless of where it is
        wiggle = std(roll_var)
        
        return (bias_score, stability_score, wiggle)
    end

    # Calculate for both
    (b_A, s_A, w_A) = calc_metrics(dates, res_model_A)
    (b_B, s_B, w_B) = calc_metrics(dates, res_model_B)

    # Print Report
    println("-"^60)
    println("STABILITY SHOWDOWN: $label_A vs $label_B")
    println("-"^60)
    
    println("1. BIAS SCORE (Target = 0.0) | Lower is Better")
    println("   $label_A: $(round(b_A, digits=4))")
    println("   $label_B: $(round(b_B, digits=4))")
    println("   Winner: $(b_A < b_B ? label_A : label_B)")
    println("")
    
    println("2. VARIANCE STABILITY (Target = 1.0) | Lower is Better")
    println("   $label_A: $(round(s_A, digits=4))")
    println("   $label_B: $(round(s_B, digits=4))")
    println("   Winner: $(s_A < s_B ? label_A : label_B)")
    println("")
    
    println("3. WIGGLINESS (Vol of Vol) | Lower is Smoother")
    println("   $label_A: $(round(w_A, digits=4))")
    println("   $label_B: $(round(w_B, digits=4))")
    println("   Winner: $(w_A < w_B ? label_A : label_B)")
    println("-"^60)
end
