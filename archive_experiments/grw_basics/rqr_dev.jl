
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


using Distributions, Random, Plots, StatsPlots, DataFrames, Statistics, Dates

# ==============================================================================
# 1. CORE MATH: Bayesian RQR Calculation
# ==============================================================================

"""
    calculate_bayesian_rqr(λ_samples, r_samples, y_obs)

Calculates the Randomized Quantile Residual (RQR) for a single observation
using the full Posterior Predictive Distribution (PPD).

1. Computes P(Y <= y_obs) and P(Y <= y_obs - 1) for EVERY parameter sample.
2. Averages these probabilities to marginalize out parameter uncertainty.
3. Samples a uniform random value u between the averaged bounds.
4. Transforms u to a Standard Normal quantile.
"""
function calculate_bayesian_rqr(λ_samples::Vector{Float64}, r_samples::Vector{Float64}, y_obs::Int)
    # Number of MCMC samples (usually 2000-4000)
    N = length(λ_samples)
    
    # Pre-allocate arrays for CDF values
    p_low_accum = 0.0
    p_high_accum = 0.0

    # Iterate through every MCMC sample (Bayesian Integration)
    @inbounds for i in 1:N
        λ = λ_samples[i]
        r = r_samples[i]

        # Convert Mean(λ) + Shape(r) to Julia's NegativeBinomial(r, p)
        # p = r / (r + λ)  -> Probability of success
        p = r / (r + λ)
        
        # Clamp for numerical stability
        p = clamp(p, 1e-8, 1.0 - 1e-8)
        
        dist = NegativeBinomial(r, p)

        # Accumulate CDF values
        # F(y-1): Probability mass strictly less than y_obs
        if y_obs > 0
            p_low_accum += cdf(dist, y_obs - 1)
        end
        
        # F(y): Probability mass less than or equal to y_obs
        p_high_accum += cdf(dist, y_obs)
    end

    # Average to get the Marginal Posterior Predictive probabilities
    p_low_bayes = p_low_accum / N
    p_high_bayes = p_high_accum / N

    # --- The "Randomized" Part (Feng et al.) ---
    # We essentially smooth the step function of the discrete count.
    # We pick a random spot between the lower step and upper step.
    
    # Safety clamp to prevent log(0) in quantile function
    p_low_bayes = clamp(p_low_bayes, 1e-9, 1.0 - 1e-9)
    p_high_bayes = clamp(p_high_bayes, 1e-9, 1.0 - 1e-9)
    
    # If the model is extremely confident y_obs is impossible, bounds might cross 
    # due to float precision, so we sort them.
    u_min, u_max = minmax(p_low_bayes, p_high_bayes)
    
    # Draw uniform random number
    u = rand(Uniform(u_min, u_max))
    
    # Inverse Standard Normal CDF
    return quantile(Normal(0, 1), u)
end

# ==============================================================================
# 2. ORCHESTRATION: Running the Diagnostics
# ==============================================================================

function run_residual_diagnostics(experiments, ds, model_name_pattern="grw_neg_bin")
    
    # --- A. Select Model ---
    exp_idx = findfirst(e -> occursin(model_name_pattern, e.config.name), experiments)
    if isnothing(exp_idx)
        error("Model matching '$model_name_pattern' not found!")
    end
    exp_target = experiments[exp_idx]
    println(">>> Analyzing Model: $(exp_target.config.name)")

    # --- B. Extract Predictions ---
    println(">>> Extracting OOS predictions (this may take a moment)...")
    # Assuming this function returns a DataFrame with columns: match_id, λ_h (vector), λ_a (vector), r (vector)
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_target)

    # --- C. Map Actuals ---
    println(">>> Mapping Actual Outcomes...")
    # Create a fast lookup: match_id -> (home_score, away_score, date)
    actuals_map = Dict{Int, Tuple{Int, Int, Date}}()
    for row in eachrow(ds.matches)
        actuals_map[row.match_id] = (row.home_score, row.away_score, row.match_date)
    end

    # --- D. Compute Residuals ---
    home_residuals = Float64[]
    away_residuals = Float64[]
    dates = Date[]

    println(">>> Computing Bayesian RQRs for $(nrow(latents.df)) matches...")

    for row in eachrow(latents.df)
        mid = row.match_id
        
        # Skip if match not found in dataset
        if !haskey(actuals_map, mid) continue end
        
        (h_score, a_score, match_date) = actuals_map[mid]

        # Extract vectors of samples (Bayesian Posterior)
        # Note: If 'r' is shared, it might be a vector. If distinct per team, adjust accordingly.
        # Assuming `r` column contains vectors of shape parameter samples.
        λ_h_s = vec(row.λ_h)
        λ_a_s = vec(row.λ_a)
        r_s   = vec(row.r) 

        # Calculate Residuals
        rqr_h = calculate_bayesian_rqr(λ_h_s, r_s, h_score)
        rqr_a = calculate_bayesian_rqr(λ_a_s, r_s, a_score)

        push!(home_residuals, rqr_h)
        push!(away_residuals, rqr_a)
        push!(dates, match_date)
    end

    # --- E. Plotting ---
    println(">>> Generating Diagnostic Plots...")
    generate_plots(home_residuals, dates, "Home Goals ($(exp_target.config.name))")
    generate_plots(away_residuals, dates, "Away Goals ($(exp_target.config.name))")

    return (home_residuals, away_residuals, dates)
end

# ==============================================================================
# 3. VISUALIZATION: EWMA & QQ Plots
# ==============================================================================

function ewma(series::Vector{Float64}; span::Float64=10.0)
    alpha = 2.0 / (span + 1.0)
    output = similar(series)
    output[1] = series[1]
    for i in 2:length(series)
        output[i] = alpha * series[i] + (1 - alpha) * output[i-1]
    end
    return output
end

function ewm_var(series::Vector{Float64}; span::Float64=10.0)
    alpha = 2.0 / (span + 1.0)
    mus = ewma(series, span=span)
    vars = Float64[]
    current_var = 1.0 
    for i in 1:length(series)
        sq_dev = (series[i] - mus[i])^2
        current_var = alpha * sq_dev + (1 - alpha) * current_var
        push!(vars, current_var)
    end
    return vars
end

function generate_plots(residuals, dates, label)
    # Sort chronologically
    perm = sortperm(dates)
    sorted_res = residuals[perm]
    sorted_dates = dates[perm]

    # 1. Q-Q Plot (Normality Check)
    # If points curve off the line, your tails are too fat/thin compared to Normal
    p1 = qqplot(Normal(0,1), sorted_res, 
        title="$label: Q-Q Plot", xlabel="Theoretical Quantiles", ylabel="RQR",
        legend=false, color=:blue, markerstrokewidth=0, markersize=3, alpha=0.5)
    plot!(p1, [-3, 3], [-3, 3], color=:red, linestyle=:dash) # 1:1 line reference

    # 2. Drift / Bias Check (EWMA)
    # If the red line drifts far from 0, the model is systematically over/under estimating
    drift = ewma(sorted_res, span=20.0)
    p2 = scatter(sorted_dates, sorted_res, 
        title="Regime Stability (Bias)", alpha=0.15, color=:black, label="", markersize=2, ylabel="Residual")
    plot!(p2, sorted_dates, drift, 
        color=:red, linewidth=2, label="EWMA Drift (Span=50)")
    hline!(p2, [0.0], color=:blue, linestyle=:dash)

    # 3. Volatility Check
    # If this is > 1.0, your model is "Overconfident" (data is wilder than model expects)
    # If this is < 1.0, your model is "Underconfident" (data is boring, model expects chaos)
    vol = ewm_var(sorted_res, span=20.0)
    p3 = plot(sorted_dates, vol, 
        title="Variance Stability (Target = 1.0)", 
        color=:purple, linewidth=2, legend=false, ylabel="Rolling Var")
    hline!(p3, [1.0], color=:black, linestyle=:dash)

    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000), margin=5Plots.mm)
    display(final_plot)
end

#= 


 Experiments in: ./data/exp/grw_basics
==============================================================================================================
IDX  | NAME                      | MODEL                | SPLITTER           | SAMPLER         | PATH ID
--------------------------------------------------------------------------------------------------------------
[1]  | grw_poisson_v2            | GRWPoisson           | CVConfig           | NUTSConfig      | grw_poisson_v2_20260205_223951
[2]  | grw_neg_bin_mu            | GRWNegativeBinomia.. | CVConfig           | NUTSConfig      | grw_neg_bin_mu_20260205_114249
[3]  | grw_neg_bin_phi           | GRWNegativeBinomia.. | CVConfig           | NUTSConfig      | grw_neg_bin_phi_20260203_133849
[4]  | grw_neg_bin_v2            | GRWNegativeBinomia.. | CVConfig           | NUTSConfig      | grw_neg_bin_v2_20260123_123116
[5]  | grw_bivariate_poisson     | GRWBivariatePoisso.. | CVConfig           | NUTSConfig      | grw_bivariate_poisson_20260122_010953
[6]  | grw_neg_bin               | GRWNegativeBinomia.. | CVConfig           | NUTSConfig      | grw_neg_bin_20260121_163240
[7]  | grw_dixon_coles           | GRWDixonColes        | CVConfig           | NUTSConfig      | grw_dixon_coles_20260121_080400
[8]  | grw_poisson               | GRWPoisson           | CVConfig           | NUTSConfig      | grw_poisson_20260120_193947
==============================================================================================================

=#


# Run the diagnostics
(home_res, away_res, dates) = run_residual_diagnostics(loaded_results, ds, "grw_neg_bin_v2")
(home_res_mu, away_res_mu, dates) = run_residual_diagnostics(loaded_results, ds, "grw_neg_bin_mu")
(home_res_phi, away_res_phi, dates) = run_residual_diagnostics(loaded_results, ds, "grw_neg_bin_phi")

# Check statistics (optional)
using Statistics
println("Home Residual Mean: ", mean(home_res)) # Should be close to 0
println("Home Residual Std:  ", std(home_res))  # Should be close to 1

println("Home Residual Mean: ", mean(home_res_mu)) # Should be close to 0
println("Home Residual Std:  ", std(home_res_mu))  # Should be close to 1

println("Home Residual Mean: ", mean(home_res_phi)) # Should be close to 0
println("Home Residual Std:  ", std(home_res_phi))  # Should be close to 1



a = generate_plots(home_res, dates, "neg_bin")
savefig("home_goals_diagnostics.png")

a = generate_plots(home_res_mu, dates, "neg_bin")
savefig("home_goals_diagnostics_mu.png")

a = generate_plots(home_res_phi, dates, "neg_bin_phi")
savefig(a, "home_goals_diagnostics_phi.png")



using Distributions, Statistics, Dates, DataFrames

"""
    apply_calibration_robust(ds, experiments, s_curve, dates_cal)

Applies the calibration scalar S(t) to the dispersion parameter r (Phi),
with safety clamps to prevent NaN/DomainErrors.
"""
function apply_calibration_robust(ds, experiments, s_curve, dates_cal)
    # 1. Sanitize the S-Curve (Fix Infs/NaNs)
    # If std was 0, S becomes Inf. We replace Inf with a cap (e.g., 5.0x multiplier)
    # If S is NaN, we fallback to 1.0 (no change)
    clean_s = replace(s_curve, Inf => 5.0, NaN => 1.0)
    
    # Extract raw predictions
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, experiments)
    
    calibrated_residuals = Float64[]
    calibrated_dates = Date[]
    
    # Create lookup
    date_to_S = Dict(d => s for (d,s) in zip(dates_cal, clean_s))
    global_S = mean(filter(isfinite, clean_s)) # Safe fallback
    
    actuals_map = Dict(r.match_id => (r.home_score, r.match_date) for r in eachrow(ds.matches))

    for row in eachrow(latents.df)
        mid = row.match_id
        if !haskey(actuals_map, mid) continue end
        (h_score, match_date) = actuals_map[mid]
        
        # 1. Get raw parameters
        # Handle cases where 'r' might be a vector (from MCMC) or scalar
        raw_r = mean(vec(row.r)) 
        raw_lam = mean(vec(row.λ_h))
        
        # 2. Get S(t)
        S_t = get(date_to_S, match_date, global_S)
        
        # 3. Apply Correction
        calibrated_r = raw_r * S_t 
        
        # --- SAFETY CHECKS ---
        # 4a. Handle Infinite r (Poisson limit)
        if calibrated_r > 1e4 
             # If r is huge, it converges to Poisson(λ)
             # We approximate by capping r at a large number
             calibrated_r = 1e4
        end
        
        # 4b. Handle Tiny r (Geometric limit)
        if calibrated_r < 1e-4
            calibrated_r = 1e-4
        end
        
        # 5. Recalculate p with clamps
        # p = r / (r + λ)
        denom = calibrated_r + raw_lam
        if denom ≈ 0.0
            p_val = 1.0
        else
            p_val = calibrated_r / denom
        end
        
        # Force p into valid range (0, 1]
        p_val = clamp(p_val, 1e-6, 1.0 - 1e-6)
        
        # 6. Construct Distribution
        dist = NegativeBinomial(calibrated_r, p_val)
        
        # 7. Calculate RQR (Simple Point Estimate Version for Speed)
        # (You can swap this for your full Bayesian version if preferred)
        u = rand(Uniform(cdf(dist, h_score - 1), cdf(dist, h_score)))
        u = clamp(u, 1e-9, 1.0 - 1e-9)
        val = quantile(Normal(0,1), u)
        
        push!(calibrated_residuals, val)
        push!(calibrated_dates, match_date)
    end
    
    return calibrated_residuals, calibrated_dates
end



using Statistics, RollingFunctions, Plots, DataFrames

"""
    calibrate_seasonality(residuals, dates; window_size=30)

1. Calculates the rolling Std Dev of the residuals.
2. Derives a time-dependent scalar S(t) to force Std -> 1.0.
3. Returns a 'Calibration Curve' you can apply to future predictions.
"""
function build_calibration_layer(residuals, dates; window_size=30)
    # 1. Sort by date
    perm = sortperm(dates)
    s_res = residuals[perm]
    s_dates = dates[perm]
    
    # 2. Compute Rolling Std Dev (The "Error" in your confidence)
    # We use a centered window for historical analysis
    roll_std = runstd(s_res, window_size)
    
    # Handle edge cases (fill NaNs with global std)
    replace!(roll_std, NaN => std(s_res))
    
    # 3. Calculate Correction Factor S(t)
    # If roll_std < 1.0 (Under-confident), we need to INCREASE Phi (tighten variance).
    # Heuristic: S ≈ 1 / (std^2) works well for Variance scaling
    correction_curve = 1.0 ./ (roll_std .^ 2)
    
    # 4. Smooth the curve (Safety layer)
    # We don't want to react to single-week noise, so we smooth the correction
    smooth_correction = runmean(correction_curve, window_size)
    
    return s_dates, smooth_correction
end

# --- Execution ---

# 1. Build the Curve from your "Mu" model residuals
dates_cal, s_curve = build_calibration_layer(home_res_mu, dates, window_size=40)

# 2. Visualize the "Seasonality of Chaos"
p_cal = plot(dates_cal, s_curve, 
    title="The Calibration Layer: S(t)",
    ylabel="Phi Multiplier (S)",
    label="Correction Factor",
    color=:green, lw=3,
    legend=:topleft
)
hline!(p_cal, [1.0], label="Perfect Calibration", color=:black, linestyle=:dash)

# Interpretation: 
# If the line is ABOVE 1.0, your model was "Panicking" (Phi too low), so we boost Phi.
# If the line is BELOW 1.0, your model was "Overconfident" (Phi too high), so we shrink Phi.
display(p_cal)
savefig("calibration_layer_curve.png")

# --- 3. Apply the Layer (The Fix) ---

# Run the fix
cal_res, cal_dates = apply_calibration_robust(ds, loaded_results[2], s_curve, dates_cal)

# Verify
println("Original Std:   ", std(home_res_mu))
println("Calibrated Std: ", std(cal_res)) # Should be very close to 1.0


# Generate diagnostics for the Calibrated Model
a = generate_plots(cal_res, cal_dates, "Calibrated Mu Model")
savefig("calibrated_diagnostics.png")
println("Saved final diagnostics to calibrated_diagnostics.png")



##
using Statistics, RollingFunctions, DataFrames

function optimize_window_size(residuals, dates; test_windows=[10, 20, 30, 40, 50, 75, 100])
    results = DataFrame(Window=[], Global_Std=[], Variance_Stability_RMSE=[])

    println("Testing Window Sizes...")
    
    for w in test_windows
        # 1. Build Layer with window 'w'
        dates_cal, s_curve = build_calibration_layer(residuals, dates, window_size=w)
        
        # 2. Apply it (Simulated application)
        # We don't need to re-run the full loop; we can approximate the effect
        # Calibrated_Res ≈ Raw_Res / S(t)^0.5  (Roughly speaking)
        # But let's just use the 'apply_calibration_robust' function you already have
        # to be precise.
        
        # Note: This requires passing your full 'ds' and 'loaded_results' objects
        # For speed, we will just use the S-curve to check the 'Fit' to the residuals
        # The S-curve essentially tries to make the local std dev 1.0.
        
        # Metric 1: Global Calibration (Should be 1.0)
        # The S-curve is derived from history, so applied back to history implies:
        # Calibrated_Res[t] = Res[t] * sqrt(S[t]) (Approximation for NegBin)
        
        # Let's run the real function to be safe:
        cal_res, _ = apply_calibration_robust(ds, loaded_results[2], s_curve, dates_cal)
        
        # Metric: How far does the rolling variance deviate from 1.0?
        # We want the purple plot to be FLAT at 1.0.
        rolling_vars = runvar(cal_res, w)
        rmse_stability = sqrt(mean((rolling_vars .- 1.0).^2))
        
        push!(results, (w, std(cal_res), rmse_stability))
        println("Window $w: Std=$(round(std(cal_res),digits=3)), Stability Error=$(round(rmse_stability,digits=3))")
    end
    
    sort!(results, :Variance_Stability_RMSE)
    return results
end

# Run the optimization
opt_results = optimize_window_size(home_res_mu, dates)
display(opt_results)


using StatsPlots

# 1. Create a DataFrame with Residuals and Game Weeks
df_resid = DataFrame(
    resid = home_res_mu, 
    week = ds.matches.match_week[1:length(home_res_mu)] # Ensure alignment!
)

# 2. Group by Week and calculate Variance
week_stats = combine(groupby(df_resid, :week), :resid => var => :week_var)
sort!(week_stats, :week)

# 3. Plot
plot(week_stats.week, week_stats.week_var, 
    title="Residual Variance by Game Week",
    xlabel="Game Week (1-38)", ylabel="Variance (Target=1.0)",
    label="Variance", lw=2, color=:blue, legend=false)
hline!([1.0], color=:red, linestyle=:dash)

savefig("game_week_phi.png")



### -- 
using Statistics, RollingFunctions, Dates, DataFrames, Distributions

# ==============================================================================
# 1. BUILD: The Learning Phase (Run once per week on historical data)
# ==============================================================================

"""
    train_calibration_layer(residuals, dates; optimal_window=40)

Learns the volatility schedule S(t) from historical residuals.
Uses the optimal window size found via optimization (N=40).
"""
function train_calibration_layer(residuals, dates; optimal_window=40)
    # Sort chronologically
    perm = sortperm(dates)
    s_res = residuals[perm]
    s_dates = dates[perm]
    
    # 1. Compute Rolling Std (The Error Signal)
    # We use runstd from RollingFunctions
    roll_std = runstd(s_res, optimal_window)
    
    # Fill startup NaNs with global std
    replace!(roll_std, NaN => std(s_res))
    
    # 2. Derive Correction Curve S(t)
    # Logic: If std < 1.0, we need to INCREASE Phi (tighten variance).
    # Formula: S = 1 / std^2
    correction_curve = 1.0 ./ (roll_std .^ 2)
    
    # 3. Smooth the Correction (Safety)
    # Apply a secondary smooth to prevent jagged jumps
    smooth_S = runmean(correction_curve, optimal_window)
    
    return s_dates, smooth_S
end

# ==============================================================================
# 2. APPLY: The Production Phase (Run for every prediction)
# ==============================================================================

"""
    apply_calibration_production(raw_r, raw_lambda, match_date, s_dates, s_curve)

Applies the learned calibration factor S(t) to a specific match prediction.
"""
function apply_calibration_production(raw_r, raw_lambda, match_date, s_dates, s_curve)
    # 1. Find the applicable S(t)
    # In production, we assume the 'Current Regime' is the one from the most recent historical match.
    # We essentially "Forward Fill" the last known S value.
    
    if match_date > last(s_dates)
        # Future game: Use the very latest learned factor
        S_t = last(s_curve)
    else
        # Backtest game: Find the specific historical factor
        # (Naive lookup for demo; use Dict/BinarySearch for speed if needed)
        idx = findfirst(d -> d >= match_date, s_dates)
        if isnothing(idx)
            S_t = 1.0 
        else
            S_t = s_curve[idx]
        end
    end
    
    # 2. Apply Correction
    calibrated_r = raw_r * S_t
    
    # 3. Safety Clamps (The Robust Fix)
    calibrated_r = clamp(calibrated_r, 1e-4, 1e4)
    
    # 4. Recalculate Probability p
    denom = calibrated_r + raw_lambda
    p_val = (denom ≈ 0.0) ? 1.0 : (calibrated_r / denom)
    p_val = clamp(p_val, 1e-6, 1.0 - 1e-6)
    
    return calibrated_r, p_val
end




# ------ 
using DataFrames, Distributions, Statistics, Dates

# ==============================================================================
# 1. SETUP: The Calibration Injector
# ==============================================================================

"""
    inject_calibration!(latents, ds, s_curve, dates_cal)

Modifies the latent predictions in-place by applying the S(t) volatility schedule.
"""
function inject_calibration!(latents, ds, s_curve, dates_cal)
    println(">>> Injecting Calibration Layer into Latents...")
    
    # 1. Prepare Lookup Table (Date -> S_t)
    # We map every match date to its corresponding S factor
    date_to_S = Dict(d => s for (d,s) in zip(dates_cal, s_curve))
    global_S = mean(s_curve)
    
    # Match ID to Date Map
    match_date_map = Dict(r.match_id => r.match_date for r in eachrow(ds.matches))

    # 2. Iterate and Modify
    # We modify the 'r' (dispersion) samples directly in the dataframe
    for row in eachrow(latents.df)
        mid = row.match_id
        if !haskey(match_date_map, mid) continue end
        
        m_date = match_date_map[mid]
        
        # Get S(t) - Forward fill logic if date is new
        S_t = get(date_to_S, m_date) do 
            # If date missing (e.g. future), use the latest available S
            m_date > last(dates_cal) ? last(s_curve) : global_S
        end
        
        # APPLY THE FIX: Multiply the vector of samples by S_t
        # We broadcast (*=) to modify the vector in place
        row.r .*= S_t
        
        # SAFETY CLAMP: Prevent DomainErrors in NegativeBinomial
        # We clamp the samples to avoid r being too small (Underflow) or too huge
        clamp!(row.r, 1e-4, 1e5)
    end
    
    println(">>> Calibration applied. Model is now tighter/looser based on season.")
    return latents
end 

#############
# ==============================================================================
# 2. EXECUTION: The Head-to-Head Backtest
# ==============================================================================

baker = BayesianKelly()

my_signals = [baker]

# --- A. Run the Standard (Raw) Backtest ---
println("\n=== 1. Running Standard Backtest (Raw Mu) ===")
# Assuming 'loaded_results[2]' is your Mu model

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results[[2,3]], 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)



# --- B. Run the Manual (Calibrated) Backtest ---
println("\n=== 2. Running Calibrated Backtest (Injection) ===")

# 1. Manually Extract Latents
# We copy the step from inside your _process_single_experiment
latents_cal = BayesianFootball.Experiments.extract_oos_predictions(ds, loaded_results[2])

# 2. INJECT CALIBRATION
# (Requires s_curve and dates_cal from your previous step)
inject_calibration!(latents_cal, ds, s_curve, dates_cal) 

# 3. Manually Run Inference (Probabilities)
# We use the modified latents here
market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
ppd_cal = BayesianFootball.Predictions.model_inference(latents_cal; market_config=market_config)

# 4. Manually Process Signals
# We need the market data again
market_data = BayesianFootball.Data.prepare_market_data(ds)
sig_result_cal = BayesianFootball.Signals.process_signals(ppd_cal, market_data.df, my_signals; odds_column=:odds_close)

# 5. Manually Enrich (PnL & Naming)
# We basically copy the enrichment logic from your worker function
df_cal = sig_result_cal.df

# Recalculate PnL
df_cal.pnl = map(eachrow(df_cal)) do r
    if ismissing(r.is_winner) || r.stake == 0.0
        0.0
    elseif r.is_winner
        r.stake * (r.odds - 1.0)
    else
        -r.stake
    end
end

# **CRITICAL**: Rename the model so we can distinguish it in the rankings!
df_cal.model_name .= "Mu_CALIBRATED" 
df_cal.model_parameters .= "Window_40" # Or whatever your settings were


# ==============================================================================
# 3. RESULTS: The Championship
# ==============================================================================

# Combine the two ledgers
full_ledger_df = vcat(ledger.df, df_cal)




# Run your Ranking Function
final_rankings = rank_strategies(full_ledger_df)
strategy_rankings = rank_strategies(full_ledger_df)

# Show the showdown
cols = [:model_name, :selection, :Win_Rate, :Growth_Rate, :Edge_Ratio, :Bet_Freq]
sort!(final_rankings, :Growth_Rate, rev=true)
display(final_rankings[!, cols])



model_names = unique(strategy_rankings.selection)
for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(strategy_rankings, :selection => ByRow(isequal(m_name)))
  show(sort(sub, :Growth_Rate, rev=true))
end




##### --- isotonic regression 
"""
    isotonic_regression(y::Vector{Float64}, weights::Vector{Float64}=ones(length(y)))

Solves the Pool Adjacent Violators Algorithm (PAVA) to find the monotonic fit.
Returns the fitted values (y_hat) that minimize squared error while ensuring non-decreasing order.
"""
function isotonic_regression(y::AbstractVector{T}, weights::AbstractVector{T}=ones(T, length(y))) where T <: AbstractFloat
    n = length(y)
    if n <= 1 return y end
    
    # We work on indices that sort the input (if x wasn't sorted, but usually we sort X outside)
    # PAVA operates on the sequence y.
    
    # Solution vectors
    solution = copy(y)
    w = copy(weights)
    
    # Blocks represented by indices
    block_indices = collect(1:n) 
    
    # Iterate and pool
    i = 1
    while i < length(solution)
        # If order is violated (decreasing)
        if solution[i] >= solution[i+1]
            # Pool i and i+1
            numerator = w[i] * solution[i] + w[i+1] * solution[i+1]
            denominator = w[i] + w[i+1]
            new_val = numerator / denominator
            new_w = denominator
            
            # Merge in solution array (conceptually shrinking the array)
            # In an optimized PAVA we use a stack, but for <100k points, vector ops are fine
            deleteat!(solution, i+1)
            deleteat!(w, i+1)
            solution[i] = new_val
            w[i] = new_w
            
            # Backtrack to check previous constraints
            if i > 1; i -= 1; end
        else
            i += 1
        end
    end
    
    # Now we have the "steps". We need to map them back to the original points.
    # (For a proper transform, usually we use a library, but let's do the "Transform" logic below)
    return solution
end


using GLM, Plots, Statistics

function calibrate_to_market_isotonic(ds, loaded_results, market_data)
    
    # 1. Get Model Probabilities
    # --------------------------
    # (Assuming you extracted them already)
    # Let's assume we have a dataframe 'preds' with [:match_id, :prob_model_home]
    # For this snippet, I'll generate dummy data to show the workflow
    preds = DataFrame(match_id = market_data.match_id, prob_model = rand(nrow(market_data))) 
    
    # JOIN: Match your model output to the Market Data
    data = leftjoin(market_data, preds, on=:match_id)
    
    # Filter for Home Team only to keep it simple (or handle 1X2 separately)
    data = filter(row -> row.selection == "home", data)
    
    # 2. Prepare Data for Isotonic Regression
    # ---------------------------------------
    # X: Your Model's Probability
    # Y: The Market's "Fair" Closing Probability
    
    # We MUST sort by X (Your Model) for Isotonic Regression to work on Y
    sort!(data, :prob_model)
    
    x_model = Float64.(data.prob_model)
    y_market = Float64.(data.prob_fair_close)
    
    # 3. Run Isotonic Regression (PAVA)
    # ---------------------------------
    # This finds the monotonic line that best fits the Market Probabilities
    y_calibrated = isotonic_regression(y_market)
    
    # 4. Visualize the Correction
    # ---------------------------
    # Plot 1: The Calibration Map
    p1 = plot(x_model, y_calibrated, 
        title="Isotonic Calibration (Model vs Market)",
        xlabel="My Model Probability",
        ylabel="Market Fair Probability (Close)",
        legend=false, color=:red, lw=2
    )
    # Reference diagonal
    plot!(p1, [0,1], [0,1], linestyle=:dash, color=:black)
    
    # 5. Apply to New Data (Interpolation)
    # ------------------------------------
    # Isotonic gives us a lookup table. We need a function to apply it.
    # Simple linear interpolation between the steps.
    
    function apply_iso(p_raw)
        # Find nearest point in x_model
        # (In production, use a fast binary search or Interp1d)
        idx = searchsortedfirst(x_model, p_raw)
        if idx > length(y_calibrated) return y_calibrated[end] end
        if idx == 1 return y_calibrated[1] end
        return y_calibrated[idx] # Step function approximation
    end
    
    return p1, apply_iso
end


a = calibrate_to_market_isotonic(ds, loaded_results[2], market_data) 




####

using DataFrames, Statistics, GLM, Plots, Distributions

# ==============================================================================
# 1. HELPER: The PAVA Solver (Isotonic Regression)
# ==============================================================================
function isotonic_regression(y::Vector{Float64})
    n = length(y)
    if n <= 1 return y end
    solution = copy(y)
    weights = ones(n)
    i = 1
    while i < length(solution)
        if solution[i] >= solution[i+1]
            numerator = weights[i] * solution[i] + weights[i+1] * solution[i+1]
            denominator = weights[i] + weights[i+1]
            new_val = numerator / denominator
            new_w = denominator
            deleteat!(solution, i+1); deleteat!(weights, i+1)
            solution[i] = new_val; weights[i] = new_w
            if i > 1; i -= 1; end
        else
            i += 1
        end
    end
    # Map steps back to original size (simple expansion for plotting)
    # Note: For strict usage, you'd build a lookup function. 
    # Here we just return the unique steps for the plot.
    return solution 
end

# ==============================================================================
# 2. ANALYSIS: Model vs Market
# ==============================================================================

function analyze_isotonic_fit(ds, exp_res, market_data)
    println(">>> 1. Calculating Model Probabilities (Home Win)...")
    
    # A. Get Latents
    latents = BayesianFootball.Experiments.extract_oos_predictions(ds, exp_res)
    
    # B. Compute P(Home Win) for every match
    # We use the integration logic from your ledger simulation
    model_probs = DataFrame(match_id = Int[], prob_model = Float64[])
    
    for row in eachrow(latents.df)
        r = mean(vec(row.r))
        lam_h = mean(vec(row.λ_h))
        lam_a = mean(vec(row.λ_a))
        
        # Construct Dists
        p_h = r / (r + lam_h); d_h = NegativeBinomial(r, p_h)
        p_a = r / (r + lam_a); d_a = NegativeBinomial(r, p_a)
        
        # # Sum prob(Home > Away)
        # # Truncated sum for speed
        # p_home_win = 0.0
        # for g_h in 0:10, g_a in 0:10
        #     if g_h > g_a
        #         p_home_win += pdf(d_h, g_h) * pdf(d_a, g_a)
        #     end
        # end
        # # Normalize roughly (ignoring >10 goals)
        # total_mass = sum(pdf(d_h, i)*pdf(d_a, j) for i in 0:10, j in 0:10)
        # p_home_win /= total_mass
        #
        # push!(model_probs, (row.match_id, p_home_win))
      # Sum prob(Total Goals > 2.5)
              p_outcome = 0.0
              
              # Loop through scores
              for g_h in 0:15, g_a in 0:15 
                  
                  # --- THE FIX ---
                  # Old Logic: if g_h > g_a (Home Win)
                  # New Logic: if g_h + g_a > 2.5 (Over 2.5)
                  if (g_h + g_a) > 2.5
                      p_outcome += pdf(d_h, g_h) * pdf(d_a, g_a)
                  end
              end
              
              # Normalize
              total_mass = sum(pdf(d_h, i)*pdf(d_a, j) for i in 0:15, j in 0:15)
              p_outcome /= total_mass
              
              push!(model_probs, (row.match_id, p_outcome))
    end
    
    println(">>> 2. Merging with Market Data...")
    # FIX: Access market_data.df, not market_data directly
    # Filter for 'Home' selection only to calibrate the Home Win model
    mkt_subset = filter(r -> r.selection == :over_25, market_data.df)
    
    # Join
    data = innerjoin(mkt_subset, model_probs, on=:match_id)
    println("    Matched $(nrow(data)) games.")
    
    println(">>> 3. Running Isotonic Regression...")
    sort!(data, :prob_model) # Sort X for PAVA
    
    x_model = data.prob_model
    y_market = data.prob_fair_close # The Truth
    
    y_iso = isotonic_regression(copy(y_market))
    
    # ==========================================================================
    # 4. PLOTTING & VERDICT
    # ==========================================================================
    
    # To plot the step function correctly, we need to repeat the steps
    # But for a quick look, scatter plot + step line is enough.
    # We will create a step function interpolator for the plot
    
    p = scatter(x_model, y_market, 
        label="Market Truth", color=:black, alpha=0.1, markersize=2,
        title="Isotonic Calibration: Home Win",
        xlabel="Model Probability", ylabel="Market Fair Prob (Close)"
    )
    
    # Plot the Diagonal (Perfect Calibration)
    plot!(p, [0,1], [0,1], label="Perfect Calibration", color=:red, linestyle=:dash)
    
    # Plot the Isotonic Fit (The S-Curve)
    # We plot the 'y_iso' against the sorted 'x_model', but we need to handle the compression
    # The simple PAVA returns unique steps. Let's just run a standard GLM or simple binning 
    # for visualization if PAVA vector logic is complex to expand.
    # Actually, simpler visual: Bin the data.
    
    # Bin Model Probs into 5% buckets and take mean of Market Prob
    data.bin = round.(data.prob_model .* 20) ./ 20
    binned = combine(groupby(data, :bin), :prob_fair_close => mean => :mkt_mean)
    sort!(binned, :bin)
    
    plot!(p, binned.bin, binned.mkt_mean, 
        label="Actual Trend (Binned)", color=:blue, lw=3, marker=:circle
    )
    
    display(p)
    savefig("isotonic_check.png")
    
    # --- THE METRIC ---
    # Measure the "Bias Slope"
    # If Slope < 1.0, you are Overconfident.
    lm_fit = lm(@formula(prob_fair_close ~ prob_model), data)
    slope = coef(lm_fit)[2]
    
    println("\n---------------------------------------------------")
    println("                 DIAGNOSTIC VERDICT                ")
    println("---------------------------------------------------")
    println("Slope (Beta): $(round(slope, digits=3))")
    
    if slope > 1.05
        println("Status: UNDER-CONFIDENT. (Market moves further than you predict).")
        println("Action: Isotonic Calibration will HELPFULLY aggressive-ize your odds.")
    elseif slope < 0.95
        println("Status: OVER-CONFIDENT. (You are too sure of yourself).")
        println("Action: Isotonic Calibration will SAVE MONEY by damping your odds.")
    else
        println("Status: CALIBRATED. (Slope ≈ 1.0).")
        println("Action: Little value in calibration. Focus on new features.")
    end
    println("---------------------------------------------------")
end

# RUN IT
analyze_isotonic_fit(ds, loaded_results[2], market_data)




using DataFrames, Statistics, Distributions, Plots

function optimize_phi_multiplier(ds, loaded_results)
    println(">>> Optimizing Phi Multiplier for Profit...")
    
    # 1. Extract Raw Latents once (Speed optimization)
    raw_latents = BayesianFootball.Experiments.extract_oos_predictions(ds, loaded_results)
    
    # 2. Define the Grid (Testing multipliers from 0.1 to 1.5)
    # We expect the winner to be < 1.0 (Deflating Phi to increase variance)
    multipliers = 0.1:0.1:1.5
    
    results = DataFrame(Multiplier=[], Growth_Rate=[], Bets=[], ROI=[])
    
    # Pre-calculate lookups
    odds_map = Dict(r.match_id => (r.home_odds, r.away_odds) for r in eachrow(ds.odds))
    actuals_map = Dict(r.match_id => (r.home_score, r.away_score) for r in eachrow(ds.matches))

    for M in multipliers
        ledger = []
        
        for row in eachrow(raw_latents.df)
            mid = row.match_id
            if !haskey(odds_map, mid) || !haskey(actuals_map, mid) continue end
            
            # --- MODIFY PHI HERE ---
            # We multiply the raw Phi by M. 
            # M < 1.0 makes the model "loos" (More Variance).
            raw_r = mean(vec(row.r))
            mod_r = clamp(raw_r * M, 1e-4, 1e4) 
            
            raw_lam_h = mean(vec(row.λ_h))
            raw_lam_a = mean(vec(row.λ_a)) # Assuming away lambda exists or is derived
            
            # Recalculate Probability with NEW Phi
            p_h = mod_r / (mod_r + raw_lam_h)
            d_h = NegativeBinomial(mod_r, p_h)
            
            p_a = mod_r / (mod_r + raw_lam_a)
            d_a = NegativeBinomial(mod_r, p_a)
            
            # Calculate Over 2.5 Prob
            prob_over = 0.0
            for g_h in 0:10, g_a in 0:10
                if (g_h + g_a) > 2.5
                    prob_over += pdf(d_h, g_h) * pdf(d_a, g_a)
                end
            end
            
            # Bet Logic (Over 2.5 Only for this test)
            # Find market odds for Over 2.5 (Assuming it's in your odds table, 
            # or we simulate 1X2 profit if that's what you have. 
            # Adapting to your previous 1X2 context:)
            
            # Let's assume we are testing 1X2 HOME WIN for simplicity of the snippet
            # (Replace with your Over 2.5 logic if you have those odds)
            prob_home = 0.0
            for g_h in 0:10, g_a in 0:10
                if g_h > g_a
                    prob_home += pdf(d_h, g_h) * pdf(d_a, g_a)
                end
            end
            
            (h_odd, a_odd) = odds_map[mid]
            (s_h, s_a) = actuals_map[mid]
            
            # Flat Stake Bet if Edge > 2%
            if (prob_home * h_odd) > 1.02
                pnl = (s_h > s_a) ? (h_odd - 1.0) : -1.0
                push!(ledger, pnl)
            end
        end
        
        # Calculate Stats
        if isempty(ledger)
            push!(results, (M, NaN, 0, NaN))
        else
            g = mean(log.(1.0 .+ (ledger .* 0.05))) # Kelly proxy (5% stake)
            roi = sum(ledger) / length(ledger)
            push!(results, (M, g, length(ledger), roi))
        end
        println("Multiplier $M: Bets=$(length(ledger)), ROI=$(round(roi*100, digits=1))%")
    end
    
    return results
end

# Run it
res = optimize_phi_multiplier(ds, loaded_results[2])
display(res)
plot(res.Multiplier, res.ROI, title="Profit vs. Phi Multiplier", xlabel="Phi Multiplier (M)", ylabel="ROI", marker=:circle)


using DataFrames, Statistics, Distributions, Plots

# ==============================================================================
# 1. HELPER: Build Odds Map from Long Format
# ==============================================================================
function build_over25_odds_map(market_df)
    # We need a dictionary: match_id -> (Over_Odds, Under_Odds)
    odds_map = Dict{Int, Tuple{Float64, Float64}}()
    
    # Filter for relevant selections
    # (Handling both Symbol and String just in case)
    is_over(s) = s == :over_25 || s == "over_25"
    is_under(s) = s == :under_25 || s == "under_25"
    
    # We'll use a temporary dict to store partial matches
    temp = Dict{Int, Dict{Symbol, Float64}}()
    
    for row in eachrow(market_df)
        mid = row.match_id
        sel = row.selection
        odd = row.odds_close # Using Closing Odds as the "Truth" to beat
        
        if is_over(sel)
            get!(temp, mid, Dict())[:over] = odd
        elseif is_under(sel)
            get!(temp, mid, Dict())[:under] = odd
        end
    end
    
    # Finalize map (only keep matches with BOTH odds)
    for (mid, d) in temp
        if haskey(d, :over) && haskey(d, :under)
            odds_map[mid] = (d[:over], d[:under])
        end
    end
    
    return odds_map
end

# ==============================================================================
# 2. OPTIMIZER: The Market Deflator
# ==============================================================================
function optimize_phi_multiplier(ds, loaded_results, market_data)
    println(">>> 1. Building Odds Map for Over/Under 2.5...")
    odds_map = build_over25_odds_map(market_data.df)
    println("    Found $(length(odds_map)) matches with Over/Under odds.")

    println(">>> 2. Extracting Raw Model Predictions...")
    raw_latents = BayesianFootball.Experiments.extract_oos_predictions(ds, loaded_results)
    
    # Get actual scores for PnL calculation
    actuals_map = Dict(r.match_id => (r.home_score, r.away_score) for r in eachrow(ds.matches))

    println(">>> 3. Running Grid Search (Phi Multiplier)...")
    
    # Grid: From 0.1 (Massive Damping) to 1.5 (Tightening)
    multipliers = 0.1:0.05:1.5
    results = DataFrame(Multiplier=[], Growth_Rate=[], Bets=[], ROI=[])
    
    for M in multipliers
        ledger = Float64[]
        
        for row in eachrow(raw_latents.df)
            mid = row.match_id
            
            # Skip if we don't have data
            if !haskey(odds_map, mid) || !haskey(actuals_map, mid) continue end
            
            (o_odd, u_odd) = odds_map[mid]
            (s_h, s_a) = actuals_map[mid]
            total_goals = s_h + s_a
            
            # --- THE ENGINE MODIFICATION ---
            # 1. Modify Phi (r)
            # M < 1.0 -> Lower Phi -> Higher Variance -> Damped Probabilities
            raw_r = mean(vec(row.r))
            mod_r = clamp(raw_r * M, 1e-4, 1e4)
            
            raw_lam_h = mean(vec(row.λ_h))
            raw_lam_a = mean(vec(row.λ_a))
            
            # 2. Re-calculate Probabilities
            # (Using point estimates for speed in the loop)
            p_h = mod_r / (mod_r + raw_lam_h); d_h = NegativeBinomial(mod_r, p_h)
            p_a = mod_r / (mod_r + raw_lam_a); d_a = NegativeBinomial(mod_r, p_a)
            
            # Calculate P(Over 2.5)
            # Sum convolution of Home + Away > 2.5
            prob_over = 0.0
            for i in 0:10, j in 0:10
                if (i + j) > 2.5
                    prob_over += pdf(d_h, i) * pdf(d_a, j)
                end
            end
            # Normalize (truncate error correction)
            total_mass = sum(pdf(d_h, i)*pdf(d_a, j) for i in 0:10, j in 0:10)
            prob_over /= total_mass
            
            # --- BETTING LOGIC ---
            # Simple Value Bet: If Edge > 2%
            threshold = 0.02
            
            # Bet Over?
            if (prob_over * o_odd) > (1.0 + threshold)
                pnl = (total_goals > 2.5) ? (o_odd - 1.0) : -1.0
                push!(ledger, pnl)
            end
            
            # Optional: Bet Under? (To see if damping helps unders too)
            # prob_under = 1.0 - prob_over
            # if (prob_under * u_odd) > (1.0 + threshold)
            #     pnl = (total_goals <= 2.5) ? (u_odd - 1.0) : -1.0
            #     push!(ledger, pnl)
            # end
        end
        
        # Stats
        if isempty(ledger)
            push!(results, (M, NaN, 0, NaN))
        else
            # Log Growth (Kelly Proxy with 5% stake)
            g = mean(log.(1.0 .+ (ledger .* 0.05)))
            roi = sum(ledger) / length(ledger)
            push!(results, (M, g, length(ledger), roi))
        end
    end
    
    return results
end

# RUN IT
# Make sure to pass 'market_data' here!
optimization_results = optimize_phi_multiplier(ds, loaded_results[2], market_data)

# Show top 5 settings
sort!(optimization_results, :Growth_Rate, rev=true)
display(first(optimization_results, 10))

# Plot the Curve
plot(optimization_results.Multiplier, optimization_results.ROI, 
    title="Profit vs. Phi Multiplier (Over 2.5)", 
    xlabel="Phi Multiplier (M)", 
    ylabel="ROI", 
    lw=2, marker=:circle, label="ROI"
)
