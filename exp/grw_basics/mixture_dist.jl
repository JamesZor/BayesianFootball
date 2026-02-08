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

# 2. Run Backtest
# ===============
baker = BayesianKelly()

my_signals = [baker]

as = AnalyticalShrinkageKelly()

kelly = KellyCriterion(1)
kelly25 = KellyCriterion(1/4)
flat_strat = FlatStake(0.05)

my_signals = [baker, as, kelly, kelly25, flat_strat]


# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)



#### dev 

ou_25 = subset(ledger.df, :stake => ByRow(!isequal(0.0)), :selection => ByRow(isequal(:over_25)))

transform!(ou_25, :pnl => ByRow(p -> ifelse(p < 0, abs(p), 0.0)) => :x,
                 :pnl => ByRow(p -> ifelse(p > 0, p, 0.0)) => :y)

using Statistics

# Mean of losses only (x)
mean_loss = mean(filter(!iszero, ou_25.x))

# Mean of wins only (y)
mean_win = mean(filter(!iszero, ou_25.y))


using StatsPlots
using Plots

# Extract clean data
wins = filter(>(0), ou_25.y)
losses = filter(>(0), ou_25.x)

# Combine for grouped plots (Box/Violin)
data = [wins; losses]
labels = [fill("Wins (y)", length(wins)); fill("Losses (x)", length(losses))]

histogram(log.(wins), label="Wins (y)", alpha=0.5, bins=50, title="PnL Distribution (Zeros Removed)")
histogram!(log.(losses), label="Losses (x)", alpha=0.5, bins=50)


# Violin plot with Boxplot overlaid
violin(labels, data, marker=(0.2, :blue), label="")
boxplot!(labels, data, fillalpha=0.5, title="PnL Comparison", ylabel="Value")

using Optim, Statistics, StatsPlots

# 1. Define the Box-Cox function
function boxcox(y, λ)
    return λ ≈ 0 ? log.(y) : (y .^ λ .- 1) ./ λ
end

# 2. Define a function to find the best λ (minimizing skewness or maximizing log-likelihood)
function find_best_λ(data)
    # We want the transformed data to have a distribution close to normal
    # One way is to maximize the log-likelihood of a normal distribution
    res = optimize(λ -> begin
        transformed = boxcox(data, λ)
        n = length(transformed)
        v = var(transformed)
        # Log-likelihood formula (simplified)
        (n/2) * log(v) - (λ - 1) * sum(log.(data))
    end, -2.0, 2.0) # Search between λ of -2 and 2
    return Optim.minimizer(res)
end

# 3. Apply it to your data
wins = filter(>(0), ou_25.y)
best_λ = find_best_λ(wins)
wins_bc = boxcox(wins, best_λ)

println("Best Lambda found: ", best_λ)

# 4. Plot
histogram(wins_bc, title="Box-Cox Transformed (λ = $(round(best_λ, digits=2)))", bins=50)


#### 
using Distributions, Statistics

# 1. Prepare the clean data (drop zeros)
# -------------------------------------
wins_data   = filter(>(0), ou_25.y)
losses_data = filter(>(0), ou_25.x)

# 2. Estimate Mixing Probability 'p' (MLE)
# -------------------------------------
# p is simply: Number of Wins / (Number of Wins + Number of Losses)
p = length(wins_data) / (length(wins_data) + length(losses_data))
println("Estimated Win Probability (p): ", round(p, digits=4))

# 3. Fit Component Distributions
# -------------------------------------
# We use LogNormal as it fits betting/financial data well (bounded at 0, heavy tail).
# You can swap this with Gamma or Weibull if preferred.
dist_Y = fit(LogNormal, wins_data)   # Distribution of Wins
dist_X = fit(LogNormal, losses_data) # Distribution of Losses (Magnitudes)

println("Win Distribution:  ", dist_Y)
println("Loss Distribution: ", dist_X)

# 4. Calculate Moments of the Mixture (R)
# -------------------------------------
# We need the raw moments (E[X], E[X^2]...) to combine them.
# Note: For the loss component (-X), E[(-X)^k] = (-1)^k * E[X^k]

# Helper to get raw moment E[D^k]
raw_moment(d::Distribution, k) = moment(d, k)

# First Moment (Mean / Expected Value)
# E[R] = p*E[Y] + (1-p)*E[-X]
ev = p * mean(dist_Y) - (1-p) * mean(dist_X)

# 1. Define the analytical moment function for LogNormal
# ----------------------------------------------------
function lognormal_moment(d::LogNormal, k::Int)
    μ, σ = params(d)
    return exp(k*μ + (k^2 * σ^2)/2)
end
# 2. Recalculate the Raw Moments of the Mixture (R)
# ----------------------------------------------------
# We use the fitted distributions you already created: dist_X and dist_Y

# First Moment (EV) - You already calculated this, but let's be consistent
# E[R] = p*E[Y] - (1-p)*E[X]
m1 = p * lognormal_moment(dist_Y, 1) - (1-p) * lognormal_moment(dist_X, 1)

# Second Raw Moment (E[R^2])
# E[R^2] = p*E[Y^2] + (1-p)*E[X^2]   (Squared negatives become positive)
m2 = p * lognormal_moment(dist_Y, 2) + (1-p) * lognormal_moment(dist_X, 2)

# Third Raw Moment (E[R^3])
# E[R^3] = p*E[Y^3] - (1-p)*E[X^3]   (Cubed negatives remain negative)
m3 = p * lognormal_moment(dist_Y, 3) - (1-p) * lognormal_moment(dist_X, 3)

# Fourth Raw Moment (E[R^4])
# E[R^4] = p*E[Y^4] + (1-p)*E[X^4]   (Fourth power negatives become positive)
m4 = p * lognormal_moment(dist_Y, 4) + (1-p) * lognormal_moment(dist_X, 4)

# 5. Convert to Central Moments (Variance, Skew, Kurt)
# -------------------------------------
variance_R = m2 - ev^2
std_dev_R  = sqrt(variance_R)

# Skewness formula using raw moments
skew_R = (m3 - 3*ev*variance_R - ev^3) / (std_dev_R^3)

# Kurtosis formula (excess kurtosis usually requires subtracting 3, 
# but here is the standard 4th standardized moment)
kurt_R = (m4 - 4*ev*m3 + 6*(ev^2)*m2 - 3*(ev^4)) / (variance_R^2)

println("-"^30)
println("Mixture Model Statistics (PnL):")
println("Expected Value (Mean PnL): ", round(ev, digits=5))
println("Variance:                  ", round(variance_R, digits=5))
println("Skewness:                  ", round(skew_R, digits=4))
println("Kurtosis:                  ", round(kurt_R, digits=4))


##
#
using Distributions, Statistics

# 1. Get the clean data again
wins_data   = filter(>(0), ou_25.y)
losses_data = filter(>(0), ou_25.x)
p = length(wins_data) / (length(wins_data) + length(losses_data))

# 2. Check Sample Means (The Truth)
real_mean_win = mean(wins_data)
real_mean_loss = mean(losses_data)
println("--- REAL DATA ---")
println("Actual Avg Win:  ", round(real_mean_win, digits=4))
println("Actual Avg Loss: ", round(real_mean_loss, digits=4))
println("Actual Net EV:   ", round(p * real_mean_win - (1-p) * real_mean_loss, digits=5))

# 3. Fit Gamma Distribution
# ------------------------
dist_Y_gamma = fit(Gamma, wins_data)
dist_X_gamma = fit(Gamma, losses_data)

println("\n--- GAMMA FIT ---")
println("Win Model Mean:  ", round(mean(dist_Y_gamma), digits=4))
println("Loss Model Mean: ", round(mean(dist_X_gamma), digits=4))

# 4. Mixture Stats with Gamma
# ------------------------
# Gamma moments are stable, so we can use the generic functions
ev_gamma = p * mean(dist_Y_gamma) - (1-p) * mean(dist_X_gamma)

# For variance, we use: Var(R) = E[R^2] - (E[R])^2
# E[X^2] for Gamma = var(X) + mean(X)^2
m2_wins   = var(dist_Y_gamma) + mean(dist_Y_gamma)^2
m2_losses = var(dist_X_gamma) + mean(dist_X_gamma)^2

# E[R^2] mixture
m2_mix = p * m2_wins + (1-p) * m2_losses
var_mix = m2_mix - ev_gamma^2

println("\n--- GAMMA RESULTS ---")
println("Model EV:        ", round(ev_gamma, digits=5))
println("Model Variance:  ", round(var_mix, digits=5))


# Helper function for Gamma Raw Moments E[X^k]
function gamma_raw_moment(d::Gamma, k::Int)
    α, θ = params(d)
    # Product: θ^k * α * (α+1) * ... * (α+k-1)
    return (θ^k) * prod(α + i for i in 0:(k-1))
end

# 1. Calculate Raw Moments for Wins (Y) and Losses (X)
# ----------------------------------------------------
# We need the 3rd and 4th moments
m3_wins   = gamma_raw_moment(dist_Y_gamma, 3)
m4_wins   = gamma_raw_moment(dist_Y_gamma, 4)

m3_losses = gamma_raw_moment(dist_X_gamma, 3)
m4_losses = gamma_raw_moment(dist_X_gamma, 4)

# 2. Combine into Mixture Moments (R)
# ----------------------------------------------------
# Recall: R = Y (with prob p) and -X (with prob 1-p)

# Mean (M1) and Second Moment (M2) - You already calculated these, bringing them forward
m1_mix = ev_gamma       # 0.0025
m2_mix = m2_mix         # 0.00466... from your previous step

# Third Raw Moment (M3)
# E[R^3] = p*E[Y^3] - (1-p)*E[X^3]
m3_mix = p * m3_wins - (1-p) * m3_losses

# Fourth Raw Moment (M4)
# E[R^4] = p*E[Y^4] + (1-p)*E[X^4]
m4_mix = p * m4_wins + (1-p) * m4_losses

# 3. Calculate Final Central Statistics
# ----------------------------------------------------
# Variance (σ²)
var_mix = m2_mix - m1_mix^2
std_mix = sqrt(var_mix)

# Skewness
skew_mix = (m3_mix - 3*m1_mix*var_mix - m1_mix^3) / (std_mix^3)

# Kurtosis (Excess)
kurt_mix_pearson = (m4_mix - 4*m1_mix*m3_mix + 6*(m1_mix^2)*m2_mix - 3*(m1_mix^4)) / (var_mix^2)
kurt_mix_excess  = kurt_mix_pearson - 3

println("-"^30)
println("Final Gamma Mixture Statistics:")
println("Skewness:        ", round(skew_mix, digits=4))
println("Excess Kurtosis: ", round(kurt_mix_excess, digits=4))



using StatsPlots

# Define the custom Mixture PDF function
function mixture_pdf(x)
    if x > 0
        return p * pdf(dist_Y_gamma, x)
    elseif x < 0
        return (1-p) * pdf(dist_X_gamma, -x)
    else
        return 0.0
    end
end

# Plot
# 1. Actual Data Density
real_pnl = [filter(>(0), ou_25.y); -filter(>(0), ou_25.x)]
density(real_pnl, label="Actual PnL", linewidth=2, fill=(0, 0.2, :blue), xlimit=(-0.3, 0.3))

# 2. Model PDF
plot!(x -> mixture_pdf(x), -0.3, 0.3, label="Gamma Mixture Model", linewidth=3, linestyle=:dash, color=:red)


# The CDF of the mixture model for x < 0 (Loss side)
# P(R ≤ x) = (1-p) * P(-X ≤ x) = (1-p) * P(X ≥ -x) = (1-p) * (1 - CDF_Gamma(-x))
function mixture_cdf_loss(x)
    if x >= 0 
        return 1.0 # We only care about the loss tail for VaR
    end
    return (1-p) * (1 - cdf(dist_X_gamma, -x))
end

# Find the 5% quantile numerically
using Roots
var_95_model = find_zero(x -> mixture_cdf_loss(x) - 0.05, (-0.5, 0.0))

# Compare with Empirical (Actual Data) VaR
using Statistics
var_95_actual = quantile(real_pnl, 0.05)

println("--- Risk Analysis (95% Confidence) ---")
println("Model VaR:  ", round(var_95_model, digits=4))
println("Actual VaR: ", round(var_95_actual, digits=4))


# The Monte Carlo Simulation
using Random, Statistics, StatsPlots

# 1. Simulation Settings
n_sims = 50000       # Number of "parallel universes" to simulate
n_bets = 1000       # Length of each simulation (approx 2 years of betting?)
start_bankroll = 1.0 # Normalized bankroll (100%)

# 2. Fast Path Generator
#    Using the pre-calculated Gamma distributions (dist_Y_gamma, dist_X_gamma)
function simulate_path(n_steps)
    pnl_stream = zeros(Float64, n_steps)
    
    # We can bulk generate random numbers for speed
    # Decide Win/Loss for all steps at once
    is_win = rand(n_steps) .< p 
    
    for i in 1:n_steps
        if is_win[i]
            pnl_stream[i] = rand(dist_Y_gamma)
        else
            pnl_stream[i] = -rand(dist_X_gamma)
        end
    end
    return cumsum(pnl_stream)
end

# 3. Calculate Max Drawdown for a single curve
function max_drawdown(equity_curve)
    peak = -Inf
    max_dd = 0.0
    
    # We assume equity_curve represents cumulative PnL
    # If starting at 0, current_equity = equity_curve[i]
    for val in equity_curve
        if val > peak
            peak = val
        end
        dd = peak - val
        if dd > max_dd
            max_dd = dd
        end
    end
    return max_dd
end

# 4. Run the Monte Carlo
# ----------------------
println("Running $n_sims simulations...")
drawdowns = Float64[]
final_pnl = Float64[]
paths = [] # Store a few paths for plotting

for i in 1:n_sims
    path = simulate_path(n_bets)
    push!(drawdowns, max_drawdown(path))
    push!(final_pnl, path[end])
    
    # Save the first 100 paths for visualization
    if i <= 100
        push!(paths, path)
    end
end

# 5. Analysis
# -----------
avg_dd = mean(drawdowns)
p95_dd = quantile(drawdowns, 0.95) # The "Worst Case" scenario (95% confidence)
prob_ruin_20 = count(d -> d >= 0.20, drawdowns) / n_sims # Prob of losing >20% units

println("-"^30)
println("Monte Carlo Results ($n_bets bets):")
println("Median Drawdown:      ", round(median(drawdowns), digits=4))
println("95% Worst Drawdown:   ", round(p95_dd, digits=4))
println("Prob. >20% Drawdown:  ", round(prob_ruin_20 * 100, digits=2), "%")
println("Expected Final PnL:   ", round(mean(final_pnl), digits=4))


# Plotting the "Cone"
plot(title="Projected Equity Curves (1000 Bets)", legend=false, xlabel="Bet Number", ylabel="Cumulative PnL")

# Plot 100 random paths in grey
for path in paths
    plot!(path, color=:grey, alpha=0.3, linewidth=1)
end

# Plot the "Average" Theoretical Path in Red
plot!(1:n_bets, (1:n_bets) .* ev_gamma, color=:red, linewidth=3, label="Expected Value")

# Add a horizontal line for the break-even point
hline!([0], color=:black, linestyle=:dash)


# ------------------------------------------------------------------------
# Rank Strategies 
# ------------------------------------------------------------------------


using DataFrames, Distributions, Statistics, StatsBase

function rank_strategies(ledger::DataFrame)
    group_cols = [:model_name, :model_parameters, :signal_name, :market_name, :selection]

    results = combine(groupby(ledger, group_cols)) do df
        # 1. Filter for active bets
        active_bets = filter(row -> abs(row.stake) > 1e-6, df)
        n_bets = nrow(active_bets)
        bet_freq = n_bets / nrow(df)
        
        # 2. Skip insignificant strategies (less than 10 bets)
        if n_bets < 10
            return (Win_Rate=NaN, Growth_Rate=NaN, Exp_Value=NaN, 
                    Kelly_Risk_Theta=NaN, Edge_Ratio=NaN, Avg_Stake=NaN, Bet_Freq=bet_freq)
        end

        # 3. Prepare vectors
        wins = filter(>(0), active_bets.pnl)
        losses = abs.(filter(<(0), active_bets.pnl)) 

        # 4. Safe Gamma Fitting
        theta_loss = NaN
        edge_ratio = NaN

        if length(losses) >= 5 && var(losses) > 1e-8
            try
                d_loss = fit(Gamma, losses)
                theta_loss = params(d_loss)[2] # Theta (Scale)
                
                if !isempty(wins) && length(wins) >= 5 && var(wins) > 1e-8
                    d_win = fit(Gamma, wins)
                    edge_ratio = mean(d_win) / mean(d_loss)
                else
                    edge_ratio = mean(wins) / mean(losses) # Fallback to simple mean ratio
                end
            catch e
                theta_loss = NaN # Fit failed (likely data issues)
            end
        elseif !isempty(losses)
            # Fallback for constant losses: Variance is 0, so risk is just the magnitude
            theta_loss = mean(losses) 
            edge_ratio = isempty(wins) ? 0.0 : mean(wins) / mean(losses)
        else
            theta_loss = 0.0 # No losses
            edge_ratio = Inf
        end

        # 5. Expected Log-Growth
        g = mean(log.(1.0 .+ active_bets.pnl))

        return (
            Win_Rate = length(wins) / n_bets,
            Growth_Rate = g,
            Exp_Value = mean(active_bets.pnl),
            Kelly_Risk_Theta = theta_loss, 
            Edge_Ratio = edge_ratio,      
            Avg_Stake = mean(active_bets.stake),
            Bet_Freq = bet_freq
        )
    end

    # Remove rows where calculation failed completely (NaN growth)
    filter!(row -> !isnan(row.Growth_Rate), results)
    
    # Sort
    sort!(results, :Growth_Rate, rev=true)
    
    return results
end

# Run the fixed function
strategy_rankings = rank_strategies(ledger.df)

model_names = unique(strategy_rankings.selection)
for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(strategy_rankings, :selection => ByRow(isequal(m_name)))
  show(sort(sub, :Growth_Rate, rev=true))
end

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

using StatsPlots, Distributions

function plot_strategy_comparison(ledger)
    # Extract data for the "Sniper" (Draws, Phi Model)
    sniper_df = filter(r -> r.selection == :draw && r.model_name == "GRWNegativeBinomialPhi", ledger)
    
    # Extract data for the "Grinder" (Over 1.5, Mu Model)
    grinder_df = filter(r -> r.selection == :over_15 && r.model_name == "GRWNegativeBinomialMu", ledger)

    # Plot PnL Density
    p = density(filter(!iszero, sniper_df.pnl), label="Sniper (Draws)", 
                linewidth=3, fill=(0, 0.2, :blue), xlimit=(-1.5, 4.0),
                title="Strategy DNA: Sniper vs. Grinder", xlabel="PnL per Bet")
    
    density!(p, filter(!iszero, grinder_df.pnl), label="Grinder (Over 1.5)", 
             linewidth=3, fill=(0, 0.2, :orange))
             
    return p
end

plot_strategy_comparison(ledger.df)



###
using DataFrames, Distributions, Statistics, Dates

# 1. Define the Decay Kernel
#    c = 0.05 implies a "half-life" of about 14 weeks (0.05 * 14 ≈ 0.7)
#    Higher c = faster reaction, more noise. Lower c = smoother, slower reaction.
function time_weight(t_current, t_event; c=0.05)
    delta = (t_current - t_event).value # Days difference
    return exp(-c * max(0, delta))
end

# 2. Weighted Gamma Fitter (Approximation)
#    Fitting distributions with weights is tricky in standard packages.
#    We can approximate it by "resampling" based on weights or using weighted moments.
function fit_weighted_gamma_moments(values, weights)
    # Calculate weighted mean and variance
    w_sum = sum(weights)
    mu = sum(values .* weights) / w_sum
    
    # Weighted Variance
    # Reliability weights formula: sum(w * (x - mu)^2) / (sum(w) - sum(w^2)/sum(w))
    numerator = sum(weights .* (values .- mu).^2)
    denominator = w_sum - (sum(weights.^2) / w_sum)
    var_w = numerator / denominator
    
    if var_w <= 1e-8 || mu <= 0 return (NaN, NaN) end

    # Method of Moments for Gamma:
    # mu = alpha * theta
    # var = alpha * theta^2
    # -> theta = var / mu
    # -> alpha = mu / theta = mu^2 / var
    
    theta = var_w / mu
    alpha = mu^2 / var_w
    
    return (alpha, theta)
end

# 3. The Portfolio Walk-Forward Loop
function run_dynamic_portfolio(ledger; decay_rate=0.01)
    # Sort by date ensures we respect causality
    sort!(ledger, :date)
    
    # Get unique weeks/dates to iterate over
    dates = unique(ledger.date)
    n_weeks = length(dates)
    
    # Store history of our dynamic assessments
    history = DataFrame()

    # Define the specific strategy we are monitoring (e.g., the "Sniper")
    # In a real system, you'd loop over ALL strategies here
    target_strat = filter(r -> r.selection == :draw && r.model_name == "GRWNegativeBinomialPhi", ledger)
    
    # Start loop (need some history first, e.g., start at week 10)
    for i in 10:n_weeks
        current_date = dates[i]
        
        # 1. Get Past Data (t < current_date)
        past_data = filter(r -> r.date < current_date, target_strat)
        if nrow(past_data) < 20 continue end
        
        # 2. Calculate Weights
        weights = [time_weight(current_date, d, c=decay_rate) for d in past_data.date]
        
        # 3. Fit Weighted Gamma for Wins and Losses
        wins_mask = past_data.pnl .> 0
        losses_mask = past_data.pnl .< 0
        
        # Fit Win Distribution
        if sum(weights[wins_mask]) > 1e-3
            # Use only positive PnL
            a_win, t_win = fit_weighted_gamma_moments(past_data.pnl[wins_mask], weights[wins_mask])
        else
            a_win, t_win = (0.0, 0.0)
        end
        
        # Fit Loss Distribution (Absolute value)
        if sum(weights[losses_mask]) > 1e-3
            a_loss, t_loss = fit_weighted_gamma_moments(abs.(past_data.pnl[losses_mask]), weights[losses_mask])
        else
            a_loss, t_loss = (0.0, 0.0)
        end
        
        # 4. Calculate Dynamic Metrics
        # Weighted Win Probability
        # p = sum(weights_win) / sum(total_weights)
        p_dyn = sum(weights[wins_mask]) / sum(weights)
        
        # Expected Growth (Weighted)
        # We can just take the weighted average of log(1+r)
        g_dyn = sum(log.(1.0 .+ past_data.pnl) .* weights) / sum(weights)
        
        # 5. Decision: Kelly Scale
        # If Growth is negative, we cut.
        # If Risk Theta is exploding, we cut.
        allocation_scale = 1.0
        if g_dyn < 0
            allocation_scale = 0.0
        elseif t_loss > 0.10 # "Panic Threshold" for fat tails
            allocation_scale = 0.5 # Cut stakes in half
        end
        
        push!(history, (
            Date = current_date,
            Dyn_Win_Rate = p_dyn,
            Dyn_Growth = g_dyn,
            Dyn_Risk_Theta = t_loss,
            Scale = allocation_scale
        ))
    end
    
    return history
end

# Run the simulation
# decay_rate=0.05 is fairly aggressive (responds to last month)
# decay_rate=0.01 is slow (responds to season trends)
dyn_stats = run_dynamic_portfolio(ledger.df, decay_rate=0.12)


using StatsPlots

function compare_equity_curves(ledger, dynamic_history)
    # 1. Join the Dynamic Decisions back to the Ledger
    #    We align decisions to the NEXT date (avoid look-ahead)
    
    # Create a mapping: Date -> Scale (from the previous day's analysis)
    # We map dynamic_history.Date to the Scale calculated ON that date
    decision_map = Dict(row.Date => row.Scale for row in eachrow(dynamic_history))
    
    # 2. Re-run the ledger
    dates = sort(unique(ledger.date))
    
    static_equity = [1.0]
    managed_equity = [1.0]
    
    current_scale = 1.0 # Default start
    
    for d in dates
        # Get bets for this day
        days_bets = filter(r -> r.date == d, ledger)
        if isempty(days_bets) continue end
        
        # Check if we have a NEW decision from the Portfolio Manager
        # (In reality, we use the decision made typically *before* the games start)
        if haskey(decision_map, d)
            current_scale = decision_map[d]
        end
        
        # Calculate daily returns
        # Static: Just sum PnL
        day_pnl_static = sum(days_bets.pnl)
        
        # Managed: Sum PnL * current_scale
        day_pnl_managed = sum(days_bets.pnl .* current_scale)
        
        # Update Equity (Simple additive for visualization, or geometric if preferred)
        push!(static_equity, static_equity[end] + day_pnl_static)
        push!(managed_equity, managed_equity[end] + day_pnl_managed)
    end
    
    # 3. Plot
    p = plot(title="Portfolio Management Impact", xlabel="Betting Days", ylabel="Cumulative Units", legend=:topleft)
    plot!(p, static_equity, label="Static (Always Bet)", linewidth=2, color=:grey, linestyle=:dash)
    plot!(p, managed_equity, label="Dynamic (Airbag Enabled)", linewidth=3, color=:blue)
    
    return p
end

compare_equity_curves(ledger.df, dyn_stats)


### Particle Filter Portfolio Manager tracking your Gamma Mixture parameters.

using Distributions, StatsFuns, Random, DataFrames, Statistics

# ------------------------------------------------------------------
# 1. State Definition
# ------------------------------------------------------------------
struct StrategyState
    logit_p::Float64      # Logit of Win Probability (allows GRW on real line)
    log_theta_loss::Float64 # Log of Loss Scale (allows GRW on real line)
end

# ------------------------------------------------------------------
# 2. Particle Filter Settings
# ------------------------------------------------------------------
const N_PARTICLES = 2000
const GRW_STD_P = 0.15       # How fast can Win Rate change per week?
const GRW_STD_THETA = 0.10   # How fast can Risk change per week?

# We fix Alpha (Shape) to global averages to stabilize the filter
# (You can get these from your static analysis)
const FIXED_ALPHA_WIN = 1.2
const FIXED_THETA_WIN = 0.05 # Assuming win size is relatively constant
const FIXED_ALPHA_LOSS = 1.5

# ------------------------------------------------------------------
# 3. Helper Functions
# ------------------------------------------------------------------
function transform_state(s::StrategyState)
    return (logistic(s.logit_p), exp(s.log_theta_loss))
end

# Likelihood of ONE bet given a specific particle's parameters
function bet_likelihood(pnl::Float64, p::Float64, theta_loss::Float64)
    # Mixture Model: P(y) = p * Gamma_Win(y) + (1-p) * Gamma_Loss(-y)
    
    if pnl > 0
        # Win Distribution
        d_win = Gamma(FIXED_ALPHA_WIN, FIXED_THETA_WIN)
        return p * pdf(d_win, pnl)
    elseif pnl < 0
        # Loss Distribution (Negative Gamma)
        d_loss = Gamma(FIXED_ALPHA_LOSS, theta_loss)
        return (1.0 - p) * pdf(d_loss, abs(pnl))
    else
        return 1.0 # Ignore 0.0 stakes/pushes
    end
end

# ------------------------------------------------------------------
# 4. The Sequential Filter
# ------------------------------------------------------------------
function run_particle_filter(ledger)
    # Group data by week for sequential processing
    sort!(ledger, :date)
    weeks = sort(unique(ledger.date))
    
    # Initialize Particles (The "Prior" Belief)
    # Start assuming roughly 30% win rate, average risk
    particles = [StrategyState(logit(0.30), log(0.04)) for _ in 1:N_PARTICLES]
    weights = ones(Float64, N_PARTICLES) ./ N_PARTICLES
    
    history = DataFrame()
    
    for w_date in weeks
        # 1. Get bets for this week
        weekly_bets = filter(r -> r.date == w_date && abs(r.stake) > 1e-6, ledger)
        
        # --- PREDICTION STEP (Before seeing this week's results) ---
        # Calculate Expected State for decision making
        avg_logit_p = mean([pts.logit_p for pts in particles])
        avg_log_theta = mean([pts.log_theta_loss for pts in particles])
        
        pred_p = logistic(avg_logit_p)
        pred_theta = exp(avg_log_theta)
        
        # Portfolio Decision Logic (The "Manager")
        # We assume loss alpha is fixed, so Risk ~ Theta * Alpha
        expected_loss_size = pred_theta * FIXED_ALPHA_LOSS
        
        # Kelly-like Scaling logic
        scale = 1.0
        if pred_p < 0.20 || pred_theta > 0.08
            scale = 0.0 # Too risky or losing too much
        elseif pred_theta > 0.05
            scale = 0.5 # Caution
        end
        
        push!(history, (
            Date = w_date,
            Tracked_WinRate = pred_p,
            Tracked_Risk_Theta = pred_theta,
            Scale = scale
        ))

        # --- UPDATE STEP (After seeing results) ---
        # If no bets this week, just diffuse (GRW) without re-weighting
        if isempty(weekly_bets)
            # Gaussian Random Walk (Diffusion only)
            for i in 1:N_PARTICLES
                new_lp = particles[i].logit_p + randn() * GRW_STD_P
                new_lt = particles[i].log_theta_loss + randn() * GRW_STD_THETA
                particles[i] = StrategyState(new_lp, new_lt)
            end
            continue
        end

        # Calculate Likelihoods for each particle
        log_weights = zeros(Float64, N_PARTICLES)
        
        for i in 1:N_PARTICLES
            p, theta = transform_state(particles[i])
            
            # Sum log-likelihood of all bets in this week for this particle
            ll = 0.0
            for row in eachrow(weekly_bets)
                lik = bet_likelihood(row.pnl, p, theta)
                ll += log(max(1e-10, lik)) # Avoid log(0)
            end
            log_weights[i] = ll
        end
        
        # Numerical stability for weights
        max_lw = maximum(log_weights)
        weights = exp.(log_weights .- max_lw)
        weights ./= sum(weights) # Normalize
        
        # Resampling (Bootstrap Filter)
        # Select particles with high likelihood to survive
        indices = rand(Categorical(weights), N_PARTICLES)
        new_particles = particles[indices]
        
        # Jitter / Mutation (The GRW Step)
        for i in 1:N_PARTICLES
            jitter_lp = randn() * GRW_STD_P
            jitter_lt = randn() * GRW_STD_THETA
            
            # Evolve state for next week
            new_particles[i] = StrategyState(
                new_particles[i].logit_p + jitter_lp,
                new_particles[i].log_theta_loss + jitter_lt
            )
        end
        particles = new_particles
    end
    
    return history
end

# Run on your "Sniper" strategy
target_strat = filter(r -> r.selection == :over_25, ledger.df)
pf_results = run_particle_filter(target_strat)


#
using StatsPlots, ProgressMeter

function run_full_portfolio_simulation(ledger)
    # 1. Identify all unique strategies
    #    Group by Model + Selection + Market
    strats = unique(ledger[!, [:model_name, :selection, :market_name]])
    
    println("Initializing Portfolio Manager for $(nrow(strats)) strategies...")
    
    # Store decisions: Dict( (Strategy_Key, Date) => Scale )
    decision_log = Dict()
    
    # 2. Run Particle Filter for EACH strategy
    @showprogress for row in eachrow(strats)
        # Extract specific strategy history
        sub_df = filter(r -> r.model_name == row.model_name && 
                             r.selection == row.selection && 
                             r.market_name == row.market_name, ledger)
        
        # Only run if we have enough data (>20 bets)
        if nrow(sub_df) < 20 continue end
        
        # Run the Particle Filter (using the function we defined previously)
        pf_history = run_particle_filter(sub_df)
        
        # Store the decisions
        key = (row.model_name, row.selection, row.market_name)
        for h_row in eachrow(pf_history)
            decision_log[(key, h_row.Date)] = h_row.Scale
        end
    end
    
    # 3. Simulate the Combined Portfolio PnL
    #    Static vs. Dynamic
    println("Simulating Portfolio Returns...")
    
    dates = sort(unique(ledger.date))
    static_equity = [1.0]
    dynamic_equity = [1.0]
    
    for d in dates
        daily_bets = filter(r -> r.date == d, ledger)
        if isempty(daily_bets) continue end
        
        day_pnl_static = 0.0
        day_pnl_dynamic = 0.0
        
        for bet in eachrow(daily_bets)
            # Standard Static PnL
            day_pnl_static += bet.pnl
            
            # Dynamic PnL
            # Lookup the scale for this specific strategy on this date
            strat_key = (bet.model_name, bet.selection, bet.market_name)
            
            # If we have a decision, use it. Default to 1.0 (or 0.0 if strict) if missing.
            scale = get(decision_log, (strat_key, d), 1.0)
            
            day_pnl_dynamic += (bet.pnl * scale)
        end
        
        push!(static_equity, static_equity[end] + day_pnl_static)
        push!(dynamic_equity, dynamic_equity[end] + day_pnl_dynamic)
    end
    
    # 4. Plot
    p = plot(title="Full Portfolio: Static vs Dynamic Manager", 
             xlabel="Betting Days", ylabel="Cumulative Units", legend=:topleft)
    plot!(p, static_equity, label="Static Portfolio", color=:grey, linestyle=:dash)
    plot!(p, dynamic_equity, label="Dynamic Portfolio (Particle Filter)", color=:blue, linewidth=2)
    
    return p, static_equity, dynamic_equity
end

# Run the master simulation
plt, stat, dyn = run_full_portfolio_simulation(ledger.df)
display(plt)


#
using StatsPlots, DataFrames

# 1. Filter the Ledger for YOUR Best Markets
#    We only keep the selections you identified as strong
best_selections = [:over_15, :over_25, :over_35]

# Filter logic: Must be OverUnder market AND one of our best selections
target_ledger = filter(r -> r.market_name == "OverUnder" && 
                            r.selection in best_selections, ledger.df)

println("Running Portfolio Manager on Core 'Overs' Strategy...")
println("Total Bets: ", nrow(target_ledger))

# 2. Run the Full Simulation on this subset
#    (Re-using the function we wrote in the previous step)
plt_overs, stat_overs, dyn_overs = run_full_portfolio_simulation(target_ledger)

# 3. Calculate Performance Metrics
static_return = stat_overs[end] - 1.0
dynamic_return = dyn_overs[end] - 1.0
improvement = ((dynamic_return - static_return) / abs(static_return)) * 100

println("-"^30)
println("Final Results (Overs Only):")
println("Static Return:      ", round(static_return, digits=2), " units")
println("Dynamic Return:     ", round(dynamic_return, digits=2), " units")
println("Manager Impact:     ", round(improvement, digits=2), "%")

# 4. Show the Plot
display(plt_overs)


#
using StatsPlots, DataFrames, ProgressMeter

function run_granular_portfolio(ledger)
    # 1. Identify Unique Strategy Streams (Model + Selection)
    #    We restrict this to your "Overs" core for this test
    targets = [:over_15, :over_25, :over_35]
    
    # Get unique combinations
    strats = unique(filter(r -> r.selection in targets, ledger[!, [:model_name, :selection, :market_name]]))
    
    println("Initializing Manager for $(nrow(strats)) distinct strategies...")
    
    # Store decisions: Dict( (Model, Selection, Date) => Scale )
    decision_log = Dict()
    
    # 2. Run Particle Filter for EACH strategy independently
    @showprogress for row in eachrow(strats)
        # Extract specific strategy history
        sub_df = filter(r -> r.model_name == row.model_name && 
                             r.selection == row.selection && 
                             r.market_name == row.market_name, ledger)
        
        # Only run if we have enough data (>20 bets) to form a belief
        if nrow(sub_df) < 20 continue end
        
        # Run PF
        pf_history = run_particle_filter(sub_df)
        
        # Log Decisions
        key = (row.model_name, row.selection)
        for h_row in eachrow(pf_history)
            decision_log[(key, h_row.Date)] = h_row.Scale
        end
    end
    
    # 3. Simulate Returns
    println("Simulating Granular Returns...")
    dates = sort(unique(ledger.date))
    static_equity = [1.0]
    dynamic_equity = [1.0]
    
    for d in dates
        daily_bets = filter(r -> r.date == d && r.selection in targets, ledger)
        if isempty(daily_bets) continue end
        
        day_pnl_static = 0.0
        day_pnl_dynamic = 0.0
        
        for bet in eachrow(daily_bets)
            # Static: Always 1.0
            day_pnl_static += bet.pnl
            
            # Dynamic: Lookup specific model decision
            strat_key = (bet.model_name, bet.selection)
            scale = get(decision_log, (strat_key, d), 1.0)
            
            day_pnl_dynamic += (bet.pnl * scale)
        end
        
        push!(static_equity, static_equity[end] + day_pnl_static)
        push!(dynamic_equity, dynamic_equity[end] + day_pnl_dynamic)
    end
    
    # 4. Final Comparison Stats
    s_ret = static_equity[end] - 1.0
    d_ret = dynamic_equity[end] - 1.0
    imp = ((d_ret - s_ret) / abs(s_ret)) * 100
    
    println("Granular Results:")
    println("Static:  ", round(s_ret, digits=2))
    println("Dynamic: ", round(d_ret, digits=2))
    println("Impact:  ", round(imp, digits=2), "%")
    
    # Plot
    p = plot(title="Granular Portfolio Management (Overs Only)", 
             xlabel="Betting Days", ylabel="Cumulative Units", legend=:topleft)
    plot!(p, static_equity, label="Static", color=:grey, linestyle=:dash)
    plot!(p, dynamic_equity, label="Dynamic (Per Model)", color=:blue, linewidth=2)
    
    return p
end

# Run it
plt_granular = run_granular_portfolio(ledger.df)
display(plt_granular)



function plot_pf_diagnostics(pf_history, strategy_name)
    # 1. Win Rate Evolution
    p1 = plot(pf_history.Date, pf_history.Tracked_WinRate, 
              label="Win Rate (p)", color=:blue, ylabel="Probability",
              title="$strategy_name: Manager Beliefs", legend=:topright, ylim=(0, 1.0))
    
    # Add simple moving average of actual win rate for comparison (optional visual aid)
    # hline!(p1, [mean(pf_history.Tracked_WinRate)], color=:blue, linestyle=:dot, label="Avg")

    # 2. Risk Evolution (Theta)
    p2 = plot(pf_history.Date, pf_history.Tracked_Risk_Theta, 
              label="Risk (Theta)", color=:red, ylabel="Tail Risk",
              legend=:topright)
    
    # Add a "Danger Line" at 0.08 (where we cut stakes)
    hline!(p2, [0.08], color=:black, linestyle=:dash, label="Cutoff")
    
    # 3. Scale Decision
    p3 = plot(pf_history.Date, pf_history.Scale, 
              label="Stake Scale", color=:green, ylabel="Multiplier",
              linetype=:step, fill=(0, 0.2, :green), ylim=(-0.1, 1.1))
    
    # Combine
    l = @layout [a; b; c]
    plot(p1, p2, p3, layout=l, size=(800, 800))
end

# Run specifically for one Over 2.5 model (pick the most active one)
# We filter for a specific model name to avoid mixing signals
target_model_name = "GRWNegativeBinomial" # Or "GRWNegativeBinomialPhi" based on your table
target_strat = filter(r -> r.selection == :over_25 && r.model_name == target_model_name, ledger.df)

pf_results_diag = run_particle_filter(target_strat)
plot_pf_diagnostics(pf_results_diag, "Over 2.5 ($target_model_name)")


#=
Updated Particle Filter (Smart Manager)

Here is the upgraded function. It tracks avg_odds and cuts stakes the moment the strategy loses its mathematical edge

Dynamic Expected Value (EV) Targeting

We need to update the Manager to track the Average Odds of the strategy. Then, instead of a hard cutoff (0.20), we compare the Tracked Win Rate against the Break-Even Win Rate.
Edget​=Tracked_pt​×(Avg_Oddst​−1)−(1−Tracked_pt​)

If Edget​<0, we stop betting immediately.
=# 

using Distributions, StatsFuns, Random, DataFrames, Statistics

# ... (Previous Structs/Constants same as before) ...
# We add a rolling window for odds tracking
const ODDS_DECAY = 0.05 

function run_smart_particle_filter(ledger)
    sort!(ledger, :date)
    weeks = sort(unique(ledger.date))
    
    # Initialize Particles
    particles = [StrategyState(logit(0.30), log(0.04)) for _ in 1:N_PARTICLES]
    weights = ones(Float64, N_PARTICLES) ./ N_PARTICLES
    
    # Initialize Rolling Odds (Start conservative, e.g., 2.0)
    avg_odds = 2.0 
    
    history = DataFrame()
    
    for w_date in weeks
        weekly_bets = filter(r -> r.date == w_date && abs(r.stake) > 1e-6, ledger)
        
        # --- 1. PREDICTION & DECISION ---
        
        # Get Current Beliefs
        avg_logit_p = mean([pts.logit_p for pts in particles])
        avg_log_theta = mean([pts.log_theta_loss for pts in particles])
        
        pred_p = logistic(avg_logit_p)
        pred_theta = exp(avg_log_theta)
        
        # Calculate Break-Even Requirement
        # If odds are 2.0, we need > 50%. If 1.5, we need > 66%.
        # We add a small "Safety Buffer" (e.g. 1%) so we don't bet on razor-thin edges
        break_even_p = 1.0 / avg_odds 
        
        # --- NEW DECISION LOGIC ---
        scale = 1.0
        
        # Condition A: The "Slow Bleed" Check (Is Edge < 0?)
        if pred_p < break_even_p
            scale = 0.0 # Cut completely. The strategy is currently -EV.
            
        # Condition B: The "Airbag" Check (Fat Tails)
        elseif pred_theta > 0.08
            scale = 0.0 # Too volatile.
            
        # Condition C: The "Caution" Zone (Edge is thin OR Volatility rising)
        elseif (pred_p < break_even_p + 0.02) || (pred_theta > 0.05)
            scale = 0.5 # Bet half size
        end
        
        push!(history, (
            Date = w_date,
            Tracked_WinRate = pred_p,
            Tracked_Risk_Theta = pred_theta,
            Break_Even_P = break_even_p, # Save this to plot later!
            Scale = scale
        ))

        # --- 2. UPDATE STEP ---
        
        if isempty(weekly_bets)
            # Diffusion only
            for i in 1:N_PARTICLES
                new_lp = particles[i].logit_p + randn() * GRW_STD_P
                new_lt = particles[i].log_theta_loss + randn() * GRW_STD_THETA
                particles[i] = StrategyState(new_lp, new_lt)
            end
            continue
        end

        # Update Average Odds (Rolling mean)
        w_avg_odds = mean(weekly_bets.odds)
        avg_odds = (1 - ODDS_DECAY) * avg_odds + ODDS_DECAY * w_avg_odds

        # Particle Update (Likelihoods)
        log_weights = zeros(Float64, N_PARTICLES)
        for i in 1:N_PARTICLES
            p, theta = transform_state(particles[i])
            ll = 0.0
            for row in eachrow(weekly_bets)
                lik = bet_likelihood(row.pnl, p, theta)
                ll += log(max(1e-10, lik))
            end
            log_weights[i] = ll
        end
        
        # Normalize & Resample
        max_lw = maximum(log_weights)
        weights = exp.(log_weights .- max_lw)
        weights ./= sum(weights)
        
        indices = rand(Categorical(weights), N_PARTICLES)
        new_particles = particles[indices]
        
        # Jitter/Diffusion
        for i in 1:N_PARTICLES
            jitter_lp = randn() * GRW_STD_P
            jitter_lt = randn() * GRW_STD_THETA
            new_particles[i] = StrategyState(
                new_particles[i].logit_p + jitter_lp,
                new_particles[i].log_theta_loss + jitter_lt
            )
        end
        particles = new_particles
    end
    
    return history
end



pf_results_diag = run_smart_particle_filter(target_strat)
plot_pf_diagnostics(pf_results_diag, "Over 2.5 ($target_model_name)")


using RollingFunctions, StatsPlots, DataFrames, Statistics

# RECOMMENDED SETTINGS (Smoother)
const GRW_STD_P = 0.02       # Drastically lower this. Win rates drift slowly.
const GRW_STD_THETA = 0.05

function plot_ev_reality_check(pf_history, strategy_ledger; window=50)
    # 1. Prepare "Reality" (Realized EV)
    # ----------------------------------
    # Sort bets chronologically
    sort!(strategy_ledger, :date)
    
    # Calculate Rolling Average PnL (Simple Moving Average)
    # This represents "How much are we actually making per bet right now?"
    realized_ev = runmean(strategy_ledger.pnl, window)
    
    # We need to align this with dates for plotting. 
    # Since there are multiple bets per day, we'll just plot every bet's smoothed value.
    dates_real = strategy_ledger.date
    
    # 2. Prepare "Belief" (Particle EV)
    # ---------------------------------
    # Derive implied average odds from the Break_Even_P column (1 / Break_Even_P)
    # Note: Break_Even_P might be 0.5 (Odds 2.0). 
    # EV = WinRate * Odds - 1
    
    # We use the Break_Even_P from history, assuming avg_odds = 1/Break_Even_P
    avg_odds_hist = 1.0 ./ pf_history.Break_Even_P
    particle_ev = (pf_history.Tracked_WinRate .* avg_odds_hist) .- 1.0
    
    # 3. The Plot
    # -----------
    p = plot(title="The Reality Check: Belief vs. Actual Performance",
             xlabel="Date", ylabel="Expected Value per Bet (Units)",
             legend=:topleft, size=(900, 500))
    
    # A. The Zero Line (Break Even)
    hline!(p, [0.0], color=:black, linestyle=:dash, label="")
    
    # B. Reality (Grey Line)
    # We plot this first so it's in the background
    plot!(p, dates_real, realized_ev, 
          label="Realized EV (Rolling $window bets)", 
          color=:grey, alpha=0.6, linewidth=1.5)
          
    # C. Belief (Blue Line)
    plot!(p, pf_history.Date, particle_ev, 
          label="Manager Belief (Particle EV)", 
          color=:blue, linewidth=3)
    
    # D. Active Betting Zones (Green Shading)
    # We want to highlight where the manager actually allowed betting
    # We scale the 'Scale' to fit on the graph (e.g. max height of EV)
    y_max = maximum(particle_ev)
    y_min = minimum(realized_ev)
    
    # Create a "Step" fill for when Scale > 0
    # We transform Scale to match plot coordinates for shading
    active_dates = pf_history.Date
    active_zones = [s > 0 ? y_max : y_min for s in pf_history.Scale]

    start_view = Date(2021, 7, 1)  # Start of 21/22 Season
    end_view   = Date(2025, 6, 1)  # Present day
    
    # Use fillrange to shade
    # (A simple way is just to plot markers where Scale > 0)
    scatter!(p, pf_history.Date[pf_history.Scale .> 0], 
             particle_ev[pf_history.Scale .> 0],
             markershape=:circle, color=:green, markersize=3, 
             xlims = (start_view, end_view),
             label="Betting Active")
             
    scatter!(p, pf_history.Date[pf_history.Scale .== 0], 
             particle_ev[pf_history.Scale .== 0],
             markershape=:x, color=:red, markersize=3, 
             label="Betting Cut")

    return p
end

# Run the visualizer
plot_ev_reality_check(pf_results_diag, target_strat)
