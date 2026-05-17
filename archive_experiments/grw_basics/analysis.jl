# exp/grw_basics/analysis.jl

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

# 2. Run Backtest
# ===============
println("\nRunning Backtest on $(length(loaded_results)) models...")

baker = BayesianKelly()
as = AnalyticalShrinkageKelly()
exact_bayesian = ExactBayesianKelly() 

my_signals = [baker]
my_signals = [as]

kelly = KellyCriterion(1)
kelly25 = KellyCriterion(1/4)

my_signals = [baker, as, exact_bayesian, kelly, kelly25, flat_strat]

flat_strat = FlatStake(0.05)
my_signals = [flat_strat]

# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
  loaded_results[4], 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

# 3. Analyze
# ==========
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

println("\n=== TEARSHEET SUMMARY ===")
println(tearsheet)

# Breakdown by Model (Selection)
println("\n=== BREAKDOWN BY MODEL ===")
model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end

for m in loaded_results 
println(m.config.model)
println("\n")
end



### 
using DataFrames, StatsPlots, Dates, Printf

# --- 1. CONFIGURATION ---
# We want to compare the "Battle" for the 2.5 Goal Line
under_sym = [:home]
over_sym  = [:away]

over_sym  = [:btts_yes, :over_15]

over_sym  = [:btts_yes, :over_15, :over_25, :over_35, :draw, :under_55]
under_sym = [:over_150]


start_view = Date(2021, 7, 1)  # Start of 21/22 Season
end_view   = Date(2025, 6, 1)  # Present day

# --- 2. DATA CLEANING (Fixes the Year 0478 Bug) ---
# We filter the ledger to remove any bad dates
clean_ledger = filter(row -> row.date >= start_view, ledger.df)

println("Data Range: ", minimum(clean_ledger.date), " to ", maximum(clean_ledger.date))

# --- 3. PREPARE PLOT ---
# Get unique models to assign consistent colors
unique_models = unique(clean_ledger[:, [:model_name, :model_parameters]])
colors = palette(:tab10)

# Helper for readable labels
function make_short_label(name, params)
    short_name = replace(name, "GRWNegativeBinomial" => "NegBin", 
                               "GRWBivariatePoisson" => "BivPois",
                               "GRWPoisson" => "Pois",
                               "GRWDixonColes" => "DC")
    m = match(r"μ=([0-9\.]+)", params)
    param_str = m === nothing ? "" : " μ=$(m.captures[1])"
    return "$short_name$param_str"
end

# Initialize ONE master plot with fixed Time Axis
p = plot(title = "Regime Battle: $(over_sym) vs $(under_sym)", 
         xlabel = "Date", ylabel = "Cumulative Profit (Units)",
         legend = :outertopright, 
         size = (1000, 600), 
         margin = 5Plots.mm,
         xlims = (start_view, end_view)) # <--- Forces correct zoom

# Add Zero line (Break Even)
hline!(p, [0.0], label="Break Even", color=:black, linestyle=:dash, alpha=0.5)

# --- 4. PLOTTING LOOP ---
for (i, model_row) in enumerate(eachrow(unique_models[[1,2,3,5],:]))
# for (i, model_row) in enumerate(eachrow(unique_models))
    
    m_name = model_row.model_name
    m_params = model_row.model_parameters
    lbl_base = make_short_label(m_name, m_params)
    
    # Assign specific color
    col_idx = mod1(i, 10)
    model_color = colors[col_idx]

    # Filter for this specific model
    model_ledger = filter(row -> row.model_name == m_name && 
                                 row.model_parameters == m_params, clean_ledger)

    # Plot OVERS (Solid Line)
    overs = filter(row -> row.selection in over_sym, model_ledger)
    if !isempty(overs)
        sort!(overs, :date)
        overs.cum_pnl = cumsum(overs.pnl)
        plot!(p, overs.date, overs.cum_pnl, 
              label="$(lbl_base) | Over", 
              lw=2, color=model_color, linestyle=:solid)
    end

    # Plot UNDERS (Dashed Line)
    unders = filter(row -> row.selection in under_sym, model_ledger)
    if !isempty(unders)
        sort!(unders, :date)
        unders.cum_pnl = cumsum(unders.pnl)
        plot!(p, unders.date, unders.cum_pnl, 
              label="$(lbl_base) | Under", 
              lw=2, color=model_color, linestyle=:dash)
    end
end

display(p)


#######
#
#######
using DataFrames, Statistics

function analyze_mixture_moments(ledger::DataFrame)
    # 1. Filter for active bets only (ignore zeros)
    active_bets = filter(row -> row.stake > 0, ledger)

    # 2. Group by Signal and Selection (Market)
    gdf = groupby(active_bets, [:signal_name, :selection])

    # 3. Compute Moments
    results = combine(gdf) do group
        N = nrow(group)
        
        # Empirical Probabilities
        wins = filter(row -> row.pnl > 0, group)
        win_rate = nrow(wins) / N
        
        # 1. Expectation (Mean Return per Bet) - The "Drift"
        # We use PnL directly as it represents fractional wealth change (if stake is fraction)
        mu = mean(group.pnl)
        
        # 2. Variance (The Volatility)
        # Var(X) = E[X^2] - (E[X])^2
        second_moment = mean(group.pnl .^ 2)
        variance = second_moment - mu^2
        
        # 3. Skewness Proxy (Upside vs Downside)
        # Ratio of Average Win to Average Loss
        avg_win = isempty(wins) ? 0.0 : mean(wins.pnl)
        avg_loss = mean(filter(row -> row.pnl <= 0, group).pnl) # will be negative
        payoff_ratio = abs(avg_win / avg_loss)

        # 4. The "Safety Score" (Sharpe-like but for this mixture)
        # How much drift do we get per unit of variance?
        safety_score = variance > 0 ? mu / sqrt(variance) : 0.0

        return (
            bets = N,
            win_rate = round(win_rate, digits=3),
            E_return = round(mu * 100, digits=4),      # Scaled for readability
            Var_return = round(variance * 1000, digits=4), # Scaled
            Payoff_Ratio = round(payoff_ratio, digits=2),
            Safety_Score = round(safety_score, digits=3)
        )
    end

    sort!(results, :Safety_Score, rev=true)
    return results
end

# Run it
moments = analyze_mixture_moments(ledger.df)
println(moments)

subset(moments, :signal_name => ByRow(in(["BayesianKelly", "AnalyticalShrinkageKelly"])))


using DataFrames, Statistics, Plots, RollingFunctions, Dates

# 1. Filter for the specific Random Variable stream (Over 2.5)
# Change "AnalyticalShrinkageKelly" to whichever signal you want to analyze
market_data = subset(ledger.df, 
    :selection => ByRow(==(:over_25)), 
    :signal_name => ByRow(==("AnalyticalShrinkageKelly")),
    :stake => ByRow(>(0.0)) # Only active bets
)

# 2. Sort by time (Crucial for time-series analysis)
sort!(market_data, :date)

# This Vector IS your sequence of realizations r_1, r_2, ... r_t
R_series = market_data.pnl
times = market_data.date

# A. The Discrete Realizations (The "Barcode" of Variance)
p1 = bar(times, R_series, 
    title="Realizations of R_i (PnL per Bet)", 
    label="R_i", 
    linecolor=:match, 
    alpha=0.6,
    ylabel="Fractional Return",
    seriescolor = [r > 0 ? :green : :red for r in R_series] # Green wins, Red losses
)

# B. The Stochastic Process (Cumulative Wealth Path)
# This is the integral of your random variable: W_t = Sum(R_i)
wealth_path = cumsum(R_series)
p2 = plot(times, wealth_path, 
    title="Cumulative Process (Wealth Log-Growth)", 
    label="Σ R_i", 
    linewidth=2, 
    color=:blue,
    legend=:topleft
)

plot(p1, p2, layout=(2,1), size=(800, 600))

# 

using DataFrames, Statistics

function estimate_time_varying_parameters(pnl_vector::Vector{Float64}; alpha=0.05)
    n = length(pnl_vector)
    
    # Initialize output vectors
    p_loss_est = zeros(Float64, n)   # p_t
    mu_win_est = zeros(Float64, n)   # E[W_t]
    mu_loss_est = zeros(Float64, n)  # E[L_t]
    expected_drift = zeros(Float64, n) # E[R_t]
    
    # Initialization (Burn-in assumptions)
    # Assume 50/50 win rate and avg win/loss from first few data points initially
    current_p = 0.5
    current_mu_w = mean(filter(x -> x > 0, pnl_vector[1:min(10,n)]))
    current_mu_l = abs(mean(filter(x -> x <= 0, pnl_vector[1:min(10,n)])))
    
    # Handle edge case if initial data is empty/zeros
    if isnan(current_mu_w) current_mu_w = 0.01 end
    if isnan(current_mu_l) current_mu_l = 0.01 end

    for t in 1:n
        r_t = pnl_vector[t]
        
        # 1. Observe Regime (Win or Loss?)
        is_loss = r_t <= 0
        
        # 2. Update Probability Estimator (EWMA)
        # If loss, input is 1.0, else 0.0
        current_p = alpha * (is_loss ? 1.0 : 0.0) + (1 - alpha) * current_p
        
        # 3. Update Conditional Magnitudes
        if is_loss
            # Update Loss Magnitude (take abs value of return)
            current_mu_l = alpha * abs(r_t) + (1 - alpha) * current_mu_l
        else
            # Update Win Magnitude
            current_mu_w = alpha * r_t + (1 - alpha) * current_mu_w
        end
        
        # 4. Store Estimates for time t
        p_loss_est[t] = current_p
        mu_win_est[t] = current_mu_w
        mu_loss_est[t] = current_mu_l
        
        # 5. Calculate Instantaneous Expectation (Drift)
        # E[R] = (1-p)*Win - p*Loss
        expected_drift[t] = (1 - current_p) * current_mu_w - (current_p * current_mu_l)
    end
    
    return DataFrame(
        p_loss = p_loss_est,
        exp_win = mu_win_est,
        exp_loss = mu_loss_est,
        drift = expected_drift
    )
end

# Usage on your specific market data
estimates = estimate_time_varying_parameters(R_series, alpha=0.03) # alpha=0.03 ~ 33 bet window

# Attach dates for plotting
estimates.date = times

p1 = plot(estimates.date, estimates.drift, 
    title="Real-Time Expected Drift E[R_t]",
    label="Estimated Edge",
    color=:purple,
    linewidth=2,
    ylabel="Exp. Return per Bet",
    legend=:topleft
)
hline!(p1, [0.0], color=:black, linestyle=:dash, label="Zero Edge")

p2 = plot(estimates.date, estimates.p_loss,
    title="Estimated Probability of Loss (p_t)",
    label="p_loss",
    color=:red,
    ylim=(0, 1),
    ylabel="Probability"
)

plot(p1, p2, layout=(2,1), size=(800, 600))


# 


using DataFrames, Statistics, RollingFunctions

"""
    RegimeFilteredSignal(base_signal, window_size)

Wraps a base signal (like AnalyticalShrinkage) and only allows betting
if the recent rolling PnL drift is positive.
"""
mutable struct RegimeFilteredSignal <: AbstractSignal
    base_signal::AbstractSignal
    rolling_window::Int
    recent_pnl::Vector{Float64} # History buffer
end

# Constructor
RegimeFilteredSignal(s::AbstractSignal; window=50) = 
    RegimeFilteredSignal(s, window, Float64[])

function compute_stake(s::RegimeFilteredSignal, dist::AbstractVector, odds::Number)
    # 1. Check Regime Health
    drift = 0.0
    if length(s.recent_pnl) >= 10
        # Calculate recent average return per bet
        # Using a simple mean of the last N bets
        n = min(length(s.recent_pnl), s.rolling_window)
        drift = mean(s.recent_pnl[end-n+1:end])
    else
        drift = 0.01 # Assume innocent until proven guilty (start phase)
    end

    # 2. The Circuit Breaker
    if drift < 0.0
        return 0.0 # STOP BETTING!
    end

    # 3. If Healthy, delegate to the base strategy
    return compute_stake(s.base_signal, dist, odds)
end

# 4. Critical: We must update the history after every match!
# You need to call this in your backtest loop after seeing the result.
function update_history!(s::RegimeFilteredSignal, pnl::Float64)
    push!(s.recent_pnl, pnl)
    # Optional: Keep buffer size constant to save memory
    if length(s.recent_pnl) > s.rolling_window * 2
        deleteat!(s.recent_pnl, 1)
    end
end

function simulate_circuit_breaker(ledger::AbstractDataFrame; window=30)
    # Sort by date ensures we process strictly in time
    sort!(ledger, :date)
    
    n = nrow(ledger)
    
    # Vectors for the simulation
    pnl_history = Float64[]
    new_pnl = zeros(Float64, n)
    is_active = fill(true, n) # Track if we are betting
    
    # Rolling sum variables
    current_sum = 0.0
    
    for i in 1:n
        # 1. Calculate Current Drift (based on History up to i-1)
        drift = 0.0
        if length(pnl_history) > 0
            # Look at last 'window' bets
            start_idx = max(1, length(pnl_history) - window + 1)
            drift = mean(pnl_history[start_idx:end])
        else
            drift = 0.1 # Start optimistic
        end
        
        # 2. Apply Rule
        original_pnl = ledger.pnl[i]
        
        if drift < 0.0
            # CIRCUIT BREAKER ACTIVE: We do not bet.
            new_pnl[i] = 0.0
            is_active[i] = false
            
            # Crucial: Do we record a "0" in history? 
            # Ideally yes, to allow drift to recover slowly (decay), 
            # OR we check "Paper Trading" results to decide when to re-enter.
            # Let's assume we "Paper Trade" (observe the market) to see when to re-enter.
            push!(pnl_history, original_pnl) 
        else
            # Betting is allowed
            new_pnl[i] = original_pnl
            is_active[i] = true
            push!(pnl_history, original_pnl)
        end
    end
    
    # Compare cumulative wealth
    ledger.pnl_filtered = new_pnl
    ledger.wealth_original = cumsum(ledger.pnl)
    ledger.wealth_filtered = cumsum(ledger.pnl_filtered)
    
    return ledger
end

function simulate_circuit_breaker(ledger::AbstractDataFrame; window=30, burn_in=50)
    sort!(ledger, :date)
    n = nrow(ledger)
    
    new_pnl = zeros(Float64, n)
    pnl_history = Float64[]
    
    for i in 1:n
        # 1. Calculate Drift
        drift = 0.0
        if length(pnl_history) > 0
            start_idx = max(1, length(pnl_history) - window + 1)
            drift = mean(pnl_history[start_idx:end])
        else
            drift = 0.0 # Neutral start
        end
        
        # 2. The Burn-In Check
        # If we haven't seen enough bets yet, force the system to WAIT.
        if i <= burn_in
            drift = -1.0 # Artificially force "No Bet" status
        end
        
        original_pnl = ledger.pnl[i]
        
        # 3. Apply Rule
        if drift < 0.0
            # BLOCKED (or Warming Up)
            new_pnl[i] = 0.0
            # We still record the result in history so the manager "learns"
            push!(pnl_history, original_pnl)
        else
            # ACTIVE TRADING
            new_pnl[i] = original_pnl
            push!(pnl_history, original_pnl)
        end
    end
    
    ledger.pnl_filtered = new_pnl
    return ledger
end


# Run simulation
sim_results = simulate_circuit_breaker(ledger.df)

# Plot comparison
plot(sim_results.date, sim_results.wealth_original, label="Original (Crash)", color=:red)
plot!(sim_results.date, sim_results.wealth_filtered, label="With Circuit Breaker", color=:green, linewidth=2)

#

function simulate_portfolio_manager(ledger::DataFrame)
    # 1. Group by Market Selection (e.g., over_25, under_25, home, draw)
    # We treat each selection as a separate "Trader"
    groups = groupby(ledger, :selection)
    
    # Store the filtered equity curves
    results = DataFrame()
    
    for df in groups
        selection_name = df.selection[1]
        
        # 2. Run the Circuit Breaker on THIS specific market only
        # This decides if we trade 'over_25' based ONLY on 'over_25' history
        simulated = simulate_circuit_breaker(df; window=30, burn_in=20)
        
        # 3. Add to our results
        append!(results, simulated)
    end
    
    # 4. Re-sort by time to see the aggregate portfolio performance
    sort!(results, :date)
    
    # 5. Calculate the "Managed Portfolio" Wealth
    # This sums up the PnL of the active strategies
    results.portfolio_wealth = cumsum(results.pnl_filtered)
    
    return results
end

# Run the split simulation
managed_ledger = simulate_portfolio_manager(ledger.df)

# Visualization: The "Unbundled" Truth
using Plots

# A. The Cumulative PnL of Overs vs Unders (Raw)
# Shows clearly that Overs are up and Unders are down
overs = subset(ledger.df, :selection => ByRow(x -> occursin("over", String(x))))
unders = subset(ledger.df, :selection => ByRow(x -> occursin("under", String(x))))

p1 = plot(sort(overs, :date).date, cumsum(sort(overs, :date).pnl), 
     label="Overs Only (Raw)", color=:green, title="Overs vs Unders (No Filter)")
plot!(p1, sort(unders, :date).date, cumsum(sort(unders, :date).pnl), 
     label="Unders Only (Raw)", color=:red)

# B. The Managed Portfolio (Automatic Cut-off)
# This shows what happens when the Circuit Breaker kills the Unders but keeps Overs
p2 = plot(managed_ledger.date, managed_ledger.portfolio_wealth, 
     label="Managed Portfolio (Overs Active / Unders Killed)", 
     color=:blue, linewidth=2, title="Meta-Portfolio Result")

plot(p1, p2, layout=(2,1), size=(800, 700))


#######
#
#######
using DataFrames, Statistics, Dates, Printf

function recommend_model(ledger_df; window=50)
    # 1. Sort by date to look at "Recent Form"
    sort!(ledger_df, :date)
    
    # 2. Identify all unique strategies (Model + Parameter + Market)
    # We group by these 3 things to find the specific "sub-strategy"
    groups = groupby(ledger_df, [:model_name, :model_parameters, :selection])
    
    println("=== 🛡️ MODEL SELECTION REPORT (Last $window Bets) ===\n")
    
    recommendations = []

    for gdf in groups
        # Extract strategy details
        m_name = gdf[1, :model_name]
        m_param = gdf[1, :model_parameters]
        market = gdf[1, :selection]
        
        # Get only the last N bets (The Window)
        if nrow(gdf) < window
            continue # Skip if not enough history
        end
        recent = last(gdf, window)
        
        # --- CALCULATE METRICS ---
        total_stake = sum(recent.stake)
        total_profit = sum(recent.pnl)
        roi = (total_profit / total_stake) * 100
        
        win_count = count(recent.is_winner)
        actual_win_rate = win_count / nrow(recent)
        
        # Implied prob = 1 / odds
        avg_implied_prob = mean(1.0 ./ recent.odds)
        edge = actual_win_rate - avg_implied_prob
        
        # --- THE DECISION LOGIC ---
        status = "🔴 STOP"
        if roi > 5.0 && edge > 0.0
            status = "🟢 GO (Full Stake)"
        elseif roi > 0.0
            status = "🟡 CAUTION (Half Stake)"
        end
        
        # Shorten name for display
        short_name = replace(m_name, "GRWNegativeBinomial" => "NegBin", 
                                   "GRWBivariatePoisson" => "BivPois")
        m_p_short = match(r"μ=([0-9\.]+)", m_param).captures[1]

        # Print Report
        if status != "🔴 STOP" # Only show me the good stuff
            Printf.@printf("%s | %s (μ=%s) - %s\n", status, short_name, m_p_short, string(market))
            Printf.@printf("   ├─ ROI: %.1f%%  (Profit: %.2f)\n", roi, total_profit)
            Printf.@printf("   └─ Edge: %.1f%% (WinRate: %.2f vs Odds: %.2f)\n\n", edge*100, actual_win_rate, avg_implied_prob)
            
            push!(recommendations, (model=m_name, params=m_param, market=market))
        end
    end
    
    if isempty(recommendations)
        println("⚠️ WARNING: No models are currently profitable. DO NOT BET this weekend.")
    end
    
    return recommendations
end

# RUN THE CHECK
active_strategies = recommend_model(ledger.df, window=50)

######## Equity Curve Trading" or "Meta-Labeling."

using Turing, StatsPlots, DataFrames, Statistics


# We copy the dataframe to 'analysis_df' to avoid property accessor issues
analysis_df = copy(ledger.df)

# Now we modify 'analysis_df', not 'ledger.df'
analysis_df.date_week = week(analysis_df.date)

analysis_df

using StatsPlots 


over25_df
histogram(over25_df.pnl, 
          title = "PnL Distribution", 
          xlabel = "Profit/Loss", 
          ylabel = "Frequency",
          bins = 200,
          legend = false, 
          xlims=[-0.1,0.1]
          )

roi_data = filter(row -> row.stake > 0, over25_df)
roi_values = roi_data.pnl ./ roi_data.stake


using StatsPlots

histogram(roi_values .* 100, 
    title = "Distribution of ROI per Trade",
    xlabel = "ROI (%)",
    ylabel = "Frequency",
    bins = 50,
    fillcolor = :red,
    linecolor = :white,
    alpha = 0.7,
    legend = false)


# Grouped density or histogram by model_name
@df roi_data density(:pnl ./ :stake .* 100, 
    group = :model_name, 
    title = "ROI Distribution by Model",
    xlabel = "ROI (%)",
    fill = (0, 0.3))

describe(roi_values)


using Distributions
# Fit a Normal just to see the overlap (though it will fit poorly due to the -1 spike)
d_fit = fit(Normal, roi_values)


# Separate the 'Total Loss' group from the 'Returns' group
losses = roi_values[roi_values .== -1.0]
outcomes = roi_values[roi_values .> -1.0]

# Plot density of the outcomes and note the loss percentage
density(outcomes, 
    title = "ROI Distribution (Excluding Total Losses)",
    annotation = (0.5, 1.0, "Total Loss Frequency: $(length(losses)/length(roi_values)*100)%"),
    fill = (0, 0.2, :blue))

# Add a vertical line at 0% to see winners vs losers
vline!([0], color = :black, lw = 2, linestyle = :dash)

describe(analysis_df.pnl)

# Double check it worked (Optional debug print)
println("Columns in analysis_df: ", names(analysis_df))



## 
using Distributions
using Optim # For maximum likelihood estimation

# 1. Separate the data
is_total_loss = roi_values .== -1.0
p_hat = mean(is_total_loss)  # The probability of hitting the -1.0 spike

# 2. Extract the continuous part (the "survivors")
continuous_part = roi_values[.!is_total_loss]

# 3. Fit a distribution to the survivors
# Let's try a Normal first, or a SkewNormal for better fit
dist_cont = fit(Normal, continuous_part) 

println("Spike at -1.0 probability (p): ", round(p_hat, digits=3))
println("Continuous part distribution: ", dist_cont)


using StatsPlots

# Histogram of raw data
histogram(roi_values, bins=50, norm=:pdf, label="Data", alpha=0.5)

# Overlay the continuous fit scaled by (1 - p_hat)
x_range = range(minimum(continuous_part), maximum(continuous_part), length=200)
y_fit = pdf.(dist_cont, x_range) .* (1 - p_hat);

plot!(x_range, y_fit, lw=3, label="Fitted Continuous Part", color=:red)
# Represent the spike visually as a thin high bar
vline!([-1.0], lw=4, label="Point Mass (p=$p_hat)", color=:black)

names(over25_df)

###

using DataFrames, Statistics

# 1. Filter for active bets and calculate ROI
df_active = filter(row -> row.stake > 0, over25_df)
df_active.roi = df_active.pnl ./ df_active.stake

# 2. Calculate mixed distribution parameters per model/params
model_comparison = combine(groupby(df_active, [:model_name, :model_parameters]), 
    # n: Total number of bets
    :roi => length => :n,
    
    # p: Prob of total loss (The Spike at -1.0)
    :roi => (x -> mean(x .== -1.0)) => :p_loss,
    
    # mu: Mean of the continuous part (The Wins)
    :roi => (x -> mean(x[x .> -1.0])) => :mu_win,
    
    # sigma: Std of the continuous part
    :roi => (x -> std(x[x .> -1.0])) => :sigma_win,
    
    # EV: Overall mean ROI
    :roi => mean => :expected_roi
)

# Sort by the best performing model
sort!(model_comparison, :expected_roi, rev = true)

using StatsPlots

@df model_comparison scatter(
    :p_loss, 
    :expected_roi, 
    group = :model_name,
    markersize = :n ./ 10, # Bubble size represents number of bets
    xlabel = "Probability of Total Loss (p)",
    ylabel = "Expected ROI (EV)",
    title = "Model Performance: Risk vs. Reward",
    legend = :outerright
)

#

ds.matches

# 
df_joined = innerjoin(
    df_active, 
    ds.matches[:, [:match_id, :season, :tournament_slug]], 
    on = :match_id
)

model_season_stats = combine(groupby(df_joined, [:season, :model_name, :model_parameters]), 
    :roi => length => :n,
    :roi => (x -> mean(x .== -1.0)) => :p_loss,
    :roi => (x -> mean(x[x .> -1.0])) => :mu_win,
    :roi => mean => :expected_roi
)

# Sort to see the best season-model combinations
sort!(model_season_stats, :expected_roi, rev = true)

@df df_joined boxplot(:season, :roi, 
    group = :model_name, 
    title = "ROI Distribution by Season",
    ylabel = "ROI",
    outliers = false)

using StatsPlots

# Filter for a specific model to keep the comparison clean
target_model = "GRWNegativeBinomial"
sub_df = filter(row -> row.model_name == target_model, df_joined)

# Create a ridge plot to see the ROI distribution "bleeding" into the negative over seasons
@df sub_df ridgeline(:roi, :season, 
    fillalpha = 0.6, 
    title = "ROI Distribution Shift: $target_model",
    xlabel = "ROI (Return on Stake)")

# We use density instead of a histogram for a smoother "curve" comparison
@df sub_df density(:roi, 
    group = :season, 
    fill = (0, 0.2), 
    title = "ROI Density Shift: $target_model",
    xlabel = "ROI (Return on Stake)",
    ylabel = "Probability Density",
    xlims = (-1.1, 1.5), # Focus on the relevant ROI range
    legend = :topright)


@df sub_df violin(:season, :roi, 
    fillalpha=0.5, 
    label="",
    title = "ROI Distribution by Season")

@df sub_df boxplot!(:season, :roi, 
    fillalpha=0.2, 
    width=0.1, 
    label="Quartiles")

### funciton s
function get_market_analysis(df, market::String, sel::Symbol)
    # Subset by market name, selection, and active stakes
    sub_df = subset(df, 
        :market_name => ByRow(==(market)),
        :selection => ByRow(==(sel)),
        :stake => ByRow(>(0.0))
    )
    
    # Calculate ROI immediately so it's ready for plotting
    sub_df.roi = sub_df.pnl ./ sub_df.stake
    
    return sub_df
end

# Usage Examples:
over25_df = get_market_analysis(ledger.df, "OverUnder", :over_25)
btts_df   = get_market_analysis(ledger.df, "BTTS", :btts_yes)
home_df   = get_market_analysis(ledger.df, "1X2", :home)

# A quick way to compare all major markets in your ledger
market_summary = combine(groupby(subset(ledger.df, :stake => ByRow(>(0))), [:market_name, :selection]), 
    :pnl => (p -> mean((p ./ ledger.df.stake[p .!= 0]) .== -1.0)) => :p_loss,
    :pnl => (p -> mean(p ./ ledger.df.stake[p .!= 0])) => :avg_roi,
    nrow => :bet_count
)

# Filter for markets where you've actually placed a significant number of bets
filter!(row -> row.bet_count > 50, market_summary)


function get_global_market_summary(ledger_df, matches_df)
    # 1. Join with matches to get season info (select only necessary columns)
    df_joined = innerjoin(
        subset(ledger_df, :stake => ByRow(>(0))), 
        matches_df[:, [:match_id, :season]], 
        on = :match_id
    )
    
    # 2. Add ROI column
    df_joined.roi = df_joined.pnl ./ df_joined.stake
    
    # 3. Aggregate by Market, Selection, Season, and Model
    summary = combine(groupby(df_joined, [:market_name, :selection, :season, :model_name, :model_parameters]), 
        :roi => length => :n,
        :roi => (x -> mean(x .== -1.0)) => :p_loss,
        :roi => (x -> mean(x[x .> -1.0])) => :mu_win,
        :roi => mean => :expected_roi
    )
    
    return sort(summary, :expected_roi, rev = true)
end

# Execution
global_stats = get_global_market_summary(ledger.df, ds.matches)

significant_stats = subset(global_stats, :n => ByRow(>=(30)))

####

# 1. PREPARE DATA
ledger.df

over25_df = subset(analysis_df, :selection => ByRow(isequal(:over_25)))


# Group ledger by week and calculate ROI for a SPECIFIC strategy (e.g., NegBin Over 2.5)
# weekly_roi = Profit / Stake
weekly_data = combine(groupby(ledger.df, :date_week), 
                      :pnl => sum => :profit, 
                      :stake => sum => :stake)

sort!(weekly_data, :date_week)

# Our observation: Realized ROI (e.g., 0.15, -0.05, ...)
y = weekly_data.profit ./ weekly_data.stake

# 2. THE META-MODEL (Latent Edge GRW)

###
# - subset of league 2 24/25 

dd = subset( ds.matches, :tournament_id => ByRow(in([57, 56])), :season => ByRow(isequal("24/25")))
ids_ref = collect( dd.match_id) 

ledge_1 = subset( ledger.df, :match_id => ByRow(in(ids_ref)))

tearsheet_1 = BayesianFootball.BackTesting.generate_tearsheet(ll)

ll = deepcopy(ledger)

subset!( ll.df, :match_id => ByRow(in(ids_ref)))


for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet_1, :selection => ByRow(isequal(m_name)))
    show(sub)
end

# 4. turing 
