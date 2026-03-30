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
exp_dir = "./data/exp/ablation_study"

println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/ablation_study"; data_dir="./data")
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

exp = loaded_results[7]


# --- MIQ --- 

# 1. Extract Latents
latents_raw = Experiments.extract_oos_predictions(ds, exp)

# 2. Prepare Market Data
# (Assuming Data.prepare_market_data exists in your Data module as you showed)
market_data = Data.prepare_market_data(ds)

# 3. Model Inference (PPD - Posterior Predictive Distribution)
# Generates betting probabilities based on latents
ppd = Predictions.model_inference(latents_raw)


# 4. Merge with Market Data
analysis_df = innerjoin(
    market_data.df,
    ppd.df,
    on = [:match_id, :market_name, :market_line, :selection]
)




using Statistics
using DataFrames

println("=== 5. CALCULATING MARKET-IMPLIED QUANTILES (MIQ) ===")

# 1. Calculate the empirical quantile for a single row
# (How much of our distribution is LESS than the market's fair probability?)
function get_miq(posterior_samples::Vector{Float64}, market_prob::Float64)
    if ismissing(market_prob) || isnan(market_prob)
        return missing
    end
    # Count samples <= market_prob, divide by total samples (usually 300)
    return sum(posterior_samples .<= market_prob) / length(posterior_samples)
end

# 2. Apply it to the entire DataFrame
analysis_df.market_quantile = [
    get_miq(dist, fair_prob) 
    for (dist, fair_prob) in zip(analysis_df.distribution, analysis_df.prob_fair_close)
]

# 3. Aggregate to find structural biases!
# We drop missings so the mean calculates cleanly
clean_analysis = dropmissing(analysis_df, :market_quantile)

market_diagnostics = combine(groupby(clean_analysis, :selection),
    :market_quantile => mean => :mean_quantile,
    :market_quantile => std  => :std_quantile,
    nrow => :sample_size
)

# Sort from Lowest Quantile (Model Overprices) to Highest Quantile (Model Underprices)
sort!(market_diagnostics, :mean_quantile)

println("\n📊 MARKET CALIBRATION DIAGNOSTICS:")
println("0.50 = Perfect agreement with the market.")
println("< 0.50 = Model chronically OVERPRICES this selection vs Market.")
println("> 0.50 = Model chronically UNDERPRICES this selection vs Market.\n")
display(market_diagnostics)


# ---
using Plots
plotlyjs() # Ensure PlotlyJS backend is active

println("=== GENERATING HTML MIQ DIAGNOSTICS ===")

# Ensure the output directory exists
output_dir = "exp/abliation_study/figures/"
mkpath(output_dir)

# --- PLOT 1: The 1X2 Market ---
# Filter for just Home, Draw, Away
df_1x2 = filter(row -> row.selection in [:home, :draw, :away], clean_analysis)

p_1x2 = histogram(df_1x2.market_quantile, 
    group = df_1x2.selection,
    bins = 0.0:0.05:1.0, # 20 bins from 0 to 1
    normalize = :probability,
    alpha = 0.6,
    layout = (3, 1),     # Stack them vertically so they don't overlap
    title = ["Home MIQ" "Draw MIQ" "Away MIQ"],
    legend = false,
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    ylabel = "Frequency",
    size = (800, 900)
);
savefig(p_1x2, joinpath(output_dir, "miq_1x2_distribution.html"))

# --- PLOT 2: The Core Goal Markets ---
# Let's look at Over/Under 2.5
df_goals = filter(row -> row.selection in [:over_25, :under_25], clean_analysis)

p_goals = histogram(df_goals.market_quantile, 
    group = df_goals.selection,
    bins = 0.0:0.05:1.0,
    normalize = :probability,
    alpha = 0.6,
    layout = (2, 1),
    title = ["Over 2.5 MIQ" "Under 2.5 MIQ"],
    legend = false,
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    size = (800, 600)
);
savefig(p_goals, joinpath(output_dir, "miq_goals_distribution.html"))

println("✅ Plots saved! Access them via your browser at:")
println("http://localhost:8080/exp/abliation_study/figures/miq_1x2_distribution.html")


# --- 
using Plots
plotlyjs()

println("=== GENERATING CONDITIONAL MIQ DIAGNOSTICS ===")

# 1. Filter out unresolved matches
# Ensure your is_winner column is boolean or 1/0, and drop missings
df_resolved = dropmissing(clean_analysis, :is_winner)

# 2. Let's look at the Home Win market as our baseline
df_home = filter(row -> row.selection == :home, df_resolved)

# 3. Plot overlapping histograms grouped by the outcome
p_conditional_home = histogram(df_home.market_quantile, 
    group = df_home.is_winner,
    bins = 0.0:0.1:1.0, 
    normalize = :probability,
    alpha = 0.6,
    color = [:red :green], # Red for Losers (0), Green for Winners (1)
    labels = ["Lost (is_winner=0)" "Won (is_winner=1)"],
    title = "Home Win MIQ: Winners vs Losers",
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    ylabel = "Relative Frequency",
    size = (1200, 1200)
);
savefig(p_conditional_home, joinpath(output_dir, "miq_conditional_home.html"))

# 4. (Optional) Run the exact same plot for Away Wins to see why it blew up
df_away = filter(row -> row.selection == :away, df_resolved)
p_conditional_away = histogram(df_away.market_quantile, 
    group = df_away.is_winner,
    bins = 0.0:0.1:1.0, 
    normalize = :probability,
    alpha = 0.6,
    color = [:red :green],
    labels = ["Lost (is_winner=0)" "Won (is_winner=1)"],
    title = "Away Win MIQ: Winners vs Losers",
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    size = (1200, 1200)
);
savefig(p_conditional_away, joinpath(output_dir, "miq_conditional_away.html"))



df_over_15 = filter(row -> row.selection == :over_15, df_resolved)
p_conditional_over_15 = histogram(df_over_15.market_quantile, 
    group = df_over_15.is_winner,
    bins = 0.0:0.1:1.0, 
    normalize = :probability,
    alpha = 0.6,
    color = [:red :green],
    labels = ["Lost (is_winner=0)" "Won (is_winner=1)"],
    title = "Over_15 Win MIQ: Winners vs Losers",
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    size = (1200, 1200)
);
savefig(p_conditional_over_15, joinpath(output_dir, "miq_conditional_over_15.html"))

df_over_25 = filter(row -> row.selection == :over_25, df_resolved)
p_conditional_over_25 = histogram(df_over_25.market_quantile, 
    group = df_over_25.is_winner,
    bins = 0.0:0.1:1.0, 
    normalize = :probability,
    alpha = 0.6,
    color = [:red :green],
    labels = ["Lost (is_winner=0)" "Won (is_winner=1)"],
    title = "Over_25 Win MIQ: Winners vs Losers",
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    size = (900, 900)
);
savefig(p_conditional_over_25, joinpath(output_dir, "miq_conditional_over_25.html"))


df_over_35 = filter(row -> row.selection == :over_35, df_resolved)
p_conditional_over_35 = histogram(df_over_35.market_quantile, 
    group = df_over_35.is_winner,
    bins = 0.0:0.1:1.0, 
    normalize = :probability,
    alpha = 0.6,
    color = [:red :green],
    labels = ["Lost (is_winner=0)" "Won (is_winner=1)"],
    title = "Over_25 Win MIQ: Winners vs Losers",
    xlabel = "Market Implied Quantile (0.0 to 1.0)",
    size = (900, 900)
);
savefig(p_conditional_over_35, joinpath(output_dir, "miq_conditional_over_35.html"))


println("✅ Conditional plots saved to server.")

analysis_df

ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    exp, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)


model_names = model_names[1:12]

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end



# --- layer 2 meta model 
using Statistics
using DataFrames

println("=== RUNNING LAYER 2 MIQ META-MODEL SIMULATION ===")

# 1. Filter for resolved 1X2 matches
df_1x2 = filter(row -> row.market_name == "1X2" && !ismissing(row.is_winner), analysis_df)

# 2. Define the Standalone Base Kelly (Baker-McHale Eq 5 Analytical for speed)
function calc_base_kelly(dist::AbstractVector, odds::Float64)
    if ismissing(odds) || odds <= 1.0 return 0.0 end
    
    p_mean = mean(dist)
    p_var = var(dist)
    b = odds - 1.0
    
    s_star = ((b + 1.0) * p_mean - 1.0) / b
    if s_star <= 0.0 return 0.0 end
    
    term = ((b + 1.0) / b)^2
    k_factor = (s_star^2) / (s_star^2 + term * p_var)
    
    return s_star * k_factor
end

# 3. Define the Meta-Model Logic (The MIQ Gatekeeper)
function calc_bounded_kelly(dist::AbstractVector, odds::Float64, selection::Symbol, miq::Float64, bounds_dict::Dict)
    # Get the raw stake first
    base_stake = calc_base_kelly(dist, odds)
    if base_stake <= 0.0 return 0.0 end
    
    # Lookup safe bounds (Default to 0.0 - 1.0 if not specified)
    lower_bound, upper_bound = get(bounds_dict, selection, (0.0, 1.0))
    
    # The Filter: If MIQ is outside the safe zone, veto the bet!
    if ismissing(miq) || isnan(miq) || miq < lower_bound || miq > upper_bound
        return 0.0 
    end
    
    return base_stake
end

# 4. Define our empirically discovered safe zones based on your charts
safe_bounds = Dict(
    :home => (0.0, 1.0),   # Home is well calibrated, no limits
    :draw => (0.0, 1.0),   # Draw is highly profitable, let it run
    :away => (0.45, 1.0)   # CUT OFF THE TOXIC LEFT TAIL!
)

# 5. Apply the stakes to the DataFrame
df_1x2.base_stake = [calc_base_kelly(d, o) for (d, o) in zip(df_1x2.distribution, df_1x2.odds_close)]
df_1x2.bounded_stake = [calc_bounded_kelly(d, o, s, m, safe_bounds) 
                        for (d, o, s, m) in zip(df_1x2.distribution, df_1x2.odds_close, df_1x2.selection, df_1x2.market_quantile)]

# 6. Calculate PnL (Units)
function get_pnl(stake, odds, is_win)
    if stake <= 0.0 return 0.0 end
    return is_win ? stake * (odds - 1.0) : -stake
end

df_1x2.base_pnl = [get_pnl(s, o, w) for (s, o, w) in zip(df_1x2.base_stake, df_1x2.odds_close, df_1x2.is_winner)]
df_1x2.bounded_pnl = [get_pnl(s, o, w) for (s, o, w) in zip(df_1x2.bounded_stake, df_1x2.odds_close, df_1x2.is_winner)]

# 7. Aggregate and Compare Results
results = combine(groupby(df_1x2, :selection),
    # Base Model Metrics
    :base_stake => (x -> sum(x .> 0)) => :Base_Bets,
    :base_stake => sum => :Base_Turnover,
    :base_pnl => sum => :Base_Profit,
    
    # Layer 2 Bounded Metrics
    :bounded_stake => (x -> sum(x .> 0)) => :Bounded_Bets,
    :bounded_stake => sum => :Bounded_Turnover,
    :bounded_pnl => sum => :Bounded_Profit
)

# Calculate ROI %
results.Base_ROI_Pct = round.((results.Base_Profit ./ results.Base_Turnover) .* 100, digits=2)
results.Bounded_ROI_Pct = round.((results.Bounded_Profit ./ results.Bounded_Turnover) .* 100, digits=2)

# Display the final showdown
display(select(results, :selection, :Base_Bets, :Bounded_Bets, :Base_ROI_Pct, :Bounded_ROI_Pct, :Base_Profit, :Bounded_Profit))


using DataFrames

println("=== RUNNING MIQ BOUNDS OPTIMIZER (GRID SEARCH) ===")

function optimize_market_bounds(df_market::DataFrame, selection::Symbol; min_bets::Int=50)
    # 1. Filter data to only the market we care about
    df_sub = filter(row -> row.selection == selection && !ismissing(row.is_winner), df_market)
    
    # Pre-calculate base stakes to save time in the loop
    df_sub.base_stake = [calc_base_kelly(d, o) for (d, o) in zip(df_sub.distribution, df_sub.odds_close)]
    
    # Prepare an array to hold our grid search results
    results = []

    # 2. Iterate through every possible Lower and Upper bound (Step size 0.05)
    step_size = 0.05
    for lower in 0.0:step_size:0.95
        for upper in (lower + step_size):step_size:1.0
            
            total_profit = 0.0
            total_turnover = 0.0
            bets_placed = 0
            
            # 3. Test this specific (lower, upper) combination against history
            for row in eachrow(df_sub)
                miq = row.market_quantile
                if !ismissing(miq) && miq >= lower && miq <= upper
                    stake = row.base_stake
                    if stake > 0.0
                        profit = row.is_winner ? stake * (row.odds_close - 1.0) : -stake
                        total_profit += profit
                        total_turnover += stake
                        bets_placed += 1
                    end
                end
            end
            
            # 4. Only record it if it passes our anti-overfitting constraint
            if bets_placed >= min_bets
                roi = (total_profit / total_turnover) * 100
                push!(results, (Lower_Bound=round(lower, digits=2), 
                                Upper_Bound=round(upper, digits=2), 
                                Bets=bets_placed, 
                                Profit=round(total_profit, digits=2), 
                                ROI_Pct=round(roi, digits=2)))
            end
        end
    end
    
    # Convert to DataFrame and sort by Total Profit
    res_df = DataFrame(results)
    if nrow(res_df) == 0
        println("No bounds found that meet the minimum bet threshold of $min_bets.")
        return DataFrame()
    end
    
    sort!(res_df, :Profit, rev=true)
    return res_df
end

# Let's run the optimizer on the problematic Away market!
# We demand at least 100 historical bets to prove the edge is robust.
away_optimal = optimize_market_bounds(analysis_df, :draw, min_bets=100)

println("\n🏆 Top 5 Optimal Bounds for Away Market (Min 100 Bets):")
display(first(away_optimal, 5))



# ----
using Dates
using DataFrames
using Statistics

println("=== RUNNING WALK-FORWARD (ONLINE) OPTIMIZATION ===")

# Ensure data is sorted chronologically!
sort!(analysis_df, :date)

function walk_forward_backtest(df::DataFrame, selection::Symbol; train_months=6, test_months=1)
    # Filter for the specific market
    df_sub = filter(row -> row.selection == selection && !ismissing(row.is_winner), df)
    
    # Pre-calculate base stakes
    df_sub.base_stake = [calc_base_kelly(d, o) for (d, o) in zip(df_sub.distribution, df_sub.odds_close)]
    
    # Get the date range
    min_date = minimum(df_sub.date)
    max_date = maximum(df_sub.date)
    
    current_train_start = min_date
    
    total_oos_profit = 0.0
    total_oos_turnover = 0.0
    total_oos_bets = 0
    
    println("Starting Walk-Forward for: $selection")
    println("--------------------------------------------------")
    
    # The Rolling Loop
    while true
        # Define the dynamic windows
        train_end = current_train_start + Month(train_months)
        test_end = train_end + Month(test_months)
        
        if test_end > max_date + Month(1)
            break # We've run out of future data
        end
        
        # 1. Isolate the Training Data (The Past)
        train_df = filter(row -> row.date >= current_train_start && row.date < train_end, df_sub)
        
        # 2. Optimize bounds ONLY on the training data
        # We lower min_bets because it's only a 6 month window, not the whole dataset
        optimal_df = optimize_market_bounds(train_df, selection, min_bets=25) 
        
        # Default bounds if optimizer finds no safe edge
        lower_b, upper_b = 0.0, 1.0 
        if nrow(optimal_df) > 0
            lower_b = optimal_df.Lower_Bound[1]
            upper_b = optimal_df.Upper_Bound[1]
        else
            println("  [$(Dates.monthname(train_end))] Market too noisy. Sitting out this month.")
            # If we find no edge, we set bounds to something impossible so we don't bet
            lower_b, upper_b = -1.0, -1.0 
        end
        
        # 3. Isolate the Test Data (The Unseen Future)
        test_df = filter(row -> row.date >= train_end && row.date < test_end, df_sub)
        
        # 4. Trade the Test Data using the Training Bounds
        month_profit = 0.0
        month_turnover = 0.0
        month_bets = 0
        
        for row in eachrow(test_df)
            miq = row.market_quantile
            if !ismissing(miq) && miq >= lower_b && miq <= upper_b
                stake = row.base_stake
                if stake > 0.0
                    profit = row.is_winner ? stake * (row.odds_close - 1.0) : -stake
                    month_profit += profit
                    month_turnover += stake
                    month_bets += 1
                end
            end
        end
        
        # 5. Log the Out-of-Sample Results
        if lower_b != -1.0
            println("  Testing $(Dates.monthname(train_end)) $(Dates.year(train_end)) | Bounds: [$lower_b - $upper_b] | Bets: $month_bets | PnL: $(round(month_profit, digits=2))")
        end
        
        # Accumulate global stats
        total_oos_profit += month_profit
        total_oos_turnover += month_turnover
        total_oos_bets += month_bets
        
        # Slide the window forward by 1 month
        current_train_start += Month(test_months)
    end
    
    # Final Report
    oos_roi = total_oos_turnover > 0 ? (total_oos_profit / total_oos_turnover) * 100 : 0.0
    println("--------------------------------------------------")
    println("TRUE OUT-OF-SAMPLE RESULTS FOR $selection")
    println("Total Bets: $total_oos_bets")
    println("Total Profit: $(round(total_oos_profit, digits=2)) Units")
    println("True ROI: $(round(oos_roi, digits=2))%")
    println("==================================================\n")
end

# Run it for your markets!
walk_forward_backtest(analysis_df, :away)
walk_forward_backtest(analysis_df, :over_25)
walk_forward_backtest(analysis_df, :draw)
walk_forward_backtest(analysis_df, :home)
walk_forward_backtest(analysis_df, :over_15)

# ---

using Dates
using DataFrames
using Statistics

println("=== RUNNING EXPANDING WINDOW WALK-FORWARD ===")

function expanding_walk_forward(df::DataFrame, selection::Symbol; initial_train_months=6)
    df_sub = filter(row -> row.selection == selection && !ismissing(row.is_winner), df)
    
    # Pre-calculate base stakes to ensure speed
    df_sub.base_stake = [calc_base_kelly(d, o) for (d, o) in zip(df_sub.distribution, df_sub.odds_close)]
    
    min_date = minimum(df_sub.date)
    max_date = maximum(df_sub.date)
    
    # LOCK the start date! This makes it an expanding window.
    train_start = min_date 
    
    total_oos_profit = 0.0
    total_oos_turnover = 0.0
    total_oos_bets = 0
    
    println("Starting Expanding Walk-Forward for: $selection")
    println("--------------------------------------------------")
    
    # The first training window ends X months after the start
    train_end = train_start + Month(initial_train_months)
    
    while train_end <= max_date
        test_start = train_end
        test_end = test_start + Month(1) # We always trade the next 1 month
        
        # 1. Isolate Training Data (From the beginning of time until 'train_end')
        train_df = filter(row -> row.date >= train_start && row.date < train_end, df_sub)
        
        # Dynamic Min Bets: Calculate months of data without triggering the Day->Month error
        months_of_data = (Dates.year(train_end) - Dates.year(train_start)) * 12 + (Dates.month(train_end) - Dates.month(train_start))
        dynamic_min_bets = max(20, round(Int, months_of_data * 5)) # E.g., 5 bets per month average
        
        # 2. Optimize
        optimal_df = optimize_market_bounds(train_df, selection, min_bets=dynamic_min_bets)
        
        lower_b, upper_b = 0.0, 1.0
        if nrow(optimal_df) > 0
            lower_b = optimal_df.Lower_Bound[1]
            upper_b = optimal_df.Upper_Bound[1]
        else
            # If the optimizer fails, we don't sit out. We default to safe, generic bounds!
            lower_b = 0.0
            upper_b = 0.30 
            println("  [$(Dates.monthname(test_start))] Optimizer failed. Using fallback bounds [0.0 - 0.30]")
        end
        
        # 3. Trade the Test Data
        test_df = filter(row -> row.date >= test_start && row.date < test_end, df_sub)
        
        month_profit = 0.0
        month_turnover = 0.0
        month_bets = 0
        
        for row in eachrow(test_df)
            miq = row.market_quantile
            if !ismissing(miq) && miq >= lower_b && miq <= upper_b
                stake = row.base_stake
                if stake > 0.0
                    profit = row.is_winner ? stake * (row.odds_close - 1.0) : -stake
                    month_profit += profit
                    month_turnover += stake
                    month_bets += 1
                end
            end
        end
        
        if lower_b != 0.0 || upper_b != 0.30 # Only print if we found custom bounds
             println("  Testing $(Dates.monthname(test_start)) $(Dates.year(test_start)) | Bounds: [$lower_b - $upper_b] | Bets: $month_bets | PnL: $(round(month_profit, digits=2))")
        end
        
        total_oos_profit += month_profit
        total_oos_turnover += month_turnover
        total_oos_bets += month_bets
        
        # EXPAND the window by moving the train_end forward 1 month
        train_end += Month(1)
    end
    
    oos_roi = total_oos_turnover > 0 ? (total_oos_profit / total_oos_turnover) * 100 : 0.0
    println("--------------------------------------------------")
    println("TRUE OOS RESULTS (EXPANDING WINDOW) FOR $selection")
    println("Total Bets: $total_oos_bets")
    println("Total Profit: $(round(total_oos_profit, digits=2)) Units")
    println("True ROI: $(round(oos_roi, digits=2))%")
    println("==================================================\n")
end

# Fire it up!
expanding_walk_forward(analysis_df, :away)
expanding_walk_forward(analysis_df, :draw)
expanding_walk_forward(analysis_df, :over_15)
expanding_walk_forward(analysis_df, :over_25)


# ----
using Dates
using DataFrames
using Statistics

println("=== RUNNING TIME-WEIGHTED BOUNDS OPTIMIZER ===")

function optimize_time_weighted_bounds(df_market::DataFrame, selection::Symbol, current_date::Date; decay_lambda=0.01, min_weighted_bets=10.0)
    df_sub = filter(row -> row.selection == selection && !ismissing(row.is_winner), df_market)
    
    if nrow(df_sub) == 0 return DataFrame() end

    df_sub.base_stake = [calc_base_kelly(d, o) for (d, o) in zip(df_sub.distribution, df_sub.odds_close)]
    
    # 1. Calculate the Exponential Weight for every historical row relative to "current_date"
    df_sub.days_ago = [Dates.value(current_date - d) for d in df_sub.date]
    df_sub.time_weight =  Base.exp.(-decay_lambda .* df_sub.days_ago)

    results = []
    step_size = 0.05
    
    for lower in 0.0:step_size:0.95
        for upper in (lower + step_size):step_size:1.0
            
            weighted_profit = 0.0
            weighted_turnover = 0.0
            sum_of_weights = 0.0 # This replaces our old 'bets_placed' counter
            
            for row in eachrow(df_sub)
                miq = row.market_quantile
                if !ismissing(miq) && miq >= lower && miq <= upper
                    stake = row.base_stake
                    if stake > 0.0
                        profit = row.is_winner ? stake * (row.odds_close - 1.0) : -stake
                        
                        # Apply the exponential decay penalty to everything!
                        weighted_profit += profit * row.time_weight
                        weighted_turnover += stake * row.time_weight
                        sum_of_weights += row.time_weight
                    end
                end
            end
            
            # 2. Check the Robustness Threshold
            # A sum of weights > 10.0 means we need the equivalent of 10 "brand new" bets,
            # or 50 "really old" bets to trust this parameter combination.
            if sum_of_weights >= min_weighted_bets
                w_roi = (weighted_profit / weighted_turnover) * 100
                push!(results, (Lower_Bound=round(lower, digits=2), 
                                Upper_Bound=round(upper, digits=2), 
                                Weighted_Bets=round(sum_of_weights, digits=2), 
                                Weighted_Profit=round(weighted_profit, digits=3), 
                                Weighted_ROI_Pct=round(w_roi, digits=2)))
            end
        end
    end
    
    res_df = DataFrame(results)
    if nrow(res_df) == 0 return DataFrame() end
    
    # Sort by the time-weighted profit!
    sort!(res_df, :Weighted_Profit, rev=true)
    return res_df
end


function weighted_walk_forward(df::DataFrame, selection::Symbol; initial_train_months=6)
    df_sub = filter(row -> row.selection == selection && !ismissing(row.is_winner), df)
    df_sub.base_stake = [calc_base_kelly(d, o) for (d, o) in zip(df_sub.distribution, df_sub.odds_close)]
    
    min_date = minimum(df_sub.date)
    max_date = maximum(df_sub.date)
    train_start = min_date 
    
    total_oos_profit = 0.0
    total_oos_turnover = 0.0
    total_oos_bets = 0
    
    println("Starting Weighted Walk-Forward for: $selection")
    println("--------------------------------------------------")
    
    train_end = train_start + Month(initial_train_months)
    
    while train_end <= max_date
        test_start = train_end
        test_end = test_start + Month(1) 
        
        train_df = filter(row -> row.date >= train_start && row.date < train_end, df_sub)
        
        # 1. Use the new Time-Weighted Optimizer
        # Pass test_start as the anchor point so it knows how old the data is!
        optimal_df = optimize_time_weighted_bounds(train_df, selection, test_start, decay_lambda=0.01, min_weighted_bets=5.0)
        
        lower_b, upper_b = 0.0, 1.0
        if nrow(optimal_df) > 0
            lower_b = optimal_df.Lower_Bound[1]
            upper_b = optimal_df.Upper_Bound[1]
        else
            lower_b = 0.0
            upper_b = 0.30 
        end
        
        # 2. Trade the Test Data
        test_df = filter(row -> row.date >= test_start && row.date < test_end, df_sub)
        
        month_profit = 0.0
        month_turnover = 0.0
        month_bets = 0
        
        for row in eachrow(test_df)
            miq = row.market_quantile
            if !ismissing(miq) && miq >= lower_b && miq <= upper_b
                stake = row.base_stake
                if stake > 0.0
                    profit = row.is_winner ? stake * (row.odds_close - 1.0) : -stake
                    month_profit += profit
                    month_turnover += stake
                    month_bets += 1
                end
            end
        end
        
        println("  Testing $(Dates.monthname(test_start)) $(Dates.year(test_start)) | Bounds: [$lower_b - $upper_b] | Bets: $month_bets | PnL: $(round(month_profit, digits=2))")
        
        total_oos_profit += month_profit
        total_oos_turnover += month_turnover
        total_oos_bets += month_bets
        
        train_end += Month(1)
    end
    
    oos_roi = total_oos_turnover > 0 ? (total_oos_profit / total_oos_turnover) * 100 : 0.0
    println("--------------------------------------------------")
    println("TRUE OOS RESULTS (TIME-WEIGHTED) FOR $selection")
    println("Total Bets: $total_oos_bets")
    println("Total Profit: $(round(total_oos_profit, digits=2)) Units")
    println("True ROI: $(round(oos_roi, digits=2))%")
    println("==================================================\n")
end

# Run the final test!
weighted_walk_forward(analysis_df, :over_15)
weighted_walk_forward(analysis_df, :draw)

weighted_walk_forward(analysis_df, :over_25)

weighted_walk_forward(analysis_df, :over_35)





# ----
using Statistics
using DataFrames

println("=== RUNNING BIAS-AWARE KELLY PROOF OF CONCEPT ===")

# 1. Define the Comparison Math
function compare_kelly(dist::AbstractVector, odds::Float64, miq::Float64; scaler::Float64=0.20)
    if ismissing(odds) || odds <= 1.0 || ismissing(miq) || isnan(miq)
        return (0.0, 0.0)
    end

    p_mean = mean(dist)
    p_var = var(dist)
    b = odds - 1.0

    # Naive Kelly
    s_star = ((b + 1.0) * p_mean - 1.0) / b
    if s_star <= 0.0
        return (0.0, 0.0)
    end

    term = ((b + 1.0) / b)^2

    # --- STANDARD BAKER-MCHALE (Variance Only) ---
    k_base = (s_star^2) / (s_star^2 + term * p_var)
    base_stake = s_star * k_base

    # --- BIAS-AWARE BAKER-MCHALE (MSE: Variance + Bias^2) ---
    # Calculate how biased this specific bet is
    miq_deviation = abs(miq - 0.50)
    beta = miq_deviation * scaler # The bias estimate
    mse = p_var + (beta^2)
    
    k_bias = (s_star^2) / (s_star^2 + term * mse)
    bias_stake = s_star * k_bias

    return (base_stake, bias_stake)
end

# 2. Filter for resolved bets
df_resolved = filter(row -> !ismissing(row.is_winner), analysis_df)

# 3. Calculate both stakes for every row
stakes = [compare_kelly(d, o, m, scaler=0.25) for (d, o, m) in zip(df_resolved.distribution, df_resolved.odds_close, df_resolved.market_quantile)]

df_resolved.base_stake = first.(stakes)
df_resolved.bias_stake = last.(stakes)

# 4. Calculate PnL for both strategies
function get_pnl(stake, odds, is_win)
    if stake <= 0.0 return 0.0 end
    return is_win ? stake * (odds - 1.0) : -stake
end

df_resolved.base_pnl = [get_pnl(s, o, w) for (s, o, w) in zip(df_resolved.base_stake, df_resolved.odds_close, df_resolved.is_winner)]
df_resolved.bias_pnl = [get_pnl(s, o, w) for (s, o, w) in zip(df_resolved.bias_stake, df_resolved.odds_close, df_resolved.is_winner)]

# 5. Aggregate and compare
# We only count a bet if the stake is larger than 0.5% of the bankroll (0.005)
min_practical_bet = 0.005 

results = combine(groupby(df_resolved, :selection),
    :base_stake => (x -> sum(x .> min_practical_bet)) => :Base_Bets,
    :base_stake => sum => :Base_Turnover,
    :base_pnl => sum => :Base_Profit,

    :bias_stake => (x -> sum(x .> min_practical_bet)) => :Bias_Bets,
    :bias_stake => sum => :Bias_Turnover,
    :bias_pnl => sum => :Bias_Profit
)

# 6. Calculate ROIs
results.Base_ROI = round.((results.Base_Profit ./ results.Base_Turnover) .* 100, digits=2)
results.Bias_ROI = round.((results.Bias_Profit ./ results.Bias_Turnover) .* 100, digits=2)

# 7. Sort to put the most heavily bet markets at the top
sort!(results, :Base_Bets, rev=true)

println("\n📊 BIAS-AWARE VS STANDARD KELLY (Whole Dataset):")
display(select(results, :selection, :Base_Bets, :Bias_Bets, :Base_ROI, :Bias_ROI, :Base_Profit, :Bias_Profit))



# ----

using RxInfer, Dates, DataFrames, Statistics

using RxInfer, Distributions
import RxInfer: @model

@model function calibration_monitor(y, p_model)
    # 1. Step One: Define the starting point
    x[1] ~ Normal(mean = 0.0, precision = 1.0)
    y[1] ~ Bernoulli(probit(p_model[1] + x[1]))
    
    # 2. Step Two: Let the graph build the rest automatically
    for i in 2:length(y)
        # Random walk transition (precision 100.0 = variance 0.01)
        x[i] ~ Normal(mean = x[i-1], precision = 100.0) 
        
        # Observation
        y[i] ~ Bernoulli(probit(p_model[i] + x[i]))
    end
end

# 2. Data Preparation
# Let's look at the toxic :away market from your analysis_df
df_away = filter(row -> row.selection == :away && !ismissing(row.is_winner), analysis_df)
sort!(df_away, :date)

y_data = Int.(df_away.is_winner)
p_data = df_away.prob_fair_close # Your model's 'Fair' probability

# 3. Run Inference (Message Passing)
# This is nearly instantaneous compared to Turing MCMC
results = infer(
    model = calibration_monitor(p_model = p_data),
    data  = (y = y_data,),
    iterations = 10, # Variational iterations
    free_energy = true
)

# 4. Extract the Moving Bias
# x_means will show you exactly how your model's bias drifted over the 374 bets
x_means = mean.(results.posteriors[:x])
x_stds  = std.(results.posteriors[:x])

println("=== RxInfer MONITORING COMPLETE ===")
println("Final Estimated Bias for :away market: $(round(x_means[end], digits=3))")

using RxInfer, Distributions, StatsFuns
import RxInfer: @model

# 1. Prepare Data & Convert to Log-Odds
# We use the :away data we isolated earlier
y_data = Int.(df_away.is_winner)

# Safely convert your probabilities to log-odds so they can't exceed 100%
logit_p_data = logit.(clamp.(df_away.prob_fair_close, 0.01, 0.99))

# 2. Define the Model
@model function calibration_monitor(y, logit_p)
    # Step 1: Initial state
    x[1] ~ Normal(mean = 0.0, precision = 1.0)
    y[1] ~ Bernoulli(logistic(logit_p[1] + x[1]))
    
    # Step 2: The Online Learner Loop
    for i in 2:length(y)
        # Random walk transition (Precision 100 means it doesn't jump too wildly)
        x[i] ~ Normal(mean = x[i-1], precision = 100.0) 
        
        # Observation using the safe logistic link
        y[i] ~ Bernoulli(logistic(logit_p[i] + x[i]))
    end
end

# 3. Run Inference
println("Running RxInfer Message Passing...")
results = infer(
    model = calibration_monitor(logit_p = logit_p_data),
    data  = (y = y_data,),
    iterations = 10
)

# 4. Extract the moving bias
x_means = mean.(results.posteriors[:x])

println("=== RxInfer MONITORING COMPLETE ===")
println("Final Estimated Logit Bias for :away market: $(round(x_means[end], digits=3))")


using RxInfer, Distributions, StatsFuns
import RxInfer: @model, @meta

# 1. Define the Model (Keep it perfectly clean)
@model function calibration_monitor(y, logit_p)
    x[1] ~ Normal(mean = 0.0, precision = 1.0)
    y[1] ~ Bernoulli(logistic(logit_p[1] + x[1]))
    
    for i in 2:length(y)
        x[i] ~ Normal(mean = x[i-1], precision = 100.0) 
        y[i] ~ Bernoulli(logistic(logit_p[i] + x[i]))
    end
end

# 2. Define the Meta Block (The instruction manual for non-linear math)
@meta function calibration_meta()
    # This tells the engine: "Whenever you see a 'logistic' node in this graph, 
    # approximate it using Linearization."
    logistic() -> DeltaMeta(method = Linearization())
end

# 3. Run Inference
println("Running RxInfer Message Passing...")
results = infer(
    model = calibration_monitor(logit_p = logit_p_data),
    data  = (y = y_data,),
    meta  = calibration_meta(),
    iterations = 10,
    options = (limit_stack_depth = 500,) # <-- THE FIX: Tells Julia not to panic on deep graphs
)

# 4. Extract the moving bias
x_means = mean.(results.posteriors[:x])

println("=== RxInfer MONITORING COMPLETE ===")
println("Final Estimated Logit Bias for :away market: $(round(x_means[end], digits=3))")


using RxInfer, Distributions, StatsFuns
import RxInfer: @model, @meta, @initialization

# 1. Define the Initial Guess for the Engine
my_init = @initialization begin
    # 'q' stands for the variational posterior. We guess it starts at 0.
    q(x) = NormalMeanPrecision(0.0, 1.0)
end

# 2. Run Inference (with the init block)
println("Running RxInfer Message Passing...")
results = infer(
    model = calibration_monitor(logit_p = logit_p_data),
    data  = (y = y_data,),
    meta  = calibration_meta(),
    initialization = my_init,     # <-- THE FIX: Gives the engine its starting point
    iterations = 10,
    options = (limit_stack_depth = 500,)
)

# 3. Extract the moving bias
x_means = mean.(results.posteriors[:x])

println("=== RxInfer MONITORING COMPLETE ===")
println("Final Estimated Logit Bias for :away market: $(round(x_means[end], digits=3))")


using RxInfer, Distributions, StatsFuns
import RxInfer: @model, @meta, @initialization

# 1. The Corrected Initialization Block
my_init = @initialization begin
    # We must initialize BOTH the marginals (q) and the messages (μ) for the array x
    q(x) = NormalMeanPrecision(0.0, 1.0)
    μ(x) = NormalMeanPrecision(0.0, 1.0)
end

# 2. Run Inference
println("Running RxInfer Message Passing...")
results = infer(
    model = calibration_monitor(logit_p = logit_p_data),
    data  = (y = y_data,),
    meta  = calibration_meta(),
    initialization = my_init,
    iterations = 10,
    options = (limit_stack_depth = 500,)
)

# 3. Extract
x_means = mean.(results.posteriors[:x])
println("Final Estimated Logit Bias: $(round(x_means[end], digits=3))")

# ---
using StatsFuns, Statistics

function track_moving_bias(y_actual, p_model; learning_rate=0.05)
    n = length(y_actual)
    moving_bias = zeros(n)
    current_bias = 0.0 # Start with the assumption that the model is perfect (Bias = 0)
    
    for i in 1:n
        # 1. Record our current belief about the bias BEFORE the match starts
        moving_bias[i] = current_bias
        
        # 2. Calculate what our prediction *should* be, given our known bias
        safe_p = clamp(p_model[i], 0.01, 0.99)
        adjusted_p = logistic(logit(safe_p) + current_bias)
        
        # 3. Observe the result and calculate the error 
        # (If y is 1 and we predicted 0.6, error is +0.4 -> we need to increase bias)
        error = y_actual[i] - adjusted_p
        
        # 4. Update the bias state for the NEXT match
        current_bias += learning_rate * error
    end
    
    return moving_bias
end

# --- RUN IT ON YOUR DATA ---
# Assuming you still have your df_away filtered
y_data = Int.(df_away.is_winner)
p_data = df_away.prob_fair_close

# Run the tracker
bias_trend = track_moving_bias(y_data, p_data, learning_rate=0.05)

println("=== PURE JULIA TRACKER COMPLETE ===")
println("Final Estimated Logit Bias for :away market: $(round(bias_trend[end], digits=3))")


# ---
using DataFrames, Statistics, StatsFuns

println("=== RUNNING CAUSAL BIAS-AWARE BACKTEST ===")

# 1. The Pure Julia Causal Bias Tracker (No look-ahead leak)
function track_moving_bias(y_actual, p_model; learning_rate=0.03)
    n = length(y_actual)
    moving_bias = zeros(n)
    current_bias = 0.0 
    
    for i in 1:n
        # Record belief BEFORE observing the match result
        moving_bias[i] = current_bias
        
        # Observe and update for the NEXT match
        safe_p = clamp(p_model[i], 0.01, 0.99)
        adjusted_p = logistic(logit(safe_p) + current_bias)
        error = y_actual[i] - adjusted_p
        current_bias += learning_rate * error
    end
    return moving_bias
end

# 2. The Kelly Stake Calculator
function compare_stakes(dist, odds, current_logit_bias)
    if ismissing(odds) || odds <= 1.0 
        return (0.0, 0.0) 
    end
    
    p_mean = clamp(mean(dist), 0.01, 0.99)
    p_var = var(dist)
    b = odds - 1.0
    
    s_star = ((b + 1.0) * p_mean - 1.0) / b
    if s_star <= 0.0 return (0.0, 0.0) end
    
    term = ((b + 1.0) / b)^2
    
    # --- Standard Baker-McHale ---
    k_base = (s_star^2) / (s_star^2 + term * p_var)
    base_stake = s_star * k_base
    
    # --- Bias-Aware Baker-McHale ---
    # Convert the logit bias into a direct Probability Penalty (Beta)
    p_adjusted = logistic(logit(p_mean) + current_logit_bias)
    prob_bias_penalty = abs(p_mean - p_adjusted)
    
    # Upgrade Variance to Mean Squared Error (Variance + Bias^2)
    mse = p_var + (prob_bias_penalty^2)
    k_bias = (s_star^2) / (s_star^2 + term * mse)
    bias_stake = s_star * k_bias
    
    return (base_stake, bias_stake)
end

# 3. Main Backtesting Loop
df_eval = filter(row -> !ismissing(row.is_winner) && !ismissing(row.odds_close), analysis_df)
all_results = DataFrame()

for market in unique(df_eval.selection)
    # SORT BY DATE: This is the ultimate protection against data leaks
    df_mkt = sort(filter(row -> row.selection == market, df_eval), :date)
    
    if nrow(df_mkt) < 20
        continue # Skip ghost markets
    end
    
    y_vals = Int.(df_mkt.is_winner)
    p_vals = df_mkt.prob_fair_close
    
    # Run the causal tracker
    bias_trend = track_moving_bias(y_vals, p_vals, learning_rate=0.03)
    
    # Calculate stakes match-by-match
    stakes = [compare_stakes(d, o, b) for (d, o, b) in zip(df_mkt.distribution, df_mkt.odds_close, bias_trend)]
    
    base_stakes = first.(stakes)
    bias_stakes = last.(stakes)
    
    # Compute PnL (Bet only if stake > 0.5% of bankroll)
    min_bet = 0.005 
    
    base_pnl = [s > min_bet ? (w ? s*(o-1) : -s) : 0.0 for (s, o, w) in zip(base_stakes, df_mkt.odds_close, df_mkt.is_winner)]
    bias_pnl = [s > min_bet ? (w ? s*(o-1) : -s) : 0.0 for (s, o, w) in zip(bias_stakes, df_mkt.odds_close, df_mkt.is_winner)]
    
    push!(all_results, (
        Market = market,
        Base_Bets = sum(base_stakes .> min_bet),
        Bias_Bets = sum(bias_stakes .> min_bet),
        Base_Profit = sum(base_pnl),
        Bias_Profit = sum(bias_pnl),
        Base_Turnover = sum(base_stakes[base_stakes .> min_bet]),
        Bias_Turnover = sum(bias_stakes[bias_stakes .> min_bet])
    ))
end

# 4. Format and Display
all_results.Base_ROI = round.((all_results.Base_Profit ./ all_results.Base_Turnover) .* 100, digits=2)
all_results.Bias_ROI = round.((all_results.Bias_Profit ./ all_results.Bias_Turnover) .* 100, digits=2)

sort!(all_results, :Base_Bets, rev=true)
display(select(all_results, :Market, :Base_Bets, :Bias_Bets, :Base_ROI, :Bias_ROI, :Base_Profit, :Bias_Profit))
