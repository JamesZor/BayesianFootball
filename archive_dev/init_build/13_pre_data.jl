using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics

#########

data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(data_store)


odds = data_store.odds

odds.open_close_ratio = ( odds.decimal_odds .- odds.initial_decimal ) ./ odds.decimal_odds


describe(odds.open_close_ratio)

using StatsPlots
using DataFramesMeta


@rsubset(odds, abs(:open_close_ratio) >= 1)
histogram(odds.open_close_ratio)


outliers = @rsubset(odds, :open_close_ratio < -1.0)
println("Count of extreme outliers: ", nrow(outliers))

for g in groupby(outliers, :season_id)
  n = nrow(g) 
  println("season : $(unique(g.season_id)) = $n")
end

# 1. Group the entire dataset by season
gdf = groupby(odds, :season_id)

# 2. Calculate Total Rows vs. Bad Rows for each season
season_stats = combine(gdf, 
    nrow => :total_rows,
    :open_close_ratio => (x -> count(v -> v < -1.0, x)) => :outlier_count
)

# 3. Calculate the Percentage
transform!(season_stats, 
    [:outlier_count, :total_rows] => ByRow((n, t) -> n / t * 100) => :error_pct
)

# 4. Sort by the worst offenders (highest percentage)
sort!(season_stats, :error_pct, rev=true)



outliers


# 1. Define what constitutes a "Corrupted Record"
# A ratio < -1.0 implies the odds dropped > 50% (e.g. 4.0 -> 2.0), which is our flag.
is_corrupt(ratio) = ratio < -1.0 || ratio > 0.6

# 2. Identify (Match, MarketGroup) pairs that contain at least one corruption
# We group by Match AND Market Group to isolate the specific bad feed (e.g., Match Goals)
corrupted_groups = @chain odds begin
    @rsubset(is_corrupt(:open_close_ratio))
    select(:match_id, :market_group) 
    unique()
    # Create a "key" column for easy filtering later
    @rtransform(:bad_key = (:match_id, :market_group))
end

# 3. Create a set of keys for fast lookup
bad_keys_set = Set(corrupted_groups.bad_key)

# 4. Filter the original dataset
# We keep a row ONLY if its (match, group) pair is NOT in the bad set
odds_clean = @rsubset(odds, !((:match_id, :market_group) in bad_keys_set))

# Verification
println("Original Rows: ", nrow(odds))
println("Cleaned Rows:  ", nrow(odds_clean))


###
using ProgressMeter

function generate_kelly_analysis_df(
    model, 
    predict_config, 
    oos_results::Dict, 
    data_store::BayesianFootball.Data.DataStore
)
    rows = []
    skipped_matches = 0
    
    # Pre-fetch match metadata for faster lookup
    meta_lookup = Dict(r.match_id => (league_id=r.tournament_id, match_week=r.match_week) 
                       for r in eachrow(data_store.matches))

    # List of markets to process (explicitly filters for what you asked)
    # Or use keys(preds) to get everything available
    target_markets = [:over_05, :under_05, :btts_yes, :btts_no, :over_35, :under_35, 
                      :over_45, :under_45, :over_15, :under_15, :over_25, :under_25, 
                      :home, :draw, :away]

    @showprogress desc="Calculating Kelly Strategies..." for (match_id, params) in oos_results
        
        # --- 1. Data Fetching ---
        local open_odds, close_odds, results
        try
            open_odds, close_odds, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, data_store.odds)
        catch e
            skipped_matches += 1
            continue 
        end

        # --- 2. Prediction ---
        preds = BayesianFootball.Predictions.predict_market(model, predict_config, params...)
        meta = get(meta_lookup, match_id, (league_id="Unknown", match_week=0))

        # --- 3. Market Iteration ---
        for market in keys(preds)
            # Skip if not in our target list (optional, remove if you want ALL markets)
            if !(market in target_markets) continue end

            if haskey(open_odds, market) && haskey(results, market)
                
                # Inputs
                chain = getproperty(preds, market)
                o_open = getproperty(open_odds, market)
                # Handle missing close odds
                o_close = haskey(close_odds, market) ? getproperty(close_odds, market) : missing
                outcome = getproperty(results, market)
                
                # Stats
                prob_mean = mean(chain)
                prob_std = std(chain)
                
                # --- KELLY CALCULATIONS ---
                
                # A. Raw / Plug-in
                k_raw = BayesianFootball.Signals.kelly_fraction(o_open, prob_mean)
                
                # B. Analytical (Eq 5) - Fast Approx
                k_analytical = BayesianFootball.Signals.calc_analytical_shrinkage(chain, o_open)
                
                # C. Bayes Optimal (Eq 2) - Slow Integration
                # Only run if we have a raw edge, otherwise it's 0.0 anyway
                k_bayes = 0.0
                if k_raw > 0.0
                     k_bayes = BayesianFootball.Signals.bayesian_kelly(chain, o_open)
                end

                # Push Row
                push!(rows, (;
                    match_id = match_id,
                    tournament_id = meta.league_id,
                    match_week = meta.match_week,
                    market = market,
                    
                    # Odds Data
                    odds_open = o_open,
                    odds_close = o_close,
                    prob_model = prob_mean,
                    prob_sigma = prob_std,
                    
                    # Result
                    result = outcome,
                    
                    # Kelly Fractions (0.0 to 1.0)
                    kelly_raw = k_raw,
                    kelly_analytical = k_analytical,
                    kelly_bayes = k_bayes,
                    
                    # Metadata for filtering
                    edge = (prob_mean * o_open) - 1.0
                ))
            end
        end
    end

    if skipped_matches > 0
        println("⚠️ Warning: Skipped $skipped_matches matches due to missing odds.")
    end

    return DataFrame(rows)
end



function calculate_equity_curve(df::DataFrame; bankroll=1000.0, max_stake=0.25)
    # We create new columns for the running bankroll of each strategy
    # Sort by date or match_id to ensure correct time series
    sdf = sort(df, :match_week) 
    
    # Initialize bankrolls
    curr_raw = bankroll
    curr_anal = bankroll
    curr_bayes = bankroll
    curr_flat = bankroll
    
    # History vectors
    hist_raw = Float64[]
    hist_anal = Float64[]
    hist_bayes = Float64[]
    hist_flat = Float64[]
    
    for row in eachrow(sdf)
        odds = row.odds_open
        # Outcome: 1.0 if Win, -1.0 if Loss
        r_mult = row.result ? (odds - 1.0) : -1.0
        
        # --- Strategy 1: Raw Kelly ---
        # Cap stake at max_stake (e.g., 5%)
        s_raw = min(row.kelly_raw, max_stake) * curr_raw
        curr_raw += s_raw * r_mult
        
        # --- Strategy 2: Analytical Shrinkage ---
        s_anal = min(row.kelly_analytical, max_stake) * curr_anal
        curr_anal += s_anal * r_mult
        
        # --- Strategy 3: Bayes Optimal ---
        s_bayes = min(row.kelly_bayes, max_stake) * curr_bayes
        curr_bayes += s_bayes * r_mult

        # --- Strategy 4: Flat Stake (e.g. 1% if edge > 0) ---
        s_flat = (row.edge > 0) ? (curr_flat * 0.01) : 0.0
        curr_flat += s_flat * r_mult
        
        push!(hist_raw, curr_raw)
        push!(hist_anal, curr_anal)
        push!(hist_bayes, curr_bayes)
        push!(hist_flat, curr_flat)
    end
    
    # Return a summary DataFrame for plotting
    return DataFrame(
        idx = 1:length(hist_raw),
        raw_bank = hist_raw,
        analytical_bank = hist_anal,
        bayes_bank = hist_bayes,
        flat_bank = hist_flat
    )
end


"""
Calculates the Maximum Drawdown (MDD) of a time series.
Returns a negative percentage (e.g., -0.45 for a 45% drop).
"""
function calculate_mdd(curve::Vector{Float64})
    peak = curve[1]
    max_drawdown = 0.0
    
    for val in curve
        if val > peak
            peak = val
        end
        dd = (val - peak) / peak
        if dd < max_drawdown
            max_drawdown = dd
        end
    end
    return max_drawdown
end

using Printf

function display_equity_summary(equity_df::DataFrame; initial_bank=100.0)
    strategies = [
        (:raw_bank, "Raw Kelly"),
        (:analytical_bank, "Analytical (Eq5)"),
        (:bayes_bank, "Bayes Optimal"),
        (:flat_bank, "Flat Stake")
    ]

    printstyled("══════════════════════════════════════════════════════════════════════\n", color=:blue)
    printstyled(" STRATEGY PERFORMANCE SUMMARY \n", bold=true, color=:white)
    printstyled("══════════════════════════════════════════════════════════════════════\n", color=:blue)
    
    # Header
    @printf("%-18s | %-10s | %-10s | %-10s | %-10s\n", 
        "Strategy", "Final", "ROI %", "Min Bank", "Max DD %")
    println("-"^70)

    for (col_sym, label) in strategies
        if !hasproperty(equity_df, col_sym) continue end
        
        curve = equity_df[!, col_sym]
        
        # Metrics
        final_val = curve[end]
        roi = (final_val - initial_bank) / initial_bank
        min_val = minimum(curve)
        mdd = calculate_mdd(curve) # This will be negative

        # Formatting Colors
        roi_color = roi >= 0 ? :green : :red
        
        # Render
        printstyled(@sprintf("%-18s | ", label), color=:white)
        printstyled(@sprintf("%10.2f | ", final_val), color=:white)
        printstyled(@sprintf("%9.2f%% | ", roi * 100), color=roi_color)
        
        # Highlight Risk of Ruin scenarios (Min Bank < 20% of start)
        min_color = min_val < (initial_bank * 0.2) ? :red : :white
        printstyled(@sprintf("%10.2f | ", min_val), color=min_color)
        
        # Highlight severe drawdowns (> 50%)
        dd_color = mdd < -0.5 ? :red : :yellow
        printstyled(@sprintf("%9.2f%%\n", mdd * 100), color=dd_color)
    end
    printstyled("══════════════════════════════════════════════════════════════════════\n", color=:blue)
    
    # Contextual Analysis based on your specific data patterns
    best_strat = strategies[argmax([equity_df[end, s[1]] for s in strategies])][2]
    
    println()
    printstyled("Analysis:\n", color=:cyan, bold=true)
    println(" • Winner: The '$best_strat' strategy ended with the highest wealth.")
    
    # Check for Overconfidence
    raw_mdd = calculate_mdd(equity_df.raw_bank)
    if raw_mdd < -0.70
        printstyled(" • WARNING: Raw Kelly suffered a $(round(raw_mdd*100, digits=1))% drawdown.\n", color=:light_red)
        println("   This indicates your model probabilities are likely overconfident/uncalibrated.")
        println("   The 'Shrinkage' methods (Bayes/Analytical) are successfully buffering this risk.")
    end
end



dss = BayesianFootball.Data.DataStore(
    df,
    odds_clean,
    data_store.incidents
)


predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )
# 1. Generate the detailed data

dss = BayesianFootball.Data.DataStore(
    ds.matches,
    odds_clean,
    ds.incidents
)

df_analysis = generate_kelly_analysis_df(model, predict_config, all_oos_results, dss)

# 2. Inspect a specific market type (e.g., Home Wins)
home_only = filter(row -> row.market == :over_25 , df_analysis)
ghosts = filter(row -> row.odds_open > 3.0, home_only)

# 3. Run the simulation on just Home bets
equity = calculate_equity_curve(home_only, bankroll=100.0, max_stake=0.2)

display_equity_summary(equity)

# 4. Plot
using StatsPlots

plot(equity.idx, equity.raw_bank, label="Raw Kelly", lw=2)
plot!(equity.idx, equity.bayes_bank, label="Bayes Shrunk", lw=2)
plot!(equity.idx, equity.flat_bank, label="Flat Stake", lw=2, linestyle=:dash)


describe(equity[:, [:raw_bank, :analytical_bank, :bayes_bank, :flat_bank]])



histogram(home_only.edge, 
    title="Distribution of Model Edge",
    label="Edge",
    xlabel="Predicted Edge",
    bins=50
)

describe(home_only.edge)


using Random, Statistics

function run_monte_carlo(df, kelly_key, n_sims=10_000)
    final_banks = Float64[]
    busts = 0
    
    # We will simulate 100 different timelines
    for i in 1:n_sims
        # Shuffle the order of the matches
        shuffled_df = df[shuffle(1:nrow(df)), :]
        
        # Recalculate equity on this new timeline
        # We use the raw_kelly logic manually here for speed
        current_bank = 100.0
        
        for row in eachrow(shuffled_df)
            stake_pct = row[kelly_key] # The stake % creates the volatility
            
            # Cap stake at 0.2 (20%) as you did before
            stake_pct = min(stake_pct, 0.2)
            
            if stake_pct > 0
                wager = current_bank * stake_pct
                if row.result # If won
                    current_bank += wager * (row.odds_open - 1)
                else # If lost
                    current_bank -= wager
                end
            end
            
            # Check for Bust (Bankroll < 5.0)
            if current_bank < 5.0
                current_bank = 0.0
                break
            end
        end
        
        push!(final_banks, current_bank)
        if current_bank == 0.0
            busts += 1
        end
    end
    
    return final_banks, busts
end

# Run the simulation
sim_results, num_busts = run_monte_carlo(home_only, "kelly_bayes")

# Display the reality check
println("--- Monte Carlo Stress Test (100 Runs) ---")
println("Times you went Bust (Zero Money): ", num_busts, "/100")
println("Average Final Bank:               ", mean(sim_results))
println("Median Final Bank:                ", median(sim_results))
println("Worst Case Scenario:              ", minimum(sim_results))


sim_results, num_busts = run_monte_carlo(home_only, "kelly_raw")

# Display the reality check
println("--- Monte Carlo Stress Test (100 Runs) ---")
println("Times you went Bust (Zero Money): ", num_busts / length(sim_results) * 100)
println("Average Final Bank:               ", mean(sim_results))
println("Median Final Bank:                ", median(sim_results))
println("Worst Case Scenario:              ", minimum(sim_results))


# 1. Bin data
home_only.prob_bin = round.(home_only.prob_model, digits=1)

# 2. Aggregate
calibration = combine(groupby(home_only, :prob_bin), 
    nrow => :count,
    :prob_model => mean => :avg_predicted_prob,
    :result => mean => :actual_win_rate
)

# 3. Filter noise
calibration = filter(row -> row.count > 5, calibration)
sort!(calibration, :prob_bin)

# 4. Plot
plot(calibration.avg_predicted_prob, calibration.actual_win_rate, 
    seriestype=:scatter, 
    label="Your Model", legend=:bottomright,
    xlabel="Model Says (Predicted)", ylabel="Reality (Actual)",
    title="Calibration: The Reason Raw Kelly Busts",
    xlims=(0,1), ylims=(0,1), aspect_ratio=:equal, 
    color=:red, markersize=6
)
plot!([0,1], [0,1], label="Perfect Truth", line=(:dash, :gray, 2))


using DataFrames, Statistics, Printf

# 1. Helper Function to Calculate ECE
function calculate_ece(probs, results, n_bins=10)
    # Create DataFrame
    df = DataFrame(prob = probs, outcome = results)
    
    # Binning
    df.bin = round.(df.prob, digits=1)
    
    # Aggregate
    agg = combine(groupby(df, :bin), 
        nrow => :count,
        :prob => mean => :avg_prob,
        :outcome => mean => :actual_rate
    )
    
    # ECE Calculation: Weighted Average of Absolute Error
    # ECE = Σ (count/total) * |avg_prob - actual_rate|
    total_rows = sum(agg.count)
    ece = sum(agg.count ./ total_rows .* abs.(agg.avg_prob .- agg.actual_rate))
    
    return ece, agg
end

# 2. Prepare the Probabilities
# Note: We use 1/Odds. This implies the "Bookie Probability" (including the margin/vig).
# This is the hurdle you must clear.
probs_model = home_only.prob_model
probs_open  = 1.0 ./ home_only.odds_open
probs_close = 1.0 ./ home_only.odds_close

# 3. Calculate ECE for all three
ece_model, _ = calculate_ece(probs_model, home_only.result)
ece_open, _  = calculate_ece(probs_open, home_only.result)
ece_close, _ = calculate_ece(probs_close, home_only.result)

# 4. Display the "Scorecard"
function display_ece()
  println("══════════════════════════════════════")
  println("      CALIBRATION SCORECARD (ECE)     ")
  println("      (Lower is Better)               ")
  println("══════════════════════════════════════")
  @printf("Model ECE:         %.4f (%.1f%% error)\n", ece_model, ece_model*100)
  @printf("Market Open ECE:   %.4f (%.1f%% error)\n", ece_open, ece_open*100)
  @printf("Market Close ECE:  %.4f (%.1f%% error)\n", ece_close, ece_close*100)
  println("══════════════════════════════════════") 
end 

display_ece()

# 5. Interpretation
if ece_model < ece_close
    println("✅ Incredible! Your model is better calibrated than the Closing Line.")
else
    println("⚠️ The Market is better calibrated. Use Bayes/Shrinkage to fix this.")
end


# 1. Filter: Keep only rows where Model Probability > 0.25
# This removes the noisy 0.2 bin but keeps the profitable 0.3/0.4 bins
high_conviction = filter(row -> row.prob_model > 0.25, home_only)

println("Original Bets: ", nrow(home_only))
println("Filtered Bets: ", nrow(high_conviction))

# 2. Run the Equity Curve on this safer subset
equity_safe = calculate_equity_curve(high_conviction, bankroll=100.0, max_stake=0.2)

# 3. Compare Results
display_equity_summary(equity_safe)
