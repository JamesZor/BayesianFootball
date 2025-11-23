using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics


# --- Phase 1: Globals (D, M, G) --- (Same as before)
data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

# filter for one season for quick training
df = filter(row -> row.season=="24/25", data_store.matches)

# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)

# 1. Define your "special cases" mapping
split_map = Dict(37 => 1, 38 => 2, 39 => 3)

# 2. Use get() with a default value of 0
#    We use Ref(split_map) to tell Julia to treat the Dict as a single object
#    and not try to broadcast over its elements.
ds.matches.split_col = get.(Ref(split_map), ds.matches.match_week, 0);

# large v2 
ds.matches.split_col = max.(0, ds.matches.match_week .- 14);






splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

# --- Phase 3: Define Training Configuration ---
# Sampler Config (Choose one)
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct

# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

# training_config_limited = TrainingConfig(sampler_conf, strategy_parallel_limited)
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# Then run:
# results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)
# save and load 
#
# JLD2.save_object("training_results_large.jld2", results)

# results = JLD2.load_object("training_results.jld2")
results = JLD2.load_object("training_results_large.jld2")

### get out of sample data - chains 
# 1. Define the column you want to split on
#    (You can change this to :round, :week, etc. later)
split_col_name = :split_col

# 2. Get all unique split keys (e.g., [0, 1, 2, 3])
all_splits = sort(unique(ds.matches[!, split_col_name]))

# 3. Define the splits you want to *predict* (e.g., [1, 2, 3])
#    We skip the first key (0), as it was for training the first model
prediction_split_keys = all_splits[2:end] 

# 4. Group the data ONCE
grouped_matches = groupby(ds.matches, split_col_name)

# 5. Create the vector of DataFrames (as efficient SubDataFrame views)
#    This is the new argument for your function
dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]


# --- 6. Call your new function ---
# It's now much cleaner and more flexible
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results
)


#################################################
# Dev area v2  kelly util 
#################################################

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

match_id = rand(keys(all_oos_results))
r1 =  all_oos_results[match_id]
subset( ds.matches, :match_id => ByRow(isequal(match_id)))

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

open, close, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)
open[:over_25]
close[:over_25]
model_odds[:over_25]
results[:over_25]

using StatsPlots
kell = BayesianFootball.Signals.kelly_fraction(open[:home], match_predict[:home])

BayesianFootball.Signals.kelly_positive(kell, 0.0)

density(kell)

using Optim, Statistics

"""
Calculates the Optimal 'Shrunken' Kelly Stake using the full posterior.
Implements the logic of Baker & McHale (2013) Eq. 2 via Monte Carlo.
"""
function optimize_bayesian_kelly(chain_probs::Vector{Float64}, offered_odds::Float64)
    # b in the paper is "fractional odds" (Decimal Odds - 1)
    b = offered_odds - 1.0
    
    # The Utility Function (Log Growth)
    # If we bet fraction 'f', and the result is Win (1) or Loss (0):
    # Wealth grows by (1 + b*f) on Win
    # Wealth shrinks to (1 - f) on Loss
    function expected_utility(f)
        # Constraint: We cannot bet more than we have, or negative amounts
        if f < 0.0 || f >= 1.0
            return -Inf 
        end

        # We calculate the utility for EVERY sample in the chain
        # "p" is one sample from your posterior (e.g., 0.54)
        utilities = [
            p * log(1 + b * f) + (1 - p) * log(1 - f) 
            for p in chain_probs
        ]
        
        # We want to Maximize the MEAN utility across the chain
        # (Optim minimizes by default, so we return negative mean)
        return -mean(utilities)
    end

    # Optimization: Find 'f' between 0.0 and 0.99 that maximizes utility
    # We use a simple bounded search
    result = optimize(expected_utility, 0.0, 0.99)
    
    optimal_f = Optim.minimizer(result)
    
    return optimal_f
end


# v2 

"""
Calculates the Optimal Shrinkage Factor 'k' using the Baker & McHale (2013) 
Bootstrap/Resampling method (Eq. 2).

This simulates the penalty of acting on noisy probability estimates.
"""
function optimize_bayesian_kelly_v2(chain_probs::AbstractVector, offered_odds::Number)
    b = offered_odds - 1.0
    
    # 1. We treat the Mean of the posterior as the "Ground Truth" for this simulation
    p_true = mean(chain_probs)
    
    # If the mean suggests no bet, we can't shrink what doesn't exist.
    s_mean = kelly_fraction(offered_odds, p_true)
    if s_mean <= 1e-6
        return 0.0
    end

    # 2. We generate the "Naive" bets we would have made for every sample in the chain.
    # Ideally, we calculate s*(q) for every q.
    # This represents the variability of our decision making process.
    naive_bets = [kelly_fraction(offered_odds, q) for q in chain_probs]

    # 3. Objective Function: 
    # Find k such that if we shrink ALL our naive bets by k, 
    # we maximize growth against the "p_true".
    function objective(k)
        utility_sum = 0.0
        n = length(naive_bets)
        
        for s_q in naive_bets
            # The bet we actually place is the Naive Bet * Shrinkage k
            actual_stake = k * s_q
            
            # Constraint check
            if actual_stake >= 0.999 
                return Inf 
            elseif actual_stake < 0.0
                actual_stake = 0.0
            end
            
            # CRITICAL STEP:
            # We evaluate the utility of this stake against p_true (The Mean).
            # We do NOT evaluate against q. 
            # This measures: "How bad is it if I bet s_q when the truth is actually p_true?"
            
            u = p_true * log(1.0 + b * actual_stake) + (1.0 - p_true) * log(1.0 - actual_stake)
            utility_sum += u
        end
        
        return -(utility_sum / n)
    end

    # Optimize k between 0.0 (full shrinkage) and 1.0 (no shrinkage)
    # Baker & McHale prove k* is always < 1 when uncertainty exists
    res = optimize(objective, 0.0, 1.0)
    best_k = Optim.minimizer(res)
    
    # Return the FINAL bet size (s_mean * k)
    # We apply the optimal shrinkage to our best estimate.
    return s_mean * best_k
end

function calc_analytical_shrinkage(chain_probs, offered_odds)
    # 1. Stats from Chain
    p_mean = mean(chain_probs)
    p_var  = var(chain_probs) # This is sigma^2
    b = offered_odds - 1.0
    
    # 2. Standard Kelly (using Mean)
    s_star = ((b + 1) * p_mean - 1) / b
    
    if s_star <= 0
        return 0.0 # No edge
    end

    # 3. Baker & McHale Eq. 5 (Shrinkage Factor)
    # k = s^2 / (s^2 + term * sigma^2)
    term = ((b + 1) / b)^2
    k_factor = (s_star^2) / (s_star^2 + term * p_var)
    
    # 4. Final Shrunken Bet
    return s_star * k_factor
end


# --- The Display Function ---

"""
Calculates the Standard Kelly Fraction using a point estimate (Plug-in approach).
"""
function kelly_fraction(decimal_odds::Number, probability::Number)
    if decimal_odds <= 1.0 return 0.0 end
    return max(0.0, probability - ((1.0 - probability) / (decimal_odds - 1.0)))
end

using Statistics, Printf, Optim, Dates

function display_kelly_bits(predict, open_odds_set, close_odds_set, results, symbol)
    # 1. Extract Data
    if !haskey(predict, symbol) || !haskey(open_odds_set, symbol)
        println("Symbol :$symbol not found in data.")
        return
    end

    chain = predict[symbol]
    o_odds = open_odds_set[symbol]
    c_odds = close_odds_set[symbol] # Optional: handle if missing
    outcome = haskey(results, symbol) ? results[symbol] : "Pending"

    # 2. Basic Stats
    n_samples = length(chain)
    p_mean = mean(chain)
    p_std  = std(chain)
    p_median = median(chain)
    
    # 3. Market Context
    implied_prob = 1.0 / o_odds
    edge_mean = (p_mean * o_odds) - 1.0
    
    # 4. Kelly Calculations
   # A. Raw (Plug-in)
    k_raw = kelly_fraction(o_odds, p_mean)

    # B. Analytical (Baker McHale Eq 5)
    k_approx = calc_analytical_shrinkage(chain, o_odds)

    # C. Bayes Optimal (Integral)
    raw_bayes = optimize_bayesian_kelly_v2(chain, o_odds)
    
    k_bayes = raw_bayes < 0.0001 ? 0.0 : raw_bayes

    # Shrinkage Ratio
    s_ratio = k_raw > 0 ? (k_bayes / k_raw) : 0.0

    # --- PRINT OUTPUT ---
    
    printstyled("══════════════════════════════════════════════════════\n", color=:blue)
    printstyled(@sprintf(" MARKET ANALYSIS: :%s \n", symbol), bold=true, color=:white)
    printstyled("══════════════════════════════════════════════════════\n", color=:blue)
    
    println("Outcome:           ", outcome ? "WIN" : "LOSS")
    @printf("Open Odds :        %.3f  (Implied: %.1f%%)\n", o_odds, implied_prob*100)
    @printf("Close Odds:        %.3f  (Implied: %.1f%%)  \n", c_odds, (1 / c_odds) *100)
    @printf("Model Odds: (mean) %.3f\n", 1 / p_mean )
    println()
    
    printstyled("--- Model Posterior (MCMC) ---\n", color=:cyan)
    @printf("Mean Prob:         %.1f%%  (Edge: %.2f%%)\n", p_mean*100, edge_mean*100)
    @printf("Uncertainty (σ):   %.3f\n", p_std)
    @printf("90%% CI:            [%.1f%% - %.1f%%]\n", quantile(chain, 0.05)*100, quantile(chain, 0.95)*100)
    println()
    
    printstyled("--- Kelly Strategy Comparison ---\n", color=:yellow)
    
    # Helper to print bars
    function print_bar(label, val, raw_val)
        bar_len = Int(round(val * 40)) # Scale: 100% = 40 chars
        bar = repeat("█", bar_len)
        reduction = raw_val > 0 ? (1.0 - (val/raw_val)) * 100 : 0.0
        @printf("%-18s | %5.2f%% %s", label, val*100, bar)
        if reduction > 0.1
            @printf(" (Shrunk %.1f%%)", reduction)
        end
        println()
    end

    print_bar("Raw (Plug-in)", k_raw, k_raw)
    print_bar("Analytical (Eq5)", k_approx, k_raw)
    print_bar("Bayes Optimal", k_bayes, k_raw)

    println()
    
    # Interpretation
    if k_bayes > 0.0
        # We have a positive bet
        if s_ratio < 0.5
             # Bayes suggests shrinking by more than 50%
             printstyled("Advice: Bet with Caution. High uncertainty detected.\n", color=:yellow)
        else
             printstyled("Advice: Value Identified. Bet suggested.\n", color=:green)
        end
    elseif k_raw > 0.0 && k_bayes == 0.0
        # Raw says yes, Bayes says no
        printstyled("Advice: NO BET. Uncertainty eliminates the edge.\n", color=:light_red)
    else
        # No edge
        printstyled("Advice: NO BET. No value.\n", color=:red)
    end
    printstyled("══════════════════════════════════════════════════════\n", color=:blue)
end

display_kelly_bits(match_predict, open, close, results, :over_25) 
display_kelly_bits(match_predict, open, close, results, :under_25) 
display_kelly_bits(match_predict, open, close, results, :over_15) 
display_kelly_bits(match_predict, open, close, results, :under_15) 
display_kelly_bits(match_predict, open, close, results, :home) 
display_kelly_bits(match_predict, open, close, results, :away) 

display_kelly_bits(match_predict, open, close, results, :over_05) 
display_kelly_bits(match_predict, open, close, results, :under_05) 


display_kelly_bits(match_predict, open, close, results, :over_35) 
display_kelly_bits(match_predict, open, close, results, :under_35) 


display_kelly_bits(match_predict, open, close, results, :btts_yes) 
display_kelly_bits(match_predict, open, close, results, :btts_no) 

keys(match_predict)

mean(kell)
describe(kell)
mean(match_predict[:home])




open[:under_25]
close[:under_25]
model_odds[:under_25]
results[:under_25]


kell = BayesianFootball.Signals.kelly_fraction(open[:under_25], match_predict[:under_25]);
density(kell)
describe(kell)
optimize_bayesian_kelly(match_predict[:under_25], open[:under_25])
calc_analytical_shrinkage(match_predict[:under_25], open[:under_25])

BayesianFootball.Signals.kelly_fraction(open[:under_25],mean(match_predict[:under_25]))


#################################################
#  dev cts 
#################################################


using DataFrames, Statistics, Optim, ProgressMeter

# --- 1. Helper Functions (The Math) ---

"""Standard Plug-in Kelly"""
function kelly_fraction(decimal_odds::Number, probability::Number)
    if decimal_odds <= 1.0 return 0.0 end
    return max(0.0, probability - ((1.0 - probability) / (decimal_odds - 1.0)))
end

"""Baker-McHale Eq 5 (Analytical Approx)"""
function calc_analytical_shrinkage(chain_probs::AbstractVector, offered_odds::Number)
    p_mean = mean(chain_probs)
    p_var = var(chain_probs)
    b = offered_odds - 1.0
    s_star = ((b + 1) * p_mean - 1) / b
    if s_star <= 0 return 0.0 end
    term = ((b + 1) / b)^2
    k_factor = (s_star^2) / (s_star^2 + term * p_var)
    return s_star * k_factor
end

"""Baker-McHale Eq 2 (Bayesian Bootstrap Optimization)"""
function optimize_bayesian_kelly_v3(chain_probs::AbstractVector, offered_odds::Number)
    b = offered_odds - 1.0
    p_true = mean(chain_probs)
    s_mean = kelly_fraction(offered_odds, p_true)
    
    # PERFORMANCE OPTIMIZATION: 
    # If there is no edge in the mean, there is no need to optimize shrinkage.
    if s_mean <= 0.0 return 0.0 end

    # Pre-calculate naive bets for the chain
    naive_bets = [kelly_fraction(offered_odds, q) for q in chain_probs]

    function objective(k)
        utility_sum = 0.0
        n = length(naive_bets)
        for s_q in naive_bets
            actual_stake = k * s_q
            if actual_stake >= 0.999 return Inf end
            if actual_stake < 0.0 actual_stake = 0.0 end
            
            # Utility evaluated against the Mean (p_true)
            u = p_true * log(1.0 + b * actual_stake) + (1.0 - p_true) * log(1.0 - actual_stake)
            utility_sum += u
        end
        return -(utility_sum / n)
    end

    res = optimize(objective, 0.0, 1.0)
    best_k = Optim.minimizer(res)
    return s_mean * best_k
end

# --- 2. The DataFrame Generator ---

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
                k_raw = kelly_fraction(o_open, prob_mean)
                
                # B. Analytical (Eq 5) - Fast Approx
                k_analytical = calc_analytical_shrinkage(chain, o_open)
                
                # C. Bayes Optimal (Eq 2) - Slow Integration
                # Only run if we have a raw edge, otherwise it's 0.0 anyway
                k_bayes = 0.0
                if k_raw > 0.0
                     k_bayes = optimize_bayesian_kelly_v3(chain, o_open)
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



function calculate_equity_curve_v1(df::DataFrame; bankroll=1000.0, max_stake=1.0)
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


# 1. Generate the detailed data
df_analysis = generate_kelly_analysis_df(model, predict_config, all_oos_results, ds)

# 2. Inspect a specific market type (e.g., Home Wins)
home_only = filter(row -> row.market == :away , df_analysis)

# 3. Run the simulation on just Home bets
equity = calculate_equity_curve_v1(home_only, bankroll=100.0, max_stake=0.2)

# 4. Plot
using StatsPlots
plot(equity.idx, equity.raw_bank, label="Raw Kelly", lw=2)
plot!(equity.idx, equity.bayes_bank, label="Bayes Shrunk", lw=2)


plot!(equity.idx, equity.flat_bank, label="Flat Stake", lw=2, linestyle=:dash)


describe(equity[:, [:raw_bank, :analytical_bank, :bayes_bank, :flat_bank]])

display_equity_summary(equity)


# 
function plot_bet_size_distribution(df_analysis::DataFrame)
    # Filter only bets where Raw Kelly > 0 (Active bets)
    active_bets = filter(r -> r.kelly_raw > 0, df_analysis)
    
    p = density(active_bets.kelly_raw, label="Raw Bet Size", fill=true, alpha=0.3, color=:red)
    density!(p, active_bets.kelly_bayes, label="Bayes Bet Size", fill=true, alpha=0.3, color=:blue)
    
    title!(p, "Effect of Parameter Uncertainty on Bet Sizing")
    xlabel!(p, "Fraction of Bankroll")
    return p
end

# Run this:
plot_bet_size_distribution(df_analysis)

function plot_equity_curves(equity_df::DataFrame; title="Kelly Strategy Performance")
    
    # Ensure we are sorted by time/index
    sort!(equity_df, :idx)
    
    # We plot on a Log Scale often for Kelly, but Linear is better for detecting Ruin
    p = plot(
        equity_df.idx, 
        equity_df.flat_bank, 
        label="Flat Stake", 
        lw=2, 
        color=:grey, 
        linestyle=:dash,
        legend=:topleft,
        title=title,
        xlabel="Bets Placed",
        ylabel="Bankroll (Currency)"
    )
    
    # Add Raw Kelly
    plot!(p, equity_df.idx, equity_df.raw_bank, label="Raw Kelly", color=:red, lw=1.5)
    
    # Add Bayes/Shrunken
    # We can see how closely it hugs the Raw line vs the Flat line
    plot!(p, equity_df.idx, equity_df.bayes_bank, label="Bayes Optimal", color=:blue, lw=2)
    
    # Highlight the "Ruin" line (usually 0 or a low threshold)
    hline!(p, [0], color=:black, label="")

    return p
end

p = plot_equity_curves(equity)

#################################################
# Dev area 
#################################################


# here we want to use the open line odds


"""
Parses a fractional odds string (e.g., "19/10") into a decimal value (e.g., 2.9).
Returns 0.0 if parsing fails (e.g., for "SP", missing, or "1").
"""
function parse_fractional_to_decimal(s::AbstractString)
    parts = split(s, '/')
    
    # Must be exactly two parts (numerator and denominator)
    if length(parts) != 2
        return 0.0
    end
    
    try
        n = parse(Float64, parts[1])
        d = parse(Float64, parts[2])
        
        # Avoid division by zero
        if d == 0.0
            return 0.0
        end
        
        # Convert from fractional (e.g., 1.9) to decimal (e.g., 2.9)
        return (n / d) + 1.0
    catch e
        # This will catch errors if parts[1] or parts[2] are not valid numbers (e.g., "SP")
        return 0.0
    end
end


ds_odds_initial = deepcopy(ds.odds)

# Convert the initial fractional string to a decimal
ds_odds_initial.initial_decimal = parse_fractional_to_decimal.(ds_odds_initial.initial_fractional_value)

# Filter out rows where parsing failed or odds were 0
filter!(row -> row.initial_decimal > 1.0, ds_odds_initial)

# --- THIS IS THE "TRICK" ---
# Overwrite the `decimal_odds` column with our new initial odds.
# Your `get_market` function will now read this column,
# thinking it's the final odds.
ds_odds_initial.decimal_odds = round.(ds_odds_initial.initial_decimal, digits=2)

# create new data store.
ds = BayesianFootball.Data.DataStore(
    df,
    ds_odds_initial,
    data_store.incidents
)




# +++ Compute kelly 

"""
Calculates the optimal Kelly criterion fraction to bet.

# Arguments
- `decimal_odds`: The decimal odds offered by the bookmaker (e.g., 2.5, 3.0).
- `probability`: Your estimated true probability of the event occurring (e.g., 0.45).

# Returns
- `f`: The fraction of your bankroll to bet (from 0.0 to 1.0).
  A value of 0.0 means the bet has no value (p < 1/decimal_odds).
"""
function kelly_fraction(decimal_odds::Number, probability::Number)
  return max(0.0, probability - ( (1 - probability) / (decimal_odds - 1.0)))
end 

function kelly_fraction(decimal_odds::Number, probability::AbstractVector)
  return kelly_fraction.(decimal_odds, probability)
end 

function kelly_fraction(odds::NamedTuple, probabilities::NamedTuple) 
  common_keys = keys(odds) ∩ keys(probabilities)
  return NamedTuple(
          k => kelly_fraction(odds[k], probabilities[k])
          for k in common_keys
      )
end


# --- compute kelly dev 

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

match_id = rand(keys(all_oos_results))
r1 =  all_oos_results[match_id]
subset( ds.matches, :match_id => ByRow(isequal(match_id)))

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

closing_match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)

kf = kelly_fraction(match_odds, match_predict)

# +++ Kelly signals 
function kelly_positive( kelly_dist::AbstractVector, c::Number)
  return mean(kelly_dist .> c)
end 

function kelly_decision_rule( kelly_dist::AbstractVector, c::Number, b::Number)::Bool
  return kelly_positive(kelly_dist, c) >= b 
end 



# --- kelly signal dev 


kelly_positive( kf[:home], 0.02)
kelly_decision_rule( kf[:home], 0.02, 0.5)

kelly_positive( kf[:over_25], 0.0)
kelly_positive( kf[:under_25], 0.0)

kelly_decision_rule( kf[:over_25], 0.06, 0.3)
kelly_decision_rule( kf[:under_25], 0.06, 0.3)

kelly_positive( kf[:over_35], 0.05)
kelly_positive( kf[:under_35], 0.05)

using StatsPlots
density( kf[:over_35], label="over")
density!( kf[:under_35], label="under")


# +++ kelly stake, 
function kellys_stake_precent(kelly_dist::AbstractVector, kellys_fraction::Number)::Float64 
  return kellys_fraction * median(kelly_dist)
end 

kellys_stake_precent( kf[:home], 1)


# +++ strategy 
function kelly_strategy(kelly_dist::AbstractVector, c::Number, b::Number, f::Number)::Number 
  return kelly_decision_rule(kelly_dist, c, b) * kellys_stake_precent(kelly_dist, f) 
end 


kelly_strategy(kf[:home], 0.05, 0.3, 1)





# +++ 

# --- 1. Load SciML Optimization Packages ---
using Optimization
using OptimizationBBO # Black-Box Optimization (derivative-free)
using SciMLBase # For NoAD()

"""
Calculates the geometric equity curve from a series of fractional stakes.
This correctly models compounding growth and risk of ruin.
"""
function calculate_equity_curve(
    stakes::AbstractVector{<:Number}, 
    odds::AbstractVector{<:Number}, 
    results::AbstractVector{Bool}
    )::Vector{Float64}
    
    # Start with a 1.0 unit bankroll
    bankroll = 1.0
    equity_curve = [bankroll]

    for (stake_frac, odd, result) in zip(stakes, odds, results)
        if stake_frac <= 0.0 || stake_frac > 1.0 # No bet or invalid stake
            push!(equity_curve, bankroll)
            continue
        end

        bet_amount = bankroll * stake_frac
        
        if result == true
            # Win
            bankroll += bet_amount * (odd - 1.0)
        else
            # Loss
            bankroll -= bet_amount
        end

        # Handle ruin
        if bankroll <= 0.0
            bankroll = 0.0
            push!(equity_curve, bankroll)
            break 
        end
        
        push!(equity_curve, bankroll)
    end
    return equity_curve
end

"""
Calculates the Calmar Ratio from a geometric equity curve.
We define it as Total Geometric Return / Max Drawdown.
"""
function calculate_calmar_ratio(equity_curve::Vector{Float64})::Float64
    if isempty(equity_curve) || equity_curve[1] <= 0.0
        return -1e9 # Penalize errors
    end

    # 1. Total Geometric Return (e.g., 1.5 -> 50% return)
    total_return = (equity_curve[end] / equity_curve[1]) - 1.0

    # 2. Maximum Drawdown
    peak = -Inf
    max_drawdown = 0.0
    
    for value in equity_curve
        if value > peak
            peak = value
        end
        
        if peak > 0.0
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown
                max_drawdown = drawdown
            end
        elseif peak == 0.0 # We hit ruin
            max_drawdown = 1.0 # 100% drawdown
            break
        end
    end

    # --- Scoring ---
    if total_return <= 0.0
        return total_return * 10.0 # Penalize
    end

    if max_drawdown == 0.0
        return total_return > 0.0 ? 1e9 : 0.0 # Huge score if profitable
    end

    return total_return / max_drawdown
end

"""
NEW objective function to optimize for Win/Loss Percentage.
It calculates the win rate *only* on the bets that are
actually placed by the strategy.
"""
function objective_function_win_pct(params, p_unused)
    c, b, f = params
    
    # Basic bounds check (optional but good practice)
    if c < 0.0 || b < 0.0 || b > 1.0 || f <= 0.0 || f > 1.0
        return Inf # Penalize invalid parameters
    end

    total_bets_placed = 0
    total_wins = 0

    # Iterate through all matches to see which ones we bet on
    for match in market_data
        # Calculate the stake based on our parameters
        # We only care if the stake is > 0, not its size.
        stake = kelly_strategy(match.k_dist, c, b, f)
        
        # Check if the strategy decided to bet
        if stake > 0.0
            total_bets_placed += 1
            
            # Check if the bet was a winner
            # match.result is a Bool (true/false)
            if match.result == true
                total_wins += 1
            end
        end
    end

    # Calculate the win percentage
    win_percentage = if total_bets_placed == 0
        0.0 # No bets placed = 0% win rate.
    else
        total_wins / total_bets_placed
    end

    # We want to MAXIMIZE win_percentage,
    # so we return the NEGATIVE value for the minimizer.
    return -win_percentage
end

# --- Helper Functions for Sortino Ratio ---

"""
Calculates a simple mean. (Use Statistics.mean if you have it)
"""
function simple_mean(vec::AbstractVector{<:Number})
    if isempty(vec)
        return 0.0
    end
    return sum(vec) / length(vec)
end

"""
Converts a cumulative equity curve into a list of 
per-period fractional returns (e.g., [1.0, 1.1, 1.05] -> [0.1, -0.045])
"""
function get_returns(equity_curve::Vector{Float64})
    if length(equity_curve) < 2
        return Float64[]
    end
    
    returns = Vector{Float64}(undef, length(equity_curve) - 1)
    
    for i in 2:length(equity_curve)
        old_val = equity_curve[i-1]
        new_val = equity_curve[i]
        
        if old_val <= 0.0 # We are at ruin, no more returns
            # Set subsequent returns to 0, or just break
            returns[i-1] = 0.0 
        else
            returns[i-1] = (new_val / old_val) - 1.0
        end
    end
    return returns
end


"""
Calculates the Sortino Ratio from a geometric equity curve.
We use a target return of 0.0.
"""
function calculate_sortino_ratio(equity_curve::Vector{Float64}; target_return::Float64 = 0.0)::Float64
    if length(equity_curve) < 2
        return -1e9 # Penalize
    end

    # 1. Get the list of per-period returns
    # This list will include 0.0 returns for "no bet" periods
    returns = get_returns(equity_curve)
    
    if isempty(returns)
        return 0.0 # No returns, no risk, no score
    end

    # 2. Calculate average return
    avg_return = simple_mean(returns)
    
    # 3. Calculate Downside Deviation
    downside_sum_of_squares = 0.0
    
    for r in returns
        if r < target_return
            # We only care about returns *below* our target
            downside_sum_of_squares += (r - target_return)^2
        end
    end

    # The deviation is the sum of squares / N (total periods)
    # This correctly penalizes strategies that lose infrequently
    # but heavily.
    downside_deviation = sqrt(downside_sum_of_squares / length(returns))

    if downside_deviation == 0.0
        # No downside returns at all. This is a perfect strategy.
        return avg_return > target_return ? 1e9 : 0.0 # Huge score if profitable
    end

    # 4. Calculate Sortino Ratio
    sortino_ratio = (avg_return - target_return) / downside_deviation

    if !isfinite(sortino_ratio)
        return -1e9 # Penalize
    end

    # Penalize negative-return strategies to help the optimizer
    if avg_return <= target_return
         return sortino_ratio * 10.0 # Make it a large negative number
    end

    return sortino_ratio
end

"""
NEW objective function to optimize for the Sortino Ratio.
"""
function objective_function_sortino(params, p_unused)
    c, b, f = params
    
    if c < 0.0 || b < 0.0 || b > 1.0 || f <= 0.0 || f > 1.0
        return Inf # Return a "very bad" score
    end

    # 1. Generate stakes
    stakes = Vector{Float64}(undef, length(market_data))
    odds_list = Vector{Float64}(undef, length(market_data))
    results_list = Vector{Bool}(undef, length(market_data))

    for (i, match) in enumerate(market_data)
        stakes[i] = kelly_strategy(match.k_dist, c, b, f)
        odds_list[i] = match.odds
        results_list[i] = match.result
    end
    
    # 2. Calculate the equity curve
    equity_curve = calculate_equity_curve(stakes, odds_list, results_list)
    
    # 3. Calculate the final score
    score = calculate_sortino_ratio(equity_curve)

    if !isfinite(score)
        return Inf # "Very bad"
    end

    # We want to MAXIMIZE Sortino, so we return the NEGATIVE score
    return -score
end


# (This function was also used for Sortino)
function simple_mean(vec::AbstractVector{<:Number})
    if isempty(vec)
        return 0.0
    end
    return sum(vec) / length(vec)
end

"""
Finds all distinct peak-to-trough drawdown periods in an equity curve.
Returns a vector of the drawdowns (e.g., [0.1, 0.05, 0.2] for 10%, 5%, 20% drawdowns).
"""
function get_all_drawdowns(equity_curve::Vector{Float64})::Vector{Float64}
    if length(equity_curve) < 2
        return Float64[]
    end

    drawdowns = Float64[]
    peak = equity_curve[1]
    current_trough = equity_curve[1]

    for value in equity_curve[2:end]
        if value > peak
            # We hit a new peak, so the previous drawdown (if any) is over.
            if current_trough < peak && peak > 0.0
                dd = (peak - current_trough) / peak
                push!(drawdowns, dd)
            end
            
            # Start the new period
            peak = value
            current_trough = value
        else
            # We are at or below the peak, update the trough
            current_trough = min(current_trough, value)
        end
    end

    # Handle the final drawdown period after the loop ends
    if current_trough < peak && peak > 0.0
        dd = (peak - current_trough) / peak
        push!(drawdowns, dd)
    end
    
    # Handle ruin case (100% drawdown)
    if any(v -> v == 0.0, equity_curve) && (isempty(drawdowns) || maximum(drawdowns) < 1.0)
         push!(drawdowns, 1.0)
    end

    return drawdowns
end


"""
Calculates the Sterling Ratio (Total Return / Average Drawdown).
"""
function calculate_sterling_ratio(equity_curve::Vector{Float64})::Float64
    if isempty(equity_curve) || equity_curve[1] <= 0.0
        return -1e9 # Penalize
    end

    # 1. Total Geometric Return
    total_return = (equity_curve[end] / equity_curve[1]) - 1.0

    # 2. Get all drawdowns
    drawdowns = get_all_drawdowns(equity_curve)

    # --- Scoring ---
    if total_return <= 0.0
        return total_return * 10.0 # Penalize
    end

    if isempty(drawdowns) || simple_mean(drawdowns) == 0.0
        # No drawdowns, perfect score
        return 1e9 
    end
    
    avg_drawdown = simple_mean(drawdowns)

    return total_return / avg_drawdown
end


"""
Calculates the Burke Ratio (Total Return / Sqrt(Sum of Drawdowns^2)).
"""
function calculate_burke_ratio(equity_curve::Vector{Float64})::Float64
    if isempty(equity_curve) || equity_curve[1] <= 0.0
        return -1e9 # Penalize
    end

    # 1. Total Geometric Return
    total_return = (equity_curve[end] / equity_curve[1]) - 1.0

    # 2. Get all drawdowns
    drawdowns = get_all_drawdowns(equity_curve)

    # --- Scoring ---
    if total_return <= 0.0
        return total_return * 10.0 # Penalize
    end

    if isempty(drawdowns)
        # No drawdowns, perfect score
        return 1e9
    end

    # 3. Calculate denominator
    sum_of_squares = sum(dd^2 for dd in drawdowns)
    
    if sum_of_squares == 0.0
         return 1e9 # Perfect score
    end

    denominator = sqrt(sum_of_squares)
    return total_return / denominator
end



"""
Objective function to optimize for the Sterling Ratio.
"""
function objective_function_sterling(params, p_unused)
    c, b, f = params
    
    if c < 0.0 || b < 0.0 || b > 1.0 || f <= 0.0 || f > 1.0
        return Inf 
    end

    # 1. Generate stakes
    stakes = [kelly_strategy(m.k_dist, c, b, f) for m in market_data]
    odds_list = [m.odds for m in market_data]
    results_list = [m.result for m in market_data]
    
    # 2. Calculate the equity curve
    equity_curve = calculate_equity_curve(stakes, odds_list, results_list)
    
    # 3. Calculate the final score
    score = calculate_sterling_ratio(equity_curve)

    if !isfinite(score)
        return Inf
    end

    return -score # MAXIMIZE Sterling Ratio
end


"""
Objective function to optimize for the Burke Ratio.
"""
function objective_function_burke(params, p_unused)
    c, b, f = params
    
    if c < 0.0 || b < 0.0 || b > 1.0 || f <= 0.0 || f > 1.0
        return Inf
    end

    # 1. Generate stakes
    stakes = [kelly_strategy(m.k_dist, c, b, f) for m in market_data]
    odds_list = [m.odds for m in market_data]
    results_list = [m.result for m in market_data]
    
    # 2. Calculate the equity curve
    equity_curve = calculate_equity_curve(stakes, odds_list, results_list)
    
    # 3. Calculate the final score
    score = calculate_burke_ratio(equity_curve)

    if !isfinite(score)
        return Inf
    end

    return -score # MAXIMIZE Burke Ratio
end

###

HomeMarketData = @NamedTuple{
    k_dist::Vector{Float64}, # Kelly distribution
    odds::Float64,           # Opening odds
    result::Bool             # Did the home team win?
}


# --- 3. Pre-process Data (Modified for :under_25) ---

# Define the market we are targeting
TARGET_MARKET = :over_25 # <-- THE ONLY CHANGE IS HERE


MarketData = @NamedTuple{
    k_dist::Vector{Float64}, # Kelly distribution
    odds::Float64,           # Opening odds
    result::Bool             # Did the home team win?
}



println("--- Starting Data Pre-processing for: $(TARGET_MARKET) ---")
predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())
processed_count = 0
error_count = 0
market_data = MarketData[] # <-- Generic name
for match_id in keys(all_oos_results)
    try
        r1 = all_oos_results[match_id]
        match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
        match_odds = BayesianFootball.Predictions.get_market_opening_lines(match_id, predict_config, ds.odds)
        # match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
        match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)


        if TARGET_MARKET ∉ keys(match_odds) || 
           TARGET_MARKET ∉ keys(match_predict) || 
           match_odds[TARGET_MARKET] <= 1.0
            continue # Skip if market doesn't exist or odds are invalid
        end
        
      k_dist_home = kelly_fraction(match_odds[TARGET_MARKET], match_predict[TARGET_MARKET])

        push!(market_data, (
            k_dist = k_dist_home,
            odds = match_odds[TARGET_MARKET],
            result = match_results[TARGET_MARKET]
        ))
        processed_count += 1
    catch e
        error_count += 1
    end
end

println("--- Pre-processing Complete ---")
println("Successfully processed $(processed_count) matches for $(TARGET_MARKET).")
println("Skipped $(error_count) matches due to errors or missing data.")



"""
This is the function Optimization.jl will call.
It must take params `u` (our vector) and `p` (static data, which we don't use).
It runs a full backtest and returns a score (Negative Calmar).
"""
function objective_function(params, p_unused)
    c, b, f = params
    
    # We can put bounds-checking here, though Optimization.jl handles it
    if c < 0.0 || b < 0.0 || b > 1.0 || f <= 0.0 || f > 1.0
        return Inf # Return a "very bad" score (positive infinity)
    end

    # 1. Generate stakes
    # Note: We access the pre-processed data from the global scope
    # A more advanced setup would pass `home_market_data` as `p_unused`
    stakes = Vector{Float64}(undef, length(market_data))
    odds_list = Vector{Float64}(undef, length(market_data))
    results_list = Vector{Bool}(undef, length(market_data))

    for (i, match) in enumerate(market_data)
        stakes[i] = kelly_strategy(match.k_dist, c, b, f)
        odds_list[i] = match.odds
        results_list[i] = match.result
    end
    
    # 2. Calculate the equity curve
    equity_curve = calculate_equity_curve(stakes, odds_list, results_list)
    
    # 3. Calculate the final score
    score = calculate_calmar_ratio(equity_curve)

    if !isfinite(score)
        return Inf # "Very bad"
    end

    # We want to MAXIMIZE Calmar, so we return the NEGATIVE score
    return -score
end



# --- 5. Setup and Run Optimization.jl ---

println("\n--- Starting Optimization.jl ---")

# 1. Define the function for the optimizer
#    We explicitly state there is No Automatic Differentiation (NoAD)
#    because our function is a complex, non-differentiable backtest.

opt_func = OptimizationFunction(objective_function, SciMLBase.NoAD())
# opt_func = OptimizationFunction(objective_function_win_pct, SciMLBase.NoAD())
# opt_func = OptimizationFunction(objective_function_sortino, SciMLBase.NoAD())
opt_func = OptimizationFunction(objective_function_sterling, SciMLBase.NoAD())
opt_func = OptimizationFunction(objective_function_burke, SciMLBase.NoAD())
# 2. Define search space and initial guess
#    [c,     b,     f]
u0 = [0.01,  0.5,   0.1]  # Initial guess (c, b, f)
lower_bounds = [0.0,   0.5,   0.0]
upper_bounds = [0.2,   0.6,   0.2]

# 3. Create the OptimizationProblem
prob = OptimizationProblem(
    opt_func, 
    u0, 
    lb = lower_bounds, 
    ub = upper_bounds
)


# 4. Solve the problem!
# We choose a solver from the OptimizationBBO package.
# `BBO_adaptive_de_rand_1_bin_radiuslimited()` is a robust global optimizer.

sol = solve(
    prob, 
    BBO_adaptive_de_rand_1_bin_radiuslimited(), # The chosen algorithm
    maxiters=1000  # Number of iterations
)

println("--- Optimization Complete ---")

# 5. Show the results
best_score = sol.objective
best_params = sol.u

println("Best (Negative) Calmar: $best_score")
println("Actual Calmar: $(-best_score)")
println("Optimal Parameters (c, b, f):")
println("  c (Kelly Thresh): $(best_params[1])")
println("  b (Conf Thresh):  $(best_params[2])")
println("  f (Kelly Frac):   $(best_params[3])")

# 6. Re-run final backtest (same as before)
println("\n--- Final Backtest with Optimal Params ---")
c_opt, b_opt, f_opt = best_params
stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];
odds_opt = [m.odds for m in market_data];
results_opt = [m.result for m in market_data];

final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);
final_calmar = calculate_calmar_ratio(final_curve)

println("Final Calmar (re-calculated): $final_calmar")
println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
println("Final Bankroll (starting from 1.0): $(final_curve[end])")


#= 
# calmer ration 
--- Final Backtest with Optimal Params ---

julia> c_opt, b_opt, f_opt = best_params
3-element Vector{Float64}:
 0.006766780047802295
 0.51695544614808
 0.19993751549615882

julia> stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];

julia> odds_opt = [m.odds for m in market_data];

julia> results_opt = [m.result for m in market_data];

julia> final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);

julia> final_calmar = calculate_calmar_ratio(final_curve)
62.94943921607101

julia> println("Final Calmar (re-calculated): $final_calmar")
Final Calmar (re-calculated): 62.94943921607101

julia> println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
Total Bets Placed: 178 / 514

julia> println("Final Bankroll (starting from 1.0): $(final_curve[end])")
Final Bankroll (starting from 1.0): 20.63592553218873


### win pct opt
--- Final Backtest with Optimal Params ---

julia> c_opt, b_opt, f_opt = best_params
3-element Vector{Float64}:
 0.026242489334867926
 0.5002810934044016
 0.10072990143426294

julia> stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];

julia> odds_opt = [m.odds for m in market_data];

julia> results_opt = [m.result for m in market_data];

julia> final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);

julia> final_calmar = calculate_calmar_ratio(final_curve)
24.550545535961007

julia> 

julia> println("Final Calmar (re-calculated): $final_calmar")
Final Calmar (re-calculated): 24.550545535961007

julia> println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
Total Bets Placed: 171 / 514

julia> println("Final Bankroll (starting from 1.0): $(final_curve[end])")
Final Bankroll (starting from 1.0): 5.1666750508924135

julia> 

julia> plot(final_curve, title="Optimal Equity Curve", label="Bankroll", legend=:topleft)


### sortino_ratio 


--- Final Backtest with Optimal Params ---

julia> c_opt, b_opt, f_opt = best_params
3-element Vector{Float64}:
 0.009208363890064655
 0.511276827006579
 0.00023248241896002192

julia> stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];

julia> odds_opt = [m.odds for m in market_data];

julia> results_opt = [m.result for m in market_data];

julia> final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);

julia> final_calmar = calculate_calmar_ratio(final_curve)
9.912280632943942

julia> println("Final Calmar (re-calculated): $final_calmar")
Final Calmar (re-calculated): 9.912280632943942

julia> println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
Total Bets Placed: 178 / 514

julia> println("Final Bankroll (starting from 1.0): $(final_curve[end])")
Final Bankroll (starting from 1.0): 1.0040978424701088



#### Sterling 
--- Final Backtest with Optimal Params ---

julia> c_opt, b_opt, f_opt = best_params
3-element Vector{Float64}:
 0.002431739385395072
 0.5008297266854782
 0.19935580525696847

julia> stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];

julia> odds_opt = [m.odds for m in market_data];

julia> results_opt = [m.result for m in market_data];

julia> final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);

julia> final_calmar = calculate_calmar_ratio(final_curve)
61.89332770760149

julia> println("Final Calmar (re-calculated): $final_calmar")
Final Calmar (re-calculated): 61.89332770760149

julia> println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
Total Bets Placed: 184 / 514

julia> println("Final Bankroll (starting from 1.0): $(final_curve[end])")
Final Bankroll (starting from 1.0): 20.25769013218774


## Burke ration 

--- Final Backtest with Optimal Params ---

julia> c_opt, b_opt, f_opt = best_params
3-element Vector{Float64}:
 0.013678552316016763
 0.5024920202729628
 0.19999746081092082

julia> stakes_opt = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in market_data];

julia> odds_opt = [m.odds for m in market_data];

julia> results_opt = [m.result for m in market_data];

julia> final_curve = calculate_equity_curve(stakes_opt, odds_opt, results_opt);

julia> final_calmar = calculate_calmar_ratio(final_curve)
63.058933380127854

julia> println("Final Calmar (re-calculated): $final_calmar")
Final Calmar (re-calculated): 63.058933380127854

julia> println("Total Bets Placed: $(count(s -> s > 0, stakes_opt)) / $(length(stakes_opt))")
Total Bets Placed: 178 / 514

julia> println("Final Bankroll (starting from 1.0): $(final_curve[end])")
Final Bankroll (starting from 1.0): 20.675201526504544




=# 

# If you have Plots.jl, you can uncomment this
using Plots
plot(final_curve, title="Optimal Equity Curve", label="Bankroll", legend=:topleft)





#### sens 

using Printf # For pretty-printing the table

# --- 1. Define Your Parameter Grid ---

# Test Kelly thresholds (c) from 0% to 5% in 11 steps
c_range = LinRange(0.0, 0.05, 11)

# Test Kelly Fractions (f) from 5% to 50% in 10 steps
f_range = LinRange(0.05, 0.5, 10)

# *** Your Fixed Parameter ***
b_fixed = 0.5

# --- 2. Create Storage for Results ---
# Rows will be c, Columns will be f
calmar_grid = zeros(length(c_range), length(f_range))

println("--- Starting Sensitivity Analysis (v1, b=0.5) ---")
println("Mapping $(length(calmar_grid)) parameter combinations...")

# --- 3. Run the Grid Search (Nested Loop) ---

for (i, c) in enumerate(c_range)
    for (j, f) in enumerate(f_range)
        
        # 1. Define the parameter set [c, b, f]
        params = [c, b_fixed, f]
        
        # 2. Call your original 3-parameter objective function
        #    (This function must be defined in your session)
        negative_calmar = objective_function(params, nothing)
        
        # 3. Store the *positive* Calmar
        actual_calmar = -negative_calmar
        if actual_calmar < -1.0 # This was a negative return
             actual_calmar /= 10.0 # Revert the penalty
        end

        calmar_grid[i, j] = actual_calmar
    end
end

println("--- Analysis Complete ---")

# --- 4. Print the Results as a Table ---

# Print header row (f values)
@printf "c_thresh |"
for f in f_range
    @printf "%8.2f" f
end
println("\n" * "-"^90) # Separator

# Print each row (c_thresh) and the Calmar scores
for (i, c) in enumerate(c_range)
    @printf "  %0.3f  |" c # Print c with 3 decimal places
    for j in 1:length(f_range)
        score = calmar_grid[i, j]
        if score <= 0.0
            @printf "%8s" "----"
        else
            @printf "%8.2f" score
        end
    end
    println() # New line for the next row
end

function print_sensitivity_analysis_table(c_range, f_range, b_fixed) 

      calmar_grid = zeros(length(c_range), length(f_range))

      println("--- Starting Sensitivity Analysis (v1, b=0.5) ---")
      println("Mapping $(length(calmar_grid)) parameter combinations...")

      # --- 3. Run the Grid Search (Nested Loop) ---

      for (i, c) in enumerate(c_range)
          for (j, f) in enumerate(f_range)
              
              # 1. Define the parameter set [c, b, f]
              params = [c, b_fixed, f]
              
              # 2. Call your original 3-parameter objective function
              #    (This function must be defined in your session)
              negative_calmar = objective_function(params, nothing)
              
              # 3. Store the *positive* Calmar
              actual_calmar = -negative_calmar
              if actual_calmar < -1.0 # This was a negative return
                   actual_calmar /= 10.0 # Revert the penalty
              end

              calmar_grid[i, j] = actual_calmar
          end
      end

      # Print header row (f values)
      @printf "c_th - f |"
      for f in f_range
          @printf "%8.2f" f
      end
      println("\n" * "-"^90) # Separator

      # Print each row (c_thresh) and the Calmar scores
      for (i, c) in enumerate(c_range)
          @printf "  %0.3f  |" c # Print c with 3 decimal places
          for j in 1:length(f_range)
              score = calmar_grid[i, j]
              if score <= 0.0
                  @printf "%8s" "----"
              else
                  @printf "%8.2f" score
              end
          end
          println() # New line for the next row
      end
end 




c_range = LinRange(0.0, 0.2, 11)

# Test Kelly Fractions (f) from 5% to 50% in 10 steps
f_range = LinRange(0.05, 0.5, 10)

# *** Your Fixed Parameter ***
b_fixed = 0.4


print_sensitivity_analysis_table(c_range, f_range, b_fixed)




#### ++++

# --- 1. Define Your WFO Structure ---
n_total_matches = length(market_data)
is_size = 300 # In-sample: Use 300 matches to find params
oos_size = 100 # Out-of-sample: Test params on the next 100

all_oos_curves = [] # We'll store the small equity curve from each OOS chunk

u0 = [0.01,  0.5,   0.2]  # Initial guess (c, b, f)
lb = [0.0,   0.5,   0.0001]
ub = [0.1,   0.6,   0.2]


# --- 2. The Main WFO Loop ---
# We slide the window across the data
for start_index in 1:oos_size:(n_total_matches - is_size)
    
    is_end = start_index + is_size - 1
    oos_start = is_end + 1
    oos_end = min(oos_start + oos_size - 1, n_total_matches)
    
    if oos_end <= oos_start
        break # Not enough data left
    end

    # --- 3. Define the In-Sample Problem ---
    
    # Get the data for this *specific* in-sample chunk
    is_data_chunk = market_data[start_index:is_end]
    
    # Create the objective function *for this chunk*
    # This is a "closure" - it captures the `is_data_chunk`
    
    opt_func_is = OptimizationFunction(objective_function, SciMLBase.NoAD())
    prob_is = OptimizationProblem(opt_func_is, u0, lb=lb, ub=ub) # Use your bounds
    
    # --- 4. SOLVE the IS chunk ---
    # Find the best params *only for this past data*
    sol = solve(prob_is, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=100)
    best_params_for_this_chunk = sol.u
    
    # --- 5. TEST on OOS chunk ---
    # Get the unseen, out-of-sample data
    oos_data_chunk = market_data[oos_start:oos_end]
    
    # Apply the strategy *using the params from step 4*
    c_opt, b_opt, f_opt = best_params_for_this_chunk
    
    # Run the backtest on the OOS data
    stakes_oos = [kelly_strategy(m.k_dist, c_opt, b_opt, f_opt) for m in oos_data_chunk]
    odds_oos = [m.odds for m in oos_data_chunk]
    results_oos = [m.result for m in oos_data_chunk]
    
    # --- 6. Record the OOS Results ---
    # We save the *equity curve itself*
    oos_curve_chunk = calculate_equity_curve(stakes_oos, odds_oos, results_oos)
    push!(all_oos_curves, oos_curve_chunk)
end

# --- 7. Analyze the Final Result ---
# Now, we "stitch" the OOS results together.
# We take the *compounded return* of each chunk.
final_stitched_curve = [1.0]
total_bankroll = 1.0

for oos_curve in all_oos_curves
    # Get the fractional return of this chunk (e.g., 1.15 for +15%)
    chunk_return_frac = oos_curve[end] / oos_curve[1] 
    
    # Apply this to our total bankroll
    total_bankroll *= chunk_return_frac
    push!(final_stitched_curve, total_bankroll)
end

println("--- Walk-Forward Analysis Complete ---")
println("Final OOS Bankroll (starting from 1.0): $(final_stitched_curve[end])")

# You can now calculate the Calmar, Sharpe, etc. on `final_stitched_curve`
# This is the true, robust test of your strategy.

plot(final_stitched_curve)
