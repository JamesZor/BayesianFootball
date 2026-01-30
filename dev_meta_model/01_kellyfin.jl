using DataFrames, Statistics, Distributions

"""
    DynamicMetaModel

A struct to hold the current state of the market regime.
"""
struct MarketRegime
    bias::Float64           # (Predicted - Realized) over window
    empirical_var::Float64  # The adjusted variance sigma^2
    regime_k::Float64       # The calculated shrinkage factor
    status::String          # "Safe", "Caution", or "Regime Shift"
end

"""
    get_dynamic_shrinkage(history_df, current_prob, current_odds; 
                          window=50, lambda=1.0, intrinsic_sigma=0.05)

Calculates the optimal bet size using the Baker-McHale shrinkage formula 
adjusted for empirical regime bias.

# Arguments
- `history_df`: DataFrame containing recent history with columns `:is_winner` and `:predicted_prob`
- `current_prob`: Your model's probability for the UPCOMING match
- `current_odds`: The decimal odds offered (e.g., 2.0)
- `window`: Number of past bets to analyze for regime detection
- `lambda`: Penalty factor for bias (higher = stricter)
- `intrinsic_sigma`: The standard deviation of your model's posterior (default ~5%)
"""
function get_dynamic_shrinkage(history_df::DataFrame, current_prob::Float64, current_odds::Float64; 
                               window::Int=50, lambda::Float64=1.0, intrinsic_sigma::Float64=0.05)
    
    # 1. RAW KELLY CALCULATION
    # ------------------------
    b = current_odds - 1.0
    s_star = ((b + 1) * current_prob - 1) / b
    
    # If raw Kelly says don't bet, return 0 immediately
    if s_star <= 0
        return 0.0, MarketRegime(0.0, 0.0, 0.0, "No Value")
    end

    # 2. REGIME DETECTION (Calculate Bias)
    # ------------------------------------
    # Check if we have enough history to make a decision
    if nrow(history_df) < window
        # Fallback: Use a conservative half-Kelly if not enough data
        println("⚠️ Not enough history for dynamic shrinking. Defaulting to k=0.5")
        return s_star * 0.5, MarketRegime(0.0, intrinsic_sigma^2, 0.5, "Insufficient Data")
    end

    # Get the last N bets
    recent_history = last(history_df, window)
    
    # Calculate Realized Win Rate (Empirical Truth)
    # Note: Ensure your DF has 'is_winner' as boolean or 1/0
    p_realized = mean(recent_history.is_winner)
    
    # Calculate Average Predicted Probability (Model Belief)
    # Note: If you don't store predicted_prob, we approximate it using 1/odds from history 
    # (assuming efficient market) or you must add it to your ledger.
    p_predicted_avg = mean(recent_history.predicted_prob)
    
    # BIAS = How far off is the model?
    # If model says 55% but reality is 40%, Bias is +0.15 (Dangerous!)
    bias = p_predicted_avg - p_realized
    
    # 3. VARIANCE INFLATION
    # ---------------------
    # Sigma_Empirical^2 = Intrinsic_Variance + Lambda * Bias^2
    # This is the "Meta-Model" core: Penalty for being wrong.
    sigma_emp_sq = (intrinsic_sigma^2) + (lambda * (bias^2))

    # 4. BAKER & MCHALE SHRINKAGE FORMULA
    # -----------------------------------
    # k = s*^2 / (s*^2 + ((b+1)/b)^2 * sigma^2)
    odds_scaling = ((b + 1) / b)^2
    
    k_optimal = (s_star^2) / ((s_star^2) + (odds_scaling * sigma_emp_sq))
    
    # Cap k at 1.0 (Never swell beyond Kelly in this risk-averse framework)
    k_optimal = min(k_optimal, 1.0)
    
    # Determine Status for logging
    status = if k_optimal < 0.2
        "🛑 REGIME SHIFT (Heavy Shrinkage)"
    elseif k_optimal < 0.7
        "⚠️ CAUTION (Moderate Bias)"
    else
        "✅ SAFE (High Confidence)"
    end

    # 5. FINAL STAKE
    final_stake = s_star * k_optimal
    
    return final_stake, MarketRegime(bias, sigma_emp_sq, k_optimal, status)
end



# --- MOCK DATA SETUP ---
# Create a dummy history representing a "Bad Run" (Regime Shift)
# 100 bets where we predicted 0.55 probability, but only won 40 times.
bad_run_df = DataFrame(
    is_winner = [rand() < 0.40 for _ in 1:100],  # 40% Win Rate (Realized)
    predicted_prob = fill(0.55, 100)             # 55% Win Rate (Predicted)
)

# --- CURRENT BET ---
# We have a new signal: Model says 55%, Odds are 2.0
curr_prob = 0.55
curr_odds = 2.0

# --- EXECUTION ---
stake, regime = get_dynamic_shrinkage(bad_run_df, curr_prob, curr_odds)

# --- REPORT ---
println("-"^40)
println("DYNAMIC KELLY REPORT")
println("-"^40)
println("Status:          $(regime.status)")
println("Model Bias:      $(round(regime.bias * 100, digits=2))%")
println("Inflated Var:    $(round(regime.empirical_var, digits=4))")
println("Shrinkage (k):   $(round(regime.regime_k, digits=3))")
println("-"^40)
println("Raw Kelly Stake: $(round(((curr_odds-1)*curr_prob - (1-curr_prob))/(curr_odds-1) * 100, digits=2))%")
println("Final Stake:     $(round(stake * 100, digits=2))%")
println("-"^40)


history_segment = subset(ledger.df, 
    :market_name => ByRow(==("OverUnder")), 
    :selection => ByRow(==(:over_25))
)

using DataFrames, Statistics, Plots

# 1. Setup Data Segment
# ---------------------
# Filter for Over 2.5 market and ensure it's sorted by date
history_segment = subset(ledger.df, 
    :market_name => ByRow(==("OverUnder")), 
    :selection => ByRow(==(:over_25))
)
sort!(history_segment, :date)

# 2. Simulation Parameters
# ------------------------
WINDOW = 20         # Lookback window for regime detection
LAMBDA = 1.0         # Penalty strength for bias
INTRINSIC_SIGMA = 0.05
STARTING_BANKROLL = 100.0

# Create columns to store our simulation results
history_segment.dynamic_stake = zeros(Float64, nrow(history_segment))
history_segment.dynamic_k     = zeros(Float64, nrow(history_segment))
history_segment.regime_bias   = zeros(Float64, nrow(history_segment))

# 3. The Backtest Loop
# --------------------
println("Running Dynamic Kelly Backtest on $(nrow(history_segment)) bets...")

for i in 1:nrow(history_segment)
    # A. Current Bet Details
    row = history_segment[i, :]
    curr_odds = row.odds
    # Use 1/odds as a proxy for predicted prob if 'predicted_prob' column is missing, 
    # otherwise use row.predicted_prob
    curr_prob = hasproperty(row, :predicted_prob) ? row.predicted_prob : (1.0 / row.odds) # simplified proxy

    # B. Calculate Raw Kelly (The "Signal")
    b = curr_odds - 1.0
    s_raw = 0.0
    if b > 0
        s_raw = ((b + 1) * curr_prob - 1) / b
    end
    s_raw = max(0.0, s_raw)

    # C. Dynamic Shrinkage Logic
    k_optimal = 1.0 # Default to full trust if no history
    bias = 0.0
    
    # Only run logic if we have enough history (i > WINDOW)
    if i > WINDOW
        # Get window of *previous* bets (1 to i-1)
        # Note: In real trading, you only know the result of bets settled before today.
        # We assume chronological order approximates this settlement order.
        window_start = i - WINDOW
        window_end = i - 1
        
        # Slicing the dataframe for the window
        past_bets = history_segment[window_start:window_end, :]
        
        # Calculate Realized Metrics
        # "is_winner" needs to be numeric (0.0 or 1.0) or boolean
        realized_win_rate = mean(past_bets.is_winner)
        
        # Calculate Predicted Average
        # Again, using 1/odds proxy if predicted_prob is missing
        if hasproperty(past_bets, :predicted_prob)
            avg_predicted = mean(past_bets.predicted_prob)
        else
            avg_predicted = mean(1.0 ./ past_bets.odds)
        end

        # 1. Bias Calculation (Regime Detection)
        bias = avg_predicted - realized_win_rate
        
        # 2. Variance Inflation
        sigma_emp_sq = (INTRINSIC_SIGMA^2) + (LAMBDA * (bias^2))
        
        # 3. Baker-McHale Shrinkage
        odds_scaling = ((b + 1) / b)^2
        
        # Avoid division by zero if s_raw is 0
        if s_raw > 1e-6
            k_optimal = (s_raw^2) / ((s_raw^2) + (odds_scaling * sigma_emp_sq))
        else
            k_optimal = 0.0
        end
        
        # Cap at 1.0
        k_optimal = min(k_optimal, 1.0)
    else
        # Warm-up period: Use conservative constant
        k_optimal = 0.5 
    end

    # D. Store Results
    history_segment.dynamic_k[i] = k_optimal
    history_segment.regime_bias[i] = bias
    history_segment.dynamic_stake[i] = s_raw * k_optimal
end

# 4. Calculate PnL Curves
# -----------------------
# Original (Actual) PnL
history_segment.pnl_original = cumsum(history_segment.pnl)

# Dynamic Simulation PnL
# Dynamic PnL = (If Winner: Odds-1, Else: -1) * Stake * Bankroll_Factor?
# For simple comparison, let's assume flat units or proportional growth. 
# Let's use simple unit PnL: (Outcome * Odds - 1) * Stake
history_segment.pnl_dynamic_step = [
    (row.is_winner ? (row.odds - 1) : -1.0) * row.dynamic_stake 
    for row in eachrow(history_segment)
]
history_segment.pnl_dynamic = cumsum(history_segment.pnl_dynamic_step)

println("Backtest Complete.")

# 5. Visualization
# ----------------
p1 = plot(history_segment.date, history_segment.pnl_original, label="Original Strategy", lw=2, title="Bankroll Comparison")
plot!(p1, history_segment.date, history_segment.pnl_dynamic, label="Dynamic Kelly", lw=2, color=:green)

p2 = plot(history_segment.date, history_segment.dynamic_k, label="Shrinkage Factor (k)", color=:red, alpha=0.5, title="Regime Trust (k) over Time")
# Add a line for Bias to see correlation
# plot!(p2, history_segment.date, history_segment.regime_bias, label="Bias", color=:blue, alpha=0.3)

plot(p1, p2, layout=(2,1), size=(800, 800))

#####
using DataFrames, Statistics, Printf

# 1. Setup Data Segment (Same as before)
# ---------------------
history_segment = subset(ledger.df, 
    :market_name => ByRow(==("OverUnder")), 
    :selection => ByRow(==(:over_25))
)
sort!(history_segment, :date)

# 2. Parameters
# ------------------------
WINDOW = 50
LAMBDA = 1.0
INTRINSIC_SIGMA = 0.05

println("DEBUG LOG START")
println("Total Bets in Segment: $(nrow(history_segment))")
println("-"^60)

# Define a helper to print row details
function print_debug_row(i, row, window_df)
    # A. Inputs
    curr_odds = row.odds
    # Proxy for predicted prob if column missing: 1/odds
    curr_prob = hasproperty(row, :predicted_prob) ? row.predicted_prob : (1.0 / row.odds)
    
    # B. Raw Kelly
    b = curr_odds - 1.0
    s_raw = (b > 0) ? ((b + 1) * curr_prob - 1) / b : 0.0
    s_raw = max(0.0, s_raw)

    # C. Regime Calc
    if i > WINDOW
        realized_win_rate = mean(window_df.is_winner)
        
        if hasproperty(window_df, :predicted_prob)
            avg_predicted = mean(window_df.predicted_prob)
        else
            avg_predicted = mean(1.0 ./ window_df.odds)
        end
        
        bias = avg_predicted - realized_win_rate
        sigma_emp_sq = (INTRINSIC_SIGMA^2) + (LAMBDA * (bias^2))
        
        odds_scaling = ((b + 1) / b)^2
        k_calc = (s_raw > 1e-6) ? (s_raw^2) / ((s_raw^2) + (odds_scaling * sigma_emp_sq)) : 0.0
        k_optimal = min(k_calc, 1.0)
        
        @printf("IDX: %4d | Date: %s\n", i, row.date)
        @printf("   INPUTS: Odds=%.2f, Prob=%.3f, RawKelly=%.3f%%\n", curr_odds, curr_prob, s_raw*100)
        @printf("   WINDOW: WinRate=%.3f, AvgPred=%.3f, BIAS=%.4f\n", realized_win_rate, avg_predicted, bias)
        @printf("   MATH:   SigmaSq=%.5f, OddsScale=%.2f\n", sigma_emp_sq, odds_scaling)
        @printf("   RESULT: k=%.4f -> FinalStake=%.3f%%\n", k_optimal, (s_raw * k_optimal)*100)
        println("-"^40)
    else
        @printf("IDX: %4d | Date: %s | WARMUP (k=0.5)\n", i, row.date)
    end
end

# 3. Print Specific Chunks
# ------------------------

# Chunk 1: The very start (Warmup)
println(">>> START OF DATA")
for i in 1:5
    print_debug_row(i, history_segment[i, :], DataFrame())
end

# Chunk 2: The Middle (Likely 23/24 season)
mid_idx = floor(Int, nrow(history_segment) / 2)
println("\n>>> MIDDLE OF DATA (Checking for Regime Shift)")
for i in mid_idx:(mid_idx+10)
    # Get previous window
    win_start = i - WINDOW
    win_end = i - 1
    window_df = history_segment[win_start:win_end, :]
    print_debug_row(i, history_segment[i, :], window_df)
end

# Chunk 3: The End (Current Season)
println("\n>>> END OF DATA")
last_idx = nrow(history_segment)
for i in (last_idx-5):last_idx
    win_start = i - WINDOW
    win_end = i - 1
    window_df = history_segment[win_start:win_end, :]
    print_debug_row(i, history_segment[i, :], window_df)
end


###
using DataFrames, Statistics, Printf

# 1. Setup Data Segment
history_segment = subset(ledger.df, 
    :market_name => ByRow(==("OverUnder")), 
    :selection => ByRow(==(:over_25))
)
sort!(history_segment, :date)

# 2. Parameters
WINDOW = 50
LAMBDA = 1.0
INTRINSIC_SIGMA = 0.05

println("REVISED DEBUG LOG START (Inferring Probabilities from Stake)")
println("-"^60)

function print_revised_debug(i, row, window_df)
    # A. Use EXISTING Stake as the Raw Signal
    s_raw = row.stake 
    curr_odds = row.odds
    b = curr_odds - 1.0
    
    # B. Infer what the model's probability was
    # Reverse Kelly: p = (s * b + 1) / (b + 1)
    # If stake is 0, we assume the model agreed with the market (1/odds) or lower.
    # For bias calculation, we default to 1/odds if stake is 0.
    curr_prob_inferred = (s_raw > 0) ? ((s_raw * b) + 1) / curr_odds : (1.0 / curr_odds)

    # C. Regime Calc
    if i > WINDOW
        # Calculate Realized Win Rate
        realized_win_rate = mean(window_df.is_winner)
        
        # Calculate Average Predicted Probability in the window
        # We must infer prob for every row in the window
        probs_in_window = map(eachrow(window_df)) do r
            b_local = r.odds - 1.0
            (r.stake > 0) ? ((r.stake * b_local) + 1) / r.odds : (1.0 / r.odds)
        end
        avg_predicted = mean(probs_in_window)
        
        # Bias Calculation
        bias = avg_predicted - realized_win_rate
        
        # Variance Inflation
        sigma_emp_sq = (INTRINSIC_SIGMA^2) + (LAMBDA * (bias^2))
        
        # Shrinkage
        odds_scaling = ((b + 1) / b)^2
        
        # Only calculate k if we actually have a raw stake to shrink
        k_optimal = 0.0
        if s_raw > 1e-6
            k_optimal = (s_raw^2) / ((s_raw^2) + (odds_scaling * sigma_emp_sq))
        else
            # If raw stake was 0, k is effectively 0 (or undefined), but let's show what it WOULD be
            # if we had a hypothetical 5% stake, just to see the regime trust.
            hypothetical_s = 0.05
            k_optimal = (hypothetical_s^2) / ((hypothetical_s^2) + (odds_scaling * sigma_emp_sq))
        end
        
        k_optimal = min(k_optimal, 1.0)
        
        @printf("IDX: %4d | Date: %s | Winner: %s\n", i, row.date, row.is_winner)
        @printf("   INPUTS: Odds=%.2f, Stake=%.3f%% -> InferredProb=%.3f\n", curr_odds, s_raw*100, curr_prob_inferred)
        @printf("   WINDOW: WinRate=%.3f, AvgProb=%.3f, BIAS=%.4f\n", realized_win_rate, avg_predicted, bias)
        @printf("   STATS:  SigmaSq=%.5f, Trust(k)=%.4f\n", sigma_emp_sq, k_optimal)
        
        if s_raw > 0
            @printf("   ACTION: RawStake: %.3f%% -> DynamicStake: %.3f%%\n", s_raw*100, (s_raw * k_optimal)*100)
        else
            @printf("   ACTION: No Bet (Regime Trust Level: %.2f%%)\n", k_optimal*100)
        end
        println("-"^40)
    end
end

# Print the Middle Chunk (Regime Shift Area)
mid_idx = floor(Int, nrow(history_segment) / 2)
println("\n>>> MIDDLE OF DATA")
for i in mid_idx:(mid_idx+10)
    win_start = i - WINDOW
    win_end = i - 1
    print_revised_debug(i, history_segment[i, :], history_segment[win_start:win_end, :])
end



function print_revised_debug_asymmetric(i, row, window_df)
    # [Setup inputs as before...]
    s_raw = row.stake 
    curr_odds = row.odds
    b = curr_odds - 1.0
    
    # Infer Model Probability from Stake
    curr_prob_inferred = (s_raw > 0) ? ((s_raw * b) + 1) / curr_odds : (1.0 / curr_odds)

    if i > WINDOW
        realized_win_rate = mean(window_df.is_winner)
        
        # Calculate Avg Predicted Prob in Window
        probs_in_window = map(eachrow(window_df)) do r
            b_local = r.odds - 1.0
            (r.stake > 0) ? ((r.stake * b_local) + 1) / r.odds : (1.0 / r.odds)
        end
        avg_predicted = mean(probs_in_window)
        
        # --- THE FIX IS HERE ---
        raw_bias = avg_predicted - realized_win_rate
        
        # Only penalize if we are OVER-estimating (Predicted > Realized)
        # If we predict 57% but win 80%, bias is negative -> penalty becomes 0
        penalty_bias = max(0.0, raw_bias)
        
        sigma_emp_sq = (INTRINSIC_SIGMA^2) + (LAMBDA * (penalty_bias^2))
        # -----------------------

        odds_scaling = ((b + 1) / b)^2
        
        k_optimal = 0.0
        if s_raw > 1e-6
            k_optimal = (s_raw^2) / ((s_raw^2) + (odds_scaling * sigma_emp_sq))
        else
            hypothetical_s = 0.05
            k_optimal = (hypothetical_s^2) / ((hypothetical_s^2) + (odds_scaling * sigma_emp_sq))
        end
        
        k_optimal = min(k_optimal, 1.0)
        
        @printf("IDX: %4d | Date: %s | Winner: %s\n", i, row.date, row.is_winner)
        @printf("   WINDOW: Realized=%.2f, Pred=%.2f, RawBias=%.3f\n", realized_win_rate, avg_predicted, raw_bias)
        
        if raw_bias < 0
            println("   NOTE:   Winning Streak! (Bias Negative). No Penalty Applied.")
        else
            @printf("   NOTE:   Losing Streak! Penalty Bias: %.3f\n", penalty_bias)
        end
        
        @printf("   STATS:  SigmaSq=%.5f, Trust(k)=%.4f\n", sigma_emp_sq, k_optimal)
        
        if s_raw > 0
            @printf("   ACTION: RawStake: %.3f%% -> DynamicStake: %.3f%%\n", s_raw*100, (s_raw * k_optimal)*100)
        else
            @printf("   ACTION: No Bet (Regime Trust Level: %.2f%%)\n", k_optimal*100)
        end
        println("-"^40)
    end
end


mid_idx = floor(Int, nrow(history_segment) / 2)
println("\n>>> MIDDLE OF DATA (ASYMMETRIC TEST)")

for i in mid_idx:(mid_idx+10)
    win_start = i - WINDOW
    win_end = i - 1
    # Calling the NEW function name here:
    print_revised_debug_asymmetric(i, history_segment[i, :], history_segment[win_start:win_end, :])
end
