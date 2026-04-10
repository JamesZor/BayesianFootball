
market_data = Data.prepare_market_data(ds)



exp = loaded_results_[2]


latents = Predictions.extract_oos_predictions(ds, exp)



ppd = Predictions.model_inference(latents)


# ========================================================================
# 1. THE CALIBRATION ENGINE
# ========================================================================
using DataFrames
using GLM
using Statistics

function calculate_shift(ppd_df::DataFrame, market_df::DataFrame, target_selection::Symbol)
    println("\n--- Layer 2 Calibration for: :$target_selection ---")
    
    # 1. Isolate the specific market predictions
    sub_df = filter(:selection => ==(target_selection), ppd_df)
    
    # 2. Join with market data to get the actual results
    calib_df = innerjoin(
        market_df,
        sub_df,
        on = [:match_id, :market_name, :market_line, :selection]
    )
    
    # 3. Use the pre-calculated 'is_winner' boolean
    calib_df.actual = Float64.(calib_df.is_winner)
    
    # 4. Extract the MEAN probability of the MCMC distribution
    calib_df.mean_prob = [mean(dist) for dist in calib_df.distribution]
    
    println("Original Model Mean: ", round(mean(calib_df.mean_prob), digits=4))
    println("Actual Hit Rate:     ", round(mean(calib_df.actual), digits=4))

    # 5. Convert to Log-Odds
    eps = 1e-6
    clamped = clamp.(calib_df.mean_prob, eps, 1.0 - eps)
    calib_df.logit_prob = log.(clamped ./ (1.0 .- clamped))
    
    # 6. Fit the GLM with the offset keyword argument
    model = glm(@formula(actual ~ 1), calib_df, Binomial(), LogitLink(), offset=calib_df.logit_prob)
    
    C_shift = coef(model)[1]
    println(">> Calculated Logit Shift (C): ", round(C_shift, digits=4))
    
    return C_shift
end

# ========================================================================
# 2. THE SHIFT APPLIER
# ========================================================================
"""
Applies the calculated shift to EVERY individual sample in the MCMC 
distribution array, preserving the uncertainty shape for Kelly betting.
"""
function apply_shift!(ppd_df::DataFrame, target_selection::Symbol, C_shift::Float64)
    # Find the rows we want to mutate
    mask = ppd_df.selection .== target_selection
    eps = 1e-6
    
    # Map over the array of arrays, shifting every single sample
    ppd_df.distribution[mask] = map(ppd_df.distribution[mask]) do dist_array
        
        # 1. Clamp to prevent log(0)
        clamped = clamp.(dist_array, eps, 1.0 - eps)
        
        # 2. Convert to Logits
        logits = log.(clamped ./ (1.0 .- clamped))
        
        # 3. Apply the Shift
        shifted_logits = logits .+ C_shift
        
        # 4. Convert back to Probabilities (Sigmoid)
        new_probs = 1.0 ./ (1.0 .+ Base.exp.(.-shifted_logits))
        
        return new_probs
    end
    
    println("Successfully shifted all MCMC samples for :$target_selection in place.")
end


# ========================================================================
# 3. RUN IT
# ========================================================================

# Let's assume your ppd is a DataFrame (if it's a struct, use ppd.df)
# Let's assume market_data is a DataFrame (if struct, use market_data.df)

# 1. Find the shift required for Over 2.5
C_over = calculate_shift(ppd.df, market_data.df, :over_25)

# 2. Apply it directly to the PPD object in memory
apply_shift!(ppd.df, :over_25, C_over)

# (Optional) Do the exact same for Under 2.5
# C_under = calculate_shift(ppd, market_data.df, :under_25)
# apply_shift!(ppd, :under_25, C_under)
#
ppd.df



sig_result = Signals.process_signals(ppd, market_data.df, my_signals; odds_column=:odds_close)



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
    
    m_name = model_name(exp.config.model)
    m_params = model_parameters(exp.config.model)

    df.model_name = fill(m_name, nrow(df))
    df.model_parameters = fill(m_params, nrow(df))


subset(df, :selection => ByRow(isequal(:over_25)))


# 1. Isolate the Over 2.5 DataFrame
over_25_df = subset(df, :selection => ByRow(isequal(:over_25)))

# 2. Calculate the core metrics
total_stake = sum(over_25_df.stake)
total_pnl = sum(over_25_df.pnl)

# Handle the edge case where no bets were placed to avoid dividing by zero
if total_stake > 0
    roi = (total_pnl / total_stake) * 100
    bets_placed = count(>(0), over_25_df.stake)
    
    println("--- Over 2.5 Backtest Results ---")
    println("Matches Evaluated: ", nrow(over_25_df))
    println("Bets Placed:       ", bets_placed)
    println("Total Staked:      ", round(total_stake, digits=2), " units")
    println("Total PnL:         ", round(total_pnl, digits=2), " units")
    println("ROI:               ", round(roi, digits=2), "%")
else
    println("No bets were placed (Total stake is 0).")
end



# ========================================================================
# 4. THE EVALUATION ENGINE (PnL, ROI, Win Rate)
# ========================================================================
function evaluate_market(ppd_obj, market_df, signals_obj, target_selection::Symbol, label::String)
    # 1. Run the Kelly Signal Agent
    sig_result = Signals.process_signals(ppd_obj, market_df, signals_obj; odds_column=:odds_close)
    df = sig_result.df

    # 2. Calculate PnL
    df.pnl = map(eachrow(df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    # 3. Isolate the specific market
    target_df = subset(df, :selection => ByRow(isequal(target_selection)))
    
    # 4. Calculate Metrics
    total_stake = sum(target_df.stake)
    total_pnl = sum(target_df.pnl)
    
    println("=== $label ($target_selection) ===")
    
    if total_stake > 0
        roi = (total_pnl / total_stake) * 100
        
        # Calculate Win Rate
        bets_placed = count(>(0), target_df.stake)
        winning_bets = count(r -> r.stake > 0 && r.is_winner == true, eachrow(target_df))
        win_rate = (winning_bets / bets_placed) * 100
        
        println("Matches Evaluated: ", nrow(target_df))
        println("Bets Placed:       ", bets_placed)
        println("Winning Bets:      ", winning_bets)
        println("Win Rate:          ", round(win_rate, digits=2), "%")
        println("Total Staked:      ", round(total_stake, digits=2), " units")
        println("Total PnL:         ", round(total_pnl, digits=2), " units")
        println("ROI:               ", round(roi, digits=2), "%\n")
    else
        println("Matches Evaluated: ", nrow(target_df))
        println("No bets were placed (Total stake is 0).\n")
    end
    
    return target_df
end

# ========================================================================
# 5. THE SIDE-BY-SIDE COMPARISON RUN
# ========================================================================

ppd = Predictions.model_inference(latents)

sym = :over_15
# STEP A: Test the Uncalibrated (Raw) Model First
# We do this before applying the shift since apply_shift! mutates memory.
raw_df = evaluate_market(ppd, market_data.df, my_signals, sym, "1. UNCALIBRATED BASELINE");

# STEP B: Find the shift and apply it to the PPD
C_over = calculate_shift(ppd.df, market_data.df, sym)
apply_shift!(ppd.df, sym, C_over)

# STEP C: Test the Calibrated Model
# The ppd object now holds the centered probabilities
calibrated_df = evaluate_market(ppd, market_data.df, my_signals, sym, "2. LAYER 2 CALIBRATED");

#=
julia> raw_df = evaluate_market(ppd, market_data.df, my_signals, :over_25, "1. UNCALIBRATED BASELINE")                                                                                                                                                                                                                      
=== 1. UNCALIBRATED BASELINE (over_25) ===                                                                                                                                                                                                                                                                                  
Matches Evaluated: 1192                                                                                                                                                                                                                                                                                                     
Bets Placed:       262                                                                                                                                                                                                                                                                                                      
Winning Bets:      149                                                                                                                                                                                                                                                                                                      
Win Rate:          56.87%                                                                                                                                                                                                                                                                                                   
Total Staked:      8.15 units                                                                                                                                                                                                                                                                                               
Total PnL:         1.58 units                                                                                                                                                                                                                                                                                               
ROI:               19.42%   



julia> calibrated_df = evaluate_market(ppd, market_data.df, my_signals, :over_25, "2. LAYER 2 CALIBRATED")                                                                                                                                                                                                                  
=== 2. LAYER 2 CALIBRATED (over_25) ===                                                                                                                                                                                                                                                                                     
Matches Evaluated: 1192                                                                                                                                                                                                                                                                                                     
Bets Placed:       450                                                                                                                                                                                                                                                                                                      
Winning Bets:      251                                                                                                                                                                                                                                                                                                      
Win Rate:          55.78%                                                                                                                                                                                                                                                                                                   
Total Staked:      17.83 units                                                                                                                                                                                                                                                                                              
Total PnL:         2.33 units                                                                                                                                                                                                                                                                                               
ROI:               13.04%      

# -------
julia> # STEP A: Test the Uncalibrated (Raw) Model First
       # We do this before applying the shift since apply_shift! mutates memory.
       raw_df = evaluate_market(ppd, market_data.df, my_signals, sym, "1. UNCALIBRATED BASELINE");
=== 1. UNCALIBRATED BASELINE (over_35) ===
Matches Evaluated: 1134
Bets Placed:       273
Winning Bets:      81
Win Rate:          29.67%
Total Staked:      5.55 units
Total PnL:         0.76 units
ROI:               13.71%


julia> # STEP B: Find the shift and apply it to the PPD
       C_over = calculate_shift(ppd.df, market_data.df, sym)

--- Layer 2 Calibration for: :over_35 ---
Original Model Mean: 0.3024
Actual Hit Rate:     0.3201
>> Calculated Logit Shift (C): 0.0842
0.08418444140271787

julia> apply_shift!(ppd.df, sym, C_over)
Successfully shifted all MCMC samples for :over_35 in place.

julia> # STEP C: Test the Calibrated Model
       # The ppd object now holds the centered probabilities
       calibrated_df = evaluate_market(ppd, market_data.df, my_signals, sym, "2. LAYER 2 CALIBRATED");
=== 2. LAYER 2 CALIBRATED (over_35) ===
Matches Evaluated: 1134
Bets Placed:       408
Winning Bets:      130
Win Rate:          31.86%
Total Staked:      10.07 units
Total PnL:         0.81 units
ROI:               8.05%




julia> sym = :over_15
:over_15

julia> # STEP A: Test the Uncalibrated (Raw) Model First
       # We do this before applying the shift since apply_shift! mutates memory.
       raw_df = evaluate_market(ppd, market_data.df, my_signals, sym, "1. UNCALIBRATED BASELINE");
=== 1. UNCALIBRATED BASELINE (over_15) ===
Matches Evaluated: 1134
Bets Placed:       91
Winning Bets:      79
Win Rate:          86.81%
Total Staked:      3.08 units
Total PnL:         0.54 units
ROI:               17.47%


julia> # STEP B: Find the shift and apply it to the PPD
       C_over = calculate_shift(ppd.df, market_data.df, sym)

--- Layer 2 Calibration for: :over_15 ---
Original Model Mean: 0.7408
Actual Hit Rate:     0.7884
>> Calculated Logit Shift (C): 0.2685
0.26845227240579467

julia> apply_shift!(ppd.df, sym, C_over)
Successfully shifted all MCMC samples for :over_15 in place.

julia> # STEP C: Test the Calibrated Model
       # The ppd object now holds the centered probabilities
       calibrated_df = evaluate_market(ppd, market_data.df, my_signals, sym, "2. LAYER 2 CALIBRATED");
=== 2. LAYER 2 CALIBRATED (over_15) ===
Matches Evaluated: 1134
Bets Placed:       435
Winning Bets:      341
Win Rate:          78.39%
Total Staked:      26.32 units
Total PnL:         1.54 units
ROI:               5.87%

=#
