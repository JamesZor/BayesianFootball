using Revise
using BayesianFootball
using DataFrames
using Dates
using JSON3
using CSV
using Statistics
using SHA


# ==============================================================================
# PART 2: MATCH DAY EXECUTION SCRIPT
# ==============================================================================

println("=== 1. INITIALIZATION & DATA LOADING ===")

ds = BayesianFootball.Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

println("Loading Model & Features...")
# UPDATE THIS PATH to your target model!
saved_folders = BayesianFootball.Experiments.list_experiments("exp/market_runs/april"; data_dir="./data")
m1 = BayesianFootball.Experiments.load_experiment(saved_folders[1])

feature_collection = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, 
    m1.config.splitter
)

last_split_idx = length(m1.training_results)
chain1 = m1.training_results[last_split_idx][1]
feature_set = feature_collection[last_split_idx][1]

# ==============================================================================
# PART 3: Inferences
# ==============================================================================

#=
julia> feature_set.data[:team_map]
Dict{String, Int64} with 23 entries:
  "edinburgh-city-fc"            => 10
  "arbroath"                     => 3
  "clyde-fc"                     => 5
  "queen-of-the-south"           => 19
  "stirling-albion"              => 21
  "east-kilbride"                => 9
  "hamilton-academical"          => 14
  "forfar-athletic"              => 13
  "peterhead"                    => 18
  "stenhousemuir"                => 20
  "bonnyrigg-rose"               => 4
  "dumbarton"                    => 7
  "inverness-caledonian-thistle" => 15
  "the-spartans-fc"              => 23
  "alloa-athletic"               => 1
  "kelty-hearts-fc"              => 16
  "montrose"                     => 17
  "stranraer"                    => 22
  "cove-rangers"                 => 6
  "falkirk-fc"                   => 12
  "elgin-city"                   => 11
  "east-fife"                    => 8
  "annan-athletic"               => 2
=#
using Dates 

match_to_predict = DataFrame(
    match_id = [i for i in 1:9],
    match_week = [ 32 for _ in 1:9],
    match_date = [ today() for _ in 1:9],
    home_team = ["cove-rangers", "kelty-hearts-fc", "montrose", "stenhousemuir", 
                  "clyde-fc", "east-kilbride", "edinburgh-city-fc", "forfar-athletic", "stranraer"], 
    away_team = ["alloa-athletic", "queen-of-the-south", "east-fife", "hamilton-academical",
                 "stirling-albion", "annan-athletic", "the-spartans-fc", "elgin-city", "dumbarton"]
)

raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m1.config.model, match_to_predict, feature_set, chain1
)

# 5. Convert to LatentStates (Preserving the restored String IDs)
function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    # Assumes all entries have same keys
    first_entry = raw_preds[ids[1]]
    for k in keys(first_entry)
        cols[k] = [raw_preds[i][k] for i in ids]
    end
    return DataFrame(cols)
end


latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m1.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents)

# ========================================================================
# 1. NON-MUTATING SHIFT APPLIER
# ========================================================================
# ========================================================================
# 1. THE CALIBRATION ENGINE
# ========================================================================
using DataFrames
using GLM
using Statistics


market_data = Data.prepare_market_data(ds)

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


"""
Takes an array of MCMC probability samples and applies a logit shift, 
returning a newly allocated array of shifted probabilities.
"""
function apply_logit_shift(dist_array::Vector{Float64}, C_shift::Float64)
    eps = 1e-6
    # 1. Clamp to prevent log(0) or log(1)
    clamped = clamp.(dist_array, eps, 1.0 - eps)
    
    # 2. Convert to Logits
    logits = log.(clamped ./ (1.0 .- clamped))
    
    # 3. Apply the Shift
    shifted_logits = logits .+ C_shift
    
    # 4. Convert back to Probabilities (Inverse Logit / Sigmoid)
    return 1.0 ./ (1.0 .+ Base.exp.(.-shifted_logits))
end

# ========================================================================
# AUTOMATED CALIBRATION LOOP
# ========================================================================

"""
Iterates over a list of market selections, runs the GLM calibration, 
and returns a populated dictionary of logit shifts.
"""
function compute_all_shifts(ppd_df::DataFrame, market_df::DataFrame, selections::Vector{Symbol})
    # Initialize an empty dictionary with explicit types
    calculated_shifts = Dict{Symbol, Float64}()
    
    println("=== Starting Global Calibration ===")
    
    for sel in selections
        try
            # Run your existing single-market calibration function
            shift_val = calculate_shift(ppd_df, market_df, sel)
            calculated_shifts[sel] = shift_val
            
        catch e
            println("⚠️ Warning: Failed to calculate shift for :$sel.")
            println("   Reason: ", e)
            println("   Defaulting shift to 0.0 (No Shift applied).")
            # If the GLM fails, default to 0.0 so the pipeline survives
            calculated_shifts[sel] = 0.0
        end
    end
    
    println("\n=== Global Calibration Complete ===")
    return calculated_shifts
end
target_markets = [:btts_yes, :draw, :over_15, :over_25, :over_35]

# NOTE: Need to ran on the back test exp, since we need to compare historical ppd to the matches - @ablation_study/plat_cat.jl
calculated_shifts = compute_all_shifts(ppd.df, market_data.df, target_markets)
#=
=== Starting Global Calibration ===

--- Layer 2 Calibration for: :btts_yes ---
Original Model Mean: 0.5199
Actual Hit Rate:     0.5416
>> Calculated Logit Shift (C): 0.0882

--- Layer 2 Calibration for: :draw ---
Original Model Mean: 0.2455
Actual Hit Rate:     0.2551
>> Calculated Logit Shift (C): 0.0514

--- Layer 2 Calibration for: :over_15 ---
Original Model Mean: 0.7409
Actual Hit Rate:     0.7875
>> Calculated Logit Shift (C): 0.2629

--- Layer 2 Calibration for: :over_25 ---
Original Model Mean: 0.5103
Actual Hit Rate:     0.5347
>> Calculated Logit Shift (C): 0.0997

--- Layer 2 Calibration for: :over_35 ---
Original Model Mean: 0.3025
Actual Hit Rate:     0.3196
>> Calculated Logit Shift (C): 0.0812

=== Global Calibration Complete ===
Dict{Symbol, Float64} with 5 entries:
  :btts_yes => 0.0882462
  :over_35  => 0.0812359
  :draw     => 0.0513683
  :over_15  => 0.262941
  :over_25  => 0.0997077
=#
ppds_compare = copy(ppds)
ppds_compare.shifted_distribution = copy(ppds_compare.distribution)

# Apply the shifts dynamically based on the dictionary
for (target_selection, shift_val) in calculated_shifts
    mask = ppds_compare.selection .== target_selection
    
    # Only map over the rows that match the current selection
    ppds_compare.shifted_distribution[mask] = map(
        d -> apply_logit_shift(d, shift_val), 
        ppds_compare.distribution[mask]
    )
end

# ========================================================================
# 3. CALCULATE SHIFTED ODDS
# ========================================================================

q_low  = 0.30  # 10th percentile (Conservative / Ask)
q_high = 0.70  # 90th percentile (Optimistic / Bid)

# Calculate probabilities for the newly shifted distributions
ppds_compare.shift_prob_lower = [quantile(d, q_low) for d in ppds_compare.shifted_distribution]
ppds_compare.shift_prob_mean  = [mean(d) for d in ppds_compare.shifted_distribution]
ppds_compare.shift_prob_upper = [quantile(d, q_high) for d in ppds_compare.shifted_distribution]

# Translate Shifted Probabilities to Odds
ppds_compare.shift_odds_ask  = 1.0 ./ ppds_compare.shift_prob_lower 
ppds_compare.shift_odds_mean = 1.0 ./ ppds_compare.shift_prob_mean  
ppds_compare.shift_odds_bid  = 1.0 ./ ppds_compare.shift_prob_upper 


# 2. Calculate probabilities for these quantiles
ppd.df.prob_lower = [quantile(d, q_low) for d in ppd.df.distribution]
ppd.df.prob_mean  = [mean(d) for d in ppd.df.distribution]
ppd.df.prob_upper = [quantile(d, q_high) for d in ppd.df.distribution]

# 3. Translate Probabilities to Odds (Odds = 1 / Probability)
ppd.df.odds_ask  = 1.0 ./ ppd.df.prob_lower # Max price you'd offer (Bookie Ask)
ppd.df.odds_mean = 1.0 ./ ppd.df.prob_mean  # True "Fair" Market Odds
ppd.df.odds_bid  = 1.0 ./ ppd.df.prob_upper # Min price you'd accept (Bookie Bid)

market_selection = [ :btts_yes, :draw, :over_15, :over_25, :over_35]

ppds_selected = subset( ppd.df, :selection => ByRow(in(market_selection)))

ppds = leftjoin( ppds_selected, match_to_predict, on=:match_id)


# ========================================================================
# 4. VIEW THE DIFFERENCES
# ========================================================================

# Select the relevant columns to view them side-by-side
final_view = select(ppds_compare,
    :match_id, 
    :home_team, 
    :away_team, 
    :market_name, 
    :selection,
    # Raw Odds
    :odds_ask => :raw_ask, 
    :odds_mean => :raw_mean, 
    :odds_bid => :raw_bid,
    # Shifted Odds
    :shift_odds_ask => :shifted_ask, 
    :shift_odds_mean => :shifted_mean, 
    :shift_odds_bid => :shifted_bid
)

sort!(final_view, [:match_id, :market_name])
display(final_view)


# -----

using Optim
using DataFrames

# (Assuming your BayesianKelly struct and compute_stake function are already defined here)
#

"""
Generates a DataFrame mapping bookmaker odds to recommended Bayesian Kelly 
stake sizes for both the Raw and Calibrated distributions.
"""
function generate_betfair_sheet(match_row::DataFrameRow, odds_range::AbstractVector; min_edge::Float64=0.0)
    signal = BayesianKelly(min_edge=min_edge)
    
    raw_stakes = Float64[]
    calib_stakes = Float64[]
    
    for odds in odds_range
        # 1. Calculate stake for the Uncalibrated (Raw) MCMC array
        raw_s = BayesianFootball.Signals.compute_stake(signal, match_row.distribution, odds)
        push!(raw_stakes, raw_s)
        
        # 2. Calculate stake for the Calibrated (Shifted) MCMC array
        calib_s = BayesianFootball.Signals.compute_stake(signal, match_row.shifted_distribution, odds)
        push!(calib_stakes, calib_s)
    end
    
    # Format into a clean DataFrame (Converting decimals to percentages for readability)
    df = DataFrame(
        Betfair_Price = odds_range,
        Raw_Stake_Pct = round.(raw_stakes .* 100, digits=2),
        Calib_Stake_Pct = round.(calib_stakes .* 100, digits=2)
    )
    
    # Filter out rows where both stakes are 0.0 to keep the sheet concise
    return filter(row -> row.Raw_Stake_Pct > 0 || row.Calib_Stake_Pct > 0, df)
end


# ========================================================================
# GENERATE LIVE TRADING SHEETS
# ========================================================================

# 1. Define the range of prices you expect to see on Betfair (e.g., 1.10 to 3.00 in 0.05 ticks)
live_prices = collect(1.10:0.05:3.00)

# 2. Isolate the specific match and markets you want to trade
match_name = "cove-rangers" 
match_data = filter(:home_team => ==(match_name), ppds_compare)

# Extract the specific rows for Over 1.5 and Over 2.5
row_o15 = filter(:selection => ==(:over_15), match_data)[1, :]
row_o25 = filter(:selection => ==(:over_25), match_data)[1, :]

# 3. Generate the sheets (We'll use a 0.0% min_edge filter here so you can see the pure Kelly curve)
println("=== BETFAIR TRADING SHEET: $(match_name) (OVER 1.5) ===")
sheet_o15 = generate_betfair_sheet(row_o15, live_prices, min_edge=0.0)
display(sheet_o15)

println("\n=== BETFAIR TRADING SHEET: $(match_name) (OVER 2.5) ===")
sheet_o25 = generate_betfair_sheet(row_o25, live_prices, min_edge=0.0)
display(sheet_o25)

describe(subset(market_data.df , :selection => ByRow(isequal(:over_15))).odds_close)


describe(subset(market_data.df , :selection => ByRow(isequal(:over_25))).odds_close)


# ========================================================================
# 4. BATCH GENERATION OF TRADING SHEETS
# ========================================================================

# Define realistic Betfair ranges based on your historical describe() data
const MARKET_ODDS_RANGES = Dict(
    # Finer ticks (0.02) for low-odds markets, capping at historical max
    :over_15  => collect(1.04:0.02:1.80), 
    
    # Standard ticks (0.05) for mid-odds markets
    :over_25  => collect(1.70:0.01:1.90), 
    :btts_yes => collect(1.40:0.05:3.00),
    
    # Wider ticks (0.10) for higher-odds markets
    :over_35  => collect(2.00:0.10:6.00),
    :draw     => collect(2.50:0.10:5.50)
)

"""
Generates a DataFrame mapping bookmaker odds to recommended Bayesian Kelly 
stake sizes for both the Raw and Calibrated distributions. (Unfiltered version)
"""
function generate_betfair_sheet(match_row::DataFrameRow, odds_range::AbstractVector; min_edge::Float64=0.0)
    signal = BayesianKelly(min_edge=min_edge)
    
    raw_stakes = Float64[]
    calib_stakes = Float64[]
    
    for odds in odds_range
        # Calculate stakes
        raw_s = BayesianFootball.Signals.compute_stake(signal, match_row.distribution, odds)
        push!(raw_stakes, raw_s)
        
        calib_s = BayesianFootball.Signals.compute_stake(signal, match_row.shifted_distribution, odds)
        push!(calib_stakes, calib_s)
    end
    
    # Return the FULL DataFrame without filtering out the 0.0s
    return DataFrame(
        Betfair_Price = round.(odds_range, digits=2), # Rounded for clean display
        Raw_Stake_Pct = round.(raw_stakes .* 100, digits=2),
        Calib_Stake_Pct = round.(calib_stakes .* 100, digits=2)
    )
end


# Execute the batch generator
target_trading_markets = [:over_25]
all_sheets = generate_all_trading_sheets(ppds_compare, target_trading_markets, min_edge=0.0)

# (Optional) If you wanted to isolate a specific sheet later in your code:
# over_25_sheet = all_sheets["cove-rangers_vs_alloa-athletic"][:over_25]
#
#
using DataFrames
using Printf

# 1. Create the Live Market DataFrame from the screenshots
live_market = DataFrame(
    home_team = ["cove-rangers", "kelty-hearts-fc", "montrose", "stenhousemuir", 
                 "clyde-fc", "east-kilbride", "edinburgh-city-fc", "forfar-athletic", "stranraer"],
    live_odds_o15 = [1.25, 1.28, 1.22, 1.33, 1.18, 1.16, 1.25, 1.34, 1.24],
    live_odds_o25 = [1.75, 1.86, 1.68, 1.98, 1.57, 1.54, 1.73, 2.02, 1.74]
)

# 2. Initialize your Kelly Signal (Using 0.0% min_edge to see the raw output)
# Note: You can change this to 0.01 (1%) to filter out marginal trades!
signal = BayesianKelly(min_edge=0.0)

println("===============================================================")
println(" 📝 PAPER TRADING BOARD: EXACT KELLY STAKES")
println("===============================================================")

# 3. Loop through the matches and calculate stakes
for row in eachrow(live_market)
    team = row.home_team
    
    # Isolate the match predictions from your main DataFrame
    match_preds = filter(:home_team => ==(team), ppds_compare)
    
    # Over 1.5 Calculations
    row_o15 = filter(:selection => ==(:over_15), match_preds)[1, :]
    raw_s_o15 = BayesianFootball.Signals.compute_stake(signal, row_o15.distribution, row.live_odds_o15)
    calib_s_o15 = BayesianFootball.Signals.compute_stake(signal, row_o15.shifted_distribution, row.live_odds_o15)
    
    # Over 2.5 Calculations
    row_o25 = filter(:selection => ==(:over_25), match_preds)[1, :]
    raw_s_o25 = BayesianFootball.Signals.compute_stake(signal, row_o25.distribution, row.live_odds_o25)
    calib_s_o25 = BayesianFootball.Signals.compute_stake(signal, row_o25.shifted_distribution, row.live_odds_o25)
    
    # Only print if at least ONE of the calibrated stakes is > 0
    if calib_s_o15 > 0 || calib_s_o25 > 0
        println("\nMATCH: $(uppercase(team))")
        
        if calib_s_o15 > 0
            @printf("  OVER 1.5 (Odds: %.2f) | Raw Stake: %5.2f%% | Calib Stake: %5.2f%%\n", 
                    row.live_odds_o15, raw_s_o15 * 100, calib_s_o15 * 100)
        end
        
        if calib_s_o25 > 0
            @printf("  OVER 2.5 (Odds: %.2f) | Raw Stake: %5.2f%% | Calib Stake: %5.2f%%\n", 
                    row.live_odds_o25, raw_s_o25 * 100, calib_s_o25 * 100)
        end
    end
end
println("\n===============================================================")
