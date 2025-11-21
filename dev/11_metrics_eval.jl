using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics

save_dir = "dev_exp/simple_poisson/"

#####
# --- basic set up
######

data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)


# ---  Define Training Configuration ---
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=600, n_chains=2, n_warmup=100) # Use renamed struct
# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)



seasons_to_train = ["20/21","21/22","22/23","23/24","24/25"]

season_to_load = seasons_to_train[1]
season_to_load_str = save_dir * "s_" * replace(season_to_load, "/" => "_") * ".jld2"
results = JLD2.load_object(season_to_load_str)


df = filter(row -> row.season==season_to_load, data_store.matches)
# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)
df.split_col = max.(0, df.match_week .- 14);

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)

BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)


split_col_name = :split_col
all_splits = sort(unique(ds.matches[!, split_col_name]))
prediction_split_keys = all_splits[2:end] 
grouped_matches = groupby(ds.matches, split_col_name)

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


##############################
# +++ helper functions 
##############################
function remove_vig(odds::NamedTuple)
    # Dictionary to store the new fair odds
    fair_data = Dict{Symbol, Float64}()

    # --- Helper Function ---
    # Calculates fair odds using basic normalization: FairOdd = Odd * Overround
    function normalize_market(market_odds...)
        # 1. Calculate Implied Probabilities (1 / Odd)
        implied_probs = [1.0 / o for o in market_odds]
        
        # 2. Calculate the Overround (Sum of implied probabilities)
        overround = sum(implied_probs)
        
        # 3. Return Fair Odds (Odd * Overround)
        # This is mathematically equivalent to: 1 / (Implied_Prob / Overround)
    return [round(o * overround, digits=2) for o in market_odds]
    end

    # --- 1. Handle 1x2 (Home/Draw/Away) ---
    if haskey(odds, :home) && haskey(odds, :draw) && haskey(odds, :away)
        f_home, f_draw, f_away = normalize_market(odds.home, odds.draw, odds.away)
        fair_data[:home] = f_home
        fair_data[:draw] = f_draw
        fair_data[:away] = f_away
    end

    # --- 2. Handle BTTS (Both Teams To Score) ---
    if haskey(odds, :btts_yes) && haskey(odds, :btts_no)
        f_yes, f_no = normalize_market(odds.btts_yes, odds.btts_no)
        fair_data[:btts_yes] = f_yes
        fair_data[:btts_no]  = f_no
    end

    # --- 3. Handle Over/Under Markets (Dynamic) ---
    # Iterate over keys to find all "over_XX" and match with "under_XX"
    for k in keys(odds)
        s_key = string(k)
        if startswith(s_key, "over_")
            # Extract the suffix (e.g., "25" from "over_25")
            suffix = s_key[6:end] 
            under_sym = Symbol("under_" * suffix)

            # If the matching under exists, process the pair
            if haskey(odds, under_sym)
                val_over = getproperty(odds, k)
                val_under = getproperty(odds, under_sym)
                
                f_over, f_under = normalize_market(val_over, val_under)
                
                fair_data[k] = f_over
                fair_data[under_sym] = f_under
            end
        end
    end

    # Convert the Dict back to a NamedTuple
    return (; fair_data...)
end

function get_margins(odds::NamedTuple)
    margins = Dict{Symbol, Float64}()
    
    # Helper to calc margin
    calc_margin(odds...) = sum(1.0 ./ o for o in odds)

    # 1x2
    if haskey(odds, :home)
        margins[:x12] = calc_margin(odds.home, odds.draw, odds.away)
    end

    # BTTS
    if haskey(odds, :btts_yes)
        margins[:btts] = calc_margin(odds.btts_yes, odds.btts_no)
    end

    # Over/Under
    for k in keys(odds)
        s_key = string(k)
        if startswith(s_key, "over_")
            suffix = s_key[6:end]
            u_key = Symbol("under_" * suffix)
            if haskey(odds, u_key)
                margins[Symbol("ou_" * suffix)] = calc_margin(getproperty(odds, k), getproperty(odds, u_key))
            end
        end
    end
    
    return margins
end

"""
(UPDATED) Calculates the Binary KL Divergence, D_KL(Q || P).
"""
function kl_divergence(q::Number, p::Number)
    epsilon = 1e-9 # Prevent log(0)
    q = clamp(q, epsilon, 1.0 - epsilon)
    p = clamp(p, epsilon, 1.0 - epsilon)
    return (q * (log(q) - log(p))) + ((1.0 - q) * (log(1.0 - q) - log(1.0 - p)))
end


##############################
# +++ dev area
##############################
predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())

match_id = rand(collect(keys(all_oos_results)))

open_odds, close_odds, result_odds = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)


r1 = all_oos_results[match_id];
match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);

model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict))
model_probs = Dict(key => median( value) for (key, value) in pairs(match_predict))

open_odds
close_odds

subset( ds.matches, :match_id => ByRow(isequal(match_id)))



open_odds
close_odds
fair_open_odds = remove_vig(open_odds)
fair_close_odds = remove_vig(close_odds)
model_odds
result_odds
get_margins(open_odds)
get_margins(close_odds)


# kl 
# Information Capture (IC)
# Returns 1.0 if perfect, < 0 if worse than opening line
#

market_error = kl_divergence(1/close_odds.over_25 , 1/open_odds.over_25)
model_error = kl_divergence.(1/close_odds.over_25,  match_predict.over_25)
IC =  1.0 .- (model_error ./ market_error)

using StatsPlots
density(IC)

mean(IC)
describe(IC)



function check_bet_stability(predictions, p_close)
    # Calculate KL for every sample
    errors = [kl_divergence(1/p_close, p) for p in predictions]
    
    # If the standard deviation of the error is high, the model is unstable
    stability = std(errors)
    
    return stability
end

# Usage
stab = check_bet_stability(match_predict.over_25, close_odds.over_25)
println("Model Instability: $stab")

model_odds[:over_25]
open_odds[:over_25]
fair_open_odds[:over_25]
fair_close_odds[:over_25]
close_odds[:over_25]


######
# tear sheet 
######

using ProgressMeter
using DataFrames
using Statistics


function generate_performance_dataframe(
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

    # Helper for Binary Log Loss
    function calc_log_loss(prob, outcome)
        p = clamp(prob, 1e-9, 1.0-1e-9) # Avoid log(0)
        y = outcome ? 1.0 : 0.0
        return - (y * log(p) + (1 - y) * log(1 - p))
    end

    # Helper for Brier Score
    calc_brier(prob, outcome) = (prob - (outcome ? 1.0 : 0.0))^2

    @showprogress desc="Evaluating Matches" for (match_id, params) in oos_results
        
        # --- ERROR HANDLING BLOCK ---
        # We wrap the data fetching in a try-catch to skip corrupt rows
        local open_odds, close_odds, results
        try
            open_odds, close_odds, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, data_store.odds)
        catch e
            # If data is missing (KeyError) or corrupt, skip this match
            skipped_matches += 1
            continue 
        end
        # ----------------------------

        # 1. Get Prediction Chains
        preds = BayesianFootball.Predictions.predict_market(model, predict_config, params...)
        
        # 2. Metadata lookup
        meta = get(meta_lookup, match_id, (league_id="Unknown", match_week=0))

        # 3. Iterate through every market
        for market in keys(preds)
            # Check if this specific market exists in the odds we fetched
            if haskey(open_odds, market) && haskey(close_odds, market) && haskey(results, market)
                
                # --- Inputs ---
                chain = getproperty(preds, market)
                prob_model = mean(chain)          # Point Estimate
                prob_median = median(chain)       # Robust Estimate
                
                o_open = getproperty(open_odds, market)
                o_close = getproperty(close_odds, market)
                outcome = getproperty(results, market) 
                
                # --- Metric 1: CLV ---
                clv = log(o_open / o_close)

                # --- Metric 2: IC ---
                p_market_open = 1.0 / o_open
                p_market_close = 1.0 / o_close
                
                kld_market = kl_divergence(p_market_close, p_market_open)
                kld_model  = kl_divergence(p_market_close, prob_model)
                
                ic = (kld_market < 1e-7) ? 0.0 : (1.0 - (kld_model / kld_market))

                # --- Metric 3: Accuracy ---
                ll = calc_log_loss(prob_model, outcome)
                brier = calc_brier(prob_model, outcome)

                # --- Metric 4: PnL ---
                ev = (prob_model * o_open) - 1.0
                bet_placed = ev > 0 # Simple strategy: Bet if +EV
                profit = 0.0
                if bet_placed
                    profit = outcome ? (o_open - 1.0) : -1.0
                end

                push!(rows, (;
                    match_id = match_id,
                    league_id = meta.league_id,
                    match_week = meta.match_week,
                    market = market,
                    prob_model = prob_model,
                    prob_median = prob_median,
                    odds_open = o_open,
                    odds_close = o_close,
                    outcome = outcome,
                    log_loss = ll,
                    brier_score = brier,
                    clv_pct = clv,
                    ic_score = ic,
                    ev = ev,
                    bet_placed = bet_placed,
                    profit = profit
                ))
            end
        end
    end

    if skipped_matches > 0
        println("⚠️ Warning: Skipped $skipped_matches matches due to missing/corrupt odds data.")
    end

    return DataFrame(rows)
end

# --- RUN IT ---
df_results = generate_performance_dataframe(model, predict_config, all_oos_results, ds)


function summarize_performance(df::DataFrame)
    # Filter for only bets we actually placed
    bets = filter(row -> row.bet_placed, df)
    
    n_bets = nrow(bets)
    if n_bets == 0
        return "No bets placed."
    end

    total_profit = sum(bets.profit)
    turnover = n_bets # Assuming 1 unit per bet
    roi = total_profit / turnover
    
    # Variance of returns
    # (Win = Odds-1, Loss = -1)
    returns_std = std(bets.profit)
    
    # Archie Score
    # A score > 2.0 implies <5% chance this result is random luck
    archie = (n_bets * (roi^2)) / (returns_std^2)

    println("--- Performance Summary ---")
    println("Total Bets:    $n_bets")
    println("Total Profit:  $(round(total_profit, digits=2)) units")
    println("ROI:           $(round(roi*100, digits=2))%")
    println("CLV (Avg):     $(round(mean(bets.clv_pct)*100, digits=2))%")
    println("Archie Score:  $(round(archie, digits=2))")
    
    if archie > 2.0
        println("✅ Result is Statistically Significant")
    else
        println("❌ Result could be variance/luck")
    end
    
    return (roi=roi, archie=archie, count=n_bets)
end


function calc_ece(df::DataFrame; bins=10)
    # We only care about Probability vs Outcome
    # Sort by probability
    df_sorted = sort(df, :prob_model)
    
    # Create bins
    df_sorted.bin = cut(df_sorted.prob_model, range(0, 1, length=bins+1))
    
    ece = 0.0
    N = nrow(df_sorted)
    
    # Group by bin to calculate stats
    gdf = groupby(df_sorted, :bin)
    
    println("--- Calibration Table ---")
    
    for key in keys(gdf)
        sub = gdf[key]
        n_sub = nrow(sub)
        
        if n_sub > 0
            avg_pred = mean(sub.prob_model)
            avg_actual = mean(sub.outcome) # True=1.0, False=0.0
            
            # Weighted difference
            diff = abs(avg_pred - avg_actual)
            ece += (n_sub / N) * diff
            
            # Optional: Print bins to spot bias
            # @printf("Bin %.2f: Pred %.2f vs Actual %.2f (N=%d)\n", 
            #         avg_pred, avg_pred, avg_actual, n_sub)
        end
    end
    
    println("ECE Score: $(round(ece, digits=4))")
    if ece < 0.05
        println("✅ Model is Well Calibrated")
    else
        println("⚠️ Model is Miscalibrated (Target < 0.05)")
    end
    
    return ece
end

using Printf

function calc_ece(df::DataFrame; bins=10)
    # 1. Create explicit copy to avoid modifying original
    local_df = copy(df)
    
    # 2. Manual Binning (0.0-0.1, 0.1-0.2, etc.)
    # We map 0.05 -> Bin 1, 0.95 -> Bin 10
    # floor(0.05 * 10) + 1 = 1
    local_df.bin_idx = floor.(Int, local_df.prob_model .* bins) .+ 1
    
    # Handle the edge case where prob == 1.0 (it belongs to the last bin, not bin+1)
    local_df.bin_idx = clamp.(local_df.bin_idx, 1, bins)

    ece = 0.0
    N = nrow(local_df)
    
    # 3. Group by the integer bin index
    gdf = groupby(local_df, :bin_idx)
    
    println("\n--- Calibration Table (ECE) ---")
    @printf("%-5s | %-8s | %-8s | %-5s\n", "Bin", "Avg Pred", "Actual", "Count")
    println("-"^36)

    for key in keys(gdf)
        sub = gdf[key]
        n_sub = nrow(sub)
        
        if n_sub > 0
            avg_pred = mean(sub.prob_model)
            avg_actual = mean(sub.outcome) # True=1.0, False=0.0
            
            # Weighted absolute difference
            diff = abs(avg_pred - avg_actual)
            ece += (n_sub / N) * diff
            
            # Print row for visual check
            @printf("%-5d | %.3f    | %.3f    | %d\n", 
                    key.bin_idx, avg_pred, avg_actual, n_sub)
        end
    end
    
    println("-"^36)
    println("ECE Score: $(round(ece, digits=4))")
    
    if ece < 0.05
        println("✅ Model is Well Calibrated (Accurate Confidence)")
    else
        println("⚠️ Model is Miscalibrated (Over/Under Confident)")
    end
    
    return ece
end


# 1. Create the Master DataFrame
df_perf = generate_performance_dataframe(model, predict_config, all_oos_results, ds)
summarize_performance(df_perf)
calc_ece(df_perf)
describe(df_perf.log_loss)
describe(df_perf.brier_score)


# 2. Filter for Premier League (League ID for EPL is usually "E0" in football-data.co.uk or similar)
# Check your ds.matches to confirm the league_id string
pl_data = filter(row -> row.league_id == "E0", df_perf)
other_data = filter(row -> row.league_id != "E0", df_perf)

# 3. Compare Over 2.5 Market specifically
println("\n=== PREMIER LEAGUE (Over 2.5) ===")
pl_over = filter(row -> row.market == :over_25, pl_data)

println("\n=== OTHER LEAGUES (Over 2.5) ===")
other_over = filter(row -> row.market == :over_25, other_data)
summarize_performance(other_over)
calc_ece(other_over)

### ece 

function generate_fair_performance_dataframe(
    model, 
    predict_config, 
    oos_results::Dict, 
    data_store::BayesianFootball.Data.DataStore
)
    rows = []
    
    # Helper: Standardize odds to fair probabilities
    function get_fair_prob(odds_val, market_name, full_market_odds)
        # This is a simplified generic vig remover
        # It tries to find the companion odd in the named tuple
        
        # 1. Calculate the Overround for this specific market
        # We grab all values from the NamedTuple that belong to this market group
        # (This is a heuristic, assuming the named tuple is grouped correctly)
        implied_sum = 0.0
        
        # Simple detection for 1x2 vs Over/Under
        if market_name in (:home, :draw, :away)
            implied_sum = (1/full_market_odds.home) + (1/full_market_odds.draw) + (1/full_market_odds.away)
        elseif occursin("over_", string(market_name)) || occursin("under_", string(market_name))
            # Extract suffix (e.g. "25")
            s = string(market_name)
            suffix = s[findlast('_', s)+1:end]
            
            o_sym = Symbol("over_" * suffix)
            u_sym = Symbol("under_" * suffix)
            
            if haskey(full_market_odds, o_sym) && haskey(full_market_odds, u_sym)
                 implied_sum = (1/getproperty(full_market_odds, o_sym)) + (1/getproperty(full_market_odds, u_sym))
            else
                return 1.0 / odds_val # Fallback to raw
            end
        else
            return 1.0 / odds_val # Fallback
        end

        # 2. Normalize
        raw_prob = 1.0 / odds_val
        return raw_prob / implied_sum
    end

    @showprogress desc="Grading Fairly" for (match_id, params) in oos_results
        # Wrap in try-catch for data safety
        try 
            open_odds, close_odds, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, data_store.odds)
            preds = BayesianFootball.Predictions.predict_market(model, predict_config, params...)
            
            for market in keys(preds)
                if haskey(open_odds, market) && haskey(close_odds, market)
                    
                    # --- The New Logic ---
                    # We calculate FAIR probability for the market
                    prob_fair_open  = get_fair_prob(getproperty(open_odds, market), market, open_odds)
                    prob_fair_close = get_fair_prob(getproperty(close_odds, market), market, close_odds)
                    
                    prob_model = mean(getproperty(preds, market))
                    outcome = getproperty(results, market)

                    push!(rows, (;
                        match_id = match_id,
                        market = market,
                        prob_model = prob_model,
                        
                        # Store BOTH for comparison
                        prob_market_raw_close  = 1.0 / getproperty(close_odds, market),
                        prob_market_fair_close = prob_fair_close,
                        
                        outcome = outcome
                    ))
                end
            end
        catch e
            continue
        end
    end
    return DataFrame(rows)
end

fair_df = generate_fair_performance_dataframe(model, predict_config, all_oos_results, ds)
"""
Calculates ECE for a specific probability column.
"""
function calc_ece_generic(df::DataFrame, prob_col::Symbol; bins=10)
    local_df = copy(df)
    
    # Handle Missing Data
    dropmissing!(local_df, [prob_col, :outcome])

    # Create Bins
    local_df.bin_idx = floor.(Int, local_df[!, prob_col] .* bins) .+ 1
    local_df.bin_idx = clamp.(local_df.bin_idx, 1, bins)

    ece = 0.0
    N = nrow(local_df)
    gdf = groupby(local_df, :bin_idx)

    for key in keys(gdf)
        sub = gdf[key]
        n_sub = nrow(sub)
        avg_pred = mean(sub[!, prob_col])
        avg_actual = mean(sub.outcome)
        
        diff = abs(avg_pred - avg_actual)
        ece += (n_sub / N) * diff
    end
    
    return ece
end

"""
Runs ECE comparison for Model vs Open vs Close.
"""
function compare_calibration(df::DataFrame)
    println("--- Calibration Championship (Lower ECE is Better) ---")
    
    # 1. Prepare Probabilities from Odds (removing vig for fairness is complex, 
    #    so we typically use raw implied probs for a rough check, 
    #    or normalized if you have the 'fair' odds data available).
    
    # Here we use raw implied probs (1/Odds). 
    # Note: Because of Vig, Market ECE is naturally worse (probs sum > 1).
    # A fair comparison requires removing vig first.
    
    df.prob_open  = 1.0 ./ df.odds_open
    df.prob_close = 1.0 ./ df.odds_close
    
    # 2. Calculate Scores
    ece_model = calc_ece_generic(df, :prob_model)
    ece_open  = calc_ece_generic(df, :prob_open)
    ece_close = calc_ece_generic(df, :prob_close)
    
    @printf("1. Your Model:   %.4f  %s\n", ece_model, (ece_model < ece_close ? "🏆" : ""))
    @printf("2. Closing Line: %.4f\n", ece_close)
    @printf("3. Opening Line: %.4f\n", ece_open)
    
    println("----------------------------------------------------")
    if ece_model < ece_close
        println("✅ Your model is BETTER calibrated than the Closing Line.")
    else
        println("❌ The Closing Line is smarter/better calibrated.")
    end
end


fair_df = generate_fair_performance_dataframe(model, predict_config, all_oos_results, ds)
compare_calibration(df_perf)

function compare_fair_calibration(df::DataFrame)
    println("--- ⚖️ Fair Calibration Championship (Vig Removed) ---")
    println("Lower ECE = More Accurate Probability")
    println("-"^50)
    
    # 1. Calculate Scores using the generic helper we wrote earlier
    # (Ensure calc_ece_generic is defined in your session)
    ece_model = calc_ece_generic(df, :prob_model)
    ece_fair_close = calc_ece_generic(df, :prob_market_fair_close)
    
    # We can also check the 'Raw' close if you kept that column
    has_raw = "prob_market_raw_close" in names(df)
    ece_raw_close = has_raw ? calc_ece_generic(df, :prob_market_raw_close) : NaN

    # 2. Print Results
    @printf("1. Your Model:        %.4f\n", ece_model)
    @printf("2. Fair Closing Line: %.4f\n", ece_fair_close)
    
    if has_raw
        @printf("3. Raw Closing Line:  %.4f (Biased by Vig)\n", ece_raw_close)
    end
    
    println("-"^50)
    
    # 3. The Verdict
    if ece_model < ece_fair_close
        println("🏆 UNICORN ALERT: Your model is beating the FAIR closing line.")
        println("   This implies a genuine mathematical edge on probability.")
    elseif ece_model < ece_raw_close
        println("✅ SOLID MODEL: You beat the Raw line (Vig), but not the Fair line.")
        println("   You are profitable because you don't pay the vig on predictions,")
        println("   but the market is still 'smarter' at finding the true center.")
    else
        println("⚠️ CALIBRATION LEAK: The market is much better calibrated.")
    end
end


compare_fair_calibration(fair_df)

###

function full_tear_sheet(df::DataFrame)
    # Filter for active bets for financial metrics
    bets = filter(row -> row.bet_placed, df)
    
    # 1. Financial Metrics (On Bets Placed)
    roi = 0.0
    profit = 0.0
    archie = 0.0
    count = 0
    
    if nrow(bets) > 0
        count = nrow(bets)
        profit = sum(bets.profit)
        roi = profit / count
        std_ret = std(bets.profit)
        archie = (count * (roi^2)) / (std_ret^2)
    end

    # 2. Skill Metrics (On ALL Predictions)
    # We evaluate skill on everything, even bets we didn't take.
    avg_log_loss = mean(df.log_loss)
    avg_brier = mean(df.brier_score)
    avg_ic = mean(df.ic_score)
    avg_clv = mean(df.clv_pct) # Note: calculating CLV on all matches shows market tracking ability

    println("=========================================")
    println("         MODEL PERFORMANCE REPORT        ")
    println("=========================================")
    
    println("\n--- 💰 FINANCIALS (Bets Placed) ---")
    @printf("Total Bets:      %d\n", count)
    @printf("Total Profit:    %.2f units\n", profit)
    @printf("ROI:             %.2f%%\n", roi * 100)
    @printf("Archie Score:    %.2f %s\n", archie, (archie > 2.0 ? "✅" : "⚠️"))
    
    println("\n--- 🧠 SKILL (All Predictions) ---")
    println("[Lower is Better]")
    @printf("Log Loss:        %.4f  (Ref: <0.60 is good)\n", avg_log_loss)
    @printf("Brier Score:     %.4f\n", avg_brier)
    
    println("\n[Higher is Better]")
    @printf("Info Capture:    %.3f\n", avg_ic)
    @printf("Avg CLV:         %.2f%%\n", avg_clv * 100)
    
    println("=========================================")
end


full_tear_sheet(df_perf)
over = 
over = filter(row -> row.market == :over_25, df_perf)
full_tear_sheet(over)

draw = filter(row -> row.market == :draw, df_perf)
full_tear_sheet(draw)

home = filter(row -> row.market == :home, other_data);
full_tear_sheet(home)



pl_data = filter(row -> row.league_id == 54, df_perf)
other_data = filter(row -> row.league_id != 54, df_perf)

# 3. Compare Over 2.5 Market specifically
println("\n=== PREMIER LEAGUE (Over 2.5) ===")
pl_over = filter(row -> row.market == :over_25, pl_data)
other_over = filter(row -> row.market == :over_25, other_data)


pl_under = filter(row -> row.market == :under_25, pl_data)
other_under = filter(row -> row.market == :under_25, other_data)


println("\n=== OTHER LEAGUES (Over 2.5) ===")
other_over = filter(row -> row.market == :over_25, other_data)
# summarize_performance(other_over)
full_tear_sheet(other_over)
calc_ece(other_over)

full_tear_sheet(other_under)
calc_ece(other_under)

full_tear_sheet(pl_over)
calc_ece(pl_over)






"""
14 │  9398869         57          31  over_25    0.441785     0.430804      12.0         1.8      true  0.816933    0.311604    1.
2 │  8824194         56          19  over_25    0.256019     0.230891      15.0         2.15    false  0.295739    0.0655455   1.


"""

subset(ds.odds, :match_id => ByRow(isequal(9398869)))

# Check for odds that are clearly wrong for an Over 2.5 market
bad_data = filter(row -> row.odds_open > 5.0, df_perf)
println("Found $(nrow(bad_data)) suspicious matches.")
println(bad_data[:, [:match_id, :league_id, :odds_open, :odds_close, :outcome]])


names(df_perf)
a = df_perf.odds_close .- df_perf.odds_open
describe(a)
histogram(a)

function is_sane_market(row; max_prob_shift=0.20)
    # 1. Hard Cap (Sanity)
    # No legitimate pre-game football market should have odds > 500 or < 1.01
    if row.odds_open > 100.0 || row.odds_close > 100.0 || row.odds_open < 1.01
        return false
    end

    # 2. Calculate Implied Probabilities
    p_open = 1.0 / row.odds_open
    p_close = 1.0 / row.odds_close

    # 3. Calculate the "Real" Shift
    prob_diff = abs(p_open - p_close)

    # 4. The Filter
    # If the market implied probability changed by more than 20%, 
    # it is likely a data error (e.g., 1.50 -> 3.00 is a 17% shift).
    # A shift of 20% (0.20) covers almost all legitimate team news drifts.
    if prob_diff > max_prob_shift
        return false
    end

    return true
end

# --- Apply and Check ---

df_clean = filter(is_sane_market, df_perf)

println("Original Rows: $(nrow(df_perf))")
println("Cleaned Rows:  $(nrow(df_clean))")

# Visualizing the cleanup
# We expect the massive outliers (-24, +30) to be gone
drift_clean = df_clean.odds_close .- df_clean.odds_open
describe(drift_clean)

full_tear_sheet(df_perf)
full_tear_sheet(df_clean)

compare_calibration(df_clean)

over = filter(row -> row.market == :over_25, df_perf);
full_tear_sheet(over)

over_clean = filter(row -> row.market == :over_25, df_clean);
full_tear_sheet(over_clean)



