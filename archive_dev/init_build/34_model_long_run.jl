using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)

using Distributions


data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


model_1 = Models.PreGame.StaticPoisson()
name_1 = "normal"

model_2 = Models.PreGame.GRWPoisson()
name_2 = "grw"

model_3 = Models.PreGame.AR1Poisson()
name_3 = "ar1"


cfg_1 = Experiments.experiment_config_models(model_1, name_1)
cfg_2 = Experiments.experiment_config_models(model_2, name_2)
cfg_3 = Experiments.experiment_config_models(model_3, name_3)


results1 = Experiments.run_experiment(ds, cfg_1)
Experiments.save_experiment(results1)

results2 = Experiments.run_experiment(ds, cfg_2)
Experiments.save_experiment(results2)


results3 = Experiments.run_experiment(ds, cfg_3)
Experiments.save_experiment(results3)

"""

"""


# 1. Grab the first stored result
#    results2.training_results is Vector{Tuple{Chains, SplitMetaData}}
(chain_1, meta_1) = results2.training_results[1];

println("Type of Metadata: ", typeof(meta_1))
println("Metadata Content: ", meta_1)

# Expected Output: SplitMetaData(tournament_id=..., time_step=..., etc.)


# 2. Regenerate all splits from the DataStore
#    (This is fast because it creates Views, not Copies)
all_splits = BayesianFootball.Data.create_data_splits(ds, results2.config.splitter);

# 3. Match the correct split
#    We assume 1-to-1 correspondence. Let's verify.
(df_train_1, meta_recalc_1) = all_splits[1];

if meta_1 == meta_recalc_1
    println("✅ Metadata matches perfectly.")
else
    println("⚠️ Metadata mismatch! (Check index alignment)")
end

println("Training Rows: ", nrow(df_train_1))



# 4. Create the Local Vocabulary for THIS specific split
local_vocab_1 = BayesianFootball.Features.adapt_vocabulary(results2.vocabulary, df_train_1);

println("Global Vocab Size: ", results2.vocabulary.mappings[:n_teams])
println("Local Vocab Size:  ", local_vocab_1.mappings[:n_teams])

# Quick sanity check: Does this match the chain?

chain_names = names(chain_1, :z_att_steps)
println("Chain Parameters:  ", length(chain_names))
# (Should be Local_Vocab_Size * (Steps) roughly)


df_test_1 = BayesianFootball.Data.get_next_matches(ds, meta_1, results2.config.splitter);

println("Test Matches Found: ", nrow(df_test_1))
show(df_test_1[:, [:match_date, :home_team, :away_team]], allcols=false)


# 6. Extract Parameters
#    Note: We pass 'local_vocab_1' here!
model_preds_1 = BayesianFootball.Models.PreGame.extract_parameters(
    results2.config.model,
    df_test_1,
    local_vocab_1, 
    chain_1
);

println("Predictions generated for $(length(model_preds_1)) matches.")


model_preds_1



using ProgressMeter

function build_robust_analysis(results, data_store)
    println("🚀 Starting Robust Analysis for: $(results.config.name)")
    
    # 1. Regenerate Training Splits (to get Local Vocabs)
    println("   ↳ Regenerating training views...")
    training_splits = BayesianFootball.Data.create_data_splits(data_store, results.config.splitter)
    
    # 2. Configs
    predict_config = BayesianFootball.Predictions.PredictionConfig(BayesianFootball.Markets.get_standard_markets())
    master_rows = []
    
    # 3. Threaded Loop (or Sequential with Progress)
    #    Iterate over the stored results
    @showprogress for (i, (chain, meta)) in enumerate(results.training_results)
        
        # A. Get the matching training data (for Vocab)
        #    We assume the order is deterministic and identical
        (df_train, meta_check) = training_splits[i]
        
        # Sanity Check
        if meta.time_step != meta_check.time_step
            @warn "Index mismatch at $i: Result has $(meta.time_step), Splitter has $(meta_check.time_step)"
            continue
        end

        # B. Reconstruct Local Vocabulary
        local_vocab = BayesianFootball.Features.adapt_vocabulary(results.vocabulary, df_train)

        # C. Get Test Data (Next Matches)
        df_test = BayesianFootball.Data.get_next_matches(data_store, meta, results.config.splitter)
        
        if nrow(df_test) == 0; continue; end

        # D. Predict
        try
            # Extraction
            model_preds = BayesianFootball.Models.PreGame.extract_parameters(
                results.config.model,
                df_test,
                local_vocab, 
                chain
            )
            
            # Market Resolution
            for (match_id, latents) in model_preds
                # ... (Standard Market Matching Logic) ...
                # Fetch row for metadata
                row_data = df_test[df_test.match_id .== match_id, :][1, :]
                
                # Fetch Odds
                open_odds, close_odds, outcomes = BayesianFootball.Predictions.get_market_data(
                     match_id, predict_config, data_store.odds
                )
                
                # Convert Latents to Probs
                probs = BayesianFootball.Predictions.predict_market(
                    results.config.model, predict_config, latents...
                )
                
                # Flatten
                for market in keys(probs)
                    if haskey(open_odds, market)
                        push!(master_rows, (;
                            split_i = i,
                            match_id = match_id,
                            season = row_data.season,
                            date = row_data.match_date,
                            league = row_data.tournament_id,
                            home = row_data.home_team,
                            away = row_data.away_team,
                            market = market,
                            prob_model = mean(getproperty(probs, market)),
                            odds_open = getproperty(open_odds, market),
                            odds_close = getproperty(close_odds, market),
                            outcome = getproperty(outcomes, market)
                        ))
                    end
                end
            end
            
        catch e
            @warn "Failed at Split $i" exception=e
        end
    end
    
    return DataFrame(master_rows)
end


df_model_2 = build_robust_analysis(results2, ds)

df_model_3 = build_robust_analysis(results3, ds)

# 1. Apply the metrics logic we defined earlier
#    (If you haven't defined this function in your REPL yet, paste it from dev/36 first)
add_performance_metrics!(df_model_2)
add_performance_metrics!(df_model_3)

# 2. Quick Sanity Check: Did we find value?
println("Mean EV: ", round(mean(df_model_2.ev) * 100, digits=2), "%")
println("Mean CLV: ", round(mean(df_model_2.clv) * 100, digits=2), "%")


println("Mean EV: ", round(mean(df_model_3.ev) * 100, digits=2), "%")
println("Mean CLV: ", round(mean(df_model_3.clv) * 100, digits=2), "%")


using Printf

function quick_calibration_check(df)
    # Bin probabilities into 10% buckets
    df.bin = floor.(Int, df.prob_model .* 10) ./ 10
    
    calib = combine(groupby(df, :bin), 
        :outcome => mean => :actual_prob,
        :prob_model => mean => :predicted_prob,
    )
    
    # Simple ASCII Plot
    println("\n--- Calibration Check (Pred vs Actual) ---")
    println("Bin   | Pred  | Actual | Diff")
    println("--------------------------------")
    for r in eachrow(sort(calib, :bin))
        diff = r.predicted_prob - r.actual_prob
        flag = abs(diff) > 0.05 ? "⚠️" : "✅"
        @printf("%.1f   | %.3f | %.3f  | %s %.3f\n", r.bin, r.predicted_prob, r.actual_prob, flag, diff)
    end
end

quick_calibration_check(df_model_2)
quick_calibration_check(df_model_3)




using DataFrames, Statistics

# 1. Aggregate errors by Home Team
team_audit = combine(groupby(df_model_2, :home), 
    :log_loss => mean => :avg_log_loss,
    :prob_model => mean => :avg_confidence,
    :ev => mean => :avg_ev,
)

# 2. Filter for teams with enough data (e.g., > 15 games)
#    (Adjust '15' depending on your dataset size)
filter!(r -> r.count >= 15, team_audit)

# 3. Sort by "Hardest to Predict" (Highest Log Loss)
sort!(team_audit, :avg_log_loss, rev=true)

println("\n--- 🛑 The 'Problem' Teams (Highest Error) ---")
# These are the teams your model "doesn't get". 
# If they are all chaotic teams (e.g., Leeds under Bielsa), that's expected.
# If they are defensive boring teams, you are missing a 'volatility' parameter.
display(first(team_audit, 5))

println("\n--- ✅ The 'Solved' Teams (Lowest Error) ---")
display(last(team_audit, 5))


using DataFrames, Statistics, Printf

function run_championship(df_grw, df_ar1)
    println("🥊 MODEL CHAMPIONSHIP: GRW vs AR1 🥊")
    println("=====================================")
    
    # 1. Skill Metrics (Lower is Better)
    ll_grw = mean(df_grw.log_loss)
    ll_ar1 = mean(df_ar1.log_loss)
    
    println("\n--- 🧠 SKILL (Log Loss) ---")
    println("GRW:  ", round(ll_grw, digits=5))
    println("AR1:  ", round(ll_ar1, digits=5))
    diff = (ll_grw - ll_ar1) / ll_grw
    println("Improvement: $(round(diff * 100, digits=2))% " * (ll_ar1 < ll_grw ? "🏆 (AR1 Wins)" : "❌ (GRW Wins)"))

    # 2. The "Problem Team" Test
    # Did AR1 fix the teams that GRW struggled with?
    
    # Get GRW's worst teams
    grw_teams = combine(groupby(df_grw, :home), :log_loss => mean => :err)
    sort!(grw_teams, :err, rev=true)
    worst_teams = first(grw_teams, 5).home
    
    println("\n--- 🚑 RECOVERY WARD (Fixing GRW's Worst Teams) ---")
    println("Team             | GRW Err | AR1 Err | Improved?")
    println("------------------------------------------------")
    
    for team in worst_teams
        err_g = filter(r -> r.home == team, df_grw).log_loss |> mean
        err_a = filter(r -> r.home == team, df_ar1).log_loss |> mean
        
        better = err_a < err_g
        flag = better ? "✅" : "❌"
        @printf("%-16s | %.4f  | %.4f  | %s\n", team, err_g, err_a, flag)
    end

    # 3. Financials (Flat Betting EV > 5%)
    println("\n--- 💰 FINANCIALS (Blind EV Betting) ---")
    
    function calc_pnl(df, name)
        # Filter for the same "confident" bets that broke GRW
        bets = filter(r -> r.ev > 0.05 && r.odds_open < 10.0, df)
        if nrow(bets) == 0 return end
        
        profit = sum((bets.outcome .* (bets.odds_open .- 1.0)) .- (.~bets.outcome))
        roi = profit / nrow(bets)
        @printf("%s:  %4d bets | Profit: %6.2f | ROI: %5.2f%%\n", name, nrow(bets), profit, roi * 100)
    end
    
    calc_pnl(df_grw, "GRW")
    calc_pnl(df_ar1, "AR1")
end

run_championship(df_model_2, df_model_3)


using DataFrames, Statistics, Printf

function financial_autopsy_fixed(df, name)
    println("\n🔎 FINANCIAL AUTOPSY: $name 🔍")
    println("==============================")
    
    # Filter for active bets
    bets = filter(r -> r.ev > 0.05 && r.odds_open < 10.0, df)
    
    # 1. Odds Bucket Logic (Manual)
    function get_bucket(o)
        if o <= 1.5 return "1.0 - 1.5 (Fav)"
        elseif o <= 2.0 return "1.5 - 2.0 (Solid)"
        elseif o <= 3.0 return "2.0 - 3.0 (Value)"
        elseif o <= 5.0 return "3.0 - 5.0 (Dog)"
        else return "5.0+ (Longshot)"
        end
    end
    
    # Create the bucket column safely
    bets.odds_bucket = map(get_bucket, bets.odds_open)

    # 2. Breakdown by Odds Range
    println("\n--- 📊 By Odds Range ---")
    by_odds = combine(groupby(bets, :odds_bucket), 
        :outcome => length => :count,
        :clv => (x -> mean(x) * 100) => :avg_clv,
        [:outcome, :odds_open] => ((o, odds) -> sum((o .* (odds .- 1.0)) .- (.~o))) => :profit
    )
    by_odds.roi = by_odds.profit ./ by_odds.count .* 100
    
    # Sort for readability
    sort!(by_odds, :odds_bucket)
    
    for r in eachrow(by_odds)
        @printf("%-18s | Bets: %4d | CLV: %5.2f%% | ROI: %6.2f%%\n", r.odds_bucket, r.count, r.avg_clv, r.roi)
    end
end

financial_autopsy_fixed(df_model_3, "AR1 Model")


function run_surgical_strategy(df)
    println("\n--- 🏥 SURGICAL STRATEGY SIMULATION ---")
    
    # The Rules derived from your Autopsy:
    # 1. EV > 5% (Standard)
    # 2. NO Draws (Poisson weakness)
    # 3. NO Home Wins (Negative CLV leak)
    # 4. Odds < 4.0 (Safety against variance)
    
    strategy_bets = filter(r -> 
        r.ev > 0.05 && 
        r.market != :draw && 
        r.market != :home &&
        r.odds_open < 4.0, 
        df
    )
    
    if nrow(strategy_bets) == 0
        println("No bets found matching criteria.")
        return
    end

    profit = sum((strategy_bets.outcome .* (strategy_bets.odds_open .- 1.0)) .- (.~strategy_bets.outcome))
    roi = profit / nrow(strategy_bets)
    clv = mean(strategy_bets.clv) * 100

    println("Criteria: No Draws, No Home Wins, Odds < 4.0")
    println("Bets Placed: $(nrow(strategy_bets))")
    println("Total Profit: $(round(profit, digits=2)) units")
    println("ROI:          $(round(roi * 100, digits=2))%")
    println("Avg CLV:      $(round(clv, digits=2))%")
end

    #    Models often fail on high odds (> 4.0) or massive favorites (< 1.3)
run_surgical_strategy(df_model_3)


using DataFrames, Statistics, Printf

function financial_autopsy(df, name)
    println("\n🔎 FINANCIAL AUTOPSY: $name 🔍")
    println("==============================")
    
    # Filter for the bets we actually placed
    bets = filter(r -> r.ev > 0.05 && r.odds_open < 10.0, df)
    
    # 1. Closing Line Value (The Truth Teller)
    #    If CLV > 0, you are beating the market, and losses are just bad luck.
    #    If CLV < 0, your model is the "sucker" at the table.
    mean_clv = mean(bets.clv) * 100
    println("Overall CLV: $(round(mean_clv, digits=2))% " * (mean_clv > 0 ? "✅ (Beating Market)" : "❌ (Losing to Market)"))

    # 2. Breakdown by Market
    #    Are we losing on 1x2 or Totals?
    println("\n--- 🛒 By Market Type ---")
    by_market = combine(groupby(bets, :market), 
        :outcome => length => :count,
        :clv => (x -> mean(x) * 100) => :avg_clv,
        [:outcome, :odds_open] => ((o, odds) -> sum((o .* (odds .- 1.0)) .- (.~o))) => :profit
    )
    by_market.roi = by_market.profit ./ by_market.count .* 100
    sort!(by_market, :roi)
    
    for r in eachrow(by_market)
        @printf("%-10s | Bets: %4d | CLV: %5.2f%% | ROI: %6.2f%%\n", r.market, r.count, r.avg_clv, r.roi)
    end

end

financial_autopsy(df_model_3, "AR1 Model")
financial_autopsy(df_model_2, "grw")


using DataFrames, Statistics, Printf

function league_autopsy(df)
    println("\n🌍 LEAGUE PERFORMANCE AUDIT 🌍")
    println("================================")
    
    # Apply the same "Surgical" filters first
    bets = filter(r -> 
        r.ev > 0.05 && 
        r.market != :draw && 
        r.market != :home &&
        r.odds_open < 4.0, 
        df
    )
    
    # Group by League (Tournament ID)
    league_stats = combine(groupby(bets, :league), 
        :outcome => length => :count,
        :clv => (x -> mean(x) * 100) => :avg_clv,
        [:outcome, :odds_open] => ((o, odds) -> sum((o .* (odds .- 1.0)) .- (.~o))) => :profit
    )
    
    league_stats.roi = league_stats.profit ./ league_stats.count .* 100
    sort!(league_stats, :roi)
    
    println("Strategy: Surgical (No Draw/Home, < 4.0)")
    println("---------------------------------------------------------")
    println("League ID | Bets |   CLV   |   Profit   |   ROI   ")
    println("---------------------------------------------------------")
    
    for r in eachrow(league_stats)
        @printf("%-9s | %4d | %5.2f%% | %8.2f u | %6.2f%%\n", 
            r.league, r.count, r.avg_clv, r.profit, r.roi)
    end
end

league_autopsy(df_model_3)



using DataFrames, Statistics, Printf

function deep_dive_autopsy(df)
    println("\n🔬 DEEP DIVE: PER LEAGUE & MARKET 🔬")
    println("======================================")
    
    # 1. Apply Surgical Filters (consistent with your current view)
    #    (EV > 5%, No Draws, No Home, Odds < 4.0)
    bets = filter(r -> 
        r.ev > 0.05 && 
        r.market != :draw && 
        r.market != :home &&
        r.odds_open < 4.0, 
        df
    )
    
    # 2. Group by League AND Market
    breakdown = combine(groupby(bets, [:league, :market]), 
        :outcome => length => :count,
        :clv => (x -> mean(x) * 100) => :avg_clv,
        [:outcome, :odds_open] => ((o, odds) -> sum((o .* (odds .- 1.0)) .- (.~o))) => :profit
    )
    
    breakdown.roi = breakdown.profit ./ breakdown.count .* 100
    
    # 3. Sort logic: By League, then by Profit (worst to best)
    sort!(breakdown, [:league, :profit])
    
    # 4. Display
    current_league = -1
    
    for r in eachrow(breakdown)
        if r.league != current_league
            println("\n---------------------------------------------------------------")
            println("🌍 LEAGUE $(r.league) BREAKDOWN")
            println("---------------------------------------------------------------")
            println("Market      | Bets |   CLV   |    Profit   |    ROI   ")
            println("---------------------------------------------------------------")
            current_league = r.league
        end
        
        # Color coding for the eye
        flag = r.roi > 0 ? "✅" : (r.roi < -10 ? "🩸" : "❌")
        
        @printf("%-11s | %4d | %5.2f%% | %8.2f u | %6.2f%% %s\n", 
            r.market, r.count, r.avg_clv, r.profit, r.roi, flag)
    end
end

# Run it
deep_dive_autopsy(df_model_3)
deep_dive_autopsy(df_model_2)



"""

"""


using DataFrames, Statistics, Printf

"""
    detect_bias(df, market_sym; league=nothing)

Compares Avg Model Probability vs Actual Outcome Frequency.
Returns the 'adjustment' needed (Actual - Predicted).
"""
function detect_bias(df, market_sym; league=nothing)
    # Filter by market (and optionally league)
    subset = filter(r -> r.market == market_sym, df)
    if !isnothing(league)
        filter!(r -> r.league == league, subset)
    end
    
    if nrow(subset) < 10
        println("⚠️ Not enough data for $market_sym (N=$(nrow(subset)))")
        return 0.0
    end
    
    avg_pred = mean(subset.prob_model)
    # Convert outcome (Bool) to Float (Frequency)
    actual_freq = mean(subset.outcome) 
    
    diff = actual_freq - avg_pred
    
    label = isnothing(league) ? "Global" : "League $league"
    
    println("\n--- ⚖️ Bias Check: $label ($market_sym) ---")
    println("Matches:          $(nrow(subset))")
    println("Avg Predicted:    $(round(avg_pred, digits=3))")
    println("Actual Frequency: $(round(actual_freq, digits=3))")
    
    if diff > 0
        printstyled("Result:           UNDER-predicting by +$(round(diff, digits=3))\n", color=:green)
    else
        printstyled("Result:           OVER-predicting by $(round(diff, digits=3))\n", color=:red)
    end
    
    return diff
end


"""
    backtest_calibrated(df, adjustment, market_sym; league=nothing, threshold=0.06)

Simulates betting after adding 'adjustment' to the model's probabilities.
"""
function backtest_calibrated(df, adjustment, market_sym; league=nothing, threshold=0.06)
    # 1. Filter Target Data
    #    Make a copy so we don't mess up the original results
    test_df = filter(r -> r.market == market_sym, df)
    
    if !isnothing(league)
        filter!(r -> r.league == league, test_df)
    end
    
    # 2. Apply Calibration (Additive Shift)
    #    New Prob = Old Prob + Adjustment
    #    Clamp to keep it valid [1%, 99%]
    #    (Note: If adjustment is negative, it lowers the prob)
    pred_probs = [r.prob_model for r in eachrow(test_df)]
    new_probs  = clamp.(pred_probs .+ adjustment, 0.01, 0.99)
    
    # 3. Calculate New Edge
    #    Edge = (New_Prob * Odds) - 1
    odds = [r.odds_open for r in eachrow(test_df)]
    edges = (new_probs .* odds) .- 1.0
    
    # 4. Select Bets
    #    Indices where edge > threshold
    bet_indices = findall(edges .> threshold)
    
    if isempty(bet_indices)
        println("No bets found with calibrated edge > $(threshold*100)%")
        return
    end
    
    # 5. Calculate P&L
    outcomes = [test_df.outcome[i] for i in bet_indices]
    bet_odds = odds[bet_indices]
    
    # Profit = (Outcome * (Odds-1)) - (Not Outcome)
    pnl_vector = (outcomes .* (bet_odds .- 1.0)) .- (.~outcomes)
    total_pnl = sum(pnl_vector)
    roi = (total_pnl / length(bet_indices)) * 100
    
    # Display
    lbl = isnothing(league) ? "Global" : "L$league"
    sign_str = adjustment > 0 ? "+" : ""
    
    println("\n--- 🛠️ Calibrated Strategy ($lbl $market_sym) ---")
    println("Adjustment:    $(sign_str)$(round(adjustment, digits=3))")
    println("Bets Placed:   $(length(bet_indices))")
    println("Total Profit:  $(round(total_pnl, digits=2)) units")
    printstyled("ROI:           $(round(roi, digits=2))%\n", color = roi>0 ? :green : :red)
end



# 1. Check & Fix the "League One Defense" Bias (League 56, Under 1.5)
#    (We expect a negative bias here, e.g., -0.05)
bias_l56_u15 = detect_bias(df_model_3, :under_15, league=56)
backtest_calibrated(df_model_3, bias_l56_u15, :under_15, league=56)

# 2. Check & Fix the "Premiership Overs" Bias (League 54, Over 2.5)
#    (We expect a positive bias here, e.g., +0.05)
bias_l54_o25 = detect_bias(df_model_3, :under_25, league=54)
backtest_calibrated(df_model_3, bias_l54_o25, :under_25, league=54)

# 3. Check Global Home Bias (The "Home Win" Trap)
bias_home = detect_bias(df_model_3, :home)
# Try calibrating it: If we fix the bias, do we make money?
backtest_calibrated(df_model_3, bias_home, :home)


using DataFrames, Statistics, Dates, Printf

"""
    validate_calibration_strategies(df; train_split=0.6)

Splits the data by time. Learns the bias on the 'Past' and bets on the 'Future'.
"""
function validate_calibration_strategies(df; train_split=0.5)
    # 1. Sort strictly by date to simulate reality
    df_sorted = sort(df, :date)
    
    n_total = nrow(df_sorted)
    cut_idx = floor(Int, n_total * train_split)
    
    # 2. Split into Past (Train) and Future (Test)
    df_train = df_sorted[1:cut_idx, :]
    df_test  = df_sorted[cut_idx+1:end, :]
    
    println("\n⏳ TIME TRAVELLING VALIDATION (Split: $(Int(train_split*100)) / $(Int((1-train_split)*100))) ⏳")
    println("==========================================================")
    println("Learning Phase:  $(min(df_train.date...)) to $(max(df_train.date...)) [$(nrow(df_train)) matches]")
    println("Betting Phase:   $(min(df_test.date...)) to $(max(df_test.date...)) [$(nrow(df_test)) matches]")
    
    # 3. Define the Hypotheses to Validate
    #    Format: (Name, Market, LeagueID (or nothing))
    hypotheses = [
        ("L56 (Under 1.5) - The 'Defense' Fix", :under_15, 56),
        ("L54 (Over 2.5)  - The 'Celtic' Fix",  :over_25,  54),
        ("Global Home     - The 'Favorite' Fix", :home,     nothing)
    ]
    
    for (name, target_market, target_league) in hypotheses
        println("\n---------------------------------------------------")
        println("🧪 Testing: $name")
        
        # --- STEP A: LEARN BIAS (PAST) ---
        train_subset = filter(r -> r.market == target_market, df_train)
        if !isnothing(target_league)
            filter!(r -> r.league == target_league, train_subset)
        end
        
        if nrow(train_subset) < 20
            println("   ⚠️ Not enough history to learn bias. Skipping.")
            continue
        end
        
        # Calculate Bias: Actual Outcome - Model Probability
        learned_bias = mean(train_subset.outcome) - mean(train_subset.prob_model)
        
        sign_str = learned_bias > 0 ? "+" : ""
        printstyled("   Phase 1 (Learn): Found bias of $(sign_str)$(round(learned_bias, digits=3))\n", color=:cyan)
        
        # --- STEP B: BET (FUTURE) ---
        test_subset = filter(r -> r.market == target_market, df_test)
        if !isnothing(target_league)
            filter!(r -> r.league == target_league, test_subset)
        end
        
        if nrow(test_subset) == 0
            println("   Phase 2 (Bet): No future matches found.")
            continue
        end
        
        # Apply the *LEARNED* bias to the *FUTURE* probabilities
        # Important: We clamp to keep probs valid
        future_probs = clamp.(test_subset.prob_model .+ learned_bias, 0.01, 0.99)
        
        # Calculate Edge using Real Odds
        future_edges = (future_probs .* test_subset.odds_open) .- 1.0
        
        # Bet if Edge > 5% (Standard threshold)
        bet_mask = future_edges .> 0.05
        n_bets = sum(bet_mask)
        
        if n_bets == 0
            println("   Phase 2 (Bet): No bets triggered with adjusted edge.")
        else
            outcomes = test_subset.outcome[bet_mask]
            odds     = test_subset.odds_open[bet_mask]
            
            profit = sum((outcomes .* (odds .- 1.0)) .- (.~outcomes))
            roi = profit / n_bets * 100
            
            # Formatting Output
            color = roi > 0 ? :green : :red
            flag = roi > 0 ? "✅" : "❌"
            
            println("   Phase 2 (Bet): Placed $n_bets bets on unseen data.")
            printstyled("   RESULT: Profit $(round(profit, digits=2)) units | ROI $(round(roi, digits=2))% $flag\n", color=color)
        end
    end
end

# Run the validation
validate_calibration_strategies(df_model_3)





using DataFrames, Statistics, Dates, Printf

function validate_all_strategies(df; train_split=0.5, min_bets=30)
    # 1. Sort & Split (Time Travel)
    df_sorted = sort(df, :date)
    cut_idx = floor(Int, nrow(df_sorted) * train_split)
    
    df_train = df_sorted[1:cut_idx, :]
    df_test  = df_sorted[cut_idx+1:end, :]
    
    println("\n⏳ GRID SEARCH VALIDATION (Split: $(Int(train_split*100))/$(Int((1-train_split)*100))) ⏳")
    println("==========================================================")
    println("Train: $(min(df_train.date...)) -> $(max(df_train.date...))")
    println("Test:  $(min(df_test.date...)) -> $(max(df_test.date...))")
    
    results = DataFrame()

    # 2. Define the Grid (Global Markets + League Specific)
    combos = unique(select(df_train, :league, :market))
    
    # Add a "Global" league placeholder
    global_markets = unique(df_train.market)
    for m in global_markets
        push!(combos, (league = -1, market = m)) 
    end
    
    # 3. Iterate and Test
    for row in eachrow(combos)
        target_league = row.league
        target_market = row.market
        
        # --- PHASE 1: LEARN (TRAIN) ---
        if target_league == -1
            train_sub = filter(r -> r.market == target_market, df_train)
            test_sub  = filter(r -> r.market == target_market, df_test)
            name = "Global - $target_market"
        else
            train_sub = filter(r -> r.market == target_market && r.league == target_league, df_train)
            test_sub  = filter(r -> r.market == target_market && r.league == target_league, df_test)
            name = "L$target_league - $target_market"
        end
        
        if nrow(train_sub) < min_bets
            continue 
        end
        
        # Calculate Bias (Actual - Model)
        bias = mean(train_sub.outcome) - mean(train_sub.prob_model)
        
        # --- PHASE 2: BET (TEST) ---
        if nrow(test_sub) == 0 continue end
        
        # Apply Calibration
        future_probs = clamp.(test_sub.prob_model .+ bias, 0.01, 0.99)
        future_edges = (future_probs .* test_sub.odds_open) .- 1.0
        
        # Bet Simulation (Edge > 5%)
        bet_mask = future_edges .> 0.05
        n_bets = sum(bet_mask)
        
        if n_bets > 0
            outcomes = test_sub.outcome[bet_mask]
            odds     = test_sub.odds_open[bet_mask]
            profit   = sum((outcomes .* (odds .- 1.0)) .- (.~outcomes))
            roi      = profit / n_bets * 100
            
            push!(results, (;
                Strategy = name,
                Bias_Learned = bias,
                Test_Bets = n_bets,
                Profit = profit,
                ROI = roi,
                Status = roi > 0 ? "✅" : "❌"
            ))
        end
    end
    
    # 4. Display Results (Sorted by Profit)
    sort!(results, :Profit, rev=true)
    
    println("\n🏆 TOP 15 VALIDATED STRATEGIES (Sorted by Test Profit) 🏆")
    println("-----------------------------------------------------------------------")
    println(rpad("Strategy", 22) * " | " * rpad("Bias", 6) * " | Bets | " * rpad("Profit", 8) * " | " * rpad("ROI", 7) * " | Status")
    println("-----------------------------------------------------------------------")
    
    # FIX: Use eachrow() here
    for r in eachrow(first(results, 15)) 
        bias_fmt = (r.Bias_Learned > 0 ? "+" : "") * @sprintf("%.3f", r.Bias_Learned)
        prof_fmt = @sprintf("%6.2f", r.Profit)
        roi_fmt  = @sprintf("%5.1f%%", r.ROI)
        
        println(rpad(r.Strategy, 22) * " | " * rpad(bias_fmt, 6) * " | " * lpad(string(r.Test_Bets), 4) * " | " * prof_fmt * " u | " * roi_fmt * " | " * r.Status)
    end

    println("\n⚠️ WORST FAILURES (Avoid these!)")
    println("-----------------------------------------------------------------------")
    sort!(results, :Profit)
    
    # FIX: Use eachrow() here too
    for r in eachrow(first(results, 5))
         bias_fmt = (r.Bias_Learned > 0 ? "+" : "") * @sprintf("%.3f", r.Bias_Learned)
         println(rpad(r.Strategy, 22) * " | " * rpad(bias_fmt, 6) * " | " * lpad(string(r.Test_Bets), 4) * " | " * @sprintf("%6.2f", r.Profit) * " u | " * @sprintf("%5.1f%%", r.ROI) * " | " * r.Status)
    end
end


validate_all_strategies(df_model_3)
validate_all_strategies(df_model_2)
