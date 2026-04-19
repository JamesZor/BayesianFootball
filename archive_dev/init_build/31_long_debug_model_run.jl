using BayesianFootball


using JLD2
using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

BLAS.set_num_threads(1) 


# data pre 
tournament_id = 56 
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(subset( data_store.matches, 
           :tournament_id => ByRow(isequal(tournament_id)),
                                      :season => ByRow(isequal("24/25")))),
    data_store.odds,
    data_store.incidents
)


model = BayesianFootball.Models.PreGame.AR1Poisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 

# here want to start the expanding window cv ( 1 -38) so 38 - 35 = 3 +1 ( since we have zero ) 4
ds.matches.split_col = max.(0, ds.matches.match_week .- 22);

splitter_config = BayesianFootball.Data.ExpandingWindowCV(
    train_seasons = [], 
    test_seasons = ["24/25"], 
    window_col = :split_col,      # 1. WINDOWING: Split chunks based on this (0, 1, 2...)
    method = :sequential,
    dynamics_col = :match_week      # 2. DYNAMICS: Inside the chunk, evolve time based on this
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

# API is now clean: no extra kwargs needed
feature_sets = BayesianFootball.Features.create_features(
    data_splits, 
    vocabulary, 
    model, 
    splitter_config 
)



# sampler 

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=2, n_warmup=300) # Use renamed struct
training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, feature_sets)

# JLD2.save_object("./dev_exp/eval_debug/debug_ar1_poisson_t55.jld2", results)

JLD2.save_object("./dev_exp/eval_debug/debug_ar1_poisson_t56.jld2", results)




""" compare to a static model """ 
model_1 = BayesianFootball.Models.PreGame.StaticPoisson()
results_1 = BayesianFootball.Training.train(model_1, training_config, feature_sets)
# JLD2.save_object("./dev_exp/eval_debug/debug_static_poisson_t55.jld2", results_1)


JLD2.save_object("./dev_exp/eval_debug/debug_static_poisson_t56.jld2", results_1)

""" compare to grw model """ 
model_2 = BayesianFootball.Models.PreGame.GRWPoisson()
results_2 = BayesianFootball.Training.train(model_2, training_config, feature_sets)

# JLD2.save_object("./dev_exp/eval_debug/debug_grw_poisson_t55.jld2", results_2)

JLD2.save_object("./dev_exp/eval_debug/debug_grw_poisson_t56.jld2", results_2)



####
"""

manual checking 

"""

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)


split_col_sym = :split_col
all_split = sort(unique(ds.matches[!, split_col_sym]))
prediction_split_keys = all_split[2:end] 
grouped_matches = groupby(ds.matches, split_col_sym)

dfs_to_predict = [
    grouped_matches[(; split_col_sym => key)] 
    for key in prediction_split_keys
]



oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict, 
    vocabulary,
    results
)

oos_results_1 = BayesianFootball.Models.PreGame.extract_parameters(
    model_1,
    dfs_to_predict, 
    vocabulary,
    results_1
)

oos_results_2 = BayesianFootball.Models.PreGame.extract_parameters(
    model_2,
    dfs_to_predict, 
    vocabulary,
    results_2
)




models_to_compare = [
    (
        name    = "AR1 Poisson", 
        model   = model,            # Your specific model struct
        results = oos_results       # Your results dictionary
    ),
    (
        name    = "static Poisson", 
        model   = model_1,            # Your specific model struct
        results = oos_results_1       # Your results dictionary
    ),
    (
        name    = "GRW Poisson", 
        model   = model_2,            # Your specific model struct
        results = oos_results_2       # Your results dictionary
    ),
];


mp = filter( row -> row.split_col >= 1 , ds.matches)

num = 9
match_id = mp[num, :match_id]
mp[num, :]

compare_models(match_id, ds, predict_config, models_to_compare, 
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no]
            )


#############################

using Statistics
using Printf
using Dates

# --- 1. Statistical Helper ---

"""
    summarize_chain(chain, market_odds)

Calculates the Model Probability, Fair Odds, Edge, and Kelly Stake.
"""
function summarize_chain(chain, market_odds)
    if isempty(chain)
        return 0.0, 0.0, -1.0, 0.0
    end

    # 1. Model Probability (Mean of posterior)
    model_prob = mean(chain)
    
    # 2. Fair Odds (1 / Probability)
    fair_odds = model_prob > 0 ? 1.0 / model_prob : Inf
    
    # 3. Edge (Expected Value)
    # EV = (Probability * Market Odds) - 1
    edge = (model_prob * market_odds) - 1.0
    
    return model_prob, fair_odds, edge
end

# --- 2. Main Comparison Engine ---

"""
    compare_models(match_id, ds, predict_config, model_inputs; markets=[:home, :draw, :away])

Compare multiple models side-by-side for a specific match.

# Arguments
- `match_id`: Int
- `ds`: The DataStore
- `predict_config`: PredictionConfig
- `model_inputs`: A Vector of NamedTuples: `[(name="Name", model=m, results=r), ...]`
"""
function compare_models(match_id::Int, 
                        ds, 
                        predict_config, 
                        model_inputs::Vector; 
                        markets=[:home, :draw, :away, :over_25, :under_25, :btts_yes])

    # --- A. Setup Match Data ---
    # Fetch market data (Open, Close, Results)
    # We use ds.odds because get_market_data needs the odds dataframe
    open_odds, close_odds, outcomes = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)
    
    match_row = subset(ds.matches, :match_id => ByRow(isequal(match_id)))
    if nrow(match_row) == 0
        println("Match ID $match_id not found in DataStore.")
        return
    end
    home_team = match_row.home_team[1]
    away_team = match_row.away_team[1]
    m_date = match_row.match_date[1]

    # --- B. Pre-Calculate Predictions for all models ---
    # We store these in a Dict to avoid re-calculating inside the market loop
    # Structure: predictions[model_name] = (prediction_dict, kelly_dict)
    model_data = Dict()

    for entry in model_inputs
        name = entry.name
        model = entry.model
        results_dict = entry.results

        if !haskey(results_dict, match_id)
            model_data[name] = nothing # Model doesn't have data for this match
            continue
        end

        # Extract params and predict
        params = results_dict[match_id]
        
        # Predict Market returns Dict{Symbol, Chain}
        pred_market = BayesianFootball.Predictions.predict_market(model, predict_config, params...)
        
        # Calculate Kelly (we need 'open' odds for this)
        # We pass 'open' because we want to see what the stake would have been at open
        kelly_res = BayesianFootball.Signals.bayesian_kelly(pred_market, open_odds)

        model_data[name] = (preds=pred_market, kelly=kelly_res)
    end

    # --- C. Display Dashboard ---
    printstyled("\n══════════════════════════════════════════════════════════════════════════════\n", color=:magenta)
    printstyled(@sprintf(" MATCH %d: %s vs %s \n", match_id, home_team, away_team), bold=true, color=:white)
    printstyled(@sprintf(" Date: %s \n", m_date), color=:light_black)
    printstyled("══════════════════════════════════════════════════════════════════════════════\n", color=:magenta)

    for market_sym in markets
        # 1. Check if market data exists
        if !haskey(open_odds, market_sym)
            continue
        end

        o_price = open_odds[market_sym]
        c_price = haskey(close_odds, market_sym) ? close_odds[market_sym] : 0.0
        
        has_result = haskey(outcomes, market_sym)
        is_win = has_result ? outcomes[market_sym] : false
        res_str = has_result ? (is_win ? "WIN" : "LOSS") : "PENDING"
        res_col = has_result ? (is_win ? :green : :red) : :yellow

        # 2. Market Header
        printstyled("──────────────────────────────────────────────────────────────────────────────\n", color=:light_black)
        printstyled(@sprintf(" %-10s ", string(market_sym)), bold=true, color=:cyan)
        printstyled("Result: ", color=:light_black)
        printstyled("$res_str ", bold=true, color=res_col)
        printstyled(@sprintf("| Open: %.2f | Close: %.2f\n", o_price, c_price), color=:light_black)
        println()

        # 3. Table Header
        printstyled(@sprintf(" %-15s | %-8s | %-8s | %-8s | %-8s\n", "Model", "Prob", "Fair", "Edge", "Kelly"), color=:light_blue)
        println(" " * "-"^65)

        # 4. Loop through models and print rows
        for entry in model_inputs
            name = entry.name
            
            if model_data[name] === nothing
                printstyled(@sprintf(" %-15s | %-30s\n", name, "No Data for Match"), color=:light_black)
                continue
            end

            (preds, kelly_dict) = model_data[name]

            if haskey(preds, market_sym)
                chain = preds[market_sym]
                
                prob, fair, edge = summarize_chain(chain, o_price)
                kelly_stake = get(kelly_dict, market_sym, 0.0)

                # Formatting Colors
                edge_col = edge > 0 ? :green : :light_black
                kelly_col = kelly_stake > 0 ? :yellow : :light_black
                
                # Print Row
                printstyled(@sprintf(" %-15s | %5.1f%%   | %6.2f   | ", name, prob*100, fair), color=:white)
                printstyled(@sprintf("%+5.1f%%", edge*100), color=edge_col)
                printstyled(@sprintf("   | %5.1f%%\n", kelly_stake*100), color=kelly_col)
            else
                printstyled(@sprintf(" %-15s | N/A\n", name), color=:light_black)
            end
        end
        println()
    end
    printstyled("══════════════════════════════════════════════════════════════════════════════\n", color=:magenta)
end






########


"""
    create_calibration_dataframe(ds, predict_config, model_inputs; markets, odds_type=:close)

Generates a DataFrame comparing Model Predictions vs Market Odds vs Outcomes.
Skips matches where market data cannot be retrieved.
"""
function create_calibration_dataframe(ds, predict_config, model_inputs; 
                                      markets=[:home, :draw, :away, :over_25, :under_25],
                                      odds_type=:close)
    rows = []
    
    # 1. Gather all unique Match IDs processed by the models
    all_match_ids = Set{Int}()
    for entry in model_inputs
        union!(all_match_ids, keys(entry.results))
    end
    sorted_match_ids = sort(collect(all_match_ids))

    println("Generating calibration data for $(length(sorted_match_ids)) matches...")

    # 2. Loop through matches
    for match_id in sorted_match_ids
        
        # --- SAFE MARKET DATA RETRIEVAL ---
        # We wrap this in try-catch to handle missing keys in the odds dataframe
        open_odds, close_odds, outcomes = try
            BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)
        catch e
            # Optional: Print warning if you want to see which matches failed
            # println("Warning: Skipping Match $match_id due to missing market data. ($e)")
            continue 
        end

        # Match Metadata
        match_row = subset(ds.matches, :match_id => ByRow(isequal(match_id)))
        if nrow(match_row) == 0; continue; end
        
        m_date = match_row.match_date[1]
        
        # Select which market odds to use as the reference
        ref_odds_dict = odds_type == :open ? open_odds : close_odds

        # 3. Loop through Models
        for entry in model_inputs
            model_name = entry.name
            model = entry.model
            results_dict = entry.results
            
            if !haskey(results_dict, match_id); continue; end
            
            # Predict Market Chains
            params = results_dict[match_id]
            pred_chains = BayesianFootball.Predictions.predict_market(model, predict_config, params...)
            
            for mkt in markets
                if haskey(pred_chains, mkt)
                    
                    # --- Model Stats ---
                    chain = pred_chains[mkt]
                    pred_prob = mean(chain)
                    pred_odds = pred_prob > 0 ? 1.0 / pred_prob : Inf
                    
                    # --- Market Stats ---
                    market_odd = get(ref_odds_dict, mkt, missing)
                    outcome = get(outcomes, mkt, missing) # true/false/missing
                    
                    # Market implied probability
                    market_prob = ismissing(market_odd) ? missing : 1.0 / market_odd

                    push!(rows, (;
                        match_id = match_id,
                        date = m_date,
                        model = model_name,
                        market = mkt,
                        # Predictions
                        pred_prob = pred_prob,
                        pred_odds = pred_odds,
                        # Market
                        market_odds = market_odd,
                        market_prob = market_prob,
                        # Results
                        outcome = outcome,
                        outcome_int = ismissing(outcome) ? missing : (outcome ? 1 : 0)
                    ))
                end
            end
        end
    end
    
    return DataFrame(rows)
end




# Create the DataFrame
df_calib = create_calibration_dataframe(
    ds, 
    predict_config, 
    models_to_compare, 
    markets=[:home, :draw, :away, :under_05, :over_05, :under_15, :over_15, :over_25, :under_25, :over_35, :under_35, :btts_yes, :btts_no],
    odds_type=:close # Using closing odds is best for checking calibration
)

# Clean up: Drop rows where we don't have market odds (optional)
df_calib = dropmissing(df_calib, :market_odds)

# Preview
first(df_calib, 5)


using Statistics

"""
    get_correlations(df)

Calculates the Pearson correlation between Model Probabilities and Market Probabilities
grouped by Model and Market.
"""
function get_correlations(df)
    # 1. Filter out missing data to ensure valid correlation calculation
    # We need rows where we have BOTH a prediction and a market price
    clean_df = dropmissing(df, [:pred_prob, :market_prob])

    # 2. Group and Calculate
    cor_df = combine(groupby(clean_df, [:model, :market]),
        # Calculate Pearson Correlation (r)
        [:pred_prob, :market_prob] => ((p, m) -> cor(p, m)) => :pearson_r,
        
        # Count samples (useful to know if N is too small)
        nrow => :n_samples
    )
    
    # 3. Sort by Market then by Correlation (highest first)
    sort!(cor_df, [:market, :pearson_r], rev=[false, true])
    
    return cor_df
end

# --- Run the calculation ---
df_corr = get_correlations(df_calib)

# Display the top results
show(df_corr, allrows=true)



##

using Plots 

# Filter for a specific market (e.g., Home Wins)
sub_df = filter(row -> row.market == :over_25, df_calib)

scatter(
    sub_df.market_prob, 
    sub_df.pred_prob, 
    group=sub_df.model,
    xlabel="Market Implied Probability",
    ylabel="Model Predicted Probability",
    title="Model vs Market Consensus (Home Win)",
    legend=:topleft,
    alpha=0.6,
    aspect_ratio=:equal
)
plot!([0, 1], [0, 1], label="Perfect Correlation", color=:black, linestyle=:dash)

#


function plot_calibration(df, model_name; n_bins=10)
    # Filter for specific model and remove missing outcomes
    m_df = filter(row -> row.model == model_name && !ismissing(row.outcome), df)
    
    if nrow(m_df) == 0
        println("No data found for model: $model_name")
        return
    end

    # --- Manual Binning ---
    # We map probabilities 0.0-1.0 to integer indices 0 to (n_bins-1)
    # e.g., 0.15 becomes bin 1 (if n_bins=10)
    m_df.bin_idx = floor.(Int, m_df.pred_prob .* n_bins)
    
    # Handle the edge case where prob == 1.0 (clamp it to the last bin)
    m_df.bin_idx = clamp.(m_df.bin_idx, 0, n_bins - 1)
    
    # Group by this new bin index
    gdf = groupby(m_df, :bin_idx)
    calib = combine(gdf, 
        :outcome_int => mean => :actual_rate,
        :pred_prob => mean => :avg_pred_prob,
        nrow => :count
    )
    
    # Sort by probability so the line connects correctly (if you switch to line plot later)
    sort!(calib, :avg_pred_prob)

    # --- Plotting ---
    p = plot(calib.avg_pred_prob, calib.actual_rate, 
         seriestype=:scatter, 
         label=model_name, 
         xlabel="Predicted Probability", 
         ylabel="Actual Win Rate",
         title="Calibration: $model_name",
         xlims=(0,1), ylims=(0,1),
         legend=:bottomright,
         markersize=6,
         color=:blue)
    
    # Add the "Perfect Calibration" diagonal line
    plot!(p, [0, 1], [0, 1], color=:black, linestyle=:dash, label="Perfect")
    
    # Optional: Add error bars or bubble sizes based on count
    # scatter!(p, calib.avg_pred_prob, calib.actual_rate, markersize=log.(calib.count).+2, alpha=0.3, label=nothing, color=:blue)

    display(p)
end


plot_calibration(df_calib, "AR1 Poisson")

# Group by Model and Market to get a scorecard
scorecard = combine(groupby(dropmissing(df_calib, :outcome_int), [:model, :market]),
    :pred_prob => (p -> mean((p .- df_calib[axes(p,1), :outcome_int]).^2)) => :brier_score,
    :pred_prob => (p -> -mean(log.(clamp.(p, 1e-6, 1.0)) .* df_calib[axes(p,1), :outcome_int] .+ 
                               log.(clamp.(1 .- p, 1e-6, 1.0)) .* (1 .- df_calib[axes(p,1), :outcome_int]))) => :log_loss,
    nrow => :n_matches
)

sort!(scorecard, [:market, :brier_score])


# edge 
using Plots

function plot_model_market_divergence(df, model_name, market_sym)
    # Filter data
    sub_df = filter(row -> row.model == model_name && row.market == market_sym, df)
    dropmissing!(sub_df, [:pred_prob, :market_prob])

    # Plot
    p = scatter(sub_df.market_prob, sub_df.pred_prob, 
        xlabel="Market Probability", 
        ylabel="Model Probability ($model_name)", 
        title="Divergence: $model_name vs Market ($market_sym)",
        legend=:topleft,
        alpha=0.7,
        label="Match"
    )
    
    # Reference Line
    plot!(p, [0, 1], [0, 1], color=:red, linestyle=:dash, label="Perfect Agreement")
    
    # Add quadrants to see bias
    # If points are mostly ABOVE the line, Model loves the outcome more than Market
    # If points are mostly BELOW the line, Model hates the outcome more than Market
    
    display(p)
end

# Run this to see if AR1 is Over-bullish or Over-bearish on goals
plot_model_market_divergence(df_calib, "AR1 Poisson", :over_25)
plot_model_market_divergence(df_calib, "GRW Poisson", :over_25)

using Plots

function plot_model_market_divergence(df, model_name, market_sym)
    # Filter data
    sub_df = filter(row -> row.model == model_name && row.market == market_sym, df)
    dropmissing!(sub_df, [:pred_prob, :market_prob, :outcome_int]) # Ensure outcome is there

    # Split into Winners (1) and Losers (0) for coloring
    winners = filter(row -> row.outcome_int == 1, sub_df)
    losers  = filter(row -> row.outcome_int == 0, sub_df)

    # Base Plot
    p = plot(
        xlabel="Market Probability (Implied)", 
        ylabel="Model Probability ($model_name)", 
        title="Divergence: $model_name vs Market ($market_sym)",
        legend=:topleft,
        aspect_ratio=:equal,
        xlims=(0,1), ylims=(0,1)
    )
    
    # Reference Line (y=x)
    plot!(p, [0, 1], [0, 1], color=:grey, linestyle=:dash, label="Agreement")

    # Plot Losers (Red) first (so winners sit on top)
    scatter!(p, losers.market_prob, losers.pred_prob, 
        color=:red, alpha=0.6, label="Outcome: FALSE", markersize=4)

    # Plot Winners (Green)
    scatter!(p, winners.market_prob, winners.pred_prob, 
        color=:green, alpha=0.6, label="Outcome: TRUE", markersize=4)
        
    display(p)
end


###
"""
    add_fair_market_probs!(df)

Removes the vig/margin from the market probabilities using basic normalization.
(Prop_True = Prop_Market / (Prop_Over + Prop_Under))
"""
function add_fair_market_probs!(df)
    # 1. Pivot to get Over and Under side-by-side
    df_over = filter(r -> r.market == :over_25, df)
    df_under = filter(r -> r.market == :under_25, df)
    
    # Join on match_id and model to align them
    df_joined = innerjoin(df_over, df_under, on=[:match_id, :model], makeunique=true)
    
    # 2. Calculate Total Implied Probability (The "Overround")
    # market_prob is from Over, market_prob_1 is from Under
    df_joined.total_implied = df_joined.market_prob .+ df_joined.market_prob_1
    
    # 3. Calculate "Fair" Market Probability
    df_joined.fair_market_prob_over = df_joined.market_prob ./ df_joined.total_implied
    
    return df_joined
end

# Run and Plot
df_fair = add_fair_market_probs!(df_calib)

scatter(df_fair.fair_market_prob_over, df_fair.pred_prob, 
    xlabel="Fair Market Prob (Vig Removed)", 
    ylabel="Model Prob (AR1)",
    title="True Disagreement (Vig Removed)",
    legend=false,
    aspect_ratio=:equal,
    xlims=(0,1), ylims=(0,1)
)
plot!([0,1], [0,1], color=:red, linestyle=:dash)


##
using DataFrames, Statistics, Plots

"""
    backtest_strategy(df, model_name, market_sym; threshold=0.0)

Simulates betting on a specific model and market whenever there is positive Expected Value (EV).

# Arguments
- `threshold`: Minimum edge required to bet (e.g., 0.05 for 5% edge).
"""
function backtest_strategy(df, model_name, market_sym; threshold=0.0)
    # 1. Filter for the specific model and market
    # We only want rows where we have an outcome (to calculate profit) and odds
    strat_df = filter(row -> 
        row.model == model_name && 
        row.market == market_sym && 
        !ismissing(row.outcome) && 
        !ismissing(row.market_odds), 
    df)
    
    sort!(strat_df, :date) # Ensure chronological order

    # 2. Calculate Edge and Stakes
    # Edge = (ModelProb * Odds) - 1
    strat_df.edge = (strat_df.pred_prob .* strat_df.market_odds) .- 1.0
    
    # Filter for value bets (Edge > threshold)
    bets = filter(row -> row.edge > threshold, strat_df)
    
    if nrow(bets) == 0
        println("No bets found with edge > $threshold")
        return nothing
    end

    # 3. Calculate Results (Flat Stake = 1 unit)
    # If outcome is TRUE (Win): Profit = Odds - 1
    # If outcome is FALSE (Loss): Profit = -1
    bets.pnl = [row.outcome ? (row.market_odds - 1.0) : -1.0 for row in eachrow(bets)]
    
    # Cumulative Profit
    bets.cum_pnl = cumsum(bets.pnl)

    # 4. Summary Stats
    total_bets = nrow(bets)
    total_profit = sum(bets.pnl)
    roi = (total_profit / total_bets) * 100
    win_rate = mean(bets.outcome) * 100
    
    println("--- Backtest Results: $model_name on $market_sym ---")
    println("Total Bets:   $total_bets")
    println("Win Rate:     $(round(win_rate, digits=1))%")
    println("Total Profit: $(round(total_profit, digits=2)) units")
    println("Yield (ROI):  $(round(roi, digits=2))%")
    println("------------------------------------------------")

    return bets
end


# Run backtest for AR1 Poisson on Under 2.5
# We use a 0.0 (0%) threshold - betting on ANY positive edge
bets_df = backtest_strategy(df_calib, "AR1 Poisson", :under_25, threshold=0.0)

# Visualization: Cumulative Profit Chart
if !isnothing(bets_df)
    plot(bets_df.cum_pnl, 
        title="Cumulative P&L: AR1 Poisson (Under 2.5)",
        label="Profit (Flat Stake)",
        xlabel="Bet Number",
        ylabel="Units Won",
        legend=:topleft,
        linewidth=2,
        color=:green,
        fill=(0, 0.2, :green) # Shading under the line
    )
    # Add a zero line
    hline!([0], color=:black, linestyle=:dash, label="")
end

###


# pp1. Calculate the "Hurdle Rate" (The Vig)

"""
    check_market_margin(df)

Calculates the average bookmaker margin (overround) for the Over/Under market.
Margin = (1/Over_Odds + 1/Under_Odds) - 1
"""
function check_market_margin(df)
    # Filter for Over/Under 2.5
    df_over = filter(r -> r.market == :over_25 && !ismissing(r.market_odds), df)
    df_under = filter(r -> r.market == :under_25 && !ismissing(r.market_odds), df)

    # Join to get matching odds for the same match
    df_joined = innerjoin(df_over, df_under, on=[:match_id, :model], makeunique=true)
    
    # Calculate Margin per match
    # Margin = (1/Over + 1/Under) - 1.0
    df_joined.margin = (1.0 ./ df_joined.market_odds) .+ (1.0 ./ df_joined.market_odds_1) .- 1.0
    
    avg_margin = mean(df_joined.margin)
    max_margin = maximum(df_joined.margin)
    
    println("--- Market Efficiency Report (Scottish L1) ---")
    println("Average Margin (Vig):  $(round(avg_margin * 100, digits=2))%")
    println("Max Margin Observed:   $(round(max_margin * 100, digits=2))%")
    println("Breakeven Win Rate @ 1.83 odds: $(round((1/1.83)*100, digits=1))%")
    
    return df_joined
end

# Check how steep the hill is
margin_data = check_market_margin(df_calib)


# 1. Strict Backtest: Only bet if Edge > Vig (6%)
bets_strict = backtest_strategy(df_calib, "static Poisson", :under_25, threshold=0.06)

# 2. Visualize the Result
if !isnothing(bets_strict)
    p = plot(bets_strict.cum_pnl, 
         title="P&L: AR1 Poisson (Edge > 6%)", 
         label="Profit (Flat Stake)", 
         xlabel="Bet Number",
         ylabel="Units Won",
         legend=:topleft,
         linewidth=2,
         color=:blue,
         fill=(0, 0.1, :blue)
    )
    hline!(p, [0], color=:black, linestyle=:dash, label="")
    display(p)
else
    println("No bets found with an edge > 6%. Model is too conservative relative to the vig.")
end


# Re-generate using OPENING odds for realistic backtesting
df_calib_open = create_calibration_dataframe(
    ds, 
    predict_config, 
    models_to_compare, 
    markets=[:home, :draw, :away, :over_25, :under_25],
    odds_type=:open  # <--- CHANGED to Open
)

# Now check the margin again (Opening margins are often slightly higher/different)
check_market_margin(df_calib_open)

# And run the backtest on the Opening prices
bets_open = backtest_strategy(df_calib_open, "AR1 Poisson", :over_25, threshold=0.06)


using Statistics, Plots

"""
    check_goal_bias(df)

Compares the average 'Expected Goals' (implied by model probability) vs Actual Goals.
"""
function check_goal_bias(df)
    # Filter for Over 2.5 market to get probabilities
    df_goals = filter(r -> r.market == :over_25, df)
    
    # Group by Model to see the average "Confidence" in goals
    bias_check = combine(groupby(df_goals, :model),
        :pred_prob => mean => :avg_prob_over_25,
        :outcome_int => mean => :actual_freq_over_25,
        nrow => :n_matches
    )
    
    # Calculate the Bias (Model - Reality)
    bias_check.bias = bias_check.avg_prob_over_25 .- bias_check.actual_freq_over_25
    
    return bias_check
end

# Run the check
bias_df = check_goal_bias(df_calib_open)
show(bias_df, allrows=true)
"""

julia> # Run the check
       bias_df = check_goal_bias(df_calib_open)
3×5 DataFrame
 Row │ model           avg_prob_over_25  actual_freq_over_25  n_matches  bias      
     │ String          Float64           Float64              Int64      Float64   
─────┼─────────────────────────────────────────────────────────────────────────────
   1 │ AR1 Poisson             0.411404             0.649351         77  -0.237947
   2 │ static Poisson          0.416786             0.649351         77  -0.232565
   3 │ GRW Poisson             0.4242               0.649351         77  -0.225151

julia> show(bias_df, allrows=true)
3×5 DataFrame
 Row │ model           avg_prob_over_25  actual_freq_over_25  n_matches  bias      
     │ String          Float64           Float64              Int64      Float64   
─────┼─────────────────────────────────────────────────────────────────────────────
   1 │ AR1 Poisson             0.411404             0.649351         77  -0.237947
   2 │ static Poisson          0.416786             0.649351         77  -0.232565
   3 │ GRW Poisson             0.4242               0.649351         77  -0.225151
julia> 

"""



using Distributions

"""
    calibrate_model_goals(df, model_name)

Calculates a 'Goal Multiplier' to align model predictions with recent reality.
"""
function calibrate_model_goals(df, model_name; filter_sym = :over_25)
    # 1. Get Model's Implied Lambda (Expected Goals)
    # We approximate this from the Over 2.5 probability using Poisson inverse
    # P(X > 2.5) = 1 - P(0) - P(1) - P(2)
    # This is hard to invert analytically, so we approximate or use the raw probabilities.
    
    # Simpler approach: Look at the Bias in Probabilities we found earlier
    # Bias = -0.23 (Model is 23% too low on Over 2.5)
    
    # Let's find the scalar to add to probabilities
    sub_df = filter(r -> r.model == model_name && r.market == filter_sym, df)
    
    avg_pred = mean(sub_df.pred_prob)
    actual_freq = mean(sub_df.outcome_int)
    
    diff = actual_freq - avg_pred
    ratio = actual_freq / avg_pred
    
    println("--- Calibration Factor: $model_name ---")
    println("Avg Predicted Prob: $(round(avg_pred, digits=3))")
    println("Actual Frequency:   $(round(actual_freq, digits=3))")
    println("Deficit:            $(round(diff, digits=3))")
    println("Multiplier Needed:  $(round(ratio, digits=2))x")
    
    return diff
end

adjustment = calibrate_model_goals(df_calib_open, "static Poisson")

adjustment = calibrate_model_goals(df_calib_open, "AR1 Poisson", filter_sym=:home)

adjustment = calibrate_model_goals(df_calib_open, "static Poisson", filter_sym=:under_25)


adjustment = calibrate_model_goals(df_calib_open, "AR1 Poisson", filter_sym=:under_15)

"""
julia> adjustment = calibrate_model_goals(df_calib_open, "AR1 Poisson")
--- Calibration Factor: AR1 Poisson ---
Avg Predicted Prob: 0.411
Actual Frequency:   0.649
Deficit:            0.238
Multiplier Needed:  1.58x
0.23794702933002554

julia> adjustment = calibrate_model_goals(df_calib_open, "GRW Poisson")
--- Calibration Factor: GRW Poisson ---
Avg Predicted Prob: 0.424
Actual Frequency:   0.649
Deficit:            0.225
Multiplier Needed:  1.53x
0.22515079471170535

julia> adjustment = calibrate_model_goals(df_calib_open, "static Poisson")
--- Calibration Factor: static Poisson ---
Avg Predicted Prob: 0.417
Actual Frequency:   0.649
Deficit:            0.233
Multiplier Needed:  1.56x
0.232565059352695



"""


function backtest_calibrated(df, model_name, adjustment; filter_sym = :over_25)
    # Copy dataframe to avoid mutating original
    test_df = copy(df)
    
    # Filter for Over 2.5
    filter!(r -> r.model == model_name && r.market == filter_sym, test_df)
    
    # Apply Calibration: Boost the probability by the observed deficit
    # We clamp at 0.99 to avoid impossible probs
    test_df.pred_prob = clamp.(test_df.pred_prob .+ adjustment, 0.01, 0.99)
    
    # Re-calculate Edge
    # Edge = (New_Prob * Odds) - 1
    test_df.edge = (test_df.pred_prob .* test_df.market_odds) .- 1.0
    
    # Bet if Edge > 6% (Vig)
    bets = filter(r -> r.edge > 0.06, test_df)
    
    if nrow(bets) == 0
        println("No bets found after calibration.")
        return
    end
    
    # Calculate P&L
    bets.pnl = [r.outcome ? (r.market_odds - 1.0) : -1.0 for r in eachrow(bets)]
    total_pnl = sum(bets.pnl)
    roi = (total_pnl / nrow(bets)) * 100
    
    println("--- Calibrated Backtest ($(String(filter_sym)) ---")
    println("Adjustment:   +$(round(adjustment, digits=2)) to probs")
    println("Bets Placed:  $(nrow(bets))")
    println("Total Profit: $(round(total_pnl, digits=2)) units")
    println("ROI:          $(round(roi, digits=2))%")
end

# Run the simulation with the +0.23 adjustment we found
backtest_calibrated(df_calib_open, "AR1 Poisson", 0.23)
backtest_calibrated(df_calib_open, "GRW Poisson", 0.23)
backtest_calibrated(df_calib_open, "static Poisson", -0.12 , filter_sym = :under_25)



backtest_calibrated(df_calib_open, "AR1 Poisson", 0.02 , filter_sym=:home)
