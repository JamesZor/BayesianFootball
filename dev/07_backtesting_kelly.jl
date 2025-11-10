using Revise
using BayesianFootball
using DataFrames
using JLD2


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
# Dev area 
#################################################

using DataFramesMeta
using Statistics, Distributions

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )



match_id = rand(keys(all_oos_results))
r1 =  all_oos_results[match_id]
subset( ds.matches, :match_id => ByRow(isequal(match_id)))

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)


model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)



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




kf = kelly_fraction(match_odds, match_predict)

Dict( k => median(v) for (k, v) in pairs(kf))

function get_confidence(kelly_dist; threshold = 0.0)
    return mean(kelly_dist .> threshold)
end

function get_positive_median(kelly_dist; threshold = 0.0)
    positive_stakes = kelly_dist[kelly_dist .> threshold]
    return isempty(positive_stakes) ? 0.0 : median(positive_stakes)
end

data_rows = []


key_i = rand(keys(kf))
threshold = 0.5

k_i = mean(kf[:away] .> threshold)

get_confidence(kf[:away], threshold=0.7)
get_positive_median(kf[:away], threshold=0.0)
get_positive_median(kf[:away], threshold=0.7)



"""
Calculates the Profit and Loss for a given bet.
- stake: The fraction of bankroll to bet.
- odds: The decimal odds for the bet.
- winner: A Bool, true if the bet won, false if it lost.
"""
function calculate_pnl(stake::Number, odds::Number, winner::Bool)
    if stake <= 0.0
        return 0.0 # No bet was placed
    end
    
    if winner
        return stake * (odds - 1.0) # Profit
    else
        return -stake # Loss
    end
end


"""
Analyzes a single market's Kelly distribution against a range of thresholds.

# Arguments
- `market_key`: The symbol for the market (e.g., :away).
- `kelly_dists`: The NamedTuple containing the full Kelly distributions (your `kf`).
- `market_odds`: A NamedTuple mapping market keys to their single decimal odds.
- `market_results`: A NamedTuple mapping market keys to their boolean win/loss result.
- `threshold_range`: A range of thresholds to test (e.g., 0.0:0.01:0.5).
"""
function analyze_market_thresholds(
    market_key::Symbol,
    kelly_dists::NamedTuple,
    market_odds::NamedTuple,
    market_results::NamedTuple;
    threshold_range=0.0:0.01:0.5 # Default range
)

    # 2. Extract the data for this specific market
    kelly_dist = kelly_dists[market_key]
    odds = market_odds[market_key]
    winner = market_results[market_key]

    # 3. Calculate the metrics that are constant (don't depend on the loop)
    pos_median_zero = get_positive_median(kelly_dist, threshold=0.0)
    pnl_zero = calculate_pnl(pos_median_zero, odds, winner)

    # 4. Initialize an array to hold our row data
    #    We use NamedTuples for type stability and performance
    data_rows = []

    # 5. Loop over the threshold range and calculate metrics
    for thresh in threshold_range
        
        # Calculate threshold-dependent metrics
        confidence = get_confidence(kelly_dist, threshold=thresh)
        pos_median_thresh = get_positive_median(kelly_dist, threshold=thresh)
        pnl_thresh = calculate_pnl(pos_median_thresh, odds, winner)

        # Create the row
        row = (
            market = market_key,
            kelly_threshold = thresh,
            confidence = confidence,
            positive_median_zero = pos_median_zero,
            positive_median_thresh = pos_median_thresh,
            winner = winner,
            pnl_zero = pnl_zero,
            pnl_thresh = pnl_thresh
        )
        
        push!(data_rows, row)
    end

    # 6. Convert the array of NamedTuples into a DataFrame
    return DataFrame(data_rows)
end

thresholds_to_test = 0.0:0.05:0.5
df_away = analyze_market_thresholds(
    :away,
    kf,
    match_odds,
    match_results,
    threshold_range=thresholds_to_test
)

"""
Analyzes all common markets for a single match and combines them into one DataFrame.

# Arguments
- `kelly_dists`: NamedTuple of (market => kelly_distribution_vector)
- `market_odds`: NamedTuple of (market => decimal_odds)
- `market_results`: NamedTuple of (market => winner_bool)
- `threshold_range`: A range of thresholds to test (e.g., 0.0:0.01:0.5).
"""
function analyze_match_thresholds(
    kelly_dists::NamedTuple,
    market_odds::NamedTuple,
    market_results::NamedTuple;
    threshold_range=0.0:0.01:0.5
)
    # Find markets that exist in all three inputs
    common_keys = keys(kelly_dists) ∩ keys(market_odds) ∩ keys(market_results)
    
    if isempty(common_keys)
        println("Warning: No common markets found. Returning an empty DataFrame.")
        return DataFrame() # Return an empty DF
    end
    
    # Use a comprehension to generate a DataFrame for each market
    all_market_dfs = [
        analyze_market_thresholds(
            key, 
            kelly_dists, 
            market_odds, 
            market_results, 
            threshold_range=threshold_range
        )
        for key in common_keys
    ]
    
    # Concatenate all of them vertically into one big DataFrame
    # The '...' splats the array into individual arguments for vcat
    return vcat(all_market_dfs...)
end


df_all_markets = analyze_match_thresholds(
    kf,
    match_odds,
    match_results,
    threshold_range=thresholds_to_test
)


"""
Runs the full out-of-sample threshold analysis for all matches.

# Arguments
- `all_oos_results`: Your Dict of match_id => model_posteriors.
- `model`: Your BayesianFootball model object.
- `predict_config`: Your configuration object.
- `ds`: Your main dataset (or at least `ds.odds`).
- `threshold_range`: The range of thresholds to test (e.g., 0.0:0.01:0.5).

# Returns
- A single DataFrame containing the analysis for all matches,
  with a `match_id` column.
"""
function run_full_oos_analysis(
    all_oos_results::Dict, 
    model, 
    predict_config, 
    ds; 
    threshold_range=0.0:0.01:0.5
)
    
    all_match_dfs = DataFrame[] # Initialize an array to hold DataFrames

    println("Starting analysis for $(length(all_oos_results)) matches...")

    # Loop over every match in your results dictionary
    for (match_id, r1) in all_oos_results
        try
            # --- This is your "recipe" from the prompt ---
            
            # 1. Get model predictions
            match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
            
            # 2. Get bookmaker odds
            match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)

            # 3. Get match results
            match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)

            # 4. Calculate Kelly Fractions for all markets
            #    (This uses the kelly_fraction NamedTuple method)
            kf = kelly_fraction(match_odds, match_predict)

            # 5. Run the threshold analysis for this single match
            df_one_match = analyze_match_thresholds(
                kf, 
                match_odds, 
                match_results, 
                threshold_range=threshold_range
            )

            # 6. (IMPORTANT) Add the match_id column
            if !isempty(df_one_match)
                df_one_match.match_id .= match_id
                push!(all_match_dfs, df_one_match)
            else
                println("  - No common markets for match $match_id. Skipping.")
            end

        catch e
            println("  - ⚠️ WARNING: Skipping match $match_id due to error: $e")
        end
    end

    println("...Analysis complete. Combining $(length(all_match_dfs)) DataFrames.")

    if isempty(all_match_dfs)
        return DataFrame() # Return empty if no matches worked
    end
    
    # 7. Combine all individual DataFrames into one
    return vcat(all_match_dfs...)
end


thresholds_to_test = 0.0:0.02:0.9

# Run the analysis over all 520 matches
df_all_matches = run_full_oos_analysis(
    all_oos_results, 
    model, 
    predict_config, 
    ds, 
    threshold_range=thresholds_to_test
)



"""
Calculates the maximum drawdown for a vector of PnLs.
Assumes PnLs are in chronological order.
"""
function calculate_max_drawdown(pnl_series::AbstractVector)
    # A 0.0 at the start represents the initial bankroll
    portfolio_value = [0.0; cumsum(pnl_series)]
    
    # Calculate the running "peak" of the portfolio
    peaks = accumulate(max, portfolio_value)
    
    # The drawdowns are the difference between the peak and the current value
    drawdowns = peaks - portfolio_value
    
    # The max drawdown is the largest of these differences
    max_dd = maximum(drawdowns)
    
    # Return 1.0 if no drawdown to avoid division by zero
    return max_dd > 0.0 ? max_dd : 1.0 
end


function analyze_total_pnl(df_all_matches::DataFrame)
    
    # Group by each strategy
    grouped = groupby(df_all_matches, [:market, :kelly_threshold])

    # Combine to sum up PnL and get average confidence
    profit_curve = combine(grouped) do g
        (
            total_pnl = sum(g.pnl_thresh),
            avg_confidence = mean(g.confidence),
            # Count matches where this strategy placed a bet
            bet_count = sum(g.positive_median_thresh .> 0.0) 
        )
    end
    
    # Sort to find the best threshold for each market
    sort!(profit_curve, [:market, :total_pnl], rev=[false, true])
    
    return profit_curve
end

profit_report = analyze_total_pnl(df_all_matches)

sort(profit_report, :total_pnl, rev=true)



function analyze_calmar_ratio(df_all_matches::DataFrame)
    
    # CRITICAL: You must sort by a time/match ID first!
    sort!(df_all_matches, :match_id)

    # Group by each strategy
    grouped = groupby(df_all_matches, [:tournament_slug, :market, :kelly_threshold])

    # Combine to get the full PnL series for each strategy
    strategy_report = combine(grouped) do g
        
        pnl_series = g.pnl_thresh
        
        # Calculate our key metrics for this PnL series
        total_pnl = sum(pnl_series)
        max_drawdown = calculate_max_drawdown(pnl_series)
        
        # Calculate the Calmar Ratio
        calmar_ratio = total_pnl / max_drawdown 
        
        return (
            total_pnl = total_pnl,
            max_drawdown = max_drawdown,
            calmar_ratio = calmar_ratio,
            avg_confidence = mean(g.confidence),
            bet_count = sum(g.positive_median_thresh .> 0.0)
        )
    end
    
    # Sort to find the best risk-adjusted strategies
    sort!(strategy_report, :calmar_ratio, rev=true)
    
    return strategy_report
end


# --- How to run it ---
# (Assuming 'df_all_matches' is your giant DataFrame)

risk_report = analyze_calmar_ratio(df_all_matches)

# Show the top 10 best strategies overall
# println(first(risk_report, 10))

# To see the best risk-adjusted strategy for each market:
best_risk_strategies = combine(groupby(risk_report, :market)) do g
    # Find the row with the maximum Calmar ratio within that group
    g[argmax(g.calmar_ratio), :]
end
# println(best_risk_strategies)


# Filter for strategies that are profitable (Calmar > 0)
profitable_strategies = filter(row -> row.calmar_ratio > 0, best_risk_strategies)

# Sort them by their risk-adjusted return
sort!(profitable_strategies, :calmar_ratio, rev=true)

println(profitable_strategies)



########

names(df_all_matches)


match_info = unique(ds.matches[:, [:match_id, :tournament_slug, :season, :match_date]], :match_id)
master_df = leftjoin(df_all_matches, match_info, on = :match_id)
master_df


risk_report = analyze_calmar_ratio(master_df)

sort(profit_report, :total_pnl, rev=true)

best_risk_strategies = combine(groupby(risk_report, [:tournament_slug, :market])) do g
    # Find the row with the maximum Calmar ratio within that group
    g[argmax(g.calmar_ratio), :]
end

# println(best_risk_strategies)



#############

function analyze_market_thresholds(
    market_key::Symbol,
    kelly_dists::NamedTuple,
    market_odds::NamedTuple,
    market_results::NamedTuple;
    threshold_range=0.0:0.01:0.5
)

    if !(market_key in keys(kelly_dists) && 
         market_key in keys(market_odds) && 
         market_key in keys(market_results))
        error("Market key :$market_key not found in all inputs.")
    end

    kelly_dist = kelly_dists[market_key]
    odds = market_odds[market_key]
    winner = market_results[market_key]
    
    pos_median_zero = get_positive_median(kelly_dist, threshold=0.0)
    pnl_zero = calculate_pnl(pos_median_zero, odds, winner)

    data_rows = []
    for thresh in threshold_range
        confidence = get_confidence(kelly_dist, threshold=thresh)
        pos_median_thresh = get_positive_median(kelly_dist, threshold=thresh)
        pnl_thresh = calculate_pnl(pos_median_thresh, odds, winner)

        row = (
            market = market_key,
            kelly_threshold = thresh,
            confidence = confidence,
            positive_median_zero = pos_median_zero,
            positive_median_thresh = pos_median_thresh,
            winner = winner,
            pnl_zero = pnl_zero,
            pnl_thresh = pnl_thresh,
            odds = odds  # <--- THIS IS THE NEW LINE
        )
        push!(data_rows, row)
    end
    return DataFrame(data_rows)
end

df_all_matches = run_full_oos_analysis(
    all_oos_results, 
    model, 
    predict_config, 
    ds, 
    threshold_range=thresholds_to_test # Use your defined range
)


match_info = unique(ds.matches[:, [:match_id, :tournament_slug, :season, :match_date]], :match_id)
master_df = leftjoin(df_all_matches, match_info, on = :match_id)


"""
Calculates the geometric wealth accumulation over time.
Starts with a bankroll of 1.0.
Applies a kelly_fraction multiplier to all stakes.
"""
function calculate_geometric_wealth(
    stakes::AbstractVector, 
    odds::AbstractVector, 
    winners::AbstractVector,
    kelly_fraction::Number 
)
    wealth = 1.0
    wealth_series = [1.0] # Start with 1.0
    
    for i in 1:length(stakes)
        # Apply the fractional kelly multiplier
        stake = stakes[i] * kelly_fraction # <-- NEW LINE
        
        if stake > 0.0
            # Ensure stake is never > 1.0 (can happen with bad data)
            stake = min(stake, 1.0)
            
            stake_amount = wealth * stake
            
            if winners[i]
                wealth += stake_amount * (odds[i] - 1.0)
            else
                wealth -= stake_amount
            end
        end
        
        if wealth <= 0.0
            wealth = 0.0
            append!(wealth_series, zeros(length(stakes) - i + 1))
            break
        end
        
        push!(wealth_series, wealth)
    end
    
    return wealth_series
end

"""
Calculates the max drawdown for a wealth series.
"""
function calculate_max_drawdown(wealth_series::AbstractVector)
    peaks = accumulate(max, wealth_series)
    drawdowns = (peaks - wealth_series) ./ peaks
    # Set NaNs (from 0.0 / 0.0) to 0.0
    drawdowns[isnan.(drawdowns)] .= 0.0 
    return maximum(drawdowns) # Return max percentage drawdown
end


function analyze_strategy_performance(master_df::DataFrame; grouping_cols=[:market], kelly_fraction::Number)
    
    # Define the full strategy group
    strategy_group = [grouping_cols..., :kelly_threshold]
    
    # Group by the strategy
    grouped = groupby(master_df, strategy_group, sort=true)

    # Run the backtest for each strategy
    report = combine(grouped) do g
        
        # CRITICAL: Sort by date to get a correct time-series
        sort!(g, :match_date)

        # Get the inputs for this strategy
        stakes = g.positive_median_thresh
        odds = g.odds
        winners = g.winner
        
        bet_count = sum(stakes .> 0.0)
        
        # Don't analyze strategies that never bet
        if bet_count == 0
            return (
                final_wealth = 1.0,
                total_pnl_additive = 0.0,
                roi = 0.0,
                win_rate = 0.0,
                max_drawdown_pct = 0.0,
                calmar_ratio = 0.0,
                sharpe_ratio = 0.0,
                sortino_ratio = 0.0,
                bet_count = 0
            )
        end
        
        # 1. Run Geometric Backtest
        wealth_series = calculate_geometric_wealth(stakes, odds, winners, kelly_fraction)
        final_wealth = last(wealth_series)
        
        # 2. Calculate PnL / ROI
        # Additive PnL is still a useful metric
        total_pnl_additive = sum(g.pnl_thresh) 
        # ROI based on additive model (Total PnL / Total Stake)
        total_staked_additive = sum(stakes)
        roi = total_staked_additive > 0 ? total_pnl_additive / total_staked_additive : 0.0
        
        # 3. Calculate Win Rate (of bets placed)
        win_rate = mean(winners[stakes .> 0.0])
        
        # 4. Calculate Risk Ratios (Calmar)
        max_drawdown_pct = calculate_max_drawdown(wealth_series)
        # Geometric profit (e.g., 1.5 wealth -> 0.5 profit)
        geo_profit = final_wealth - 1.0 
        local calmar_ratio::Float64
        if geo_profit <= 0.0
            # If not profitable, Calmar is 0 or negative
            calmar_ratio = max_drawdown_pct > 0 ? geo_profit / max_drawdown_pct : 0.0
        elseif max_drawdown_pct == 0.0 && geo_profit > 0.0
            # "Perfect" strategy: profitable with zero drawdown
            calmar_ratio = Inf
        else
        # Standard case: profitable with some drawdown
            calmar_ratio = geo_profit / max_drawdown_pct
        end

        
        # 5. Calculate Sharpe & Sortino Ratios
        # We use log returns of the wealth series for stability
        # Filter for wealth > 0 to avoid log(0)
        valid_wealth = wealth_series[wealth_series .> 0]
        log_returns = diff(log.(valid_wealth))
        
        if isempty(log_returns)
           (sharpe_ratio = 0.0, sortino_ratio = 0.0)
        else
            avg_return = mean(log_returns)
            std_dev = std(log_returns)
            
            negative_log_returns = filter(r -> r < 0, log_returns)
            downside_dev = isempty(negative_log_returns) ? 1.0 : std(negative_log_returns, mean=0)

            sharpe_ratio = std_dev > 0 ? avg_return / std_dev : 0.0
            sortino_ratio = downside_dev > 0 ? avg_return / downside_dev : 0.0
        end

        return (
            final_wealth = final_wealth,
            total_pnl_additive = total_pnl_additive,
            roi = roi,
            win_rate = win_rate,
            max_drawdown_pct = max_drawdown_pct,
            calmar_ratio = calmar_ratio,
            sharpe_ratio = sharpe_ratio,
            sortino_ratio = sortino_ratio,
            bet_count = bet_count
        )
    end
    
    return report
end

# --- EXAMPLE 1: Group by Market (as before) ---

# Run the analysis
report_by_market = analyze_strategy_performance(
    master_df, 
    grouping_cols=[:market]
)

best_wealth_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.final_wealth), :]
end;

sort(best_wealth_strategies, :final_wealth, rev=true)

best_sortino_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.sortino_ratio), :]
end;

sort(best_sortino_strategies, :final_wealth, rev=true)


best_calmar_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.calmar_ratio), :]
end;

sort(best_calmar_strategies, :final_wealth, rev=true)


best_sharpe_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.sharpe_ratio), :]
end;

sort(best_sharpe_strategies, :final_wealth, rev=true)



#####

report_by_market_tenth_kelly = analyze_strategy_performance(
    master_df, 
    grouping_cols=[:market],
    kelly_fraction=0.15  
)

best_wealth_strategies = combine(groupby(report_by_market_tenth_kelly, :market)) do g
    g[argmax(g.final_wealth), :]
end;

sort(best_wealth_strategies, :final_wealth, rev=true)

best_sortino_strategies = combine(groupby(report_by_market_tenth_kelly, :market)) do g
    g[argmax(g.sortino_ratio), :]
end;

sort(best_sortino_strategies, :final_wealth, rev=true)


best_calmar_strategies = combine(groupby(report_by_market_tenth_kelly, :market)) do g
    g[argmax(g.calmar_ratio), :]
end;

sort(best_calmar_strategies, :final_wealth, rev=true)


best_sharpe_strategies = combine(groupby(report_by_market_tenth_kelly, :market)) do g
    g[argmax(g.sharpe_ratio), :]
end;

sort(best_sharpe_strategies, :final_wealth, rev=true)

####

report_by_market_tenth_kelly = analyze_strategy_performance(
    master_df, 
    grouping_cols=[:tournament_slug, :market],
    kelly_fraction=0.3  # <-- HERE IS THE MAGIC
)



best_wealth_strategies = combine(groupby(report_by_market_tenth_kelly, [:tournament_slug, :market])) do g
    g[argmax(g.final_wealth), :]
end;

sort(best_wealth_strategies, :final_wealth, rev=true)

best_sortino_strategies = combine(groupby(report_by_market_tenth_kelly, [:tournament_slug,:market])) do g
    g[argmax(g.sortino_ratio), :]
end;

sort(best_sortino_strategies, :final_wealth, rev=true)


best_calmar_strategies = combine(groupby(report_by_market_tenth_kelly, [:tournament_slug, :market])) do g
    g[argmax(g.calmar_ratio), :]
end;

sort(best_calmar_strategies, :final_wealth, rev=true)


best_sharpe_strategies = combine(groupby(report_by_market_tenth_kelly, [:tournament_slug, :market])) do g
    g[argmax(g.sharpe_ratio), :]
end;

sort(best_sharpe_strategies, :final_wealth, rev=true)



# --- EXAMPLE 2: Group by Tournament AND Market ---


best_wealth_strategies = combine(groupby(report_by_market, [:tournament_slug, :market])) do g
    g[argmax(g.final_wealth), :]
end;

sort(best_wealth_strategies, :final_wealth, rev=true)

best_sortino_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.sortino_ratio), :]
end;

sort(best_sortino_strategies, :final_wealth, rev=true)


best_calmar_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.calmar_ratio), :]
end;

sort(best_calmar_strategies, :final_wealth, rev=true)


best_sharpe_strategies = combine(groupby(report_by_market, :market)) do g
    g[argmax(g.sharpe_ratio), :]
end;

sort(best_sharpe_strategies, :final_wealth, rev=true)






# Run the analysis
report_by_tournament = analyze_strategy_performance(
    master_df, 
    grouping_cols=[:tournament_slug, :market]
)

# Find the best strategy based on CALMAR ratio
best_calmar_strategies = combine(groupby(report_by_tournament, [:tournament_slug, :market])) do g
    g[argmax(g.calmar_ratio), :]
end

println("\n--- Best Strategies by Tournament/Market (via Calmar Ratio) ---")
# Filter for only profitable ones to see what works
println(filter(row -> row.calmar_ratio > 0, best_calmar_strategies))



####
# 1. Define the grid of fractions you want to test
fraction_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# 2. Initialize a place to store the "best strategy" from each run
all_best_strategies = []

# 3. Loop over the grid
for frac in 0.02:0.02:0.9
    println("--- Running backtest for kelly_fraction = $frac ---")
    
    # 3a. Run your full analysis
    report = analyze_strategy_performance(
        master_df, 
        grouping_cols=[:market],
        kelly_fraction=frac
    )
    
    # 3b. Find the best Calmar strategy for each market *at this fraction*
    best_calmar_for_this_fraction = combine(groupby(report, :market)) do g
        g[argmax(g.calmar_ratio), :]
    end
    
    # 3c. Add a column so we know which fraction this was
    best_calmar_for_this_fraction.kelly_fraction .= frac
    
    push!(all_best_strategies, best_calmar_for_this_fraction)
end

# 4. Combine all results into one big DataFrame
final_report = vcat(all_best_strategies...)

# 5. NOW, find the best of the best
# Sort by Calmar to find the single best (market, threshold, fraction) combo
sort!(final_report, :final_wealth, rev=true)



bb = combine(groupby(final_report, :market)) do g
    g[argmax(g.calmar_ratio), :]
end;

sort(bb, :final_wealth, rev=true)


bb = combine(groupby(final_report, :market)) do g
    g[argmax(g.final_wealth), :]
end;

sort(bb, :final_wealth, rev=true)





####
unique_matches = unique(master_df[:, [:match_id, :match_date]])
println("Total unique matches found: $(nrow(unique_matches))")

# Find the 70% split index
n_total_matches = nrow(unique_matches)
n_train_matches = round(Int, n_total_matches * 0.5)

# Get the set of match_ids for the training data
# A 'Set' is much faster for filtering than an array
train_ids = Set(unique_matches[1:n_train_matches, :match_id])


# Filter the main DataFrame based on the Set of IDs
train_df = filter(row -> row.match_id in train_ids, master_df)
test_df = filter(row -> !(row.match_id in train_ids), master_df)

# --- Verification ---
println("Total rows in master_df: $(nrow(master_df))")
println("Rows in train_df (70%): $(nrow(train_df))")
println("Rows in test_df (30%): $(nrow(test_df))")
println("Total rows accounted for: $(nrow(train_df) + nrow(test_df))")


# 1. Define the grid of fractions you want to test
fraction_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
all_best_strategies = []

# 2. Loop over the grid
for frac in fraction_grid
    
    # Run analysis ONLY on train_df
    report = analyze_strategy_performance(
        train_df, 
        grouping_cols=[:market],
        kelly_fraction=frac
    )
    
    # Find the best Calmar strategy for each market
    best_calmar_for_this_fraction = combine(groupby(report, :market)) do g
        g[argmax(g.calmar_ratio), :]
    end
    
    best_calmar_for_this_fraction.kelly_fraction .= frac
    push!(all_best_strategies, best_calmar_for_this_fraction)
end

# 3. Combine all results
final_report = vcat(all_best_strategies...)

# 4. Get our final "Rule Book"
# We find the best (threshold, fraction) pair for each market,
# based *only* on the training data.
strategy_rules = combine(groupby(final_report, :market)) do g
    g[argmax(g.calmar_ratio), :]
end

# 5. Filter for only the profitable strategies
final_portfolio_rules = filter(row -> row.calmar_ratio > 0 && row.bet_count > 0, strategy_rules)

println("--- Final Strategy Rules (found from 70% train data) ---")
println(final_portfolio_rules[:, [:market, :kelly_threshold, :kelly_fraction, :calmar_ratio]])


"""
Runs a full portfolio backtest on a new dataset,
using a pre-defined set of strategy rules.
"""
function run_portfolio_backtest(
    test_data::DataFrame, 
    rules::DataFrame;
    conflict_groups = [
        [:home, :draw, :away],
        [:over_05, :under_05],
        [:over_15, :under_15],
        [:over_25, :under_25],
        [:over_35, :under_35],
        [:over_45, :under_45],
        [:btts_yes, :btts_no]
    ]
)
    
    # Create a Dict for fast rule lookup
    # e.g., :home => (threshold=0.25, fraction=0.02)
    rule_dict = Dict(
        row.market => (
            threshold = row.kelly_threshold, 
            fraction = row.kelly_fraction
        )
        for row in eachrow(rules)
    )

    wealth = 1.0
    wealth_series = [1.0]
    
    # Loop over each match *day* (or just match_id if sorted)
    # We must group by match_id to handle conflicts
    for match_group in groupby(test_data, :match_id)
        
        total_stake_for_this_match = 0.0
        bets_to_make = [] # Store potential bets

        # 1. Check all rows for this match (home, away, etc.)
        for row in eachrow(match_group)
            market = row.market
            
            # Check if we have a rule for this market
            if haskey(rule_dict, market)
                rule = rule_dict[market]
                
                # Check if the stake beats our threshold
                if row.positive_median_thresh > rule.threshold
                    push!(bets_to_make, (
                        market = market,
                        stake = row.positive_median_thresh,
                        fraction = rule.fraction,
                        odds = row.odds,
                        winner = row.winner
                    ))
                end
            end
        end
        
        if isempty(bets_to_make)
            push!(wealth_series, wealth) # Bankroll didn't change
            continue
        end

        # 2. Resolve Conflicts
        final_bets = []
        for group in conflict_groups
            conflicting_bets = filter(b -> b.market in group, bets_to_make)
            
            if length(conflicting_bets) > 1
                # Rule: Pick the one with the highest *raw stake*
                best_bet = argmax(b -> b.stake, conflicting_bets)
                push!(final_bets, best_bet)
            elseif length(conflicting_bets) == 1
                push!(final_bets, conflicting_bets[1])
            end
        end
        
        # Add non-conflicting bets (e.g., if a group has btts_yes but not btts_no)
        non_conflicting = filter(
            b -> !any(g -> b.market in g, conflict_groups), 
            bets_to_make
        )
        append!(final_bets, non_conflicting)

        # 3. Place the final, non-conflicting bets
        for bet in final_bets
            # Stake is the median_stake * optimal_fraction
            stake = bet.stake * bet.fraction
            stake_amount = wealth * stake
            
            if stake_amount > wealth # Safety check
                stake_amount = wealth
            end

            if bet.winner
                wealth += stake_amount * (bet.odds - 1.0)
            else
                wealth -= stake_amount
            end
            
            if wealth <= 0.0
                wealth = 0.0
                break # Bankroll busted
            end
        end
        
        push!(wealth_series, wealth)
        if wealth == 0.0
             # Fill rest of series with 0
            append!(wealth_series, zeros(nrow(test_data) - length(wealth_series) + 1))
            break
        end
    end
    
    # 4. Calculate final metrics for the *test* period
    final_wealth = last(wealth_series)
    max_dd = calculate_max_drawdown(wealth_series)
    geo_profit = final_wealth - 1.0
    calmar_ratio = max_dd > 0 ? geo_profit / max_dd : (geo_profit > 0 ? Inf : 0.0)
    
    return (
        final_wealth = final_wealth,
        max_drawdown_pct = max_dd,
        calmar_ratio = calmar_ratio,
        wealth_series = wealth_series
    )
end



out_of_sample_results = run_portfolio_backtest(
    test_df, 
    final_portfolio_rules
)

println("--- Out-of-Sample Test Results (on 30% unseen data) ---")
println("Final Wealth: $(out_of_sample_results.final_wealth)")
println("Max Drawdown: $(round(out_of_sample_results.max_drawdown_pct * 100, digits=2))%")
println("Calmar Ratio: $(out_of_sample_results.calmar_ratio)")



println("--- Starting Out-of-Sample Test (Per-Market) ---")

# We will store the 8 independent backtest results here
all_market_test_results = []

for rule in eachrow(final_portfolio_rules)
    market = rule.market
    threshold = rule.kelly_threshold
    fraction = rule.kelly_fraction

    println("Testing :$(market) (thresh=$(threshold), frac=$(fraction))...")

    # 1. Filter the test_df for *only* the rows that match this one strategy
    market_test_data = filter(
        row -> row.market == market && row.kelly_threshold == threshold, 
        test_df
    )

    if isempty(market_test_data)
        println("  -> No data found for this rule in the test set. Skipping.")
        continue
    end

    # 2. Run the backtest for *only* this market's data
    #    We pass its specific, optimal kelly_fraction
    market_report = analyze_strategy_performance(
        market_test_data, 
        grouping_cols=[:market],
        kelly_fraction=fraction
    )
    
    # 3. Add the fraction back in for clarity
    market_report.kelly_fraction .= fraction
    
    push!(all_market_test_results, market_report)
end

println("...Testing complete.")

# 4. Combine all 8 results into one final report
if isempty(all_market_test_results)
    println("No results found. Did the test set have any matching data?")
else
    out_of_sample_report = vcat(all_market_test_results...)

    # 5. Sort by Calmar Ratio to see what *actually* worked
    sort!(out_of_sample_report, :calmar_ratio, rev=true)
    
    println(out_of_sample_report)
end



using Parsers
using ParseUtil

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


# Make a deep copy to avoid changing your original data
ds_odds_initial = deepcopy(ds.odds)

# Convert the initial fractional string to a decimal
ds_odds_initial.initial_decimal = parse_fractional_to_decimal.(ds_odds_initial.initial_fractional_value)

# Filter out rows where parsing failed or odds were 0
filter!(row -> row.initial_decimal > 1.0, ds_odds_initial)

# --- THIS IS THE "TRICK" ---
# Overwrite the `decimal_odds` column with our new initial odds.
# Your `get_market` function will now read this column,
# thinking it's the final odds.
ds_odds_initial.decimal_odds = ds_odds_initial.initial_decimal

println("Original odds count: $(nrow(ds.odds))")
println("New initial odds count: $(nrow(ds_odds_initial))")

# 1. Create your master DataFrame using the MODIFIED ds
#    (I'm assuming 'ds' is a container object you pass to the function)

ds = BayesianFootball.Data.DataStore(
    df,
    ds_odds_initial,
    data_store.incidents
)

df_all_matches_initial = run_full_oos_analysis(
    all_oos_results, 
    model, 
    predict_config, 
    ds, 
    threshold_range=0.0:0.02:0.9 # Or your original range
)

# 2. Add your match info
master_df_initial = leftjoin(df_all_matches_initial, match_info, on = :match_id)

# 3. Create Train/Test Split
sort!(master_df_initial, :match_date)
unique_matches_initial = unique(master_df_initial[:, [:match_id, :match_date]])
n_total_initial = nrow(unique_matches_initial)
n_train_initial = round(Int, n_total_initial * 0.7)
train_ids_initial = Set(unique_matches_initial[1:n_train_initial, :match_id])

train_df_initial = filter(row -> row.match_id in train_ids_initial, master_df_initial)
test_df_initial = filter(row -> !(row.match_id in train_ids_initial), master_df_initial)

# 4. Find Optimal Rules (on train data)
# ... (run your grid search on train_df_initial) ...
# ... (this finds your new 'final_portfolio_rules_initial') ...

fraction_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
all_best_strategies = []

# 2. Loop over the grid
for frac in fraction_grid
    
    # Run analysis ONLY on train_df
    report = analyze_strategy_performance(
        train_df_initial, 
        grouping_cols=[:market],
        kelly_fraction=frac
    )
    
    # Find the best Calmar strategy for each market
    best_calmar_for_this_fraction = combine(groupby(report, :market)) do g
        g[argmax(g.calmar_ratio), :]
    end
    
    best_calmar_for_this_fraction.kelly_fraction .= frac
    push!(all_best_strategies, best_calmar_for_this_fraction)
end

# 3. Combine all results
final_report = vcat(all_best_strategies...)

# 4. Get our final "Rule Book"
# We find the best (threshold, fraction) pair for each market,
# based *only* on the training data.
strategy_rules = combine(groupby(final_report, :market)) do g
    g[argmax(g.calmar_ratio), :]
end

# 5. Filter for only the profitable strategies
final_portfolio_rules_initial = filter(row -> row.calmar_ratio > 0 && row.bet_count > 0, strategy_rules)

println("--- Final Strategy Rules (found from 70% train data) ---")
println(final_portfolio_rules_initial[:, [:market, :kelly_threshold, :kelly_fraction, :calmar_ratio]])


# 5. Run the Out-of-Sample Backtest (on test data)
out_of_sample_results_initial = run_portfolio_backtest(
    test_df_initial, 
    final_portfolio_rules_initial
)

# 6. CHECK THE RESULTS
println("--- FINAL BACKTEST (AGAINST OPENING LINE) ---")
println(out_of_sample_results_initial)


println("--- Out-of-Sample Test Results (on 30% unseen data) ---")
println("Final Wealth: $(out_of_sample_results_initial.final_wealth)")
println("Max Drawdown: $(round(out_of_sample_results_initial.max_drawdown_pct * 100, digits=2))%")
println("Calmar Ratio: $(out_of_sample_results_initial.calmar_ratio)")

println("--- Starting Out-of-Sample Test (Per-Market) ---")

# We will store the 8 independent backtest results here
all_market_test_results = []

for rule in eachrow(final_portfolio_rules_initial)
    market = rule.market
    threshold = rule.kelly_threshold
    fraction = rule.kelly_fraction

    println("Testing :$(market) (thresh=$(threshold), frac=$(fraction))...")

    # 1. Filter the test_df for *only* the rows that match this one strategy
    market_test_data = filter(
        row -> row.market == market && row.kelly_threshold == threshold, 
        test_df
    )

    if isempty(market_test_data)
        println("  -> No data found for this rule in the test set. Skipping.")
        continue
    end

    # 2. Run the backtest for *only* this market's data
    #    We pass its specific, optimal kelly_fraction
    market_report = analyze_strategy_performance(
        market_test_data, 
        grouping_cols=[:market],
        kelly_fraction=fraction
    )
    
    # 3. Add the fraction back in for clarity
    market_report.kelly_fraction .= fraction
    
    push!(all_market_test_results, market_report)
end

println("...Testing complete.")

# 4. Combine all 8 results into one final report
if isempty(all_market_test_results)
    println("No results found. Did the test set have any matching data?")
else
    out_of_sample_report = vcat(all_market_test_results...)

    # 5. Sort by Calmar Ratio to see what *actually* worked
    sort!(out_of_sample_report, :calmar_ratio, rev=true)
    
    println(out_of_sample_report)
end


"""
Runs a full portfolio backtest on a new dataset,
using a pre-defined set of strategy rules.
Can optionally override the fractions in the rules.
"""
function run_portfolio_backtest(
    test_data::DataFrame, 
    rules::DataFrame;
    global_kelly_fraction::Union{Nothing, Number} = nothing, # <-- NEW
    conflict_groups = [
        [:home, :draw, :away],
        [:over_05, :under_05],
        [:over_15, :under_15],
        [:over_25, :under_25],
        [:over_35, :under_35],
        [:over_45, :under_45],
        [:btts_yes, :btts_no]
    ]
)
    
    # Create a Dict for fast rule lookup
    rule_dict = Dict(
        row.market => (
            threshold = row.kelly_threshold, 
            fraction = row.kelly_fraction # This is the "default" fraction
        )
        for row in eachrow(rules)
    )

    wealth = 1.0
    wealth_series = [1.0]
    
    for match_group in groupby(test_data, :match_id)
        
        bets_to_make = [] # Store potential bets

        # 1. Check all rows for this match
        for row in eachrow(match_group)
            market = row.market
            
            if haskey(rule_dict, market)
                rule = rule_dict[market]
                
                if row.positive_median_thresh > rule.threshold
                    
                    # --- NEW LOGIC ---
                    # Use the rule's fraction, *unless* a global one is provided
                    final_fraction = isnothing(global_kelly_fraction) ? rule.fraction : global_kelly_fraction
                    # --- END NEW LOGIC ---

                    push!(bets_to_make, (
                        market = market,
                        stake = row.positive_median_thresh,
                        fraction = final_fraction, # <-- Use the new variable
                        odds = row.odds,
                        winner = row.winner
                    ))
                end
            end
        end
        
        if isempty(bets_to_make)
            push!(wealth_series, wealth) 
            continue
        end

        # 2. Resolve Conflicts (Code is identical)
        final_bets = []
        for group in conflict_groups
            conflicting_bets = filter(b -> b.market in group, bets_to_make)
            
            if length(conflicting_bets) > 1
                best_bet = argmax(b -> b.stake, conflicting_bets)
                push!(final_bets, best_bet)
            elseif length(conflicting_bets) == 1
                push!(final_bets, conflicting_bets[1])
            end
        end
        
        non_conflicting = filter(
            b -> !any(g -> b.market in g, conflict_groups), 
            bets_to_make
        )
        append!(final_bets, non_conflicting)

        # 3. Place the final, non-conflicting bets (Code is identical)
        for bet in final_bets
            stake = bet.stake * bet.fraction
            stake_amount = wealth * stake
            
            if stake_amount > wealth 
                stake_amount = wealth
            end

            if bet.winner
                wealth += stake_amount * (bet.odds - 1.0)
            else
                wealth -= stake_amount
            end
            
            if wealth <= 0.0
                wealth = 0.0
                break 
            end
        end
        
        push!(wealth_series, wealth)
        if wealth == 0.0
            append!(wealth_series, zeros(nrow(test_data) - length(wealth_series) + 1))
            break
        end
    end
    
    # 4. Calculate final metrics
    final_wealth = last(wealth_series)
    max_dd = calculate_max_drawdown(wealth_series)
    geo_profit = final_wealth - 1.0
    calmar_ratio = max_dd > 0 ? geo_profit / max_dd : (geo_profit > 0 ? Inf : 0.0)
    
    return (
        final_wealth = final_wealth,
        max_drawdown_pct = max_dd,
        calmar_ratio = calmar_ratio,
        wealth_series = wealth_series
    )
end


# Run the backtest on the 30% test_df,
# using the 8 rules, but overriding with a safe global fraction
sane_results = run_portfolio_backtest(
    test_df_initial, 
    final_portfolio_rules_initial,
    global_kelly_fraction = 0.2  # <-- Tame the bull!
)

println("--- FINAL SANE Backtest (5% Global Fraction) ---")
println("Final Wealth: $(sane_results.final_wealth)")
println("Max Drawdown: $(round(sane_results.max_drawdown_pct * 100, digits=2))%")
println("Calmar Ratio: $(sane_results.calmar_ratio)")





"""
The Real Story: Signal vs. Noise

This new table is your "smoking gun." You've successfully separated the "signal" (a real edge) from the "noise" (overfit, fake edges).

    The Signal (Your Edge):

        Row 1: :over_15 -> This is a genuinely robust, profitable strategy.

        final_wealth = 1.58 (+58% profit)

        calmar_ratio = 2.14 (This is excellent. Your profit was 2x your max drawdown).

        win_rate = 0.92

        This is the "space" you were looking for. It's real.

    The Noise (The Overfit Losers):

        Row 6: :home -> final_wealth = 0.45 (-55% loss)

        Row 5: :over_25 -> final_wealth = 0.44 (-56% loss)

        Row 7: :over_45 -> final_wealth = 0.73 (-27% loss)

        Row 8: :over_35 -> final_wealth = 0.72 (-28% loss)

Your portfolio failed because the 56% loss from :over_25 and the 55% loss from :home completely wiped out the 58% gain from :over_15.

You didn't fail. You just proved that your final portfolio should only contain one strategy: the :over_15 one.

🚀 The Path Forward: Your "Stability" Problem

Your next steps are exactly what you outlined. You need to confirm why :over_15 worked and the others failed. Your ideas are the correct professional workflow.

Here is how you execute your plan.

1. Your Idea: "Analyze the differences between the initial odds v the game line odds"

    What this is: Closing Line Value (CLV) analysis.

    How to do it: This is a fast analysis. You don't need a full backtest. Create a simple report:

        Filter your master_df for only the rows matching your one winning strategy: (market == :over_15 && kelly_threshold == 0.86).

        This gives you all the matches (in both train and test) where this strategy would have bet.

        Compare the initial_decimal (which you'll need to parse) to the decimal_odds (the closing line).

        Ask the question: In what percentage of these bets did the final odds (decimal_odds) drop?

    What it proves: If the final odds consistently dropped (e.g., 60-70% of the time), you have positive CLV. This proves your model's edge is real and the "smart money" (the market) agrees with your model after you've placed your bet.

2. Your Idea: "need an online learning for the paramters of the kelly... a rolling one"

    What this is: A Walk-Forward Analysis. This is the ultimate test of stability.

    How to do it: This is more computationally expensive, but you can speed it up. You don't need to re-run your entire 15-market grid search. You can just test your one winning strategy:

        Window 1 (Weeks 1-10): Run your backtest (calculate_geometric_wealth) on this window for the :over_15 strategy. Is it profitable?

        Window 2 (Weeks 2-11): Slide the window forward. Run the backtest again.

        Window 3 (Weeks 3-12): Slide again.

    What it proves: This gives you a time series of your strategy's performance. You can plot the rolling 10-week Calmar Ratio. This directly answers your "stability" question. Does the edge persist over time, or did your 30% test set just happen to be a lucky 10-week run?

3. Your Idea: "need to find a way to measue the it more stable" (The Root Cause)

    What this is: This is the core problem of non-stationarity. Team form changes.

    The Solution: This is where you connect back to our earlier discussion. The reason your :home and :over_25 strategies failed so badly is that they were "overfit" to static, average strengths. The reason your :over_15 strategy might be working is that "high-scoring teams tend to stay high-scoring" is a more stable, simpler pattern.

    The Next Model: To make all your markets more stable, you must move to a Dynamic Model (AR(1) / State-Space). This model is designed to handle changing form.

What I Recommend You Do Next

You have a winning horse (:over_15). Now you need to make sure it's not a fluke.

    Today (Fast): Run the CLV analysis on your :over_15 strategy. This gives you a fast "yes/no" on whether your model's predictions are fundamentally sound.

    This Week (Medium): Run the Walk-Forward Analysis (rolling window) on your :over_15 strategy. This gives you the "stability plot" and tells you if the profit is consistent.

    Long-Term (The "Real" Fix): If the strategy still looks good, your next major project is to build the Dynamic (AR(1)) Model. This will likely improve the stability of all your markets, potentially turning your other "losing" strategies into "winners."


"""



"""
Where to Look: The "Lost" Textbooks

The problems you've found (parameter instability, robust backtesting, Kelly optimization) are the entire focus of quantitative trading.

Here are the people, concepts, and books that actually solve this.

1. The Core Problem: Backtest Overfitting & Stability

This is exactly what you described as "online learning" to find "stable" parameters.

    Key Concept: Walk-Forward Optimization (or Rolling Window Analysis). This is the direct solution to your "single split" problem. You've already described it perfectly: you optimize on a 1-year window, bet on the next month, then slide the window forward and re-optimize. This tests for parameter stability.

    The "Bible": "Advances in Financial Machine Learning" by Marcos López de Prado. This book is the modern standard. It dedicates entire chapters to why simple backtests (like the 70/30 split, and the one in your handout) are statistically guaranteed to fail and produce overfit, false strategies. He provides the code and theory for robust methods, including walk-forward analysis.

    The Practical Guide: "Quantitative Trading" by Ernie Chan. This is a more accessible, practical guide on how to build and, most importantly, validate a strategy. He lives and breathes the "out-of-sample" validation that you've just discovered.

2. The Kelly Problem: "Optimizing the Fractions"

Yes, this is a huge topic. You're right that argmax is naive.

    Key Concept: "The Metamodel." Your realization that the "best" (threshold, fraction) pair is itself a parameter to be optimized is the key.

    The "Bible": "The Mathematics of Money Management" by Ralph Vince. This is the book on Kelly optimization (he calls it the "Optimal f"). He discusses how to apply it, the "risk of ruin" problem, and how to "fractionalize" it, which is exactly what you're doing.

    The Problem You Found: You've discovered that the Kelly parameters are "non-stationary" (they change over time). This is why your "rolling online" idea is the correct one. You're not looking for one set of parameters; you're trying to build a system that adapts to the current optimal parameters.

3. The Model Problem: "Is it solved by particle filters?"

    Yes, this is one valid path. What you described ("posteriors as priors") is a Sequential Monte Carlo (SMC) / Particle Filter approach. It's one way to solve the "static vs. dynamic" model problem.

    The other (more common) way is the State-Space Model (the AR(1) / Random Walk) we discussed.

    Key Search Term: "Dynamic Linear Models (DLMs)" or "State-Space Models." This is the academic field for the AR(1) approach. A search for "Bayesian Dynamic Linear Models for sports" will yield much more advanced papers than the static ones you've seen.

Your Next Step (My Recommendation)

You have two problems to solve: a weak model (static) and a weak metamodel (brittle 70/30 split).

Do not try to solve both at once.

    First, prove you have a real edge.

        Do the "Closing Line Value (CLV)" analysis. This is fast and simple.

        It completely separates the model from the Kelly optimization.

        It answers one question: "Is my static Dixon-Coles model (betting the opening line) smarter than the market?"

        If Yes, you have a provable edge. Now you can spend the time building the "Online Bayesian Optimizer" to exploit it.

        If No (which is my suspicion), then no amount of Kelly optimization will ever make you money. This proves you must build a Dynamic (AR(1)) Model first.

Your framework is sound. Your instincts are 100% correct. You just need to look in the quant finance world for the "metamodel" answers and focus on building a dynamic model to find a real edge.

"""
