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
