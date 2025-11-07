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


##############################
# ---  predict 
##############################
using Statistics, Distributions

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

# get an id 

match_id = rand(keys(all_oos_results))

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, all_oos_results[match_id]...)

model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)

market_odds = subset( ds.odds, :match_id => ByRow(isequal(match_id)))

subset( ds.matches, :match_id => ByRow(isequal(match_id)))



##############################
# --- Kelly 
##############################

"""
Calculates the optimal Kelly criterion fraction to bet.

# Arguments
- `decimal_odds`: The decimal odds offered by the bookmaker (e.g., 2.5, 3.0).
- `probability`: Your estimated true probability of the event occurring (e.g., 0.45).

# Returns
- `f`: The fraction of your bankroll to bet (from 0.0 to 1.0).
  A value of 0.0 means the bet has no value (p < 1/decimal_odds).
"""
function kelly_fraction(decimal_odds, probability)
    # 1. Get p (probability of winning) and q (probability of losing)
    p = probability
    q = 1.0 - p

    # 2. Get b (net odds) from the decimal odds
    # b = profit from a 1-unit bet
    b = decimal_odds - 1.0

    # 3. Handle invalid odds
    if b <= 0.0
        return 0.0  # Bet has no profit potential or is invalid
    end

    # 4. Calculate the Kelly fraction: f = p - (q / b)
    f = p - (q / b)

    # 5. A negative fraction means "don't bet"
    return max(0.0, f)
end

function get_confidence(kelly_dist; cutoff = 0.0)
    return mean(kelly_dist .> cutoff)
end

h = kelly_fraction.(match_odds.home, match_predict.home);
a = kelly_fraction.(match_odds.away, match_predict.away);
d = kelly_fraction.(match_odds.draw, match_predict.draw);


kelly_value(a, 0.0)
kelly_value(h, 0.0)
kelly_value(d, 0.0)


a_positive = a[a .> 0.0]
recommended_stake_away = median(a_positive)

using StatsPlots

density(a, label="away")
density!(h, label="home")
density!(d, label="draw")

mean(h)
median(h)

kelly_fraction(match_odds.away, median(match_predict.away))
kelly_fraction(match_odds.away, mean(match_predict.away))


####

function get_confidence(kelly_dist; cutoff = 0.0)
    return mean(kelly_dist .> cutoff)
end

function get_positive_median(kelly_dist)
    positive_stakes = kelly_dist[kelly_dist .> 0.0]
    return isempty(positive_stakes) ? 0.0 : median(positive_stakes)
end


# v1 - had been improved below
# 2. Define a function to process all matches
function generate_backtest_data(model, all_oos_results, predict_config, ds, model_name::String)
    
    # Use a Vector of NamedTuples for performance.
    # It's much faster than pushing to a DataFrame.
    bet_data_rows = []
    
    println("Processing $(length(all_oos_results)) matches for $model_name...")
    
    for match_id in keys(all_oos_results)
        
        # A. Get data for this match
        match_predict = BayesianFootball.Predictions.predict_market(
            model, predict_config, all_oos_results[match_id]...
        )
        match_odds = BayesianFootball.Predictions.get_market(
            match_id, predict_config, ds.odds
        )
        match_results = BayesianFootball.Predictions.get_market_results(
            match_id, predict_config, ds.odds
        )
        
        # B. Iterate over every market for this match
        for market_key in keys(match_odds)
            
            p_dist = match_predict[market_key]
            decimal_odds = match_odds[market_key]
            result = match_results[market_key]
            
            # C. Skip if we don't have odds for this market
            if isnothing(decimal_odds) || ismissing(decimal_odds) || decimal_odds <= 1.0
                continue
            end
            
            # D. This is the core: get the Kelly distribution
            f_dist = kelly_fraction.(decimal_odds, p_dist)
            
            # E. Pre-calculate all our "ingredients"
            row = (
                match_id = match_id,
                model_name = model_name,
                market = market_key,
                decimal_odds = decimal_odds,
                was_winner = result,
                
                # Model's raw probability
                p_median = median(p_dist),
                
                # Kelly distribution summaries (our "decision" stats)
                conf_0 = get_confidence(f_dist, cutoff=0.0),
                stake_median = get_positive_median(f_dist),
                
                # Optional: higher-confidence checks
                conf_1pct = get_confidence(f_dist, cutoff=0.01)
            )
            
            push!(bet_data_rows, row)
        end
    end
    
    # F. Convert to DataFrame and join with match info
    # This is the "master" DataFrame
    bet_df = DataFrame(bet_data_rows)
    
    # Join with ds.matches to get league, date, etc.
    match_info = unique(ds.matches[:, [:match_id, :tournament_slug, :season, :match_date]], :match_id)
    
    master_df = leftjoin(bet_df, match_info, on = :match_id)
    
    return master_df
end


# v2 to avoid missing odds 

function generate_backtest_data(model, all_oos_results, predict_config, ds, model_name::String)
    
    bet_data_rows = []
    
    println("Processing $(length(all_oos_results)) matches for $model_name...")
    
    for match_id in keys(all_oos_results)
        
        # We must declare these vars outside the try block to use them after
        local match_predict, match_odds, match_results
        
        try
            # A. Get data for this match
            match_predict = BayesianFootball.Predictions.predict_market(
                model, predict_config, all_oos_results[match_id]...
            )
            match_odds = BayesianFootball.Predictions.get_market(
                match_id, predict_config, ds.odds
            )
            match_results = BayesianFootball.Predictions.get_market_results(
                match_id, predict_config, ds.odds
            )
        catch e
            if e isa KeyError
                # This match is missing data (e.g., the "Over" key)
                # Print a warning and skip this match_id
                println("SKIPPING match_id $match_id: Could not find key $(e.key).")
                continue # This jumps to the next iteration of the outer loop
            else
                # It's a different error, so we should stop and see it
                rethrow(e)
            end
        end
        
        # B. Iterate over every market for this match
        # This code is only reached if the 'try' block succeeded
        for market_key in keys(match_odds)
            
            p_dist = match_predict[market_key]
            decimal_odds = match_odds[market_key]
            result = match_results[market_key]
            
            # C. Skip if we don't have odds for this market
            if isnothing(decimal_odds) || ismissing(decimal_odds) || decimal_odds <= 1.0
                continue
            end
            
            # D. This is the core: get the Kelly distribution
            f_dist = kelly_fraction.(decimal_odds, p_dist)
            
            # E. Pre-calculate all our "ingredients"
            row = (
                match_id = match_id,
                model_name = model_name,
                market = market_key,
                decimal_odds = decimal_odds,
                was_winner = result,
                
                # Model's raw probability
                p_median = median(p_dist),
                
                # Kelly distribution summaries (our "decision" stats)
                conf_0 = get_confidence(f_dist, cutoff=0.0),
                stake_median = get_positive_median(f_dist),
                
                # Optional: higher-confidence checks
                conf_1pct = get_confidence(f_dist, cutoff=0.01)
            )
            
            push!(bet_data_rows, row)
        end
    end
    
    # F. Convert to DataFrame and join with match info
    bet_df = DataFrame(bet_data_rows)
    
    # Join with ds.matches to get league, date, etc.
    match_info = unique(ds.matches[:, [:match_id, :tournament_slug, :season, :match_date]], :match_id)
    
    master_df = leftjoin(bet_df, match_info, on = :match_id)
    
    return master_df
end


master_df_v1 = generate_backtest_data(model, all_oos_results, predict_config, ds, "basic_poisson")


# part 2 
using DataFramesMeta, Statistics

# This struct is a good practice for passing many parameters
struct Strategy
    conf_thresh::Float64
    stake_thresh::Float64
    stake_type::Symbol  # :stake_median
    model_name::String
end

function analyze_strategy(df::AbstractDataFrame, strategy::Strategy)
    
    # 1. Filter for the bets this strategy would have placed
   bets_placed = @subset(df,
    :model_name .== strategy.model_name,           # Add a dot
    :conf_0 .> strategy.conf_thresh,               # Add a dot
    $(strategy.stake_type) .> strategy.stake_thresh # Add $ and a dot
)
    
    if isempty(bets_placed)
        return (num_bets=0, roi=0.0, profit=0.0, win_rate=0.0, avg_odds=0.0)
    end

    # 2. Calculate profit and loss for those bets
    # We use the stake amount (e.g., :stake_median) to size the bet
    bets_placed.profit = @. ifelse(
        bets_placed.was_winner,
        bets_placed[!, strategy.stake_type] * (bets_placed.decimal_odds - 1),
        -bets_placed[!, strategy.stake_type]
    )
    
    # 3. Aggregate metrics
    total_staked = sum(bets_placed[!, strategy.stake_type])
    total_profit = sum(bets_placed.profit)
    roi = total_profit / total_staked
    
    return (
        num_bets = nrow(bets_placed),
        roi = roi,
        profit = total_profit,
        win_rate = mean(bets_placed.was_winner),
        avg_odds = mean(bets_placed.decimal_odds)
    )
end

dff = subset( master_df_v1, :market => ByRow(isequal(:home)))
mean(dff.was_winner)

# Assume `all_models_df` is loaded from Phase 1
all_models_df = master_df_v1

# s = Strategy(0.7, 0.005, :stake_median, "basic_poisson")
# metrics = analyze_strategy(all_models_df, s)

# --- Experiment 1: Find optimal confidence threshold for "Model_v1" ---
println("--- Model_v1 Threshold Test ---")
for conf in 0.0:0.02:0.99
    s = Strategy(conf, 0.005, :stake_median, "basic_poisson")
    metrics = analyze_strategy(all_models_df, s)
    
    println("Conf: $conf | Bets: $(metrics.num_bets) | ROI: $(round(metrics.roi, digits=3))")
end

# Output might be:
# Conf: 0.5 | Bets: 1205 | ROI: 0.051
# Conf: 0.55 | Bets: 980 | ROI: 0.062
# Conf: 0.60 | Bets: 750 | ROI: 0.075  <-- Looks promising

# --- Experiment 2: Compare models at the "optimal" threshold ---
println("\n--- Model Comparison at 60% Confidence ---")
s_v1 = Strategy(0.60, 0.005, :stake_median, "basic_poisson")
# s_v2 = Strategy(0.60, 0.005, :stake_median, "Model_v2")

metrics_v1 = analyze_strategy(all_models_df, s_v1)
# metrics_v2 = analyze_strategy(all_models_df, s_v2)

println("Model_v1 ROI: $(metrics_v1.roi) on $(metrics_v1.num_bets) bets")
# println("Model_v2 ROI: $(metrics_v2.roi) on $(metrics_v2.num_bets) bets")


# --- Experiment 3: Group by market (as you asked) ---
println("\n--- Market Breakdown for Model_v1 ---")
strategy_df = @subset(all_models_df,
    :model_name .== "basic_poisson",
    :conf_0 .> 0.60,
    :stake_median .> 0.005
)

# This is the power of DataFrames:
grouped_results = groupby(strategy_df, :market)

# We can now analyze each market just like we did the whole group
# (This is more complex, but `combine` is the function you'd use)
for (key, group) in pairs(grouped_results)
    s = Strategy(0.7, 0.005, :stake_median, "basic_poisson")
    # We pass the *sub-dataframe* for just that market
    metrics = analyze_strategy(group, s) 
    
    if metrics.roi > 0.0
        println("Market: $(key.market) | Bets: $(metrics.num_bets) | ROI: $(round(metrics.roi, digits=3))")
    end
end

# Output:
# Market: :home | Bets: 150 | ROI: 0.081
# Market: :away | Bets: 132 | ROI: 0.102
# Market: :over_25 | Bets: 95 | ROI: -0.050 <-- Clearly shows a weak market!

using DataFrames, DataFramesMeta, Statistics

# (Your existing analyze_strategy and Strategy struct are needed)

"""
Finds the optimal thresholds for a single market by grid searching.
"""
function find_best_market_strategy(market_df::AbstractDataFrame;
                                   model_name::String,
                                   min_bets::Int = 5, # Our key constraint
                                   conf_range = 0.05:0.05:0.95,
                                   stake_range = 0.00:0.005:0.20)
    
    best_roi = 0.0
    best_strategy = (conf=0.0, stake=0.0)
    best_metrics = (num_bets=0, roi=0.0)
    
    # Grid search
    for conf in conf_range
        for stake in stake_range
            
            s = Strategy(conf, stake, :stake_median, model_name)
            
            # Analyze this specific (conf, stake) combo
            metrics = analyze_strategy(market_df, s)
            
            # --- This is the core logic ---
            # Is this ROI better than our current best?
            # AND does it meet our minimum bet constraint?
            if (metrics.roi > best_roi) && (metrics.num_bets >= min_bets)
                best_roi = metrics.roi
                best_strategy = (conf=conf, stake=stake)
                best_metrics = metrics
            end
        end
    end
    
    # If no strategy was found (e.g., all ROIs were negative or 0 bets)
    # return "disabled" thresholds
    if best_roi == -Inf
        return (
            market = market_df.market[1],
            opt_conf_thresh = Inf, # "Disable" this market
            opt_stake_thresh = Inf,
            best_roi = 0.0,
            num_bets = 0
        )
    end
    
    return (
        market = market_df.market[1],
        opt_conf_thresh = best_strategy.conf,
        opt_stake_thresh = best_strategy.stake,
        best_roi = best_metrics.roi,
        num_bets = best_metrics.num_bets
    )
end


# 1. Filter for the model you want to optimize
model_df = @subset(all_models_df, :model_name .== "basic_poisson")

# 2. Group by market
grouped_by_market = groupby(model_df, :market)

# 3. Apply (combine) our "finder" function to each group
# We use a do-block for clarity
strategy_portfolio = combine(grouped_by_market) do market_group
    
    # Pass each sub-dataframe (one for each market)
    # into our optimization function
    find_best_market_strategy(market_group,
                              model_name="basic_poisson",
                              min_bets=5, 
                             conf_range = 0.05:0.05:0.95,
                             stake_range = 0.0)
end

println(strategy_portfolio)

sort(strategy_portfolio, :best_roi, rev=true)

grouped_by_market = groupby(model_df, [:market, :tournament_slug])


# Your grouped_by_market is already defined
# grouped_by_market = groupby(model_df, [:market, :tournament_slug])

# Use combine with a 'do' block to run calculations on each group
market_metrics_df = combine(grouped_by_market) do df
    
    # 1. Number of bets
    n_bets = nrow(df)
    
    # Handle empty groups to avoid division by zero
    if n_bets == 0
        return (
            n_bets = 0,
            n_wins = 0,
            win_ratio = 0.0,
            total_profit = 0.0,
            roi = 0.0
        )
    end
    
    # 2. Number of winning bets
    n_wins = sum(df.was_winner)
    
    # 3. Win ratio
    win_ratio = n_wins / n_bets
    
    # 4. Calculate ROI (Return on Investment)
    # We assume a 1-unit stake on every bet.
    # If win: profit = decimal_odds - 1
    # If loss: profit = -1
    profit_vec = ifelse.(df.was_winner, df.decimal_odds .- 1.0, -1.0)
    
    # Sum up all profits
    total_profit = sum(profit_vec)
    
    # ROI = Total Profit / Total Stakes
    # Since total stakes = n_bets * 1 unit, we just divide by n_bets
    roi = total_profit / n_bets
    
    # Return a NamedTuple of our new metrics.
    # DataFrames.jl will turn this into columns.
    return (
        n_bets = n_bets,
        n_wins = n_wins,
        win_ratio = win_ratio,
        total_profit = total_profit,
        roi = roi
    )
end

# Optional: Sort by the most profitable groups
sort!(market_metrics_df, :roi, rev=true)

# Display the result
println(market_metrics_df)


###
using DataFrames, DataFramesMeta, Statistics

# --- 1. Your Helper Struct & Functions (from your notes) ---

struct Strategy
    conf_thresh::Float64
    stake_thresh::Float64
    stake_type::Symbol  # :stake_median
    model_name::String
end

function analyze_strategy(df::AbstractDataFrame, strategy::Strategy)
    
    # 1. Filter for the bets this strategy would have placed
    bets_placed = @subset(df,
        :model_name .== strategy.model_name,
        :conf_0 .> strategy.conf_thresh,
        $(strategy.stake_type) .> strategy.stake_thresh
    )
    
    if isempty(bets_placed)
        return (num_bets=0, roi=-Inf, profit=0.0, win_rate=0.0, avg_odds=0.0)
    end

    # 2. Calculate profit and loss for those bets
    # We use the stake amount (e.g., :stake_median) to size the bet
    profit_vec = @. ifelse(
        bets_placed.was_winner,
        bets_placed[!, strategy.stake_type] * (bets_placed.decimal_odds - 1),
        -bets_placed[!, strategy.stake_type]
    )
    
    # 3. Aggregate metrics
    total_staked = sum(bets_placed[!, strategy.stake_type])
    total_profit = sum(profit_vec)
    
    # Handle case where total_staked is zero
    if total_staked == 0.0
         return (num_bets=0, roi=-Inf, profit=0.0, win_rate=0.0, avg_odds=0.0)
    end
    
    roi = total_profit / total_staked
    
    return (
        num_bets = nrow(bets_placed),
        roi = roi,
        profit = total_profit,
        win_rate = mean(bets_placed.was_winner),
        avg_odds = mean(bets_placed.decimal_odds)
    )
end

"""
Finds the optimal thresholds for a single market by grid searching.
"""
function find_best_market_strategy(market_df::AbstractDataFrame;
                                   model_name::String,
                                   min_bets::Int = 10, # Increased for more reliability
                                   conf_range = 0.05:0.05:0.95,
                                   stake_range = 0.00:0.005:0.10) # Smaller range
    
    best_roi = -Inf # Start at -Inf so any profit is better
    best_strategy = (conf=0.0, stake=0.0)
    best_metrics = (num_bets=0, roi=0.0)
    
    # Grid search
    for conf in conf_range
        for stake in stake_range
            
            s = Strategy(conf, stake, :stake_median, model_name)
            
            # Analyze this specific (conf, stake) combo
            metrics = analyze_strategy(market_df, s)
            
            # --- This is the core logic ---
            # Is this ROI better than our current best?
            # AND does it meet our minimum bet constraint?
            if (metrics.roi > best_roi) && (metrics.num_bets >= min_bets)
                best_roi = metrics.roi
                best_strategy = (conf=conf, stake=stake)
                best_metrics = metrics
            end
        end
    end
    
    # If no strategy was found (e.g., all ROIs were negative or 0 bets)
    if best_roi == -Inf
        return (
            market = market_df.market[1],
            opt_conf_thresh = Inf, # "Disable" this market
            opt_stake_thresh = Inf,
            best_roi = 0.0,
            num_bets = 0
        )
    end
    
    return (
        market = market_df.market[1], # Get the market name from the group
        opt_conf_thresh = best_strategy.conf,
        opt_stake_thresh = best_strategy.stake,
        best_roi = best_metrics.roi,
        num_bets = best_metrics.num_bets
    )
end

# --- 2. Your Data (Assuming 'model_df' and 'grouped_by_market') ---
# model_df = master_df_v1 
# grouped_by_market = groupby(model_df, [:market, :tournament_slug])


# --- 3. The Solution: Apply the Grid Search to Each Group ---

println("Finding optimal strategies for all groups...")

# Define the model and constraints you want to test
MODEL_TO_TEST = "basic_poisson"
MIN_BETS_CONSTRAINT = 10 # Only find strategies with at least 10 bets

# This is the magic!
# It runs `find_best_market_strategy` on each 'df' in your 'grouped_by_market'
optimal_strategies_df = combine(grouped_by_market) do df
    find_best_market_strategy(
        df,
        model_name = MODEL_TO_TEST,
        min_bets = MIN_BETS_CONSTRAINT,
        conf_range = 0.05:0.05:0.95,
         stake_range = 0.0)
end

# Sort to see the most profitable market/league strategies
sort!(optimal_strategies_df, :best_roi, rev=true)

println("--- Optimal Strategy Map ---")
println(optimal_strategies_df)


# 1. Create large, robust groups
grouped_by_market_global = groupby(model_df, :market)



## test 
# 1. Get the vector of match IDs (your code is correct)
mask = subset(ds.matches, :tournament_id => ByRow(isequal(54))).match_id
# 2. Subset model_df to keep rows where :match_id IS IN the mask
df_in_mask = subset(model_df, :match_id => ByRow(in(mask)))

grouped_by_market_global = groupby(df_in_mask, :market)

grouped_by_market_global = groupby(model_df, [:market, :tournament_slug])

# 2. Re-run your optimization with a *much* higher constraint
MIN_BETS_CONSTRAINT = 100 # Find strategies that work over 100+ bets

robust_strategies_df = combine(grouped_by_market_global) do df
    find_best_market_strategy(
        df,
        model_name = "basic_poisson",
        min_bets = MIN_BETS_CONSTRAINT,
        conf_range = 0.05:0.05:0.95,
        stake_range = 0.0:0.01:0.0  # Keep your smart 1D optimization
    )
end

# 3. View the new, more realistic results
sort!(robust_strategies_df, :best_roi, rev=true)
println(robust_strategies_df)

#=
# without top league ( scottish pl ) 
julia> println(robust_strategies_df)
15×5 DataFrame
 Row │ market    opt_conf_thresh  opt_stake_thresh  best_roi     num_bets 
     │ Symbol    Float64          Float64           Float64      Int64    
─────┼────────────────────────────────────────────────────────────────────
   1 │ over_35              0.1                0.0   0.0302184        106
   2 │ over_25              0.05               0.0   0.0288288        118
   3 │ over_15              0.1                0.0   0.0091187        102
   4 │ btts_yes             0.05               0.0   0.00686831       109
   5 │ draw               Inf                Inf     0.0                0
   6 │ over_05            Inf                Inf     0.0                0
   7 │ btts_no              0.15               0.0  -0.030039         108
   8 │ under_15             0.2                0.0  -0.0460552        100
   9 │ under_45             0.05               0.0  -0.0461105        118
  10 │ home                 0.05               0.0  -0.0486298        121
  11 │ over_45              0.1                0.0  -0.0546463        106
  12 │ under_35             0.05               0.0  -0.0796503        121
  13 │ away                 0.1                0.0  -0.133093         103
  14 │ under_25             0.05               0.0  -0.182812         122
  15 │ under_05             0.05               0.0  -0.431173         109

# with top league scottish pl 
julia> println(robust_strategies_df)
15×5 DataFrame
 Row │ market    opt_conf_thresh  opt_stake_thresh  best_roi     num_bets 
     │ Symbol    Float64          Float64           Float64      Int64    
─────┼────────────────────────────────────────────────────────────────────
   1 │ away                 0.6                0.0   0.11908          114
   2 │ over_25              0.15               0.0   0.0392629        362
   3 │ over_15              0.5                0.0   0.0366276        121
   4 │ under_35             0.75               0.0   0.0297232        111
   5 │ over_35              0.15               0.0   0.0265944        349
   6 │ over_05              0.2                0.0   0.0127666        203
   7 │ home                 0.1                0.0   0.00484031       440
   8 │ under_15             0.75               0.0  -0.00273327       116
   9 │ under_45             0.7                0.0  -0.0167387        116
  10 │ btts_no              0.75               0.0  -0.0287006        131
  11 │ over_45              0.15               0.0  -0.0346528        323
  12 │ btts_yes             0.05               0.0  -0.0891045        408
  13 │ under_25             0.7                0.0  -0.122127         158
  14 │ draw                 0.05               0.0  -0.358193         372
  15 │ under_05             0.3                0.0  -0.433909         238

=# 


####
"""
Runs a compounding bankroll simulation based on a strategy map.

# Arguments
- `model_df`: The master DataFrame of all possible bets.
- `strategy_map_df`: Your `robust_strategies_df` DataFrame.
- `min_roi_to_bet`: Only include markets from the map with an ROI > this value.
- `initial_bankroll`: The starting bankroll (e.g., 1.0 unit).
"""
function simulate_compounding_growth(
    model_df::AbstractDataFrame, 
    strategy_map_df::AbstractDataFrame;
    min_roi_to_bet::Float64 = 0.0,
    initial_bankroll::Float64 = 1.0,
    kelly_scalar::Float64 = 1.0  
)
    
    # --- 1. Create the Strategy Map (a Dict for fast lookups) ---
    # We only want to bet on markets that were profitable in our test!
    profitable_strategies = @subset(strategy_map_df, :best_roi .> min_roi_to_bet)
    
    # Convert to a Dict: :market => opt_conf_thresh
    strategy_map = Dict(
        row.market => row.opt_conf_thresh for row in eachrow(profitable_strategies)
    )
    
    if isempty(strategy_map)
        println("Warning: No profitable strategies found with min_roi > $min_roi_to_bet")
        return DataFrame(date=Date[], bankroll=Float64[])
    end

    println("Simulating with $(length(strategy_map)) profitable markets...")
    
    # --- 2. Filter the master list for *only* bets we would place ---
    
    # This is much faster than filtering the whole DataFrame
    bets_to_place_rows = []
    for row in eachrow(model_df)
        # Check if this market is in our strategy
        conf_thresh = get(strategy_map, row.market, Inf) # Inf means "don't bet"
        
        # Check if the bet passes our confidence threshold
        if row.conf_0 > conf_thresh
            push!(bets_to_place_rows, row)
        end
    end
    
    if isempty(bets_to_place_rows)
        println("Warning: Strategy found no bets to place.")
        return DataFrame(date=Date[], bankroll=Float64[])
    end
    
    bets_to_place = DataFrame(bets_to_place_rows)
    
    # --- 3. Sort by date ---
    # This is essential for a "wealth over time" simulation
    sort!(bets_to_place, :match_date)

    # --- 4. Run the compounding simulation loop ---
    bankroll = initial_bankroll
    bankroll_history = []
    
    # Add an initial point for plotting
    push!(bankroll_history, (
        date = minimum(bets_to_place.match_date) - Day(1),
        bankroll = initial_bankroll,
        bet_num = 0
    ))

    for (i, row) in enumerate(eachrow(bets_to_place))

        stake_fraction_full = row.stake_median
        
        # Get stake fraction from the Kelly median
        stake_fraction = stake_fraction_full * kelly_scalar
        
        # Ensure stake is not too large (e.g., cap at 20% of bankroll)
        stake_fraction = min(stake_fraction, 0.2) 
        
        actual_stake = bankroll * stake_fraction

        # Don't allow a bet to ruin the bankroll
        if actual_stake >= bankroll
            actual_stake = bankroll * 0.99 # Bet 99%
        end

        # Calculate profit/loss and update bankroll
        if row.was_winner
            profit = actual_stake * (row.decimal_odds - 1.0)
            bankroll += profit
        else
            loss = actual_stake
            bankroll -= loss
        end
        
        # Stop if we are ruined
        if bankroll <= 0.0
            bankroll = 0.0
            push!(bankroll_history, (
                date = row.match_date, 
                bankroll = 0.0,
                bet_num = i
            ))
            break # Stop the simulation
        end

        # Save the new bankroll value after the bet
        push!(bankroll_history, (
            date = row.match_date, 
            bankroll = bankroll,
            bet_num = i
        ))
    end
    
    println("Simulation complete. Placed $(nrow(bets_to_place)) bets.")
    println("Initial bankroll: $initial_bankroll")
    println("Final bankroll: $(round(bankroll, digits=2))")
    
    return DataFrame(bankroll_history)
end


# v2 
function simulate_compounding_growth(
    model_df::AbstractDataFrame,  
    strategy_map_df::AbstractDataFrame;
    min_roi_to_bet::Float64 = 0.0,
    initial_bankroll::Float64 = 1.0,
    kelly_scalar::Float64 = 1.0  
)
    
    # --- 1. Create the Strategy Map (a Dict for fast lookups) ---
    profitable_strategies = @subset(strategy_map_df, :best_roi .> min_roi_to_bet)
    
    # --- FIX 1: Use a Tuple key (league, market) ---
    strategy_map = Dict(
        (row.tournament_slug, row.market) => row.opt_conf_thresh 
        for row in eachrow(profitable_strategies)
    )
    
    if isempty(strategy_map)
        println("Warning: No profitable strategies found with min_roi > $min_roi_to_bet")
        return DataFrame(date=Date[], bankroll=Float64[])
    end

    println("Simulating with $(length(strategy_map)) profitable markets...")
    
    # --- 2. Filter the master list for *only* bets we would place ---
    
    bets_to_place_rows = []
    for row in eachrow(model_df)
        
        # --- FIX 2: Create the same Tuple key to look up ---
        key = (row.tournament_slug, row.market)
        conf_thresh = get(strategy_map, key, Inf) # Inf means "don't bet"
        
        # Check if the bet passes our confidence threshold
        if row.conf_0 > conf_thresh
            push!(bets_to_place_rows, row)
        end
    end
    
    if isempty(bets_to_place_rows)
        println("Warning: Strategy found no bets to place.")
        return DataFrame(date=Date[], bankroll=Float64[])
    end
    
    bets_to_place = DataFrame(bets_to_place_rows)
    
    # --- 3. Sort by date ---
    sort!(bets_to_place, :match_date)

    # --- 4. Run the compounding simulation loop (no changes needed here) ---
    bankroll = initial_bankroll
    bankroll_history = []
    
    push!(bankroll_history, (
        date = minimum(bets_to_place.match_date) - Day(1),
        bankroll = initial_bankroll,
        bet_num = 0
    ))

    for (i, row) in enumerate(eachrow(bets_to_place))
        stake_fraction_full = row.stake_median
        stake_fraction = stake_fraction_full * kelly_scalar
        stake_fraction = min(stake_fraction, 0.2) 
        
        actual_stake = bankroll * stake_fraction

        if actual_stake >= bankroll
            actual_stake = bankroll * 0.99 
        end

        if row.was_winner
            profit = actual_stake * (row.decimal_odds - 1.0)
            bankroll += profit
        else
            loss = actual_stake
            bankroll -= loss
        end
        
        if bankroll <= 0.0
            bankroll = 0.0
            push!(bankroll_history, (
                date = row.match_date, 
                bankroll = 0.0,
                bet_num = i
            ))
            break 
        end

        push!(bankroll_history, (
            date = row.match_date, 
            bankroll = bankroll,
            bet_num = i
        ))
    end
    
    println("Simulation complete. Placed $(nrow(bets_to_place)) bets.")
    println("Initial bankroll: $initial_bankroll")
    println("Final bankroll: $(round(bankroll, digits=2))")
    
    return DataFrame(bankroll_history)
end



using Dates
# 1. Run the simulation
# We set `min_roi_to_bet = 0.0` to include all 7 profitable markets
# bankroll_df = simulate_compounding_growth(
#     model_df,                # Your master DataFrame
#     robust_strategies_df,    # Your strategy map
#     min_roi_to_bet = 0.0     # We want to bet on all positive ROI markets
# )

bankroll_df_scaled = simulate_compounding_growth(
    model_df,
    robust_strategies_df,
    min_roi_to_bet = 0.0,
    kelly_scalar = 0.12 
)

# 2. See the results
println(bankroll_df)


# 1. Get a list of the profitable markets to test
profitable_markets_df = @subset(robust_strategies_df, :best_roi .> 0.0)

# 2. Create a dictionary to hold all the resulting DataFrames
all_market_simulations = Dict{Symbol, DataFrame}()

println("--- Running Per-Market Simulations (Full Kelly) ---")

for strategy_row in eachrow(robust_strategies_df)
    
    market_name = strategy_row.market
    market_leauge = strategy_row.tournament_slug
    market_roi = strategy_row.best_roi
    
    # 3. Create a temporary, 1-row DataFrame for the strategy
    # This tells the simulation function to *only* bet on this one market
    single_market_strategy_map = DataFrame(strategy_row)

    println("Simulating market: $market_leauge - $market_name (OOS ROI: $(round(market_roi*100, digits=1))%)")
    
    # 4. Run the *exact same* simulation function as before
    bankroll_df = simulate_compounding_growth(
        model_df,
        single_market_strategy_map, # Pass in the 1-row strategy
        min_roi_to_bet = 0.0,
        kelly_scalar = 0.34  # <-- HERE IS THE FIX
    )
    
    # 5. Store the result and print the summary
    all_market_simulations[market_name] = bankroll_df
    
    # Get the final bankroll from the last row
    final_bankroll = isempty(bankroll_df) ? 0.0 : last(bankroll_df.bankroll)
    println("  -> Final Bankroll: $(round(final_bankroll, digits=3))\n")
    
end

# You can now access the full simulation for a market, e.g.:
# println(all_market_simulations[:away])

function find_optimal_scalar(
    model_df, 
    strategy_map_df;
    scalar_range = 0.02:0.02:0.5 # Test from 2% to 50% Kelly
)
    
    println("--- Finding Optimal Kelly Scalar ---")
    best_scalar = 0.0
    best_final_bankroll = 0.0

    for scalar in scalar_range
        
        sim_df = simulate_compounding_growth(
            model_df,
            strategy_map_df,
            min_roi_to_bet = 0.0,
            kelly_scalar = scalar
        )
        
        final_bankroll = isempty(sim_df) ? 0.0 : last(sim_df.bankroll)
        
        println("Scalar: $(round(scalar, digits=2)) -> Final Bankroll: $(round(final_bankroll, digits=3))")
        
        if final_bankroll > best_final_bankroll
            best_final_bankroll = final_bankroll
            best_scalar = scalar
        end
    end
    
    println("\n--- Optimal Result ---")
    println("Best Scalar: $best_scalar")
    println("Best Final Bankroll: $(round(best_final_bankroll, digits=3))")
    
    return (best_scalar, best_final_bankroll)
end

# --- Run the Optimization ---
# This will test 24 different scalar values and find the best one
find_optimal_scalar(model_df, robust_strategies_df)

##############################
# --- Metrics
##############################
#
# Log prob 

# get market log prob 

match_odds

a = collect(predict_config.markets)
a1 = a[1]

"""
    market_log_loss(market, odds, results) -> Float64

Calculates the log-loss for the *normalized* bookmaker probabilities
(odds with overround removed) given the actual match results.

This function uses multiple dispatch to select the correct
calculation for each market type.
"""
function market_log_loss end # Create the generic function

"""
Log-loss calculation for the BTTS market.
"""
function market_log_loss(
    market::BayesianFootball.Markets.MarketBTTS,
    odds::NamedTuple,
    results::NamedTuple
)
    implied_prob_yes = 1 / odds.btts_yes
    implied_prob_no  = 1 / odds.btts_no

    y_yes = results.btts_yes
    y_no  = results.btts_no

    #  normalize probabilities to remove overround
    total_prob = implied_prob_yes + implied_prob_no
    p_yes = implied_prob_yes / total_prob
    p_no  = implied_prob_no / total_prob

    # 3. calculate log-loss
  return y_yes * (-log(p_yes)) + y_no * (-log(p_no))

end


market_log_loss(a1, match_odds, match_results)



"""
Log-loss calculation for the 1X2 market.
"""
function market_log_loss(
    market::BayesianFootball.Markets.Market1X2,
    odds::NamedTuple,
    results::NamedTuple
)
    # 1. Define keys and access data
    implied_prob_home = 1 / odds.home
    implied_prob_draw = 1 / odds.draw
    implied_prob_away = 1 / odds.away

    # 2. Normalize probabilities
    total_prob = implied_prob_home + implied_prob_draw + implied_prob_away
    p_home = implied_prob_home / total_prob
    p_draw = implied_prob_draw / total_prob
    p_away = implied_prob_away / total_prob

  return results.home * ( -log(p_home)) + results.draw * ( -log(p_draw)) + results.away * (-log(p_away)) 

end

a_1x2 = a[4]
market_log_loss(a_1x2, match_odds, match_results)


"""
Log-loss calculation for the Under Over market.
"""
function market_log_loss(
    market::BayesianFootball.Markets.MarketOverUnder,
    odds::NamedTuple,
results::NamedTuple
)
    line_str = replace(string(market.line), "." => "")
    # e.g., over_key = :over_15
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)

    implied_prob_under = 1 / odds[under_key]
    implied_prob_over  = 1 / odds[over_key]


    #  normalize probabilities to remove overround
    total_prob = implied_prob_under + implied_prob_over
    p_under = implied_prob_under / total_prob
    p_over  = implied_prob_over / total_prob

    # 3. calculate log-loss
  return results[under_key] * (-log(p_under)) + results[over_key] * (-log(p_over))

end

a_uo_25 = a[2]

market_log_loss(a_uo_25, match_odds, match_results)




"""
Returns a symbol to use as a key for a market's log-loss.
"""
function get_market_loss_key end # Generic function

function get_market_loss_key(market::Markets.MarketBTTS)
    return :btts_loss
end

function get_market_loss_key(market::Markets.Market1X2)
    return :_1x2_loss
end

function get_market_loss_key(market::Markets.MarketOverUnder)
    line_str = replace(string(market.line), "." => "")
    return Symbol("ou_", line_str, "_loss")
end


# --- NEW WRAPPER FUNCTION ---

"""
Wrapper function to calculate log-loss for all markets
in a PredictionConfig. Returns a NamedTuple with the results.
"""
function market_log_loss(
    predict_config::Predictions.PredictionConfig,
    odds::NamedTuple,
    results::NamedTuple
)
    # Create a generator that produces (Key, Value) pairs
    loss_generator = (
        (get_market_loss_key(market), market_log_loss(market, odds, results))
        for market in predict_config.markets
    )

    # Convert the (Key, Value) pairs into a NamedTuple
    return NamedTuple(loss_generator)
end



market_log_loss(predict_config, match_odds, match_results)


get



###########


function predict_log_loss(
    market::BayesianFootball.Markets.Market1X2, 
    model_probs::NamedTuple,
    results::NamedTuple
)
  return results.home .* ( -log.(model_probs.home)) .+ results.draw * ( -log.(model_probs.draw)) .+ results.away * (-log.(model_probs.away)) 
end 


b = predict_log_loss(a_1x2, match_predict, match_results) 

dens

# --- LOG-LOSS FUNCTIONS FOR MODEL PROBABILITY CHAINS ---

"""
    predict_log_loss(market, model_probs, results) -> Vector{Float64}

Calculates the log-loss for the model's posterior probability chains.

Returns a vector (one loss value per MCMC sample) by dispatching
on the market type.
"""
function predict_log_loss end # Create the generic function

"""
Log-loss chain calculation for the 1X2 market.
"""
function predict_log_loss(
    market::BayesianFootball.Markets.Market1X2,
    model_probs::NamedTuple,
    results::NamedTuple
)
    # Note: .log is element-wise, * is broadcast
    return results.home .* (-log.(model_probs.home)) .+
           results.draw .* (-log.(model_probs.draw)) .+
           results.away .* (-log.(model_probs.away))
end

"""
Log-loss chain calculation for the BTTS market.
"""
function predict_log_loss(
    market::BayesianFootball.Markets.MarketBTTS,
    model_probs::NamedTuple,
    results::NamedTuple
)
    return results.btts_yes .* (-log.(model_probs.btts_yes)) .+
           results.btts_no .* (-log.(model_probs.btts_no))
end

"""
Log-loss chain calculation for the Under Over market.
"""
function predict_log_loss(
    market::BayesianFootball.Markets.MarketOverUnder,
    model_probs::NamedTuple,
    results::NamedTuple
)
    # 1. Generate keys
    line_str = replace(string(market.line), "." => "")
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)

    # 2. Access chains and results (using index [])
    return results[under_key] .* (-log.(model_probs[under_key])) .+
           results[over_key] .* (-log.(model_probs[over_key]))
end


# --- NEW HELPER FUNCTIONS FOR PREDICT WRAPPER ---

"""
Returns a symbol to use as a key for a model's log-loss chain.
"""
function get_predict_loss_key end # Generic function

function get_predict_loss_key(market::Markets.MarketBTTS)
    return :btts
end

function get_predict_loss_key(market::Markets.Market1X2)
    return :_1x2
end

function get_predict_loss_key(market::Markets.MarketOverUnder)
    line_str = replace(string(market.line), "." => "")
    return Symbol("ou_", line_str)
end


# --- NEW WRAPPER FUNCTION FOR MODEL PREDICTIONS ---

"""
Wrapper function to calculate model log-loss chains for all markets
in a PredictionConfig. Returns a NamedTuple with the resulting vectors.
"""
function predict_log_loss(
    predict_config::BayesianFootball.Predictions.PredictionConfig,
    model_probs::NamedTuple,
    results::NamedTuple
)
    # Create a generator that produces (Key, Value) pairs
    # Value will be a Vector{Float64} from predict_log_loss
    loss_generator = (
        (get_predict_loss_key(market), predict_log_loss(market, model_probs, results))
        for market in predict_config.markets
    )

    # Convert the (Key, Value) pairs into a NamedTuple
    return NamedTuple(loss_generator)
end



predict_log_loss(predict_config,  match_predict, match_results)



####
#
## --- PREDICTIVE RPS (BRIER SCORE) FUNCTIONS ---

"""
    predict_rps(market, model_probs, results) -> Vector{Float64}

Calculates the Ranked Probability Score (RPS) chain.
For binary markets (BTTS, O/U), this is identical to the Brier Score.
"""
function predict_rps end # Generic function

"""
RPS chain calculation for the 1X2 market (3-class ordered).
"""
function predict_rps(
    market::BayesianFootball.Markets.Market1X2,
    model_probs::NamedTuple,
    results::NamedTuple
)
    P_H_chain = model_probs.home
    P_D_chain = model_probs.draw
    
    # 1. Cumulative probability chain
    P_HD_chain = P_H_chain .+ P_D_chain

    # 2. Cumulative outcomes (scalars: 1 or 0)
    O_H = results.home
    O_HD = results.home + results.draw

    # 3. Calculate RPS: Sum of squared errors of cumulative probs
    # RPS = (P_H - O_H)^2 + (P_H+D - O_H+D)^2
    term1 = (P_H_chain .- O_H).^2
    term2 = (P_HD_chain .- O_HD).^2
    
    return term1 .+ term2
end

"""
RPS chain calculation for the BTTS market (Binary -> Brier Score).
"""
function predict_rps(
    market::BayesianFootball.Markets.MarketBTTS,
    model_probs::NamedTuple,
    results::NamedTuple
)
    p_yes_chain = model_probs.btts_yes
    y_yes = results.btts_yes # Scalar: 1 or 0
    
    # Brier Score: (p - y)^2
    return (p_yes_chain .- y_yes).^2
end

"""
RPS chain calculation for the Under Over market (Binary -> Brier Score).
"""
function predict_rps(
    market::BayesianFootball.Markets.MarketOverUnder,
    model_probs::NamedTuple,
    results::NamedTuple
)
    # 1. Generate keys
    line_str = replace(string(market.line), "." => "")
    over_key = Symbol("over_", line_str)
    # under_key = Symbol("under_", line_str) # Not needed

    # 2. Access chains and results
    p_over_chain = model_probs[over_key]
    y_over = results[over_key] # Scalar: 1 or 0

    # 3. Brier Score: (p - y)^2
    return (p_over_chain .- y_over).^2
end


# --- PREDICT RPS HELPERS & WRAPPER ---

"""
Returns a symbol to use as a key for a model's RPS chain.
"""
function get_predict_rps_key end # Generic function

function get_predict_rps_key(market::BayesianFootball.Markets.MarketBTTS)
    return :btts
end

function get_predict_rps_key(market::BayesianFootball.Markets.Market1X2)
    return :_1x2
end

function get_predict_rps_key(market::BayesianFootball.Markets.MarketOverUnder)
    line_str = replace(string(market.line), "." => "")
    return Symbol("ou_", line_str)
end

"""
Wrapper function to calculate model RPS chains for all markets
in a PredictionConfig. Returns a NamedTuple with the resulting vectors.
"""
function predict_rps(
    predict_config::BayesianFootball.Predictions.PredictionConfig,
    model_probs::NamedTuple,
    results::NamedTuple
)
    # Create a generator that produces (Key, Value) pairs
    rps_generator = (
        (get_predict_rps_key(market), predict_rps(market, model_probs, results))
        for market in predict_config.markets
        # Check if the model_probs has the keys for this market
    )

    # Convert the (Key, Value) pairs into a NamedTuple
    return NamedTuple(rps_generator)
end


b = predict_rps(predict_config, match_predict, match_results )
