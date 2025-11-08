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


