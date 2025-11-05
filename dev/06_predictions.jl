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
# JLD2.save_object("training_results.jld2", results)

results = JLD2.load_object("training_results.jld2")


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
p_match =

match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, all_oos_results[match_id]...)

model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

match_odds = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
match_results = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds.odds)

market_odds = subset( ds.odds, :match_id => ByRow(isequal(match_id)))


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
