# src/predictions/prediction-module.jl

module Predictions
#
# include("./markets.jl")
# include("./calculations.jl")

using ..Models
using ..Features
using ..Data
using ..Markets
using ..Calculations
using Turing
using DataFrames
using Statistics

export PredictionConfig


"""
Defines what to predict (your Config_p).
"""
struct PredictionConfig
    # A list of markets to calculate, e.g. [Market1X2(), MarketOverUnder(2.5)]
    markets::AbstractSet{AbstractMarket}
    # A list of calculations to perform, e.g. [CalcProbability(), CalcExpectedValue()]
    calculations::AbstractSet{AbstractCalculation}
end





# --- This is your helper function from scripts/dev_train_predict.jl ---
#
# It's now a private helper inside the Prediction module
function _calculate_betting_distributions(prediction_chain::Chains)
    # ... (your exact code from) ...
    #
    # This function is perfect as-is. It returns a Dict{String, Vector{Float64}}
end

# --- This is the main "morphism" h ---
"""
    predict(model, chains, features, config, data_store)

Generates predictions from a trained model.
"""
function predict(
    model::Models.AbstractFootballModel, 
    chains::Chains, 
    features::FeatureSet, 
    config::PredictionConfig,
    data_store::DataStore # Need this to get market odds
)
    # 1. Get the raw goal posteriors (as seen in your script)
    # We need to build the prediction-time model
    # This assumes we are predicting on the *training* set.
    # A more advanced version would take a separate `test_features`.

    home_ids = vcat(features.round_home_ids...) #
    away_ids = vcat(features.round_away_ids...) #

    # Build the model defined in your script
    prediction_turing_model = Models.PreGame.build_turing_model(
        model,
        features.n_teams, 
        home_ids, 
        away_ids, 
        missing, 
        missing
    )

    # Run Turing.predict
    prediction_chain = Turing.predict(prediction_turing_model, chains)

    # 2. Calculate Market Probability Distributions
    # This uses your helper
    market_dists = _calculate_betting_distributions(prediction_chain)

    # 3. Build the final results DataFrame
    # This is "P", your set of results.
    # We'll cross-join markets and calculations

    results_df = DataFrame()

    for market in config.markets
        # Get the key for this market (e.g., "over_25")
        market_key = _get_market_key(market) # e.g., MarketOverUnder(2.5) -> "over_25"

        !haskey(market_dists, market_key) && continue

        dist = market_dists[market_key]

        for calc in config.calculations
            if calc isa CalcProbability
                # Your code from
                mean_prob = mean(dist) 
                results_df[!, Symbol(market_key, "_prob")] = [mean_prob]

            elseif calc isa CalcExpectedValue
                # This is where we'd fetch odds from data_store
                # and run your EV logic
                # ev = ...
                # results_df[!, Symbol(market_key, "_ev")] = [ev]
            end
        end
    end

    return results_df # This is your result "P"
end

# Helper to convert Market structs to the Dict keys
function _get_market_key(market::Market1X2) 
    # This is a simplification; you'd need "home_win", "draw", "away_win"
    return "home_win" 
end

function _get_market_key(market::MarketOverUnder)
    if market.line == 0.5; return "over_05"; end
    if market.line == 1.5; return "over_15"; end
    if market.line == 2.5; return "over_25"; end #
    # ... etc.
end

function _get_market_key(market::MarketBTTS)
    return "btts_yes" #
end

end # module Predictions
