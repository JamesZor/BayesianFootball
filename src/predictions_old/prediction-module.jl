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
using ..TypesInterfaces

using Turing
using DataFrames
using Statistics
using Distributions


export PredictionConfig, predict_market


"""
Defines what to predict (your Config_p).
"""
struct PredictionConfig
    # A list of markets to calculate, e.g. [Market1X2(), MarketOverUnder(2.5)]
    markets::AbstractSet{AbstractMarket}
    # # A list of calculations to perform, e.g. [CalcProbability(), CalcExpectedValue()]
    # calculations::AbstractSet{AbstractCalculation}
end





include("./pregame/predict-abstractpoisson.jl")
include("./pregame/predict-dixoncoles.jl")
include("./markets-helpers.jl")

end 

