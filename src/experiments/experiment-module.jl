
module Experiments


# imports
using ..Models: AbstractFootballModel
using ..Sampling: AbstractTrainingMethod
using ..Predictions: PredictionConfig


include("./types.jl")

# exports 
export Experiment, AbstractSplitter, PredictionConfig


end
