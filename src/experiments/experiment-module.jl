
module Experiments


# imports
using ..Models: AbstractFootballModel
using ..Sampling: AbstractTrainingMethod


include("./types.jl")

# exports 
export Experiment, AbstractSplitter, PredictionConfig


end
