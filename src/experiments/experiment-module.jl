
module Experiments


# imports
using ..Models: AbstractFootballModel
using ..Sampling: AbstractTrainingMethod
using ..Predictions: PredictionConfig


include("./types.jl")
# include("./runner.jl")
# exports 
export Experiment, AbstractSplitter, PredictionConfig

# export run_experiment
# export ExperimentRunner

end
