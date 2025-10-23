
module Experiments


# imports
using ..Models: AbstractFootballModel
# using ..Sampling: AbstractTrainingMethod
using ..Predictions: PredictionConfig
using ..Data: AbstractSplitter


# include("./types.jl")
# include("./runner.jl")
# exports 
export Experiment, PredictionConfig


# --- Experiment struct ---
struct Experiment
    name::String
    model::AbstractFootballModel
    splitter::AbstractSplitter 
    # sampler_config::AbstractTrainingMethod
end


# export run_experiment
# export ExperimentRunner

end
