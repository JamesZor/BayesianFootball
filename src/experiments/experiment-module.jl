# src/Experiments/experiment-module

module Experiments

# using Reexport
# using Accessors # Need this here for the macros to work if exported

using ..Models
using ..Data
using ..Features
using ..Training
# using ..Predictions
using ..Samplers

include("./types.jl")
include("./presets.jl") 
include("./runner.jl")
include("./display.jl")

include("post_processing.jl")

export ExperimentConfig, ExperimentResults, run_experiment
export BENCHMARK_DEFAULTS, create_benchmark_config
export LatentStates, extract_oos_predictions 
end
