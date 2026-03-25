# src/evaluation/evaluation-module.jl 

module Evaluation 

  using ..Data
  using ..Predictions
  using ..Experiments

  using DataFrames
  using Statistics
  using Distributions
  using HypothesisTests
  using Random

include("./types.jl")
include("./interfaces.jl")
include("./translator.jl") 

# metric methods
include("./metrics_methods/rqr.jl")
include("./metrics_methods/crps.jl")


end
