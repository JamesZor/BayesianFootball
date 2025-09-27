using BayesianFootball
using DataFrames
using Dates
using Statistics, StatsBase, StatsPlots, Distributions


include("/home/james/bet_project/models_julia/workspace/basic_state_space/setup.jl")
include("/home/james/bet_project/models_julia/workspace/bivariate_maher_model/match_day_utils.jl")
using .AR1NegativeBinomial

include("/home/james/bet_project/models_julia/workspace/basic_state_space/prediction.jl")

using .AR1NegBiPrediction


all_model_paths = Dict(
  "ssm_bineg" => "/home/james/bet_project/models_julia/experiments/ar1_poisson_test/ar1_negbi_2425_to_2526_20250926-173118"
)


