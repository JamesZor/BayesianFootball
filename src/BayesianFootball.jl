module BayesianFootball

using CategoricalArrays # maybe 
# Package dependencies
using CSV, DataFrames
using Statistics, Dates
using Turing, StatsPlots, LinearAlgebra, Distributions, Random, MCMCChains
using Base.Threads, ThreadsX
using JLD2, JSON

# Include all components in dependency order
include("types.jl")

# Data pipeline
include("data/loader.jl")
include("data/mapping.jl") 

include("data/splitting.jl")
# export TimeSeriesSplitsConfig, TimeSeriesSplits
export time_series_splits, summarize_splits

# Data Utils 
include("./data/utils_incidents.jl")
export get_match_results, process_matches_results
export get_line_active_minutes, process_matches_active_minutes

# data market odds
include("./data/utils_market_odds.jl")
export get_processed_game_line_odds, process_matches_odds, default_marketodds_config 



# Features
# include("features/base.jl")  # TODO: Create this file
include("features/maher.jl")
include("features/maher_variants.jl")
export feature_map_maher_league_ha

# Models
# include("models/base.jl")  # TODO: Create this file
include("models/maher.jl")
include("models/maher_variants.jl")
export maher_league_ha_model

# Training
include("training/morphisms.jl")
# include("training/sampling.jl")  # TODO: Create this file
include("training/pipeline.jl")

# Prediction 
include("prediction/basic_maher.jl")
export Predictions
export extract_posterior_samples, extract_samples, predict_match_chain_ft, predict_match_chain_ht, predict_match_ft_ht_chain, predict_round_chains, predict_target_season
# export MatchPredict, MatchHalfChainPredicts

# Evaluation - TODO: Create these files
# eval - kelly 
include("./evaluation/kelly.jl")
export process_matches_kelly, apply_kelly_to_match
# include("evaluation/diagnostics.jl")
include("evaluation/metrics.jl")

# Experiments
include("experiments/runner.jl")
include("experiments/registry.jl")   # ADD THIS LINE

include("experiments/persistence.jl")  # TODO: Create this file
export ExperimentRun, prepare_run, save, load_run # ADD THESE
# include("experiments/comparison.jl")  # TODO: Create this file

# Export main API
export DataFiles, DataStore
export MappingFunctions, MappedData, create_list_mapping
export ModelConfig, ModelSampleConfig, ExperimentConfig
export BasicMaherModels, TrainedChains, ExperimentResult
export basic_maher_model_raw, feature_map_basic_maher_model
export run_experiment, train_all_splits
export create_experiment_config

# experiments/persistence 
# export load_experiment, save_experiment


# Nice packages to have

export DataFrames, Statistics, Plots, Distributions, KernelDensity, Plots, StatsPlots, Turing




# Export any additional functions you need
export compose_training_morphism

end
