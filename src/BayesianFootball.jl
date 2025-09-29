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
export time_series_splits, summarize_splits

# Data Utils 
include("./data/utils_incidents.jl")
export get_match_results, process_matches_results
export get_line_active_minutes, process_matches_active_minutes

# data market odds
include("./data/utils_market_odds.jl")
export get_processed_game_line_odds, process_matches_odds, default_marketodds_config 


# Models
include("features/core.jl")
include("models/maher.jl")
include("models/maher_variants.jl")



# Training
include("training/morphisms.jl")
include("training/pipeline.jl")

include("prediction/api.jl") # NEW: Include the generic prediction API
# Prediction 
# include("prediction/basic_maher.jl")

# export Predictions
# export extract_posterior_samples, extract_samples, predict_match_chain_ft, predict_match_chain_ht, predict_match_ft_ht_chain, predict_round_chains, predict_target_season

# Evaluation - TODO: Create these files
# eval - kelly 
include("./evaluation/kelly.jl")
export process_matches_kelly, apply_kelly_to_match
# include("evaluation/diagnostics.jl")
include("evaluation/metrics.jl")

# Experiments
include("experiments/runner.jl")
include("experiments/registry.jl")   
include("experiments/persistence.jl") 
export ExperimentRun, prepare_run, save, load_run, save_experiment_metadata



# ============================================================================
# --- Main API Exports ---
# ============================================================================
# model 
export AbstractStateSpaceModel
# --- Core Types ---
export DataFiles, DataStore, MappingFunctions, MappedData
export TrainedModel # NEW: Export our main model container

# --- Model Definition Protocol ---
export AbstractModelDefinition, MaherBasic, MaherLeagueHA
export build_turing_model, get_required_features

# --- Training & Experimentation ---
export ExperimentConfig, ExperimentResult, TrainedChains
export create_master_features
export run_experiment, train_all_splits, time_series_splits
export create_experiment_config
export prepare_run, save, load_run
export list_runs, load_model # NEW: Export the user-friendly loading functions

# --- Prediction ---
export predict_target_season # NEW: Export the main high-level prediction function
export Predictions # Export the Predictions module to access its types

# --- Utility packages ---
export DataFrames, Statistics, Plots, Turing

# # Data
# export DataFiles, DataStore
# export MappingFunctions, MappedData, create_list_mapping
#
# # Model Definition Protocol
# export AbstractModelDefinition, MaherBasic, MaherLeagueHA
# export build_turing_model, get_required_features # Export the protocol functions
#
# # Training & Experimentation
# export ModelSampleConfig, ExperimentConfig, ExperimentResult, BasicMaherModels, TrainedChains
# export create_master_features # Export the main feature generator
# export run_experiment, train_all_splits
# export create_experiment_config
# export compose_training_morphism
#
# # Utility packages
# export DataFrames, Statistics, Plots, Distributions, KernelDensity, Plots, StatsPlots, Turing
#
end
