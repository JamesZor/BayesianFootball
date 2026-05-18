# src/features/features-module.jl

"""
This module is responsible for transforming raw data from a DataStore
into a model-ready FeatureSet using the new relational SplitBoundary architecture.
"""
module Features

using DataFrames
using Dates
using Base.Threads
using ..Data
using ..TypesInterfaces

export FeatureSet, create_features, required_features, add_feature!

# Core Architecture
include("./model_requirements.jl")
include("./vocabulary.jl")
include("./map_builders.jl")
include("./builder.jl")

# Relational Extractors
include("./market_inverse_utils.jl")
include("./extractors/core_extractors.jl")
include("./extractors/time_extractors.jl")
include("./extractors/stats_extractors.jl")
include("./extractors/market_extractors.jl")
include("./extractors/player_extractors.jl")

end # module
