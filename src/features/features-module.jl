
"""
This module is responsible for transforming raw data from a DataStore
into a model-ready FeatureSet by using a global Vocabulary.
"""
module Features

using DataFrames
using Dates
using ..Data
# Features depends on the central interfaces for types...
using ..TypesInterfaces
# ...and on Models for the concrete model types and their contract methods.
using ..Models: required_mapping_keys


export Vocabulary, FeatureSet, create_vocabulary, create_features

# --- Constants for required columns ---
const REQUIRED_MATCH_COLS = [
    :home_team, :away_team, :home_score, :away_score, :match_date, :tournament_slug
]

# Include the separated logic files.
include("./vocabulary.jl")
include("./feature-sets.jl")





end



