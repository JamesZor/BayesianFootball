# src/experiments/diagnostics/diagnostics.jl

"""
    Diagnostics

Sub-module for analyzing the convergence and stability of Bayesian models
across walk-forward cross-validation splits.
"""
module Diagnostics

using DataFrames
using Statistics
using MCMCChains
using HypothesisTests
using ...Data
using ...Features
using ...Models.PreGame
using ..Experiments

export extract_chains
export check_convergence, check_stability
export ExperimentChains, ChainDiagnostic, StabilityDiagnostic

include("types.jl")
include("utils.jl")
include("extraction.jl")
include("convergence.jl")
include("stability.jl")
include("display.jl")

end # module
