# src/signals/signals-module.jl

module Signals

using DataFrames
using Statistics
using Optim
using ..TypesInterfaces: AbstractFootballModel
using ..Predictions: PPD

# 1. Types & Interfaces
include("types.jl")
include("interfaces.jl")

# 2. Processing Logic
include("process_signals.jl")

# 3. Implementations
# Create a folder named 'implementations' inside src/signals/
include("implementations/flat.jl")
include("implementations/kelly.jl")

# Exports
export 
    # Types
    AbstractSignal, 
    SignalsResult,
    
    # Concrete Signals
    FlatStake,
    KellyCriterion,
    BayesianKelly,
    AnalyticalShrinkageKelly,
    ExactBayesianKelly,
    
    # Functions
    process_signals,
    signal_name,
    signal_parameters,
    signal_description

end
