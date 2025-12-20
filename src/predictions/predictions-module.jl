# src/predictions/predictions-module 

module Predictions

using DataFrames
using Statistics
using Base.Threads
using Distributions

using ..Models
# using ..Data
using ..TypesInterfaces

# include("./markets.jl")
# using ..Markets

# 1. Types 
include("./types.jl")



# 2. Logic (Functions added to Predictions namespace)
# include("./market_data.jl")
# include("./inference.jl") # Dispatcher

# 3. Method Implementations (The math)
# We can include these here or inside inference.jl. 
# Including here makes it clear what models are supported.
# include("methods/poisson.jl")
# include("methods/dixon_coles.jl")

export 
    # Types
    PPD
    
    # Functions
    # prepare_market_data

# model_inference, predict_market
end
