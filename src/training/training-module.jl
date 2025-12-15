# src/training/training-module.jl

module Training

using Turing
using ProgressMeter
using Base.Threads
using ..Samplers
using ..Models
using ..Features
using ..Models.PreGame: build_turing_model

# 1. Configuration & Persistence
include("./types.jl")
include("./checkpointing.jl")

# 2. Strategies
include("./strategies/independent.jl")
# include("strategies/sequential.jl") # Placeholder for future
include("./method.jl")


end # module
