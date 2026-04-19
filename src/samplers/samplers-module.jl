# src/samplers/samplers-module.jl

module Samplers

using Turing
# using Zygote
using Optimization      
using OptimizationOptimJL 
using Optim
using ReverseDiff, Memoization

# 1. Base Types & Interfaces
include("./types.jl")
include("./interface.jl")

# 2. Initialization Sub-system
# Must be loaded before engines so NUTSConfig can see UniformInit/MapInit
include("./initialisation/types.jl")
include("./initialisation/uniform.jl")
include("./initialisation/map.jl")

# 3. Engines
include("./engines/nuts.jl")
include("./engines/map.jl")
include("./engines/sgld.jl")
include("./engines/advi.jl")

end # module
