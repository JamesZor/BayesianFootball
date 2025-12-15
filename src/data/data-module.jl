# src/data/data-module.jl

module Data

using DataFrames
using CSV
using Dates
using InlineStrings

# 1. Load Constants & Types
include("./constants.jl")
include("./types.jl")
include("./display.jl")

# 2. Load IO Logic
include("./io.jl")

# 3. Load Preprocessing
# Note: Ensure src/data/preprocessing.jl DOES NOT wrap itself in 'module DataPreprocessing'
# It should just be a script of functions like the other files.
include("./preprocessing.jl") 

# 4. Load Splitting Logic
include("./splitting/types.jl")
include("./splitting/methods.jl")
include("./splitting/display.jl")

end # module
