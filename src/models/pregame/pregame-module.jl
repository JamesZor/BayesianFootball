# src/models/pregame/pregame-module.jl


"""
This module organizes all pre-game models.
It includes the interfaces and all concrete model implementations.
"""
module PreGame

# Shared abstract types for all pre-game models
include("interfaces.jl")
# Shared, reusable likelihood functions
include("turing_helpers.jl")

# Directory for concrete model implementations
module Implementations
    using ....Features: FeatureSet 
    # Each model is now in its own self-contained file
    include("./models-src/static-poisson.jl")
    include("./models-src/static-simplex-poisson.jl")
    include("./models-src/hierarchical-simplex-poisson.jl")
end

# Export the model structs to be used in scripts
using .Implementations
export StaticPoisson, HierarchicalSimplexPoisson, build_turing_model

end
