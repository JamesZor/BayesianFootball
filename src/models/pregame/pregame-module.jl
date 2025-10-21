# src/models/pregame/pregame-module.jl

"""
This module organizes all pre-game models.
"""
module PreGame

# This module also only depends on the central interfaces.
# The '...' goes up two levels from PreGame -> Models -> BayesianFootball to find TypesInterfaces.
using ...TypesInterfaces: AbstractFootballModel, AbstractPregameModel, FeatureSet

# Shared abstract types are now in the main interfaces file.
include("interfaces.jl")
# Shared, reusable likelihood functions
include("turing_helpers.jl")

# Directory for concrete model implementations
module Implementations
    # It also only needs TypesInterfaces for its contracts.
    # '....' goes up three levels from Implementations -> PreGame -> Models -> BayesianFootball
    using ....TypesInterfaces: FeatureSet
    
    # Each model is now in its own self-contained file
    include("./models-src/static-poisson.jl")
    include("./models-src/static-simplex-poisson.jl")
    include("./models-src/hierarchical-simplex-poisson.jl")
end

# Export the model structs to be used in scripts
using .Implementations
export StaticPoisson, StaticSimplexPoisson, HierarchicalSimplexPoisson, build_turing_model, predict

# This is where we define the specific methods for our contract.
# This extends the function originally defined in TypesInterfaces.
import ...TypesInterfaces: required_mapping_keys

required_mapping_keys(model::StaticPoisson) = [:team_map, :n_teams]
required_mapping_keys(model::StaticSimplexPoisson) = [:team_map, :n_teams]
# Add more specific model implementations here, e.g.:
# required_mapping_keys(model::HierarchicalPoisson) = [:team_map, :n_teams, :league_map, :n_leagues]

end
