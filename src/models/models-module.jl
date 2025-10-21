"""
This module brings together all the different model types.
"""
module Models

# Models now ONLY depends on the central TypesInterfaces module for its contracts
# and shared structs. It has NO dependency on Features.
using ..TypesInterfaces

# --- Include and export sub-modules ---
include("pregame/pregame-module.jl")
include("ingame/ingame-module.jl")

# Expose the sub-modules to the rest of the package
export PreGame, InGame
# We must re-export the contract function so other modules can use it.
export required_mapping_keys

end
