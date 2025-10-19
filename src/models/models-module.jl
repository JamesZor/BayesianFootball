"""
This module brings together all the different model types (pre-game, in-game, etc.)
into a single, organized unit. It defines the highest-level abstract type.
"""
module Models

using ..TypesInterfaces
using ..Features

# --- Include and export sub-modules ---
include("pregame/pregame-module.jl")
include("ingame/ingame-module.jl")

# Expose the sub-modules to the rest of the package
export PreGame, InGame

end
