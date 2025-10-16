"""
This module brings together all the different model types (pre-game, in-game, etc.)
into a single, organized unit. It defines the highest-level abstract type.
"""
module Models

# This is the highest-level abstract type for all models in the package.
abstract type AbstractFootballModel end

# --- Include and export sub-modules ---
include("pregame/pregame-module.jl")
include("ingame/ingame-module.jl")

# Expose the sub-modules to the rest of the package
export PreGame, InGame

end
