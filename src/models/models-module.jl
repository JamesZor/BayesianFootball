"""
This module brings together all the different model types (pre-game, in-game, etc.)
into a single, organized unit.
"""
module Models

# This is the highest-level abstract type for all models in the package.
abstract type AbstractFootballModel end

# --- Include and export sub-modules ---

# Pre-Game Models
include("pregame/interfaces.jl")
include("pregame/components.jl")
include("pregame/turing_helpers.jl")
include("pregame/api.jl")

# In-Game Models (Placeholder)
include("ingame/ingame-module.jl")

# Expose the sub-modules to the rest of the package
export PreGame, InGame, AbstractFootballModel

end
