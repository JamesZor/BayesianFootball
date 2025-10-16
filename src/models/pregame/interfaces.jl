
"""
Defines the abstract types that form the contracts for pre-game models.
"""
module PreGameInterfaces

# Bring the top-level abstract type into this module's scope
using ...Models: AbstractFootballModel

export AbstractPregameModel

# A specific abstract type for any pre-game model.
# All concrete model structs will subtype this.
abstract type AbstractPregameModel <: AbstractFootballModel end

end
