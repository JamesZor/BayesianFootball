module TypesInterfaces

export AbstractFootballModel, AbstractPregameModel, AbstractInGameModel

# This is the highest-level type
abstract type AbstractFootballModel end

# Sub-types
abstract type AbstractPregameModel <: AbstractFootballModel end
abstract type AbstractInGameModel <: AbstractFootballModel end

end # module Interfaces
