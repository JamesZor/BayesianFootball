"""
Defines the abstract types that form the contracts for pre-game models.
"""
module PreGameInterfaces

# Bring the top-level abstract type into this module's scope
using ..Models: AbstractFootballModel

export AbstractPregameModel, GoalDistribution, TimeDynamic

# A specific abstract type for any pre-game model
abstract type AbstractPregameModel <: AbstractFootballModel end

# Abstract types for the model components (the "plug-ins")
abstract type GoalDistribution end
abstract type TimeDynamic end

end
