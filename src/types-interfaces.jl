module TypesInterfaces

using DataFrames

export AbstractFootballModel, AbstractPregameModel, AbstractInGameModel, AbstractPregameModel, AbstractInflatedDiagonalPoissonModel, AbstractNegBinModel
export AbstractDixonColesModel

export AbstractStaticPoissonModel, AbstractDynamicPoissonModel
# export AbstractGRWPoissonModel
export Vocabulary, FeatureSet, required_mapping_keys

# This is the highest-level type
abstract type AbstractFootballModel end

# Sub-types
abstract type AbstractPregameModel <: AbstractFootballModel end
abstract type AbstractInGameModel <: AbstractFootballModel end

abstract type AbstractPoissonModel <: AbstractPregameModel end

abstract type AbstractStaticPoissonModel <: AbstractPoissonModel end 
abstract type AbstractDynamicPoissonModel <: AbstractPoissonModel end 



abstract type AbstractDixonColesModel <: AbstractPregameModel end
abstract type AbstractNegBinModel <: AbstractPregameModel end
abstract type AbstractInflatedDiagonalPoissonModel <: AbstractPregameModel end



# --- Flexible Feature Structs ---
# By defining these here, both Models and Features can use them without depending on each other.

"""
    Vocabulary (Your 'G')
"""
struct Vocabulary
    mappings::Dict{Symbol, Any}
end

"""
    FeatureSet (Your 'F_i')
"""
struct FeatureSet
    data::Dict{Symbol, Any}
end

"""
    required_mapping_keys(model::AbstractFootballModel)::Vector{Symbol}

This is the "contract". Each model implementation should override
this function to return a list of symbols for the global
mappings it needs the Vocabulary to build.
"""
function required_mapping_keys(model::AbstractFootballModel)
    # By default, we assume all models need at least a team mapping.
    return [:team_map, :n_teams]
end

end
