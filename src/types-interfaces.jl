module TypesInterfaces

using DataFrames

export AbstractFootballModel, AbstractPregameModel, AbstractInGameModel, AbstractPregameModel, AbstractInflatedDiagonalPoissonModel, AbstractNegBinModel
export AbstractDixonColesModel

export AbstractStaticPoissonModel, AbstractDynamicPoissonModel
export Vocabulary, FeatureSet, required_mapping_keys

# This is the highest-level type
abstract type AbstractFootballModel end

# Sub-types
abstract type AbstractPregameModel <: AbstractFootballModel end
abstract type AbstractInGameModel <: AbstractFootballModel end

abstract type AbstractPoissonModel <: AbstractPregameModel end
abstract type AbstractBivariatePoissonModel <: AbstractPregameModel end

abstract type AbstractStaticPoissonModel <: AbstractPoissonModel end 

abstract type AbstractStaticBivariatePoissonModel <: AbstractBivariatePoissonModel end
export AbstractBivariatePoissonModel, AbstractStaticBivariatePoissonModel

abstract type AbstractDynamicPoissonModel <: AbstractPoissonModel end 

abstract type AbstractDixonColesModel <: AbstractPregameModel end
abstract type AbstractNegBinModel <: AbstractPregameModel end
abstract type AbstractInflatedDiagonalPoissonModel <: AbstractPregameModel end

# --- Flexible Feature Structs ---

"""
    FeatureSet
    
A self-contained container for model inputs. 
It holds both the numerical data (e.g., `flat_home_goals`) AND 
the metadata mappings (e.g., `team_map`) required to interpret them.
"""
struct FeatureSet <: AbstractDict{Symbol, Any}
    data::Dict{Symbol, Any}
end

# Constructor helper
FeatureSet(pairs::Pair...) = FeatureSet(Dict{Symbol, Any}(pairs...))

# --- AbstractDict Interface Implementation ---
Base.getindex(fs::FeatureSet, key::Symbol) = getindex(fs.data, key)
Base.setindex!(fs::FeatureSet, value, key::Symbol) = setindex!(fs.data, value, key)
Base.length(fs::FeatureSet) = length(fs.data)
Base.iterate(fs::FeatureSet, state...) = iterate(fs.data, state...)
Base.keys(fs::FeatureSet) = keys(fs.data)
Base.haskey(fs::FeatureSet, key::Symbol) = haskey(fs.data, key)
Base.get(fs::FeatureSet, key::Symbol, default) = get(fs.data, key, default)


export FeatureCollection

"""
    FeatureCollection{M}

A wrapper around a sequence of (FeatureSet, Metadata) tuples.
Provides a clean interface for training loops.
"""
struct FeatureCollection{M} <: AbstractVector{Tuple{FeatureSet, M}}
    items::Vector{Tuple{FeatureSet, M}}
end

# --- Interface Implementation (Makes it behave like a Vector) ---
Base.size(fc::FeatureCollection) = size(fc.items)
Base.getindex(fc::FeatureCollection, i::Int) = getindex(fc.items, i)
Base.setindex!(fc::FeatureCollection, v, i::Int) = setindex!(fc.items, v, i)
Base.IndexStyle(::Type{<:FeatureCollection}) = IndexLinear()

# HACK: - update this to something more useful - placeholder
# Optional: Specialized constructor or show methods
function Base.show(io::IO, ::MIME"text/plain", fc::FeatureCollection)
    print(io, "FeatureCollection with $(length(fc)) splits")
end


# HACK: Remove this after updating 
# Legacy/Deprecated (Optional to keep for now)
struct Vocabulary
    mappings::Dict{Symbol, Any}
end

"""
    required_mapping_keys(model::AbstractFootballModel)::Vector{Symbol}

This is the "contract". Each model implementation should override
this function to return a list of symbols for the global
mappings it needs.
"""
function required_mapping_keys(model::AbstractFootballModel)
    # By default, we assume all models need at least a team mapping.
    return [:team_map, :n_teams]
end

end
