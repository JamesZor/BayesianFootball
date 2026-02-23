# src/data/splitting/types.jl

using DataFrames

export AbstractSplitter, AbstractSplitMetaData, SplitMetaData
export StaticSplit, ExpandingWindowCV, WindowCV, CVConfig
export TimeSeriesSplits
# --- NEW EXPORTS ---
export GroupedSplitMetaData, GroupedCVConfig

# --- Metadata ---

abstract type AbstractSplitMetaData end

"""
    SplitMetaData
Contains the context for a specific data split (fold).
"""
struct SplitMetaData <: AbstractSplitMetaData
    tournament_id::Int
    train_season::String       
    target_season::String      
    history_depth::Int         
    time_step::Int             
    warmup_period::Int         
end

function Base.show(io::IO, meta::SplitMetaData)
    print(io, "Split(Tourn: $(meta.tournament_id), Season: $(meta.train_season), Week: $(meta.time_step), Hist: $(meta.history_depth))")
end

"""
    GroupedSplitMetaData
Contains the context for a specific data split encompassing multiple grouped tournaments.
"""
struct GroupedSplitMetaData <: AbstractSplitMetaData
    tournament_ids::Vector{Int}
    train_season::String       
    target_season::String      
    history_depth::Int         
    time_step::Int             
    warmup_period::Int         
end

function Base.show(io::IO, meta::GroupedSplitMetaData)
    print(io, "GroupedSplit(Tourns: $(meta.tournament_ids), Season: $(meta.train_season), Week: $(meta.time_step), Hist: $(meta.history_depth))")
end

# --- Splitter Configurations ---

abstract type AbstractSplitter end

Base.@kwdef struct StaticSplit <: AbstractSplitter
    train_seasons::Vector{String}
    window_col::Symbol = :round
    dynamics_col::Union{Symbol, Nothing} = nothing 
end

Base.@kwdef struct ExpandingWindowCV <: AbstractSplitter
    train_seasons::Vector{String}
    test_seasons::Vector{String}
    window_col::Symbol 
    dynamics_col::Union{Symbol, Nothing} = nothing
    method::Symbol = :sequential
end

struct WindowCV <: AbstractSplitter
    base_seasons::Vector{String}
    target_seasons::Vector{String}
    window_col::Symbol
    window_size::Number
    ordering::Symbol
end

Base.@kwdef struct CVConfig <: AbstractSplitter
    tournament_ids::Vector{Int} = [1] 
    target_seasons::Vector{String}
    history_seasons::Int = 0
    dynamics_col::Symbol = :match_week
    warmup_period::Int = 5
    end_dynamics::Union{Int, Nothing} = nothing 
    stop_early::Bool = false                     
end

Base.@kwdef struct GroupedCVConfig <: AbstractSplitter
    tournament_groups::Vector{Vector{Int}} = [[1]] 
    target_seasons::Vector{String}
    history_seasons::Int = 0
    dynamics_col::Symbol = :match_week
    warmup_period::Int = 5
    end_dynamics::Union{Int, Nothing} = nothing 
    stop_early::Bool = false                     
end

# --- Iterator Structs ---
struct TimeSeriesSplits
    base_indices::Vector{Int}
    target_rounds_by_season::Dict{String, Vector}
    target_round_sequence::Vector{Tuple{String, Any}}
    original_df::DataFrame 
    window_col::Symbol 
    window_size::Union{Number, Nothing} 
end
