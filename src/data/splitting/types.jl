# src/data/splitting/types.jl

using DataFrames

export AbstractSplitter, AbstractSplitMetaData, SplitMetaData
export StaticSplit, ExpandingWindowCV, WindowCV, CVConfig
# TimeSeriesSplits is exported for internal consistency if needed, though mostly used by methods
export TimeSeriesSplits

# --- Metadata ---

abstract type AbstractSplitMetaData end

"""
    SplitMetaData
Contains the context for a specific data split (fold).
"""
struct SplitMetaData <: AbstractSplitMetaData
    tournament_id::Int
    train_season::String       # The primary target season (e.g. "20/21")
    target_season::String      # Usually same as train_season in expanding window
    history_depth::Int         # Number of historical seasons included
    time_step::Int             # The dynamics index (e.g. match_week 5)
    warmup_period::Int         # The starting dynamics index
end

function Base.show(io::IO, meta::SplitMetaData)
    print(io, "Split(Tourn: $(meta.tournament_id), Season: $(meta.train_season), Week: $(meta.time_step), Hist: $(meta.history_depth))")
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

# --- Iterator Structs ---

"""
    TimeSeriesSplits
State container for the rolling window iterator.
"""
struct TimeSeriesSplits
    base_indices::Vector{Int}
    target_rounds_by_season::Dict{String, Vector}
    target_round_sequence::Vector{Tuple{String, Any}}
    original_df::DataFrame 
    window_col::Symbol 
    window_size::Union{Number, Nothing} 
end
