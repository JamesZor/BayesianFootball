# src/data/splitting.jl

using DataFrames
using Base: collect

export AbstractSplitter, StaticSplit, ExpandingWindowCV, WindowCV, create_data_splits


# --- 1. SPLITTER TYPES ---
# --- Abstract Type ---
abstract type AbstractSplitter end

# --- 1. Static Split (Standard) ---
Base.@kwdef struct StaticSplit <: AbstractSplitter
    train_seasons::Vector{String}
    window_col::Symbol = :round
    
    # NEW: Optional time column for model dynamics
    # If explicitly passed as nothing in constructor, we handle it in the logic
    dynamics_col::Union{Symbol, Nothing} = nothing 
end

# --- 2. Expanding Window (Backtesting) ---
Base.@kwdef struct ExpandingWindowCV <: AbstractSplitter
    train_seasons::Vector{String}
    test_seasons::Vector{String}
    window_col::Symbol           # The column used for WINDOWING (e.g., :split_col)
    dynamics_col::Union{Symbol, Nothing} = nothing
    method::Symbol = :sequential
end

"""
old version 
"""

# abstract type AbstractSplitter end
#
# struct StaticSplit <: AbstractSplitter
#     train_seasons::Vector{String}
#     round_col::Union{Symbol, AbstractVector{Symbol}}
# end
#
# function StaticSplit(; train_seasons::Vector{String}, round_col = Symbol[])
#     return StaticSplit(train_seasons, round_col)
# end
#
# struct ExpandingWindowCV <: AbstractSplitter
#     base_seasons::Vector{String}
#     target_seasons::Vector{String}
#     round_col::Symbol
#     ordering::Symbol
# end
#
struct WindowCV <: AbstractSplitter
    base_seasons::Vector{String}
    target_seasons::Vector{String}
    window_col::Symbol
    window_size::Number
    ordering::Symbol
end

########################################
# --- 2. REVISED UNIFIED API FUNCTION ---
########################################

"""
    create_data_splits(data_store::DataStore, splitter::AbstractSplitter)
"""
function create_data_splits(data_store::DataStore, splitter::StaticSplit)::Vector{Tuple{SubDataFrame, String}}
    println("Creating a single static data split (using view)...")
    row_indices = findall(s -> s in splitter.train_seasons, data_store.matches.season)
    train_view = view(data_store.matches, row_indices, :)
    split_metadata = "static_seasons_$(join(splitter.train_seasons, "_"))"
    return [(train_view, split_metadata)]
end

# older 
# function create_data_splits(data_store::DataStore, splitter::ExpandingWindowCV)::Vector{Tuple{SubDataFrame, String}}
#     println("Creating TimeSeriesSplits (Expanding Window)...")
#
#     # Initialize iterator with window_size = nothing (Infinite history)
#     ts_iterator = TimeSeriesSplits(
#         data_store.matches, 
#         splitter.base_seasons, 
#         splitter.target_seasons, 
#         splitter.round_col, 
#         splitter.ordering,
#         nothing # No window size
#     )
#
#     return Vector{Tuple{SubDataFrame, String}}(collect(ts_iterator))
# end
#
# new version 
function create_data_splits(data_store::DataStore, splitter::ExpandingWindowCV)::Vector{Tuple{SubDataFrame, String}}
    println("Creating TimeSeriesSplits (Expanding Window)...")
    
    # Initialize iterator with window_size = nothing (Infinite history)
    ts_iterator = TimeSeriesSplits(
        data_store.matches, 
        splitter.train_seasons,    # CHANGED: base_seasons -> train_seasons
        splitter.test_seasons,     # CHANGED: target_seasons -> test_seasons
        splitter.window_col, 
        splitter.method,           # CHANGED: ordering -> method
        nothing # No window size
    )
    
    return Vector{Tuple{SubDataFrame, String}}(collect(ts_iterator))
end


function create_data_splits(data_store::DataStore, splitter::WindowCV)::Vector{Tuple{SubDataFrame, String}}
    println("Creating TimeSeriesSplits (Sliding Window size=$(splitter.window_size))...")

    # Initialize iterator with specific window_size
    ts_iterator = TimeSeriesSplits(
        data_store.matches, 
        splitter.base_seasons, 
        splitter.target_seasons, 
        splitter.window_col, 
        splitter.ordering,
        splitter.window_size # Pass the constraint
    )

    return Vector{Tuple{SubDataFrame, String}}(collect(ts_iterator))
end


##############################
# --- 3. REWORKED TimeSeriesSplits ---
##############################

struct TimeSeriesSplits
    base_indices::Vector{Int}
    target_rounds_by_season::Dict{String, Vector}
    target_round_sequence::Vector{Tuple{String, Any}}
    original_df::DataFrame 
    window_col::Symbol 
    # New Field: If nothing, it behaves like Expanding. If Number, behaves like Window.
    window_size::Union{Number, Nothing} 
end

# Updated Constructor
function TimeSeriesSplits(df::DataFrame, 
                          base_seasons::AbstractVector, 
                          target_seasons::AbstractVector, 
                          window_col::Symbol, 
                          ordering::Symbol,
                          window_size::Union{Number, Nothing}=nothing) # Default to expanding
    
    base_indices = findall(row -> row.season in base_seasons, eachrow(df))
    
    target_rounds_by_season = Dict{String, Vector}()
    for season in target_seasons
        season_data = filter(row -> row.season == season, df)
        if nrow(season_data) > 0
            target_rounds_by_season[String(season)] = sort(unique(season_data[!, window_col]))
        else
            target_rounds_by_season[String(season)] = []
        end
    end
    
    target_round_sequence = Tuple{String, Any}[]
    
    if ordering == :sequential
        for season in target_seasons
            season_str = String(season)
            for round in target_rounds_by_season[season_str]
                push!(target_round_sequence, (season_str, round))
            end
        end
    elseif ordering == :interleaved
        max_rounds = maximum(length(rounds) for rounds in values(target_rounds_by_season); init=0)
        for round_idx in 1:max_rounds
            for season in target_seasons
                season_str = String(season)
                if round_idx <= length(target_rounds_by_season[season_str])
                    push!(target_round_sequence, (season_str, target_rounds_by_season[season_str][round_idx]))
                end
            end
        end
    else
        error("Unknown ordering: $ordering. Use :sequential or :interleaved")
    end
    
    return TimeSeriesSplits(base_indices, target_rounds_by_season, target_round_sequence, df, window_col, window_size) 
end

# Iterator Interface
Base.length(ts::TimeSeriesSplits) = length(ts.target_round_sequence)

function Base.iterate(ts::TimeSeriesSplits, state=1)
    if state > length(ts.target_round_sequence)
        return nothing
    end
    
    season, round_val = ts.target_round_sequence[state]
    round_info = "$season/Round_$(round_val)"
    
    # 1. Identify rows in the current target season up to the current round
    target_indices = findall(row -> row.season == season && row[ts.window_col] <= round_val, eachrow(ts.original_df))

    # 2. Combine base history with current target history (The "Expanding" Set)
    current_indices = sort(unique(vcat(ts.base_indices, target_indices)))

    # 3. Apply Sliding Window Logic (The "Pruning" Step)
    if !isnothing(ts.window_size)
        # Calculate cutoff value
        # e.g., if Round is 10 and window is 4, we want > 6 (i.e., 7,8,9,10)
        cutoff_val = round_val - ts.window_size
        
        # Filter the current_indices based on the value in window_col
        # Note: This filters both base_indices and target_indices equally based on the window_col value.
        filter!(idx -> ts.original_df[idx, ts.window_col] > cutoff_val, current_indices)
    end

    train_view = view(ts.original_df, current_indices, :)
    
    return ((train_view, round_info), state + 1)
end
