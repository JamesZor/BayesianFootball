# src/data/splitting.jl

using DataFrames
using Base.Iterators: collect

export AbstractSplitter, StaticSplit, ExpandingWindowCV, create_data_splits
# (Keep TimeSeriesSplits struct and constructor as they are)

# --- 1. SPLITTER TYPES (Unchanged) ---
abstract type AbstractSplitter end
struct StaticSplit <: AbstractSplitter
    train_seasons::Vector{String}
end
struct ExpandingWindowCV <: AbstractSplitter
    base_seasons::Vector{String}
    target_seasons::Vector{String}
    round_col::Symbol
    ordering::Symbol
end

# --- 2. REVISED UNIFIED API FUNCTION ---

"""
    create_data_splits(data_store::DataStore, splitter::AbstractSplitter)

Returns a Vector containing (SubDataFrame, String) tuples, where each tuple
represents a data split (D_i) and its associated metadata.
Uses views (SubDataFrame) for memory efficiency.
"""
function create_data_splits(data_store::DataStore, splitter::StaticSplit)::Vector{Tuple{SubDataFrame, String}}
    println("Creating a single static data split (using view)...")
    
    # Find row indices matching the seasons
    row_indices = findall(s -> s in splitter.train_seasons, data_store.matches.season)
    
    # Create a view of the original DataFrame
    train_view = view(data_store.matches, row_indices, :)
    
    split_metadata = "static_seasons_$(join(splitter.train_seasons, "_"))"
    
    # Return a Vector containing the single split
    return [(train_view, split_metadata)]
end

function create_data_splits(data_store::DataStore, splitter::ExpandingWindowCV)::Vector{Tuple{SubDataFrame, String}}
    println("Creating TimeSeriesSplits (expanding window) iterator and collecting results...")
    
    # Create the underlying iterator (which yields views)
    ts_iterator = TimeSeriesSplits(
        data_store.matches, 
        splitter.base_seasons, 
        splitter.target_seasons, 
        splitter.round_col, 
        splitter.ordering
    )
    
    # Collect the results into the desired Vector format
    # The iterator yields (SubDataFrame, String) already, matching our target type
    collected_splits = collect(ts_iterator)
    
    # Ensure the type matches explicitly, though `collect` should handle it.
    return Vector{Tuple{SubDataFrame, String}}(collected_splits)
end


# --- 3. EXISTING TimeSeriesSplits CODE (Needs slight modification in `iterate`) ---
struct TimeSeriesSplits
    base_indices::Vector{Int}
    target_rounds_by_season::Dict{String, Vector}
    target_round_sequence::Vector{Tuple{String, Any}}
    original_df::DataFrame 
    round_col::Symbol 
end

function TimeSeriesSplits(df::DataFrame, 
                          base_seasons::AbstractVector, 
                          target_seasons::AbstractVector, 
                          round_col::Symbol, 
                          ordering::Symbol)
     base_indices = findall(row -> row.season in base_seasons, eachrow(df))
    
    target_rounds_by_season = Dict{String, Vector}()
    for season in target_seasons
        season_data = filter(row -> row.season == season, df)
        if nrow(season_data) > 0
            target_rounds_by_season[String(season)] = sort(unique(season_data[!, round_col]))
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
    
    # Pass the original df to the struct
    return TimeSeriesSplits(base_indices, target_rounds_by_season, target_round_sequence, df, round_col) 
end

# Iterator Interface
Base.length(ts::TimeSeriesSplits) = length(ts.target_round_sequence)

function Base.iterate(ts::TimeSeriesSplits, state=1)
    if state > length(ts.target_round_sequence)
        return nothing
    end
    
    season, round_val = ts.target_round_sequence[state] # Renamed `round` to `round_val` to avoid conflict
    round_info = "$season/Round_$(round_val)"
    
    # Find indices using the stored round_col
    # Use the correctly passed `ts.round_col` field
    target_indices = findall(row -> row.season == season && row[ts.round_col] <= round_val, eachrow(ts.original_df))

    current_indices = sort(unique(vcat(ts.base_indices, target_indices)))
    train_view = view(ts.original_df, current_indices, :)
    
    return ((train_view, round_info), state + 1)
end
