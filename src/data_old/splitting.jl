# src/data/splitting.jl

using DataFrames
using Base: collect

export AbstractSplitter, StaticSplit, ExpandingWindowCV, WindowCV, create_data_splits
export AbstractSplitter, AbstractSplitMetaData, SplitMetaData, CVConfig, create_data_splits

# --- 1. METADATA STRUCTURES ---

abstract type AbstractSplitMetaData end

"""
    SplitMetaData
Contains the context for a specific data split (fold).
Replaces the old String metadata to allow programmatic access downstream.
"""
struct SplitMetaData <: AbstractSplitMetaData
    tournament_id::Int
    train_season::String       # The primary target season (e.g. "20/21")
    target_season::String      # Usually same as train_season in expanding window
    history_depth::Int         # Number of historical seasons included
    time_step::Int             # The dynamics index (e.g. match_week 5)
    warmup_period::Int         # The starting dynamics index
end

# Helpful for debugging/logging
function Base.show(io::IO, meta::SplitMetaData)
    print(io, "Split(Tourn: $(meta.tournament_id), Season: $(meta.train_season), Week: $(meta.time_step), Hist: $(meta.history_depth))")
end


# --- 2. CONFIGURATION (THE "RECIPE") ---

abstract type AbstractSplitter end
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


"""
    CVConfig
Configuration for creating cross-validation splits across multiple tournaments and seasons.

# Fields
- `tournament_ids`: List of tournament IDs to process (e.g., [1, 56]). Splits are generated for each independently.
- `target_seasons`: The seasons to perform the expanding window on (e.g., ["20/21", "21/22"]).
- `history_seasons`: How many previous seasons to append to the training set (e.g., 1 implies adding "19/20" when target is "20/21").
- `dynamics_col`: Column used for time evolution (e.g., :match_week).
- `warmup_period`: The starting index for `dynamics_col` (inclusive).
- `end_dynamics`: (Optional) The final index for `dynamics_col`. If nothing, runs to the end of data.
"""
Base.@kwdef struct CVConfig <: AbstractSplitter
    # Filtering
    tournament_ids::Vector{Int} = [1] 
    
    # Season Logic
    target_seasons::Vector{String}
    history_seasons::Int = 0
    
    # Dynamics
    dynamics_col::Symbol = :match_week
    warmup_period::Int = 5
    
    # Future extensibility (not used yet, but good to have)
    # window_type::Symbol = :expanding 

  # Stopping Logic
    end_dynamics::Union{Int, Nothing} = nothing # Explicit override (e.g., stop at week 30)
    stop_early::Bool = false                    # If true, auto-stops at (Max - 1)
end


# --- 3. CORE LOGIC ---

"""
    create_data_splits(data_store, config::CVConfig)

Generates a vector of (DataFrameView, SplitMetaData) tuples based on the configuration.
Automatically handles history lookup and tournament filtering.
"""
function create_data_splits(data_store, config::CVConfig)::Vector{Tuple{SubDataFrame, SplitMetaData}}
    splits = Vector{Tuple{SubDataFrame, SplitMetaData}}()
    
    df = data_store.matches
    
    # 1. Pre-calculate sorted seasons for history resolution
    # Assumes "YY/YY" format where lexicographical sort works (e.g. "20/21" < "21/22")
    all_seasons = sort(unique(df.season))
    
    for tourn_id in config.tournament_ids
        # Get indices for this tournament to avoid scanning the whole DF constantly
        tourn_mask = df.tournament_id .== tourn_id
        
        # Optimization: If tournament has no data, skip
        if !any(tourn_mask); continue; end

        for target_season in config.target_seasons
            # --- A. Resolve History ---
            # Find where the target season sits in the full timeline
            target_idx = findfirst(==(target_season), all_seasons)
            
            if isnothing(target_idx)
                @warn "Target season $target_season not found for tournament $tourn_id. Skipping."
                continue
            end
            
            # Determine range of seasons to include
            # Start index is target minus history depth (clamped to 1)
            start_idx = max(1, target_idx - config.history_seasons)
            
            # Seasons to include: History + Target
            # History seasons are treated as "static" (full data included)
            history_seasons_list = all_seasons[start_idx : target_idx-1]
            
            # --- B. Get Indices ---
            
            # 1. History Indices (Static)
            # All rows belonging to the history seasons for this tournament
            history_indices = findall(
                tourn_mask .& (in.(df.season, Ref(history_seasons_list)))
            )
            
            # 2. Target Indices (Dynamic)
            # All rows belonging to the target season for this tournament
            target_indices = findall(
                tourn_mask .& (df.season .== target_season)
            )
            
            if isempty(target_indices)
                @warn "No data found for target season $target_season (Tournament $tourn_id)."
                continue
            end
            
            # --- C. Iterate Dynamics (Expanding Window) ---
            # Extract unique time steps from the target season
            # season_dynamics = unique(df[target_indices, config.dynamics_col])
            # sort!(season_dynamics)
            #
            # # Filter valid steps based on warmup/end
            # valid_steps = filter(t -> t >= config.warmup_period, season_dynamics)
            # if !isnothing(config.end_dynamics)
            #     filter!(t -> t <= config.end_dynamics, valid_steps)
            # end
      # --- 
      # --- C. Iterate Dynamics (Expanding Window) ---
            # Extract unique time steps from the target season
            season_dynamics = unique(df[target_indices, config.dynamics_col])
            sort!(season_dynamics)
            
            # 1. Determine the effective end of the season
            max_week = maximum(season_dynamics)
            
            # 2. Apply "Stop Early" logic (Backtesting Mode)
            # If we are backtesting, we often stop 1 week before the end 
            # so the last split has a 'future' week to predict within the dataset.
            effective_end = config.stop_early ? (max_week - 1) : max_week

            # 3. Filter valid steps
            # Start at warmup, and don't go past explicit end_dynamics OR effective_end
            valid_steps = filter(t -> t >= config.warmup_period, season_dynamics)
            
            # Apply explicit end limit if provided
            if !isnothing(config.end_dynamics)
                filter!(t -> t <= config.end_dynamics, valid_steps)
            end
            
            # Apply the calculated effective end (for stop_early)
            filter!(t -> t <= effective_end, valid_steps)
           
      # ---
            for t in valid_steps
                # For the target season, include only rows up to time t
                # We reuse the pre-calculated target_indices and filter them by value
                # (Note: direct array access is fast here)
                current_target_indices = filter(idx -> df[idx, config.dynamics_col] <= t, target_indices)
                
                # Combine History + Current Window
                combined_indices = vcat(history_indices, current_target_indices)
                
                # Sort indices for better memory access pattern in the View
                sort!(combined_indices)
                
                # Create the View
                train_view = view(df, combined_indices, :)
                
                # Create Metadata
                meta = SplitMetaData(
                    tourn_id,
                    target_season,
                    target_season, # target is same as train here
                    config.history_seasons,
                    t,
                    config.warmup_period
                )
                
                push!(splits, (train_view, meta))
            end
        end
    end
    
    return splits
end


export get_next_matches

function get_next_matches(ds::DataStore, meta::SplitMetaData, config::CVConfig)::AbstractDataFrame 

    # 1. Filter by Tournament & Season
    oos_df = subset( ds.matches, 
           :tournament_id => ByRow(isequal(meta.tournament_id)),
           :season => ByRow(isequal(meta.target_season)),
           config.dynamics_col => ByRow(isequal( meta.time_step + 1 )) 
           )

    return oos_df
end

