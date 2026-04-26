# src/data/splitting/methods.jl

using DataFrames
using ..TypesInterfaces: FeatureSet

export create_data_splits, get_next_matches

# --- 1. Constructors for the Iterator ---

function TimeSeriesSplits(df::DataFrame, 
                          base_seasons::AbstractVector, 
                          target_seasons::AbstractVector, 
                          window_col::Symbol, 
                          ordering::Symbol,
                          window_size::Union{Number, Nothing}=nothing)
    
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

# --- 2. Iterator Implementation ---

Base.length(ts::TimeSeriesSplits) = length(ts.target_round_sequence)

function Base.iterate(ts::TimeSeriesSplits, state=1)
    if state > length(ts.target_round_sequence)
        return nothing
    end
    
    season, round_val = ts.target_round_sequence[state]
    round_info = "$season/Round_$(round_val)"
    
    target_indices = findall(row -> row.season == season && row[ts.window_col] <= round_val, eachrow(ts.original_df))
    current_indices = sort(unique(vcat(ts.base_indices, target_indices)))

    if !isnothing(ts.window_size)
        cutoff_val = round_val - ts.window_size
        filter!(idx -> ts.original_df[idx, ts.window_col] > cutoff_val, current_indices)
    end

    train_view = view(ts.original_df, current_indices, :)
    
    return ((train_view, round_info), state + 1)
end

# --- 3. High-Level Split Creators ---

function create_data_splits(data_store::DataStore, splitter::StaticSplit)::Vector{Tuple{SubDataFrame, String}}
    println("Creating a single static data split (using view)...")
    row_indices = findall(s -> s in splitter.train_seasons, data_store.matches.season)
    train_view = view(data_store.matches, row_indices, :)
    split_metadata = "static_seasons_$(join(splitter.train_seasons, "_"))"
    return [(train_view, split_metadata)]
end

function create_data_splits(data_store::DataStore, splitter::ExpandingWindowCV)::Vector{Tuple{SubDataFrame, String}}
    println("Creating TimeSeriesSplits (Expanding Window)...")
    ts_iterator = TimeSeriesSplits(
        data_store.matches, 
        splitter.train_seasons,    
        splitter.test_seasons,     
        splitter.window_col, 
        splitter.method,           
        nothing 
    )
    return Vector{Tuple{SubDataFrame, String}}(collect(ts_iterator))
end

function create_data_splits(data_store::DataStore, splitter::WindowCV)::Vector{Tuple{SubDataFrame, String}}
    println("Creating TimeSeriesSplits (Sliding Window size=$(splitter.window_size))...")
    ts_iterator = TimeSeriesSplits(
        data_store.matches, 
        splitter.base_seasons, 
        splitter.target_seasons, 
        splitter.window_col, 
        splitter.ordering,
        splitter.window_size 
    )
    return Vector{Tuple{SubDataFrame, String}}(collect(ts_iterator))
end

# --- 4. Shared Split Logic (Extracted Helper) ---

function _process_tournament_group(df::DataFrame, group_ids::Vector{Int}, config::Union{CVConfig, GroupedCVConfig}, meta_type::Type)
    splits = Vector{Tuple{SubDataFrame, AbstractSplitMetaData}}()
    all_seasons = sort(unique(df.season))
    
    # Check if tournament ID is within the current group
    tourn_mask = in.(df.tournament_id, Ref(group_ids))
    if !any(tourn_mask); return splits; end

    for target_season in config.target_seasons
        target_idx = findfirst(==(target_season), all_seasons)
        if isnothing(target_idx)
            @warn "Target season $target_season not found for tournament group $group_ids. Skipping."
            continue
        end
        
        start_idx = max(1, target_idx - config.history_seasons)
        history_seasons_list = all_seasons[start_idx : target_idx-1]
        
        history_indices = findall(tourn_mask .& (in.(df.season, Ref(history_seasons_list))))
        target_indices = findall(tourn_mask .& (df.season .== target_season))
        
        if isempty(target_indices)
            @warn "No data found for target season $target_season (Tournament group $group_ids)."
            continue
        end
        
        season_dynamics = unique(df[target_indices, config.dynamics_col])
        sort!(season_dynamics)
        
        max_week = maximum(season_dynamics)
        effective_end = config.stop_early ? (max_week - 1) : max_week

        valid_steps = filter(t -> t >= config.warmup_period, season_dynamics)
        if !isnothing(config.end_dynamics)
            filter!(t -> t <= config.end_dynamics, valid_steps)
        end
        filter!(t -> t <= effective_end, valid_steps)
      
        for t in valid_steps
            current_target_indices = filter(idx -> df[idx, config.dynamics_col] <= t, target_indices)
            combined_indices = vcat(history_indices, current_target_indices)
            sort!(combined_indices)
            
            train_view = view(df, combined_indices, :)
            
            # Dispatch the proper MetaData type
            if meta_type === SplitMetaData
                meta = SplitMetaData(
                    group_ids[1], # Old config expects single Int
                    target_season,
                    target_season, 
                    config.history_seasons,
                    t,
                    config.warmup_period
                )
            else
                meta = GroupedSplitMetaData(
                    group_ids,    # New config expects Vector{Int}
                    target_season,
                    target_season, 
                    config.history_seasons,
                    t,
                    config.warmup_period
                )
            end
            push!(splits, (train_view, meta))
        end
    end
    return splits
end

# --- 5. CVConfig Split Wrappers (Unified API via Dispatch) ---

# Legacy API (Fully backward compatible)
function create_data_splits(data_store, config::CVConfig)::Vector{Tuple{SubDataFrame, SplitMetaData}}
    splits = Vector{Tuple{SubDataFrame, SplitMetaData}}()
    for tourn_id in config.tournament_ids
        group_splits = _process_tournament_group(data_store.matches, [tourn_id], config, SplitMetaData)
        for (v, m) in group_splits
            push!(splits, (v, m::SplitMetaData))
        end
    end
    return splits
end

# New Grouped API
function create_data_splits(data_store, config::GroupedCVConfig)::Vector{Tuple{SubDataFrame, GroupedSplitMetaData}}
    splits = Vector{Tuple{SubDataFrame, GroupedSplitMetaData}}()
    for group in config.tournament_groups
        group_splits = _process_tournament_group(data_store.matches, group, config, GroupedSplitMetaData)
        for (v, m) in group_splits
            push!(splits, (v, m::GroupedSplitMetaData))
        end
    end
    return splits
end

# --- 6. Next Matches Helpers ---

# # Legacy helper
# function get_next_matches(ds::DataStore, meta::SplitMetaData, config::CVConfig)::AbstractDataFrame 
#     return subset(ds.matches, 
#            :tournament_id => ByRow(isequal(meta.tournament_id)),
#            :season => ByRow(isequal(meta.target_season)),
#            config.dynamics_col => ByRow(isequal(meta.time_step + 1)) 
#     )
# end
#
# function get_next_matches(ds::DataStore, fs::Tuple{FeatureSet, SplitMetaData}, cvconf::CVConfig)::AbstractDataFrame 
#   return get_next_matches(ds, fs[2], cvconf) 
# end
#
# # New Grouped helper
# function get_next_matches(ds::DataStore, meta::GroupedSplitMetaData, config::GroupedCVConfig)::AbstractDataFrame 
#     return subset(ds.matches, 
#            :tournament_id => ByRow(in(meta.tournament_ids)), # Checks array inclusion
#            :season => ByRow(isequal(meta.target_season)),
#            config.dynamics_col => ByRow(isequal(meta.time_step + 1)) 
#     )
# end
#
# function get_next_matches(ds::DataStore, fs::Tuple{FeatureSet, GroupedSplitMetaData}, cvconf::GroupedCVConfig)::AbstractDataFrame 
#   return get_next_matches(ds, fs[2], cvconf) 
# end
#
# ==============================================================================
# FETCH NEXT MATCHES (Inference Data Retrieval)
# ==============================================================================

# 1. Base Logic for Single Tournament
function get_next_matches(
    ds::Data.DataStore, 
    meta::Data.SplitMetaData, 
    config::Data.CVConfig
)::AbstractDataFrame 
    return subset(ds.matches, 
           :tournament_id => ByRow(isequal(meta.tournament_id)),
           :season => ByRow(isequal(meta.target_season)),
           config.dynamics_col => ByRow(isequal(meta.time_step + 1)) 
    )
end

# 2. Base Logic for Grouped Tournaments
function get_next_matches(
    ds::Data.DataStore, 
    meta::Data.GroupedSplitMetaData, 
    config::Data.GroupedCVConfig
)::AbstractDataFrame 
    return subset(ds.matches, 
           :tournament_id => ByRow(in(meta.tournament_ids)), 
           :season => ByRow(isequal(meta.target_season)),
           config.dynamics_col => ByRow(isequal(meta.time_step + 1)) 
    )
end

# 3. The "Catch-All" Tuple Wrapper (Replaces all the redundant functions!)
# This will automatically work if you pass it `boundaries_with_meta[1]` 
# OR if you pass it `feature_collection[1]`.
function get_next_matches(
    ds::Data.DataStore, 
    fold_tuple::Tuple{Any, <:Data.AbstractSplitMetaData}, 
    config::Union{Data.CVConfig, Data.GroupedCVConfig}
)::AbstractDataFrame 
    # fold_tuple[2] is always the MetaData object
    return get_next_matches(ds, fold_tuple[2], config) 
end


# -----
export create_id_boundaries # Export the new API

# 1. The ID-based internal helper

function _process_tournament_group_ids(
    df::DataFrame, 
    group_ids::Vector{Int}, 
    config::Union{Data.CVConfig, Data.GroupedCVConfig}, 
    meta_type::Type
)
    splits = Vector{Tuple{SplitBoundary, Data.AbstractSplitMetaData}}()
    all_seasons = sort(unique(df.season))
    
    tourn_mask = in.(df.tournament_id, Ref(group_ids))
    if !any(tourn_mask); return splits; end

    for target_season in config.target_seasons
        target_idx = findfirst(==(target_season), all_seasons)
        if isnothing(target_idx)
            @warn "Target season $target_season not found for tournament group $group_ids. Skipping."
            continue
        end
        
        start_idx = max(1, target_idx - config.history_seasons)
        history_seasons_list = all_seasons[start_idx : target_idx-1]
        
        # --- Extract universal match_ids ---
        history_ids = df[tourn_mask .& in.(df.season, Ref(history_seasons_list)), :match_id]
        
        target_pool = df[tourn_mask .& (df.season .== target_season), [:match_id, config.dynamics_col]]
        
        if isempty(target_pool)
            @warn "No data found for target season $target_season (Tournament group $group_ids)."
            continue
        end
        
        season_dynamics = unique(target_pool[!, config.dynamics_col])
        sort!(season_dynamics)
        
        max_week = maximum(season_dynamics)
        effective_end = config.stop_early ? (max_week - 1) : max_week

        valid_steps = filter(t -> t >= config.warmup_period, season_dynamics)
        if !isnothing(config.end_dynamics)
            filter!(t -> t <= config.end_dynamics, valid_steps)
        end
        filter!(t -> t <= effective_end, valid_steps)
        # -----------------------------------------------------------------
        # CREATE FOLDS
        # -----------------------------------------------------------------
        fold_counter = 1
        
        # --- 1. Inject the Baseline Fold (History Only, t=0) ---
        # ONLY inject this if we actually have history!
        if length(history_ids) > 0
            boundary_zero = SplitBoundary(
                fold_counter,
                0, # Target step 0 (Baseline)
                copy(history_ids),
                Int[] # No target matches yet!
            )
            
            if meta_type === Data.SplitMetaData
                meta_zero = Data.SplitMetaData(group_ids[1], target_season, target_season, config.history_seasons, 0, config.warmup_period)
            else
                meta_zero = Data.GroupedSplitMetaData(group_ids, target_season, target_season, config.history_seasons, 0, config.warmup_period)
            end
            
            push!(splits, (boundary_zero, meta_zero))
            fold_counter += 1
        end

        # --- 2. Inject the Dynamic Folds (Walk Forward) ---
        for t in valid_steps
            # Get target match IDs up to step t
            current_target_ids = target_pool[target_pool[!, config.dynamics_col] .<= t, :match_id]
            
            boundary = SplitBoundary(
                fold_counter,
                t,
                copy(history_ids),       # frozen history
                copy(current_target_ids) # expanding window
            )
            
            if meta_type === Data.SplitMetaData
                meta = Data.SplitMetaData(group_ids[1], target_season, target_season, config.history_seasons, t, config.warmup_period)
            else
                meta = Data.GroupedSplitMetaData(group_ids, target_season, target_season, config.history_seasons, t, config.warmup_period)
            end
            
            push!(splits, (boundary, meta))
            fold_counter += 1
        end
      end
    return splits
end 



# 2. The Public APIs
function create_id_boundaries(data_store, config::CVConfig)
    splits = Vector{Tuple{SplitBoundary, SplitMetaData}}()
    for tourn_id in config.tournament_ids
        group_splits = _process_tournament_group_ids(data_store.matches, [tourn_id], config, SplitMetaData)
        for (b, m) in group_splits
            push!(splits, (b, m::SplitMetaData))
        end
    end
    return splits
end

function create_id_boundaries(data_store, config::GroupedCVConfig)
    return _process_tournament_group_ids(data_store.matches, config.tournament_ids, config, GroupedSplitMetaData)
end
