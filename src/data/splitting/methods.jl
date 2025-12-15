# src/data/splitting/methods.jl

using DataFrames

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

# --- 4. CVConfig Split Logic (Unified API) ---

function create_data_splits(data_store, config::CVConfig)::Vector{Tuple{SubDataFrame, SplitMetaData}}
    splits = Vector{Tuple{SubDataFrame, SplitMetaData}}()
    df = data_store.matches
    all_seasons = sort(unique(df.season))
    
    for tourn_id in config.tournament_ids
        tourn_mask = df.tournament_id .== tourn_id
        if !any(tourn_mask); continue; end

        for target_season in config.target_seasons
            target_idx = findfirst(==(target_season), all_seasons)
            if isnothing(target_idx)
                @warn "Target season $target_season not found for tournament $tourn_id. Skipping."
                continue
            end
            
            start_idx = max(1, target_idx - config.history_seasons)
            history_seasons_list = all_seasons[start_idx : target_idx-1]
            
            history_indices = findall(tourn_mask .& (in.(df.season, Ref(history_seasons_list))))
            target_indices = findall(tourn_mask .& (df.season .== target_season))
            
            if isempty(target_indices)
                @warn "No data found for target season $target_season (Tournament $tourn_id)."
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
                
                meta = SplitMetaData(
                    tourn_id,
                    target_season,
                    target_season, 
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

function get_next_matches(ds::DataStore, meta::SplitMetaData, config::CVConfig)::AbstractDataFrame 
    return subset(ds.matches, 
           :tournament_id => ByRow(isequal(meta.tournament_id)),
           :season => ByRow(isequal(meta.target_season)),
           config.dynamics_col => ByRow(isequal(meta.time_step + 1)) 
    )
end
