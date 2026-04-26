# src/features/feature-sets.jl

# -------------------------------------------------------------------------
# 1. The Core Feature Extractor (For CVConfig / GroupedCVConfig)
# -------------------------------------------------------------------------
function create_features(
    data_split::AbstractDataFrame,
    model::AbstractFootballModel, # <-- CHANGED: Now works for ALL models
    splitter_config::Union{CVConfig, GroupedCVConfig}
)::FeatureSet

    F_data = Dict{Symbol, Any}()

    # 1. Build Base Mappings & Clean Data
    merge!(F_data, build_mappings(data_split, model))
    team_map = F_data[:team_map]::Dict{<:AbstractString, Int}
    
    needed_cols = target_columns(model)
    matches_df = dropmissing(data_split, needed_cols)
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df, view=false)
    
    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)
    matches_df = apply_model_specific_logic(model, matches_df)
    F_data[:matches_df] = matches_df

    # 2. Time Grouping Logic (Dual Grouping)
    grouping_col = splitter_config.dynamics_col
    if !hasproperty(matches_df, grouping_col)
        error("The time column ':$grouping_col' was not found.")
    end

    history_mask = .!in.(matches_df.season, Ref(splitter_config.target_seasons))
    history_grouped = groupby(matches_df[history_mask, :], :season, sort=true)
    target_grouped  = groupby(matches_df[.!history_mask, :], grouping_col, sort=true)
    
    all_groups = vcat(collect(history_grouped), collect(target_grouped))
    
    F_data[:n_rounds] = length(all_groups)
    F_data[:n_history_steps] = length(history_grouped)
    F_data[:n_target_steps] = length(target_grouped)

    # 3. ---> DYNAMIC FEATURE PIPELINE <---
    features_to_extract = required_features(model)
    
    for f in features_to_extract
        add_feature!(F_data, Val(f), all_groups, team_map)
    end

    # 4. Target Extraction (Goals vs Funnel)
    extract_targets!(F_data, model, all_groups)

    return FeatureSet(F_data)
end

# -------------------------------------------------------------------------
# 2. The Legacy Splitter Fallback (Optional but recommended)
# -------------------------------------------------------------------------
# If you ever run tests using StaticSplit or WindowCV (which don't have dual grouping)
function create_features(
    data_split::AbstractDataFrame,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter # Catches anything that ISN'T CVConfig
)::FeatureSet

    F_data = Dict{Symbol, Any}()

    # 1. Build Base Mappings & Clean Data
    merge!(F_data, build_mappings(data_split, model))
    team_map = F_data[:team_map]::Dict{<:AbstractString, Int}
    
    needed_cols = target_columns(model)
    matches_df = dropmissing(data_split, needed_cols)
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df, view=false)
    
    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)
    matches_df = apply_model_specific_logic(model, matches_df)
    F_data[:matches_df] = matches_df

    # 2. Single Grouping Logic
    grouping_col = isnothing(splitter_config.dynamics_col) ? splitter_config.window_col : splitter_config.dynamics_col
    if !hasproperty(matches_df, grouping_col)
        error("The time column ':$grouping_col' was not found.")
    end
    all_groups = groupby(matches_df, grouping_col, sort=true)
    F_data[:n_rounds] = length(all_groups)

    # 3. ---> DYNAMIC FEATURE PIPELINE <---
    features_to_extract = required_features(model)
    for f in features_to_extract
        add_feature!(F_data, Val(f), all_groups, team_map)
    end

    # 4. Target Extraction
    extract_targets!(F_data, model, all_groups)

    return FeatureSet(F_data)
end

# -------------------------------------------------------------------------
# 3. The Vector Wrapper for FeatureCollections
# -------------------------------------------------------------------------
function create_features(
    data_splits::Vector{<:Tuple{<:AbstractDataFrame, M}},
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureCollection{M} where M  

    raw_vector = [
        (create_features(data, model, splitter_config), meta) 
        for (data, meta) in data_splits
    ]

    return FeatureCollection(raw_vector)
end




# ---- updated 
# ==============================================================================
# RELATIONAL FEATURE BUILDER (DataStore & SplitBoundary Architecture)
# ==============================================================================

# 1. The Macro Loop (Vector Dispatch)
function create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary, <:Any}}, 
    ds::Data.DataStore, 
    model::AbstractFootballModel,
    dynamics_col::Symbol = :match_month
)
    raw_vector = [
        (create_features(boundary, ds, model, dynamics_col), meta) 
        for (boundary, meta) in splits
    ]
    return FeatureCollection(raw_vector)
end

# 2. The Micro Builder (Single Boundary Dispatch)
function create_features(
    boundary::Data.SplitBoundary, 
    ds::Data.DataStore, 
    model::AbstractFootballModel,
    dynamics_col::Symbol
)
    F_data = Dict{Symbol, Any}()
    
    # 1. COMBINE IDs for the full sequence (History + Target)
    all_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
    
    # 2. Extract just the matches for this specific fold
    matches_df = subset(ds.matches, :match_id => ByRow(id -> id in all_ids))
    
    # 3. BUILD VOCABULARY (Strings -> Integers)
    all_teams = unique(vcat(matches_df.home_team, matches_df.away_team))
    team_map = Dict(name => i for (i, name) in enumerate(sort(all_teams)))
    
    F_data[:n_teams] = length(team_map)
    F_data[:team_map] = team_map

    # 4. GENERATE TIME INDICES 
    history_df = subset(matches_df, :match_id => ByRow(id -> id in boundary.history_match_ids))
    target_df  = subset(matches_df, :match_id => ByRow(id -> id in boundary.target_match_ids))
    
    history_groups = groupby(history_df, :season, sort=true)
    target_groups  = groupby(target_df, dynamics_col, sort=true)
    
    time_indices = Int[]
    t_idx = 1
    for g in history_groups
        append!(time_indices, fill(t_idx, nrow(g)))
        t_idx += 1
    end
    n_history = length(history_groups)
    
    for g in target_groups
        append!(time_indices, fill(t_idx, nrow(g)))
        t_idx += 1
    end
    n_target = length(target_groups)
    
    ordered_ids = Int.(vcat(history_df.match_id, target_df.match_id))
    
    F_data[:time_indices] = time_indices
    F_data[:n_history_steps] = n_history
    F_data[:n_target_steps] = n_target
    F_data[:n_rounds] = n_history + n_target

    # 5. DYNAMIC PIPELINE
    for trait in required_features(model)
        add_feature!(F_data, Val(trait), ordered_ids, team_map, ds)
    end

    # Return using your package's existing FeatureSet struct!
    return FeatureSet(F_data)
end
