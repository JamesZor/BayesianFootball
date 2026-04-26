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
# -------------------------------------------------------------------------
# NEW RELATIONAL ARCHITECTURE (ID-Based)
# -------------------------------------------------------------------------

# The Macro Loop (Vector Dispatch)
function create_features(
    splits::Vector{<:Tuple{Data.SplitBoundary, <:Any}}, 
    ds::Data.DataStore, 
    model::AbstractFootballModel
)
    raw_vector = [
        (create_features(boundary, ds, model), meta) 
        for (boundary, meta) in splits
    ]

    # Reusing your existing FeatureCollection wrapper
    return FeatureCollection(raw_vector) 
end

# The Micro Builder (Single Boundary Dispatch)
function create_features(
    boundary::Data.SplitBoundary, 
    ds::Data.DataStore, 
    model::AbstractFootballModel
)
    F_data = Dict{Symbol, Any}()
    
    F_data[:dynamics_step] = boundary.target_step
    F_data[:n_history_matches] = length(boundary.history_match_ids)
    F_data[:n_target_matches] = length(boundary.target_match_ids)

    # Dynamic pipeline
    for trait in required_features(model)
        add_feature!(F_data, Val(trait), boundary, ds)
    end

    return FeatureSet(F_data)
end
