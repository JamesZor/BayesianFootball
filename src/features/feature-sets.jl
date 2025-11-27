# src/features/feature-sets.jl 

# 1. Define the hook (Default behavior: do nothing)
function apply_model_specific_logic(model::AbstractStaticPoissonModel, df::DataFrame)
    return df
end

# 2. Define the hook for GRW (Behavior: Sort by time)
# Note: You need to ensure AbstractGRWPoissonModel is available here or use a specific concrete type
function apply_model_specific_logic(model::AbstractDynamicPoissonModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end

function create_features(
    data_split::AbstractDataFrame,
    vocabulary::Vocabulary,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureSet

    G = vocabulary.mappings
    F_data = Dict{Symbol, Any}()

    team_map = G[:team_map]::Dict{<:AbstractString, Int}
    n_teams = G[:n_teams]::Int
    F_data[:team_map] = team_map
    F_data[:n_teams] = n_teams

    # --- Process D_i ---
    # REFACTOR 1: Use dropmissing for cleaner syntax
    matches_df_filtered = dropmissing(data_split, [:home_score, :away_score])

    # Filter teams (making a copy)
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df_filtered, view=false)

    # Convert scores
    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)

    # REFACTOR 2: Apply the model-specific hook
    # This will sort the data ONLY if the model is a GRW model
    matches_df = apply_model_specific_logic(model, matches_df)

    F_data[:matches_df] = matches_df

    # --- Build split-specific data (F_i) ---
    # Ensure we respect the sort order when grouping
    grouped = groupby(matches_df, splitter_config.round_col, sort=true)
    F_data[:n_rounds] = length(grouped)

    # Extract team IDs (these are already Int)
    F_data[:round_home_ids] = [ [team_map[name] for name in g.home_team] for g in grouped]
    F_data[:round_away_ids] = [ [team_map[name] for name in g.away_team] for g in grouped]

    # Extract goals (these will now be Int)
    F_data[:round_home_goals] = [g.home_score for g in grouped]
    F_data[:round_away_goals] = [g.away_score for g in grouped]

    # Flatten the vectors (the element types will be preserved as Int)
    F_data[:flat_home_ids] = vcat(F_data[:round_home_ids]...)
    F_data[:flat_away_ids] = vcat(F_data[:round_away_ids]...)
    F_data[:flat_home_goals] = vcat(F_data[:round_home_goals]...) # Should now be Vector{Int}
    F_data[:flat_away_goals] = vcat(F_data[:round_away_goals]...) # Should now be Vector{Int}


    return FeatureSet(F_data)
end



"""
    create_features(data_splits_vector::Vector{Tuple{<:AbstractDataFrame, String}}, vocabulary::Vocabulary, model::AbstractFootballModel)

Applies the feature creation process to each data split in the input vector,
returning a vector of (FeatureSet, metadata) tuples.
"""
function create_features(
    data_splits_vector::Vector{<:Tuple{<:AbstractDataFrame, String}}, # More general Tuple type
    vocabulary::Vocabulary,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::Vector{Tuple{FeatureSet, String}}

    feature_sets_vector = [
        begin
            fs = create_features( # Call the existing single-split method
                data_split_view, # D_i (SubDataFrame or DataFrame)
                vocabulary,      # G
                model,            # M
                splitter_config,
            )
            (fs, split_metadata) # Return tuple (F_i, metadata)
        end
        for (data_split_view, split_metadata) in data_splits_vector
    ]

    return feature_sets_vector
end
