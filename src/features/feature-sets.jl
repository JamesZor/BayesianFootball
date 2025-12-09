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

    # --- Process Data ---
    matches_df_filtered = dropmissing(data_split, [:home_score, :away_score])
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df_filtered, view=false)

    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)
    matches_df = apply_model_specific_logic(model, matches_df)
    F_data[:matches_df] = matches_df

    # --- DETERMINING THE TIME GROUPING ---
    # LOGIC: Use config.dynamics_col if it exists. 
    # If it is nothing, fallback to config.window_col.
    grouping_col = isnothing(splitter_config.dynamics_col) ? splitter_config.window_col : splitter_config.dynamics_col
    
    # Check if column exists to prevent obscure errors
    if !hasproperty(matches_df, grouping_col)
        error("The time column ':$grouping_col' was not found in the DataFrame. Check your splitter_config.")
    end

    # Group by the Time Column (e.g., :match_week)
    grouped = groupby(matches_df, grouping_col, sort=true)
    F_data[:n_rounds] = length(grouped)

    # Extract Data
    F_data[:round_home_ids] = [ [team_map[name] for name in g.home_team] for g in grouped]
    F_data[:round_away_ids] = [ [team_map[name] for name in g.away_team] for g in grouped]
    F_data[:round_home_goals] = [g.home_score for g in grouped]
    F_data[:round_away_goals] = [g.away_score for g in grouped]

    # Flatten
    F_data[:flat_home_ids] = vcat(F_data[:round_home_ids]...)
    F_data[:flat_away_ids] = vcat(F_data[:round_away_ids]...)
    F_data[:flat_home_goals] = vcat(F_data[:round_home_goals]...)
    F_data[:flat_away_goals] = vcat(F_data[:round_away_goals]...)

    # Time Indices
    time_indices = Int[]
    for (t, round_matches) in enumerate(F_data[:round_home_ids])
        n_matches_in_round = length(round_matches)
        append!(time_indices, fill(t, n_matches_in_round))
    end
    F_data[:time_indices] = time_indices

    return FeatureSet(F_data)
end

# Vector wrapper (remains clean, just passes the config)
function create_features(
    data_splits_vector::Vector{<:Tuple{<:AbstractDataFrame, String}},
    vocabulary::Vocabulary,
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::Vector{Tuple{FeatureSet, String}}

    return [
        (create_features(data, vocabulary, model, splitter_config), meta) 
        for (data, meta) in data_splits_vector
    ]
end

