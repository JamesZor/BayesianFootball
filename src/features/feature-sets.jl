# src/features/feature-sets.jl 

# 1. Define the hook (Default behavior: do nothing)

function apply_model_specific_logic(model::AbstractStaticPoissonModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticDixonColesModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticMixCopulaModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticMVPLNModel, df::DataFrame)
    return df
end

function apply_model_specific_logic(model::AbstractStaticBivariatePoissonModel, df::DataFrame)
    return df
end

# 2. Define the hook for GRW (Behavior: Sort by time)
function apply_model_specific_logic(model::AbstractDynamicPoissonModel, df::DataFrame)
    # Sort by season and date to ensure time flows forward
    return sort(df, [:season, :match_date])
end

"""
    build_mappings(df, model)

Internal helper to create the necessary ID maps (e.g. Team A -> 1)
based specifically on the data present in `df`.
"""
function build_mappings(df::AbstractDataFrame, model::AbstractFootballModel)
    keys_needed = required_mapping_keys(model)
    mappings = Dict{Symbol, Any}()

    # --- Team Mapping Factory ---
    if :team_map in keys_needed || :n_teams in keys_needed
        present_teams = Set{String}()
        
        if hasproperty(df, :home_team)
            union!(present_teams, df.home_team)
        end
        if hasproperty(df, :away_team)
            union!(present_teams, df.away_team)
        end

        # Sort for deterministic ordering (Crucial for reproducibility)
        sorted_teams = sort(collect(present_teams))
        
        # Create dense map (1..N)
        team_map = Dict(t => i for (i, t) in enumerate(sorted_teams))
        
        mappings[:team_map] = team_map
        mappings[:n_teams] = length(sorted_teams)
    end

    # --- League/Tournament Factory (Example extension) ---
    if :league_map in keys_needed || :n_leagues in keys_needed
        if hasproperty(df, :tournament_slug)
            leagues = unique(df.tournament_slug)
            mappings[:league_map] = Dict(l => i for (i, l) in enumerate(leagues))
            mappings[:n_leagues] = length(leagues)
        end
    end

    return mappings
end

function create_features(
    data_split::AbstractDataFrame,
    # vocabulary::Vocabulary,  <-- REMOVED
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureSet

    # 1. Initialize data dictionary
    F_data = Dict{Symbol, Any}()

    # 2. Build Mappings internally based on THIS split
    mappings = build_mappings(data_split, model)
    merge!(F_data, mappings) # Store mappings directly in FeatureSet

    # Retrieve map for processing
    team_map = F_data[:team_map]::Dict{<:AbstractString, Int}
    
    # --- Process Data ---
    matches_df_filtered = dropmissing(data_split, [:home_score, :away_score])
    # Filter to ensure we only keep rows where teams are in our map (redundant here but safe)
    matches_df = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df_filtered, view=false)

    matches_df.home_score = Int.(matches_df.home_score)
    matches_df.away_score = Int.(matches_df.away_score)
    matches_df = apply_model_specific_logic(model, matches_df)
    
    # Store the processed DF in the FeatureSet for later reference (e.g. in extract_parameters)
    F_data[:matches_df] = matches_df

    # --- DETERMINING THE TIME GROUPING ---
    grouping_col = isnothing(splitter_config.dynamics_col) ?
                   splitter_config.window_col : splitter_config.dynamics_col
    
    if !hasproperty(matches_df, grouping_col)
        error("The time column ':$grouping_col' was not found in the DataFrame.")
    end

    # Group by the Time Column
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

function create_features(
    data_splits::Vector{<:Tuple{<:AbstractDataFrame, M}},
    model::AbstractFootballModel,
    splitter_config::AbstractSplitter
)::FeatureCollection{M} where M  # <--- Update Return Type

    # Generate the vector as before
    raw_vector = [
        (
            create_features(data, model, splitter_config), 
            meta
        ) 
        for (data, meta) in data_splits
    ]

    # Wrap it
    return FeatureCollection(raw_vector)
end



