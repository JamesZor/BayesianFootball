# src/features/feature-sets.jl 


# """
#     FeatureSet (Your 'F_i')
#
# A flexible container for the *data* of a specific split (D_i),
# built using a global Vocabulary (G).
# """
# struct FeatureSet
#     data::Dict{Symbol, Any}
# end
#

#
# # --- Private Helper Functions ---
# function _add_global_round_column(matches_df::DataFrame)
#     df = sort(matches_df, :match_date)
#     num_matches = nrow(df)
#     global_rounds = Vector{Int}(undef, num_matches)
#     global_round_counter = 1
#     teams_in_current_round = Set{String}()
#
#     for (i, row) in enumerate(eachrow(df))
#         home_team, away_team = row.home_team, row.away_team
#         if home_team in teams_in_current_round || away_team in teams_in_current_round
#             global_round_counter += 1
#             empty!(teams_in_current_round)
#         end
#         global_rounds[i] = global_round_counter
#         push!(teams_in_current_round, home_team, away_team)
#     end
#
#     df.global_round = global_rounds
#     return df
# end
#
# function _add_global_round_column(matches_df::AbstractDataFrame) # Accept AbstractDataFrame
#     # Make a copy if you intend to modify, otherwise views don't allow adding columns
#     df = copy(matches_df) # IMPORTANT: Work on a copy if adding columns
#     sort!(df, :match_date) # Sort the copy
#     num_matches = nrow(df)
#     global_rounds = Vector{Int}(undef, num_matches)
#     global_round_counter = 1
#     teams_in_current_round = Set{String}()
#
#     for (i, row) in enumerate(eachrow(df))
#         home_team, away_team = row.home_team, row.away_team
#         if home_team in teams_in_current_round || away_team in teams_in_current_round #
#             global_round_counter += 1
#             empty!(teams_in_current_round)
#         end
#         global_rounds[i] = global_round_counter
#         push!(teams_in_current_round, home_team, away_team)
#     end
#
#     df.global_round = global_rounds
#     return df
# end
#
#

# --- Main API Functions ---

"""
    create_features(data_split::DataFrame, vocabulary::Vocabulary, model::AbstractFootballModel)

Creates a FeatureSet (F_i) for a specific data split (D_i)
using the pre-computed global Vocabulary (G). This is your `f_i: D_i x G x M -> F_i`.
"""
function create_features(
    data_split::DataFrame, 
    vocabulary::Vocabulary, 
    model::AbstractFootballModel
)::FeatureSet
    
    G = vocabulary.mappings
    F_data = Dict{Symbol, Any}() # This will hold the data for the FeatureSet
    
    # --- 1. Copy global info from G to F_i, asserting types for stability ---
    team_map = G[:team_map]::Dict{<:AbstractString, Int}
    n_teams = G[:n_teams]::Int
    F_data[:team_map] = team_map
    F_data[:n_teams] = n_teams

    # --- 2. Process D_i (the data split) ---
    matches_df = dropmissing(data_split, [:home_score, :away_score])
    
    # Filter out matches with teams not present in the global vocabulary
    filter!(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df)
    
    matches_df = _add_global_round_column(matches_df)
    F_data[:matches_df] = matches_df

    # --- 3. Build split-specific data (F_i) ---
    grouped = groupby(matches_df, :global_round)
    F_data[:n_rounds] = length(grouped)
    
    F_data[:round_home_ids] = [ [team_map[name] for name in g.home_team] for g in grouped]
    F_data[:round_away_ids] = [ [team_map[name] for name in g.away_team] for g in grouped]
    F_data[:round_home_goals] = [g.home_score for g in grouped]
    F_data[:round_away_goals] = [g.away_score for g in grouped]
    
    F_data[:flat_home_ids] = vcat(F_data[:round_home_ids]...)
    F_data[:flat_away_ids] = vcat(F_data[:round_away_ids]...)
    F_data[:flat_home_goals] = vcat(F_data[:round_home_goals]...)
    F_data[:flat_away_goals] = vcat(F_data[:round_away_goals]...)

    # --- 4. Add model-specific data ---
    # This is where you would add logic for a new model type, e.g.:
    # if model isa HierarchicalModel 
    #   league_map = G[:league_map]::Dict{String, Int}
    #   F_data[:flat_league_ids] = [league_map[r.tournament_slug] for r in eachrow(matches_df)]
    # end
    
    return FeatureSet(F_data)
end

"""
    create_features(data_split::AbstractDataFrame, vocabulary::Vocabulary, model::AbstractFootballModel) # <-- CHANGE HERE

Creates a FeatureSet (F_i) for a specific data split (D_i)
using the pre-computed global Vocabulary (G). This is your `f_i: D_i x G x M -> F_i`.
"""
function create_features(
    data_split::AbstractDataFrame, # <-- CHANGE HERE to AbstractDataFrame
    vocabulary::Vocabulary,
    model::AbstractFootballModel
)::FeatureSet
    
    G = vocabulary.mappings
    F_data = Dict{Symbol, Any}()
    
    team_map = G[:team_map]::Dict{<:AbstractString, Int}
    n_teams = G[:n_teams]::Int
    F_data[:team_map] = team_map #
    F_data[:n_teams] = n_teams #

    # --- Process D_i (the data split) ---
    # IMPORTANT: Since _add_global_round_column now makes a copy,
    # we don't need to copy `data_split` here if that's the only modification.
    # However, be mindful if other operations modify it in place.
    matches_df_with_missing = data_split # Work directly with the view/dataframe

    # Drop missings *after* potential copying if needed, or work with views carefully
    matches_df_filtered = filter(row -> !ismissing(row.home_score) && !ismissing(row.away_score), matches_df_with_missing)

    # Filter out matches with teams not present in the global vocabulary
    # Use view=true for efficiency if matches_df_filtered is already a SubDataFrame or you don't need a copy
    matches_df_teams_filtered = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), matches_df_filtered, view=!(matches_df_filtered isa SubDataFrame))

    # Add global round (this function now returns a *new* DataFrame copy)
    matches_df = _add_global_round_column(matches_df_teams_filtered)
    F_data[:matches_df] = matches_df # Store the processed DataFrame

    # --- Build split-specific data (F_i) ---
    grouped = groupby(matches_df, :global_round)
    F_data[:n_rounds] = length(grouped)
   
    F_data[:round_home_ids] = [ [team_map[name] for name in g.home_team] for g in grouped] #
    F_data[:round_away_ids] = [ [team_map[name] for name in g.away_team] for g in grouped] #
    F_data[:round_home_goals] = [g.home_score for g in grouped] #
    F_data[:round_away_goals] = [g.away_score for g in grouped] #
    
    F_data[:flat_home_ids] = vcat(F_data[:round_home_ids]...) #
    F_data[:flat_away_ids] = vcat(F_data[:round_away_ids]...) #
    F_data[:flat_home_goals] = vcat(F_data[:round_home_goals]...) #
    F_data[:flat_away_goals] = vcat(F_data[:round_away_goals]...) #

    # --- Add model-specific data ---
    # ... (no changes needed here)

    return FeatureSet(F_data)
end
