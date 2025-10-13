"""
This module is responsible for feature engineering. It takes a DataStore
and transforms the raw data into a format suitable for modeling.
"""
module Features

using DataFrames
using ..Data # Use the Data module from the parent BayesianFootball module

export create_features

# --- Constants ---

# Define the columns from the raw matches data that are needed for feature engineering
const MATCHES_FEATURE_COLS = [
    :match_id,
    :season_id,
    :match_date,
    :home_team,
    :away_team,
    :home_score,
    :away_score
]

# --- Structs ---
#
# Define a struct to hold the engineered features
struct FeatureSet
    data::DataFrame
    team_to_id::Dict{String, Int}
    id_to_team::Dict{Int, String}
    n_teams::Int
end

# --- Private Helper Functions ---

"""
    _add_global_round_column!(matches_df::DataFrame)

Adds a `:global_round` column in-place to the provided DataFrame.
This is a private helper function intended to be used on a copy of the data.
"""
function _add_global_round_column!(matches_df::DataFrame)
    sort!(matches_df, :match_date)
    num_matches = nrow(matches_df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(matches_df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end
    
    matches_df.global_round = global_rounds
    return matches_df
end


# --- Public API ---

"""
    create_features(data_store::DataStore)

Takes a DataStore object and creates features for modeling.

This function creates a copy of the matches data and then adds new
feature columns, such as team IDs and a global round counter.
"""
function create_features(data_store::Data.DataStore)
    # Select only the necessary columns and create a copy to work with.
    matches_df = select(data_store.matches, MATCHES_FEATURE_COLS; copycols=true)

    # --- Feature Pipeline ---
    # 1. Add the global round column
    _add_global_round_column!(matches_df)

    # 2. Create team ID mappings
    home_teams = unique(matches_df.home_team)
    away_teams = unique(matches_df.away_team)
    all_teams = sort(unique(vcat(home_teams, away_teams)))

    team_to_id = Dict(team => i for (i, team) in enumerate(all_teams))
    id_to_team = Dict(i => team for (i, team) in enumerate(all_teams))
    n_teams = length(all_teams)

    # 3. Add the new ID columns to the DataFrame
    matches_df.home_team_id = [team_to_id[team] for team in matches_df.home_team]
    matches_df.away_team_id = [team_to_id[team] for team in matches_df.away_team]

    return FeatureSet(matches_df, team_to_id, id_to_team, n_teams)
end


end

