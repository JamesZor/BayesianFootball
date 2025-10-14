"""
This module is responsible for transforming raw data from a DataStore
into a model-ready FeatureSet.
"""
module Features

using DataFrames
using ..Data: DataStore

export FeatureSet, create_features

# --- Constants for required columns ---
const REQUIRED_MATCH_COLS = [
    :home_team, :away_team, :home_score, :away_score, :match_date
]

# --- Struct Definition ---
"""
    FeatureSet

A container for all data needed by a model. This struct is now focused
on the grouped data required by dynamic models.

# Fields
- `matches_df::DataFrame`: The main DataFrame with added features.
- `team_map::Dict{String, Int}`: A mapping from team names to integer IDs.
- `n_teams::Int`: The total number of unique teams.
- `n_rounds::Int`: The total number of unique global rounds.
- `round_home_ids::Vector{Vector{Int}}`: A vector where each element is a vector of home team IDs for that round.
- `round_away_ids::Vector{Vector{Int}}`: A vector where each element is a vector of away team IDs for that round.
- `round_home_goals::Vector{Vector{Int}}`: A vector for home goals, grouped by round.
- `round_away_goals::Vector{Vector{Int}}`: A vector for away goals, grouped by round.
"""
struct FeatureSet
    matches_df::DataFrame
    team_map::Dict{String, Int}
    n_teams::Int
    n_rounds::Int
    round_home_ids::Vector{Vector{Int}}
    round_away_ids::Vector{Vector{Int}}
    round_home_goals::Vector{Vector{Int}}
    round_away_goals::Vector{Vector{Int}}
end


# --- Private Helper Functions ---
function _add_global_round_column(matches_df::DataFrame)
    df = sort(matches_df, :match_date)
    num_matches = nrow(df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end
    
    df.global_round = global_rounds
    return df
end

# --- Main API Function ---
function create_features(data_store::DataStore)::FeatureSet
    # 1. Start with the matches data and drop any rows with missing scores.
    #    This is the key fix to ensure clean data for the models.
    matches_df = dropmissing(data_store.matches, [:home_score, :away_score])

    # 2. Select only the necessary columns
    matches_df = select(matches_df, REQUIRED_MATCH_COLS)

    # 3. Add the global_round column
    matches_df = _add_global_round_column(matches_df)

    # 4. Create team-to-integer mappings
    all_teams = unique(vcat(matches_df.home_team, matches_df.away_team))
    team_map = Dict(team_name => i for (i, team_name) in enumerate(all_teams))
    n_teams = length(team_map)

    # 5. Create data structures for DYNAMIC models (grouped by round)
    grouped = groupby(matches_df, :global_round)
    n_rounds = length(grouped)
    round_home_ids = [ [team_map[name] for name in g.home_team] for g in grouped]
    round_away_ids = [ [team_map[name] for name in g.away_team] for g in grouped]
    round_home_goals = [g.home_score for g in grouped]
    round_away_goals = [g.away_score for g in grouped]

    # 6. Return the completed FeatureSet
    return FeatureSet(
        matches_df, team_map, n_teams, n_rounds,
        round_home_ids, round_away_ids, round_home_goals, round_away_goals
    )
end

end
