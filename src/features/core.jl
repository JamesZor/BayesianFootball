# src/features/core.jl

"""
    create_master_features(data::SubDataFrame, mapping::MappedData)

Generates a comprehensive NamedTuple of features required by any model in the project.
This is the single source of truth for feature generation.
"""
function create_master_features(data::AbstractDataFrame, mapping::MappedData)
    # Map team and league names to integer IDs
    home_team_ids = [mapping.team[team] for team in data.home_team]
    away_team_ids = [mapping.team[team] for team in data.away_team]
    league_ids = [mapping.league[string(id)] for id in data.tournament_id]

    # Get counts
    n_teams = length(mapping.team)
    n_leagues = length(mapping.league)

    return (
        home_team_ids = home_team_ids,
        away_team_ids = away_team_ids,
        league_ids = league_ids,
        n_teams = n_teams,
        n_leagues = n_leagues,
        goals_home_ht = data.home_score_ht,
        goals_away_ht = data.away_score_ht,
        goals_home_ft = data.home_score,
        goals_away_ft = data.away_score,
    )
end
