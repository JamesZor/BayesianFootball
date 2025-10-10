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
        global_round = data.global_round
    )
end



# --- 3. Utility Functions ---
"""
    add_global_round_column!(matches_df::DataFrame)

Adds a `:global_round` column in-place to the DataFrame.
"""
function add_global_round_column!(matches_df::DataFrame)
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
    println("✅ Successfully added `:global_round` column. Found $(global_round_counter) unique time steps.")
    return matches_df
end

