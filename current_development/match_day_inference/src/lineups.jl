# current_development/match_day_inference/src/lineups.jl

using JSON3
using DataFrames
using Dates

"""
    load_lineup_from_json(filepath::String)

Loads SofaScore-format JSON lineups file and returns structured player entries.
"""
function load_lineup_from_json(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    try
        content = read(filepath, String)
        data = JSON3.read(content)
        
        extract_players(side_data) = map(side_data.players) do p
            (
                player_id = Int(p.player.id),
                player_name = String(p.player.name),
                position = String(p.position),
                substitute = Bool(p.substitute)
            )
        end
        
        return (
            home = extract_players(data.home),
            away = extract_players(data.away)
        )
    catch e
        @warn "Failed to parse lineup JSON file $filepath: $e"
        return nothing
    end
end

"""
    get_most_recent_lineup(ds::Data.DataStore, team_name::String)

Finds the team's most recent starting/squad lineup from historical database records.
"""
function get_most_recent_lineup(ds::Data.DataStore, team_name::String)
    # Find matches involving this team
    team_matches = subset(ds.matches, 
        [:home_team, :away_team] => ByRow((h, a) -> h == team_name || a == team_name)
    )
    if isempty(team_matches)
        @warn "No historical matches found for team: $team_name"
        return []
    end
    
    # Sort chronologically to get the latest match
    sort!(team_matches, :match_date, rev=true)
    latest_match = first(team_matches)
    latest_mid = latest_match.match_id
    
    # Identify which side the team was playing on in that latest match
    side = latest_match.home_team == team_name ? "home" : "away"
    
    # Filter lineups for this match and side
    m_lineups = subset(ds.lineups, 
        :match_id => ByRow(==(latest_mid)),
        :team_side => ByRow(s -> String(s) == side)
    )
    
    if isempty(m_lineups)
        @warn "No lineups found in database for team $team_name in match $latest_mid"
        return []
    end
    
    return map(eachrow(m_lineups)) do row
        (
            player_id = Int(row.player_id),
            player_name = ismissing(row.player_name) ? "Unknown" : String(row.player_name),
            position = ismissing(row.position) ? "M" : String(row.position),
            substitute = coalesce(row.is_substitute, false)
        )
    end
end

"""
    get_matchday_lineup(ds::Data.DataStore, match_id::Int, home_team::String, away_team::String, json_dir::String)

Retrieves lineup for today's match using local JSON file if available, or falls back to database.
"""
function get_matchday_lineup(ds::Data.DataStore, match_id::Int, home_team::String, away_team::String, json_dir::String)
    filepath = joinpath(json_dir, "$(match_id).json")
    if isfile(filepath)
        println("└─ [Lineup] Loaded from JSON file: $filepath")
        return load_lineup_from_json(filepath)
    else
        println("└─ [Lineup] JSON not found for match $match_id. Falling back to most recent database lineup...")
        home_players = get_most_recent_lineup(ds, home_team)
        away_players = get_most_recent_lineup(ds, away_team)
        return (home = home_players, away = away_players)
    end
end
