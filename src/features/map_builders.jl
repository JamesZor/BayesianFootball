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
