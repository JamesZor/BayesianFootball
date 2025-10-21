# src/features/Vocabulary.jl


# --- Struct Definitions (Flexible Containers) ---

# """
#     Vocabulary (Your 'G')
#
# A flexible container for all *global* mappings created from the
# *entire* dataset. This is created once and reused for all data splits.
# It holds a dictionary of mappings, e.g.,
# :team_map => Dict("Team A" => 1, ...)
# :n_teams  => 20
# """
# struct Vocabulary
#     mappings::Dict{Symbol, Any}
# end
#
"""
    create_vocabulary(data_store::DataStore, model::AbstractFootballModel)

Creates the global Vocabulary (G) from the entire DataStore (D) by asking
the model what mappings it requires. This is your `G_map: D × M -> G`.
"""
function create_vocabulary(data_store::DataStore, model::AbstractFootballModel)::Vocabulary
    # 1. Ask the model what mappings it needs via the contract
    mapping_keys = required_mapping_keys(model)
    
    G_dict = Dict{Symbol, Any}()

    # 2. Run the "factories" for the requested keys
    
    # --- Team Factory ---
    if :team_map in mapping_keys || :n_teams in mapping_keys
        all_teams = unique(vcat(data_store.matches.home_team, data_store.matches.away_team))
        team_map = Dict(name => i for (i, name) in enumerate(all_teams))
        G_dict[:team_map] = team_map
        G_dict[:n_teams] = length(team_map)
    end

    # --- League Factory (Example for future extension) ---
    if :league_map in mapping_keys || :n_leagues in mapping_keys
        all_leagues = unique(data_store.matches.tournament_slug)
        league_map = Dict(name => i for (i, name) in enumerate(all_leagues))
        G_dict[:league_map] = league_map
        G_dict[:n_leagues] = length(league_map)
    end

    return Vocabulary(G_dict)
end
