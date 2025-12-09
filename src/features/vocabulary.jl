# src/features/Vocabulary.jl

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
    team_factory!(data_store, mapping_keys, G_dict)
    league_factory!(data_store, mapping_keys, G_dict)
    
    return Vocabulary(G_dict)
end


function team_factory!(data_store::DataStore, mapping_keys::Vector{Symbol}, G_dict::Dict{Symbol, Any}) 

    if :team_map in mapping_keys || :n_teams in mapping_keys
        all_teams = unique(vcat(data_store.matches.home_team, data_store.matches.away_team))
        team_map = Dict(name => i for (i, name) in enumerate(all_teams))
        G_dict[:team_map] = team_map
        G_dict[:n_teams] = length(team_map)
    end

end


function league_factory!(data_store::DataStore, mapping_keys::Vector{Symbol}, G_dict::Dict{Symbol, Any}) 

    if :league_map in mapping_keys || :n_leagues in mapping_keys
        all_leagues = unique(data_store.matches.tournament_slug)
        league_map = Dict(name => i for (i, name) in enumerate(all_leagues))
        G_dict[:league_map] = league_map
        G_dict[:n_leagues] = length(league_map)
    end

end
