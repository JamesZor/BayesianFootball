# src/features/maher_variants.jl

function feature_map_maher_league_ha(data::SubDataFrame, mapping::MappedData)
    # 1. Get all the features from the basic model's map
    base_features = feature_map_basic_maher_model(data, mapping)

    # 2. Add the new features required by our new model
    league_ids = [mapping.league[string(id)] for id in data.tournament_id]
    n_leagues = length(mapping.league)

    # 3. Merge and return them all in a NamedTuple
    return (
        base_features..., # Splats the contents of the base tuple
        n_leagues = n_leagues,
        league_ids = league_ids
    )
end
