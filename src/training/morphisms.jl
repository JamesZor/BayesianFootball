# src/training/morphisms.jl

# Morphism: TuringModels -> Chains (This function is unchanged)
function sampling_morphism(sample_config::ModelSampleConfig)
    return models -> begin
        ht_chain = sample(models.ht, NUTS(), MCMCSerial(),
                          sample_config.steps, 1;
                          progress=sample_config.bar)
        ft_chain = sample(models.ft, NUTS(), MCMCSerial(),
                          sample_config.steps, 1;
                          progress=sample_config.bar)
        ModelChain(ht_chain, ft_chain)
    end
end

"""
    compose_training_morphism(model_def, sample_config, mapping)

Composes the entire training pipeline from data to chains using the ModelDefinition protocol.
"""
function compose_training_morphism(
    model_def::AbstractModelDefinition,
    sample_config::ModelSampleConfig,
    mapping::MappedData
)
    # Get the list of required core feature symbols ONCE
    required_features = get_required_features(model_def)

    # This is the composed function: data -> chains
    return (data, info) -> begin
        # 1. Generate all possible features from the raw data
        all_features = create_master_features(data, mapping)

        # 2. Filter to get only the core features the model needs
        # This solves the performance issue by creating a minimal NamedTuple.
        model_features = (; (k => all_features[k] for k in required_features)...)

        # 3. Build the specific Turing models for Half-Time and Full-Time
        ht_model = build_turing_model(
            model_def,
            model_features,
            all_features.goals_home_ht,
            all_features.goals_away_ht
        )
        ft_model = build_turing_model(
            model_def,
            model_features,
            all_features.goals_home_ft,
            all_features.goals_away_ft
        )
        models = BasicMaherModels(ht_model, ft_model)

        # 4. Sample the models to get the chains
        chains = sampling_morphism(sample_config)(models)

        # 5. Package the results
        TrainedChains(
            chains.ht,
            chains.ft,
            info,
            nrow(data)
        )
    end
end


# Core morphism functions
# function features_morphism(model_config::ModelConfig, mapping::MappedData)
#     return data -> model_config.feature_map(data, mapping)
# end
#
# # Morphism: Features -> TuringModels
# function models_morphism(model_config::ModelConfig)
#     return features -> begin
#
#         # Manually unpack features for the Half-Time model
#         ht_model = model_config.model(
#             features.home_team_ids,
#             features.away_team_ids,
#             features.goals_home_ht,
#             features.goals_away_ht,
#             features.n_teams
#         )
#
#         # Manually unpack features for the Full-Time model
#         ft_model = model_config.model(
#             features.home_team_ids,
#             features.away_team_ids,
#             features.goals_home_ft,
#             features.goals_away_ft,
#             features.n_teams
#         )
#
#         BasicMaherModels(ht_model, ft_model)
#     end
# end
#
#
# # src/training/morphisms.jl
#
# function models_morphism(model_config::ModelConfig)
#     return features -> begin
#
#         # Check if the features for the league model are present
#         is_league_model = haskey(features, :league_ids) && haskey(features, :n_leagues)
#
#         if is_league_model
#             # --- Call for League-Specific Models ---
#             ht_model = model_config.model(
#                 features.home_team_ids, features.away_team_ids,
#                 features.goals_home_ht, features.goals_away_ht,
#                 features.n_teams, features.n_leagues, features.league_ids
#             )
#             ft_model = model_config.model(
#                 features.home_team_ids, features.away_team_ids,
#                 features.goals_home_ft, features.goals_away_ft,
#                 features.n_teams, features.n_leagues, features.league_ids
#             )
#         else
#             # --- Call for Basic Models ---
#             ht_model = model_config.model(
#                 features.home_team_ids, features.away_team_ids,
#                 features.goals_home_ht, features.goals_away_ht,
#                 features.n_teams
#             )
#             ft_model = model_config.model(
#                 features.home_team_ids, features.away_team_ids,
#                 features.goals_home_ft, features.goals_away_ft,
#                 features.n_teams
#             )
#         end
#
#         BasicMaherModels(ht_model, ft_model)
#     end
# end
#
# # NOTE: Don't pass named dict, turing doesn't like them, get the step size wrong
#
# # TODO: add number of chains to config
# # Morphism: TuringModels -> Chains 
# function sampling_morphism(sample_config::ModelSampleConfig)
#     return models -> begin
#         ht_chain = sample(models.ht, NUTS(), MCMCSerial(), 
#                          sample_config.steps, 1; 
#                          progress=sample_config.bar)
#         ft_chain = sample(models.ft, NUTS(), MCMCSerial(), 
#                          sample_config.steps, 1;
#                          progress=sample_config.bar)
#         ModelChain(ht_chain, ft_chain)
#     end
# end
# # Composed training morphism: SubDataFrame -> TrainedChains
# function compose_training_morphism(
#     model_config::ModelConfig,
#     sample_config::ModelSampleConfig,
#     mapping::MappedData
# )
#     # Compose the morphisms: data -> features -> models -> chains
#     return (data, info) -> begin
#         features = features_morphism(model_config, mapping)(data)
#         models = models_morphism(model_config)(features)
#         chains = sampling_morphism(sample_config)(models)
#
#         TrainedChains(
#             chains.ht,
#             chains.ft,
#             info,
#             nrow(data)
#         )
#     end
# end
