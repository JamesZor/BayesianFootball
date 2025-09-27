# src/training/morphisms.jl

# Morphism: TuringModels -> Chains (This function is unchanged)

using Base.Threads
# if running a single model 
function sampling_morphism(sample_config::ModelSampleConfig)
    return models -> begin
        # @spawn starts each task on a separate available thread
        # ht_task = @spawn sample(models.ht, NUTS(), MCMCSerial(),
        #                         sample_config.steps, 1;
        #                         progress=sample_config.bar)
        #
        # ft_task = @spawn sample(models.ft, NUTS(), MCMCSerial(),
        #                         sample_config.steps, 1;
        #                         progress=sample_config.bar)


        ft_chain = sample(models.ft, NUTS(), MCMCSerial(),
                                sample_config.steps, 1;
                                progress=sample_config.bar)
        ht_chain = sample(models.ht, NUTS(), MCMCSerial(),
                                sample_config.steps, 1;
                                progress=sample_config.bar)
        # fetch waits for the tasks to finish and gets their results
        # ht_chain = fetch(ht_task)
        # ft_chain = fetch(ft_task)

        ModelChain(ht_chain, ft_chain)
    end
end



"""
    compose_training_morphism(model_def, sample_config, mapping)

Composes the entire training pipeline from data to chains using the ModelDefinition protocol.

This version is generalized to handle different model scopes (e.g., :HT, :FT, or both)
by using the `model_scope` trait.
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
        # 1. Generate all possible features from the raw data [cite: 100]
        all_features = create_master_features(data, mapping) 

        # 2. Filter to get only the core features the model needs [cite: 101]
        model_features = (; (k => all_features[k] for k in required_features)...)


        # HACK:
        # 3. Get the model scope to determine which models to build
        # scope = model_scope(model_def)
        # scope = (:HT, :FT)
        scope = (:HT, :FT )
        
        println("  [Thread $(threadid())] Building model(s) for scope: $scope")

        ht_task = nothing
        ft_task = nothing

        # 4. Conditionally build and sample models in parallel
        if :HT in scope
            ht_model = build_turing_model(
                model_def,
                model_features,
                all_features.goals_home_ht,
                all_features.goals_away_ht
            )
            ht_task = @spawn sample(ht_model, NUTS(), MCMCSerial(),
                                    sample_config.steps, 1;
                                    progress=sample_config.bar)
        end

        if :FT in scope
            ft_model = build_turing_model(
                model_def,
                model_features,
                all_features.goals_home_ft,
                all_features.goals_away_ft
            )
            ft_task = @spawn sample(ft_model, NUTS(), MCMCSerial(),
                                    sample_config.steps, 1;
                                    progress=sample_config.bar)
        end

        # 5. Fetch results, handling cases where a model was not run
        ht_chain = isnothing(ht_task) ? nothing : fetch(ht_task)
        ft_chain = isnothing(ft_task) ? nothing : fetch(ft_task)

        # 6. Package the results (assuming TrainedChains can handle `nothing`)
        TrainedChains(
            ht_chain,
            ft_chain,
            info,
            nrow(data)
        )
    end
end





# """
#     compose_training_morphism(model_def, sample_config, mapping)
#
# Composes the entire training pipeline from data to chains using the ModelDefinition protocol.
# """
# function compose_training_morphism(
#     model_def::AbstractModelDefinition,
#     sample_config::ModelSampleConfig,
#     mapping::MappedData
# )
#     # Get the list of required core feature symbols ONCE
#     required_features = get_required_features(model_def)
#
#     # This is the composed function: data -> chains
#     return (data, info) -> begin
#         # 1. Generate all possible features from the raw data
#         all_features = create_master_features(data, mapping)
#
#         # 2. Filter to get only the core features the model needs
#         # This solves the performance issue by creating a minimal NamedTuple.
#         model_features = (; (k => all_features[k] for k in required_features)...)
#
#         # 3. Build the specific Turing models for Half-Time and Full-Time
#         ht_model = build_turing_model(
#             model_def,
#             model_features,
#             all_features.goals_home_ht,
#             all_features.goals_away_ht
#         )
#         ft_model = build_turing_model(
#             model_def,
#             model_features,
#             all_features.goals_home_ft,
#             all_features.goals_away_ft
#         )
#         models = BasicMaherModels(ht_model, ft_model)
#
#         # 4. Sample the models to get the chains
#         chains = sampling_morphism(sample_config)(models)
#
#         # 5. Package the results
#         TrainedChains(
#             chains.ht,
#             chains.ft,
#             info,
#             nrow(data)
#         )
#     end
# end
