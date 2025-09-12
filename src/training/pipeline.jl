# src/training/pipeline.jl

# src/training/pipeline.jl
using Dates
"""
    train_all_splits(data_store, cv_config, model_def, sample_config, mapping_funcs; parallel=true)

Main training pipeline that applies the composed morphisms to all CV splits.
Returns an ExperimentResult containing all trained chains.
"""
function train_all_splits(
    data_store::DataStore,
    cv_config::TimeSeriesSplitsConfig,
    model_def::AbstractModelDefinition, # <-- MODIFIED from model_config
    sample_config::ModelSampleConfig,
    mapping_funcs::MappingFunctions;
    parallel::Bool = true
)::ExperimentResult
    
    start_time = time()
    
    # Create mapping once for all splits
    mapping = MappedData(data_store, mapping_funcs)
    
    # Create the composed training morphism
    training_morphism = compose_training_morphism(
        model_def, # <-- MODIFIED from model_config
        sample_config,
        mapping
    )
    
    # Generate splits
    splits = time_series_splits(data_store, cv_config, :sequential)
    split_data = collect(splits)
    
    println("Training on $(length(split_data)) splits...")
    
    # Apply training morphism to all splits
    chains_sequence = if parallel
        println("Using parallel execution with $(Threads.nthreads()) threads")
        ThreadsX.map(split_data) do (data, info)
            println("  Training split: $info")
            training_morphism(data, info)
        end
    else
        println("Using sequential execution")
        map(split_data) do (data, info)
            println("  Training split: $info")
            training_morphism(data, info)
        end
    end
    
    # Hash config for tracking
    config_hash = hash((
        cv_config.base_seasons,
        cv_config.target_seasons,
        sample_config.steps,
        nameof(typeof(model_def)) # Using the type name of the model_def struct
    ))
    
    total_time = time() - start_time
    println("Training completed in $(round(total_time, digits=2)) seconds")
    
    return ExperimentResult(
        chains_sequence,
        mapping,
        config_hash,
        total_time
    )
end

function predict_target_season(model::TrainedModel, target_matches::DataFrame)
    
    predictions_dict = Dict{Int, Any}()

    for round_group in groupby(target_matches, :round)
        
        first_match = round_group[1, :]
        chains_for_round = get_chains_for_match(model, first_match)

        for (i, match_row) in enumerate(eachrow(round_group))
            
            match_subframe = SubDataFrame(round_group, i:i, :)

            # 1. Generate features for this single match
            features = create_master_features(
                match_subframe,
                model.result.mapping
            )

            # 2. Call the wrapper that handles both HT and FT
            match_predictions = predict_match_lines(
                model.config.model_def,
                chains_for_round,
                features,
                model.result.mapping
            )
            
            predictions_dict[match_row.match_id] = match_predictions
        end
    end
    
    return predictions_dict
end

"""
    get_chains_for_match(model::TrainedModel, match_row::DataFrameRow)

Finds the most up-to-date `TrainedChains` object from the model's
sequence that can be used to predict the given match.

This is the model trained on all data *before* the match's round.
"""
function get_chains_for_match(model::TrainedModel, match_row::DataFrameRow)
    # The sequence of chains is ordered by the training data progression.
    # chains_sequence[1] is the base model, used for the first round of predictions.
    # chains_sequence[2] is trained on base + round 1, used for round 2 predictions.
    # Therefore, to predict a match in `round_num`, we need the chain at that index.
    
    round_num = match_row.round # Assuming a :round column
    
    if round_num > length(model.result.chains_sequence) || round_num < 1
        error("Cannot find a suitable model for round $(round_num). Check data.")
    end
    
    # Simple direct mapping: for predicting round `k`, use chain at index `k`.
    return model.result.chains_sequence[round_num]
end

