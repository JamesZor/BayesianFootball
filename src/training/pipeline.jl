# src/training/pipeline.jl
# Core training pipeline implementation

"""
    train_all_splits(data_store, cv_config, model_config, sample_config, mapping_funcs; parallel=true)

Main training pipeline that applies the composed morphisms to all CV splits.
Returns an ExperimentResult containing all trained chains.
"""

function train_all_splits(
    data_store::DataStore,
    cv_config::TimeSeriesSplitsConfig,
    model_config::ModelConfig,
    sample_config::ModelSampleConfig,
    mapping_funcs::MappingFunctions;
    parallel::Bool = true
)::ExperimentResult
    
    start_time = time()
    
    # Create mapping once for all splits
    mapping = MappedData(data_store, mapping_funcs)
    
    # Create the composed training morphism
    training_morphism = compose_training_morphism(
        model_config,
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
        nameof(model_config.model)
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


"""
    train_single_split(training_data, model_config, sample_config, mapping; info="single_split")

Trains a model on a single provided AbstractDataFrame (e.g., a DataFrame or SubDataFrame).
This is useful for debugging or one-off model runs.

# Arguments
- `training_data`: The DataFrame/SubDataFrame to train the model on.
- `model_config`: The configuration defining the model and feature map.
- `sample_config`: The configuration for MCMC sampling (e.g., number of steps).
- `mapping`: A pre-computed `MappedData` object. Must be created from the full dataset.
- `info`: An optional string to describe the split.

# Returns
- A `TrainedChains` object containing the results of the single training run.
"""
function train_single_split(
    training_data::AbstractDataFrame,
    model_config::ModelConfig,
    sample_config::ModelSampleConfig,
    mapping::MappedData;
    info::String = "manual_single_split"
)::TrainedChains
    
    println("Starting training for single split: $info")
    start_time = time()

    # 1. Create the composed training morphism (reusing the core logic)
    training_morphism = compose_training_morphism(
        model_config,
        sample_config,
        mapping
    )

    # 2. Apply the morphism directly to the provided data
    # This is the core of the function, replacing the loop from train_all_splits
    trained_chains = training_morphism(training_data, info)

    total_time = time() - start_time
    println("Training completed in $(round(total_time, digits=2)) seconds")

    return trained_chains
end

"""
    sample_basic_maher_model(models, sample_config)

Execute MCMC sampling for both half-time and full-time models.
"""
function sample_basic_maher_model(
    models::BasicMaherModels,
    sample_config::ModelSampleConfig
)::ModelChain
    
    # Sample half-time model
    ht_chain = sample(
        models.ht, 
        NUTS(0.65),  # Acceptance rate target
        MCMCThreads(), 
        sample_config.steps, 
        4;  # Number of chains
        progress = sample_config.bar
    )
    
    # Sample full-time model
    ft_chain = sample(
        models.ft, 
        NUTS(0.65),
        MCMCThreads(), 
        sample_config.steps, 
        4;
        progress = sample_config.bar
    )
    
    return ModelChain(ht_chain, ft_chain)
end
