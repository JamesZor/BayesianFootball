# src/experiments/presets.jl

using ..Data
using ..Training
using ..Samplers
using ..Models

export create_benchmark_config

"""
    create_benchmark_config(model, name; kwargs...)

Factory function to create a standard experiment configuration.
You can override specific parts by passing them as keyword arguments.
"""
function create_benchmark_config(
    model, 
    name::String;
    # 1. Allow overriding high-level components
    splitter = nyothing, 
    
    # 2. Allow overriding specific common parameters (The "Shortcuts")
    samples = 500,
    chains = 2,
    warmup = 300,
    parallel = true
)

    # --- A. Handle Splitter Defaults ---
    # If the user didn't pass a specific splitter, create the standard one
    final_splitter = isnothing(splitter) ? 
        ExpandingWindowCV(
            train_seasons = [],
            test_seasons = ["24/25"],
            window_col = :split_col,
            method = :sequential,
            dynamics_col = :match_week
        ) : splitter

    # --- B. Handle Training Defaults ---
    # We build this fresh every time using the passed arguments
    sampler_config = NUTSConfig(
        n_samples = samples, 
        n_chains = chains, 
        n_warmup = warmup
    )
    
    train_strategy = Independent(
        parallel = parallel, 
        max_concurrent_splits = 4
    )

    training_config = TrainingConfig(sampler_config, train_strategy)

    # --- C. Return the Config ---
    return ExperimentConfig(
        name = name,
        model = model,
        splitter = final_splitter,
        training_config = training_config,
        tags = ["benchmark", "standard"],
        description = "Created via factory with N=$samples, Chains=$chains"
    )
end



export experiment_config_models
"""
    experiment_config_models(model, name)

Creates a standard experiment configuration for the 21/22 season benchmark.
"""
function experiment_config_models(model, name::String)::ExperimentConfig

    # 1. Define Fixed Data Config
    cv_config = Data.CVConfig(
        tournament_ids = [55],
        target_seasons = ["21/22"],
        history_seasons = 0,
        dynamics_col = :match_week,
        warmup_period = 34,
        stop_early = true
    )

    # 2. Define Fixed Training Config
    train_cfg = Training.Independent(
        parallel = true, 
        max_concurrent_splits = 2
    )
    
    sampler_conf = Samplers.NUTSConfig(
        n_samples = 100, 
        n_chains = 1, 
        n_warmup = 100
    )

    training_config = Training.TrainingConfig(sampler_conf, train_cfg)

    # 3. Assemble
    return ExperimentConfig(
        name = name,
        model = model,
        splitter = cv_config,
        training_config = training_config,
        tags = ["benchmark", "dev"],
        description = "Standard benchmark for 21/22 season"
    )
end
