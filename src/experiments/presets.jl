# src/experiments/presets.jl

using ..Data
using ..Training
using ..Samplers
using ..Models

export create_benchmark_config, create_experiment_task

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
        description = "Created via factory with N=$samples, Chains=$chains", 
        save_dir = "./data/experiment_run_dev/"
    )
end



export experiment_config_models

"""
    experiment_config_models(model, name; kwargs...)

Creates a standard experiment configuration. All internal settings can be 
overridden via keyword arguments.
"""
function experiment_config_models(
    model, 
    name::String;
    # --- Splitter (CVConfig) Defaults ---
    tournament_ids::AbstractVector{<:Integer} = [54, 55, 56, 57], 
    target_seasons::AbstractVector{<:String} = ["21/22", "22/23", "24/25"],
    history_seasons::Int = 0,
    dynamics_col::Symbol = :match_week,
    warmup_period::Int = 12,
    stop_early::Bool = true,

    # --- Training Strategy Defaults ---
    parallel::Bool = true, 
    max_concurrent_splits::Int = 4,

    # --- Sampler (NUTS) Defaults ---
    n_samples::Int = 1_000, 
    n_chains::Int = 2, 
    n_warmup::Int = 500,

    # --- Metadata Defaults ---
    tags::Vector{String} = ["benchmark", "dev"],
    description::String = "Standard benchmark config"
)::ExperimentConfig

    # 1. Define Data Config
    cv_config = Data.CVConfig(
        tournament_ids = tournament_ids,
        target_seasons = target_seasons,
        history_seasons = history_seasons,
        dynamics_col = dynamics_col,
        warmup_period = warmup_period,
        stop_early = stop_early
    )

    # 2. Define Training Strategy
    train_cfg = Training.Independent(
        parallel = parallel, 
        max_concurrent_splits = max_concurrent_splits
    )
    
    # 3. Define Sampler
    sampler_conf = Samplers.NUTSConfig(
        n_samples = n_samples, 
        n_chains = n_chains, 
        n_warmup = n_warmup
    )

    # 4. Bundle Training Config
    training_config = Training.TrainingConfig(sampler_conf, train_cfg)

    # 5. Assemble Final Experiment Config
    return ExperimentConfig(
        name = name,
        model = model,
        splitter = cv_config,
        training_config = training_config,
        tags = tags,
        description = description
    )
end

"""
    create_experiment_task(ds::DataStore, model, name::String, save_dir::String; kwargs...)

Creates an `ExperimentTask` containing both the dataset and the configuration, ready to be run.
Defaults are provided for all major parameters to ensure a clean user experience.

# Arguments
- `ds::DataStore`: The dataset to run the experiment on.
- `model::AbstractFootballModel`: The Bayesian model engine.
- `name::String`: Name label for the experiment (e.g. "ab_std_player").
- `save_dir::String`: Directory path to save the experiment artifacts.

# Keywords
- `target_seasons`: A vector of season strings (e.g., `["2025", "2026"]` or `["24/25"]`). Must match the format of the `DataStore`.
- `history_seasons`: Number of historical seasons to include for burn-in (default `2`).
- `dynamics_col`: The time column used for grouped CV (default `:match_month`).
- `warmup_period`: Warmup period for evaluating early season matches (default `0`).
- `samples`: NUTS sampling steps (default `500`).
- `chains`: NUTS chains (default `4`).
- `warmup`: NUTS warmup steps (default `200`).
- `accept_rate`: NUTS acceptance rate (default `0.65`).
- `parallel`: Run cross-validation folds in parallel (default `true`).
"""
function create_experiment_task(
    ds::Data.DataStore, 
    model, 
    name::String, 
    save_dir::String;
    # --- Splitter Overrides ---
    target_seasons::Vector{String},
    history_seasons::Int = 2,
    dynamics_col::Symbol = :match_month,
    warmup_period::Int = 0,
    stop_early::Bool = false,
    
    # --- Sampler Defaults ---
    samples::Int = 500,
    chains::Int = 4,
    warmup::Int = 200,
    accept_rate::Float64 = 0.65,
    max_depth::Int = 10,
    
    # --- Training/Execution Defaults ---
    parallel::Bool = true,
    max_concurrent_splits::Int = 4
)

    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = history_seasons,
        dynamics_col = dynamics_col,
        warmup_period = warmup_period,
        stop_early = stop_early
    )

    sampler_conf = Samplers.NUTSConfig(
        samples, 
        chains, 
        warmup, 
        accept_rate, 
        max_depth,  
        Samplers.UniformInit(-1, 1),
        false #  display the chain progress 
    )

    train_cfg = Training.Independent(
        parallel = parallel,
        max_concurrent_splits = max_concurrent_splits
    )
    
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    config = ExperimentConfig(
        name = name,
        model = model, 
        splitter = cv_config,
        training_config = training_config,
        save_dir = save_dir
    )

    return ExperimentTask(ds, config)
end
