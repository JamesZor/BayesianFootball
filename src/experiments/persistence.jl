# src/experiments/persistence.jl

using JLD2, JSON, Dates
using ..BayesianFootball: ExperimentConfig, ExperimentResult

"""
    ExperimentRun

A struct to manage all aspects of a single model training execution.
"""
struct ExperimentRun
    experiment_name::String
    config::ExperimentConfig
    base_path::String
    run_path::String
end

function prepare_run(experiment_name::String, config::ExperimentConfig, base_path::String)::ExperimentRun
    model_variant_name = config.name
    
    run_path = joinpath(base_path, experiment_name, model_variant_name)
    mkpath(run_path)
    
    println("Preparing run in: $run_path")
    return ExperimentRun(experiment_name, config, base_path, run_path)
end

function save(run::ExperimentRun, result::ExperimentResult)
    jldsave(joinpath(run.run_path, "chains.jld2"); chains=result.chains_sequence)

    metadata = Dict(
        "experiment_name" => run.experiment_name,
        "model_variant_name" => run.config.name,
        "run_path" => run.run_path,
        "timestamp" => basename(run.run_path),
        "total_time_seconds" => result.total_time,
        "config_hash" => result.config_hash,
        "config" => Dict(
             "model" => String(nameof(run.config.model_config.model)),
             "feature_map" => String(nameof(run.config.model_config.feature_map)),
             "base_seasons" => run.config.cv_config.base_seasons,
             "target_seasons" => run.config.cv_config.target_seasons,
             "mcmc_steps" => run.config.sample_config.steps
        )
    )
    
    open(joinpath(run.run_path, "metadata.json"), "w") do f
        JSON.print(f, metadata, 4)
    end
    
    println("✅ Successfully saved run for '$(run.config.name)'.")
end

function load_run(path::String)
    if !isdir(path)
        error("Directory not found: $path")
    end
    
    chains = JLD2.load(joinpath(path, "chains.jld2"))["chains"]
    metadata = JSON.parsefile(joinpath(path, "metadata.json"))
    
    return (chains=chains, metadata=metadata)
end
