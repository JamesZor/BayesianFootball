# src/experiments/persistence.jl
using JLD2, JSON, Dates, DataFrames, Glob
using ..BayesianFootball: ExperimentConfig, ExperimentResult, TrainedModel



"""
    load_model(run_path::String) -> TrainedModel

Loads a saved experiment run from a directory and packages it into a
convenient `TrainedModel` object, ready for prediction.
"""
function load_model(run_path::String)
    loaded_data = load_run(run_path)
    return TrainedModel(loaded_data.config, loaded_data.result)
end

"""
    list_runs(experiment_path::String)

Scans an experiment directory, reads metadata for each run,
and returns a DataFrame summarizing them.
"""
function list_runs(experiment_path::String)
    run_paths = glob("*/metadata.json", experiment_path)
    
    if isempty(run_paths)
        println("No completed runs found in: $experiment_path")
        return DataFrame()
    end
    
    summaries = []
    for (id, path) in enumerate(run_paths)
        metadata = JSON.parsefile(path)
        config = metadata["config"]
        
        summary = (
            id = id,
            model_name = metadata["model_variant_name"],
            model_type = config["model"],
            timestamp = metadata["timestamp"],
            mcmc_steps = config["mcmc_steps"],
            path = dirname(path) # Store the path to the run folder
        )
        push!(summaries, summary)
    end
    
    return DataFrame(summaries)
end


"""
    ExperimentRun

A struct to manage all aspects of a single model training execution,
including file paths and metadata.
"""
struct ExperimentRun
    experiment_name::String
    config::ExperimentConfig
    base_path::String
    run_path::String
    timestamp::String
end

function prepare_run(experiment_name::String, config::ExperimentConfig, base_path::String)::ExperimentRun
    model_variant_name = config.name
    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    run_folder_name = "$(model_variant_name)_$(timestamp)"
    run_path = joinpath(base_path, experiment_name, run_folder_name)
    mkpath(run_path)
    println("Preparing run in: $run_path")
    return ExperimentRun(experiment_name, config, base_path, run_path, timestamp)
end

function save(run::ExperimentRun, result::ExperimentResult)
    jldsave(joinpath(run.run_path, "run_data.jld2"); result=result, config=run.config)
    
    metadata = Dict(
        "experiment_name" => run.experiment_name,
        "model_variant_name" => run.config.name,
        "run_path" => run.run_path,
        "timestamp" => run.timestamp,
        "total_time_seconds" => result.total_time,
        "config_hash" => result.config_hash,
        "config" => Dict(
            "model" => String(nameof(typeof(run.config.model_def))),
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
    if !isdir(path) error("Directory not found: $path") end
    
    run_data_path = joinpath(path, "run_data.jld2")
    metadata_path = joinpath(path, "metadata.json")

    if !isfile(run_data_path) || !isfile(metadata_path)
        error("Run directory is incomplete. Missing run_data.jld2 or metadata.json in $path")
    end
    
    run_data = JLD2.load(run_data_path)
    metadata = JSON.parsefile(metadata_path)
    
    return (result=run_data["result"], config=run_data["config"], metadata=metadata)
end

"""
    save_experiment_metadata(experiment_name, base_path, data_path, cv_config, sample_config)

Saves a JSON file with metadata common to all runs within an experiment.
"""
function save_experiment_metadata(experiment_name::String, base_path::String, data_path::String, 
                                  cv_config::TimeSeriesSplitsConfig, sample_config::ModelSampleConfig)
    
    experiment_path = joinpath(base_path, experiment_name)
    mkpath(experiment_path)
    
    metadata = Dict(
        "experiment_name" => experiment_name,
        "data_path" => data_path,
        "timestamp_created" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "cv_config" => Dict(
            "base_seasons" => cv_config.base_seasons,
            "target_seasons" => cv_config.target_seasons,
            "round_col" => String(cv_config.round_col)
        ),
        "sample_config" => Dict(
            "steps" => sample_config.steps,
            "progress_bar" => sample_config.bar
        )
    )
    
    open(joinpath(experiment_path, "experiment_metadata.json"), "w") do f
        JSON.print(f, metadata, 4)
    end
    println("Saved experiment metadata to: $experiment_path")
end





#
# # olds
# using JLD2, JSON, Dates
# using ..BayesianFootball: ExperimentConfig, ExperimentResult, TimeSeriesSplitsConfig, ModelSampleConfig
#
# """
#     ExperimentRun
#
# A struct to manage all aspects of a single model training execution,
# including file paths and metadata.
# """
# struct ExperimentRun
#     experiment_name::String
#     config::ExperimentConfig
#     base_path::String
#     run_path::String
#     timestamp::String
# end
#
# """
#     prepare_run(experiment_name, config, base_path)
#
# Creates a unique, timestamped directory for a new experiment run.
# """
# function prepare_run(experiment_name::String, config::ExperimentConfig, base_path::String)::ExperimentRun
#     model_variant_name = config.name
#     timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
#
#     # Create a unique name for the run folder to avoid overwriting
#     run_folder_name = "$(model_variant_name)_$(timestamp)"
#
#     run_path = joinpath(base_path, experiment_name, run_folder_name)
#     mkpath(run_path)
#
#     println("Preparing run in: $run_path")
#     return ExperimentRun(experiment_name, config, base_path, run_path, timestamp)
# end
#
# """
#     save(run, result)
#
# Saves the MCMC chains and run metadata to the experiment's directory.
# """
# function save(run::ExperimentRun, result::ExperimentResult)
#     # Save the chains using JLD2 for efficient storage
#     jldsave(joinpath(run.run_path, "chains.jld2"); chains=result.chains_sequence)
#
#     # Create a metadata dictionary with all relevant run information
#     metadata = Dict(
#         "experiment_name" => run.experiment_name,
#         "model_variant_name" => run.config.name,
#         "run_path" => run.run_path,
#         "timestamp" => run.timestamp,
#         "total_time_seconds" => result.total_time,
#         "config_hash" => result.config_hash,
#         "config" => Dict(
#             # Get the model's type name from the model_def struct
#             "model" => String(nameof(typeof(run.config.model_def))),
#             # The concept of a single feature_map is gone, so we remove it.
#             "base_seasons" => run.config.cv_config.base_seasons,
#             "target_seasons" => run.config.cv_config.target_seasons,
#             "mcmc_steps" => run.config.sample_config.steps
#         )
#     )
#
#     # Save the metadata as a human-readable JSON file
#     open(joinpath(run.run_path, "metadata.json"), "w") do f
#         JSON.print(f, metadata, 4)
#     end
#
#     println("✅ Successfully saved run for '$(run.config.name)'.")
# end
#
# """
#     load_run(path)
#
# Loads the chains and metadata from a specified run directory.
# """
# function load_run(path::String)
#     if !isdir(path)
#         error("Directory not found: $path")
#     end
#
#     chains_path = joinpath(path, "chains.jld2")
#     metadata_path = joinpath(path, "metadata.json")
#
#     if !isfile(chains_path) || !isfile(metadata_path)
#         error("Run directory is incomplete. Missing chains.jld2 or metadata.json in $path")
#     end
#
#     chains = JLD2.load(chains_path)["chains"]
#     metadata = JSON.parsefile(metadata_path)
#
#     return (chains=chains, metadata=metadata)
# end
#
