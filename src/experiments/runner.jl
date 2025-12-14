# src/experiments/runner.jl

using Dates
using DataFrames
using JLD2, JSON3
using ProgressMeter

# Internal Imports
using ..Data
using ..Features
using ..Training
using ..Experiments: ExperimentConfig, ExperimentResults

export run_experiment, save_experiment, load_experiment, list_experiments

# ==============================================================================
# 1. EXPERIMENT ORCHESTRATION (The Logic)
# ==============================================================================

"""
    run_experiment(data_store, config::ExperimentConfig)

Executes the experiment pipeline.
Returns an `ExperimentResults` object. 
NOTE: Does NOT save automatically. Call `save_experiment(results)` to persist.
"""
function run_experiment(data_store, config::ExperimentConfig)
    
    _log_header(config.name)

    # --- Step 1: Vocabulary ---
    _log_step(1, "Creating Vocabulary")
    vocabulary = Features.create_vocabulary(data_store, config.model)

    # --- Step 2: Data Splitting ---
    _log_step(2, "Generating Data Splits")
    data_splits = Data.create_data_splits(data_store, config.splitter)
    _log_info("Generated $(length(data_splits)) splits")

    # --- Step 3: Feature Engineering ---
    _log_step(3, "Building Feature Sets")
    feature_sets = Features.create_features(
        data_splits, 
        vocabulary, 
        config.model, 
        config.splitter 
    )

    # --- Step 4: Training ---
    _log_step(4, "Executing Training Strategy")
    training_results = Training.train(
        config.model, 
        config.training_config, 
        feature_sets
    )

    _log_footer()

    # Construct and return results (InMemory)
    # We generate a default save path here, but don't write to it yet.
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    default_path = joinpath(config.save_dir, "$(config.name)_$(timestamp)")

    return ExperimentResults(
        config,
        training_results,
        vocabulary,
        default_path
    )
end

# ==============================================================================
# 2. PERSISTENCE (Saving)
# ==============================================================================

"""
    save_experiment(results::ExperimentResults; path=nothing, quiet=false)

Persists the experiment results to disk.
"""
function save_experiment(results::ExperimentResults; path=nothing, quiet=false)
    # Use provided path override, or the one stored in results
    target_path = isnothing(path) ? results.save_path : path
    mkpath(target_path)

    # 1. Save Binary (Results + Vocab + Chains)
    jldsave(joinpath(target_path, "results.jld2"); results)

    # 2. Save Config (JSON for readability)
    open(joinpath(target_path, "config.json"), "w") do io
        JSON3.pretty(io, results.config)
    end

    if !quiet
        printstyled("\n[IO] Artifacts saved to: ", color=:green, bold=true)
        println(target_path)
    end
end

# ==============================================================================
# 3. DISCOVERY & LOADING (The Smart Workflow)
# ==============================================================================

"""
    list_experiments(base_dir="./experiments") -> Vector{String}

Scans the directory, prints a numbered table of experiments, and returns 
the list of paths corresponding to those numbers.
"""
function list_experiments(base_dir::String="./experiments")
    if !isdir(base_dir)
        println("Directory not found: $base_dir")
        return String[]
    end

    # 1. Get and Sort Subdirectories (Newest First)
    subdirs = filter(isdir, readdir(base_dir, join=true))
    sort!(subdirs, by=mtime, rev=true)

    if isempty(subdirs)
        println("No experiments found in $base_dir")
        return String[]
    end

    # 2. Print Header
    println("\n experiments in: $base_dir")
    println("="^85)
    println(
        rpad("IDX", 5), " | ", 
        rpad("NAME", 35), " | ", 
        rpad("MODEL", 20), " | ", 
        "TIMESTAMP"
    )
    println("-"^85)

    # 3. Print Rows
    for (i, path) in enumerate(subdirs)
        meta = _read_experiment_metadata(path)
        
        # Formatting
        idx_str = "[$i]"
        name_str = length(meta.name) > 33 ? meta.name[1:33] * ".." : meta.name
        model_str = length(meta.model) > 18 ? meta.model[1:18] * ".." : meta.model
        time_str = basename(path) # Fallback if no timestamp in meta

        # Highlight the first one (most recent)
        row_color = i == 1 ? :white : :light_black
        
        printstyled(rpad(idx_str, 5), " | ", color=:cyan, bold=(i==1))
        printstyled(rpad(name_str, 35), " | ", color=row_color)
        printstyled(rpad(model_str, 20), " | ", color=row_color)
        println(time_str)
    end
    println("="^85, "\n")

    return subdirs
end

"""
    load_experiment(path::String)
    load_experiment(experiment_list::Vector{String}, index::Int)

Loads an experiment. Can take a direct path OR a list+index from `list_experiments`.
"""
function load_experiment(path::String)
    # Helper to fix path if pointing to dir
    file_path = endswith(path, ".jld2") ? path : joinpath(path, "results.jld2")

    if !isfile(file_path)
        error("Results file not found: $file_path")
    end

    printstyled("Loading: ", color=:green)
    println(basename(dirname(file_path)))
    
    data = load(file_path)
    return data["results"]
end

function load_experiment(experiment_list::Vector{String}, index::Int)
    if index < 1 || index > length(experiment_list)
        error("Index $index out of bounds (1 to $(length(experiment_list)))")
    end
    return load_experiment(experiment_list[index])
end


# ==============================================================================
# 4. INTERNAL HELPERS (Private)
# ==============================================================================

function _log_header(name)
    printstyled("\n>> EXPERIMENT: ", color=:magenta, bold=true)
    printstyled(name, "\n", color=:white, bold=true)
    println("-"^60)
end

function _log_step(step_num, msg)
    printstyled(" [$step_num] ", color=:cyan, bold=true)
    println(msg, "...")
end

function _log_info(msg)
    printstyled("     > ", color=:light_black)
    println(msg)
end

function _log_footer()
    println("-"^60)
    printstyled("DONE.\n", color=:green, bold=true)
end

function _read_experiment_metadata(path)
    # Tries to read config.json quickly for display
    # Returns NamedTuple (name, model)
    config_path = joinpath(path, "config.json")
    default = (name=basename(path), model="Unknown")
    
    if isfile(config_path)
        try
            json_str = read(config_path, String)
            cfg = JSON3.read(json_str)
            
            # Extract fields safely
            n = haskey(cfg, :name) ? cfg.name : default.name
            
            # Model might be a string type name or a struct
            m = default.model
            if haskey(cfg, :model)
                if cfg.model isa String
                    m = cfg.model
                elseif haskey(cfg.model, :type) # if JSON3 serialized type info
                    m = string(cfg.model.type)
                else
                    # Fallback: Just say "Configured"
                    m = "Model"
                end
            end
            return (name=n, model=m)
        catch
            return default
        end
    end
    return default
end

