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
    # vocabulary = Features.create_vocabulary(data_store, config.model)

    # --- Step 2: Data Splitting ---
    _log_step(2, "Generating Data Splits")
    data_splits = Data.create_data_splits(data_store, config.splitter)
    _log_info("Generated $(length(data_splits)) splits")

    # --- Step 3: Feature Engineering ---
    _log_step(3, "Building Feature Sets")
    feature_sets = Features.create_features(
    Data.create_data_splits(data_store, config.splitter), 
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
# FIX: 1 
    return ExperimentResults(
        config,
        training_results,
        nothing,
        default_path
    )
end

# ==============================================================================
# 2. PERSISTENCE (Saving)
# ==============================================================================

"""
    save_experiment(results::ExperimentResults; path=nothing, quiet=false)

Persists the experiment results to disk.
Creates a `meta.json` sidecar for fast listing.
"""
function save_experiment(results::ExperimentResults; path=nothing, quiet=false)
    target_path = isnothing(path) ? results.save_path : path
    mkpath(target_path)

    # 1. Save Binary (Heavy Data)
    jldsave(joinpath(target_path, "results.jld2"); results)

    # 2. Save Config (Full JSON)
    open(joinpath(target_path, "config.json"), "w") do io
        JSON3.pretty(io, results.config)
    end

    # 3. Save Metadata Sidecar (Lightweight for Listing)
    # We explicitly extract the Type Names here
    meta = Dict(
        :name => results.config.name,
        :model => string(nameof(typeof(results.config.model))),
        :splitter => string(nameof(typeof(results.config.splitter))),
        :sampler => string(nameof(typeof(results.config.training_config.sampler))),
        :timestamp => basename(target_path) # usually contains timestamp
    )
    
    open(joinpath(target_path, "meta.json"), "w") do io
        JSON3.pretty(io, meta)
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

Scans the directory for `meta.json` files and prints a detailed dashboard.
"""
function list_experiments(dir::String;data_dir::String="./data")
    base_dir = joinpath(data_dir, dir)

    if !isdir(base_dir)
        println("Directory not found: $base_dir")
        return String[]
    end

    subdirs = filter(isdir, readdir(base_dir, join=true))
    sort!(subdirs, by=mtime, rev=true)

    if isempty(subdirs)
        println("No experiments found in $base_dir")
        return String[]
    end

    # Header
    println("\n Experiments in: $base_dir")
    println("="^110)
    println(
        rpad("IDX", 4), " | ", 
        rpad("NAME", 25), " | ", 
        rpad("MODEL", 20), " | ", 
        rpad("SPLITTER", 18), " | ", 
        rpad("SAMPLER", 15), " | ", 
        "PATH ID"
    )
    println("-"^110)

    for (i, path) in enumerate(subdirs)
        # Try to read the new meta.json, fall back to config parsing, fall back to folder name
        m = _read_meta(path)

        # Formatting
        idx_str = "[$i]"
        
        # Truncation logic
        name_s = length(m.name) > 23 ? m.name[1:23] * ".." : m.name
        model_s = length(m.model) > 18 ? m.model[1:18] * ".." : m.model
        split_s = length(m.splitter) > 16 ? m.splitter[1:16] * ".." : m.splitter
        samp_s = length(m.sampler) > 13 ? m.sampler[1:13] * ".." : m.sampler

        # Color the latest run
        c = i == 1 ? :white : :light_black
        
        printstyled(rpad(idx_str, 4), " | ", color=:cyan, bold=(i==1))
        printstyled(rpad(name_s, 25), " | ", color=c, bold=(i==1))
        printstyled(rpad(model_s, 20), " | ", color=c)
        printstyled(rpad(split_s, 18), " | ", color=c)
        printstyled(rpad(samp_s, 15), " | ", color=c)
        println(basename(path))
    end
    println("="^110, "\n")

    return subdirs
end

"""
    load_experiment(path_or_list, [index])
"""
function load_experiment(path::String)
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
        error("Index $index out of bounds.")
    end
    return load_experiment(experiment_list[index])
end

# ==============================================================================
# 4. INTERNAL HELPERS (Private)
# ==============================================================================

function _read_meta(path)
    meta_path = joinpath(path, "meta.json")
    default = (name=basename(path), model="?", splitter="?", sampler="?")

    if isfile(meta_path)
        try
            return JSON3.read(read(meta_path, String))
        catch
            return default
        end
    end

    # Fallback: Try to read old config.json if meta.json doesn't exist yet
    config_path = joinpath(path, "config.json")
    if isfile(config_path)
        try
            cfg = JSON3.read(read(config_path, String))
            return (
                name = get(cfg, :name, default.name),
                model = "Legacy Config",
                splitter = "?",
                sampler = "?"
            )
        catch
        end
    end
    return default
end

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

