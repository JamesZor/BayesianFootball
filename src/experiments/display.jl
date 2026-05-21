# src/experiments/display.jl

using ..Models
using ..Data
using ..Training
using ..Samplers
using Distributions

# --- 1. Helper for ASCII Trees ---
const TREE_V = "|"
const TREE_J = "+--"
const TREE_L = "└──" 
const TREE_M = "├──"
const TREE_S = "   "

# --- 2. ExperimentTask (The Dashboard) ---
function Base.show(io::IO, ::MIME"text/plain", task::ExperimentTask)
    printstyled(io, "ExperimentTask", color=:magenta, bold=true)
    printstyled(io, " [Ready for run_experiment(task)]", color=:yellow, bold=true)
    printstyled(io, "\n (Bundled DataStore & Config)\n", color=:light_black)
    println(io, "=========")

    # High-level Summary
    printstyled(io, "  Name:       ", color=:light_black)
    printstyled(io, task.config.name, "\n", color=:white, bold=true)
    printstyled(io, "  Model:      ", color=:light_black)
    printstyled(io, nameof(typeof(task.config.model)), "\n", color=:cyan)
    printstyled(io, "  Save Dir:   ", color=:light_black)
    printstyled(io, task.config.save_dir, "\n", color=:white)
    
    # Calculate Folds
    try
        folds = length(Data.create_id_boundaries(task.ds, task.config.splitter))
        printstyled(io, "  Total Folds: ", color=:light_black)
        printstyled(io, folds, "\n", color=:cyan, bold=true)
    catch
        printstyled(io, "  Total Folds: ", color=:light_black)
        printstyled(io, "Unknown (Run create_id_boundaries)\n", color=:cyan)
    end
    println(io, "=========\n")

    # Data Source
    printstyled(io, "[Data Source] ", color=:cyan, bold=true)
    printstyled(io, "access via: task.ds\n", color=:light_black)
    show(io, task.ds)
    println(io, "\n")

    # Pass the rest to ExperimentConfig display
    show(io, MIME("text/plain"), task.config)
end

function Base.show(io::IO, task::ExperimentTask)
    print(io, "ExperimentTask(ds=$(typeof(task.ds.segment)), config=\"$(task.config.name)\")")
end

# --- 3. ExperimentConfig ---
function Base.show(io::IO, ::MIME"text/plain", config::ExperimentConfig)
    printstyled(io, "[Configuration] ", color=:cyan, bold=true)
    printstyled(io, "access via: task.config\n", color=:light_black)
    
    printstyled(io, "  Tags: ", color=:light_black)
    println(io, isempty(config.tags) ? "None" : join(config.tags, ", "))
    println(io)

    # Model Section
    printstyled(io, "[Model] ", color=:cyan, bold=true)
    printstyled(io, "access via: task.config.model\n", color=:light_black)
    show(io, MIME("text/plain"), config.model) 
    println(io)

    # Splitter Section
    printstyled(io, "[Splitter] ", color=:cyan, bold=true)
    printstyled(io, "access via: task.config.splitter\n", color=:light_black)
    show(io, MIME("text/plain"), config.splitter)
    println(io)

    # Training Section
    printstyled(io, "[Training Strategy] ", color=:cyan, bold=true)
    printstyled(io, "access via: task.config.training_config\n", color=:light_black)
    show(io, MIME("text/plain"), config.training_config)
    println(io)
end

function Base.show(io::IO, config::ExperimentConfig)
    print(io, "ExperimentConfig(name=\"$(config.name)\")")
end



# --- 5. Training Config ---
function Base.show(io::IO, ::MIME"text/plain", t::TrainingConfig)
    strat_type = nameof(typeof(t.strategy))
    strat_info = t.strategy.parallel ? "Parallel (Max=$(t.strategy.max_concurrent_splits))" : "Sequential"
    
    printstyled(io, "  $strat_type Strategy", color=:green, bold=true)
    println(io)
    printstyled(io, "  ├── ", color=:light_black)
    printstyled(io, "Execution: ", color=:white)
    printstyled(io, strat_info, "\n", color=:cyan)
    
    # Sampler
    sampler_name = nameof(typeof(t.sampler))
    printstyled(io, "  └── ", color=:light_black)
    printstyled(io, "$sampler_name Sampler\n", color=:green, bold=true)
    
    s = t.sampler
    if hasfield(typeof(s), :n_samples)
        printstyled(io, "      ├── ", color=:light_black)
        printstyled(io, "Samples: ", color=:white)
        printstyled(io, "$(s.n_samples)\n", color=:cyan)
        
        printstyled(io, "      ├── ", color=:light_black)
        printstyled(io, "Warmup:  ", color=:white)
        printstyled(io, "$(s.n_warmup)\n", color=:cyan)
        
        printstyled(io, "      ├── ", color=:light_black)
        printstyled(io, "Chains:  ", color=:white)
        printstyled(io, "$(s.n_chains)\n", color=:cyan)

        if hasfield(typeof(s), :initialisation)
            printstyled(io, "      ├── ", color=:light_black)
            printstyled(io, "Init:    ", color=:white)
            printstyled(io, "$(nameof(typeof(s.initialisation)))\n", color=:cyan)
        end
        
        if hasfield(typeof(s), :show_progress)
            printstyled(io, "      ├── ", color=:light_black)
            printstyled(io, "Progress:", color=:white)
            printstyled(io, " $(s.show_progress)\n", color=:cyan)
        end

        if hasfield(typeof(t.strategy), :max_concurrent_splits)
            max_threads = s.n_chains * t.strategy.max_concurrent_splits
            printstyled(io, "      └── ", color=:light_black)
            printstyled(io, "Max CPU Threads: ", color=:white)
            printstyled(io, "$max_threads ", color=:cyan, bold=true)
            printstyled(io, "(Chains: $(s.n_chains) × Splits: $(t.strategy.max_concurrent_splits))\n", color=:light_black)
        else
            printstyled(io, "      └── ", color=:light_black)
            printstyled(io, "Max CPU Threads: ", color=:white)
            printstyled(io, "$(s.n_chains)\n", color=:cyan, bold=true)
        end
    end
end
