# src/experiments/display.jl

using ..Models
using ..Data
using ..Training
using ..Samplers
using Distributions

# --- 1. Helper for ASCII Trees ---
# Using standard ASCII to ensure compatibility with bare-bones terminals
const TREE_V = "|"
const TREE_J = "+--"
const TREE_L = "+--" 
const TREE_S = "   " # Spacer

# --- 2. ExperimentConfig (The Dashboard) ---
function Base.show(io::IO, ::MIME"text/plain", config::ExperimentConfig)
    println(io, "Experiment: ", config.name)
    println(io, "========================================")
    
    # Model Section
    println(io, "[Model]")
    show(io, MIME("text/plain"), config.model) # Delegate to Model's show
    println(io, "")

    # Splitter Section
    println(io, "[Data & Splitter]")
    show(io, MIME("text/plain"), config.splitter) # Delegate to Splitter's show
    println(io, "")

    # Training Section
    println(io, "[Training Strategy]")
    show(io, MIME("text/plain"), config.training_config) # Delegate to Training
    println(io, "")

    # Metadata
    print(io, "[Tags]: ")
    println(io, isempty(config.tags) ? "None" : join(config.tags, ", "))
end

# --- 3. Models (Math Notation) ---
# This assumes your models have a 'prior' field or similar. 
# We target the abstract type to cover all models.

function Base.show(io::IO, ::MIME"text/plain", m::AbstractFootballModel)
    # Get the specific type name (e.g., "StaticPoisson")
    model_name = nameof(typeof(m))
    
    # ASCII Math representation
    println(io, TREE_S, model_name)
    println(io, TREE_S, "-----------------")
    
    # Introspect fields to find distributions
    # This automatically finds fields like 'prior' and prints them like "prior ~ Normal(...)"
    for name in fieldnames(typeof(m))
        val = getfield(m, name)
        if val isa Distribution
            # Convert Distribution to string but clean it up if needed
            dist_str = string(val) 
            println(io, TREE_S, "  ", name, " ~ ", dist_str)
        else
            println(io, TREE_S, "  ", name, " = ", val)
        end
    end
end

# --- 4. Splitter (Config View) ---

function Base.show(io::IO, ::MIME"text/plain", s::CVConfig)
    println(io, TREE_S, "Type: CVConfig")
    println(io, TREE_S, TREE_J, " Tournaments: ", s.tournament_ids)
    println(io, TREE_S, TREE_J, " Target:      ", join(s.target_seasons, ", "))
    println(io, TREE_S, TREE_J, " History:     ", s.history_seasons, " seasons")
    println(io, TREE_S, TREE_L, " Dynamics:    ", s.dynamics_col, " (Warmup=", s.warmup_period, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::ExpandingWindowCV)
    println(io, TREE_S, "Type: ExpandingWindowCV")
    println(io, TREE_S, TREE_J, " Test Seasons: ", join(s.test_seasons, ", "))
    println(io, TREE_S, TREE_L, " Window Col:   ", s.window_col)
end

# --- 5. Training Config ---

function Base.show(io::IO, ::MIME"text/plain", t::TrainingConfig)
    # Strategy
    strat_type = nameof(typeof(t.strategy))
    strat_info = t.strategy.parallel ? "Parallel (Max=$(t.strategy.max_concurrent_splits))" : "Sequential"
    
    println(io, TREE_S, "Strategy: ", strat_type)
    println(io, TREE_S, TREE_L, " Mode: ", strat_info)
    
    # Sampler
    println(io, TREE_S, "Sampler:  ", nameof(typeof(t.sampler)))
    s = t.sampler
    # Assuming NUTSConfig has these fields
    if hasfield(typeof(s), :n_samples)
        println(io, TREE_S, TREE_S, "Samples: ", s.n_samples)
        println(io, TREE_S, TREE_S, "Chains:  ", s.n_chains)
        println(io, TREE_S, TREE_S, "Warmup:  ", s.n_warmup)
    end
end
