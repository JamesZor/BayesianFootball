# src/experiments/types.jl



abstract type AbstractSplitter end

"""
Defines a single static train/test split.
"""
struct StaticSplit <: AbstractSplitter
    train_seasons::Vector{String}
end

"""
Defines an expanding window CV strategy (your Config_f).
This struct holds the *parameters* for your TimeSeriesSplits iterator.
"""
struct ExpandingWindowCV <: AbstractSplitter
    base_seasons::Vector{String}
    target_seasons::Vector{String}
    round_col::Symbol
    ordering::Symbol 
end


# --- MODIFIED: Experiment struct ---
struct Experiment
    name::String
    model::AbstractFootballModel
    splitter::AbstractSplitter 
    sampler_config::AbstractTrainingMethod
end



