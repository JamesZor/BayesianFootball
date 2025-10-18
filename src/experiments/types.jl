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


"""
Defines what to predict (your Config_p).
"""
struct PredictionConfig
    markets::Vector{Symbol} # e.g., [:1x2, :over_under_25, :btts]
    calculate_ev::Bool
    calculate_kelly::Bool
end


# --- MODIFIED: Experiment struct ---
struct Experiment
    name::String
    model::AbstractFootballModel        # Your M
    splitter::AbstractSplitter        # <-- REPLACES FeatureConfig
    sampler_config::AbstractTrainingMethod  # Your Config_s
    prediction_config::PredictionConfig # Your Config_p
end



