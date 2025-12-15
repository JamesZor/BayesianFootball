# src/training/types.jl

export AbstractExecutionStrategy, Independent, SequentialPriorUpdate
export TrainingConfig

abstract type AbstractExecutionStrategy end

"""
    Independent(; parallel=false, max_concurrent_splits=...)

Strategy where each data split is processed independently.
"""
Base.@kwdef struct Independent <: AbstractExecutionStrategy
    parallel::Bool = false
    max_concurrent_splits::Int = max(1, Threads.nthreads() ÷ 2)
end

"""
    SequentialPriorUpdate(...)
Strategy where the posterior from one split informs the prior for the next.
"""
Base.@kwdef struct SequentialPriorUpdate <: AbstractExecutionStrategy
    prior_update_method::Symbol = :MvNormalFit
end

"""
    TrainingConfig

Configuration for the training loop, including sampler, strategy, and resilience settings.

# Fields
- `sampler`: The sampling algorithm (NUTS, VI, etc.).
- `strategy`: Execution strategy (Independent, Sequential).
- `checkpoint_dir`: Path to save intermediate results. If `nothing`, runs in memory only.
- `cleanup_checkpoints`: If `true`, deletes checkpoints after successful completion.
"""
Base.@kwdef struct TrainingConfig
    sampler::Any # Typed as Any to avoid circular deps, or AbstractSamplerConfig if loaded
    strategy::AbstractExecutionStrategy
    checkpoint_dir::Union{String, Nothing} = nothing
    cleanup_checkpoints::Bool = false
end

function Base.show(io::IO, c::TrainingConfig)
    print(io, "TrainingConfig(strategy=$(nameof(typeof(c.strategy))), checkpointing=$(!isnothing(c.checkpoint_dir)))")
end
