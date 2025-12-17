# src/training/types.jl

export AbstractExecutionStrategy, Independent, SequentialPriorUpdate
export TrainingConfig
export TrainingResults

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


"""
    TrainingResults{C, M}

A container for the results of a training run.
- `C`: Type of the Model Result (e.g., Turing.Chains)
- `M`: Type of the Metadata (e.g., SplitMetaData)
"""
struct TrainingResults{C, M} <: AbstractVector{Tuple{C, M}}
    items::Vector{Tuple{C, M}}
end

# Allow construction from a loosely typed vector by attempting to tighten types
function TrainingResults(items::Vector)
    # This comprehension forces Julia to look at the actual types of elements
    # and infer the tightest common type (e.g., Vector{Tuple{Chains, Meta}})
    tightened = [i for i in items] 
    return TrainingResults(tightened)
end

# --- Interface Implementation ---
Base.size(tr::TrainingResults) = size(tr.items)
Base.getindex(tr::TrainingResults, i::Int) = getindex(tr.items, i)
Base.setindex!(tr::TrainingResults, v, i::Int) = setindex!(tr.items, v, i)
Base.IndexStyle(::Type{<:TrainingResults}) = IndexLinear()

# Optional: Pretty Printing
function Base.show(io::IO, ::MIME"text/plain", tr::TrainingResults)
    C = eltype(tr).parameters[1]
    n = length(tr)
    print(io, "TrainingResults: $n completed splits ($C)")
end
