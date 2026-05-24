# src/samplers/types.jl

export AbstractSamplerConfig, AbstractNUTSConfig
export AbstractOptimizationConfig, MAPConfig, MLEConfig

"""
    AbstractSamplerConfig
Abstract parent for all sampling strategies (NUTS, SGLD, VI, MAP).
"""
abstract type AbstractSamplerConfig end

"""
    AbstractNUTSConfig
Abstract parent for NUTS sampling configurations.
"""
abstract type AbstractNUTSConfig <: AbstractSamplerConfig end

"""
    AbstractOptimizationConfig
Abstract parent for Optimization mode estimation strategies (MAP, MLE).
"""
abstract type AbstractOptimizationConfig <: AbstractSamplerConfig end

"""
    MAPConfig
Configuration for Maximum A Posteriori (MAP) estimation.
"""
Base.@kwdef struct MAPConfig <: AbstractOptimizationConfig
    optimizer::Any = LBFGS()
    maxiters::Int = 1000
    adtype::Any = AutoReverseDiff(compile=true)
    initial_params::Union{Symbol, AbstractVector} = :prior
    show_progress::Bool = true
end

"""
    MLEConfig
Configuration for Maximum Likelihood Estimation (MLE).
"""
Base.@kwdef struct MLEConfig <: AbstractOptimizationConfig
    optimizer::Any = LBFGS()
    maxiters::Int = 1000
    adtype::Any = AutoReverseDiff(compile=true)
    initial_params::Union{Symbol, AbstractVector} = :prior
    show_progress::Bool = true
end
