# src/samplers/types.jl

export AbstractSamplerConfig

"""
    AbstractSamplerConfig
Abstract parent for all sampling strategies (NUTS, SGLD, VI, MAP).
"""
abstract type AbstractSamplerConfig end
