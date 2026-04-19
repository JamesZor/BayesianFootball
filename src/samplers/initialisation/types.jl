# src/samplers/initialisation/types.jl

export AbstractInitStrategy

"""
    AbstractInitStrategy
Abstract parent for all chain initialisation strategies.
"""
abstract type AbstractInitStrategy end


# interface
function get_init_params(model, strategy::AbstractInitStrategy, n_chains::Int)
    error("get_init_params not implemented for AbstractInitStrategy type: $(typeof(strategy))")
end
