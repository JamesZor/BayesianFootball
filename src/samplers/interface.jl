# src/samplers/interface.jl

export run_sampler

"""
    run_sampler(model, config::AbstractSamplerConfig)

Main entry point for inference. Dispatches based on the configuration type.
"""
function run_sampler(turing_model, config::AbstractSamplerConfig)
    error("run_sampler not implemented for sampler config type: $(typeof(config))")
end
