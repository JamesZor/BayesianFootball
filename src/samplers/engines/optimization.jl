# src/samplers/engines/optimization.jl

using MCMCChains

# --- 1. Execution ---

function run_sampler(turing_model, config::MAPConfig)
    # println("Sampling with MAP...")
    
    # Handle initial parameters gracefully if needed, Turing uses InitFromPrior() by default
    kwargs = Dict{Symbol, Any}()
    if config.initial_params != :prior
        kwargs[:initial_params] = InitFromParams(config.initial_params)
    end

    map_estimate = maximum_a_posteriori(
        turing_model, 
        config.optimizer; 
        adtype=config.adtype,
        maxiters=config.maxiters,
        kwargs...
    )
    
    if config.show_progress
        println("Optimization Result (MAP):")
        println("  Log-Probability: ", map_estimate.lp)
        println("  Retcode: ", map_estimate.optim_result.retcode)
    end

    return mode_result_to_chains(turing_model, map_estimate)
end

function run_sampler(turing_model, config::MLEConfig)
    # println("Sampling with MLE...")
    
    kwargs = Dict{Symbol, Any}()
    if config.initial_params != :prior
        kwargs[:initial_params] = InitFromParams(config.initial_params)
    end

    mle_estimate = maximum_likelihood(
        turing_model, 
        config.optimizer; 
        adtype=config.adtype,
        maxiters=config.maxiters,
        kwargs...
    )

    if config.show_progress
        println("Optimization Result (MLE):")
        println("  Log-Likelihood: ", mle_estimate.lp)
        println("  Retcode: ", mle_estimate.optim_result.retcode)
    end

    return mode_result_to_chains(turing_model, mle_estimate)
end

import StatsBase

function _flatten_params!(names, vals, prefix::String, x::Number)
    push!(names, prefix)
    push!(vals, Float64(x))
end

function _flatten_params!(names, vals, prefix::String, x::AbstractArray{<:Number})
    for i in eachindex(x)
        push!(names, "$(prefix)[$i]")
        push!(vals, Float64(x[i]))
    end
end

function _flatten_params!(names, vals, prefix::String, x)
    # Generic fallback for NamedTuple or VarNamedTuple
    for (k, v) in pairs(x)
        new_prefix = isempty(prefix) ? string(k) : "$(prefix).$(k)"
        _flatten_params!(names, vals, new_prefix, v)
    end
end

function safe_mode_extractor(estimate)
    # 1. Try standard StatsBase interface (used by newer Turing versions)
    try
        names = String.(StatsBase.coefnames(estimate))
        vals = Float64.(StatsBase.coef(estimate))
        if !isempty(names) && names[1] != "1"
            return names, vals
        end
    catch
    end

    # 2. Try Turing.Optimisation.vector_names_and_params (older Turing)
    if isdefined(Turing, :Optimisation) && isdefined(Turing.Optimisation, :vector_names_and_params)
        return Turing.Optimisation.vector_names_and_params(estimate)
    elseif isdefined(Turing, :vector_names_and_params)
        return Turing.vector_names_and_params(estimate)
    end

    # 3. Recursive manual flattening (Rock-solid fallback)
    # Prefer `params` (VarNamedTuple) over `values` (often just a raw Vector)
    vals_dict = hasproperty(estimate, :params) ? estimate.params : estimate.values
    
    varnames = String[]
    vals = Float64[]
    _flatten_params!(varnames, vals, "", vals_dict)
    
    return varnames, vals
end

"""
    mode_result_to_chains(model, estimate::Turing.Optimisation.ModeResult)

Converts a point estimate (MAP/MLE) into a 1-sample `MCMCChains.Chains` object.
This ensures all downstream extractors, predictions, and diagnostic tools that 
expect an array of samples continue to work unmodified.
"""
function mode_result_to_chains(model, estimate)
    # Get variable names and their optimized values safely
    varnames, values = safe_mode_extractor(estimate)
    
    # Extract string names for the Chain structure
    names = String.(varnames)
    
    # Optional: include the log probability in the chain metadata
    push!(names, "lp")
    push!(values, estimate.lp)
    
    # Shape must be (samples, parameters, chains)
    # Here: 1 sample, length(values) parameters, 1 chain
    data = reshape(values, 1, length(values), 1)
    
    return MCMCChains.Chains(data, names)
end
