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

# --- 2. The Bridge (Point-Mass Chain) ---

function safe_mode_extractor(estimate)
    # Check if the utility exists in Turing (handling version differences)
    if isdefined(Turing, :Optimisation) && isdefined(Turing.Optimisation, :vector_names_and_params)
        return Turing.Optimisation.vector_names_and_params(estimate)
    elseif isdefined(Turing, :vector_names_and_params)
        return Turing.vector_names_and_params(estimate)
    end

    # Manual extraction fallback
    vals_dict = hasproperty(estimate, :values) ? estimate.values : estimate.params
    
    varnames = String[]
    vals = Float64[]
    
    for (k, v) in pairs(vals_dict)
        name_base = string(k)
        if v isa Number
            push!(varnames, name_base)
            push!(vals, Float64(v))
        elseif v isa AbstractArray
            for i in eachindex(v)
                push!(varnames, "$(name_base)[$i]")
                push!(vals, Float64(v[i]))
            end
        end
    end
    
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
