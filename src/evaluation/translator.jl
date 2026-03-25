# src/evaluation/translator.jl

export to_dataframe_row

# Base Case: It's just a number. Wrap it in a NamedTuple.
function unroll(prefix::String, val::Real)
    return NamedTuple{(Symbol(prefix),)}((val,))
end

# Recursive Case: It's a nested component (like DistributionStats). Dive inside!
function unroll(prefix::String, comp::AbstractMetricComponent)
    keys = propertynames(comp)
    # Recursively unroll each field
    unrolled_tuples = [unroll("$(prefix)_$(k)", getproperty(comp, k)) for k in keys]
    return merge(unrolled_tuples...)
end

"""
    to_dataframe_row(model_name::String, metric_name::String, result::AbstractEvaluationResult)

Flattens any nested AbstractEvaluationResult into a single, flat NamedTuple 
that can be easily pushed into a DataFrame.
"""
function to_dataframe_row(exp::ExperimentResults, result::AbstractEvaluationResult)
    keys = propertynames(result)

    model_name = Experiments.get_model_name(exp)

    metric_name = get_metric_method_name(result)
    
    # Start the unrolling process at the top level
    unrolled_tuples = [unroll("$(metric_name)_$(k)", getproperty(result, k)) for k in keys]
    
    # Merge all the flattened tuples together
    flat_data = merge(unrolled_tuples...)
    
    # Attach the model name to the front
    return merge((model = model_name,), flat_data)
end
