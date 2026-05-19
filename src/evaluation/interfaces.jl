# src/evaluation/interfaces.jl 

"""
    compute_metric(metric::AbstractScoringRule, exp::ExperimentResults, ds::DataStore, latents::Any)

Main implementation point for metrics. Takes pre-extracted latents to avoid redundant computations.
"""
function compute_metric(metric::AbstractScoringRule, exp::ExperimentResults, ds::DataStore, latents::Any)
    error("compute_metric not implemented for $(typeof(metric)) with provided latents")
end

"""
    compute_metric(metric::AbstractScoringRule, exp::ExperimentResults, ds::DataStore)

Wrapper that extracts latents automatically if not provided.
"""
function compute_metric(metric::AbstractScoringRule, exp::ExperimentResults, ds::DataStore)
    latents = Experiments.extract_oos_predictions(ds, exp)
    return compute_metric(metric, exp, ds, latents)
end

function get_metric_method_name(metric::AbstractEvaluationResult)
    error("get_metric_method_name not implemented for $(typeof(metric))")
end

function get_metric_method_name(metric::AbstractScoringRule)
    error("get_metric_method_name not implemented for $(typeof(metric))")
end
