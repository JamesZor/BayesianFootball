# src/evaluation/interfaces.jl 

function compute_metric(metric::AbstractScoringRule, exp::ExperimentResults, ds::DataStore)
    error("compute_metric not implemented for $(typeof(metric))")
end

function get_metric_method_name(metric::AbstractEvaluationResult)
    error("get_metric_method_name not implemented for $(typeof(metric))")
end

function get_metric_method_name(metric::AbstractScoringRule)
    error("get_metric_method_name not implemented for $(typeof(metric))")
end

