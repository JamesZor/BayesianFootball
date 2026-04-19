# src/evaluation/types.jl

export AbstractScoringRule
export AbstractEvaluationResult, AbstractMetricComponent

# --- The Triggers ---
abstract type AbstractScoringRule end

# --- The Containers ---
abstract type AbstractEvaluationResult end
abstract type AbstractMetricComponent end

