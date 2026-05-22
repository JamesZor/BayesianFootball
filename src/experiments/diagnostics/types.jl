# src/experiments/diagnostics/types.jl

abstract type AbstractDiagnostic end

"""
    ExperimentChains

A long-format DataFrame containing MCMC parameter samples across multiple walk-forward splits.
Columns typically include: `fold`, `target_season`, `week`, `train_season`, `parameter`, `entity`, `mean`, `std`, etc.
"""
struct ExperimentChains
    df::DataFrame
end

"""
    ChainDiagnostic

Contains metrics for MCMC convergence per fold, such as R-hat and Effective Sample Size (ESS).
"""
struct ChainDiagnostic <: AbstractDiagnostic
    df::DataFrame
end

"""
    StabilityDiagnostic

Contains statistics for parameter stability across walk-forward folds, including 
the Augmented Dickey-Fuller (ADF) test p-values.
"""
struct StabilityDiagnostic <: AbstractDiagnostic
    df::DataFrame
end
