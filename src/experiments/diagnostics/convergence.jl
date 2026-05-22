# src/experiments/diagnostics/convergence.jl

"""
    check_convergence(chains::ExperimentChains) -> ChainDiagnostic

Analyzes the pre-computed convergence metrics (R-hat and ESS) in the extracted chains.
Creates a ChainDiagnostic object that highlights any unstable folds or parameters.
"""
function check_convergence(chains::ExperimentChains)
    # We just need to filter out rows where rhat is missing/NaN
    df = filter(row -> !isnan(row.rhat), chains.df)
    
    return ChainDiagnostic(df)
end
