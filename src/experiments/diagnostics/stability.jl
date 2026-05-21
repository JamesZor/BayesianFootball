# src/experiments/diagnostics/stability.jl

"""
    check_stability(chains::ExperimentChains) -> StabilityDiagnostic

Groups parameters across all temporal folds and computes the Augmented Dickey-Fuller
(ADF) test to evaluate stationarity (stability over time).
"""
function check_stability(chains::ExperimentChains)
    gdf = groupby(chains.df, [:parameter, :entity])
    
    results = []
    for g in gdf
        means = convert(Vector{Float64}, g.mean)
        n_folds = length(means)
        
        adf_pval = NaN
        is_stable = false
        
        # Need enough history to run a meaningful unit root test
        if n_folds >= 5
            try
                # Use default ADFTest. Usually tests if process has a unit root.
                # p-value < 0.05 rejects unit root -> implies stationarity (stable).
                test_result = ADFTest(means, :constant, 1) # 1 lag
                adf_pval = pvalue(test_result)
                is_stable = adf_pval < 0.05
            catch e
                # Can fail if the variance is zero (constant parameter across all folds)
                if std(means) ≈ 0.0
                    is_stable = true
                    adf_pval = 0.0
                end
            end
        else
            # If not enough folds, fall back to simple variance check
            is_stable = std(means) < 0.5 * abs(mean(means))
        end
        
        push!(results, Dict(
            :parameter => first(g.parameter),
            :entity => first(g.entity),
            :n_folds => n_folds,
            :mean_of_means => mean(means),
            :std_of_means => std(means),
            :skewness => StatsBase.skewness(means),
            :kurtosis => StatsBase.kurtosis(means),
            :min_val => minimum(means),
            :max_val => maximum(means),
            :adf_pvalue => adf_pval,
            :is_stable => is_stable
        ))
    end
    
    # Sort by stability so unstable parameters appear first
    df = sort(DataFrame(results), [:is_stable, :parameter, :entity])
    
    return StabilityDiagnostic(df)
end
