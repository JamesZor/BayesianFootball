# src/experiments/diagnostics/display.jl

function Base.show(io::IO, ::MIME"text/plain", chains::ExperimentChains)
    printstyled(io, "ExperimentChains\n", color=:magenta, bold=true)
    
    n_folds = length(unique(chains.df.fold))
    n_params = length(unique(chains.df.raw_symbol))
    
    printstyled(io, "  ├── ", color=:light_black)
    printstyled(io, "Folds: ", color=:white)
    printstyled(io, "$n_folds\n", color=:cyan)
    
    printstyled(io, "  ├── ", color=:light_black)
    printstyled(io, "Parameters tracked: ", color=:white)
    printstyled(io, "$n_params\n", color=:cyan)
    
    printstyled(io, "  └── ", color=:light_black)
    printstyled(io, "Rows: ", color=:white)
    printstyled(io, "$(nrow(chains.df))\n", color=:cyan)
end

function Base.show(io::IO, ::MIME"text/plain", diag::ChainDiagnostic)
    printstyled(io, "ChainDiagnostic (Convergence)\n", color=:cyan, bold=true)
    
    bad_rhat = subset(diag.df, :rhat => r -> r .> 1.05)
    
    if isempty(bad_rhat)
        printstyled(io, "  ✅ All parameters converged (R-hat ≤ 1.05)\n", color=:green)
    else
        printstyled(io, "  ⚠️  WARNING: $(nrow(bad_rhat)) instances of high R-hat detected!\n", color=:yellow)
        
        # Group by fold to show which folds had issues
        # Print all folds in order (even those with 0, if possible? We only have bad_rhat data here though)
        bad_folds = sort(unique(bad_rhat.fold))
        for (i, f) in enumerate(bad_folds)
            f_bad = subset(bad_rhat, :fold => x -> x .== f)
            prefix = (i == length(bad_folds)) ? "    └── " : "    ├── "
            printstyled(io, prefix, color=:light_black)
            printstyled(io, "Fold $f: ", color=:white)
            println(io, "$(nrow(f_bad)) unstable parameters (Max R-hat: $(round(maximum(f_bad.rhat), digits=3)))")
            
            # Display all the bad parameters for this fold
            bad_params = sort(unique(f_bad.raw_symbol))
            param_prefix = (i == length(bad_folds)) ? "        " : "    │   "
            printstyled(io, param_prefix, color=:light_black)
            println(io, "⚠️  " * join(string.(bad_params), ", "))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", diag::StabilityDiagnostic)
    printstyled(io, "StabilityDiagnostic (Stationarity)\n", color=:cyan, bold=true)
    
    unstable = subset(diag.df, :is_stable => s -> s .== false)
    
    if isempty(unstable)
        printstyled(io, "  ✅ All $(nrow(diag.df)) parameters exhibit cross-fold stability.\n", color=:green)
    else
        printstyled(io, "  ⚠️  WARNING: $(nrow(unstable)) parameters failed stability checks!\n", color=:yellow)
        
        # Display all unstable parameters
        sort!(unstable, :adf_pvalue, rev=true)
        n_unstable = nrow(unstable)
        for i in 1:n_unstable
            row = unstable[i, :]
            prefix = "    ├── "
            
            p_val_str = isnan(row.adf_pvalue) ? "NaN" : string(round(row.adf_pvalue, digits=3))
            
            # Use yellow warning symbol for unstable
            printstyled(io, prefix, "⚠️  ", color=:yellow)
            printstyled(io, "$(row.parameter) [$(row.entity)] ", color=:white)
            println(io, "- ADF p-val: $p_val_str")
            
            # Print statistical moments
            moments_prefix = "    │   "
            m_mean = round(row.mean_of_means, digits=3)
            m_std  = round(row.std_of_means, digits=3)
            m_min  = round(row.min_val, digits=3)
            m_max  = round(row.max_val, digits=3)
            m_skew = round(row.skewness, digits=3)
            m_kurt = round(row.kurtosis, digits=3)
            
            printstyled(io, moments_prefix, color=:light_black)
            println(io, "Mean: $m_mean | Std: $m_std | Min: $m_min | Max: $m_max | Skew: $m_skew | Kurtosis: $m_kurt")
        end
        
        # Display stable parameters
        stable = subset(diag.df, :is_stable => s -> s .== true)
        sort!(stable, :adf_pvalue)
        n_stable = nrow(stable)
        
        if n_stable > 0
            printstyled(io, "  ✅ STABLE PARAMETERS ($n_stable):\n", color=:green)
            for i in 1:n_stable
                row = stable[i, :]
                prefix = (i == n_stable) ? "    └── " : "    ├── "
                
                p_val_str = isnan(row.adf_pvalue) ? "NaN" : string(round(row.adf_pvalue, digits=3))
                
                printstyled(io, prefix, "✅ ", color=:green)
                printstyled(io, "$(row.parameter) [$(row.entity)] ", color=:white)
                println(io, "- ADF p-val: $p_val_str")
                
                moments_prefix = (i == n_stable) ? "        " : "    │   "
                m_mean = round(row.mean_of_means, digits=3)
                m_std  = round(row.std_of_means, digits=3)
                m_min  = round(row.min_val, digits=3)
                m_max  = round(row.max_val, digits=3)
                m_skew = round(row.skewness, digits=3)
                m_kurt = round(row.kurtosis, digits=3)
                
                printstyled(io, moments_prefix, color=:light_black)
                println(io, "Mean: $m_mean | Std: $m_std | Min: $m_min | Max: $m_max | Skew: $m_skew | Kurtosis: $m_kurt")
            end
        end
    end
end
