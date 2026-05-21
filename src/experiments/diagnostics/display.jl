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
        bad_folds = sort(unique(bad_rhat.fold))
        for (i, f) in enumerate(bad_folds)
            f_bad = subset(bad_rhat, :fold => x -> x .== f)
            prefix = (i == length(bad_folds)) ? "    └── " : "    ├── "
            printstyled(io, prefix, color=:light_black)
            printstyled(io, "Fold $f: ", color=:white)
            println(io, "$(nrow(f_bad)) unstable parameters (Max R-hat: $(round(maximum(f_bad.rhat), digits=3)))")
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
        
        # Display the worst offenders
        sort!(unstable, :adf_pvalue, rev=true)
        n_show = min(10, nrow(unstable))
        for i in 1:n_show
            row = unstable[i, :]
            prefix = (i == n_show && nrow(unstable) <= 10) ? "    └── " : "    ├── "
            printstyled(io, prefix, color=:light_black)
            printstyled(io, "$(row.parameter) [$(row.entity)] ", color=:white)
            
            p_val_str = isnan(row.adf_pvalue) ? "NaN" : string(round(row.adf_pvalue, digits=3))
            println(io, "- ADF p-val: $p_val_str")
        end
        if nrow(unstable) > 10
            printstyled(io, "    └── ... and $(nrow(unstable) - 10) more.\n", color=:light_black)
        end
    end
end
