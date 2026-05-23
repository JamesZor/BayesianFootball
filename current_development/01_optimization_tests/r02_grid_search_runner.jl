# current_development/01_optimization_tests/r02_grid_search_runner.jl

using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores)
using DataFrames

# Include the loaders
include("l02_grid_search_loaders.jl")

# 1. Load Data
println("\n[1] Loading Ireland Data...")
ds = Data.load_datastore_cached(Data.Ireland())

# 2. Define Parameter Grid for days_half_life
# You can customize these values as needed
half_lives_grid = [7, 14, 21, 50, 90.0, 180.0, 360.0]

# 3. Run Grid Search (Defaults to MAP estimation for speed)
# You can pass use_mle=true if you want maximum likelihood instead of MAP.
summary_df, results_dict = run_grid_search(ds, half_lives_grid; use_map=true, use_mle=false)

# 4. Print Results Table
println("\n============================================================")
println(" 🎉 Grid Search Completed!")
println("============================================================")
println(summary_df)

# 5. Identify Best Parameter
best_row = sort(summary_df, :logloss_model)[1, :]
println("\nOptimal Parameter Choice:")
println("  Best days_half_life: ", best_row.days_half_life)
println("  Model LogLoss:       ", best_row.logloss_model)

println("\nNote: The full experiment results are preserved in the `results_dict` dictionary.")
println("For example, to inspect the parameters of the best run:")
println("  best_results = results_dict[$(best_row.days_half_life)]")
println("  extract_chains(ds, best_results)")


    # Run this in your REPL to process the results_dict in memory:
metrics = [Evaluation.LogLoss()]
summary_rows = []

for (hl, res) in sort(collect(results_dict), by=x->x[1])
    println("\n---------------------------------------------------")
    println("Evaluating days_half_life = $hl...")

    # 1. Run evaluation
    eval_df = Evaluation.evaluate_experiments(metrics, [res], ds)

    # 2. Print columns and first row for debugging
    println("  Available columns: ", names(eval_df))
    if nrow(eval_df) > 0
        println("  First row data:    ", eval_df[1, :])
    end

    # 3. Extract logloss and n_obs directly
    logloss_model = NaN
    n_obs = 0
    if nrow(eval_df) > 0
        try
            # Direct extraction
            logloss_model = eval_df.logloss_overall_model_ll[1]
            n_obs = eval_df.logloss_overall_n_obs[1]
        catch e
            println("  Warning: failed to extract columns directly: ", e)
        end
    end

    # 4. Extract parameters
    local ν_xg_val = NaN
    local σ_market_val = NaN
    local ha_σ_val = NaN
    local kap_σ_val = NaN
    local lp_val = NaN

    try
        chains = BayesianFootball.Experiments.Diagnostics.extract_chains(ds, res)
        # Re-defining get_param_value inline to ensure no UndefVarError
        get_val(cdf, sym) = begin
            r = filter(row -> row.raw_symbol == sym, cdf)
            nrow(r) > 0 ? r.mean[1] : (
                r_str = filter(row -> string(row.raw_symbol) == string(sym), cdf);
                nrow(r_str) > 0 ? r_str.mean[1] : NaN
            )
        end
        ν_xg_val = get_val(chains.df, :ν_xg)
        σ_market_val = get_val(chains.df, :σ_market)
        ha_σ_val = get_val(chains.df, Symbol("ha.σ_γ"))
        kap_σ_val = get_val(chains.df, Symbol("kap.σ_κ"))
        lp_val = get_val(chains.df, :lp)
    catch e
        println("  Warning: parameter extraction failed: ", e)
    end

    push!(summary_rows, (
        days_half_life = hl,
        logloss_model = logloss_model,
        n_obs = n_obs,
        ν_xg = ν_xg_val,
        σ_market = σ_market_val,
        ha_σ_γ = ha_σ_val,
        kap_σ_κ = kap_σ_val,
        lp = lp_val
    ))
end

summary_df = DataFrame(summary_rows)
println("\n=== Re-evaluated Summary DataFrame ===")
println(summary_df)


sort(summary_df, :logloss_model) 
