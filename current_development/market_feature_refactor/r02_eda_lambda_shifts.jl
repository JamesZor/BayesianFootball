# current_development/market_feature_refactor/r02_eda_lambda_shifts.jl
# 
# This script runs an Exploratory Data Analysis on a subset of the historical odds.
# It extracts lambdas using three different models (Double Poisson, Dixon-Coles, 
# and Regularized Frank Copula) to mathematically quantify the "stolen signal" 
# or bias caused by correlation parameters.

using Revise
using DataFrames
using BayesianFootball # This will load the new `src/features` logic we just ported!

println("==================================================")
println(" EDA: Lambda Shifts Across Inverse Models ")
println("==================================================")

# 1. Load Data
# We'll load the Scottish Lower dataset (or whichever is locally cached)
println("Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

# Filter to get a decent sample of unique match IDs
unique_match_ids = unique(ds.odds.match_id)[1:min(100, length(unique(ds.odds.match_id)))]
println("Running analysis on $(length(unique_match_ids)) matches...")

# 2. Define our Configs
dp_config = BayesianFootball.Features.DoublePoissonMarketFeature()
dc_config = BayesianFootball.Features.DixonColesMarketFeature()
# Applying regularization to prevent r from blowing up
fc_config = BayesianFootball.Features.RegularizedFrankCopulaMarketFeature(prior_r=15.0, penalty_weight=0.05)

# 3. Prepare result containers
results = DataFrame(
    match_id = Int[],
    
    dp_λ_home = Float64[],
    dp_λ_away = Float64[],
    
    dc_λ_home = Float64[],
    dc_λ_away = Float64[],
    dc_ρ      = Float64[],
    
    fc_λ_home = Float64[],
    fc_λ_away = Float64[],
    fc_r_home = Float64[],
    fc_r_away = Float64[],
    fc_κ      = Float64[]
)

# 4. Extract
odds_by_match = groupby(subset(ds.odds, :match_id => ByRow(in(unique_match_ids))), :match_id)

for match_df in odds_by_match
    match_id = first(match_df.match_id)
    
    # Run the three models
    res_dp = BayesianFootball.Features.fit_market_implied_parameters(match_df, dp_config)
    res_dc = BayesianFootball.Features.fit_market_implied_parameters(match_df, dc_config)
    res_fc = BayesianFootball.Features.fit_market_implied_parameters(match_df, fc_config)
    
    # Unroll via the generalized extraction methods
    p_dp = BayesianFootball.Features.extract_parameters(dp_config, res_dp.minimizer)
    p_dc = BayesianFootball.Features.extract_parameters(dc_config, res_dc.minimizer)
    p_fc = BayesianFootball.Features.extract_parameters(fc_config, res_fc.minimizer)
    
    push!(results, (
        match_id,
        p_dp.λ_home, p_dp.λ_away,
        p_dc.λ_home, p_dc.λ_away, p_dc.ρ,
        p_fc.λ_home, p_fc.λ_away, p_fc.r_home, p_fc.r_away, p_fc.κ
    ))
end

# 5. Analyze Differences
results.delta_λ_home_dc = results.dc_λ_home .- results.dp_λ_home
results.delta_λ_away_dc = results.dc_λ_away .- results.dp_λ_away

results.delta_λ_home_fc = results.fc_λ_home .- results.dp_λ_home
results.delta_λ_away_fc = results.fc_λ_away .- results.dp_λ_away

println("\n--- Analysis Results ---")

println("\nAverage Dixon-Coles Shift (Stolen Signal):")
println("Home λ shift: ", round(sum(results.delta_λ_home_dc) / nrow(results), digits=4))
println("Away λ shift: ", round(sum(results.delta_λ_away_dc) / nrow(results), digits=4))

println("\nAverage Frank Copula Shift:")
println("Home λ shift: ", round(sum(results.delta_λ_home_fc) / nrow(results), digits=4))
println("Away λ shift: ", round(sum(results.delta_λ_away_fc) / nrow(results), digits=4))

println("\nDispersion Stability (Checking Regularization):")
println("Average r_home: ", round(sum(results.fc_r_home) / nrow(results), digits=2))
println("Average r_away: ", round(sum(results.fc_r_away) / nrow(results), digits=2))

println("\nTop 5 matches with highest Correlation (ρ) and their Lambda shifts:")
sort!(results, :dc_ρ, rev=true)
display(first(results[:, [:match_id, :dc_ρ, :dp_λ_home, :dc_λ_home, :delta_λ_home_dc]], 5))

println("\n==================================================")
println(" EDA Complete. Run `using Plots; scatter(results.dc_ρ, results.delta_λ_home_dc)` to visualize the bias.")
