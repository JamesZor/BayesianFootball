# current_development/market_feature_refactor/r03_eda_lambda_predictive_power.jl
# 
# This script runs an Exploratory Data Analysis to evaluate the predictive power 
# of the extracted market lambdas against the ACTUAL match outcomes (goals).
# We calculate the Log-Likelihood of the actual observed scores for each model.

using Revise
using DataFrames
using BayesianFootball
using StatsBase

println("==================================================")
println(" EDA: Lambda Predictive Power vs Actual Outcomes ")
println("==================================================")

# 1. Load Data
println("Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# Join matches with odds to ensure we only test matches where we have both
# Extract unique match IDs that have odds
odds_match_ids = unique(ds.odds.match_id);

# Filter matches DataFrame
valid_matches = subset(ds.matches, :match_id => ByRow(id -> id in odds_match_ids));

# Take a subset for the EDA to run quickly (e.g. 100 matches)
# Make sure we don't exceed the number of available matches
n_samples = max(100, nrow(valid_matches));
test_matches = valid_matches[1:n_samples, :];
test_match_ids = test_matches.match_id;

println("Running predictive analysis on $(n_samples) matches...")

# 2. Define our Configs
dp_config = BayesianFootball.Features.DoublePoissonMarketFeature()
dc_config = BayesianFootball.Features.DixonColesMarketFeature()
fc_config = BayesianFootball.Features.RegularizedFrankCopulaMarketFeature(prior_r=15.0, penalty_weight=0.05)
fc_unreg_config = BayesianFootball.Features.RegularizedFrankCopulaMarketFeature(penalty_weight=0.0)
dnb_config = BayesianFootball.Features.RegularizedDoubleNegativeBinomialMarketFeature(prior_r=22.0, penalty_weight=0.05)
dcnb_config = BayesianFootball.Features.RegularizedDixonColesNegativeBinomialMarketFeature(prior_r=22.0, penalty_weight=0.05)

# 3. Prepare result containers
results = DataFrame(
    match_id   = Int[],
    home_score = Int[],
    away_score = Int[],
    
    dp_loglike = Float64[],
    dc_loglike = Float64[],
    fc_loglike = Float64[],
    fc_unreg_loglike = Float64[],
    dnb_loglike = Float64[],
    dcnb_loglike = Float64[]
)

# Filter odds to just our test matches
odds_by_match = groupby(subset(ds.odds, :match_id => ByRow(in(test_match_ids))), :match_id)

# Helper function to safely get the probability of a specific score from a matrix
function get_score_prob(P::Matrix{Float64}, h_goals::Int, a_goals::Int)
    max_idx = size(P, 1) - 1
    # Cap goals to the matrix size to prevent BoundsError
    h_idx = min(h_goals, max_idx) + 1
    a_idx = min(a_goals, max_idx) + 1
    return max(P[h_idx, a_idx], 1e-10) # Prevent log(0)
end

for match_row in eachrow(test_matches)
    m_id = match_row.match_id
    
    # We need to gracefully handle column name variations (e.g. home_score vs home_goals)
    h_goals = Int(hasproperty(match_row, :home_score) ? match_row.home_score : match_row.home_goals)
    a_goals = Int(hasproperty(match_row, :away_score) ? match_row.away_score : match_row.away_goals)
    
    # Get the odds dataframe for this match
    match_odds_df = odds_by_match[(match_id = m_id,)]
    
    # Run the models
    res_dp = BayesianFootball.Features.fit_market_implied_parameters(match_odds_df, dp_config)
    res_dc = BayesianFootball.Features.fit_market_implied_parameters(match_odds_df, dc_config)
    res_fc = BayesianFootball.Features.fit_market_implied_parameters(match_odds_df, fc_config)
    res_fc_unreg = BayesianFootball.Features.fit_market_implied_parameters(match_odds_df, fc_unreg_config)
    res_dnb = BayesianFootball.Features.fit_market_implied_parameters(match_odds_df, dnb_config)
    res_dcnb = BayesianFootball.Features.fit_market_implied_parameters(match_odds_df, dcnb_config)
    
    # Rebuild the probability matrices
    P_dp = BayesianFootball.Features.build_probability_matrix(dp_config, res_dp.minimizer, 10)
    P_dc = BayesianFootball.Features.build_probability_matrix(dc_config, res_dc.minimizer, 10)
    P_fc = BayesianFootball.Features.build_probability_matrix(fc_config, res_fc.minimizer, 10)
    P_fc_unreg = BayesianFootball.Features.build_probability_matrix(fc_unreg_config, res_fc_unreg.minimizer, 10)
    P_dnb = BayesianFootball.Features.build_probability_matrix(dnb_config, res_dnb.minimizer, 10)
    P_dcnb = BayesianFootball.Features.build_probability_matrix(dcnb_config, res_dcnb.minimizer, 10)
    
    # Calculate Log-Likelihood
    ll_dp = log(get_score_prob(P_dp, h_goals, a_goals))
    ll_dc = log(get_score_prob(P_dc, h_goals, a_goals))
    ll_fc = log(get_score_prob(P_fc, h_goals, a_goals))
    ll_fc_unreg = log(get_score_prob(P_fc_unreg, h_goals, a_goals))
    ll_dnb = log(get_score_prob(P_dnb, h_goals, a_goals))
    ll_dcnb = log(get_score_prob(P_dcnb, h_goals, a_goals))
    
    push!(results, (
        m_id, h_goals, a_goals,
        ll_dp, ll_dc, ll_fc, ll_fc_unreg, ll_dnb, ll_dcnb
    ))
end

# 5. Analyze Predictive Power
println("\n--- Predictive Power Results (Log-Likelihood) ---")
println("Note: Higher (closer to 0) is better. Represents how well the market-extracted lambdas explained the actual goals.")

sum_ll_dp = sum(results.dp_loglike)
sum_ll_dc = sum(results.dc_loglike)
sum_ll_fc = sum(results.fc_loglike)
sum_ll_fc_unreg = sum(results.fc_unreg_loglike)
sum_ll_dnb = sum(results.dnb_loglike)
sum_ll_dcnb = sum(results.dcnb_loglike)

function print_logLoss()
println("\nTotal Log-Likelihood over $(n_samples) matches:")
println("Double Poisson:  ", round(sum_ll_dp, digits=2))
println("Dixon-Coles:     ", round(sum_ll_dc, digits=2), " (Diff vs DP: ", round(sum_ll_dc - sum_ll_dp, digits=2), ")")
println("Frank Copula (Reg): ", round(sum_ll_fc, digits=2), " (Diff vs DP: ", round(sum_ll_fc - sum_ll_dp, digits=2), ")")
println("Frank Copula (Unreg):", round(sum_ll_fc_unreg, digits=2), " (Diff vs DP: ", round(sum_ll_fc_unreg - sum_ll_dp, digits=2), ")")
println("Double NegBin:   ", round(sum_ll_dnb, digits=2), " (Diff vs DP: ", round(sum_ll_dnb - sum_ll_dp, digits=2), ")")
println("Dixon-Coles NegBin:", round(sum_ll_dcnb, digits=2), " (Diff vs DP: ", round(sum_ll_dcnb - sum_ll_dp, digits=2), ")")
end 

print_logLoss()



#=
Total Log-Likelihood over 990 matches:
Double Poisson:  -2710.04
Dixon-Coles:     -2711.11 (Diff vs DP: -1.07)
Frank Copula (Reg): -2719.21 (Diff vs DP: -9.17)
Frank Copula (Unreg):-2715.91 (Diff vs DP: -5.87)
Double NegBin:   -2717.51 (Diff vs DP: -7.47)
Dixon-Coles NegBin:-2718.92 (Diff vs DP: -8.88)
=#


# Let's see which model was "closest" to reality most often
wins = argmax.(eachrow(results[:, [:dp_loglike, :dc_loglike, :fc_loglike, :fc_unreg_loglike, :dnb_loglike, :dcnb_loglike]]));
wins_counts = countmap(wins)

function print_wins()
println("\nNumber of matches where the model assigned the highest probability to the actual score:")
println("Double Poisson:       ", get(wins_counts, :dp_loglike, 0))
println("Dixon-Coles:          ", get(wins_counts, :dc_loglike, 0))
println("Frank Copula (Reg):   ", get(wins_counts, :fc_loglike, 0))
println("Frank Copula (Unreg): ", get(wins_counts, :fc_unreg_loglike, 0))
println("Double NegBin:        ", get(wins_counts, :dnb_loglike, 0))
println("Dixon-Coles NegBin:   ", get(wins_counts, :dcnb_loglike, 0))
end 

print_wins()




#=
Number of matches where the model assigned the highest probability to the actual score:
Double Poisson:       174
Dixon-Coles:          284
Frank Copula (Reg):   120
Frank Copula (Unreg): 247
Double NegBin:        87
Dixon-Coles NegBin:   78
=#


println("\n==================================================")
println(" EDA Complete. Run this script via REPL to see the predictive edge of the Copula.")
