# current_development/diagnostics_ppc/l01_ppc_diagnostics.jl

using Plots
plotlyjs() # Interactive HTML backend
using DataFrames
using Distributions
using Statistics
using StatsBase

# We need the custom RobustNegativeBinomial for simulating goals
using BayesianFootball.MyDistributions

"""
Extracts predictive parameters from the model for all matches in a fold, 
and simulates goal outcomes to create the Posterior Predictive Distribution.
"""
function generate_pp_goals(results::BayesianFootball.Experiments.ExperimentResults, ds::BayesianFootball.Data.DataStore; fold_idx=1)
    # Get the chain and meta for the specified fold
    chain, meta = results.training_results[fold_idx]
    
    # We need the matches dataframe for this fold
    matches_df = subset(ds.matches, :match_id => ByRow(id -> id in meta.target_match_ids))
    
    # Reconstruct features to pass to extractor
    boundaries = BayesianFootball.Data.create_id_boundaries(ds, results.config.splitter)
    boundary = boundaries[fold_idx][1]
    feature_set = BayesianFootball.Features.create_features(boundary, ds, results.config.model, :match_month)
    
    # Extract match parameters (λ and r arrays for every sample)
    param_dict = BayesianFootball.Models.PreGame.extract_parameters(results.config.model, matches_df, feature_set, chain)
    
    n_matches = nrow(matches_df)
    n_samples = size(chain, 1) * size(chain, 3)
    
    pp_home_goals = zeros(Int, n_samples, n_matches)
    pp_away_goals = zeros(Int, n_samples, n_matches)
    
    obs_home_goals = zeros(Int, n_matches)
    obs_away_goals = zeros(Int, n_matches)
    
    for (i, row) in enumerate(eachrow(matches_df))
        mid = Int(row.match_id)
        obs_home_goals[i] = row.home_goals
        obs_away_goals[i] = row.away_goals
        
        match_params = param_dict[mid]
        
        # Generate posterior predictive samples
        for s in 1:n_samples
            λ_h = match_params.λ_h[s]
            r_h = match_params.r_h[s]
            pp_home_goals[s, i] = rand(RobustNegativeBinomial(r_h, λ_h))
            
            λ_a = match_params.λ_a[s]
            r_a = match_params.r_a[s]
            pp_away_goals[s, i] = rand(RobustNegativeBinomial(r_a, λ_a))
        end
    end
    
    return pp_home_goals, pp_away_goals, obs_home_goals, obs_away_goals
end

"""
Plots the density/histogram overlay of simulated goals vs observed goals.
"""
function plot_ppc(pp_goals::Matrix{Int}, obs_goals::Vector{Int}, plot_title::String, filename::String)
    # Count observed frequencies
    obs_counts = countmap(obs_goals)
    max_goal = max(maximum(obs_goals), 8)
    
    # Average the simulated frequencies across all posterior samples
    sim_counts_avg = zeros(Float64, max_goal + 1)
    
    n_samples = size(pp_goals, 1)
    for s in 1:n_samples
        sample_counts = countmap(pp_goals[s, :])
        for (g, c) in sample_counts
            if g <= max_goal
                sim_counts_avg[g+1] += c / n_samples
            end
        end
    end
    
    # Prepare data for plotting
    x_vals = 0:max_goal
    
    # Normalize to proportions for density-like overlay
    total_obs = length(obs_goals)
    total_sim = size(pp_goals, 2)
    
    obs_y = [get(obs_counts, g, 0) / total_obs for g in x_vals]
    sim_y = [sim_counts_avg[g+1] / total_sim for g in x_vals]
    
    p = Plots.bar(x_vals, obs_y, label="Observed Goals", alpha=0.7, color=:darkred, 
                  title=plot_title, xlabel="Goals Scored", ylabel="Probability Density", 
                  lw=0, bar_width=0.6)
                  
    # Overlay the posterior predictive mean
    Plots.bar!(p, x_vals, sim_y, label="Posterior Predictive (Simulated)", alpha=0.5, color=:steelblue, 
               lw=0, bar_width=0.8)
    
    Plots.savefig(p, filename)
    println("✅ PPC Plot saved to $filename")
    return p
end
