module BivariatePrediction

using ..BivariateMaher # To get access to logpdf_bivariate_poisson
using BayesianFootball # For helper functions and types
using BayesianFootball.Predictions # For the output structs
using DataFrames
using StatsBase
using MCMCChains

# Export the new top-level function
export predict_bivariate_match_lines

"""
    compute_bivariate_xScore(λ_home, λ_away, γ, max_goals)

Computes the probability matrix for all scorelines up to max_goals using the
bivariate Poisson probability mass function.
"""
function compute_bivariate_xScore(λ_home::Float64, λ_away::Float64, γ::Float64, max_goals::Int)
    p = zeros(max_goals + 1, max_goals + 1)
    for h in 0:max_goals, a in 0:max_goals
        log_prob = BivariateMaher.logpdf_bivariate_poisson(h, a, λ_home, λ_away, γ)
        p[h+1, a+1] = isinf(log_prob) ? 0.0 : exp(log_prob)
    end
    return p
end

"""
    predict_bivariate_match_ft(model_def, chain, features, mapping)

Generates full-time predictions for the bivariate model.
"""
function predict_bivariate_match_ft(
    model_def::AbstractModelDefinition,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = BayesianFootball.extract_posterior_samples(model_def, chain)

    # Initialize vectors to store the full posterior for each market
    λ_home = zeros(n_samples)
    λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples)
    draw_probs = zeros(n_samples)
    away_win_probs = zeros(n_samples)

    cs_keys = vcat(
        [(h, a) for h in 0:3 for a in 0:3],
        ["other_home_win", "other_away_win", "other_draw"]
    )
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(
        key => zeros(n_samples) for key in cs_keys
    )
    
    under_05 = zeros(n_samples)
    under_15 = zeros(n_samples)
    under_25 = zeros(n_samples)
    under_35 = zeros(n_samples)
    btts = zeros(n_samples)

    for i in 1:n_samples
        λ_h, λ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a
        γ = posterior_samples.γ[i]

        prob_grid = compute_bivariate_xScore(λ_h, λ_a, γ, 10)

        hda = BayesianFootball.calculate_1x2(prob_grid)
        home_win_probs[i] = hda.hw
        draw_probs[i] = hda.dr
        away_win_probs[i] = hda.aw

        cs = BayesianFootball.calculate_correct_score_dict_ft(prob_grid)
        for (key, value) in cs
            correct_score[key][i] = value
        end

        under_05[i] = BayesianFootball.calculate_under_prob(prob_grid, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(prob_grid, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(prob_grid, 2)
        under_35[i] = BayesianFootball.calculate_under_prob(prob_grid, 3)
        btts[i] = BayesianFootball.calculate_btts(prob_grid)
    end

    # Return the formal struct defined in your types.jl file 
    return MatchFTPredictions(
        λ_home, λ_away, home_win_probs, draw_probs, away_win_probs,
        correct_score, under_05, under_15, under_25, under_35, btts
    )
end

"""
    predict_bivariate_match_ht(model_def, chain, features, mapping)

Generates half-time predictions for the bivariate model.
"""
function predict_bivariate_match_ht(
    model_def::AbstractModelDefinition,
    chain::Chains,
    features::NamedTuple,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = BayesianFootball.extract_posterior_samples(model_def, chain)

    # Initialize vectors
    λ_home = zeros(n_samples)
    λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples)
    draw_probs = zeros(n_samples)
    away_win_probs = zeros(n_samples)

    cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(
        key => zeros(n_samples) for key in cs_keys
    )
    
    under_05 = zeros(n_samples)
    under_15 = zeros(n_samples)
    under_25 = zeros(n_samples)

    for i in 1:n_samples
        λ_h, λ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a
        γ = posterior_samples.γ[i]

        prob_grid = compute_bivariate_xScore(λ_h, λ_a, γ, 10)
        
        hda = BayesianFootball.calculate_1x2(prob_grid)
        home_win_probs[i] = hda.hw
        draw_probs[i] = hda.dr
        away_win_probs[i] = hda.aw
        
        cs = BayesianFootball.calculate_correct_score_dict_ht(prob_grid)
        for (key, value) in cs
            correct_score[key][i] = value
        end
        
        under_05[i] = BayesianFootball.calculate_under_prob(prob_grid, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(prob_grid, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(prob_grid, 2)
    end
    
    # Return the formal HT struct
    return MatchHTPredictions(
        λ_home, λ_away, home_win_probs, draw_probs, away_win_probs,
        correct_score, under_05, under_15, under_25
    )
end

"""
    predict_bivariate_match_lines(model_def, round_chains, features, mapping)

Generates both HT and FT predictions for a single match using the bivariate model.
"""
function predict_bivariate_match_lines(
    model_def::AbstractModelDefinition,
    round_chains::TrainedChains,
    features::NamedTuple,
    mapping::MappedData
)
    ht_predict = predict_bivariate_match_ht(model_def, round_chains.ht, features, mapping)
    ft_predict = predict_bivariate_match_ft(model_def, round_chains.ft, features, mapping)
    return MatchLinePredictions(ht_predict, ft_predict)
end

end
