module BivariatePrediction

using ..BivariateMaher # To get access to logpdf_bivariate_poisson
using BayesianFootball
using DataFrames
using StatsBase
using MCMCChains

export predict_bivariate_match_ft

"""
    compute_bivariate_xScore(λ_home, λ_away, γ, max_goals)

Computes the probability matrix for all scorelines up to max_goals using the
bivariate Poisson probability mass function.
"""
function compute_bivariate_xScore(λ_home::Float64, λ_away::Float64, γ::Float64, max_goals::Int)
    p = zeros(max_goals + 1, max_goals + 1)
    for h in 0:max_goals, a in 0:max_goals
        # Use the logpdf function we already built and convert it back to a probability
        log_prob = BivariateMaher.logpdf_bivariate_poisson(h, a, λ_home, λ_away, γ)
        # We need to handle cases where the logp is -Inf (probability is zero)
        p[h+1, a+1] = isinf(log_prob) ? 0.0 : exp(log_prob)
    end
    return p
end

"""
    predict_bivariate_match_ft(model_def, chain, features)

Generates full-time predictions for all markets by averaging the analytical
probabilities over the entire posterior chain.
"""
function predict_bivariate_match_ft(
    model_def::AbstractModelDefinition,
    chain::Chains,
    features::NamedTuple,
)
    n_samples = size(chain, 1) * size(chain, 3)
    posterior_samples = BayesianFootball.extract_posterior_samples(model_def, chain)

    # We will accumulate the probabilities for each market here
    home_win_prob_sum = 0.0
    draw_prob_sum = 0.0
    away_win_prob_sum = 0.0

    # Loop through each sample in the posterior chain
    for i in 1:n_samples
        # 1. Get the parameters for this specific sample
        λ_h, λ_a = BayesianFootball.get_goal_rates(model_def, posterior_samples, i, features)
        γ = posterior_samples.γ[i]

        # 2. Compute the full probability grid for these parameters
        # We'll use a max_goals of 10, which is standard for these calculations
        prob_grid = compute_bivariate_xScore(λ_h, λ_a, γ, 10)

        # 3. Calculate the 1x2 probabilities for this grid and add to the running total
        hda = BayesianFootball.calculate_1x2(prob_grid)
        home_win_prob_sum += hda.hw
        draw_prob_sum += hda.dr
        away_win_prob_sum += hda.aw
    end

    # 4. Average the probabilities over all posterior samples
    # This gives us the final posterior predictive probabilities
    return (
        home = home_win_prob_sum / n_samples,
        draw = draw_prob_sum / n_samples,
        away = away_win_prob_sum / n_samples
    )
end

end
