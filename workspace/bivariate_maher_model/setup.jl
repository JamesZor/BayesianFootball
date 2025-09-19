module BivariateMaher

using Turing
using LinearAlgebra
using SpecialFunctions

export logpdf_bivariate_poisson, maher_bivariate_model

function logpdf_bivariate_poisson(X, Y, λx, λy, γ)
    # Check for valid parameters
    if λx < 0 || λy < 0 || γ < 0
        return -Inf
    end

    # Calculate the log of the summation term
    min_xy = min(X, Y)
    log_sum_term = -Inf  # Initialize with log(0)

    for k in 0:min_xy
        # Use log-scale for numerical stability
        term_k = (logfactorial(X) - logfactorial(k) - logfactorial(X - k)) +
                 (logfactorial(Y) - logfactorial(k) - logfactorial(Y - k)) +
                 logfactorial(k) +
                 k * (log(γ) - log(λx) - log(λy))

        # Log-sum-exp trick to safely add probabilities
        if isinf(log_sum_term)
            log_sum_term = term_k
        else
            log_sum_term = log(exp(log_sum_term - term_k) + 1) + term_k
        end
    end

    # Combine all parts of the log-likelihood
    logp = -(λx + λy + γ) +
           X * log(λx) - logfactorial(X) +
           Y * log(λy) - logfactorial(Y) +
           log_sum_term

    return logp
end

@model function maher_bivariate_model(
    home_team_ids, away_team_ids,
    home_goals, away_goals,
    n_teams
)
    # --- Priors for attack, defense, and home advantage (Unchanged) ---
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)

    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)

    log_δ ~ Normal(log(1.3), 0.2) # I'm renaming your γ to δ to avoid confusion with the new dependence param

    # --- NEW: Prior for the dependence parameter γ ---
    # Must be positive. The paper estimates it around 0.1
    γ ~ truncated(Normal(0.1, 0.1), 0, Inf)

    # --- Identifiability and Transformations (Unchanged) ---
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)

    α = exp.(log_α)
    β = exp.(log_β)
    δ = exp(log_δ)

    # --- Likelihood using the custom log-pdf ---
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]

        λ = α[i] * β[j] * δ # Expected home goals
        μ = α[j] * β[i] # Expected away goals

        # Replace the two Poisson lines with this:
        Turing.@addlogprob! logpdf_bivariate_poisson(home_goals[k], away_goals[k], λ, μ, γ)
    end
end

end
