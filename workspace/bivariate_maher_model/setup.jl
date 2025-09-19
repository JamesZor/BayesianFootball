module BivariateMaher

using Turing
using LinearAlgebra
using SpecialFunctions
using BayesianFootball
using DataFrames
using StatsBase # For mean function

export MaherBivariate, maher_bivariate_model, logpdf_bivariate_poisson

struct MaherBivariate <: BayesianFootball.AbstractModelDefinition end

# (The logpdf_bivariate_poisson and maher_bivariate_model functions remain the same as before)
function logpdf_bivariate_poisson(X, Y, λx, λy, γ)
    if λx < 0 || λy < 0 || γ < 0
        return -Inf
    end
    min_xy = min(X, Y)
    log_sum_term = -Inf
    for k in 0:min_xy
        term_k = (logfactorial(X) - logfactorial(k) - logfactorial(X - k)) +
                 (logfactorial(Y) - logfactorial(k) - logfactorial(Y - k)) +
                 logfactorial(k) +
                 k * (log(γ) - log(λx) - log(λy))
        if isinf(log_sum_term)
            log_sum_term = term_k
        else
            log_sum_term = log(exp(log_sum_term - term_k) + 1) + term_k
        end
    end
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
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    log_δ ~ Normal(log(1.3), 0.2)
    γ ~ truncated(Normal(0.1, 0.1), 0, Inf)
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    α = exp.(log_α)
    β = exp.(log_β)
    δ = exp(log_δ)
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]
        j = away_team_ids[k]
        λ = α[i] * β[j] * δ
        μ = α[j] * β[i]
        Turing.@addlogprob! logpdf_bivariate_poisson(home_goals[k], away_goals[k], λ, μ, γ)
    end
end


# --- Implementation of the ModelDefinition Protocol ---

"""
get_required_features(::MaherBivariate)

Declares the data features required by the bivariate Maher model.
"""
function BayesianFootball.get_required_features(::MaherBivariate)
    return [:home_team_ids, :away_team_ids, :n_teams]
end

"""
build_turing_model(::MaherBivariate, ...)

Constructs the Turing model instance for the bivariate model.
"""
function BayesianFootball.build_turing_model(::MaherBivariate, features::NamedTuple, home_goals::V, away_goals::V) where {V<:AbstractVector}
    return maher_bivariate_model(
        features.home_team_ids,
        features.away_team_ids,
        home_goals,
        away_goals,
        features.n_teams
    )
end

"""
extract_posterior_samples(::MaherBivariate, ...)

Extracts and transforms posterior samples, including the new dependence parameter γ.
"""
function BayesianFootball.extract_posterior_samples(::MaherBivariate, chain::Chains)
    log_α_raw = BayesianFootball.extract_samples(chain, "log_α_raw")
    log_β_raw = BayesianFootball.extract_samples(chain, "log_β_raw")
    log_δ = vec(Array(chain[:log_δ]))
    γ = vec(Array(chain[:γ])) # Extract the new parameter

    log_α = log_α_raw .- mean(log_α_raw, dims=2)
    log_β = log_β_raw .- mean(log_β_raw, dims=2)

    return ( α = exp.(log_α), β = exp.(log_β), δ = exp.(log_δ), γ = γ )
end

"""
get_goal_rates(::MaherBivariate, ...)

Calculates λ_home and λ_away for a single sample.
Note: The rates themselves are calculated like the standard model; the dependency
is in the bivariate likelihood, not the rate calculation.
"""
function BayesianFootball.get_goal_rates(::MaherBivariate, samples::NamedTuple, i::Int, features::NamedTuple)
    home_idx = features.home_team_ids[1]
    away_idx = features.away_team_ids[1]

    α_h = samples.α[i, home_idx]; β_h = samples.β[i, home_idx]
    α_a = samples.α[i, away_idx]; β_a = samples.β[i, away_idx]
    δ = samples.δ[i]

    λ_home = α_h * β_a * δ
    λ_away = α_a * β_h
    return (λ_home, λ_away)
end


end # end of module BivariateMaher
