using Distributions 
using StatsBase
using Turing

using Turing, Distributions, LinearAlgebra, Random

@model function basic_maher_model(
    home_team_ids, away_team_ids, home_goals, away_goals, n_teams
)
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_home_adv ~ Normal(log(1.3), 0.2)
    
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    home_ids = home_team_ids
    away_ids = away_team_ids
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ log_home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]

    home_goals .~ LogPoisson.(log_λs)
    away_goals .~ LogPoisson.(log_μs)
end



function get_maher_inputs(data)
  home_team_ids_flat = reduce(vcat, data.home_team_ids)
  away_team_ids_flat = reduce(vcat, data.away_team_ids)
  home_goals_flat = reduce(vcat, data.home_goals)
  away_goals_flat = reduce(vcat, data.away_goals)

  return  home_team_ids_flat, away_team_ids_flat, home_goals_flat, away_goals_flat
end 

function get_static_parameters(chain_static)
  log_α_raw_static = Array(group(chain_static, :log_α_raw))
  log_α_static = log_α_raw_static .- mean(log_α_raw_static, dims=2)

  log_β_raw_static = Array(group(chain_static, :log_β_raw))
  log_β_static = log_β_raw_static .- mean(log_β_raw_static, dims=2)
  return log_α_static, log_β_static
end


"""
    generate_static_predictions(chain_static, test_set)

Generates posterior predictive simulations for the static model.
"""
function generate_static_predictions(chain_static, test_set)
    log_α_s = Array(group(chain_static, :log_α_raw)) .- mean(Array(group(chain_static, :log_α_raw)), dims=2)
    log_β_s = Array(group(chain_static, :log_β_raw)) .- mean(Array(group(chain_static, :log_β_raw)), dims=2)
    log_home_adv_s = vec(Array(chain_static["log_home_adv"]))

    n_samples = size(chain_static, 1)
    predictions = Dict()

    for match in test_set
        i, j = match.home_team, match.away_team
        simulated_outcomes = []

        for s in 1:n_samples
            log_λ = log_α_s[s, i] + log_β_s[s, j] + log_home_adv_s[s]
            log_μ = log_α_s[s, j] + log_β_s[s, i]
            push!(simulated_outcomes, (hg=rand(Poisson(exp(log_λ))), ag=rand(Poisson(exp(log_μ)))))
        end
        predictions[match] = simulated_outcomes
    end
    return predictions
end

