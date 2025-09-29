# workspace/basic_state_space/models/ar1_poisson.jl

module AR1Poisson

using Turing
using LinearAlgebra
using Distributions
using DataFrames
using MCMCChains
using Statistics
using BayesianFootball

export AR1PoissonModel

#=
--------------------------------------------------------------------------------
1.  MODEL DEFINITION
--------------------------------------------------------------------------------
=#

struct AR1PoissonModel <: AbstractStateSpaceModel end

function BayesianFootball.get_required_features(::AR1PoissonModel)
    return (
        :home_team_ids, :away_team_ids, :n_teams,
        :global_round
    )
end

function BayesianFootball.build_turing_model(
    ::AR1PoissonModel,
    features::NamedTuple,
    goals_home::Vector{Int},
    goals_away::Vector{Int}
)
    temp_df = DataFrame(
        global_round = features.global_round, home_id = features.home_team_ids,
        away_id = features.away_team_ids, gh = goals_home, ga = goals_away
    )
    grouped = groupby(temp_df, :global_round)
    n_rounds = length(grouped)

    home_ids_by_round = [g.home_id for g in grouped]
    away_ids_by_round = [g.away_id for g in grouped]
    home_goals_by_round = [g.gh for g in grouped]
    away_goals_by_round = [g.ga for g in grouped]
    
    return ar1_poisson_model(
        home_ids_by_round, away_ids_by_round,
        home_goals_by_round, away_goals_by_round,
        features.n_teams, n_rounds
    )
end

#=
--------------------------------------------------------------------------------
2.  TURING @MODEL IMPLEMENTATION
--------------------------------------------------------------------------------
=#

@model function ar1_poisson_model(
    home_team_ids::Vector{<:Vector}, away_team_ids::Vector{<:Vector}, 
    home_goals::Vector{<:Vector}, away_goals::Vector{<:Vector}, 
    n_teams::Int, n_rounds::Int
)
    ρ_attack ~ Beta(10, 1.5); ρ_defense ~ Beta(10, 1.5)
    μ_log_σ_attack ~ Normal(-2.5, 0.5); τ_log_σ_attack ~ Truncated(Normal(0, 0.2), 0, Inf)
    μ_log_σ_defense ~ Normal(-2.5, 0.5); τ_log_σ_defense ~ Truncated(Normal(0, 0.2), 0, Inf)

    z_log_σ_attack ~ MvNormal(zeros(n_teams), I)
    z_log_σ_defense ~ MvNormal(zeros(n_teams), I)
    log_σ_attack = μ_log_σ_attack .+ z_log_σ_attack * τ_log_σ_attack
    log_σ_defense = μ_log_σ_defense .+ z_log_σ_defense * τ_log_σ_defense
    σ_attack = exp.(log_σ_attack); σ_defense = exp.(log_σ_defense)
    log_home_adv ~ Normal(log(1.3), 0.2)

    initial_α_z ~ MvNormal(zeros(n_teams), I); log_α_raw_t0 = initial_α_z * sqrt(0.5)
    initial_β_z ~ MvNormal(zeros(n_teams), I); log_β_raw_t0 = initial_β_z * sqrt(0.5)

    z_α ~ MvNormal(zeros(n_teams * n_rounds), I); z_α_mat = reshape(z_α, n_teams, n_rounds)
    z_β ~ MvNormal(zeros(n_teams * n_rounds), I); z_β_mat = reshape(z_β, n_teams, n_rounds)

    log_α_raw = Matrix{Real}(undef, n_teams, n_rounds)
    log_β_raw = Matrix{Real}(undef, n_teams, n_rounds)

    for t in 1:n_rounds
        if t == 1
            log_α_raw[:, 1] = log_α_raw_t0 .+ z_α_mat[:, 1] .* σ_attack
            log_β_raw[:, 1] = log_β_raw_t0 .+ z_β_mat[:, 1] .* σ_defense
        else
            log_α_raw[:, t] = ρ_attack * log_α_raw[:, t-1] .+ z_α_mat[:, t] .* σ_attack
            log_β_raw[:, t] = ρ_defense * log_β_raw[:, t-1] .+ z_β_mat[:, t] .* σ_defense
        end

        log_α_t = log_α_raw[:, t] .- mean(log_α_raw[:, t])
        log_β_t = log_β_raw[:, t] .- mean(log_β_raw[:, t])
    
        home_ids = home_team_ids[t]; away_ids = away_team_ids[t]
        
        if !isempty(home_ids)
            log_λs = log_α_t[home_ids] .+ log_β_t[away_ids] .+ log_home_adv
            log_μs = log_α_t[away_ids] .+ log_β_t[home_ids]
            home_goals[t] .~ LogPoisson.(log_λs)
            away_goals[t] .~ LogPoisson.(log_μs)
        end
    end
end

#=
--------------------------------------------------------------------------------
3.  PREDICTION LOGIC
--------------------------------------------------------------------------------
=#

"""
Extracts and reconstructs all necessary parameters from the MCMC chain for prediction.
"""
function BayesianFootball.extract_posterior_samples(
    ::AR1PoissonModel,
    chain::Chains,
    mapping::MappedData
)
    n_samples = size(chain, 1) * size(chain, 3)
    n_teams = length(mapping.team)
    z_α_flat = BayesianFootball.extract_samples(chain, "z_α")
    n_rounds = size(z_α_flat, 2) ÷ n_teams
    
    ρ_attack = vec(Array(chain[:ρ_attack])); ρ_defense = vec(Array(chain[:ρ_defense]))
    log_home_adv = vec(Array(chain[:log_home_adv]))

    μ_log_σ_attack = vec(Array(chain[:μ_log_σ_attack])); τ_log_σ_attack = vec(Array(chain[:τ_log_σ_attack]))
    z_log_σ_attack = BayesianFootball.extract_samples(chain, "z_log_σ_attack")
    σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack .* τ_log_σ_attack)

    μ_log_σ_defense = vec(Array(chain[:μ_log_σ_defense])); τ_log_σ_defense = vec(Array(chain[:τ_log_σ_defense]))
    z_log_σ_defense = BayesianFootball.extract_samples(chain, "z_log_σ_defense")
    σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense .* τ_log_σ_defense)

    initial_α_z = BayesianFootball.extract_samples(chain, "initial_α_z")
    initial_β_z = BayesianFootball.extract_samples(chain, "initial_β_z")
    z_α_mat_reshaped = reshape(z_α_flat', n_teams, n_rounds, n_samples)
    z_β_flat = BayesianFootball.extract_samples(chain, "z_β")
    z_β_mat_reshaped = reshape(z_β_flat', n_teams, n_rounds, n_samples)

    log_α_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)
    log_β_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)

    for s in 1:n_samples
        log_α_raw_t0 = initial_α_z[s, :] .* sqrt(0.5)
        log_β_raw_t0 = initial_β_z[s, :] .* sqrt(0.5)
        for t in 1:n_rounds
            if t == 1
                log_α_raw[s, :, 1] = log_α_raw_t0 .+ z_α_mat_reshaped[:, 1, s] .* σ_attack[s, :]
                log_β_raw[s, :, 1] = log_β_raw_t0 .+ z_β_mat_reshaped[:, 1, s] .* σ_defense[s, :]
            else
                log_α_raw[s, :, t] = ρ_attack[s] * log_α_raw[s, :, t-1] .+ z_α_mat_reshaped[:, t, s] .* σ_attack[s, :]
                log_β_raw[s, :, t] = ρ_defense[s] * log_β_raw[s, :, t-1] .+ z_β_mat_reshaped[:, t, s] .* σ_defense[s, :]
            end
        end
    end

    log_α_centered = similar(log_α_raw); log_β_centered = similar(log_β_raw)
    for s in 1:n_samples, t in 1:n_rounds
        log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
        log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
    end

    return (
        n_rounds=n_rounds, log_home_adv=log_home_adv, ρ_attack=ρ_attack, ρ_defense=ρ_defense,
        σ_attack=σ_attack, σ_defense=σ_defense, log_α_raw=log_α_raw, log_β_raw=log_β_raw,
        log_α_centered=log_α_centered, log_β_centered=log_β_centered
    )
end

"""
Calculates the expected goal rates (λ) for a single match and posterior sample.
"""
function get_goal_rates(
    ::AR1PoissonModel,
    samples::NamedTuple,
    i::Int,
    features::NamedTuple
)
    home_idx = features.home_team_ids[1]; away_idx = features.away_team_ids[1]
    current_round = features.global_round[1]
    
    local log_α_t, log_β_t

    if current_round <= samples.n_rounds
        log_α_t = samples.log_α_centered[i, :, current_round]
        log_β_t = samples.log_β_centered[i, :, current_round]
    else # Future prediction
        last_α_raw = samples.log_α_raw[i, :, samples.n_rounds]
        last_β_raw = samples.log_β_raw[i, :, samples.n_rounds]
        n_teams = size(last_α_raw, 1)
        future_α_raw = samples.ρ_attack[i] .* last_α_raw .+ randn(n_teams) .* samples.σ_attack[i, :]
        future_β_raw = samples.ρ_defense[i] .* last_β_raw .+ randn(n_teams) .* samples.σ_defense[i, :]
        log_α_t = future_α_raw .- mean(future_α_raw)
        log_β_t = future_β_raw .- mean(future_β_raw)
    end
    
    log_λ_home = log_α_t[home_idx] + log_β_t[away_idx] + samples.log_home_adv[i]
    log_λ_away = log_α_t[away_idx] + log_β_t[home_idx]
    
    return (λ_home = exp(log_λ_home), λ_away = exp(log_λ_away))
end

"""
Generates full-time predictions.
"""
function BayesianFootball._predict_match_ft(
    model_def::AR1PoissonModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home = zeros(n_samples); λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples); draw_probs = zeros(n_samples); away_win_probs = zeros(n_samples)
    under_05=zeros(n_samples); under_15=zeros(n_samples); under_25=zeros(n_samples); under_35=zeros(n_samples)
    btts = zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:3 for a in 0:3], ["other_home_win", "other_away_win", "other_draw"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a

        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ft(p)
        
        home_win_probs[i] = hda.hw; draw_probs[i] = hda.dr; away_win_probs[i] = hda.aw
        for (key, value) in cs; correct_score[key][i] = value; end
        
        under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
        under_35[i] = BayesianFootball.calculate_under_prob(p, 3)
        btts[i] = BayesianFootball.calculate_btts(p)
    end

    return BayesianFootball.Predictions.MatchFTPredictions(
        λ_home, λ_away, home_win_probs, draw_probs, away_win_probs,
        correct_score, under_05, under_15, under_25, under_35, btts
    )
end

"""
Generates half-time predictions.
"""
function BayesianFootball._predict_match_ht(
    model_def::AR1PoissonModel,
    chain::Chains,
    features::NamedTuple,
    posterior_samples::NamedTuple
)
    n_samples = size(chain, 1) * size(chain, 3)
    λ_home = zeros(n_samples); λ_away = zeros(n_samples)
    home_win_probs = zeros(n_samples); draw_probs = zeros(n_samples); away_win_probs = zeros(n_samples)
    under_05=zeros(n_samples); under_15=zeros(n_samples); under_25=zeros(n_samples)
    cs_keys = vcat([(h, a) for h in 0:2 for a in 0:2], ["any_unquoted"])
    correct_score = Dict{Union{Tuple{Int,Int}, String}, Vector{Float64}}(key => zeros(n_samples) for key in cs_keys)

    for i in 1:n_samples
        λ_h, λ_a = get_goal_rates(model_def, posterior_samples, i, features)
        λ_home[i] = λ_h; λ_away[i] = λ_a

        p = BayesianFootball.compute_xScore(λ_h, λ_a, 10)
        hda = BayesianFootball.calculate_1x2(p)
        cs = BayesianFootball.calculate_correct_score_dict_ht(p)

        home_win_probs[i] = hda.hw; draw_probs[i] = hda.dr; away_win_probs[i] = hda.aw
        for (key, value) in cs; correct_score[key][i] = value; end

        under_05[i] = BayesianFootball.calculate_under_prob(p, 0)
        under_15[i] = BayesianFootball.calculate_under_prob(p, 1)
        under_25[i] = BayesianFootball.calculate_under_prob(p, 2)
    end

    return BayesianFootball.Predictions.MatchHTPredictions(
        λ_home, λ_away, home_win_probs, draw_probs, away_win_probs,
        correct_score, under_05, under_15, under_25
    )
end

end # end module
