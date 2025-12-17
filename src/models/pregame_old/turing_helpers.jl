

"""
This module contains helper functions and submodels for Turing.jl.
"""
module TuringHelpers

using Turing, Distributions, LinearAlgebra, Statistics
# using ..PreGameInterfaces 
# using ...Features: FeatureSet
# using ...TypesInterfaces: AbstractFootballModel, AbstractPregameModel
using ...TypesInterfaces: AbstractFootballModel, AbstractPregameModel, FeatureSet # <-- ADD FeatureSet HERE
# using ..AbstractFootballModel, ..AbstractPregameModel
export prepare_data

"""
    prepare_data(model, feature_set)

Adapts the flexible `FeatureSet` into a type-stable `NamedTuple`
that the Turing `@model` can use efficiently.
"""
function prepare_data(model::AbstractPregameModel, feature_set::FeatureSet) 
    # This is the key adapter: it converts the flexible Dict into a
    # type-stable NamedTuple that the Turing model can use efficiently.
    return NamedTuple(feature_set.data)
end

# --- PRIORS SUBMODEL (Static) ---
@model function static_priors(n_teams::Int, has_home_advantage::Bool)
    log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
    
    home_adv = 0.0
    if has_home_advantage
        home_adv ~ Normal(log(1.3), 0.2)
    end
    
    return (; log_α_raw, log_β_raw, home_adv)
end

# --- DYNAMICS SUBMODEL (AR1) ---
@model function ar1_dynamics(n_teams::Int, n_rounds::Int)
    ρ_att ~ Normal(0.0, 0.5)
    ρ_def ~ Normal(0.0, 0.5)
    σ_att ~ Truncated(Normal(0, 1), 0, Inf)
    σ_def ~ Truncated(Normal(0, 1), 0, Inf)

    attack_raw = Matrix{Real}(undef, n_teams, n_rounds)
    defense_raw = Matrix{Real}(undef, n_teams, n_rounds)

    attack_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)
    defense_raw[:, 1] ~ MvNormal(zeros(n_teams), 0.5 * I)

    for t in 2:n_rounds
        attack_raw[:, t] ~ MvNormal(ρ_att * attack_raw[:, t - 1], σ_att * I)
        defense_raw[:, t] ~ MvNormal(ρ_def * defense_raw[:, t - 1], σ_def * I)
    end
    
    return (attack_raw=attack_raw, defense_raw=defense_raw)
end


# ---- link functions 
function calculate_goal_rates(log_α, log_β, log_home_adv, home_ids, away_ids)
    log_λs = log_α[home_ids] .+ log_β[away_ids] .+ log_home_adv
    log_μs = log_α[away_ids] .+ log_β[home_ids]
    return log_λs, log_μs
end

end




