"""
Defines the main `PregameModel` struct and the user-facing API functions
like `build_turing_model` and `required_features`.
"""
module PreGame

using ..PreGameInterfaces, ..PreGameComponents, ..TuringHelpers
using Turing


export PregameModel, build_turing_model, required_features,
       PoissonGoal, NegativeBinomial, AR1, RandomWalk, Static

struct PregameModel{D<:GoalDistribution, T<:TimeDynamic} <: AbstractPregameModel
    distribution::D
    time_dynamic::T
    home_advantage::Bool
end

function required_features(model::PregameModel)
    return [:home_team, :away_team, :home_score, :away_score, :match_date]
end

# --- Build Turing Model (dispatched for DYNAMIC models) ---
function build_turing_model(model::PregameModel{<:GoalDistribution, <:Union{AR1, RandomWalk}}, feature_set)
    @model function dynamic_model(n_teams, n_rounds, r_home_ids, r_away_ids, r_home_goals, r_away_goals)
       
        dynamics ~ ar1_dynamics(n_teams, n_rounds)
        attack_raw, defense_raw = dynamics

        # Apply sum-to-zero constraint for identifiability
        attack = attack_raw .- mean(attack_raw, dims=1)
        defense = defense_raw .- mean(defense_raw, dims=1)

        home_adv = 0.0
        if model.home_advantage
             home_adv ~ Normal(log(1.3), 0.2)
        end

        _add_likelihood(model.distribution, n_rounds, r_home_ids, r_away_ids, r_home_goals, r_away_goals, attack, defense, home_adv)
    end
    return dynamic_model(feature_set.n_teams, feature_set.n_rounds, feature_set.round_home_ids, feature_set.round_away_ids, feature_set.round_home_goals, feature_set.round_away_goals)
end

# --- Build Turing Model (dispatched for STATIC models) ---
function build_turing_model(model::PregameModel{<:GoalDistribution, Static}, feature_set)
    home_ids = vcat(feature_set.round_home_ids...)
    away_ids = vcat(feature_set.round_away_ids...)
    home_goals = vcat(feature_set.round_home_goals...)
    away_goals = vcat(feature_set.round_away_goals...)

    @model function static_model(n_teams, home_ids, away_ids, home_goals, away_goals)
        # --- 1. Call the priors submodel ---
        priors ~ static_priors(n_teams, model.home_advantage)
        
        # --- 2. Perform pure calculations (identifiability constraint) ---
        log_α = priors.log_α_raw .- mean(priors.log_α_raw)
        log_β = priors.log_β_raw .- mean(priors.log_β_raw)
        
        # --- 3. Calculate final goal rates ---
        log_λs, log_μs = _get_static_log_goal_rates(log_α, log_β, priors.home_adv, home_ids, away_ids)
        
        # --- 4. Add the likelihood ---
        _add_likelihood(model.distribution, home_goals, away_goals, log_λs, log_μs)
    end
    return static_model(feature_set.n_teams, home_ids, away_ids, home_goals, away_goals)
end

end
