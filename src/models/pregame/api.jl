"""
Defines the main `PregameModel` struct and the user-facing API functions
like `build_turing_model` and `required_features`.
"""
module PreGame

# --- Dependencies ---
using ..PreGameInterfaces
using ..PreGameComponents
using ..TuringHelpers
using Turing

# --- Exports ---
# Expose the main struct, the API functions, and all component types
export PregameModel, build_turing_model, required_features,
       PoissonGoal, NegativeBinomial, AR1, RandomWalk


# --- Struct Definition ---

"""
    PregameModel{D<:GoalDistribution, T<:TimeDynamic}

A flexible, composite model struct for pre-game predictions.

# Fields
- `distribution::D`: The statistical distribution for goal scoring (e.g., `PoissonGoal()`).
- `time_dynamic::T`: The process governing how team strengths evolve over time (e.g., `AR1()`).
- `home_advantage::Bool`: Whether to include a home advantage parameter.
"""
struct PregameModel{D<:GoalDistribution, T<:TimeDynamic} <: AbstractPregameModel
    distribution::D
    time_dynamic::T
    home_advantage::Bool
end


# --- API Functions ---

"Defines the features required by the model based on its components."
function required_features(model::PregameModel)
    # This is a placeholder; a full implementation would dispatch
    # on model.distribution and model.time_dynamic.
    return [:home_score, :away_score, :global_round, :is_home]
end

"Constructs and returns a Turing.jl model instance."
function build_turing_model(model::PregameModel, feature_set)
    @model function dynamic_pregame_model(
        home_goals,
        away_goals,
        n_teams,
        n_rounds,
        home_ids_by_round,
        away_ids_by_round
    )
        # 1. Call the dispatched helper for the time dynamics
        attack, defense = _add_time_dynamics(
            model.time_dynamic,
            n_teams,
            n_rounds,
            home_ids_by_round,
            away_ids_by_round
        )

        # 2. Call the dispatched helper for the likelihood
        _add_likelihood(
            model.distribution,
            home_goals,
            away_goals,
            attack,
            defense
        )

        # A full model would return generated quantities
        return nothing
    end

    # This is a simplified call; a real implementation would pass
    # the actual feature data extracted from the feature_set.
    return dynamic_pregame_model(
        feature_set.matches_df.home_score,
        feature_set.matches_df.away_score,
        length(feature_set.team_map),
        maximum(feature_set.matches_df.global_round),
        [], # Placeholder for home_ids_by_round
        []  # Placeholder for away_ids_by_round
    )
end

end

