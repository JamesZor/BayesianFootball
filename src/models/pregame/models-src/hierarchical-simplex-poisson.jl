# --- Model File: hierarchical_simplex_poisson.jl ---

using Turing
using DataFrames
using LinearAlgebra
# using ..PreGameInterfaces
using ..TuringHelpers

export HierarchicalSimplexPoisson, build_turing_model

# 1. DEFINE A CONCRETE STRUCT FOR THE MODEL
struct HierarchicalSimplexPoisson <: AbstractPregameModel end 


@model function hierarchical_simplex_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    # --- Priors for Scaling Factors ---
    home_att_scale ~ Normal(0, 10)
    away_att_scale ~ Normal(0, 10)
    home_def_scale ~ Normal(0, 10)
    away_def_scale ~ Normal(0, 10)

    # --- Priors for Simplexes ---
    # A Dirichlet distribution is the natural prior for a simplex.
    # A vector of ones is a non-informative prior.
    home_att_raw ~ Dirichlet(n_teams, 1.0)
    away_att_raw ~ Dirichlet(n_teams, 1.0)
    home_def_raw ~ Dirichlet(n_teams, 1.0)
    away_def_raw ~ Dirichlet(n_teams, 1.0)

    # --- Transformed Parameters (to enforce sum-to-zero) ---
    home_att = home_att_scale .* (home_att_raw .- 1.0/n_teams)
    away_att = away_att_scale .* (away_att_raw .- 1.0/n_teams)
    home_def = home_def_scale .* (home_def_raw .- 1.0/n_teams)
    away_def = away_def_scale .* (away_def_raw .- 1.0/n_teams)

    # --- Calculate Goal Rates ---
    # This model has four parameter groups, so no separate home_adv
    log_λs = home_att[home_ids] .+ away_def[away_ids]
    log_μs = away_att[away_ids] .+ home_def[home_ids]

    # --- Likelihood ---
    for i in eachindex(home_goals)
      home_goals[i] ~ LogPoisson(log_λs[i])
      away_goals[i] ~ LogPoisson(log_μs[i])
    end 

    return nothing
end




# 2. DEFINE THE TURING MODEL LOGIC
# This implementation closely follows the provided Stan model
function build_turing_model(model::HierarchicalSimplexPoisson, feature_set::FeatureSet)

   data = TuringHelpers.prepare_data(model, feature_set)

    @model function hierarchical_simplex_model(n_teams, home_ids, away_ids, home_goals, away_goals)
        # --- Priors for Scaling Factors ---
        home_att_scale ~ Normal(0, 10)
        away_att_scale ~ Normal(0, 10)
        home_def_scale ~ Normal(0, 10)
        away_def_scale ~ Normal(0, 10)

        # --- Priors for Simplexes ---
        # A Dirichlet distribution is the natural prior for a simplex.
        # A vector of ones is a non-informative prior.
        home_att_raw ~ Dirichlet(n_teams, 1.0)
        away_att_raw ~ Dirichlet(n_teams, 1.0)
        home_def_raw ~ Dirichlet(n_teams, 1.0)
        away_def_raw ~ Dirichlet(n_teams, 1.0)

        # --- Transformed Parameters (to enforce sum-to-zero) ---
        home_att = home_att_scale .* (home_att_raw .- 1.0/n_teams)
        away_att = away_att_scale .* (away_att_raw .- 1.0/n_teams)
        home_def = home_def_scale .* (home_def_raw .- 1.0/n_teams)
        away_def = away_def_scale .* (away_def_raw .- 1.0/n_teams)

        # --- Calculate Goal Rates ---
        # This model has four parameter groups, so no separate home_adv
        log_λs = home_att[home_ids] .+ away_def[away_ids]
        log_μs = away_att[away_ids] .+ home_def[home_ids]

        # --- Likelihood ---
        for i in eachindex(home_goals)
          home_goals[i] ~ LogPoisson(log_λs[i])
          away_goals[i] ~ LogPoisson(log_μs[i])
        end 

        return nothing
    end

    return hierarchical_simplex_model(data.n_teams, data.f_home_ids, data.f_away_ids, data.f_home_goals, data.f_away_goals)
end
