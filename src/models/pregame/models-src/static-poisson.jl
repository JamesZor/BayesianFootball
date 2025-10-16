# --- Model File: static-poisson.jl ---

using Turing
using LinearAlgebra
using ..PreGameInterfaces # Use the abstract type
using ..TuringHelpers

# Export the concrete model struct and its build function
export StaticPoisson, build_turing_model

# 1. DEFINE A CONCRETE STRUCT FOR THE MODEL
struct StaticPoisson <: AbstractPregameModel end

# 2. DEFINE THE TURING MODEL LOGIC
# This function is dispatched specifically on the StaticPoisson type
function build_turing_model(model::StaticPoisson, feature_set)

     data = TuringHelpers.prepare_data(model, feature_set)

    @model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
        # --- Priors ---
        log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        home_adv ~ Normal(log(1.3), 0.2)

        # --- Identifiability Constraint ---
        log_α := log_α_raw .- mean(log_α_raw) # using := to added to track vars,
        log_β := log_β_raw .- mean(log_β_raw)

        # --- Calculate Goal Rates ---
        log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
        log_μs = log_α[away_ids] .+ log_β[home_ids]

        # --- Likelihood ---
        for i in eachindex(home_goals)
          home_goals[i] ~ LogPoisson(log_λs[i])
          away_goals[i] ~ LogPoisson(log_μs[i])
        end 
      
        return nothing
      
    end

    # Return an instance of the Turing model with the data
    return static_poisson_model(data.n_teams, data.f_home_ids, data.f_away_ids, data.f_home_goals, data.f_away_goals)
end

