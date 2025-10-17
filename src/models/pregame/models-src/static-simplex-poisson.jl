
# --- Model File: static-poisson.jl ---

using Turing
using LinearAlgebra
using ..PreGameInterfaces # Use the abstract type
using ..TuringHelpers

# Export the concrete model struct and its build function
export StaticSimplexPoisson, build_turing_model

# 1. DEFINE A CONCRETE STRUCT FOR THE MODEL
struct StaticSimplexPoisson <: AbstractPregameModel end
# 2. DEFINE THE TURING MODEL LOGIC
# This function is dispatched specifically on the StaticPoisson type
function build_turing_model(model::StaticSimplexPoisson, feature_set)

     data = TuringHelpers.prepare_data(model, feature_set)


    # NOTE: Issues with sampling - neal funnel assumed.
    # @model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
    #     # --- Priors ---
    #     log_α_scale ~ Normal(0, 10)
    #     log_β_scale ~ Normal(0, 10)
    #     home_adv ~ Normal(log(1.3), 0.2)
    #
    #     # --- Identifiability Constraint ---
    #     log_α_raw ~ Dirichlet(n_teams, 1.0)
    #     log_β_raw ~ Dirichlet(n_teams, 1.0)
    #
    #     factor = 1.0/n_teams
    #
    #     log_α := log_α_scale .* ( log_α_raw .- factor)
    #     log_β := log_β_scale .* ( log_β_raw .- factor)
    #
    #     # --- Calculate Goal Rates ---
    #     log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
    #     log_μs = log_α[away_ids] .+ log_β[home_ids]
    #
    #     # --- Likelihood ---
    #     for i in eachindex(home_goals)
    #       home_goals[i] ~ LogPoisson(log_λs[i])
    #       away_goals[i] ~ LogPoisson(log_μs[i])
    #     end 
    #
    #     return nothing
    #
    # end


  @model function static_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals)
      # --- Priors ---
        log_α_scale ~ Normal(0, 10)
        log_β_scale ~ Normal(0, 10)
        home_adv ~ Normal(log(1.3), 0.2)

        # --- Non-Centered Parameterization for Identifiability ---
        # Sample n-1 raw parameters from a standard normal distribution
        # These are independent of the scale! This is the key.
        α_raw_offsets ~ MvNormal(n_teams - 1, 1.0)
        β_raw_offsets ~ MvNormal(n_teams - 1, 1.0)

        # Deterministically create the full n-team vectors
        # that sum to zero.
        α_offsets = vcat(α_raw_offsets, -sum(α_raw_offsets))
        β_offsets = vcat(β_raw_offsets, -sum(β_raw_offsets))

        # Apply the scale and mean AFTER sampling.
        # This transformation happens "outside" the sampler's main work.
        log_α := log_α_scale .* α_offsets
        log_β := log_β_scale .* β_offsets


        # --- Calculate Goal Rates ---
        log_λs = home_adv .+ log_α[home_ids] .+ log_β[away_ids]
        log_μs = log_α[away_ids] .+ log_β[home_ids]

        # --- Likelihood ---
        for i in eachindex(home_goals)
          home_goals[i] ~ LogPoisson(log_λs[i])
          away_goals[i] ~ LogPoisson(log_μs[i])
        end 
      
        return nothing

    # Return an instance of the Turing model with the data
    return static_poisson_model(data.n_teams, data.f_home_ids, data.f_away_ids, data.f_home_goals, data.f_away_goals)
end
