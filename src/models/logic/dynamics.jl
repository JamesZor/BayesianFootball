# src/models/logic/dynamics.jl
using Distributions, Statistics

export sample_latent, apply_dynamics, get_state_val

# --- A. Sampling Helper (Defines Shape) ---
# Used in the @model block to draw priors

function sample_latent(::Static, n_teams, n_rounds)
    # Static: Just 1 value per team. Ignores n_rounds.
    return filldist(Normal(0, 1), n_teams)
end

function sample_latent(::GRW, n_teams, n_rounds)
    # GRW: Needs steps for every round.
    # Returns Matrix (n_teams × n_rounds)
    return filldist(Normal(0, 1), n_teams, n_rounds)
end

# --- B. Evolution Helper (Defines Math) ---
# Used to convert raw z-scores into actual strength

function apply_dynamics(::Static, raw_z::AbstractVector, σ)
    # Static: Just scale.
    return raw_z .* σ
end

function apply_dynamics(::GRW, raw_z::AbstractMatrix, σ)
    # GRW: Scale and Cumulative Sum across time (dims=2).
    # This turns "steps" into a "walk".
    return cumsum(raw_z .* σ, dims=2)
end

# --- C. Indexing Helper (Safety) ---
# Handles looking up values in the Likelihood loop

# If Static (Vector), time 't' is ignored
get_state_val(x::AbstractVector, team_id, t) = x[team_id]

# If GRW (Matrix), we must look up [team, time]
get_state_val(x::AbstractMatrix, team_id, t) = x[team_id, t]


export get_sigma_prior 

"""
get_sigma_prior(dynamics)
Returns the correct prior distribution for the variability parameter.
"""
function get_sigma_prior(d::Static)
    return d.σ_prior
end

function get_sigma_prior(d::GRW)
    return d.σ_step_prior
end
