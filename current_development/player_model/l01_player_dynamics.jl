# current_development/player_model/l01_player_dynamics.jl

using Turing
using Distributions
using BayesianFootball
using BayesianFootball.Models
using BayesianFootball.Models.PreGame

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Use the full path to ensure standalone scripts find the abstract type
Base.@kwdef struct PositionalPlayerDynamics <: BayesianFootball.Models.PreGame.AbstractDynamicsConfig
    w_att_prior::ContinuousUnivariateDistribution = Normal(0.0, 0.2)
    w_def_prior::ContinuousUnivariateDistribution = Normal(0.0, 0.2)
end

# ==========================================
# 2. TURING SUBMODEL
# ==========================================
"""
    build_dynamics(config::PositionalPlayerDynamics, n_teams::Int)

Submodel for Player-Level dynamics. Instead of team-specific latents, 
it samples 8 global positional weights.
"""
@model function BayesianFootball.Models.PreGame.build_dynamics(config::PositionalPlayerDynamics, n_teams::Int)
    # n_teams is unused in this specific component but kept for interface compatibility
    
    # 4 global attacking weights
    w_G_att ~ config.w_att_prior
    w_D_att ~ config.w_att_prior
    w_M_att ~ config.w_att_prior
    w_F_att ~ config.w_att_prior

    # 4 global defending weights
    w_G_def ~ config.w_def_prior
    w_D_def ~ config.w_def_prior
    w_M_def ~ config.w_def_prior
    w_F_def ~ config.w_def_prior

    return (; 
        w_G_att, w_D_att, w_M_att, w_F_att,
        w_G_def, w_D_def, w_M_def, w_F_def
    )
end

# ==========================================
# 3. EXTRACTOR
# ==========================================
"""
    extract_dynamics(chain::Chains, config::PositionalPlayerDynamics, prefix::String, n_teams::Int)

Extracts the 8 positional weights from the MCMC chain.
"""
function BayesianFootball.Models.PreGame.extract_dynamics(chain::Chains, ::PositionalPlayerDynamics, prefix::String, n_teams::Int)
    # total draws across all chains
    n_samples = size(chain, 1) * size(chain, 3) 
    
    return (;
        w_G_att = vec(Array(chain[Symbol("$prefix.w_G_att")])),
        w_D_att = vec(Array(chain[Symbol("$prefix.w_D_att")])),
        w_M_att = vec(Array(chain[Symbol("$prefix.w_M_att")])),
        w_F_att = vec(Array(chain[Symbol("$prefix.w_F_att")])),
        
        w_G_def = vec(Array(chain[Symbol("$prefix.w_G_def")])),
        w_D_def = vec(Array(chain[Symbol("$prefix.w_D_def")])),
        w_M_def = vec(Array(chain[Symbol("$prefix.w_M_def")])),
        w_F_def = vec(Array(chain[Symbol("$prefix.w_F_def")]))
    )
end
