# src/models/pregame/components/dynamics/player_level.jl

# ==========================================
# 1. CONFIGURATION
# ==========================================
Base.@kwdef struct PositionalPlayerDynamics <: AbstractDynamicsConfig
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
@model function build_dynamics(config::PositionalPlayerDynamics, n_teams::Int)
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
function extract_dynamics(chain::Chains, ::PositionalPlayerDynamics, prefix::String, n_teams::Int)
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
