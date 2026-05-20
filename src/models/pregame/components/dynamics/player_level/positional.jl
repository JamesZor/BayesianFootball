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

# ==========================================
# 4. HIERARCHICAL PLAYER DYNAMICS (VARYING SLOPES)
# ==========================================

Base.@kwdef struct HierarchicalPlayerDynamicsConfig <: AbstractDynamicsConfig
    # Goalkeeper priors
    w_G_prior::ContinuousUnivariateDistribution = Normal(0.0, 0.5)
    
    # Outfield hierarchical priors (Attack)
    w_Outfield_att_base_prior::ContinuousUnivariateDistribution = Normal(0.08, 0.05)
    σ_w_att_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.05), lower=0.0)
    
    # Outfield hierarchical priors (Defense)
    w_Outfield_def_base_prior::ContinuousUnivariateDistribution = Normal(-0.08, 0.05)
    σ_w_def_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.05), lower=0.0)

    # Required for Time-Decay weights
    days_half_life::Real = 180.0
end

"""
    build_dynamics(config::HierarchicalPlayerDynamicsConfig, n_teams::Int)

Submodel for Hierarchical Player dynamics. Learns team-specific multipliers 
(tactical cohesion) for outfield players while keeping goalkeepers global.
"""
@model function build_dynamics(config::HierarchicalPlayerDynamicsConfig, n_teams::Int)
    # 1. Goalkeepers (Global)
    w_G_att ~ config.w_G_prior
    w_G_def ~ config.w_G_prior

    # 2. Outfield Attack (Hierarchical)
    w_Outfield_att_base ~ config.w_Outfield_att_base_prior
    σ_w_att             ~ config.σ_w_att_prior
    team_att_raw        ~ filldist(Normal(0, 1), n_teams)

    # 3. Outfield Defense (Hierarchical)
    w_Outfield_def_base ~ config.w_Outfield_def_base_prior
    σ_w_def             ~ config.σ_w_def_prior
    team_def_raw        ~ filldist(Normal(0, 1), n_teams)

    return (; 
        w_G_att, w_G_def,
        w_Outfield_att_base, σ_w_att, team_att_raw,
        w_Outfield_def_base, σ_w_def, team_def_raw
    )
end

"""
    extract_dynamics(chain::Chains, config::HierarchicalPlayerDynamicsConfig, prefix::String, n_teams::Int)

Extracts the hierarchical positional weights and team-specific raw values.
"""
function extract_dynamics(chain::Chains, ::HierarchicalPlayerDynamicsConfig, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)

    # 1. Global / Base Components
    nt = (;
        w_G_att = vec(Array(chain[Symbol("$prefix.w_G_att")])),
        w_G_def = vec(Array(chain[Symbol("$prefix.w_G_def")])),
        w_Outfield_att_base = vec(Array(chain[Symbol("$prefix.w_Outfield_att_base")])),
        σ_w_att = vec(Array(chain[Symbol("$prefix.σ_w_att")])),
        w_Outfield_def_base = vec(Array(chain[Symbol("$prefix.w_Outfield_def_base")])),
        σ_w_def = vec(Array(chain[Symbol("$prefix.σ_w_def")])),
    )

    # 2. Team-Specific Raw Values
    team_att_raw = zeros(n_samples, n_teams)
    team_def_raw = zeros(n_samples, n_teams)

    for i in 1:n_teams
        team_att_raw[:, i] = vec(Array(chain[Symbol("$prefix.team_att_raw[$i]")]))
        team_def_raw[:, i] = vec(Array(chain[Symbol("$prefix.team_def_raw[$i]")]))
    end

    return merge(nt, (; team_att_raw, team_def_raw))
end
