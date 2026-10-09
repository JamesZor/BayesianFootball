# QualityStyleGRW builder seam. Shares the unmodified MultiScaleGRW accumulators,
# state-count contract and GRWDynamicsDesign; quality/style have independent sites.

_cb_dynamics_supported(::CB_PG.QualityStyleGRW) = true
_dynamics_weighting_valid(::CB_PG.QualityStyleGRW) = true
_dynamics_weighting_detail(::CB_PG.QualityStyleGRW) =
    "unit likelihood weights; recency is represented by latent macro/micro states"
dynamics_match_weights(::CB_PG.QualityStyleGRW, dates::Vector{Float64}) =
    ones(Float64, length(dates))

_sites_dynamics(::CB_PG.QualityStyleGRW) = [
    Symbol("dyn.τ₀"), Symbol("dyn.τₛ"), Symbol("dyn.τₖ"),
    Symbol("dyn.r₀"), Symbol("dyn.rₛ"), Symbol("dyn.rₖ"),
    Symbol("dyn.q.z_init"), Symbol("dyn.q.z_season"), Symbol("dyn.q.z_target"),
    Symbol("dyn.s.z_init"), Symbol("dyn.s.z_season"), Symbol("dyn.s.z_target"),
]

dynamics_design(::CB_PG.QualityStyleGRW, feature_set, n_matches::Int) =
    dynamics_design(CB_PG.MultiScaleGRW(), feature_set, n_matches)

@model function _cb_dynamics_effects(
    config::CB_PG.QualityStyleGRW,
    home_ids::Vector{Int}, away_ids::Vector{Int},
    design::GRWDynamicsDesign, n_teams::Int,
)
    state ~ to_submodel(
        CB_PG._qs_pair(config, design.initial_accumulator,
                       design.season_accumulator, design.target_accumulator,
                       n_teams, size(design.season_accumulator, 1), design.n_target,
                       design.target_marker), false)
    return (;
        att_h = state.α[design.home_state_indices],
        def_a = state.β[design.away_state_indices],
        att_a = state.α[design.away_state_indices],
        def_h = state.β[design.home_state_indices],
    )
end

function _cb_extract_dynamics(chain::Chains, config::CB_PG.QualityStyleGRW,
                              prefix::String, n_teams::Int, feature_set)
    counts = CB_PG.qs_step_counts(chain, prefix)
    states = CB_PG.extract_dynamics(chain, config, prefix, n_teams,
                                     counts.n_history, counts.n_target)
    seed = Int(get(feature_set.data, :grw_oos_seed, 0))
    innovations = CB_PG._qs_oos_innovations(chain, config, prefix, n_teams,
                                             seed, counts.n_target)
    return merge(states, innovations)
end

_cb_extract_dynamics(chain::Chains, config::CB_PG.QualityStyleGRW,
                     prefix::String, n_teams::Int) =
    _cb_extract_dynamics(chain, config, prefix, n_teams,
                         (; data = Dict{Symbol,Any}()))

"Held-out next-biweek draw: final fitted state plus a σ_q,K/σ_s,K innovation."
function _cb_oos_dynamics(
    ::CB_PG.QualityStyleGRW, draw, lineup_map, match_id::Int,
    home_index::Int, away_index::Int, n_samples::Int,
)
    final = size(draw.α, 2)
    att(team) = team > 0 ? vec(draw.α[team, final, :]) .+
                           draw.oos_q[:, team] .+ draw.oos_s[:, team] : zeros(n_samples)
    def(team) = team > 0 ? vec(draw.β[team, final, :]) .-
                           draw.oos_q[:, team] .+ draw.oos_s[:, team] : zeros(n_samples)
    return (; att_h = att(home_index), def_a = def(away_index),
            att_a = att(away_index), def_h = def(home_index))
end
