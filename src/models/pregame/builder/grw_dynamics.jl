# src/models/pregame/builder/grw_dynamics.jl
#
# ==============================================================================
# Composable-builder adapter for MultiScaleGRW
# ==============================================================================
#
# The mathematics lives in
# `src/models/pregame/components/dynamics/team_level/multiscale.jl`. This file is
# only the seam that lets `CountModelBuilder` carry it: the validation hook, the
# likelihood-weighting contract, the θ site list, the per-fold design object, the
# in-model effects submodel, and the two posterior hooks.
#
# Every other dynamics component reaches the builder through the same nine
# functions, so adding this one required no change to `engine.jl` or `builder.jl`
# beyond listing the type as supported.
# ==============================================================================

# ==============================================================================
# 1. VALIDATION AND WEIGHTING CONTRACT
# ==============================================================================

_cb_dynamics_supported(::CB_PG.MultiScaleGRW) = true

# MultiScaleGRW represents recency as latent state, so the likelihood must weight
# every fixture equally. Applying time decay ON TOP of a random walk would discount
# the same evidence twice: once by shrinking an old fixture's contribution to the
# log density, and again by letting the walk drift away from the state that fitted
# it. `TimeDecayDynamics` makes the opposite choice — no state motion, decayed
# weights — and the two are alternatives rather than layers.
_dynamics_weighting_valid(::CB_PG.MultiScaleGRW) = true
_dynamics_weighting_detail(::CB_PG.MultiScaleGRW) =
    "unit likelihood weights; recency is represented by latent macro/micro states"
dynamics_match_weights(::CB_PG.MultiScaleGRW, dates::Vector{Float64}) =
    ones(Float64, length(dates))

# Declaration order inside `_grw_pair`, which is the θ layout. A fold with no
# observed target steps does not sample `σₖ`/`z_target` (see
# `_grw_trajectory_no_target`), so those two sites are absent from such a chain.
_sites_dynamics(::CB_PG.MultiScaleGRW) = [
    Symbol("dyn.α.σ₀"), Symbol("dyn.α.σₛ"), Symbol("dyn.α.σₖ"),
    Symbol("dyn.α.z_init"), Symbol("dyn.α.z_season"), Symbol("dyn.α.z_target"),
    Symbol("dyn.β.σ₀"), Symbol("dyn.β.σₛ"), Symbol("dyn.β.σₖ"),
    Symbol("dyn.β.z_init"), Symbol("dyn.β.z_season"), Symbol("dyn.β.z_target"),
]

# ==============================================================================
# 2. PER-FOLD DESIGN
# ==============================================================================

"""
    GRWDynamicsDesign

Everything about one fold's time geometry that `MultiScaleGRW` needs, computed once
outside `@model`.

The Cartesian index vectors turn "each fixture reads its own team's state at its own
time step" into a single vectorised gather on the tape, instead of a scalar loop
over fixtures. The three accumulators are the walk's cumulative sum as constant
linear maps — see the component file for why that beats `cumsum`.

`target_marker` is `Val{true}` or `Val{false}` so the no-target branch is a dispatch
rather than a runtime `if` inside the model.
"""
struct GRWDynamicsDesign{T}
    home_state_indices::Vector{CartesianIndex{2}}
    away_state_indices::Vector{CartesianIndex{2}}
    initial_accumulator::Matrix{Float64}
    season_accumulator::Matrix{Float64}
    target_accumulator::Matrix{Float64}
    target_marker::T
    n_history::Int
    n_target::Int
    n_rounds::Int
end

"""
    dynamics_design(::MultiScaleGRW, feature_set, n_matches) -> GRWDynamicsDesign

Build the fold's design, asserting the state-count contract on the way.

The `n_rounds == n_history + n_target` check is the guard against the off-by-one
that the superseded implementation of this component shipped with: a walk emitting
one state too many still runs, still samples and still scores — it just reads every
fixture against the wrong state. That failure is silent, so it is checked here
rather than trusted.
"""
function dynamics_design(::CB_PG.MultiScaleGRW, feature_set, n_matches::Int)
    d = feature_set.data
    home_ids = Vector{Int}(d[:flat_home_ids])
    away_ids = Vector{Int}(d[:flat_away_ids])
    time_indices = Vector{Int}(d[:time_indices])
    n_history = Int(d[:n_history_steps])
    n_target = Int(d[:n_target_steps])
    n_rounds = Int(d[:n_rounds])

    n_rounds == n_history + n_target || error(
        "MultiScaleGRW time contract mismatch: n_rounds=$n_rounds, " *
        "n_history=$n_history, n_target=$n_target")
    for (name, v) in (("flat_home_ids", home_ids), ("flat_away_ids", away_ids),
                      ("time_indices", time_indices))
        length(v) == n_matches || error(
            "MultiScaleGRW design vector $name has length $(length(v)); expected $n_matches")
    end
    all(t -> 1 <= t <= n_rounds, time_indices) ||
        error("MultiScaleGRW time index outside 1:$n_rounds")

    acc = CB_PG.grw_accumulators(n_history, n_target)
    return GRWDynamicsDesign(
        CartesianIndex.(home_ids, time_indices),
        CartesianIndex.(away_ids, time_indices),
        acc.initial, acc.season, acc.target,
        n_target == 0 ? Val(false) : Val(true),
        n_history, n_target, n_rounds,
    )
end

# ==============================================================================
# 3. IN-MODEL EFFECTS
# ==============================================================================

# Unlike the static components, which index states by team alone, this one indexes
# by (team, time): a fixture reads the state its own time step, so the same team
# contributes differently in September and in April.
@model function _cb_dynamics_effects(
    config::CB_PG.MultiScaleGRW,
    home_ids::Vector{Int}, away_ids::Vector{Int},
    design::GRWDynamicsDesign, n_teams::Int,
)
    state ~ to_submodel(
        CB_PG._grw_pair(
            config,
            design.initial_accumulator,
            design.season_accumulator,
            design.target_accumulator,
            n_teams, design.n_history - 1, design.n_target,
            design.target_marker),
        false)
    return (;
        att_h = state.α[design.home_state_indices],
        def_a = state.β[design.away_state_indices],
        att_a = state.α[design.away_state_indices],
        def_h = state.β[design.home_state_indices],
    )
end

# ==============================================================================
# 4. POSTERIOR HOOKS
# ==============================================================================

# The generic composable extractor hands over a chain and a team count but not the
# fold's time geometry, so the geometry is recovered from the site names.
function _cb_extract_dynamics(chain::Chains, ::CB_PG.MultiScaleGRW,
                              prefix::String, n_teams::Int)
    counts = CB_PG.grw_step_counts(chain, prefix)
    return (;
        α = CB_PG._grw_reconstruct_trajectory(
            chain, "$prefix.α", n_teams, counts.n_history, counts.n_target),
        β = CB_PG._grw_reconstruct_trajectory(
            chain, "$prefix.β", n_teams, counts.n_history, counts.n_target),
    )
end

"""
Price a held-out fixture from the LAST state the fold could see.

A walk has no opinion about a step it never observed, so the honest forecast for the
next match-biweek is the final fitted state carried forward — the random walk's own
conditional mean. Extrapolating the drift instead would invent motion the posterior
never measured.

Note the trailing `:` indexing: `draw.α` here is `(teams, time, samples)`, not the
`(samples, teams)` matrix the static components return.
"""
function _cb_oos_dynamics(
    ::CB_PG.MultiScaleGRW, draw, lineup_map, match_id::Int,
    home_index::Int, away_index::Int, n_samples::Int,
)
    final = size(draw.α, 2)
    return (;
        att_h = home_index > 0 ? vec(draw.α[home_index, final, :]) : zeros(n_samples),
        def_a = away_index > 0 ? vec(draw.β[away_index, final, :]) : zeros(n_samples),
        att_a = away_index > 0 ? vec(draw.α[away_index, final, :]) : zeros(n_samples),
        def_h = home_index > 0 ? vec(draw.β[home_index, final, :]) : zeros(n_samples),
    )
end
