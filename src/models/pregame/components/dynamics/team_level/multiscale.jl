# src/models/pregame/components/dynamics/team_level/multiscale.jl
#
# ==============================================================================
# MultiScaleGRW — two-speed Gaussian random-walk team dynamics
# ==============================================================================
#
# Team attack and defence are latent states that move at two speeds:
#
#   * one MACRO innovation per history season   (scale σₛ)
#   * one MICRO innovation per target-season step, normally a match-biweek (σₖ)
#
# starting from a season-zero level (σ₀). Every state column is zero-centred over
# teams, so the league mean is exactly zero at every point in time and the level
# is identified against the interception.
#
# ------------------------------------------------------------------------------
# THE STATE-COUNT CONTRACT
# ------------------------------------------------------------------------------
#
# The walk emits exactly `n_history + n_target` states, which is what
# `FeatureSet.data[:n_rounds]` promises and what `:time_indices` indexes into:
#
#     state 1              level only                     (first history season)
#     state s, s <= n_hist  level + (s-1) macro steps      (later history seasons)
#     state n_hist + k      ... + k micro steps            (target-season steps)
#
# so there are `n_history - 1` macro transitions, not `n_history`. The superseded
# implementation of this component wrote `cumsum(hcat(init, season, target))` with
# one season column per history season and therefore produced
# `1 + n_history + n_target` states — one too many, silently shifting every state
# against the time index that selects it. Task 007 corrected this; the accumulator
# construction below is the corrected contract and `dynamics_design` asserts it.
#
# ------------------------------------------------------------------------------
# WHY ACCUMULATOR MATRICES RATHER THAN `cumsum`
# ------------------------------------------------------------------------------
#
# The cumulative sum is written as multiplication by three constant 0/1 matrices.
# ReverseDiff preallocates the output of a matrix multiply in the compiled tape, so
# replay is allocation-stable; `cumsum(hcat(...), dims = 2)` allocates fresh scratch
# on every replay because `hcat` cannot be preallocated. Same log density to the
# last bit, materially cheaper gradients.
#
# ------------------------------------------------------------------------------
# RECENCY
# ------------------------------------------------------------------------------
#
# This component carries recency in its LATENT STATES, so it takes unit likelihood
# weights. `TimeDecayDynamics` instead downweights old fixtures in the likelihood.
# Combining both would discount the same evidence twice — see the builder hooks in
# `src/models/pregame/builder/grw_dynamics.jl`.
# ==============================================================================

# ==========================================
# 1. CONFIGURATION
# ==========================================

"""
    MultiScaleGRW

Two-speed non-centred Gaussian random-walk dynamics for team attack and defence.

`z₀`, `zₛ`, `zₖ` are the standardised innovation shapes (the non-centred z-scores)
for the initial level, the per-season macro step and the per-target-step micro step.
`α_σ*` and `β_σ*` are the corresponding scale priors for attack and defence.

Defence scales are deliberately looser than attack at the season boundary
(`β_σ₀`, `β_σₛ`) and tighter within the target season (`β_σₖ`): squads are rebuilt
between seasons but concede at a more stable rate week to week.

```julia
model = CountModelBuilder(:m00_baseline_grw) |>
    add(GlobalInterception()) |>
    add(MultiScaleGRW()) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build
```
"""
# Keep the established nine-field type layout unchanged: historical fit artifacts
# embed this concrete struct through Julia Serialization.
struct MultiScaleGRW <: AbstractDynamicsConfig
    z₀::ContinuousUnivariateDistribution
    zₛ::ContinuousUnivariateDistribution
    zₖ::ContinuousUnivariateDistribution
    α_σ₀::ContinuousUnivariateDistribution
    α_σₛ::ContinuousUnivariateDistribution
    α_σₖ::ContinuousUnivariateDistribution
    β_σ₀::ContinuousUnivariateDistribution
    β_σₛ::ContinuousUnivariateDistribution
    β_σₖ::ContinuousUnivariateDistribution
end

# The opt-in form is a separate concrete type so adding the option does not alter
# the serialized representation of established `MultiScaleGRW` fits.
struct TargetSeasonMultiScaleGRW <: AbstractDynamicsConfig
    z₀::ContinuousUnivariateDistribution
    zₛ::ContinuousUnivariateDistribution
    zₖ::ContinuousUnivariateDistribution
    α_σ₀::ContinuousUnivariateDistribution
    α_σₛ::ContinuousUnivariateDistribution
    α_σₖ::ContinuousUnivariateDistribution
    β_σ₀::ContinuousUnivariateDistribution
    β_σₛ::ContinuousUnivariateDistribution
    β_σₖ::ContinuousUnivariateDistribution
end

const AnyMultiScaleGRW = Union{MultiScaleGRW,TargetSeasonMultiScaleGRW}
_target_season_step(::MultiScaleGRW) = false
_target_season_step(::TargetSeasonMultiScaleGRW) = true
Base.getproperty(config::AnyMultiScaleGRW, name::Symbol) =
    name === :target_season_step ? _target_season_step(config) : getfield(config, name)

function MultiScaleGRW(;
    z₀::ContinuousUnivariateDistribution = Normal(0, 1),
    zₛ::ContinuousUnivariateDistribution = Normal(0, 1),
    zₖ::ContinuousUnivariateDistribution = Normal(0, 1),
    α_σ₀::ContinuousUnivariateDistribution = Gamma(2, 0.06),
    α_σₛ::ContinuousUnivariateDistribution = Gamma(2, 0.03),
    α_σₖ::ContinuousUnivariateDistribution = Gamma(2, 0.015),
    β_σ₀::ContinuousUnivariateDistribution = Gamma(2, 0.10),
    β_σₛ::ContinuousUnivariateDistribution = Gamma(2, 0.055),
    β_σₖ::ContinuousUnivariateDistribution = Gamma(2, 0.012),
    target_season_step::Bool = false,
)
    constructor = target_season_step ? TargetSeasonMultiScaleGRW : MultiScaleGRW
    return constructor(z₀, zₛ, zₖ, α_σ₀, α_σₛ, α_σₖ, β_σ₀, β_σₛ, β_σₖ)
end

# ==========================================
# 2. ACCUMULATORS
# ==========================================

"""
    grw_accumulators(n_history, n_target; target_season_step = false) -> (; initial, season, target)

The cumulative sum of the walk, expressed as three constant 0/1 linear maps.

Each returned matrix is `(n_innovations, n_rounds)` with `n_rounds = n_history +
n_target`, and entry `[i, s]` is 1 exactly when innovation `i` has already occurred
by state `s`. Multiplying the innovations by these and adding gives the walk
without a single `cumsum` or `hcat` on the AD tape.

Throws when `n_history < 1`: a walk needs a level to start from.
"""
function grw_accumulators(n_history::Int, n_target::Int;
                          target_season_step::Bool = false)
    n_history >= 1 ||
        error("MultiScaleGRW requires at least one history season; got $n_history")
    n_target >= 0 ||
        error("MultiScaleGRW target-step count is negative: $n_target")

    # The opt-in boundary innovation has an observed target state only when the
    # fold contains target steps. A zero-target fold integrates it at OOS time,
    # rather than sampling a site absent from its likelihood.
    n_season_transitions = n_history - 1 + Int(target_season_step && n_target >= 1)
    n_rounds = n_history + n_target

    # The level is present in every state.
    initial = ones(Float64, 1, n_rounds)

    # History macro transition t moves the walk from history season t into
    # season t+1, so it is present from state t+1 onward. The optional final
    # row is the summer boundary and first appears at the first target state.
    season = zeros(Float64, n_season_transitions, n_rounds)
    for transition in 1:(n_history - 1)
        season[transition, (transition + 1):n_rounds] .= 1.0
    end
    if target_season_step && n_target >= 1
        season[end, (n_history + 1):n_rounds] .= 1.0
    end

    # Micro step k is present from the k-th target state onward.
    target = zeros(Float64, n_target, n_rounds)
    for step in 1:n_target
        target[step, (n_history + step):n_rounds] .= 1.0
    end

    return (; initial, season, target)
end

# ==========================================
# 3. TURING SUBMODELS
# ==========================================

"""
    _grw_centre(raw, n_teams) -> raw minus its per-round team mean

Zero-sum across teams at every round, `raw .- mean(raw, dims = 1)`, with the mean written as a
`(1 × n_teams) * (n_teams × n_rounds)` product.

ReverseDiff 1.17 has no rule for `mean(::TrackedMatrix; dims)`: it falls back to element-wise
taping, which put ~8,700 scalar instructions per side on the W2 largest-fold tape (17,619 of its
instructions) and returned an `Array{TrackedReal}` whose subtraction allocated on every replay.
The matrix product is one preallocated instruction. Same quantity; summation order differs
(≤ 1e-12 relative, `scripts/tape_allocation_audit.jl --compare-parity`). `Float64` and ForwardDiff
evaluation go through the same product.
"""
function _grw_centre(raw, n_teams::Int)
    column_mean = fill(1.0 / n_teams, 1, n_teams) * raw
    return raw .- column_mean
end

"""
One side (attack or defence) of the walk, with target-season micro steps.

Returns a `(n_teams, n_rounds)` matrix of zero-centred states.
"""
@model function _grw_trajectory(
    z_initial_prior, z_season_prior, z_target_prior,
    scale_initial_prior, scale_season_prior, scale_target_prior,
    initial_accumulator::Matrix{Float64},
    season_accumulator::Matrix{Float64},
    target_accumulator::Matrix{Float64},
    n_teams::Int, n_season_transitions::Int, n_target::Int,
)
    σ₀ ~ scale_initial_prior
    σₛ ~ scale_season_prior
    σₖ ~ scale_target_prior

    z_init ~ filldist(z_initial_prior, n_teams)
    z_season ~ filldist(z_season_prior, n_teams, n_season_transitions)
    z_target ~ filldist(z_target_prior, n_teams, n_target)

    initial_states = reshape(z_init .* σ₀, n_teams, 1) * initial_accumulator
    season_states = (z_season .* σₛ) * season_accumulator
    target_states = (z_target .* σₖ) * target_accumulator
    raw = initial_states .+ season_states .+ target_states
    return _grw_centre(raw, n_teams)
end

"""
One side of the walk for a fold whose target block has no observed steps.

`σₖ` and `z_target` are not sampled at all, rather than sampled and multiplied by an
empty accumulator: an unused site would still widen θ and appear in the chain.
"""
@model function _grw_trajectory_no_target(
    z_initial_prior, z_season_prior,
    scale_initial_prior, scale_season_prior,
    initial_accumulator::Matrix{Float64},
    season_accumulator::Matrix{Float64},
    n_teams::Int, n_season_transitions::Int,
)
    σ₀ ~ scale_initial_prior
    σₛ ~ scale_season_prior

    z_init ~ filldist(z_initial_prior, n_teams)
    z_season ~ filldist(z_season_prior, n_teams, n_season_transitions)

    initial_states = reshape(z_init .* σ₀, n_teams, 1) * initial_accumulator
    season_states = (z_season .* σₛ) * season_accumulator
    raw = initial_states .+ season_states
    return _grw_centre(raw, n_teams)
end

"""
Attack and defence walks, sharing innovation shapes but not scales.

`has_target` is a `Val` so the branch is resolved at compile time and never appears
inside the model body, which is the AD-safety rule this repository enforces.
"""
@model function _grw_pair(
    config::AnyMultiScaleGRW,
    initial_accumulator::Matrix{Float64},
    season_accumulator::Matrix{Float64},
    target_accumulator::Matrix{Float64},
    n_teams::Int, n_season_transitions::Int, n_target::Int,
    ::Val{true},
)
    α ~ to_submodel(_grw_trajectory(
        config.z₀, config.zₛ, config.zₖ,
        config.α_σ₀, config.α_σₛ, config.α_σₖ,
        initial_accumulator, season_accumulator, target_accumulator,
        n_teams, n_season_transitions, n_target))
    β ~ to_submodel(_grw_trajectory(
        config.z₀, config.zₛ, config.zₖ,
        config.β_σ₀, config.β_σₛ, config.β_σₖ,
        initial_accumulator, season_accumulator, target_accumulator,
        n_teams, n_season_transitions, n_target))
    return (; α, β)
end

@model function _grw_pair(
    config::AnyMultiScaleGRW,
    initial_accumulator::Matrix{Float64},
    season_accumulator::Matrix{Float64},
    ::Matrix{Float64},
    n_teams::Int, n_season_transitions::Int, ::Int,
    ::Val{false},
)
    α ~ to_submodel(_grw_trajectory_no_target(
        config.z₀, config.zₛ, config.α_σ₀, config.α_σₛ,
        initial_accumulator, season_accumulator, n_teams, n_season_transitions))
    β ~ to_submodel(_grw_trajectory_no_target(
        config.z₀, config.zₛ, config.β_σ₀, config.β_σₛ,
        initial_accumulator, season_accumulator, n_teams, n_season_transitions))
    return (; α, β)
end

"""
    build_dynamics(config::AnyMultiScaleGRW, n_teams, n_history, n_target)

Legacy four-argument entry point, used by the hand-written `standard/` team engines.

The composable builder does not call this: it precomputes the accumulators once per
fold in `dynamics_design` rather than rebuilding them on every model construction.
Retained so the legacy engines keep working, and corrected to the
`n_history + n_target` state contract described at the top of this file.
"""
@model function build_dynamics(config::AnyMultiScaleGRW, n_teams::Int,
                               n_history::Int, n_target::Int)
    acc = grw_accumulators(n_history, n_target;
                           target_season_step = config.target_season_step)
    state ~ to_submodel(
        _grw_pair(config, acc.initial, acc.season, acc.target,
                  n_teams, size(acc.season, 1), n_target,
                  n_target == 0 ? Val(false) : Val(true)),
        false)
    return (; α = state.α, β = state.β)
end

# ==========================================
# 4. POSTERIOR RECONSTRUCTION
# ==========================================

"""
Resolve a chain site whose name may or may not carry spaces after the commas.

Turing's index rendering has varied across versions; both spellings are accepted so
a stored chain stays readable by a later release.
"""
function _grw_chain_symbol(chain::Chains, base::String, indices::Int...)
    isempty(indices) && return Symbol(base)
    spaced = Symbol("$base[$(join(indices, ", "))]")
    compact = Symbol("$base[$(join(indices, ","))]")
    available = Set(names(chain))
    spaced in available && return spaced
    compact in available && return compact
    return error("chain has no site $spaced or $compact")
end

"""
    _grw_reconstruct_trajectory(chain, prefix, n_teams, n_history, n_target)

Rebuild one side's full state trajectory from the posterior draws.

Returns `(n_teams, n_rounds, n_samples)` — teams, time, draws — which is the layout
the composable extractor and the out-of-sample hook index into.
"""
function _grw_reconstruct_trajectory(chain::Chains, prefix::String,
                                     n_teams::Int, n_history::Int, n_target::Int;
                                     target_season_step::Bool = false)
    n_samples = size(chain, 1) * size(chain, 3)
    n_season_transitions = n_history - 1 + Int(target_season_step && n_target >= 1)

    σ₀ = reshape(vec(Array(chain[_grw_chain_symbol(chain, "$prefix.σ₀")])), n_samples, 1, 1)
    σₛ = reshape(vec(Array(chain[_grw_chain_symbol(chain, "$prefix.σₛ")])), n_samples, 1, 1)

    z_init = Array{Float64}(undef, n_samples, n_teams, 1)
    for team in 1:n_teams
        z_init[:, team, 1] =
            vec(Array(chain[_grw_chain_symbol(chain, "$prefix.z_init", team)]))
    end

    z_season = Array{Float64}(undef, n_samples, n_teams, n_season_transitions)
    for step in 1:n_season_transitions, team in 1:n_teams
        z_season[:, team, step] =
            vec(Array(chain[_grw_chain_symbol(chain, "$prefix.z_season", team, step)]))
    end

    target_increments = if n_target == 0
        zeros(Float64, n_samples, n_teams, 0)
    else
        σₖ = reshape(
            vec(Array(chain[_grw_chain_symbol(chain, "$prefix.σₖ")])), n_samples, 1, 1)
        z_target = Array{Float64}(undef, n_samples, n_teams, n_target)
        for step in 1:n_target, team in 1:n_teams
            z_target[:, team, step] =
                vec(Array(chain[_grw_chain_symbol(chain, "$prefix.z_target", team, step)]))
        end
        z_target .* σₖ
    end

    increments = cat(z_init .* σ₀, z_season .* σₛ, target_increments; dims = 3)
    raw = cumsum(increments, dims = 3)
    centered = raw .- mean(raw, dims = 2)
    return permutedims(centered, (2, 3, 1))
end

"""
    extract_dynamics(chain, ::AnyMultiScaleGRW, prefix, n_teams, n_history, n_target)

Legacy six-argument extractor matching the legacy `build_dynamics` above.

Returns `α` and `β` as `(n_teams, n_rounds, n_samples)` arrays.
"""
function extract_dynamics(chain::Chains, config::AnyMultiScaleGRW, prefix::String,
                          n_teams::Int, n_history::Int, n_target::Int)
    return (;
        α = _grw_reconstruct_trajectory(chain, "$prefix.α", n_teams, n_history, n_target;
                                         target_season_step = config.target_season_step),
        β = _grw_reconstruct_trajectory(chain, "$prefix.β", n_teams, n_history, n_target;
                                         target_season_step = config.target_season_step),
    )
end

"""
    grw_step_counts(chain, prefix) -> (; n_history, n_target)

Recover the walk's shape from the chain itself.

The composable extractor is handed a chain and a team count but not the fold's time
geometry, so the geometry is counted back off the site names. Counting team 1's
sites is enough because `filldist` emits a full rectangular grid.
"""
function grw_step_counts(chain::Chains, prefix::String, config::AnyMultiScaleGRW)
    available = String.(names(chain))
    n_target = count(name -> startswith(name, "$prefix.α.z_target[1,"), available)
    n_season_transitions = count(name -> startswith(name, "$prefix.α.z_season[1,"), available)
    boundary_count = Int(config.target_season_step && n_target >= 1)
    n_history = n_season_transitions - boundary_count + 1
    n_history >= 1 || error("MultiScaleGRW chain has invalid macro-step geometry")
    return (; n_history, n_target)
end

grw_step_counts(chain::Chains, prefix::String) =
    grw_step_counts(chain, prefix, MultiScaleGRW())

"Deterministic, per-fold posterior-predictive summer innovations for an empty target block."
function _grw_oos_boundary_innovations(chain::Chains, prefix::String,
                                       n_teams::Int, seed::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    σₛ = reshape(vec(Array(chain[_grw_chain_symbol(chain, "$prefix.σₛ")])), n_samples, 1)
    side_seed = 2 * seed + (endswith(prefix, ".β") ? 1 : 0)
    rng = Random.MersenneTwister(side_seed)
    z = Random.randn(rng, n_samples, n_teams)
    return (z .- mean(z, dims = 2)) .* σₛ
end
