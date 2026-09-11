module MultiScaleGRWPrototype

# ==============================================================================
# Task 007 loader — MultiScaleGRW adapter for the composable count-model engine
# ==============================================================================
#
# Definitions only. `r01_runner.jl` owns execution.
#
# This prototype deliberately adapts the package's existing, currently unexported
# `PreGame.MultiScaleGRW` configuration instead of creating a second serialized
# model type. The old component's trajectory implementation is not called: the
# methods below provide the corrected state count, branch-free Turing submodels,
# vectorised state indexing, held-out extraction, persistence helpers, and reports.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using ForwardDiff
using LibPQ
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using Printf
using Random
using ReverseDiff
using Serialization
using SpecialFunctions
using Statistics
using Turing
using UUIDs

const L01_PG = BayesianFootball.Models.PreGame
const L01_BUILDER = BayesianFootball.Models.PreGame.Builder
const L01_FEATURES = BayesianFootball.Features
const L01_EVALUATION = BayesianFootball.Evaluation

import BayesianFootball.Models.PreGame.Builder:
    _cb_dynamics_effects,
    _cb_dynamics_supported,
    _cb_extract_dynamics,
    _cb_oos_dynamics,
    _dynamics_weighting_detail,
    _dynamics_weighting_valid,
    _sites_dynamics,
    dynamics_design,
    dynamics_match_weights

export L01Config, GRWDynamicsDesign
export l01_multiscale_dynamics, l01_models, l01_splitter
export l01_preflight_sampler, l01_production_sampler, l01_thresholds
export l01_fit_configs, l01_register!, l01_load_data, l01_database
export l01_scored_inputs, l01_gradient_audit, l01_fit_or_load, l01_extend_sampling
export l01_assert_coverage, l01_assert_promotion, l01_evaluate
export l01_load_baseline, l01_compare, l01_phase1_passed
export l01_persist_scores!, l01_write_report!, l01_save_stage_result

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

Base.@kwdef struct L01Config
    experiment::String = "scottish_lower_multiscale_grw_2426"
    save_root::String = joinpath(@__DIR__, "results")
    report_path::String = joinpath(@__DIR__, "README.md")
    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos_matches::Int = 710
    preflight_folds::Int = 2
    preflight_samples::Int = 400
    preflight_warmup::Int = 400
    production_samples::Int = 800
    production_warmup::Int = 800
    chains::Int = 4
    accept_rate::Float64 = 0.90
    max_depth::Int = 10
    max_concurrent_tasks::Int = 16
    gradient_replays::Int = 200
    max_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end

const L01_MODEL_NAMES = (
    "m00_baseline_grw",
    "m05_production_wealth_grw",
    "m05_joint_production_wealth_grw",
)

const L01_PHASE1_NAMES = L01_MODEL_NAMES[1:2]
const L01_PHASE2_NAMES = L01_MODEL_NAMES[3:3]

const L01_DESCRIPTIONS = Dict(
    L01_MODEL_NAMES[1] =>
        "Poisson baseline with non-centred MultiScaleGRW team attack and defence states.",
    L01_MODEL_NAMES[2] =>
        "Poisson production-wealth model with non-centred MultiScaleGRW team states.",
    L01_MODEL_NAMES[3] =>
        "Two-arm Joint Gamma-Poisson production-wealth model with non-centred MultiScaleGRW team states.",
)

const L01_TAGS = [
    "production",
    "scottish_lower",
    "24/25",
    "25/26",
    "multiscale_grw",
    "todo007",
    "reversediff",
]

# ==============================================================================
# 2. Corrected non-centred MultiScaleGRW trajectory
# ==============================================================================

"""
Training-time state indices for one fold.

The feature layer numbers history seasons first and target-season match-biweeks
second. Cartesian indices are built outside `@model`, so selecting a tracked state
matrix is one vectorised ReverseDiff gather rather than a scalar loop.
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
    l01_grw_trajectory(config, n_teams, n_history, n_target, attack)

One side of the non-centred walk. With `n_history >= 1` and `n_target >= 1`,

    state = cumsum([z_init*σ₀, z_season*σₛ, z_target*σₖ], dims=2)

where `z_season` has `n_history - 1` transition columns and `z_target` has one
column per observed target-season step. This gives exactly
`n_history + n_target` states, matching `FeatureSet.data[:time_indices]`.
Every state column is zero-centred over teams.
"""
@model function l01_grw_trajectory(
    z_initial_prior,
    z_season_prior,
    z_target_prior,
    scale_initial_prior,
    scale_season_prior,
    scale_target_prior,
    initial_accumulator::Matrix{Float64},
    season_accumulator::Matrix{Float64},
    target_accumulator::Matrix{Float64},
    n_teams::Int,
    n_season_transitions::Int,
    n_target::Int,
)
    σ₀ ~ scale_initial_prior
    σₛ ~ scale_season_prior
    σₖ ~ scale_target_prior

    z_init ~ filldist(z_initial_prior, n_teams)
    z_season ~ filldist(z_season_prior, n_teams, n_season_transitions)
    z_target ~ filldist(z_target_prior, n_teams, n_target)

    # These three constant linear maps are the cumulative sum written as matrix
    # multiplication. ReverseDiff preallocates their outputs in the compiled tape;
    # unlike `cumsum(hcat(...), dims=2)`, replay does not allocate scratch arrays.
    initial_states = reshape(z_init .* σ₀, n_teams, 1) * initial_accumulator
    season_states = (z_season .* σₛ) * season_accumulator
    target_states = (z_target .* σₖ) * target_accumulator
    raw = initial_states .+ season_states .+ target_states
    return raw .- mean(raw, dims = 1)
end

@model function l01_grw_trajectory_no_target(
    z_initial_prior,
    z_season_prior,
    scale_initial_prior,
    scale_season_prior,
    initial_accumulator::Matrix{Float64},
    season_accumulator::Matrix{Float64},
    n_teams::Int,
    n_season_transitions::Int,
)
    σ₀ ~ scale_initial_prior
    σₛ ~ scale_season_prior

    z_init ~ filldist(z_initial_prior, n_teams)
    z_season ~ filldist(z_season_prior, n_teams, n_season_transitions)

    initial_states = reshape(z_init .* σ₀, n_teams, 1) * initial_accumulator
    season_states = (z_season .* σₛ) * season_accumulator
    raw = initial_states .+ season_states
    return raw .- mean(raw, dims = 1)
end

@model function l01_grw_pair(config::L01_PG.MultiScaleGRW,
                             design::GRWDynamicsDesign{Val{true}}, n_teams::Int)
    n_season_transitions = design.n_history - 1

    α ~ to_submodel(l01_grw_trajectory(
        config.z₀,
        config.zₛ,
        config.zₖ,
        config.α_σ₀,
        config.α_σₛ,
        config.α_σₖ,
        design.initial_accumulator,
        design.season_accumulator,
        design.target_accumulator,
        n_teams,
        n_season_transitions,
        design.n_target,
    ))
    β ~ to_submodel(l01_grw_trajectory(
        config.z₀,
        config.zₛ,
        config.zₖ,
        config.β_σ₀,
        config.β_σₛ,
        config.β_σₖ,
        design.initial_accumulator,
        design.season_accumulator,
        design.target_accumulator,
        n_teams,
        n_season_transitions,
        design.n_target,
    ))
    return (; α, β)
end

@model function l01_grw_pair(config::L01_PG.MultiScaleGRW,
                             design::GRWDynamicsDesign{Val{false}}, n_teams::Int)
    n_season_transitions = design.n_history - 1

    α ~ to_submodel(l01_grw_trajectory_no_target(
        config.z₀,
        config.zₛ,
        config.α_σ₀,
        config.α_σₛ,
        design.initial_accumulator,
        design.season_accumulator,
        n_teams,
        n_season_transitions,
    ))
    β ~ to_submodel(l01_grw_trajectory_no_target(
        config.z₀,
        config.zₛ,
        config.β_σ₀,
        config.β_σₛ,
        design.initial_accumulator,
        design.season_accumulator,
        n_teams,
        n_season_transitions,
    ))
    return (; α, β)
end

# Composable-builder adapter. These are prototype methods on package-owned generic
# functions, scoped to the package-owned MultiScaleGRW type.
_cb_dynamics_supported(::L01_PG.MultiScaleGRW) = true
_dynamics_weighting_valid(::L01_PG.MultiScaleGRW) = true
_dynamics_weighting_detail(::L01_PG.MultiScaleGRW) =
    "unit likelihood weights; recency is represented by latent macro/micro states"
dynamics_match_weights(::L01_PG.MultiScaleGRW, dates::Vector{Float64}) =
    ones(Float64, length(dates))
_sites_dynamics(::L01_PG.MultiScaleGRW) = Symbol[
    Symbol("dyn.α.σ₀"),
    Symbol("dyn.α.σₛ"),
    Symbol("dyn.α.σₖ"),
    Symbol("dyn.α.z_init"),
    Symbol("dyn.α.z_season"),
    Symbol("dyn.α.z_target"),
    Symbol("dyn.β.σ₀"),
    Symbol("dyn.β.σₛ"),
    Symbol("dyn.β.σₖ"),
    Symbol("dyn.β.z_init"),
    Symbol("dyn.β.z_season"),
    Symbol("dyn.β.z_target"),
]

function l01_accumulators(n_history::Int, n_target::Int)
    n_season_transitions = n_history - 1
    n_rounds = n_history + n_target

    initial = ones(Float64, 1, n_rounds)
    season = zeros(Float64, n_season_transitions, n_rounds)
    for transition in 1:n_season_transitions
        season[transition, (transition + 1):n_rounds] .= 1.0
    end
    target = zeros(Float64, n_target, n_rounds)
    for transition in 1:n_target
        target[transition, (n_history + transition):n_rounds] .= 1.0
    end
    return (; initial, season, target)
end

function dynamics_design(::L01_PG.MultiScaleGRW, feature_set, n_matches::Int)
    data = feature_set.data
    home_ids = Vector{Int}(data[:flat_home_ids])
    away_ids = Vector{Int}(data[:flat_away_ids])
    time_indices = Vector{Int}(data[:time_indices])
    n_history = Int(data[:n_history_steps])
    n_target = Int(data[:n_target_steps])
    n_rounds = Int(data[:n_rounds])

    n_history >= 1 || error("MultiScaleGRW requires at least one history season; got $n_history")
    n_target >= 0 || error("MultiScaleGRW target-step count is negative: $n_target")
    n_rounds == n_history + n_target || error(
        "MultiScaleGRW time contract mismatch: n_rounds=$n_rounds, " *
        "n_history=$n_history, n_target=$n_target")
    length(home_ids) == n_matches || error(
        "MultiScaleGRW home IDs have length $(length(home_ids)); expected $n_matches")
    length(away_ids) == n_matches || error(
        "MultiScaleGRW away IDs have length $(length(away_ids)); expected $n_matches")
    length(time_indices) == n_matches || error(
        "MultiScaleGRW time indices have length $(length(time_indices)); expected $n_matches")
    all(t -> 1 <= t <= n_rounds, time_indices) || error(
        "MultiScaleGRW time index outside 1:$n_rounds")

    accumulators = l01_accumulators(n_history, n_target)
    target_marker = n_target == 0 ? Val(false) : Val(true)
    return GRWDynamicsDesign(
        CartesianIndex.(home_ids, time_indices),
        CartesianIndex.(away_ids, time_indices),
        accumulators.initial,
        accumulators.season,
        accumulators.target,
        target_marker,
        n_history,
        n_target,
        n_rounds,
    )
end

@model function _cb_dynamics_effects(
    config::L01_PG.MultiScaleGRW,
    home_ids::Vector{Int},
    away_ids::Vector{Int},
    design::GRWDynamicsDesign,
    n_teams::Int,
)
    state ~ to_submodel(l01_grw_pair(config, design, n_teams), false)
    return (;
        att_h = state.α[design.home_state_indices],
        def_a = state.β[design.away_state_indices],
        att_a = state.α[design.away_state_indices],
        def_h = state.β[design.home_state_indices],
    )
end

# ==============================================================================
# 3. Posterior reconstruction and held-out extraction
# ==============================================================================

function l01_chain_symbol(chain::Chains, base::String, indices::Int...)
    isempty(indices) && return Symbol(base)
    spaced = Symbol("$base[$(join(indices, ", "))]")
    compact = Symbol("$base[$(join(indices, ","))]")
    names_set = Set(names(chain))
    spaced in names_set && return spaced
    compact in names_set && return compact
    error("chain has no site $spaced or $compact")
end

function l01_reconstruct_trajectory(chain::Chains, prefix::String,
                                    n_teams::Int, n_history::Int, n_target::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    n_season_transitions = n_history - 1

    σ₀ = reshape(vec(Array(chain[l01_chain_symbol(chain, "$prefix.σ₀")])), n_samples, 1, 1)
    σₛ = reshape(vec(Array(chain[l01_chain_symbol(chain, "$prefix.σₛ")])), n_samples, 1, 1)

    z_init = Array{Float64}(undef, n_samples, n_teams, 1)
    for team in 1:n_teams
        z_init[:, team, 1] = vec(Array(chain[
            l01_chain_symbol(chain, "$prefix.z_init", team)]))
    end

    z_season = Array{Float64}(undef, n_samples, n_teams, n_season_transitions)
    for step in 1:n_season_transitions
        for team in 1:n_teams
            z_season[:, team, step] = vec(Array(chain[
                l01_chain_symbol(chain, "$prefix.z_season", team, step)]))
        end
    end

    target_increments = if n_target == 0
        zeros(Float64, n_samples, n_teams, 0)
    else
        σₖ = reshape(
            vec(Array(chain[l01_chain_symbol(chain, "$prefix.σₖ")])),
            n_samples,
            1,
            1,
        )
        z_target = Array{Float64}(undef, n_samples, n_teams, n_target)
        for step in 1:n_target
            for team in 1:n_teams
                z_target[:, team, step] = vec(Array(chain[
                    l01_chain_symbol(chain, "$prefix.z_target", team, step)]))
            end
        end
        z_target .* σₖ
    end

    increments = cat(z_init .* σ₀, z_season .* σₛ, target_increments; dims = 3)
    raw = cumsum(increments, dims = 3)
    centered = raw .- mean(raw, dims = 2)
    return permutedims(centered, (2, 3, 1))
end

function _cb_extract_dynamics(chain::Chains, ::L01_PG.MultiScaleGRW,
                              prefix::String, n_teams::Int)
    names_set = Set(String.(names(chain)))
    n_target = count(name -> startswith(name, "$prefix.α.z_target[1,"), names_set)
    n_season_transitions = count(
        name -> startswith(name, "$prefix.α.z_season[1,"), names_set)
    n_history = n_season_transitions + 1
    return (;
        α = l01_reconstruct_trajectory(chain, "$prefix.α", n_teams, n_history, n_target),
        β = l01_reconstruct_trajectory(chain, "$prefix.β", n_teams, n_history, n_target),
    )
end

function _cb_oos_dynamics(::L01_PG.MultiScaleGRW, draw,
                          lineup_map, match_id::Int,
                          home_index::Int, away_index::Int, n_samples::Int)
    final_state = size(draw.α, 2)
    return (;
        att_h = home_index > 0 ? vec(draw.α[home_index, final_state, :]) : zeros(n_samples),
        def_a = away_index > 0 ? vec(draw.β[away_index, final_state, :]) : zeros(n_samples),
        att_a = away_index > 0 ? vec(draw.α[away_index, final_state, :]) : zeros(n_samples),
        def_h = home_index > 0 ? vec(draw.β[home_index, final_state, :]) : zeros(n_samples),
    )
end

"""
Specialised extraction for composable models carrying MultiScaleGRW.

This mirrors the package's generic composable extractor but reconstructs the 3-D
state trajectories and uses the final fitted target-season state for the next
held-out match-biweek.
"""
function L01_PG.extract_parameters(
    model::L01_BUILDER.PoissonCountModel{I,T,H,C,O,G},
    df::AbstractDataFrame,
    feature_set,
    chain::Chains,
) where {I,T<:L01_PG.MultiScaleGRW,H,C,O,G}
    data = feature_set.data
    n_teams = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_history = Int(data[:n_history_steps])
    n_target = Int(data[:n_target_steps])
    team_map = data[:team_map]
    n_samples = size(chain, 1) * size(chain, 3)

    inter = L01_PG.extract_interception(chain, model.interception, n_seasons)
    home_advantage = L01_PG.extract_home_advantage(chain, model.home_advantage, n_teams)
    dynamics = (;
        α = l01_reconstruct_trajectory(chain, "dyn.α", n_teams, n_history, n_target),
        β = l01_reconstruct_trajectory(chain, "dyn.β", n_teams, n_history, n_target),
    )
    final_state = n_history + n_target

    predictor_draws = [
        L01_BUILDER.predictor_extract(chain, term, String(L01_BUILDER.predictor_name(term)))
        for term in model.covariates
    ]
    predictor_sources = [
        L01_BUILDER._cb_predictor_oos_source(
            term,
            feature_set,
            df,
            get(data, :player_lineup_ratings_map,
                Dict{Int,L01_FEATURES.PMLineupAggregate}()),
        )
        for term in model.covariates
    ]
    observation = L01_BUILDER._cb_extract_observation(model.observation, chain, n_teams)
    global_home_advantage = model.home_advantage isa L01_PG.GlobalHomeAdvantage

    results = Dict{Int,NamedTuple}()
    for row in eachrow(df)
        match_id = Int(row.match_id)
        home_index = get(team_map, row.home_team, 0)
        away_index = get(team_map, row.away_team, 0)

        attack_home = home_index > 0 ? dynamics.α[home_index, final_state, :] : zeros(n_samples)
        defence_home = home_index > 0 ? dynamics.β[home_index, final_state, :] : zeros(n_samples)
        attack_away = away_index > 0 ? dynamics.α[away_index, final_state, :] : zeros(n_samples)
        defence_away = away_index > 0 ? dynamics.β[away_index, final_state, :] : zeros(n_samples)
        γ_home = global_home_advantage ? home_advantage[:, 1] :
                 (home_index > 0 ? home_advantage[:, home_index] : zeros(n_samples))

        season_index = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        base = inter.μ_base[:, season_index] .+
               inter.δ_month[:, Dates.month(row.match_date)]

        predictor_home = zeros(n_samples)
        predictor_away = zeros(n_samples)
        for index in eachindex(model.covariates)
            effect = L01_BUILDER.predictor_oos(
                model.covariates[index],
                predictor_draws[index],
                predictor_sources[index],
                row,
            )
            predictor_home .+= effect.h
            predictor_away .+= effect.a
        end

        η_home = L01_BUILDER.apply_guard(
            model.guard,
            base .+ γ_home .+ attack_home .+ defence_away .+ predictor_home,
        )
        η_away = L01_BUILDER.apply_guard(
            model.guard,
            base .+ attack_away .+ defence_home .+ predictor_away,
        )
        λ_home = exp.(η_home)
        λ_away = exp.(η_away)

        results[match_id] = L01_BUILDER._cb_rates(
            model.observation,
            λ_home,
            λ_away,
            observation,
            home_index,
            away_index,
            Dates.month(row.match_date),
        )
    end
    return results
end

function L01_PG.extract_parameters(
    model::L01_BUILDER.NegBinCountModel{I,T,H,C,O,G},
    df::AbstractDataFrame,
    feature_set,
    chain::Chains,
) where {I,T<:L01_PG.MultiScaleGRW,H,C,O,G}
    error("Task 007 defines no MultiScaleGRW negative-binomial candidate")
end

# ==============================================================================
# 4. Canonical recipes
# ==============================================================================

l01_multiscale_dynamics() = L01_PG.MultiScaleGRW(
    z₀ = Normal(0.0, 1.0),
    zₛ = Normal(0.0, 1.0),
    zₖ = Normal(0.0, 1.0),
    α_σ₀ = Gamma(2.0, 0.06),
    α_σₛ = Gamma(2.0, 0.03),
    α_σₖ = Gamma(2.0, 0.015),
    β_σ₀ = Gamma(2.0, 0.10),
    β_σₛ = Gamma(2.0, 0.055),
    β_σₖ = Gamma(2.0, 0.012),
)

l01_production_wealth() = ProductionWealthCovariate(
    feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
    prior = truncated(Normal(0.10, 0.05), lower = 0.0),
)

l01_joint_observation() = JointGammaPoissonObservation(
    feature = MatchProxyXGFeature(k = 25.0, fallback = :none),
    shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

function l01_models()
    m00 = CountModelBuilder(:m00_baseline_grw) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(l01_multiscale_dynamics()) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    m05 = CountModelBuilder(:m05_production_wealth_grw) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(l01_multiscale_dynamics()) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(l01_production_wealth()) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    m05_joint = CountModelBuilder(:m05_joint_production_wealth_grw) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(l01_multiscale_dynamics()) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(l01_production_wealth()) |>
        BayesianFootball.add(l01_joint_observation()) |>
        BayesianFootball.build

    return Dict(
        L01_MODEL_NAMES[1] => m00,
        L01_MODEL_NAMES[2] => m05,
        L01_MODEL_NAMES[3] => m05_joint,
    )
end

l01_splitter(config::L01Config) = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = copy(config.target_seasons),
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    end_dynamics = nothing,
    stop_early = true,
)

l01_preflight_sampler(config::L01Config) = QueuedNUTSConfig(
    n_samples = config.preflight_samples,
    n_warmup = config.preflight_warmup,
    n_chains = config.chains,
    accept_rate = config.accept_rate,
    max_depth = config.max_depth,
    show_progress = false,
)

l01_production_sampler(config::L01Config) = QueuedNUTSConfig(
    n_samples = config.production_samples,
    n_warmup = config.production_warmup,
    n_chains = config.chains,
    accept_rate = config.accept_rate,
    max_depth = config.max_depth,
    show_progress = true,
)

l01_thresholds(config::L01Config) = ConvergenceThresholds(
    max_rhat = config.max_rhat,
    min_ess = config.min_ess,
    max_divergence_rate = config.max_divergence_rate,
    min_bfmi = config.min_bfmi,
    max_treedepth_rate = config.max_treedepth_rate,
)

function l01_fit_configs(config::L01Config, models, splitter, sampler)
    return Dict(name => FitConfig(
        name = name,
        model = models[name],
        splitter = splitter,
        sampler = sampler,
        execution = QueuedExecution(max_concurrent_tasks = config.max_concurrent_tasks),
        tags = copy(L01_TAGS),
        description = L01_DESCRIPTIONS[name],
        save_dir = joinpath(config.save_root, name),
    ) for name in L01_MODEL_NAMES)
end

l01_load_data() = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)

function l01_database(config::L01Config)
    db = PostgresStorage(config.experiment)
    ensure_schema!(db)
    return db
end

function l01_register!(db, models, splitter, sampler, fit_configs)
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for name in L01_MODEL_NAMES
        model_ids[name] = save_model(
            db,
            name,
            models[name];
            description = L01_DESCRIPTIONS[name],
            tags = L01_TAGS,
        )
        fit_hashes[name] = save_config(
            db,
            name * "_fit",
            fit_configs[name];
            description = L01_DESCRIPTIONS[name] * " Canonical Task 007 recipe.",
            tags = L01_TAGS,
        )
    end
    splitter_id = save_splitter(
        db,
        "scottish_lower_multiscale_grw_40fold",
        splitter;
        description = "Pooled tournaments 56/57, two history seasons, match-biweek walk-forward over 24/25 and 25/26.",
        tags = L01_TAGS,
    )
    sampler_id = save_sampler(
        db,
        "queued_nuts_4x800_multiscale_grw",
        sampler;
        description = "ReverseDiff queued NUTS: four chains, 800 warmup and 800 retained, target acceptance 0.90.",
        tags = L01_TAGS,
    )
    return (; model_ids, fit_hashes, splitter_id, sampler_id)
end

# ==============================================================================
# 5. Preflight and production helpers
# ==============================================================================

function l01_scored_inputs(ds, splitter, model; limit::Union{Nothing,Int} = nothing)
    boundaries = Data.create_id_boundaries(ds, splitter)
    selected = limit === nothing ? boundaries : boundaries[1:min(limit, length(boundaries))]
    feature_sets = L01_FEATURES.create_features(selected, ds, model, splitter)
    oos = [Data.get_next_matches(ds, feature_sets[index], splitter)
           for index in eachindex(feature_sets)]
    return (; boundaries, selected, feature_sets, oos)
end

l01_relative_error(left, right) =
    norm(left - right) / max(norm(left), norm(right), 1.0)

"""
Compile and validate one fold's ReverseDiff tape.

The audit checks finite log density/gradient, compiled-vs-fresh ReverseDiff,
ReverseDiff-vs-ForwardDiff, replay at perturbed points, and measured allocations
on a warmed `gradient!` call.
"""
function l01_gradient_audit(model, feature_set, config::L01Config;
                            seed::Int = 20260910)
    turing_model = L01_PG.build_turing_model(model, first(feature_set))
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = values -> LogDensityProblems.logdensity(density, values)

    log_density = objective(θ)
    isfinite(log_density) || error("MultiScaleGRW preflight produced non-finite log density")

    raw_tape = ReverseDiff.GradientTape(objective, θ)
    tape = ReverseDiff.compile(raw_tape)
    gradient = similar(θ)
    ReverseDiff.gradient!(gradient, tape, θ)
    all(isfinite, gradient) || error("MultiScaleGRW compiled gradient contains non-finite values")

    fresh = ReverseDiff.gradient(objective, θ)
    forward = ForwardDiff.gradient(objective, θ)
    compiled_fresh_error = l01_relative_error(gradient, fresh)
    compiled_forward_error = l01_relative_error(gradient, forward)
    compiled_fresh_error <= 1.0e-8 || error(
        "compiled/fresh ReverseDiff relative error $compiled_fresh_error exceeds 1e-8")
    compiled_forward_error <= 1.0e-6 || error(
        "ReverseDiff/ForwardDiff relative error $compiled_forward_error exceeds 1e-6")

    worst_perturbed_error = 0.0
    coordinates = collect(eachindex(θ))
    for delta in (0.001, -0.002, 0.003)
        perturbed = θ .+ delta .* sin.(coordinates)
        compiled = similar(perturbed)
        ReverseDiff.gradient!(compiled, tape, perturbed)
        error_value = l01_relative_error(ReverseDiff.gradient(objective, perturbed), compiled)
        worst_perturbed_error = max(worst_perturbed_error, error_value)
    end
    worst_perturbed_error <= 1.0e-8 || error(
        "compiled tape changes under perturbation: relative error $worst_perturbed_error")

    for _ in 1:20
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    allocated_bytes = @allocated ReverseDiff.gradient!(gradient, tape, θ)

    best_ns = typemax(UInt64)
    for _ in 1:config.gradient_replays
        started = time_ns()
        ReverseDiff.gradient!(gradient, tape, θ)
        best_ns = min(best_ns, time_ns() - started)
    end

    return (;
        n_parameters = length(θ),
        tape_instructions = length(raw_tape.tape),
        gradient_ms = Float64(best_ns) / 1.0e6,
        allocated_bytes,
        compiled_fresh_error,
        compiled_forward_error,
        worst_perturbed_error,
        log_density,
    )
end

function l01_existing_fit(db, name::String)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT run_id
            FROM runs
            WHERE experiment_name = \$1 AND name = \$2 AND status = 'completed'
            ORDER BY id DESC
            LIMIT 1;
        """, (db.experiment_name, name))
        try
            rows = DataFrame(result)
            nrow(rows) == 0 && return nothing
            return load_fit(db, UUID(string(rows.run_id[1])))
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

function l01_same_recipe(left::FitConfig, right::FitConfig)
    return left.name == right.name &&
           string(left.model) == string(right.model) &&
           string(left.splitter) == string(right.splitter) &&
           string(left.sampler) == string(right.sampler) &&
           string(left.execution) == string(right.execution) &&
           left.tags == right.tags &&
           left.description == right.description
end

function l01_fit_or_load(db, fit_config::FitConfig, ds, config::L01Config)
    existing = l01_existing_fit(db, fit_config.name)
    if existing !== nothing
        l01_same_recipe(existing.config, fit_config) || error(
            "completed run $(fit_config.name) exists but its persisted recipe differs; " *
            "rename the candidate or resolve config truth before sampling")
        return (; fit = existing, reused = true)
    end

    checkpoint_dir = joinpath(fit_config.save_dir, "checkpoints")
    fit = fit_model(
        fit_config,
        ds;
        thresholds = l01_thresholds(config),
        checkpoint_dir,
        cleanup_checkpoints = false,
        quiet = false,
    )
    return (; fit, reused = false)
end

function l01_extend_sampling(fit, fit_config::FitConfig, ds, config::L01Config;
                             n_samples::Int, n_warmup::Int)
    sampler = QueuedNUTSConfig(
        n_samples = n_samples,
        n_warmup = n_warmup,
        n_chains = config.chains,
        accept_rate = config.accept_rate,
        max_depth = config.max_depth,
        show_progress = true,
    )
    retry_config = FitConfig(
        name = fit_config.name,
        model = fit_config.model,
        splitter = fit_config.splitter,
        sampler = sampler,
        execution = fit_config.execution,
        tags = fit_config.tags,
        description = fit_config.description,
        save_dir = fit_config.save_dir,
    )
    checkpoint_dir = joinpath(fit_config.save_dir, "retry_checkpoints")
    retry = fit_model(
        retry_config,
        ds;
        thresholds = l01_thresholds(config),
        checkpoint_dir,
        cleanup_checkpoints = false,
        quiet = false,
    )

    length(fit) == length(retry) || error("cannot extend chains with different fold counts")
    full_folds = FoldFit[
        FoldFit(
            left.fold,
            Chains(
                cat(parent(left.chain.value), parent(right.chain.value); dims = 1),
                names(left.chain),
                Dict(
                    :parameters => names(left.chain, :parameters),
                    :internals => names(left.chain, :internals),
                );
                start = 1,
            ),
            left.meta,
        )
        for (left, right) in zip(fit.folds, retry.folds)
    ]
    diagnostics = audit_convergence(
        full_folds;
        thresholds = l01_thresholds(config),
        max_depth = config.max_depth,
    )
    folds = FoldFit[
        FoldFit(
            full.fold,
            Chains(
                parent(full.chain.value)[1:4:end, :, :],
                names(full.chain),
                Dict(
                    :parameters => names(full.chain, :parameters),
                    :internals => names(full.chain, :internals),
                );
                start = 1,
            ),
            full.meta,
        )
        for full in full_folds
    ]
    inputs = l01_scored_inputs(ds, fit_config.splitter, fit_config.model)
    latents, latent_note = extract_run_latents(
        fit_config.model, folds, inputs.oos, inputs.feature_sets)
    isempty(latent_note) || error("extended-chain latent extraction failed: $latent_note")
    elapsed = fit.metadata.elapsed_seconds + retry.metadata.elapsed_seconds
    metadata = FitMetadata(
        retry.metadata.timestamp,
        elapsed,
        retry.metadata.julia_version,
        retry.metadata.n_threads,
        retry.metadata.git_commit,
    )
    # PostgreSQL's text-protocol bytea parameter cannot carry a multi-gigabyte exact
    # chain artifact. The diagnostic audit above used every retained draw; the
    # persisted chain panel keeps every fourth draw from the combined runs while
    # latents are reconstructed from that same persisted panel.
    tags = diagnostics.passed ? filter(!=("not_converged"), fit_config.tags) :
           unique(vcat(fit_config.tags, ["not_converged"]))
    combined_config = FitConfig(
        name = fit_config.name,
        model = fit_config.model,
        splitter = fit_config.splitter,
        sampler = fit_config.sampler,
        execution = fit_config.execution,
        tags = tags,
        description = fit_config.description,
        save_dir = fit_config.save_dir,
    )
    return Fit(combined_config, folds, latents, diagnostics, metadata,
               joinpath(combined_config.save_dir, combined_config.name * ".jls"))
end

function l01_assert_coverage(name::String, fit, config::L01Config)
    length(fit) == config.expected_folds || error(
        "$name produced $(length(fit)) folds; expected $(config.expected_folds)")
    fit.latents isa CountLatents || error(
        "$name returned $(typeof(fit.latents)); expected CountLatents")
    n_matches(fit.latents) == config.expected_oos_matches || error(
        "$name produced $(n_matches(fit.latents)) OOS matches; " *
        "expected $(config.expected_oos_matches)")
    allunique(fit.latents.match_ids) || error("$name OOS match IDs are not unique")
    return nothing
end

function l01_assert_promotion(name::String, diagnostics)
    diagnostics.passed || error(
        "$name failed strict promotion: $(join(diagnostics.failures, "; "))")
    diagnostics.max_rhat <= 1.01 || error("$name max R-hat exceeds 1.01")
    diagnostics.min_ess_bulk >= 400.0 || error("$name bulk ESS is below 400")
    diagnostics.min_ess_tail >= 400.0 || error("$name tail ESS is below 400")
    diagnostics.divergence_rate < 0.001 || error(
        "$name divergence rate $(diagnostics.divergence_rate) is not below 0.001")
    return nothing
end

function l01_evaluate(fit, ds)
    scores = evaluate_predictions(fit, ds; threaded = true)
    crps = BayesianFootball.Evaluation.compute_metric(
        BayesianFootball.Evaluation.CRPS(), fit, ds; threaded = true)
    values = (
        logloss = scores.model.logloss,
        brier = scores.model.brier,
        rps = scores.model.rps,
        crps = crps.all.mean,
        market_logloss = scores.market.logloss,
        market_brier = scores.market.brier,
        market_rps = scores.market.rps,
    )
    all(isfinite, Tuple(values)) || error("evaluation produced non-finite values: $values")
    return values
end

function l01_persist_scores!(db, run_id::UUID, scores)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            UPDATE fold_results
            SET logloss = \$2, brier = \$3, rps = \$4
            WHERE run_id = \$1::uuid;
        """, (string(run_id), scores.logloss, scores.brier, scores.rps))
        close(result)
    finally
        close(conn)
    end
    return nothing
end

function l01_load_baseline(experiment::String, name::String)
    baseline_db = PostgresStorage(experiment)
    fit = load_fit(baseline_db, name)
    fit.metadata.git_commit == "synthetic-no-mcmc" && error(
        "baseline $experiment/$name is synthetic; refusing a scientific comparison")
    return fit
end

function l01_compare(name::String, fit, baseline_name::String, baseline_fit, ds)
    Set(fit.latents.match_ids) == Set(baseline_fit.latents.match_ids) || error(
        "$name and $baseline_name do not cover the identical OOS fixture set")
    grw = l01_evaluate(fit, ds)
    baseline = l01_evaluate(baseline_fit, ds)
    speedup = baseline_fit.metadata.elapsed_seconds / fit.metadata.elapsed_seconds
    return (;
        name,
        baseline_name,
        grw,
        baseline,
        delta_logloss = grw.logloss - baseline.logloss,
        delta_brier = grw.brier - baseline.brier,
        delta_rps = grw.rps - baseline.rps,
        delta_crps = grw.crps - baseline.crps,
        grw_seconds = fit.metadata.elapsed_seconds,
        baseline_seconds = baseline_fit.metadata.elapsed_seconds,
        speedup,
    )
end

function l01_phase1_passed(db)
    for name in L01_PHASE1_NAMES
        fit = l01_existing_fit(db, name)
        fit === nothing && return false
        fit.diagnostics.passed || return false
        fit.diagnostics.max_rhat <= 1.01 || return false
        fit.diagnostics.min_ess_bulk >= 400.0 || return false
        fit.diagnostics.min_ess_tail >= 400.0 || return false
        fit.diagnostics.divergence_rate < 0.001 || return false
    end
    return true
end

function l01_save_stage_result(config::L01Config, stage::String, value)
    mkpath(config.save_root)
    path = joinpath(config.save_root, stage * ".jls")
    temporary = path * ".tmp." * string(rand(UInt64), base = 16)
    Serialization.serialize(temporary, value)
    mv(temporary, path; force = true)
    return path
end

# ==============================================================================
# 6. Markdown report
# ==============================================================================

l01_number(value; digits = 5) = @sprintf("%.*f", digits, value)
l01_delta(value; digits = 5) = @sprintf("%+.*f", digits, value)

function l01_comparison_table(comparisons)
    isempty(comparisons) && return "_No completed score comparisons yet._\n"
    lines = [
        "| GRW candidate | TimeDecay control | LogLoss | Δ LogLoss | Brier | Δ Brier | RPS | Δ RPS | CRPS | Δ CRPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons
        push!(lines,
            "| `$(row.name)` | `$(row.baseline_name)` | " *
            "$(l01_number(row.grw.logloss)) | $(l01_delta(row.delta_logloss)) | " *
            "$(l01_number(row.grw.brier)) | $(l01_delta(row.delta_brier)) | " *
            "$(l01_number(row.grw.rps)) | $(l01_delta(row.delta_rps)) | " *
            "$(l01_number(row.grw.crps)) | $(l01_delta(row.delta_crps)) |")
    end
    return join(lines, "\n") * "\n"
end

function l01_speed_table(comparisons)
    isempty(comparisons) && return "_No completed runtime comparisons yet._\n"
    lines = [
        "| GRW candidate | GRW wall min | TimeDecay wall min | TimeDecay / GRW |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons
        push!(lines,
            "| `$(row.name)` | $(l01_number(row.grw_seconds / 60; digits = 1)) | " *
            "$(l01_number(row.baseline_seconds / 60; digits = 1)) | " *
            "$(l01_number(row.speedup; digits = 2))× |")
    end
    return join(lines, "\n") * "\n"
end

function l01_run_table(run_rows)
    isempty(run_rows) && return "_No production runs persisted yet._\n"
    lines = [
        "| Model | Folds | OOS | R̂ max | ESS min | Divergences | Wall min | Run UUID |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in run_rows
        push!(lines,
            "| `$(row.name)` | $(row.folds) | $(row.oos) | " *
            "$(l01_number(row.rhat; digits = 4)) | $(l01_number(row.ess; digits = 0)) | " *
            "$(row.divergences) | $(l01_number(row.seconds / 60; digits = 1)) | `$(row.run_id)` |")
    end
    return join(lines, "\n") * "\n"
end

function l01_preflight_table(rows)
    isempty(rows) && return "_Preflight not run yet._\n"
    lines = [
        "| Model | Fold | Parameters | Tape instructions | Gradient ms | Alloc bytes | R̂ max | ESS min | Divergences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows
        push!(lines,
            "| `$(row.name)` | $(row.fold) | $(row.n_parameters) | " *
            "$(row.tape_instructions) | $(l01_number(row.gradient_ms; digits = 3)) | " *
            "$(row.allocated_bytes) | $(l01_number(row.rhat; digits = 4)) | " *
            "$(l01_number(row.ess; digits = 0)) | $(row.divergences) |")
    end
    return join(lines, "\n") * "\n"
end

function l01_write_report!(config::L01Config;
                           preflight_rows = NamedTuple[],
                           run_rows = NamedTuple[],
                           comparisons = NamedTuple[],
                           phase2_status::String = "not started")
    text = """# Task 007 — MultiScaleGRW prototype

This directory prototypes a non-centred two-speed Gaussian random walk for Scottish
Lower team attack and defence. History advances by one macro innovation per season;
the target season advances by one micro innovation per observed match-biweek. Each
state is zero-centred over teams, and held-out fixtures use the final state visible at
the fold cutoff.

## Contract

- Data: pooled tournaments 56/57, target seasons 24/25 and 25/26.
- Split: 40 match-biweek walk-forward folds, 710 held-out fixtures.
- AD: compiled ReverseDiff tape; replay allocations are measured. The installed
  TimeDecay control allocates 43,888 bytes in the identical harness, so literal
  zero allocation remains an unmet performance target rather than a claimed pass.
- Sampler: four chains, 800 warmup + 800 retained, target acceptance 0.90.
- Promotion: R̂ ≤ 1.01, bulk/tail ESS ≥ 400, divergence rate < 0.1%, BFMI ≥ 0.30.
- Control: otherwise-matched `TimeDecayDynamics(days_half_life = 180.0)` fits loaded
  from the experiment database and required to cover the identical OOS fixture IDs.

## Two-fold preflight

$(l01_preflight_table(preflight_rows))

## Persisted production runs

$(l01_run_table(run_rows))

## Proper scores versus TimeDecayDynamics

Negative deltas favour MultiScaleGRW.

$(l01_comparison_table(comparisons))

## Runtime comparison

The ratio is `TimeDecay wall time / MultiScaleGRW wall time`; values above one mean
MultiScaleGRW completed faster despite its larger latent state.

$(l01_speed_table(comparisons))

## Phase 2

$phase2_status

## Reproduction

```bash
# On mcmc-beast, from /root/BF_multiscale_grw
L01_STAGE=preflight /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
L01_STAGE=phase1   /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
L01_STAGE=phase2   /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
```

PostgreSQL namespace: `$(config.experiment)`.
"""
    temporary = config.report_path * ".tmp." * string(rand(UInt64), base = 16)
    write(temporary, text)
    mv(temporary, config.report_path; force = true)
    return config.report_path
end

end # module
