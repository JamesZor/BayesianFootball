# ==============================================================================
# Task 015 loader — MultiScaleGRW with Market Supremacy and Market Smile anchoring
# ==============================================================================
#
# Definitions only. `r01_smoke.jl`, `r02_production_grid.jl`, `r04_evaluate.jl` and
# `r05_slate_repricing.jl` execute.
#
# THE LADDER. One foundation — Task 013's `m05_wealth_grw`, verbatim:
#
#   GlobalInterception + MultiScaleGRW + GlobalHomeAdvantage
#   + ProductionWealthCovariate(SupremacyRole) + JointGammaPoissonObservation
#
# and two market likelihood pillars bolted onto it:
#
#   m05_joint_grw_baseline               no pillar            = Task 013 run b0961bc4 (pinned)
#   m05_joint_grw_supremacy_w040         C1 supremacy @ 0.40
#   m05_joint_grw_smile_supremacy_w020   C1 + C2 smile @ 0.20 / 0.20   (light)
#   m05_joint_grw_smile_supremacy_w040   C1 + C2 smile @ 0.40 / 0.40   (moderate, Ireland default)
#   m05_joint_grw_smile_supremacy_w070   C1 + C2 smile @ 0.70 / 0.70   (strong)
#
# THE PILLARS (Ireland's `goals_smile_league.jl`, re-derived for the joint observation)
#
#   C1  η_h − η_a            ~ Normal(log λ̂_h − log λ̂_a, σ_sup)                  per training match
#   C2  log κ + log(μ_h+μ_a) + log φ_K ~ Normal(log Λ̂_K, σ_smile),  K = 0…4      per training match
#
#   σ_sup, σ_smile ~ truncated(Normal(0.15, 0.10), lower = 0.02)     SAMPLED, never fixed
#   log φ          ~ Normal(0, 0.5)^5                                 a PRICING object only
#
# Each pillar's log-likelihood is tempered by its weight and enters by `@addlogprob!`.
# λ̂ is the market's double-Poisson inversion of the closing book
# (`DoublePoissonMarketFeature`), Λ̂_K the Poisson rate implied by the de-vigged
# closing Under K.5 (`MarketSmileFeature(Kmax = 4)`).
#
# WHY κ IS IN C2 AND NOT IN C1. The joint observation prices goals at λ = κ·μ. Supremacy
# is a log-ratio, so κ cancels. A total is a level, so it does not: the market's goals
# intensity must be compared with the model's GOALS intensity κ(μ_h + μ_a), not with the
# proxy-xG latent μ — otherwise log φ would silently absorb log κ and stop meaning
# "the smile".
#
# WHY A WRAPPER AND NOT A BUILDER COMPONENT. The composable engine types its config as
# `ComposableCountModel` and its observation hides `obs.log_κ` inside `_observe`, which
# returns only a scalar. Neither can be extended from `current_development/`. So
# `MarketAnchoredCountModel` carries a built `PoissonCountModel` and its own engine that
# calls the builder's OWN submodels in the builder's OWN declaration order — the only
# new code on the likelihood path is the κ-returning copy of the shared-κ observation
# and the two pillars. Gate G0 (`gms_parity_check`) proves that claim the strong way:
# with both pillars off, the log density is bit-identical to the Task 013 `m05` builder
# model at a prior draw and three displaced points.
#
# FILTRATION. Both pillars read only the TRAINING fixtures of a fold — played matches
# whose closing prices existed before the fold's first held-out kickoff. A held-out
# fixture's price never reaches the likelihood or the extractor: OOS pricing is λ from the
# walk's final state, φ from the chain.
#
# PERSISTENCE CAVEAT (ticket T010). `save_fit(::PostgresStorage)` stores `CountLatents`
# only (`src/training/inference/db_storage.jl:618`) and `load_fit` rebuilds latents from
# `match_latents`, so a smile run cannot round-trip its φ through PostgreSQL. A smile fit is
# persisted with its latent panel DETACHED — run, config, fold diagnostics and the full
# chain artefact — and `gms_rebuild_latents` re-extracts `SmileLatents` from the persisted
# chains. That is deterministic arithmetic on identical draws, and `gms_save_and_verify`
# requires the rebuilt container to equal the in-memory one field for field.
# ==============================================================================

if !isdefined(@__MODULE__, :GPHConfig)
    include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l01_loader.jl"))
end

import SpecialFunctions
import Turing

const GMS_PG = BayesianFootball.Models.PreGame
const GMS_CB = BayesianFootball.Models.PreGame.Builder
const GMS_FEATURES = BayesianFootball.Features
const GMS_PRED = BayesianFootball.Predictions
const GMS_TI = BayesianFootball.TypesInterfaces
const GMS_LATENTS = parentmodule(BayesianFootball.latent_family)
const GMS_INF = BayesianFootball.Training.Inference

const GMS_HALF_LOG_2PI = 0.5 * log(2π)

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

"""
    GMSConfig

Every number that governs the study. Runners read `config.samples`, never `ENV`.

`extension_seasons` is the PRODUCTION split. The 2026-09-12 card is priced from Fold 43, so
r02 fits 43 folds once rather than fitting 40 and extending; r04 restricts every arm back to
the 710-fixture 24/25 + 25/26 panel before scoring, and r02 asserts that the first 40 folds
of the 43-fold split are the 40-fold split fixture for fixture.
"""
Base.@kwdef struct GMSConfig
    experiment::String = "scottish_lower_grw_market_smile"
    smoke_experiment::String = "smoke_grw_smile"
    save_root::String = joinpath(@__DIR__, "results")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    extension_seasons::Vector{String} = ["24/25", "25/26", "26/27"]
    expected_folds::Int = 40
    expected_extended_folds::Int = 43
    expected_oos::Int = 710

    smoke_folds::Int = 2
    smoke_samples::Int = 500
    smoke_warmup::Int = 500
    smoke_chains::Int = 4

    samples::Int = 1000
    warmup::Int = 500
    chains::Int = 4
    accept_rate::Float64 = 0.80
    max_depth::Int = 10
    max_concurrent_tasks::Int = 16

    persist_stride::Int = 2
    gradient_replays::Int = 200

    max_rhat::Float64 = 1.05
    strict_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end

"The Task 013 helpers take a `GPHConfig`; this is the same numbers in that shape."
gms_gph_config(c::GMSConfig) = GPHConfig(
    samples = c.samples, warmup = c.warmup, chains = c.chains,
    accept_rate = c.accept_rate, max_depth = c.max_depth,
    max_concurrent_tasks = c.max_concurrent_tasks,
    persist_stride = c.persist_stride, strict_rhat = c.strict_rhat,
    max_rhat = c.max_rhat, min_ess = c.min_ess,
    max_divergence_rate = c.max_divergence_rate, min_bfmi = c.min_bfmi,
    max_treedepth_rate = c.max_treedepth_rate,
    smoke_samples = c.smoke_samples, smoke_warmup = c.smoke_warmup,
    smoke_chains = c.smoke_chains)

const GMS_MODEL_NAMES = [
    "m05_joint_grw_baseline",
    "m05_joint_grw_supremacy_w040",
    "m05_joint_grw_smile_supremacy_w020",
    "m05_joint_grw_smile_supremacy_w040",
    "m05_joint_grw_smile_supremacy_w070",
]

"The work package's three-rung smoke ladder, at the moderate (Ireland) weight."
const GMS_SMOKE_MODEL_NAMES = GMS_MODEL_NAMES[[1, 2, 4]]

"Sampled by r02. The baseline is NOT: it is Task 013's persisted run of the identical recipe."
const GMS_GRID_MODEL_NAMES = GMS_MODEL_NAMES[2:5]

const GMS_DESCRIPTIONS = Dict(
    "m05_joint_grw_baseline" =>
        "Joint Gamma-Poisson MultiScaleGRW with production wealth; no market pillar (Task 013 m05).",
    "m05_joint_grw_supremacy_w040" =>
        "m05 joint GRW + market supremacy pillar (weight 0.40, sigma sampled).",
    "m05_joint_grw_smile_supremacy_w020" =>
        "m05 joint GRW + market supremacy and local-intensity smile pillars, light weights 0.20/0.20.",
    "m05_joint_grw_smile_supremacy_w040" =>
        "m05 joint GRW + market supremacy and local-intensity smile pillars, moderate weights 0.40/0.40.",
    "m05_joint_grw_smile_supremacy_w070" =>
        "m05 joint GRW + market supremacy and local-intensity smile pillars, strong weights 0.70/0.70.",
)

const GMS_TAGS = [
    "scottish-lower", "24/25", "25/26", "26/27", "multiscale-grw", "joint-gamma-poisson",
    "market-supremacy", "market-smile", "todo015", "reversediff",
]

"""
The baseline rung, pinned by UUID. Task 013's `m05_wealth_grw` is the SAME recipe this
loader's `m05_joint_grw_baseline` builds — `r02` asserts `string(model)` and
`string(sampler)` equality before using it — and it was extended in place to 43 folds.
"""
# Transitions the baseline control's six-part audit counted over its ORIGINAL 40 folds, copied
# from Task 013's committed `current_development/grw_player_hybrid/results/r02_production_runs.csv`
# (row `m05_wealth_grw`, run b0961bc4): 160,000 = 40 folds × 4 chains × 1,000 retained draws.
#
# Pinned here because the run's own diagnostics no longer carry it: `extend_fit` re-audits EVERY
# fold from the persisted, thinned chains when it appends folds
# (`src/training/inference/extension.jl:365`), so fold 1 of the 43-fold artefact counts
# 4 × 500 = 2,000. r02 checks both numbers.
const GMS_BASELINE_SAMPLED_TRANSITIONS_40 = 160_000


const GMS_BASELINE_CONTROL = GPHControl(
    "m05_joint_grw_baseline", "scottish_lower_grw_player_hybrid",
    UUID("b0961bc4-c40c-4dbe-9c05-57df7ae0839e"), "m05_wealth_grw",
    "Task 013 joint Gamma-Poisson: GRW + production wealth, 4×(500+1000), 43 folds")

# ==============================================================================
# 2. The market pillars
# ==============================================================================

abstract type AbstractMarketPillar end

"A pillar slot left empty. Dispatches every hook away: no site, no feature, no tape node."
struct NoMarketPillar <: AbstractMarketPillar end

"""
    MarketSupremacyPillar(; weight, σ_prior, feature, λ_lo, λ_hi)

C1: the model's log-rate difference is read by `Normal(log λ̂_h − log λ̂_a, σ_sup)` on every
training match whose closing book inverted to two plausible rates (`λ_lo < λ̂ < λ_hi`).
The plausibility window is `goals_smile_league.jl`'s: a thin close can invert to a
degenerate λ, and a supremacy of ±9 would dominate the pillar.
"""
Base.@kwdef struct MarketSupremacyPillar{D<:ContinuousUnivariateDistribution} <: AbstractMarketPillar
    weight::Float64 = 0.40
    σ_prior::D = truncated(Normal(0.15, 0.10), lower = 0.02)
    feature::GMS_FEATURES.DoublePoissonMarketFeature = GMS_FEATURES.DoublePoissonMarketFeature()
    λ_lo::Float64 = 0.02
    λ_hi::Float64 = 20.0
end

"""
    MarketSmilePillar(; weight, σ_prior, shape_sd, feature)

C2: the model's per-strike total intensity `log κ + log(μ_h + μ_a) + log φ_K` is read by
`Normal(log Λ̂_K, σ_smile)` wherever the closing Under K.5 exists and inverted cleanly.
`φ` is global, and it is a pricing object: it never touches the goals or proxy arms.
"""
Base.@kwdef struct MarketSmilePillar{D<:ContinuousUnivariateDistribution} <: AbstractMarketPillar
    weight::Float64 = 0.40
    σ_prior::D = truncated(Normal(0.15, 0.10), lower = 0.02)
    shape_sd::Float64 = 0.50
    feature::GMS_FEATURES.MarketSmileFeature = GMS_FEATURES.MarketSmileFeature(Kmax = 4)
end

gms_n_strikes(p::MarketSmilePillar) = p.feature.Kmax + 1

"""
    MarketAnchoredCountModel(base, supremacy, smile)

A built Task 013 `PoissonCountModel` plus two pillar slots.

The supertype is `AbstractPoissonModel` because every price this model makes comes from
the double-Poisson grid at `λ = κμ` — the smile variant adds a per-strike O/U curve
beside that grid, routed by `latent_family` to `SmileLatents`, never instead of it.
"""
struct MarketAnchoredCountModel{
    B<:GMS_CB.PoissonCountModel,
    S<:Union{NoMarketPillar,MarketSupremacyPillar},
    K<:Union{NoMarketPillar,MarketSmilePillar},
} <: GMS_TI.AbstractPoissonModel
    base::B
    supremacy::S
    smile::K

    function MarketAnchoredCountModel(base::B, supremacy::S, smile::K) where {B,S,K}
        base.observation isa GMS_CB.SharedKappaJoint || error(
            "MarketAnchoredCountModel is written against the shared-κ JointGammaPoissonObservation; " *
            "got $(typeof(base.observation)). The smile pillar reads obs.log_κ from that block.")
        return new{B,S,K}(base, supremacy, smile)
    end
end

const GMSSmileModel = MarketAnchoredCountModel{B,S,<:MarketSmilePillar} where {B,S}
const GMSGridModel = MarketAnchoredCountModel{B,S,NoMarketPillar} where {B,S}

# ==============================================================================
# 3. The ladder
# ==============================================================================

"Task 013's `m05_wealth_grw`, component for component and in the same add order."
gms_base_model(kind::Symbol = :m05_joint_grw) = CountModelBuilder(kind) |>
    add(GlobalInterception()) |>
    add(gph_dynamics()) |>
    add(GlobalHomeAdvantage()) |>
    add(gph_production_wealth()) |>
    add(gph_joint_observation()) |>
    build

gms_supremacy(w::Real) = MarketSupremacyPillar(weight = Float64(w))
gms_smile(w::Real) = MarketSmilePillar(weight = Float64(w))

"""
    gms_models() -> Vector{Tuple{String,Any}}

All five rungs in ladder order. The baseline is the bare builder model — not a wrapper
with two empty slots — because it is the object Task 013 sampled and the one its persisted
run's `config.model` must equal.
"""
function gms_models()
    base = gms_base_model()
    return Tuple{String,Any}[
        ("m05_joint_grw_baseline", base),
        ("m05_joint_grw_supremacy_w040",
         MarketAnchoredCountModel(base, gms_supremacy(0.40), NoMarketPillar())),
        ("m05_joint_grw_smile_supremacy_w020",
         MarketAnchoredCountModel(base, gms_supremacy(0.20), gms_smile(0.20))),
        ("m05_joint_grw_smile_supremacy_w040",
         MarketAnchoredCountModel(base, gms_supremacy(0.40), gms_smile(0.40))),
        ("m05_joint_grw_smile_supremacy_w070",
         MarketAnchoredCountModel(base, gms_supremacy(0.70), gms_smile(0.70))),
    ]
end

gms_select(models, names) = Tuple{String,Any}[(n, m) for (n, m) in models if n in names]

"The same base with both slots empty — G0's bit-identity witness. Never sampled."
gms_null_anchor(base) = MarketAnchoredCountModel(base, NoMarketPillar(), NoMarketPillar())

# ==============================================================================
# 4. Features and design (all conditional logic lives here)
# ==============================================================================

gms_pillar_features(::NoMarketPillar) = GMS_FEATURES.AbstractFeatureConfig[]
gms_pillar_features(p::MarketSupremacyPillar) = GMS_FEATURES.AbstractFeatureConfig[p.feature]
gms_pillar_features(p::MarketSmilePillar) = GMS_FEATURES.AbstractFeatureConfig[p.feature]

function GMS_FEATURES.required_features(m::MarketAnchoredCountModel)
    out = GMS_FEATURES.required_features(m.base)
    append!(out, gms_pillar_features(m.supremacy))
    append!(out, gms_pillar_features(m.smile))
    return out
end

"C1 data: the market log-supremacy, and weight × mask × match weight folded into one vector."
struct GMSSupremacyDesign
    m_sup::Vector{Float64}
    weights::Vector{Float64}
    n_observed::Int
end

"C2 data: `n_matches × n_strikes` market log-intensities and their folded weights."
struct GMSSmileDesign
    logΛ::Matrix{Float64}
    weights::Matrix{Float64}
    n_strikes::Int
    n_observed::Int
end

struct GMSMarketDesign{S,K}
    supremacy::S
    smile::K
end

gms_pillar_design(::NoMarketPillar, fs, n::Int, wts::Vector{Float64}) = nothing

function gms_pillar_design(p::MarketSupremacyPillar, fs, n::Int, wts::Vector{Float64})
    d = fs.data
    for key in (:flat_market_λ_home, :flat_market_λ_away)
        haskey(d, key) || error(
            "MarketSupremacyPillar needs $key; it is emitted by DoublePoissonMarketFeature, " *
            "which the pillar declares through required_features")
    end
    λ_h = d[:flat_market_λ_home]
    λ_a = d[:flat_market_λ_away]
    length(λ_h) == n && length(λ_a) == n || error(
        "market rate columns have lengths $(length(λ_h))/$(length(λ_a)); expected $n")

    plausible(x) = !ismissing(x) && isfinite(Float64(x)) && p.λ_lo < Float64(x) < p.λ_hi
    mask = Float64[plausible(λ_h[i]) && plausible(λ_a[i]) for i in 1:n]
    m_sup = Float64[mask[i] == 1.0 ? log(Float64(λ_h[i])) - log(Float64(λ_a[i])) : 0.0
                    for i in 1:n]
    all(isfinite, m_sup) || error("non-finite market supremacy on a masked-in match")
    return GMSSupremacyDesign(m_sup, p.weight .* wts .* mask, Int(sum(mask)))
end

function gms_pillar_design(p::MarketSmilePillar, fs, n::Int, wts::Vector{Float64})
    d = fs.data
    for key in (:flat_smile_logΛ, :flat_smile_mask, :smile_Kmax)
        haskey(d, key) || error(
            "MarketSmilePillar needs $key; it is emitted by MarketSmileFeature")
    end
    Int(d[:smile_Kmax]) == p.feature.Kmax || error(
        "FeatureSet smile_Kmax $(d[:smile_Kmax]) ≠ pillar Kmax $(p.feature.Kmax)")
    logΛ = Matrix{Float64}(d[:flat_smile_logΛ])
    mask = Matrix{Float64}(d[:flat_smile_mask])
    nK = gms_n_strikes(p)
    size(logΛ) == (n, nK) && size(mask) == (n, nK) || error(
        "smile matrices are $(size(logΛ)) / $(size(mask)); expected ($n, $nK)")
    all(x -> x == 0.0 || x == 1.0, mask) || error("smile mask must be exactly 0/1")
    all(isfinite, logΛ) || error("non-finite smile log-intensity")
    return GMSSmileDesign(logΛ, p.weight .* mask .* reshape(wts, n, 1), nK, Int(sum(mask)))
end

function gms_market_design(m::MarketAnchoredCountModel, fs, n::Int, wts::Vector{Float64})
    return GMSMarketDesign(gms_pillar_design(m.supremacy, fs, n, wts),
                           gms_pillar_design(m.smile, fs, n, wts))
end

# ==============================================================================
# 5. The engine
# ==============================================================================

"""
The shared-κ two-arm observation, returning `log κ` beside the log-likelihood.

Byte-for-byte `_observe(::SharedKappaJoint, …)` (`engine.jl:273`) — same submodel, same
site names (`obs.ν`, `obs.log_κ`), same arithmetic in the same order — except for the
return value. G0 is what makes "byte-for-byte" a measured statement.
"""
Turing.@model function _gms_joint_observe(o::GMS_CB.SharedKappaJoint,
                                          η_h, η_a,
                                          yh::Vector{Int}, ya::Vector{Int}, wts::Vector{Float64},
                                          lfh::Vector{Float64}, lfa::Vector{Float64},
                                          od::GMS_CB.JointGammaPoissonDesign)
    obs ~ DynamicPPL.to_submodel(GMS_CB._joint_gamma_poisson_params(o))
    ν = obs.ν

    ζ_h = η_h .+ obs.log_κ
    ζ_a = η_a .+ obs.log_κ
    ll_h = yh .* ζ_h .- exp.(ζ_h) .- lfh
    ll_a = ya .* ζ_a .- exp.(ζ_a) .- lfa
    goals_ll = sum(ll_h .* wts) + sum(ll_a .* wts)

    log_norm = ν * log(ν) - SpecialFunctions.loggamma(ν)
    inv_μ_h = exp.(.-η_h)
    inv_μ_a = exp.(.-η_a)
    g_h = (ν - 1.0) .* od.log_pxg_h .- (ν .* od.pxg_h) .* inv_μ_h .- ν .* η_h .+ log_norm
    g_a = (ν - 1.0) .* od.log_pxg_a .- (ν .* od.pxg_a) .* inv_μ_a .- ν .* η_a .+ log_norm
    proxy_ll = sum(g_h .* od.mask_weights) + sum(g_a .* od.mask_weights)

    return (; ll = goals_ll + proxy_ll, log_κ = obs.log_κ)
end

# An empty slot is an empty submodel: no site, no @addlogprob!, no tape instruction.
Turing.@model function _gms_supremacy(::NoMarketPillar, η_h, η_a, ::Nothing)
    return nothing
end

"""
C1. The Normal log-density is written out rather than built from `Normal.(μ, σ)`, for the
reason `engine.jl` §3 records: constructing a distribution per element inside a broadcast
widens the kernel ReverseDiff differentiates. The weights (pillar weight × availability mask
× match weight) were folded into one constant vector in the builder layer.
"""
Turing.@model function _gms_supremacy(p::MarketSupremacyPillar, η_h, η_a, md::GMSSupremacyDesign)
    σ_sup ~ p.σ_prior
    z = (md.m_sup .- (η_h .- η_a)) ./ σ_sup
    ll = -0.5 .* (z .* z) .- (log(σ_sup) + GMS_HALF_LOG_2PI)
    Turing.@addlogprob! sum(ll .* md.weights)
    return nothing
end

Turing.@model function _gms_smile(::NoMarketPillar, η_h, η_a, log_κ, ::Nothing)
    return nothing
end

"C2. `n_matches × n_strikes`, one broadcast; `log φ` enters as a `1 × n_strikes` row."
Turing.@model function _gms_smile(p::MarketSmilePillar, η_h, η_a, log_κ, md::GMSSmileDesign)
    σ_smile ~ p.σ_prior
    log_φ ~ Turing.filldist(Normal(0.0, p.shape_sd), md.n_strikes)
    log_λ_tot = log_κ .+ log.(exp.(η_h) .+ exp.(η_a))
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, md.n_strikes)
    z = (md.logΛ .- model_logΛ) ./ σ_smile
    ll = -0.5 .* (z .* z) .- (log(σ_smile) + GMS_HALF_LOG_2PI)
    Turing.@addlogprob! sum(ll .* md.weights)
    return nothing
end

"""
`composable_count_engine` (`engine.jl:443`) with the observation swapped for the κ-returning
copy and the two pillars appended.

Declaration order is inter → ha → dyn → predictors → obs → σ_sup → σ_smile → log_φ, so θ
is the base model's θ with the pillar sites APPENDED. G0 relies on that and asserts it.
The return value exists for G0's independent re-derivation of the pillar terms.
"""
Turing.@model function gms_market_anchored_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_ids::Vector{Int}, month_ids::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    dynamics_data,
    predictor_designs::Tuple,
    log_fact_h::Vector{Float64}, log_fact_a::Vector{Float64},
    observation_data,
    market::GMSMarketDesign,
    n_matches::Int, n_teams::Int, n_seasons::Int, n_months::Int,
    config::MarketAnchoredCountModel,
)
    base = config.base

    inter ~ DynamicPPL.to_submodel(GMS_PG.build_interception(base.interception, n_seasons, n_months))
    ha    ~ DynamicPPL.to_submodel(GMS_PG.build_home_advantage(base.home_advantage, n_teams))
    dyn   ~ DynamicPPL.to_submodel(GMS_CB._cb_dynamics_effects(
        base.dynamics, home_ids, away_ids, dynamics_data, n_teams))

    pred ~ DynamicPPL.to_submodel(
        GMS_CB._predictor_block(base.covariates, predictor_designs, n_matches), false)

    level = inter.μ_base[season_ids] .+ inter.δ_month[month_ids]
    η_h = GMS_CB.apply_guard(base.guard,
                             GMS_CB._predictor_shift(level .+ ha[home_ids] .+
                                                     dyn.att_h .+ dyn.def_a, pred.h))
    η_a = GMS_CB.apply_guard(base.guard,
                             GMS_CB._predictor_shift(level .+
                                                     dyn.att_a .+ dyn.def_h, pred.a))

    lik ~ DynamicPPL.to_submodel(
        _gms_joint_observe(base.observation, η_h, η_a, home_goals, away_goals,
                           match_weights, log_fact_h, log_fact_a, observation_data), false)
    Turing.@addlogprob! lik.ll

    sup ~ DynamicPPL.to_submodel(_gms_supremacy(config.supremacy, η_h, η_a, market.supremacy), false)
    sml ~ DynamicPPL.to_submodel(_gms_smile(config.smile, η_h, η_a, lik.log_κ, market.smile), false)

    return (; η_h, η_a, log_κ = lik.log_κ)
end

function GMS_PG.build_turing_model(m::MarketAnchoredCountModel, feature_set)
    z = GMS_CB.cb_design(m.base, feature_set)
    market = gms_market_design(m, feature_set, z.n_matches, z.match_weights)
    return gms_market_anchored_engine(
        z.home_ids, z.away_ids, z.season_ids, z.month_ids,
        z.home_goals, z.away_goals, z.match_weights,
        z.dynamics_data, z.predictor_designs, z.log_fact_h, z.log_fact_a, z.observation_data,
        market,
        z.n_matches, z.n_teams, z.n_seasons, z.n_months,
        m,
    )
end

# ==============================================================================
# 6. Extraction and pricing routes
# ==============================================================================

"""
Held-out rates are the base model's, untouched: the pillars carry no OOS term. The smile
variant adds `λ_tot = λ_h + λ_a` (= κ(μ_h + μ_a), the quantity C2 anchored) and the global
`φ` as an `n_draws × n_strikes` matrix — the shape `tpl_stack_smile` and
`smile_poisson.jl` read.
"""
function GMS_PG.extract_parameters(m::MarketAnchoredCountModel, df::AbstractDataFrame,
                                   feature_set, chain::Chains)
    raw = GMS_PG.extract_parameters(m.base, df, feature_set, chain)
    return gms_attach_smile(m.smile, raw, chain)
end

gms_attach_smile(::NoMarketPillar, raw, chain) = raw

function gms_smile_shape(p::MarketSmilePillar, chain::Chains)
    nK = gms_n_strikes(p)
    n_samples = size(chain, 1) * size(chain, 3)
    φ = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        φ[:, k] = exp.(vec(Array(chain[Symbol("log_φ[$k]")])))
    end
    return φ
end

function gms_attach_smile(p::MarketSmilePillar, raw, chain::Chains)
    φ = gms_smile_shape(p, chain)
    out = Dict{Int,NamedTuple}()
    for (mid, nt) in raw
        size(φ, 1) == length(nt.λ_h) || error(
            "φ has $(size(φ, 1)) draws but λ_h has $(length(nt.λ_h)) for match $mid")
        out[mid] = merge(nt, (; λ_tot = nt.λ_h .+ nt.λ_a, φ))
    end
    return out
end

GMS_LATENTS.latent_family(::GMSGridModel) = GMS_LATENTS.PoissonCountFamily()
GMS_LATENTS.latent_family(::GMSSmileModel) = GMS_LATENTS.SmilePoissonFamily()

# The legacy row route (`Portfolio.build_book` on a latent DataFrame, the replay desk's
# `_probs_from_latents`) dispatches on the MODEL type. These two pairs route the wrapper
# exactly as `smile_poisson.jl` routes the TimeDecay smile engines.
GMS_PRED.extract_params(::GMSGridModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
GMS_PRED.compute_score_matrix(::GMSGridModel, params; max_goals::Real = 12) =
    GMS_PRED._smile_poisson_grid(params.λ_h, params.λ_a; max_goals = Int(max_goals))

GMS_PRED.extract_params(::GMSSmileModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)
function GMS_PRED.compute_score_matrix(::GMSSmileModel, params; max_goals::Real = 12)
    grid = GMS_PRED._smile_poisson_grid(params.λ_h, params.λ_a; max_goals = Int(max_goals))
    Λ = Matrix{Float64}(transpose(params.λ_tot .* params.φ))
    return GMS_PRED.SmileScoreMatrix(grid, Λ)
end

# ==============================================================================
# 7. Gates
# ==============================================================================

"The sites a VarInfo holds, as strings, in θ order."
gms_site_names(vi) = [string(vn) for vn in keys(vi)]

"""
    gms_model_state(turing_model, varinfo) -> NamedTuple

The engine's return value `(; η_h, η_a, log_κ)` at the varinfo's values, through
`DynamicPPL.returned`. G0 uses it to re-derive the pillar terms without the engine's own
design objects.
"""
function gms_model_state(turing_model, varinfo)
    values = Dict{DynamicPPL.VarName,Any}(vn => varinfo[vn] for vn in keys(varinfo))
    return DynamicPPL.returned(turing_model, values)
end

"""
    gms_reference_pillars(m, fs, state, vi) -> Float64

The two pillars' contribution to the unlinked log density, written a second time from the
equations in this file's header — `Distributions.logpdf` on `Normal` and on the priors, with
the availability masks recomputed from the raw FeatureSet columns rather than read from the
engine's design.
"""
function gms_reference_pillars(m::MarketAnchoredCountModel, fs, state, vi)
    return gms_reference_supremacy(m.supremacy, fs, state, vi) +
           gms_reference_smile(m.smile, fs, state, vi)
end

gms_reference_supremacy(::NoMarketPillar, fs, state, vi) = 0.0

function gms_reference_supremacy(p::MarketSupremacyPillar, fs, state, vi)
    d = fs.data
    σ = vi[DynamicPPL.@varname(σ_sup)]
    total = logpdf(p.σ_prior, σ)
    for i in eachindex(state.η_h)
        h = d[:flat_market_λ_home][i]
        a = d[:flat_market_λ_away][i]
        (ismissing(h) || ismissing(a) || isnan(h) || isnan(a)) && continue
        (p.λ_lo < h < p.λ_hi && p.λ_lo < a < p.λ_hi) || continue
        target = log(h) - log(a)
        total += p.weight * logpdf(Normal(state.η_h[i] - state.η_a[i], σ), target)
    end
    return total
end

gms_reference_smile(::NoMarketPillar, fs, state, vi) = 0.0

function gms_reference_smile(p::MarketSmilePillar, fs, state, vi)
    d = fs.data
    σ = vi[DynamicPPL.@varname(σ_smile)]
    log_φ = vi[DynamicPPL.@varname(log_φ)]
    total = logpdf(p.σ_prior, σ) + sum(logpdf.(Normal(0.0, p.shape_sd), log_φ))
    logΛ = d[:flat_smile_logΛ]
    mask = d[:flat_smile_mask]
    for i in eachindex(state.η_h), k in 1:gms_n_strikes(p)
        mask[i, k] == 1.0 || continue
        level = state.log_κ + log(exp(state.η_h[i]) + exp(state.η_a[i])) + log_φ[k]
        total += p.weight * logpdf(Normal(level, σ), logΛ[i, k])
    end
    return total
end

"""
    gms_parity_check(model, base, feature_set; perturbations, seed) -> NamedTuple

G0. Two claims, checked at a prior draw and three displaced points:

1. **The base is untouched.** θ's leading sites are exactly the base model's sites, in
   order, and `logdensity(model, θ) − logdensity(base, θ_base)` is exactly the pillar
   contribution. For `gms_null_anchor(base)` that contribution is zero and the difference
   is required to be `0.0` — bit identity, not tolerance.
2. **The pillars are the equations.** The difference equals `gms_reference_pillars`, an
   independent re-derivation with `Distributions.logpdf`, to ≤ 1e-9 relative.

Compared through `LogDensityFunction` in the model's unlinked space, as Task 014's G0 was.
"""
function gms_parity_check(model::MarketAnchoredCountModel, base, feature_set;
                          perturbations = (0.01, -0.02, 0.035), seed::Int = 20260915)
    fs = first(feature_set)
    tm = GMS_PG.build_turing_model(model, fs)
    tb = GMS_PG.build_turing_model(base, fs)

    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(tm)
    vb = DynamicPPL.VarInfo(tb)
    names_model = gms_site_names(vi)
    names_base = gms_site_names(vb)
    n_base = length(vb[:])
    names_model[1:length(names_base)] == names_base || error(
        "G0: the wrapper's leading sites are not the base model's sites in order")

    f_model = let density = DynamicPPL.LogDensityFunction(tm)
        x -> LogDensityProblems.logdensity(density, x)
    end
    f_base = let density = DynamicPPL.LogDensityFunction(tb)
        x -> LogDensityProblems.logdensity(density, x)
    end

    # INDEPENDENT PRIOR DRAWS, not displacements of one draw. Under MultiScaleGRW several
    # sites are truncated scales, and a cos-displacement in the unlinked space steps outside
    # their support (log density −Inf on BOTH models — measured on the first r01 attempt).
    # Four prior draws are in support by construction and still rule out agreement by the
    # accident of one symmetric point.
    points = Vector{Float64}[copy(vi[:])]
    for j in eachindex(perturbations)
        push!(points, copy(DynamicPPL.VarInfo(Random.MersenneTwister(seed + j), tm)[:]))
    end

    worst_abs = 0.0
    worst_rel = 0.0
    base_deltas = Float64[]
    for point in points
        vpoint = DynamicPPL.unflatten(vi, point)
        engine = f_model(point)
        base_value = f_base(point[1:n_base])
        isfinite(engine) && isfinite(base_value) || error(
            "G0: non-finite log density (model $engine, base $base_value)")
        state = gms_model_state(tm, vpoint)
        reference = gms_reference_pillars(model, fs, state, vpoint)
        gap = (engine - base_value) - reference
        push!(base_deltas, engine - base_value)
        worst_abs = max(worst_abs, abs(gap))
        worst_rel = max(worst_rel, abs(gap) / max(abs(engine), 1.0))
    end

    return (; n_points = length(points), n_base_sites = length(names_base),
              n_pillar_sites = length(names_model) - length(names_base),
              pillar_sites = join(names_model[length(names_base)+1:end], ","),
              base_deltas, worst_abs, worst_rel)
end

"""
    gms_market_coverage(model, inputs) -> DataFrame

Per fold: how many training matches each pillar actually reads. A pillar with no observed
target is a prior and nothing else, and says nothing about whether the market helps.
"""
function gms_market_coverage(model::MarketAnchoredCountModel, inputs)
    rows = NamedTuple[]
    for (i, fs_tuple) in enumerate(inputs.feature_sets)
        fs = first(fs_tuple)
        z = GMS_CB.cb_design(model.base, fs)
        md = gms_market_design(model, fs, z.n_matches, z.match_weights)
        sup_n = md.supremacy === nothing ? 0 : md.supremacy.n_observed
        smile_rows = md.smile === nothing ? 0 :
            count(r -> any(==(1.0), view(fs.data[:flat_smile_mask], r, :)), 1:z.n_matches)
        per_k = md.smile === nothing ? "" :
            join([string(Int(sum(view(fs.data[:flat_smile_mask], :, k))))
                  for k in 1:md.smile.n_strikes], "/")
        push!(rows, (; fold = i, n_train = z.n_matches,
                       supremacy_observed = sup_n,
                       supremacy_share = sup_n / z.n_matches,
                       smile_matches = smile_rows,
                       smile_share = smile_rows / z.n_matches,
                       smile_per_strike = per_k))
    end
    return DataFrame(rows)
end

"""
    gms_latent_audit(fit) -> NamedTuple

`gph_latent_audit`'s checks on λ, plus — for a smile container — φ finite and positive,
`λ_tot == λ_home + λ_away` exactly, and the strike ladder 0.5 … 4.5.
"""
function gms_latent_audit(fit)
    lat = fit.latents
    lat isa CountLatents || lat isa SmileLatents || error(
        "latents are $(typeof(lat)); expected CountLatents or SmileLatents")
    lat.observation_params === nothing || error("a Poisson-family rung carries observation_params")
    allunique(lat.match_ids) || error("duplicate OOS match IDs in latents")
    for (side, draws) in (("home", lat.λ_home), ("away", lat.λ_away))
        all(isfinite, draws) || error("non-finite λ_$side draws")
        all(>(0.0), draws) || error("non-positive λ_$side draws")
    end
    v = vcat(vec(var(lat.λ_home; dims = 2)), vec(var(lat.λ_away; dims = 2)))
    all(>(0.0), v) || error("a fixture has zero posterior rate variance")

    φ_mean = Float64[]
    if lat isa SmileLatents
        lat.λ_tot == lat.λ_home .+ lat.λ_away || error("λ_tot ≠ λ_home + λ_away")
        all(isfinite, lat.φ) && all(>(0.0), lat.φ) || error("φ has non-finite or non-positive draws")
        lat.strikes == [0.5, 1.5, 2.5, 3.5, 4.5] || error("unexpected strike ladder $(lat.strikes)")
        φ_mean = [mean(view(lat.φ, :, s, :)) for s in 1:length(lat.strikes)]
    end
    return (; n_matches = length(lat.match_ids), n_draws = size(lat.λ_home, 2),
              family = lat isa SmileLatents ? "smile" : "count",
              mean_lambda_h = mean(lat.λ_home), mean_lambda_a = mean(lat.λ_away),
              min_sd = sqrt(minimum(v)), φ_mean = join(round.(φ_mean; digits = 3), "/"))
end

"""
    gms_smile_pricing_gate(fit; n_fixtures) -> DataFrame

For a smile container: the smile O/U price must be what the equations say, through BOTH
routes that will consume it, and must differ from the plain grid's.

* `p_under_ref` — `mean_s cdf(Poisson(λ_tot·φ_K), K)`, computed here from the container.
* `p_under_typed` — `compute_score_grid` + `price_market` (evaluation and portfolio route).
* `p_under_legacy` — `compute_score_matrix(model, extract_params(model, row))` on the
  legacy row (MatchDay replay route).
* `p_under_grid` — the double-Poisson grid at the same λ draws, for the size of the smile.
"""
function gms_smile_pricing_gate(fit; n_fixtures::Int = 6, line::Float64 = 2.5)
    lat = fit.latents
    lat isa SmileLatents || error("smile pricing gate needs SmileLatents; got $(typeof(lat))")
    model = fit.config.model
    market = Data.MarketOverUnder(line)
    K = Int(floor(line))
    twin = CountLatents(lat.match_ids, lat.λ_home, lat.λ_away, nothing)
    rows = NamedTuple[]
    for i in 1:min(n_fixtures, n_matches(lat))
        ref = mean(cdf(Poisson(lat.λ_tot[i, s] * lat.φ[i, K + 1, s]), K)
                   for s in 1:size(lat.λ_home, 2))
        typed = GMS_PRED.price_market(GMS_PRED.compute_score_grid(lat, i), market)
        grid = GMS_PRED.price_market(GMS_PRED.compute_score_grid(twin, i), market)
        under_key = Data.outcomes(market).under
        row = (λ_h = vec(lat.λ_home[i, :]), λ_a = vec(lat.λ_away[i, :]),
               λ_tot = vec(lat.λ_tot[i, :]), φ = permutedims(lat.φ[i, :, :]))
        legacy = GMS_PRED.compute_market_probs(
            GMS_PRED.compute_score_matrix(model, GMS_PRED.extract_params(model, row)), market)
        push!(rows, (; fixture = lat.match_ids[i],
                       p_under_ref = ref,
                       p_under_typed = mean(typed[under_key]),
                       p_under_legacy = mean(legacy[under_key]),
                       p_under_grid = mean(grid[under_key]),
                       smile_shift = ref - mean(grid[under_key])))
    end
    return DataFrame(rows)
end

# ==============================================================================
# 8. Persistence
# ==============================================================================

"Latents rebuilt from a fit's own persisted chains, with the fold inputs it was fitted on."
function gms_rebuild_latents(fit, inputs)
    latents, note = extract_run_latents(fit.config.model, fit.folds, inputs.oos, inputs.feature_sets)
    latents === nothing && error("latent re-extraction failed: $note")
    return latents
end

function gms_latents_equal(a, b)
    typeof(a) == typeof(b) || return false
    a.match_ids == b.match_ids && a.λ_home == b.λ_home && a.λ_away == b.λ_away || return false
    a isa SmileLatents || return true
    return a.λ_tot == b.λ_tot && a.φ == b.φ && a.strikes == b.strikes
end

"""
    gms_save_and_verify(db, fit, inputs; latent_dir) -> UUID

Persist and prove the reload.

* `CountLatents` runs go through `gph_save_and_verify` unchanged.
* `SmileLatents` runs are saved with the panel DETACHED (ticket T010), the typed container is
  written beside the results as `latent_dir/<run_id>/oos_latents.jls`, and the reload is
  verified three ways: chains identical, latents re-extracted from the PERSISTED chains equal
  to the in-memory container, and the file copy equal to both.
"""
function gms_save_and_verify(db, fit, inputs; latent_dir::AbstractString)
    fit.latents isa SmileLatents || return gph_save_and_verify(db, fit)

    existing = gph_completed_run(db, fit.config)
    existing === nothing || error(
        "recipe $(fit.config.name) is already persisted as run $existing; load it instead")
    detached = Fit(fit.config, fit.folds, nothing, fit.diagnostics, fit.metadata, fit.save_path)
    run_id = save_fit(detached, db)

    target = joinpath(latent_dir, string(run_id))
    mkpath(target)
    save_latents(target, fit.latents)

    reloaded = load_fit(db, run_id)
    length(reloaded.folds) == length(fit.folds) || error("round-trip fold count differs")
    reloaded.diagnostics.max_rhat == fit.diagnostics.max_rhat || error("round-trip R̂ differs")
    for (a, b) in zip(fit.folds, reloaded.folds)
        parent(a.chain.value) == parent(b.chain.value) || error(
            "round-trip chain differs on fold $(a.fold)")
    end
    rebuilt = gms_rebuild_latents(reloaded, inputs)
    gms_latents_equal(rebuilt, fit.latents) || error(
        "SmileLatents re-extracted from the persisted chains differ from the fitted container")
    from_file = load_latents(target)
    gms_latents_equal(from_file, fit.latents) || error("file copy of SmileLatents differs")
    return run_id
end

"""
    gms_load_fit(db, run_id, ds; splitter) -> Fit

A persisted run with a typed latent panel. Smile runs come back from PostgreSQL with no
panel (T010); it is rebuilt from the persisted chains over the run's own splitter.
"""
function gms_load_fit(db, run_id::UUID, ds; splitter)
    fit = load_fit(db, run_id)
    fit.latents === nothing || return fit
    inputs = gph_fold_inputs(ds, splitter, fit.config.model)
    length(inputs.feature_sets) == length(fit.folds) || error(
        "run $run_id holds $(length(fit.folds)) folds but the splitter builds $(length(inputs.feature_sets))")
    latents = gms_rebuild_latents(fit, inputs)
    return Fit(fit.config, fit.folds, latents, fit.diagnostics, fit.metadata, fit.save_path)
end

"The newest completed run named `name` in `db`, or `nothing`."
gms_run_by_name(db, name::AbstractString) = gph_run_by_name(db, name)

"""
    gms_split_prefix_check(ds, c) -> Int

Folds 1–40 of the 43-fold production split must be the 40-fold walk-forward split, fixture
for fixture — that is what makes a 43-fold run's first 710 fixtures the canonical panel.
"""
function gms_split_prefix_check(ds, c::GMSConfig)
    short = Data.create_id_boundaries(ds, gph_splitter(c.target_seasons))
    long = Data.create_id_boundaries(ds, gph_splitter(c.extension_seasons))
    length(short) == c.expected_folds || error("40-fold split has $(length(short)) folds")
    length(long) == c.expected_extended_folds || error(
        "production split has $(length(long)) folds; expected $(c.expected_extended_folds) — " *
        "check the DataStore snapshot date (it must end before the 2026-09-12 card)")
    # Each entry is `(boundary, split metadata)`; the fixture sets live on the first element,
    # exactly as `gph_filtration_report` reads them.
    ids(b) = (copy(first(b).history_match_ids), copy(first(b).target_match_ids))
    for i in 1:c.expected_folds
        ids(short[i]) == ids(long[i]) || error("fold $i differs between the 40- and 43-fold splits")
    end
    return length(long)
end

# ==============================================================================
# 9. Recipes and registry
# ==============================================================================

gms_smoke_sampler(c::GMSConfig) = gph_smoke_sampler(gms_gph_config(c))
gms_production_sampler(c::GMSConfig) = gph_production_sampler(gms_gph_config(c))
gms_execution(c::GMSConfig) = gph_execution(gms_gph_config(c))

function gms_fit_configs(c::GMSConfig, models, splitter, sampler; name_suffix::AbstractString = "")
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gms_execution(c),
        tags = copy(GMS_TAGS),
        description = GMS_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

"""
    gms_register!(db, models, splitter, sampler, configs) -> NamedTuple

Register every rung. `save_model` classifies with a hard-coded `isa ComposableCountModel`
(`db_storage.jl:214`), so it refuses the wrapper (first r01 attempt, G6). A wrapper therefore
registers its BASE builder model as `<name>__base_model`, and its complete recipe — pillars,
weights and priors included — as the `<name>_fit` entry, whose `FitConfig` classification and
exact `config_blob` do not depend on the model type. Recorded in ticket T010.
"""
function gms_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    for (name, model) in models
        if model isa MarketAnchoredCountModel
            model_ids[name * "__base_model"] = save_model(db, name * "__base_model", model.base;
                description = "Base builder model of $name (pillars live in $(name)_fit).",
                tags = GMS_TAGS)
        else
            model_ids[name] = save_model(db, name, model;
                                         description = GMS_DESCRIPTIONS[name], tags = GMS_TAGS)
        end
        save_config(db, name * "_fit", configs[name];
                    description = GMS_DESCRIPTIONS[name] * " Task 015 recipe.", tags = GMS_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_grw_smile_43fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25, 25/26 and 26/27 to Fold 43.",
        tags = GMS_TAGS)
    # Named from the sampler's own budget. The registry upserts on (experiment, name), so a fixed
    # name would let a larger-budget re-run of one rung overwrite the canonical sampler entry.
    sampler_name = @sprintf("queued_nuts_%dx%d_w%d_a%03d", sampler.n_chains, sampler.n_samples,
                            sampler.n_warmup, round(Int, 100 * sampler.accept_rate))
    sampler_id = save_sampler(db, sampler_name, sampler;
        description = "ReverseDiff queued NUTS: $(sampler.n_chains) chains, $(sampler.n_warmup) warmup, " *
                      "$(sampler.n_samples) retained, target acceptance $(sampler.accept_rate).",
        tags = GMS_TAGS)
    return (; model_ids, splitter_id, sampler_id)
end

"One convergence row, Task 013's columns plus the pillar posteriors pooled over folds."
function gms_convergence_row(name::AbstractString, fit, c::GMSConfig; run_id = nothing)
    row = gph_convergence_row(name, fit, gms_gph_config(c); run_id)
    return (; row..., gms_pillar_summary(fit)...)
end

function gms_pillar_summary(fit)
    pooled(sym) = begin
        draws = Float64[]
        for f in fit.folds
            sym in names(f.chain) && append!(draws, vec(Array(f.chain[sym])))
        end
        draws
    end
    q(v, p) = isempty(v) ? NaN : quantile(v, p)
    σ_sup = pooled(:σ_sup)
    σ_smile = pooled(:σ_smile)
    κ = exp.(pooled(Symbol("obs.log_κ")))
    φ = [exp.(pooled(Symbol("log_φ[$k]"))) for k in 1:5]
    return (; σ_sup_median = q(σ_sup, 0.5), σ_sup_q05 = q(σ_sup, 0.05), σ_sup_q95 = q(σ_sup, 0.95),
              σ_smile_median = q(σ_smile, 0.5), σ_smile_q05 = q(σ_smile, 0.05),
              σ_smile_q95 = q(σ_smile, 0.95), κ_median = q(κ, 0.5),
              φ_median = all(isempty, φ) ? "" : join([gph_num(q(v, 0.5); digits = 3) for v in φ], "/"))
end
