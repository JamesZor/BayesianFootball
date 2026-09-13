# ==============================================================================
# Task 016 loader — MultiScaleGRW with Market Supremacy and a 1-parameter Smile Spine
# ==============================================================================
#
# Definitions only. `r01_smoke.jl` and `r02_production_grid.jl` execute.
#
# THE QUESTION. Task 015's five free strike shapes (log φ_K ~ Normal(0, 0.5)^5) bought
# portfolio growth and cost 158–185 min per 43-fold grid with bulk ESS down to 431
# (312 at @0.70). Its pooled posterior medians were
#
#     φ(K = 0…4) = 0.843 / 0.976 / 1.001 / 1.026 / 1.069     (smile + sup @0.40)
#
# which this task replaces with ONE slope through the 2.5 line:
#
#     log φ(K) = β · (K − 2),        β ~ Normal(0.04, 0.05),        φ(2) ≡ 1
#
# READ THE K = 0 RESIDUAL BEFORE READING ANY RESULT. The least-squares slope through
# Task 015's medians is β ≈ 0.052 (`gss_task015_line_fit`), and the line misses K = 0 by
# about −0.065 in log φ (0.90 on the line vs 0.84 measured) while fitting K = 1…4 within
# ±0.04. The spine cannot represent that kink. Where Under 0.5 / Over 0.5 matter, a spine
# that loses to the five-strike smile is losing THERE, and the per-strike tables in r04
# should say so rather than average it away.
#
# THE LADDER
#
#   m05_joint_grw_baseline               no pillar                  pinned  Task 013 b0961bc4
#   m05_joint_grw_supremacy_w040         C1 @ 0.40                  pinned  Task 015 0ee58d18
#   m05_joint_grw_smile_supremacy_w020   C1 + C2 five strikes @ 0.20 pinned  Task 015 fcd5e974
#   m05_joint_grw_smile_supremacy_w040   C1 + C2 five strikes @ 0.40 pinned  Task 015 30620d3e
#   m05_joint_grw_smile_spine_w020       C1 + C2 spine @ 0.20       SAMPLED here
#   m05_joint_grw_smile_spine_w040       C1 + C2 spine @ 0.40       SAMPLED here
#
# THE SPINE PILLAR is Task 015's C2 with `log_φ` replaced by `β_spine · (K − pivot)`:
#
#   C2  log κ + log(μ_h + μ_a) + β_spine·(K − 2) ~ Normal(log Λ̂_K, σ_smile),  K = 0…4
#
# Same σ_smile prior, same tempering, same MarketSmileFeature(Kmax = 4), same masks. The site
# is `β_spine`, not `β`: MultiScaleGRW already declares `dyn.β`, and one name meaning two
# things in a chain summary is how a table gets misread.
#
# WHY A SECOND WRAPPER TYPE. Task 015's `MarketAnchoredCountModel` bounds its smile slot to
# `Union{NoMarketPillar, MarketSmilePillar}`. Widening that Union would change the type that
# the pinned five-strike artefacts were serialised with, and r04 must deserialise them. So
# this file INCLUDES Task 015's loader unchanged and adds `SpineAnchoredCountModel` beside
# it. The engine is Task 015's, copied for the type annotation only; G0a proves the copy is
# the base model bit for bit, and G0c proves the spine IS the five-strike model restricted to
# the line log φ_K = β(K − 2).
#
# TICKET T011 — ANTI-DIAGONAL REWEIGHTING (§8). `Portfolio` prices a smile container's O/U
# through λ_tot·φ(K) but sizes every stake off the plain (λ_h, λ_a) grid. Here each posterior
# draw's 12×12 grid is rescaled on its anti-diagonals G = h + a so that its totals marginal is
# the smile's, and the SAME `_finish_book` (mean grid → Kelly, per-draw grid → BakerMcHale)
# then stakes off it. Four interpretation choices, each one a place the work package is
# silent or loose:
#
#   1. A 12×12 grid has 23 anti-diagonals (G = 0…22), not 12. Every one is rescaled.
#   2. G ≤ Kmax takes the spine's increments F(G) − F(G−1), F(K) = cdf(Poisson(λ_tot·φ_K), K).
#      Mass above Kmax, 1 − F(Kmax), is spread over G = Kmax+1 … 22 IN PROPORTION TO THE
#      GRID'S OWN anti-diagonal masses. For G ≤ 11 those are exactly Poisson(λ_h + λ_a)
#      masses (every pair with h + a = G is inside the grid), so this is the work package's
#      "relative Poisson tail"; for G ≥ 12 it keeps the grid's truncated cells in the ratio the
#      allocator's scenario space already holds them. Each draw then sums to exactly 1.
#   3. A per-strike smile is a CDF only where it is monotone in K. A draw whose F decreases
#      is REFUSED with an error naming the draw, strike, λ_tot and φ — never clipped.
#   4. A draw whose φ is exactly 1 at every strike is left untouched, so a φ ≡ 1 container
#      stakes the bit-identical ledger of its grid twin (T011 acceptance). The un-shortcut
#      arithmetic at φ ≡ 1 is measured separately (`gss_identity_path_gate`) so the shortcut
#      cannot hide a real discrepancy.
#
# Reweighting is per draw, so the MEAN grid's P(total ≤ K) is the mean of cdf(Poisson(λ_tot·φ_K), K)
# exactly — the quantity `p_model` already reports — and BakerMcHale's per-draw re-solves see
# the smile too.
#
# FILTRATION. Unchanged from Task 015: both pillars read only a fold's training fixtures;
# OOS prices are λ from the walk's final state and β from the chain.
#
# PERSISTENCE (ticket T010). Unchanged from Task 015 §8: a smile run is saved with its latent
# panel detached and `gms_load_fit` re-extracts `SmileLatents` from the persisted chains. This
# loader must be included before a spine artefact is deserialised.
# ==============================================================================

if !isdefined(@__MODULE__, :MarketAnchoredCountModel)
    include(joinpath(@__DIR__, "..", "grw_market_smile", "l01_loader.jl"))
end

const GSS_PORTFOLIO = BayesianFootball.Portfolio
const GSS_EVALUATION = BayesianFootball.Evaluation
const GSS_TRAINING = BayesianFootball.Training
const GSS_MATCHDAY = BayesianFootball.MatchDay

# ==============================================================================
# 1. Experiment configuration
# ==============================================================================

"""
    gss_config(; overrides...) -> GMSConfig

Task 015's configuration struct with this task's namespaces. Every other number — folds,
panel size, sampler budget, convergence thresholds — is Task 015's, so a benchmark against it
compares models and not settings.

`smoke_samples = 1000` rather than 015's default 500: Task 015's smoke measured the pinned
baseline FAILING tail ESS (363) at 4 × (500 + 500) and passing at 4 × (500 + 1000), the budget
r02 uses. A smoke gate that fails the control on draw count says nothing about the candidate.
"""
function gss_config(; overrides...)
    defaults = (; experiment = "scottish_lower_grw_smile_spine",
                  smoke_experiment = "smoke_grw_smile_spine",
                  save_root = joinpath(@__DIR__, "results"),
                  smoke_samples = 1000)
    return GMSConfig(; merge(defaults, values(overrides))...)
end

const GSS_MODEL_NAMES = [
    "m05_joint_grw_baseline",
    "m05_joint_grw_supremacy_w040",
    "m05_joint_grw_smile_supremacy_w020",
    "m05_joint_grw_smile_supremacy_w040",
    "m05_joint_grw_smile_spine_w020",
    "m05_joint_grw_smile_spine_w040",
]

"Sampled by r01 and r02. Every other rung is a pinned, persisted run."
const GSS_GRID_MODEL_NAMES = GSS_MODEL_NAMES[5:6]

"Each spine rung and the five-strike rung it is a restriction of (G0c)."
const GSS_LINE_PAIRS = [
    "m05_joint_grw_smile_spine_w020" => "m05_joint_grw_smile_supremacy_w020",
    "m05_joint_grw_smile_spine_w040" => "m05_joint_grw_smile_supremacy_w040",
]

const GSS_DESCRIPTIONS = Dict(
    "m05_joint_grw_baseline" => GMS_DESCRIPTIONS["m05_joint_grw_baseline"],
    "m05_joint_grw_supremacy_w040" => GMS_DESCRIPTIONS["m05_joint_grw_supremacy_w040"],
    "m05_joint_grw_smile_supremacy_w020" => GMS_DESCRIPTIONS["m05_joint_grw_smile_supremacy_w020"],
    "m05_joint_grw_smile_supremacy_w040" => GMS_DESCRIPTIONS["m05_joint_grw_smile_supremacy_w040"],
    "m05_joint_grw_smile_spine_w020" =>
        "m05 joint GRW + market supremacy and 1-parameter smile spine log φ = β(K−2), light weights 0.20/0.20.",
    "m05_joint_grw_smile_spine_w040" =>
        "m05 joint GRW + market supremacy and 1-parameter smile spine log φ = β(K−2), moderate weights 0.40/0.40.",
)

const GSS_TAGS = [
    "scottish-lower", "24/25", "25/26", "26/27", "multiscale-grw", "joint-gamma-poisson",
    "market-supremacy", "market-smile", "smile-spine", "todo016", "reversediff",
]

"A ladder rung that is loaded, not sampled."
struct GSSPinnedRun
    rung::String
    experiment::String
    run_id::UUID
    source::String
end

const GSS_PINNED_RUNS = [
    GSSPinnedRun("m05_joint_grw_baseline", "scottish_lower_grw_player_hybrid",
                 UUID("b0961bc4-c40c-4dbe-9c05-57df7ae0839e"), "Task 013 m05_wealth_grw"),
    GSSPinnedRun("m05_joint_grw_supremacy_w040", "scottish_lower_grw_market_smile",
                 UUID("0ee58d18-b7e9-4168-8d78-93887b1a8c26"), "Task 015"),
    GSSPinnedRun("m05_joint_grw_smile_supremacy_w020", "scottish_lower_grw_market_smile",
                 UUID("fcd5e974-9a46-4a10-9828-6b987a5484d6"), "Task 015"),
    GSSPinnedRun("m05_joint_grw_smile_supremacy_w040", "scottish_lower_grw_market_smile",
                 UUID("30620d3e-e4bd-4c05-b1a1-85cefa36b728"), "Task 015"),
]

"""
Task 015's production grid, copied from its README (4 × (500 + 1000), 43 folds, mcmc-beast
-t 16). The benchmark H1 is judged against. Not re-measured here.
"""
const GSS_TASK015_PRODUCTION = [
    (; rung = "m05_joint_grw_supremacy_w040", wall_min = 60.0, min_ess_bulk = 814.0, min_ess_tail = 516.0, max_rhat = 1.0105),
    (; rung = "m05_joint_grw_smile_supremacy_w020", wall_min = 158.0, min_ess_bulk = 472.0, min_ess_tail = 498.0, max_rhat = 1.0139),
    (; rung = "m05_joint_grw_smile_supremacy_w040", wall_min = 185.0, min_ess_bulk = 431.0, min_ess_tail = 696.0, max_rhat = 1.0200),
]

"Task 015's pooled φ medians for smile + sup @0.40 (README, pillar posteriors), K = 0…4."
const GSS_TASK015_PHI_MEDIAN = [0.843, 0.976, 1.001, 1.026, 1.069]

# ==============================================================================
# 2. The spine pillar and the wrapper
# ==============================================================================

"""
    MarketSmileSpinePillar(; weight, σ_prior, β_prior, pivot, feature)

C2 with one shape parameter: the per-strike total intensity is
`log κ + log(μ_h + μ_a) + β_spine · (K − pivot)`. At `K == pivot` the shape term is
`β · 0.0` and φ is exactly 1 in floating point, whatever β is.

Defaults are the work package's: β ~ Normal(0.04, 0.05) — centred a little below Task 015's
implied least-squares slope (≈ 0.052) — pivot 2 (the 2.5 line, the deepest-quoted strike),
σ_smile and Kmax exactly Task 015's.
"""
Base.@kwdef struct MarketSmileSpinePillar{D<:ContinuousUnivariateDistribution,
                                          P<:ContinuousUnivariateDistribution} <: AbstractMarketPillar
    weight::Float64 = 0.40
    σ_prior::D = truncated(Normal(0.15, 0.10), lower = 0.02)
    β_prior::P = Normal(0.04, 0.05)
    pivot::Int = 2
    feature::GMS_FEATURES.MarketSmileFeature = GMS_FEATURES.MarketSmileFeature(Kmax = 4)
end

gms_n_strikes(p::MarketSmileSpinePillar) = p.feature.Kmax + 1

"`K − pivot` for K = 0…Kmax, as the Float64 row the engine multiplies β by."
gss_offsets(p::MarketSmileSpinePillar) = Float64[K - p.pivot for K in 0:p.feature.Kmax]

"""
    SpineAnchoredCountModel(base, supremacy, smile)

A built Task 013 `PoissonCountModel` plus a supremacy slot and a spine slot. Task 015's
`MarketAnchoredCountModel` with the smile slot typed for the spine — see the header for why
it is a second type rather than a widened one.
"""
struct SpineAnchoredCountModel{
    B<:GMS_CB.PoissonCountModel,
    S<:Union{NoMarketPillar,MarketSupremacyPillar},
    K<:Union{NoMarketPillar,MarketSmileSpinePillar},
} <: GMS_TI.AbstractPoissonModel
    base::B
    supremacy::S
    smile::K

    function SpineAnchoredCountModel(base::B, supremacy::S, smile::K) where {B,S,K}
        base.observation isa GMS_CB.SharedKappaJoint || error(
            "SpineAnchoredCountModel is written against the shared-κ JointGammaPoissonObservation; " *
            "got $(typeof(base.observation)). The spine pillar reads obs.log_κ from that block.")
        if smile isa MarketSmileSpinePillar
            0 <= smile.pivot <= smile.feature.Kmax || error(
                "spine pivot $(smile.pivot) is outside the strike ladder 0…$(smile.feature.Kmax); " *
                "φ(pivot) ≡ 1 only means something at a strike the pillar reads")
        end
        return new{B,S,K}(base, supremacy, smile)
    end
end

const GSSSpineModel = SpineAnchoredCountModel{B,S,<:MarketSmileSpinePillar} where {B,S}
const GSSGridModel = SpineAnchoredCountModel{B,S,NoMarketPillar} where {B,S}

# ==============================================================================
# 3. The ladder
# ==============================================================================

gss_spine(w::Real) = MarketSmileSpinePillar(weight = Float64(w))

"""
    gss_models() -> Vector{Tuple{String,Any}}

All six rungs in ladder order, built from ONE base object. Rungs 1–4 are rebuilt exactly as
Task 013/015 built them so r02 can assert `string(model)` equality with the pinned artefacts
before loading them.
"""
function gss_models()
    base = gms_base_model()
    return Tuple{String,Any}[
        ("m05_joint_grw_baseline", base),
        ("m05_joint_grw_supremacy_w040",
         MarketAnchoredCountModel(base, gms_supremacy(0.40), NoMarketPillar())),
        ("m05_joint_grw_smile_supremacy_w020",
         MarketAnchoredCountModel(base, gms_supremacy(0.20), gms_smile(0.20))),
        ("m05_joint_grw_smile_supremacy_w040",
         MarketAnchoredCountModel(base, gms_supremacy(0.40), gms_smile(0.40))),
        ("m05_joint_grw_smile_spine_w020",
         SpineAnchoredCountModel(base, gms_supremacy(0.20), gss_spine(0.20))),
        ("m05_joint_grw_smile_spine_w040",
         SpineAnchoredCountModel(base, gms_supremacy(0.40), gss_spine(0.40))),
    ]
end

"Both slots empty, through THIS file's engine copy — G0a's bit-identity witness. Never sampled."
gss_null_anchor(base) = SpineAnchoredCountModel(base, NoMarketPillar(), NoMarketPillar())

"`MatchDay.option_b_system()`'s book spec — the portfolio contract every Task 015 runner used."
gss_option_b_book() = getproperty(GSS_MATCHDAY.option_b_system(), :book)

# ==============================================================================
# 4. Features and design
# ==============================================================================

gms_pillar_features(p::MarketSmileSpinePillar) = GMS_FEATURES.AbstractFeatureConfig[p.feature]

function GMS_FEATURES.required_features(m::SpineAnchoredCountModel)
    out = GMS_FEATURES.required_features(m.base)
    append!(out, gms_pillar_features(m.supremacy))
    append!(out, gms_pillar_features(m.smile))
    return out
end

"C2 spine data: Task 015's smile design plus the `1 × n_strikes` offset row."
struct GSSSpineDesign
    logΛ::Matrix{Float64}
    weights::Matrix{Float64}
    offsets::Matrix{Float64}
    n_strikes::Int
    n_observed::Int
end

"""
The masks, weights and validation are Task 015's, reached through a five-strike pillar with
the same weight, σ prior and feature — so the two pillars cannot read different data.
"""
function gms_pillar_design(p::MarketSmileSpinePillar, fs, n::Int, wts::Vector{Float64})
    five = MarketSmilePillar(weight = p.weight, σ_prior = p.σ_prior, feature = p.feature)
    d = gms_pillar_design(five, fs, n, wts)
    offsets = reshape(gss_offsets(p), 1, d.n_strikes)
    return GSSSpineDesign(d.logΛ, d.weights, offsets, d.n_strikes, d.n_observed)
end

function gms_market_design(m::SpineAnchoredCountModel, fs, n::Int, wts::Vector{Float64})
    return GMSMarketDesign(gms_pillar_design(m.supremacy, fs, n, wts),
                           gms_pillar_design(m.smile, fs, n, wts))
end

# ==============================================================================
# 5. The engine
# ==============================================================================

"""
C2 spine. Task 015's `_gms_smile` with `reshape(log_φ, 1, K)` replaced by `β_spine .* offsets`
— the same broadcast, so at `log_φ = β .* offsets` the n × K likelihood matrix is the same
Float64 array (G0c). A new METHOD of `_gms_smile`, so the engine's call site is unchanged.
"""
Turing.@model function _gms_smile(p::MarketSmileSpinePillar, η_h, η_a, log_κ, md::GSSSpineDesign)
    σ_smile ~ p.σ_prior
    β_spine ~ p.β_prior
    log_λ_tot = log_κ .+ log.(exp.(η_h) .+ exp.(η_a))
    model_logΛ = log_λ_tot .+ β_spine .* md.offsets
    z = (md.logΛ .- model_logΛ) ./ σ_smile
    ll = -0.5 .* (z .* z) .- (log(σ_smile) + GMS_HALF_LOG_2PI)
    Turing.@addlogprob! sum(ll .* md.weights)
    return nothing
end

"""
`gms_market_anchored_engine` (Task 015 l01 §5), copied line for line because its `config`
argument is typed `MarketAnchoredCountModel`. Declaration order inter → ha → dyn → predictors
→ obs → σ_sup → σ_smile → β_spine: θ is the base model's θ with the pillar sites appended.
"""
Turing.@model function gss_spine_anchored_engine(
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
    config::SpineAnchoredCountModel,
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

function GMS_PG.build_turing_model(m::SpineAnchoredCountModel, feature_set)
    z = GMS_CB.cb_design(m.base, feature_set)
    market = gms_market_design(m, feature_set, z.n_matches, z.match_weights)
    return gss_spine_anchored_engine(
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

function GMS_PG.extract_parameters(m::SpineAnchoredCountModel, df::AbstractDataFrame,
                                   feature_set, chain::Chains)
    raw = GMS_PG.extract_parameters(m.base, df, feature_set, chain)
    return gms_attach_smile(m.smile, raw, chain)
end

"""
φ as the `n_draws × n_strikes` matrix `tpl_stack_smile` reads, from the `β_spine` draws in the
chain's own draw order (the order the base extractor reads λ in, as Task 015's `log_φ[k]`).
"""
function gms_smile_shape(p::MarketSmileSpinePillar, chain::Chains)
    β = vec(Array(chain[:β_spine]))
    offsets = gss_offsets(p)
    φ = Matrix{Float64}(undef, length(β), length(offsets))
    for (s, offset) in enumerate(offsets)
        φ[:, s] = exp.(β .* offset)
    end
    return φ
end

function gms_attach_smile(p::MarketSmileSpinePillar, raw, chain::Chains)
    φ = gms_smile_shape(p, chain)
    out = Dict{Int,NamedTuple}()
    for (mid, nt) in raw
        size(φ, 1) == length(nt.λ_h) || error(
            "φ has $(size(φ, 1)) draws but λ_h has $(length(nt.λ_h)) for match $mid")
        out[mid] = merge(nt, (; λ_tot = nt.λ_h .+ nt.λ_a, φ))
    end
    return out
end

GMS_LATENTS.latent_family(::GSSGridModel) = GMS_LATENTS.PoissonCountFamily()
GMS_LATENTS.latent_family(::GSSSpineModel) = GMS_LATENTS.SmilePoissonFamily()

# The legacy row route (MatchDay replay desk), exactly as Task 015 routes its wrapper.
GMS_PRED.extract_params(::GSSGridModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
GMS_PRED.compute_score_matrix(::GSSGridModel, params; max_goals::Real = 12) =
    GMS_PRED._smile_poisson_grid(params.λ_h, params.λ_a; max_goals = Int(max_goals))

GMS_PRED.extract_params(::GSSSpineModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)
function GMS_PRED.compute_score_matrix(::GSSSpineModel, params; max_goals::Real = 12)
    grid = GMS_PRED._smile_poisson_grid(params.λ_h, params.λ_a; max_goals = Int(max_goals))
    Λ = Matrix{Float64}(transpose(params.λ_tot .* params.φ))
    return GMS_PRED.SmileScoreMatrix(grid, Λ)
end

# ==============================================================================
# 7. Likelihood gates (G0)
# ==============================================================================

gss_log_density(turing_model) = let density = DynamicPPL.LogDensityFunction(turing_model)
    x -> LogDensityProblems.logdensity(density, x)
end

"The pillar terms of the unlinked log density, re-derived with `Distributions.logpdf`."
gss_reference_pillars(m::MarketAnchoredCountModel, fs, state, vi) =
    gms_reference_pillars(m, fs, state, vi)
gss_reference_pillars(m::SpineAnchoredCountModel, fs, state, vi) =
    gms_reference_supremacy(m.supremacy, fs, state, vi) + gss_reference_spine(m.smile, fs, state, vi)

gss_reference_spine(::NoMarketPillar, fs, state, vi) = 0.0

"""
The spine term written from the header's equation: raw FeatureSet columns (not the engine's
design), the offset recomputed as the integer `(k − 1) − pivot`, one `logpdf(Normal)` per
observed cell.
"""
function gss_reference_spine(p::MarketSmileSpinePillar, fs, state, vi)
    d = fs.data
    σ = vi[DynamicPPL.@varname(σ_smile)]
    β = vi[DynamicPPL.@varname(β_spine)]
    total = logpdf(p.σ_prior, σ) + logpdf(p.β_prior, β)
    logΛ = d[:flat_smile_logΛ]
    mask = d[:flat_smile_mask]
    for i in eachindex(state.η_h), k in 1:gms_n_strikes(p)
        mask[i, k] == 1.0 || continue
        level = state.log_κ + log(exp(state.η_h[i]) + exp(state.η_a[i])) + β * ((k - 1) - p.pivot)
        total += p.weight * logpdf(Normal(level, σ), logΛ[i, k])
    end
    return total
end

"""
    gss_parity_check(model, base, feature_set; n_extra_points, seed) -> NamedTuple

G0a/G0b. Task 015's `gms_parity_check` for either wrapper type: θ's leading sites must be the
base model's in order, and at a prior draw plus `n_extra_points` independent prior draws,
`logdensity(model) − logdensity(base)` must equal `gss_reference_pillars`. With both slots
empty the caller requires every base delta to be exactly `0.0`.
"""
function gss_parity_check(model, base, feature_set; n_extra_points::Int = 3, seed::Int = 20260916)
    fs = first(feature_set)
    tm = GMS_PG.build_turing_model(model, fs)
    tb = GMS_PG.build_turing_model(base, fs)

    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(tm)
    vb = DynamicPPL.VarInfo(tb)
    names_model = gms_site_names(vi)
    names_base = gms_site_names(vb)
    n_base = length(vb[:])
    length(names_model) >= length(names_base) &&
        names_model[1:length(names_base)] == names_base || error(
        "G0: the wrapper's leading sites are not the base model's sites in order")

    f_model = gss_log_density(tm)
    f_base = gss_log_density(tb)

    # Independent prior draws, not displacements: Task 015 measured cos-displacements stepping
    # outside the support of MultiScaleGRW's truncated scales.
    points = Vector{Float64}[copy(vi[:])]
    for j in 1:n_extra_points
        push!(points, copy(DynamicPPL.VarInfo(Random.MersenneTwister(seed + j), tm)[:]))
    end

    worst_abs = 0.0
    worst_rel = 0.0
    base_deltas = Float64[]
    for point in points
        engine = f_model(point)
        base_value = f_base(point[1:n_base])
        isfinite(engine) && isfinite(base_value) || error(
            "G0: non-finite log density (model $engine, base $base_value)")
        vpoint = DynamicPPL.unflatten(vi, point)
        state = gms_model_state(tm, vpoint)
        reference = gss_reference_pillars(model, fs, state, vpoint)
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
    gss_spine_line_identity(spine, five_strike, feature_set; n_extra_points, seed) -> NamedTuple

G0c. The spine is the five-strike model restricted to `log φ_K = β · (K − pivot)`. At a spine
θ and the five-strike θ that shares every site and sets `log_φ = β .* offsets`, the two log
densities must differ by EXACTLY their shape priors:

    [spine(θ) − logpdf(β_prior, β)] − [five(θ₅) − Σ logpdf(Normal(0, shape_sd), log φ)] ≈ 0

The pair must agree on base, supremacy slot, weight, σ prior and feature; the function refuses
otherwise, since then the difference would measure the mismatch rather than the shape.
"""
function gss_spine_line_identity(spine::SpineAnchoredCountModel, five::MarketAnchoredCountModel,
                                 feature_set; n_extra_points::Int = 3, seed::Int = 20260917)
    p = spine.smile
    q = five.smile
    p isa MarketSmileSpinePillar && q isa MarketSmilePillar || error(
        "G0c needs a spine rung and a five-strike smile rung; got $(typeof(p)) and $(typeof(q))")
    string(spine.base) == string(five.base) || error("G0c: the two rungs have different base models")
    string(spine.supremacy) == string(five.supremacy) || error("G0c: the supremacy slots differ")
    p.weight == q.weight && string(p.σ_prior) == string(q.σ_prior) && p.feature == q.feature ||
        error("G0c: the smile pillars differ in weight, σ prior or feature, not only in shape")

    fs = first(feature_set)
    ts = GMS_PG.build_turing_model(spine, fs)
    t5 = GMS_PG.build_turing_model(five, fs)
    Random.seed!(seed)
    vs = DynamicPPL.VarInfo(ts)
    v5 = DynamicPPL.VarInfo(t5)
    names_s = gms_site_names(vs)
    names_5 = gms_site_names(v5)
    names_s[end] == "β_spine" && names_5[end] == "log_φ" && names_s[1:end-1] == names_5[1:end-1] ||
        error("G0c: site layouts are not [shared…, β_spine] and [shared…, log_φ]; got " *
              "$(names_s[max(1, end-2):end]) and $(names_5[max(1, end-2):end])")
    n_shared = length(vs[:]) - 1
    length(v5[:]) == n_shared + gms_n_strikes(q) || error("G0c: log_φ is not $(gms_n_strikes(q)) values")

    offsets = gss_offsets(p)
    f_s = gss_log_density(ts)
    f_5 = gss_log_density(t5)
    points = Vector{Float64}[copy(vs[:])]
    for j in 1:n_extra_points
        push!(points, copy(DynamicPPL.VarInfo(Random.MersenneTwister(seed + j), ts)[:]))
    end

    worst_abs = 0.0
    worst_rel = 0.0
    for point in points
        β = point[end]
        log_φ = β .* offsets
        spine_value = f_s(point)
        five_value = f_5(vcat(point[1:n_shared], log_φ))
        isfinite(spine_value) && isfinite(five_value) || error(
            "G0c: non-finite log density (spine $spine_value, five-strike $five_value)")
        spine_likelihood = spine_value - logpdf(p.β_prior, β)
        five_likelihood = five_value - sum(logpdf.(Normal(0.0, q.shape_sd), log_φ))
        gap = spine_likelihood - five_likelihood
        worst_abs = max(worst_abs, abs(gap))
        worst_rel = max(worst_rel, abs(gap) / max(abs(spine_value), 1.0))
    end
    return (; n_points = length(points), n_shared_sites = n_shared, worst_abs, worst_rel)
end

"Per fold: how many training matches each pillar reads. Task 015's table, for either wrapper."
function gss_market_coverage(model, inputs)
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

# ==============================================================================
# 8. Anti-diagonal reweighting (ticket T011)
# ==============================================================================

"Per-draw scratch for `gss_reweight_grid!`: one mass and one ratio per anti-diagonal."
struct GSSReweightWorkspace
    grid_mass::Vector{Float64}
    ratio::Vector{Float64}
    max_goals::Int
end

function GSSReweightWorkspace(max_goals::Integer = GMS_PRED.TPL_MAX_GOALS)
    n_diagonals = 2 * Int(max_goals) - 1
    return GSSReweightWorkspace(zeros(n_diagonals), zeros(n_diagonals), Int(max_goals))
end

"""
    gss_reweight_grid!(S, λ_tot, φ, rw; identity_shortcut = true) -> Int

Rescale every draw `k` of the `max_goals × max_goals × n_draws` grid `S` in place so that its
total-goals marginal is the smile's: `P(total ≤ K) = cdf(Poisson(λ_tot[k] · φ[K+1, k]), K)` for
K = 0…n_strikes−1, the remainder spread over the higher anti-diagonals in the grid's own
proportions (header, choice 2). `φ` is `n_strikes × n_draws`, the `BookWorkspace` buffer shape.

Within an anti-diagonal every cell is multiplied by the same factor, so the grid's conditional
split of a total into scorelines — and hence the 1X2/BTTS structure given the total — is kept.

Returns the number of draws left untouched because their φ is exactly 1 at every strike.
Errors, naming the draw, if a draw's smile curve is not monotone in K.
"""
function gss_reweight_grid!(S::Array{Float64,3}, λ_tot::AbstractVector{Float64},
                            φ::AbstractMatrix{Float64}, rw::GSSReweightWorkspace;
                            identity_shortcut::Bool = true)
    M, M2, nd = size(S)
    M == M2 == rw.max_goals || error(
        "grid is $(size(S)); the reweighting workspace is for $(rw.max_goals) × $(rw.max_goals)")
    n_strikes = size(φ, 1)
    size(φ, 2) == nd && length(λ_tot) == nd || error(
        "φ is $(size(φ)) and λ_tot has $(length(λ_tot)); both must cover the grid's $nd draws")
    1 <= n_strikes < 2M - 1 || error(
        "$n_strikes strikes leave no anti-diagonal above Kmax in a $M × $M grid")

    n_identity = 0
    for k in 1:nd
        if identity_shortcut && _gss_is_identity(φ, k)
            n_identity += 1
            continue
        end
        _gss_reweight_draw!(S, k, λ_tot[k], φ, rw)
    end
    return n_identity
end

function _gss_is_identity(φ::AbstractMatrix{Float64}, k::Int)
    for s in axes(φ, 1)
        φ[s, k] == 1.0 || return false
    end
    return true
end

function _gss_reweight_draw!(S::Array{Float64,3}, k::Int, λ_tot::Float64,
                             φ::AbstractMatrix{Float64}, rw::GSSReweightWorkspace)
    M = rw.max_goals
    mass = rw.grid_mass
    ratio = rw.ratio
    n_strikes = size(φ, 1)

    fill!(mass, 0.0)
    for c in 1:M, r in 1:M
        mass[r + c - 1] += S[r, c, k]          # anti-diagonal index G + 1, G = (r−1) + (c−1)
    end

    F_prev = 0.0
    for s in 1:n_strikes
        K = s - 1
        # λ_tot · φ in the smile kernel's operand order (kernels.jl `price_market!`), so F is the
        # very Float64 the reported O/U price was computed from.
        F = cdf(Poisson(λ_tot * φ[s, k]), K)
        target = F - F_prev
        target >= 0.0 || error(
            "anti-diagonal reweighting: the smile curve is not a CDF on draw $k — " *
            "P(total ≤ $K) = $F < P(total ≤ $(K - 1)) = $F_prev at λ_tot = $λ_tot, " *
            "φ[$(K + 1)] = $(φ[s, k]). A per-strike smile defines a totals distribution only " *
            "where it is monotone; the container is refused rather than clipped.")
        mass[s] > 0.0 || error(
            "anti-diagonal reweighting: the grid holds zero mass on total $K at draw $k " *
            "(λ_tot = $λ_tot) and cannot be rescaled to $target")
        ratio[s] = target / mass[s]
        F_prev = F
    end

    grid_tail = 0.0
    for g in (n_strikes + 1):length(mass)
        grid_tail += mass[g]
    end
    tail = 1.0 - F_prev
    tail_ratio = 0.0
    if grid_tail > 0.0
        tail_ratio = tail / grid_tail
    elseif tail > 0.0
        error("anti-diagonal reweighting: the grid holds no mass above total $(n_strikes - 1) at " *
              "draw $k but the smile puts $tail there")
    end
    for g in (n_strikes + 1):length(ratio)
        ratio[g] = tail_ratio
    end

    for c in 1:M, r in 1:M
        S[r, c, k] *= ratio[r + c - 1]
    end
    return nothing
end

"""
    gss_price_fixture_reweighted!(w, rw, latents, i) -> nothing

`Portfolio.price_fixture!` with the reweighting inserted between the grid and the market books:
grid → smile buffers → reweight → price. 1X2 and BTTS books are therefore read off the
reweighted grid too, so every `p_model` in the ledger comes from the distribution the stake is
solved on; O/U within the strike ladder still reads the smile curve, which the reweighted grid
reproduces per draw.
"""
function gss_price_fixture_reweighted!(w::GSS_PORTFOLIO.BookWorkspace{GMS_PRED.SmileScoreGrid},
                                       rw::GSSReweightWorkspace, l::SmileLatents, i::Int)
    GMS_PRED.compute_score_grid!(w.S, w.ws, l, i)
    GMS_PRED.fill_smile_buffers!(w.λ_tot, w.φ, l, i)
    gss_reweight_grid!(w.S, w.λ_tot, w.φ, rw)
    GSS_PORTFOLIO._price_slots!(w.slots_1x2, w.grid)
    GSS_PORTFOLIO._price_slots!(w.slots_btts, w.grid)
    GSS_PORTFOLIO._price_slots!(w.slots_ou, w.grid)
    return nothing
end

"""
    gss_build_books_reweighted(spec, latents_or_fit, odds, fixtures; ...) -> (books, report)

`Portfolio.build_books_reported` with stakes solved off the anti-diagonal-reweighted grid.

* `SmileLatents` — the loop of `pricing.jl` §6 with `gss_price_fixture_reweighted!` in place of
  `price_fixture!`. `_finish_book` is called unchanged, so the mean grid it hands the Kelly solve
  and the per-draw grid BakerMcHale re-solves on are both the reweighted one.
* `CountLatents` — delegated to `build_books_reported` untouched (T011: count portfolios unchanged).
* `Fit` — the convergence verdict exactly as `pricing.jl` §7 records it, then the latents.

ONE DELIBERATE DEPARTURE: pricing and reweighting run OUTSIDE the per-fixture `try`. The
production builder turns any per-fixture error into a skipped fixture; a smile curve that is not
a CDF is a property of the container, and a portfolio that silently dropped those fixtures would
be a biased sample of the panel.
"""
function gss_build_books_reweighted(spec, l::SmileLatents, odds, fixtures;
                                    require_result::Bool = true,
                                    max_goals::Integer = GMS_PRED.TPL_MAX_GOALS,
                                    converged::Union{Nothing,Bool} = nothing,
                                    failed_gates::Vector{String} = String[],
                                    gated::Bool = false,
                                    quiet::Bool = false)
    t0 = time()
    oi = GSS_PORTFOLIO.build_odds_index(odds)
    fxs = GSS_PORTFOLIO.fixture_table(fixtures)
    w = GSS_PORTFOLIO.BookWorkspace(spec, l; max_goals = max_goals, quiet = quiet)
    rw = GSSReweightWorkspace(max_goals)

    n = n_matches(l)
    books = GSS_PORTFOLIO.MatchBook[]
    sizehint!(books, n)
    no_fixture = Int[]
    unplayed = Int[]
    no_quotes = Int[]
    no_sels = Int[]
    errored = Pair{Int,String}[]
    fb_needed = !isempty(w.slots_fb)
    no_fallback = Dict{String,Dict{Symbol,Vector{Float64}}}()

    for i in 1:n
        m_id = Int(l.match_ids[i])
        if !haskey(fxs, m_id)
            push!(no_fixture, m_id)
            continue
        end
        fx = fxs[m_id]
        if require_result && fx.score === nothing
            push!(unplayed, m_id)
            continue
        end
        if !haskey(oi.rows, m_id)
            push!(no_quotes, m_id)
            continue
        end

        gss_price_fixture_reweighted!(w, rw, l, i)

        local book
        try
            sels = GSS_PORTFOLIO.extract_selections(w, oi, m_id, spec,
                                                    fb_needed ? GSS_PORTFOLIO.fallback_probs(w) : no_fallback)
            if isempty(sels)
                push!(no_sels, m_id)
                continue
            end
            book = GSS_PORTFOLIO._finish_book(spec, w, m_id, fx, sels)
        catch err
            push!(errored, m_id => sprint(showerror, err))
            continue
        end
        push!(books, book)
    end

    sort!(books, by = b -> (b.date, b.m_id))
    report = GSS_PORTFOLIO.BuildReport(n, length(books), no_fixture, unplayed, no_quotes, no_sels,
                                       errored, GSS_PORTFOLIO.fallback_market_names(w), converged,
                                       failed_gates, gated, time() - t0)
    return (books, report)
end

gss_build_books_reweighted(spec, l::CountLatents, odds, fixtures; kw...) =
    GSS_PORTFOLIO.build_books_reported(spec, l, odds, fixtures; kw...)

function gss_build_books_reweighted(spec, fit::GSS_TRAINING.Fit, odds, fixtures;
                                    require_result::Bool = true,
                                    require_converged::Bool = false,
                                    max_goals::Integer = GMS_PRED.TPL_MAX_GOALS,
                                    quiet::Bool = false)
    passed, gates, detail = GSS_EVALUATION.convergence_verdict(fit)
    if require_converged && !passed
        throw(GSS_EVALUATION.ConvergenceRefusal(GSS_TRAINING.fit_name(fit), gates,
              vcat(detail, ["Refusing to build a staking book on this posterior."])))
    end
    passed || quiet || @warn("building reweighted books on a fit that did NOT converge",
                             fit = GSS_TRAINING.fit_name(fit), failed_gates = gates)
    return gss_build_books_reweighted(spec, GSS_EVALUATION.fit_latents(fit), odds, fixtures;
                                      require_result = require_result, max_goals = max_goals,
                                      converged = passed, failed_gates = gates,
                                      gated = require_converged, quiet = quiet)
end

# ==============================================================================
# 9. Reweighting gates (G4)
# ==============================================================================

"""
    gss_synthetic_spine_latents(; n_fixtures, n_draws, β_mean, β_sd, seed) -> SmileLatents

A deterministic spine container for G4a, checked before any sampling. Rates span 0.3–4.0 per
side so the reweighting is exercised from 0.6 to 8 expected goals.
"""
function gss_synthetic_spine_latents(; n_fixtures::Int = 6, n_draws::Int = 200,
                                     β_mean::Float64 = 0.05, β_sd::Float64 = 0.02,
                                     seed::Int = 20260913)
    rng = Random.MersenneTwister(seed)
    λ_h = 0.3 .+ 3.7 .* rand(rng, n_fixtures, n_draws)
    λ_a = 0.3 .+ 3.7 .* rand(rng, n_fixtures, n_draws)
    β = β_mean .+ β_sd .* randn(rng, n_draws)
    offsets = gss_offsets(MarketSmileSpinePillar())
    φ = Array{Float64,3}(undef, n_fixtures, length(offsets), n_draws)
    for i in 1:n_fixtures, (s, offset) in enumerate(offsets), k in 1:n_draws
        φ[i, s, k] = exp(β[k] * offset)
    end
    strikes = [(s - 1) + 0.5 for s in eachindex(offsets)]
    return SmileLatents(collect(1:n_fixtures), λ_h, λ_a, nothing, λ_h .+ λ_a, φ, strikes)
end

"""
    gss_reweight_gate(latents; fixtures, max_goals) -> DataFrame

Per fixture, the reweighted grid checked against code paths it does not share:

* `max_draw_cdf_gap`, `max_mean_cdf_gap` — the PRODUCTION grid O/U kernel on the reweighted grid
  vs the PRODUCTION smile O/U kernel, lines 0.5…4.5, per draw and in the mean.
* `max_mass_gap` — |Σ cells − 1| per draw; `min_cell` — no negative mass.
* `max_diag_spread` — (max − min)/max of reweighted ÷ original over each anti-diagonal: the
  conditional scoreline split is kept.
* `Δp_home/draw/away`, `Δp_under25` — mean shift against the plain grid, for the report.
"""
function gss_reweight_gate(l::SmileLatents; fixtures = 1:n_matches(l),
                           max_goals::Int = GMS_PRED.TPL_MAX_GOALS)
    nd = n_draws(l)
    nK = length(l.strikes)
    M = max_goals
    ws = GMS_PRED.GridWorkspace(M)
    S = GMS_PRED.alloc_score_grid(l, M)
    S0 = similar(S)
    λ_tot = Vector{Float64}(undef, nd)
    φ = Matrix{Float64}(undef, nK, nd)
    rw = GSSReweightWorkspace(M)
    m1x2 = Data.Market1X2()
    key_home, key_draw, key_away = GMS_PRED.market_keys(m1x2)

    rows = NamedTuple[]
    for i in fixtures
        GMS_PRED.compute_score_grid!(S, ws, l, i)
        copyto!(S0, S)
        GMS_PRED.fill_smile_buffers!(λ_tot, φ, l, i)
        n_identity = gss_reweight_grid!(S, λ_tot, φ, rw)
        smile = GMS_PRED.compute_score_grid(l, i; max_goals = M)

        draw_gap = 0.0
        mean_gap = 0.0
        under25_shift = NaN
        for s in 1:nK
            market = Data.MarketOverUnder((s - 1) + 0.5)
            _, key_under = GMS_PRED.market_keys(market)
            under_reweighted = GMS_PRED.price_market(S, market)[key_under]
            under_smile = GMS_PRED.price_market(smile, market)[key_under]
            draw_gap = max(draw_gap, maximum(abs.(under_reweighted .- under_smile)))
            mean_gap = max(mean_gap, abs(mean(under_reweighted) - mean(under_smile)))
            if s == 3
                under_grid = GMS_PRED.price_market(S0, market)[key_under]
                under25_shift = mean(under_reweighted) - mean(under_grid)
            end
        end

        mass_gap = maximum(abs(sum(view(S, :, :, k)) - 1.0) for k in 1:nd)

        spread = 0.0
        for k in 1:nd, G in 0:(2M - 2)
            lo = Inf
            hi = -Inf
            for r in max(1, G + 2 - M):min(M, G + 1)
                c = G + 2 - r
                S0[r, c, k] > 0.0 || continue
                q = S[r, c, k] / S0[r, c, k]
                lo = min(lo, q)
                hi = max(hi, q)
            end
            hi > 0.0 && (spread = max(spread, (hi - lo) / hi))
        end

        p_new = GMS_PRED.price_market(S, m1x2)
        p_old = GMS_PRED.price_market(S0, m1x2)
        push!(rows, (; fixture = l.match_ids[i], n_identity,
                       max_draw_cdf_gap = draw_gap, max_mean_cdf_gap = mean_gap,
                       max_mass_gap = mass_gap, min_cell = minimum(S),
                       max_diag_spread = spread,
                       Δp_home = mean(p_new[key_home]) - mean(p_old[key_home]),
                       Δp_draw = mean(p_new[key_draw]) - mean(p_old[key_draw]),
                       Δp_away = mean(p_new[key_away]) - mean(p_old[key_away]),
                       Δp_under25 = under25_shift))
    end
    return DataFrame(rows)
end

"The G4 verdict on a `gss_reweight_gate` frame, as a list of failure strings (empty = pass)."
function gss_reweight_failures(df::AbstractDataFrame; tol::Float64 = 1.0e-9,
                               spread_tol::Float64 = 1.0e-12)
    failures = String[]
    worst(col) = maximum(df[!, col])
    worst(:max_draw_cdf_gap) <= tol || push!(failures,
        @sprintf("reweighted totals ≠ smile CDF per draw (%.2e > %.0e)", worst(:max_draw_cdf_gap), tol))
    worst(:max_mean_cdf_gap) <= tol || push!(failures,
        @sprintf("reweighted totals ≠ smile CDF in the mean (%.2e > %.0e)", worst(:max_mean_cdf_gap), tol))
    worst(:max_mass_gap) <= tol || push!(failures,
        @sprintf("a reweighted draw does not sum to 1 (%.2e)", worst(:max_mass_gap)))
    minimum(df.min_cell) >= 0.0 || push!(failures, "negative reweighted mass")
    worst(:max_diag_spread) <= spread_tol || push!(failures,
        @sprintf("an anti-diagonal was not rescaled uniformly (spread %.2e)", worst(:max_diag_spread)))
    return failures
end

"""
    gss_identity_path_gate(latents; fixtures, max_goals) -> NamedTuple

The φ ≡ 1 case, both ways: through the shortcut the grid must be bit-identical and every draw
counted as identity; with the shortcut disabled the arithmetic must land within `forced_max_abs_gap`
of the plain grid (the residual is the grid's own truncation mass, redistributed).
"""
function gss_identity_path_gate(l::SmileLatents; fixtures = 1:n_matches(l),
                                max_goals::Int = GMS_PRED.TPL_MAX_GOALS)
    nd = n_draws(l)
    ws = GMS_PRED.GridWorkspace(max_goals)
    S = GMS_PRED.alloc_score_grid(l, max_goals)
    S0 = similar(S)
    ones_φ = ones(length(l.strikes), nd)
    λ_tot = Vector{Float64}(undef, nd)
    rw = GSSReweightWorkspace(max_goals)
    shortcut_identical = true
    forced_gap = 0.0
    for i in fixtures
        GMS_PRED.compute_score_grid!(S0, ws, l, i)
        λ_tot .= view(l.λ_tot, i, :)
        copyto!(S, S0)
        n_identity = gss_reweight_grid!(S, λ_tot, ones_φ, rw)
        shortcut_identical &= (n_identity == nd) && (S == S0)
        copyto!(S, S0)
        gss_reweight_grid!(S, λ_tot, ones_φ, rw; identity_shortcut = false)
        forced_gap = max(forced_gap, maximum(abs.(S .- S0)))
    end
    return (; shortcut_bit_identical = shortcut_identical, forced_max_abs_gap = forced_gap)
end

"""
    gss_refusal_gate() -> NamedTuple

A curve that is not a CDF must be refused. φ = (1, 3, 1, 1, 1) at λ_tot = 2 gives
P(total ≤ 1) = cdf(Poisson(6), 1) ≈ 0.017 < P(total ≤ 0) = e⁻² ≈ 0.135.
"""
function gss_refusal_gate(; max_goals::Int = GMS_PRED.TPL_MAX_GOALS)
    λ = fill(1.0, 1, 4)
    φ = ones(1, 5, 4)
    φ[1, 2, :] .= 3.0
    l = SmileLatents([1], λ, λ, nothing, λ .+ λ, φ, [0.5, 1.5, 2.5, 3.5, 4.5])
    S = GMS_PRED.compute_score_grid(CountLatents([1], λ, λ, nothing), 1; max_goals = max_goals)
    λ_tot = vec(l.λ_tot[1, :])
    φ_buf = Matrix{Float64}(l.φ[1, :, :])           # n_strikes × n_draws, the buffer shape
    message = ""
    refused = try
        gss_reweight_grid!(S, λ_tot, φ_buf, GSSReweightWorkspace(max_goals))
        false
    catch err
        message = sprint(showerror, err)
        occursin("not a CDF", message)
    end
    return (; refused, message)
end

"""
    gss_identity_ledger_gate(spec, latents, odds, fixtures) -> NamedTuple

T011 acceptance: a container with φ ≡ 1, staked through the reweighted builder, must produce the
bit-identical ledger — fixture order, score grid, payoff matrix, Kelly vector, shrink factor — of
its `CountLatents` twin staked through the production builder.
"""
function gss_identity_ledger_gate(spec, l::SmileLatents, odds, fixtures)
    flat = SmileLatents(l.match_ids, l.λ_home, l.λ_away, nothing, l.λ_tot, ones(size(l.φ)), l.strikes)
    twin = CountLatents(l.match_ids, l.λ_home, l.λ_away, nothing)
    smile_books, _ = gss_build_books_reweighted(spec, flat, odds, fixtures; quiet = true)
    grid_books, _ = GSS_PORTFOLIO.build_books_reported(spec, twin, odds, fixtures; quiet = true)
    n_identical = 0
    max_stake_gap = 0.0
    for (a, b) in zip(smile_books, grid_books)
        same = a.m_id == b.m_id && a.p_grid == b.p_grid && a.R == b.R &&
               a.a_kelly == b.a_kelly && a.k_shrink == b.k_shrink
        n_identical += same
        if a.m_id == b.m_id && length(a.a_kelly) == length(b.a_kelly)
            max_stake_gap = max(max_stake_gap, maximum(abs.(a.a_kelly .- b.a_kelly); init = 0.0))
        end
    end
    return (; n_books_flat = length(smile_books), n_books_grid = length(grid_books),
              n_identical, max_flat_stake_gap = max_stake_gap)
end

"""
    gss_staking_gate(spec, latents, odds, fixtures) -> NamedTuple

T011 acceptance on the real container: every staked book's `p_grid` — the distribution the
Kelly solve reads — implies `P(total ≤ K) = mean_k cdf(Poisson(λ_tot·φ_K), K)` for K = 0…4, and
the count of fixtures whose stake vector now differs from the production (plain-grid) builder's.
Task 015 measured that count at zero; above zero is φ reaching the stake.
"""
function gss_staking_gate(spec, l::SmileLatents, odds, fixtures)
    books, report = gss_build_books_reweighted(spec, l, odds, fixtures; quiet = true)
    plain, _ = GSS_PORTFOLIO.build_books_reported(spec, l, odds, fixtures; quiet = true)
    row_of = Dict(id => i for (i, id) in enumerate(l.match_ids))
    nd = n_draws(l)
    worst = 0.0
    for b in books
        i = row_of[b.m_id]
        M = isqrt(length(b.p_grid))
        P = reshape(b.p_grid, M, M)
        for s in eachindex(l.strikes)
            K = s - 1
            implied = sum(P[r, c] for c in 1:M for r in 1:M if (r - 1) + (c - 1) <= K)
            reference = mean(cdf(Poisson(l.λ_tot[i, k] * l.φ[i, s, k]), K) for k in 1:nd)
            worst = max(worst, abs(implied - reference))
        end
    end
    plain_by = Dict(b.m_id => b for b in plain)
    n_changed = count(b -> haskey(plain_by, b.m_id) && b.a_kelly != plain_by[b.m_id].a_kelly, books)
    return (; n_books = length(books), n_plain_books = length(plain),
              n_skipped = GSS_PORTFOLIO.n_skipped(report),
              max_totals_gap = worst, n_stake_changed = n_changed)
end

# ==============================================================================
# 10. Posterior summaries
# ==============================================================================

"""
    gss_task015_line_fit(; offsets) -> NamedTuple

The least-squares slope through Task 015's pooled log φ medians with the pivot held at zero,
and the residual per strike — the reference H2's "β ≈ 0.03–0.05" is read against.
"""
function gss_task015_line_fit(; offsets = gss_offsets(MarketSmileSpinePillar()))
    log_φ = log.(GSS_TASK015_PHI_MEDIAN)
    β = sum(offsets .* log_φ) / sum(offsets .^ 2)
    return (; β_least_squares = β, residuals = log_φ .- β .* offsets)
end

"Per fold: the β_spine posterior, its convergence, and the φ curve at its median."
function gss_beta_by_fold(fit)
    rows = NamedTuple[]
    offsets = gss_offsets(MarketSmileSpinePillar())
    for f in fit.folds
        :β_spine in names(f.chain) || continue
        draws = vec(Array(f.chain[:β_spine]))
        rh = DataFrame(MCMCChains.rhat(f.chain))
        eb = DataFrame(MCMCChains.ess(f.chain; kind = :bulk))
        et = DataFrame(MCMCChains.ess(f.chain; kind = :tail))
        j = findfirst(==(:β_spine), rh.parameters)
        β_median = median(draws)
        push!(rows, (; fold = f.fold, β_median, β_q05 = quantile(draws, 0.05),
                       β_q95 = quantile(draws, 0.95), β_sd = std(draws),
                       φ_at_median = join(round.(exp.(β_median .* offsets); digits = 3), "/"),
                       rhat = rh.rhat[j], ess_bulk = eb.ess[j], ess_tail = et.ess[j]))
    end
    return DataFrame(rows)
end

"Pillar posteriors pooled over folds, for either wrapper type."
function gss_pillar_summary(fit)
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
    β = pooled(:β_spine)
    φ_string = if !isempty(β)
        join([gph_num(exp(q(β, 0.5) * o); digits = 3) for o in gss_offsets(MarketSmileSpinePillar())], "/")
    else
        φ = [exp.(pooled(Symbol("log_φ[$k]"))) for k in 1:5]
        all(isempty, φ) ? "" : join([gph_num(q(v, 0.5); digits = 3) for v in φ], "/")
    end
    return (; σ_sup_median = q(σ_sup, 0.5), σ_smile_median = q(σ_smile, 0.5),
              σ_smile_q05 = q(σ_smile, 0.05), σ_smile_q95 = q(σ_smile, 0.95),
              κ_median = q(κ, 0.5),
              β_median = q(β, 0.5), β_q05 = q(β, 0.05), β_q95 = q(β, 0.95),
              β_sd = isempty(β) ? NaN : std(β),
              φ_median = φ_string)
end

"One convergence row: Task 013's columns plus the pooled pillar posteriors."
function gss_convergence_row(name::AbstractString, fit, c::GMSConfig; run_id = nothing)
    row = gph_convergence_row(name, fit, gms_gph_config(c); run_id)
    return (; row..., gss_pillar_summary(fit)...)
end

# ==============================================================================
# 11. Recipes and registry
# ==============================================================================

function gss_fit_configs(c::GMSConfig, models, splitter, sampler; name_suffix::AbstractString = "")
    return Dict(name => FitConfig(
        name = name * name_suffix,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = gms_execution(c),
        tags = copy(GSS_TAGS),
        description = GSS_DESCRIPTIONS[name],
        save_dir = joinpath(c.save_root, name * name_suffix),
    ) for (name, model) in models)
end

"""
    gss_register!(db, models, splitter, sampler, configs) -> NamedTuple

Task 015's registration rule for both wrapper types: `save_model` refuses anything that is not
a builder model (T010 related finding), so a wrapper registers its base as `<name>__base_model`
and its complete recipe — pillars, weights, priors — as the `<name>_fit` entry.
"""
function gss_register!(db, models, splitter, sampler, configs)
    model_ids = Dict{String,Int}()
    for (name, model) in models
        if model isa Union{MarketAnchoredCountModel,SpineAnchoredCountModel}
            model_ids[name * "__base_model"] = save_model(db, name * "__base_model", model.base;
                description = "Base builder model of $name (pillars live in $(name)_fit).",
                tags = GSS_TAGS)
        else
            model_ids[name] = save_model(db, name, model;
                                         description = GSS_DESCRIPTIONS[name], tags = GSS_TAGS)
        end
        save_config(db, name * "_fit", configs[name];
                    description = GSS_DESCRIPTIONS[name] * " Task 016 recipe.", tags = GSS_TAGS)
    end
    splitter_id = save_splitter(db, "scottish_lower_grw_smile_spine_43fold", splitter;
        description = "Pooled 56/57, two history seasons, match-biweek walk-forward over 24/25, 25/26 and 26/27 to Fold 43.",
        tags = GSS_TAGS)
    sampler_name = @sprintf("queued_nuts_%dx%d_w%d_a%03d", sampler.n_chains, sampler.n_samples,
                            sampler.n_warmup, round(Int, 100 * sampler.accept_rate))
    sampler_id = save_sampler(db, sampler_name, sampler;
        description = "ReverseDiff queued NUTS: $(sampler.n_chains) chains, $(sampler.n_warmup) warmup, " *
                      "$(sampler.n_samples) retained, target acceptance $(sampler.accept_rate).",
        tags = GSS_TAGS)
    return (; model_ids, splitter_id, sampler_id)
end
