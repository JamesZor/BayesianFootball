# ==============================================================================
# l01_market_inverse_loader.jl — market-inverse state-space models (TODO 023)
# ==============================================================================
#
# Definitions only; `r01_market_inverse_runner.jl` is the notebook that uses them.
#
# ------------------------------------------------------------------------------
# THE OBSERVATION MODEL (DESIGN.md §2)
# ------------------------------------------------------------------------------
#
# Each accepted fixture contributes two observations, the log of the inverted
# Betfair close (λ_mkt_h, λ_mkt_a):
#
#     y_home = μ + γ_home + α[h, t] + β[a, t] + ε,     ε ~ N(0, σ_obs²)
#     y_away = μ          + α[a, t] + β[h, t] + ε
#
# with t the WEEK of kick-off (7-day steps from the Monday of the first fixture;
# the summer break is a run of weeks with no observations, not a gap in t).
#
# Identification (DESIGN §2.2) is by zero-centring, not by a reference team:
# the latent state is an UNCONSTRAINED walk x̃ and the ratings the likelihood
# sees are α = C x̃, C = I − 11ᵀ/N. So every innovation is per-team and
# independent — which is what lets the stochastic-volatility and regime arms
# factorise across teams — and the one direction the data cannot see (the
# common mode of x̃) is a harmless random walk the filter carries along.
#
# ------------------------------------------------------------------------------
# THE FOUR ARMS (DESIGN §3)  — x ∈ {α, β}, one path per team per component
# ------------------------------------------------------------------------------
#
#   GRW1         x̃_t = x̃_{t−1} + σ_c ω                          (σ_α, σ_β)
#   MomentumGRW  x̃_t = x̃_{t−1} + ṽ_{t−1} + σ_c ω,  ṽ_t = φ ṽ_{t−1} + σ_v η
#   StochVolGRW  x̃_t = x̃_{t−1} + exp(h_t) ω,  h_t = h̄_c + γ(h_{t−1} − h̄_c) + σ_h ξ
#   RegimeGRW    x̃_t = x̃_{t−1} + σ_{c,S_t} ω,  S_t ∈ {calm, turbulent} Markov(P),
#                σ_{c,2} = σ_{c,1}(1 + Δ)
#
# plus a STATIC control (σ_α = σ_β = 0: one rating per team for the whole panel),
# which is what "is there any dynamics at all" is measured against.
#
# ------------------------------------------------------------------------------
# INFERENCE — everything is exact conditional on the right thing
# ------------------------------------------------------------------------------
#
# Conditional on the per-team innovation variances D_t, the model is linear-
# Gaussian in s_t = [μ, γ_home, x̃α, x̃β (, ṽα, ṽβ)], so:
#
#   * the Kalman filter gives the EXACT marginal likelihood p(y | θ, D) with μ,
#     γ_home and every rating path integrated out analytically;
#   * forward-filtering backward-sampling (FFBS) draws the whole path exactly.
#
# GRW1 / Momentum / Static: D is a function of θ alone, so θ is sampled from
# its collapsed posterior p(θ | y) by coordinate-wise slice sampling on the
# Kalman likelihood; paths are FFBS draws at each retained θ.
#
# StochVol / Regime: partially-collapsed Gibbs (van Dyk & Park 2008), in the
# only valid order —
#     θ | aux, y         collapsed slice (paths integrated out)
#     s_{1:T} | θ, aux, y   FFBS
#     aux | s_{1:T}, θ     per-team: elliptical slice on whitened ξ (SV) /
#                          forward-filter backward-sample of S (regime)
#     P | S                conjugate Beta (regime only)
#
# One-step-ahead prediction for EVERY arm uses the pre-week predictive of the
# filter (obs in week t predicted from information strictly before week t):
# exact Kalman for the Gaussian arms, a Rao-Blackwellised particle filter for
# SV / regime (particles over the volatility state, one Kalman filter each).
# Setting the volatility process to a point mass makes the RBPF reproduce the
# Kalman filter — that is gate G3 in `mid_gates`.
# ==============================================================================

module MarketInverseDynamics

using DataFrames
using Dates, LinearAlgebra, Statistics, Random, Printf
import Distributions
import MCMCChains
import BayesianFootball
const MID_DATA = BayesianFootball.Data
const MID_CAL = BayesianFootball.Calibration

export MarketPanel, build_market_panel, closing_book,
       AbstractArm, StaticArm, GRW1, GRW1Break, MomentumGRW, StochVolGRW, RegimeGRW,
       arm_name, param_names, constrain, init_theta,
       kalman_loglik, run_filter, ffbs, smoothed_mean,
       fit_arm, ArmFit, onestep_predictions, mid_gates,
       draws_chains, convergence_table, team_paths, prediction_metrics,
       rbpf_predictions, innovation_diagnostics, anomaly_catalog, restrict_panel,
       median_theta, fitted_logrates, active_weeks, n_obs, n_teams, schedule

# ==============================================================================
# 1. Data: the inverted close as a weekly observation panel
# ==============================================================================

"""
    closing_book(ds) -> DataFrame

Betfair close as the 20-minute time-weighted average before kick-off, de-vigged
within (match, market, line). Identical to `fast_slow_grw/l01::fsg_closing_book`
and `calibration_generative_eda/l01`, so the targets are the ones every earlier
supremacy-slope number was measured against.
"""
function closing_book(ds)
    raw = MID_DATA.summarize_odds(ds.betfair_odds, MID_DATA.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(match_id = Int.(raw.match_id),
                     market_name = String.(raw.market_name),
                     market_line = Float64.(raw.market_line),
                     selection = Symbol.(raw.selection),
                     odds_close = Float64.(raw.odds))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(groupby(odds, [:match_id, :market_name, :market_line]),
               :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close)
    return odds
end

"""
    MarketPanel

The accepted inversions laid out for the filter.

| field | is |
|---|---|
| `matches` | one row per ACCEPTED fixture: ids, date, week, season, tournament, teams, λ_mkt |
| `refusals` | every panel fixture whose close did not invert, with its reason |
| `teams` | team names; index = position |
| `n_weeks` | length of the weekly grid |
| `week_start` | Monday of each week |
| `obs_*` | one entry per observation (2 per fixture), sorted by week |
| `week_ptr` | observations of week t are `week_ptr[t]:(week_ptr[t+1]-1)` |
"""
struct MarketPanel
    matches::DataFrame
    refusals::DataFrame
    teams::Vector{String}
    n_weeks::Int
    week_start::Vector{Date}
    obs_week::Vector{Int}
    obs_home::Vector{Float64}   # 1.0 for the home side's rate, 0.0 for the away side's
    obs_att::Vector{Int}        # team whose attack generates the rate
    obs_def::Vector{Int}        # team whose defence concedes it
    obs_y::Vector{Float64}      # log λ_mkt
    obs_match::Vector{Int}
    obs_season::Vector{String}
    week_ptr::Vector{Int}
end

n_teams(p::MarketPanel) = length(p.teams)
n_obs(p::MarketPanel) = length(p.obs_y)

"""
    build_market_panel(ds; seasons, tournaments, step_days = 7) -> (panel, book, frame)

Invert the close for every fixture of `seasons` × `tournaments`, keep the accepted
ones, and index them on a `step_days` calendar grid. `frame` is the full inversion
frame (accepted and refused), `book` the de-vigged close.
"""
function build_market_panel(ds; seasons = ["24/25", "25/26"], tournaments = [56, 57],
                            step_days::Int = 7)
    m = filter(r -> !ismissing(r.season) && r.season in seasons &&
                    Int(r.tournament_id) in tournaments, ds.matches)
    nrow(m) > 0 || error("build_market_panel: no fixtures for seasons $seasons, " *
                         "tournaments $tournaments")
    book = closing_book(ds)
    ids = Int.(m.match_id)
    frame = MID_CAL.inversion_frame(MID_CAL.invert_market_rates(book; match_ids = ids))

    meta = DataFrame(match_id = Int.(m.match_id), match_date = Date.(m.match_date),
                     season = String.(m.season), tournament_id = Int.(m.tournament_id),
                     home_team = String.(m.home_team), away_team = String.(m.away_team))
    full = innerjoin(meta, frame; on = :match_id)
    nrow(full) == nrow(meta) || error("inversion frame lost fixtures: $(nrow(full)) of $(nrow(meta))")
    refusals = select(filter(:accepted => !, full), :match_id, :match_date, :season,
                      :home_team, :away_team, :n_targets, :sse, :reason)
    acc = filter(:accepted => identity, full)

    teams = sort!(unique(vcat(meta.home_team, meta.away_team)))
    tix = Dict(t => i for (i, t) in enumerate(teams))
    d0 = minimum(meta.match_date)
    d0 = d0 - Day(dayofweek(d0) - 1)                     # Monday of the first week
    acc.week = [div(Dates.value(d - d0), step_days) + 1 for d in acc.match_date]
    sort!(acc, [:week, :match_date, :match_id])
    n_weeks = div(Dates.value(maximum(meta.match_date) - d0), step_days) + 1

    n = 2 * nrow(acc)
    ow = Vector{Int}(undef, n); oh = Vector{Float64}(undef, n)
    oa = Vector{Int}(undef, n); od = Vector{Int}(undef, n)
    oy = Vector{Float64}(undef, n); om = Vector{Int}(undef, n)
    os = Vector{String}(undef, n)
    for (j, r) in enumerate(eachrow(acc))
        h = tix[r.home_team]; a = tix[r.away_team]
        k = 2j - 1
        ow[k] = r.week; oh[k] = 1.0; oa[k] = h; od[k] = a; oy[k] = log(r.lambda_mkt_h)
        om[k] = r.match_id; os[k] = r.season
        ow[k+1] = r.week; oh[k+1] = 0.0; oa[k+1] = a; od[k+1] = h; oy[k+1] = log(r.lambda_mkt_a)
        om[k+1] = r.match_id; os[k+1] = r.season
    end
    ptr = zeros(Int, n_weeks + 1); ptr[1] = 1
    for w in ow
        ptr[w+1] += 1
    end
    cumsum!(ptr, ptr)
    all(issorted(ow)) || error("observations are not week-sorted")

    panel = MarketPanel(select(acc, :match_id, :match_date, :week, :season, :tournament_id,
                               :home_team, :away_team, :lambda_mkt_h, :lambda_mkt_a, :sse),
                        refusals, teams, n_weeks,
                        [d0 + Day(step_days * (t - 1)) for t in 1:n_weeks],
                        ow, oh, oa, od, oy, om, os, ptr)
    return panel, book, frame
end

"""
    restrict_panel(panel, keep::AbstractVector{Bool}) -> MarketPanel

Same teams and weekly grid, only the observations flagged in `keep`. Used to fit θ
on one season and filter across both (the honest out-of-sample protocol).
"""
function restrict_panel(p::MarketPanel, keep::AbstractVector{Bool})
    idx = findall(keep)
    ow = p.obs_week[idx]
    ptr = zeros(Int, p.n_weeks + 1); ptr[1] = 1
    for w in ow
        ptr[w+1] += 1
    end
    cumsum!(ptr, ptr)
    return MarketPanel(p.matches, p.refusals, p.teams, p.n_weeks, p.week_start, ow,
                       p.obs_home[idx], p.obs_att[idx], p.obs_def[idx], p.obs_y[idx],
                       p.obs_match[idx], p.obs_season[idx], ptr)
end

# ==============================================================================
# 2. Arms: parameter vectors, transforms, priors
# ==============================================================================
#
# θ is always an UNCONSTRAINED vector; `constrain` maps it to a NamedTuple and
# `log_prior` includes the Jacobian of that map, so the slice sampler targets the
# right density on the unconstrained scale.

abstract type AbstractArm end
"Static ratings: σ_α = σ_β = 0. θ = (log σ_obs)."
struct StaticArm <: AbstractArm end
"1st-order GRW. θ = (log σ_obs, log σ_α, log σ_β)."
struct GRW1 <: AbstractArm end
"""
GRW1 plus a one-off jump at the season boundary: every team's α and β get an extra
N(0, σ_break²) innovation in the first week of each new season in the panel.
θ = (log σ_obs, log σ_att, log σ_def, log σ_break). A CONTROL, not one of the four
DESIGN arms: it asks how much of any "volatility clustering" is simply the summer.
"""
struct GRW1Break <: AbstractArm end
"Damped-velocity GRW. θ = (log σ_obs, log σ_α, log σ_β, logit φ, log σ_v)."
struct MomentumGRW <: AbstractArm end
"Stochastic-volatility GRW. θ = (log σ_obs, h̄_α, h̄_β, logit γ, log σ_h); aux = whitened ξ."
struct StochVolGRW <: AbstractArm end
"2-state regime GRW. θ = (log σ_obs, log σ1_α, log σ1_β, log Δ); aux = S; P conjugate."
struct RegimeGRW <: AbstractArm end

arm_name(::StaticArm) = "a0_static"
arm_name(::GRW1) = "a1_grw1"
arm_name(::GRW1Break) = "a1b_grw1_break"
arm_name(::MomentumGRW) = "a2_momentum"
arm_name(::StochVolGRW) = "a3_stochvol"
arm_name(::RegimeGRW) = "a4_regime"

param_names(::StaticArm) = ["sigma_obs"]
param_names(::GRW1) = ["sigma_obs", "sigma_att", "sigma_def"]
param_names(::GRW1Break) = ["sigma_obs", "sigma_att", "sigma_def", "sigma_break"]
param_names(::MomentumGRW) = ["sigma_obs", "sigma_att", "sigma_def", "phi", "sigma_v"]
param_names(::StochVolGRW) = ["sigma_obs", "hbar_att", "hbar_def", "gamma_h", "sigma_h"]
param_names(::RegimeGRW) = ["sigma_obs", "sigma1_att", "sigma1_def", "delta"]

logistic(x) = 1 / (1 + exp(-x))
logit(p) = log(p / (1 - p))

constrain(::StaticArm, θ) = (σ_obs = exp(θ[1]),)
constrain(::GRW1, θ) = (σ_obs = exp(θ[1]), σ_att = exp(θ[2]), σ_def = exp(θ[3]))
constrain(::GRW1Break, θ) = (σ_obs = exp(θ[1]), σ_att = exp(θ[2]), σ_def = exp(θ[3]),
                            σ_break = exp(θ[4]))
constrain(::MomentumGRW, θ) = (σ_obs = exp(θ[1]), σ_att = exp(θ[2]), σ_def = exp(θ[3]),
                              φ = logistic(θ[4]), σ_v = exp(θ[5]))
constrain(::StochVolGRW, θ) = (σ_obs = exp(θ[1]), hbar_att = θ[2], hbar_def = θ[3],
                              γ = logistic(θ[4]), σ_h = exp(θ[5]))
constrain(::RegimeGRW, θ) = (σ_obs = exp(θ[1]), σ1_att = exp(θ[2]), σ1_def = exp(θ[3]),
                            Δ = exp(θ[4]))

"""
    log_prior(arm, θ)

Log prior on the unconstrained scale (Jacobian included). Every prior is a
modelling decision; DESIGN §3 fixes the σ_x scale.

| parameter | prior | note |
|---|---|---|
| σ_obs | HalfNormal(0.20) | market log-rate pricing residual |
| σ_att, σ_def | HalfNormal(0.10) | DESIGN §3 Arm 1, per WEEK |
| φ | Beta(2, 2) | momentum persistence on (0, 1) |
| σ_v | HalfNormal(0.05) | velocity innovation per week |
| σ_break | HalfNormal(0.30) | season-boundary jump (control arm only) |
| h̄_att, h̄_def | Normal(log 0.03, 1.0) | log weekly σ; 0.03 ≈ the GRW1 scale |
| γ_h | Beta(8, 2) | SV persistence, mean 0.8 |
| σ_h | HalfNormal(0.50) | SV innovation |
| σ1_att, σ1_def | HalfNormal(0.10) | calm-regime scale (same as Arm 1) |
| Δ | Exponential(mean 2) | turbulent = calm × (1 + Δ) |
| p11, p22 | Beta(18, 2), Beta(8, 2) | calm lasts ~10 weeks, turbulence ~5 a priori |
"""
function log_prior end

halfnormal_lp(x, s) = -0.5 * (x / s)^2          # up to a constant, x > 0
normal_lp(x, m, s) = -0.5 * ((x - m) / s)^2
beta_lp(p, a, b) = (a - 1) * log(p) + (b - 1) * log1p(-p)

# log prior on the unconstrained scale = log p(constrained) + log |Jacobian|
function log_prior(::StaticArm, θ)
    s = exp(θ[1])
    return halfnormal_lp(s, 0.20) + θ[1]
end
function log_prior(::GRW1, θ)
    lp = halfnormal_lp(exp(θ[1]), 0.20) + θ[1]
    lp += halfnormal_lp(exp(θ[2]), 0.10) + θ[2]
    lp += halfnormal_lp(exp(θ[3]), 0.10) + θ[3]
    return lp
end
log_prior(::GRW1Break, θ) = log_prior(GRW1(), θ[1:3]) + halfnormal_lp(exp(θ[4]), 0.30) + θ[4]
function log_prior(::MomentumGRW, θ)
    lp = log_prior(GRW1(), θ)
    φ = logistic(θ[4])
    lp += beta_lp(φ, 2.0, 2.0) + log(φ) + log1p(-φ)
    lp += halfnormal_lp(exp(θ[5]), 0.05) + θ[5]
    return lp
end
function log_prior(::StochVolGRW, θ)
    lp = halfnormal_lp(exp(θ[1]), 0.20) + θ[1]
    lp += normal_lp(θ[2], log(0.03), 1.0) + normal_lp(θ[3], log(0.03), 1.0)
    γ = logistic(θ[4])
    lp += beta_lp(γ, 8.0, 2.0) + log(γ) + log1p(-γ)
    lp += halfnormal_lp(exp(θ[5]), 0.50) + θ[5]
    return lp
end
function log_prior(::RegimeGRW, θ)
    lp = log_prior(GRW1(), θ[1:3])
    lp += -exp(θ[4]) / 2.0 + θ[4]
    return lp
end

"Dispersed starting points for chain `c` (a jitter around a sensible centre)."
function init_theta(arm::AbstractArm, rng::AbstractRNG)
    centre = init_centre(arm)
    return centre .+ 0.5 .* randn(rng, length(centre))
end
init_centre(::StaticArm) = [log(0.1)]
init_centre(::GRW1) = [log(0.05), log(0.03), log(0.03)]
init_centre(::GRW1Break) = [log(0.05), log(0.03), log(0.03), log(0.1)]
init_centre(::MomentumGRW) = [log(0.05), log(0.03), log(0.03), 0.0, log(0.01)]
init_centre(::StochVolGRW) = [log(0.05), log(0.03), log(0.03), logit(0.8), log(0.3)]
init_centre(::RegimeGRW) = [log(0.05), log(0.02), log(0.02), log(1.0)]

const REGIME_P_PRIOR = ((18.0, 2.0), (8.0, 2.0))   # (p11 ~ Beta, p22 ~ Beta)
const REGIME_PI0 = (0.8, 0.2)                      # initial regime distribution (fixed)

# ==============================================================================
# 3. State-space layout and the Kalman filter
# ==============================================================================
#
#   s = [μ, γ_home, x̃α (N), x̃β (N)]                    n = 2 + 2N
#   s = [μ, γ_home, x̃α (N), x̃β (N), ṽα (N), ṽβ (N)]    n = 2 + 4N   (momentum)

const MU_PRIOR = (log(1.35), 0.5)       # μ ~ N(log 1.35, 0.5²)
const HOME_PRIOR = (0.15, 0.25)         # γ_home ~ N(0.15, 0.25²)
const X0_SD = 0.5                       # x̃_{i,1} ~ N(0, 0.25) — DESIGN's 0.25 read as a variance

state_dim(::AbstractArm, N) = 2 + 2N
state_dim(::MomentumGRW, N) = 2 + 4N
has_velocity(::AbstractArm) = false
has_velocity(::MomentumGRW) = true

"Initial state mean and covariance (week 1, before any observation)."
function initial_state(arm::AbstractArm, N)
    n = state_dim(arm, N)
    m = zeros(n)
    P = zeros(n, n)
    m[1] = MU_PRIOR[1]; P[1, 1] = MU_PRIOR[2]^2
    m[2] = HOME_PRIOR[1]; P[2, 2] = HOME_PRIOR[2]^2
    for i in 3:(2 + 2N)
        P[i, i] = X0_SD^2
    end
    # velocities start at exactly 0 (DESIGN §3 Arm 2 boundary condition)
    return m, P
end

"""
Observation row: h·s = μ + home·γ + (C x̃α)[att] + (C x̃β)[def]. Written into `h`.
"""
function obs_row!(h::Vector{Float64}, N::Int, home::Float64, att::Int, def::Int)
    fill!(h, 0.0)
    h[1] = 1.0
    h[2] = home
    invN = 1.0 / N
    @inbounds for i in 1:N
        h[2+i] = -invN
        h[2+N+i] = -invN
    end
    h[2+att] += 1.0
    h[2+N+def] += 1.0
    return h
end

"""
    InnovationSchedule

Per-week innovation variances for the rating block. `D[:, t]` holds the 2N
variances (α then β) of the step INTO week t (column 1 is unused). For momentum,
`vel_var` and `φ` add the velocity block.
"""
struct InnovationSchedule
    D::Matrix{Float64}      # 2N × T
    vel_var::Float64
    φ::Float64
end

"Schedule for arms whose variances are functions of θ alone."
function schedule(arm::StaticArm, θ, N, T)
    return InnovationSchedule(zeros(2N, T), 0.0, 0.0)
end
"Season-boundary weeks: the first observed week of every season after the panel's first."
season_break_weeks(p::MarketPanel) =
    sort!([minimum(p.obs_week[p.obs_season .== s]) for s in unique(p.obs_season)])[2:end]

function schedule(arm::GRW1Break, θ, N, T, breaks::AbstractVector{<:Integer})
    c = constrain(arm, θ)
    D = Matrix{Float64}(undef, 2N, T)
    D[1:N, :] .= c.σ_att^2
    D[N+1:2N, :] .= c.σ_def^2
    for t in breaks
        D[:, t] .+= c.σ_break^2
    end
    return InnovationSchedule(D, 0.0, 0.0)
end

function schedule(arm::Union{GRW1, MomentumGRW}, θ, N, T)
    c = constrain(arm, θ)
    D = Matrix{Float64}(undef, 2N, T)
    D[1:N, :] .= c.σ_att^2
    D[N+1:2N, :] .= c.σ_def^2
    arm isa MomentumGRW && return InnovationSchedule(D, c.σ_v^2, c.φ)
    return InnovationSchedule(D, 0.0, 0.0)
end

"""
SV log-volatility paths from whitened innovations ξ (2N × T; column 1 unused):
h_2 = h̄ + σ_h/√(1−γ²) ξ_2,  h_t = h̄ + γ(h_{t−1} − h̄) + σ_h ξ_t.
"""
function sv_logvol(c, ξ::AbstractMatrix, N::Int)
    T = size(ξ, 2)
    H = zeros(2N, T)
    sd0 = c.σ_h / sqrt(1 - c.γ^2)
    for i in 1:2N
        hb = i <= N ? c.hbar_att : c.hbar_def
        H[i, 2] = hb + sd0 * ξ[i, 2]
        for t in 3:T
            H[i, t] = hb + c.γ * (H[i, t-1] - hb) + c.σ_h * ξ[i, t]
        end
    end
    return H
end
function schedule(arm::StochVolGRW, θ, N, T, ξ::AbstractMatrix)
    H = sv_logvol(constrain(arm, θ), ξ, N)
    D = exp.(2 .* H)
    D[:, 1] .= 0.0
    return InnovationSchedule(D, 0.0, 0.0)
end
function schedule(arm::RegimeGRW, θ, N, T, S::AbstractMatrix{<:Integer})
    c = constrain(arm, θ)
    D = zeros(2N, T)
    for t in 2:T, i in 1:2N
        s1 = i <= N ? c.σ1_att : c.σ1_def
        σ = S[i, t] == 1 ? s1 : s1 * (1 + c.Δ)
        D[i, t] = σ^2
    end
    return InnovationSchedule(D, 0.0, 0.0)
end

"""
Kalman time update into week t (t ≥ 2), in place. F = I except the momentum arm,
where x̃ += ṽ and ṽ ← φṽ; `buf` is an n×n workspace used only for momentum.
"""
function time_update!(m, P, sch::InnovationSchedule, t::Int, N::Int, vel::Bool, buf)
    if vel
        xr = 3:(2+2N)
        vr = (3+2N):(2+4N)
        # m: x += v; v *= φ
        @views m[xr] .+= m[vr]
        @views m[vr] .*= sch.φ
        # P ← F P Fᵀ with F = [I E; 0 φI] on (s, v): done as two sparse passes
        copyto!(buf, P)
        @views buf[xr, :] .+= P[vr, :]          # rows:  F P
        @views buf[vr, :] .*= sch.φ
        copyto!(P, buf)
        @views P[:, xr] .+= buf[:, vr]          # cols: (F P) Fᵀ
        @views P[:, vr] .*= sch.φ
        for i in vr
            P[i, i] += sch.vel_var
        end
    end
    @inbounds for i in 1:2N
        P[2+i, 2+i] += sch.D[i, t]
    end
    return nothing
end

"""
    run_filter(arm, panel, sch; σ_obs, store = false, predict = false)
        -> (; loglik, pred_mean, pred_var, m_filt, P_filt)

Exact Kalman filter over the weekly grid with sequential scalar updates.

* `loglik` — log p(y | θ, D), μ, γ_home and every path integrated out.
* `predict` — also record, for every observation, the PRE-WEEK predictive mean
  and variance (information strictly before its week; same-week fixtures never
  inform each other).
* `store` — keep the filtered mean/covariance of every week (for FFBS / RTS).
"""
function run_filter(arm::AbstractArm, p::MarketPanel, sch::InnovationSchedule;
                    σ_obs::Float64, store::Bool = false, predict::Bool = false)
    N = n_teams(p)
    T = p.n_weeks
    vel = has_velocity(arm)
    m, P = initial_state(arm, N)
    n = length(m)
    h = zeros(n)
    k = zeros(n)
    buf = vel ? zeros(n, n) : zeros(0, 0)
    R = σ_obs^2
    ll = 0.0
    pm = predict ? fill(NaN, n_obs(p)) : Float64[]
    pv = predict ? fill(NaN, n_obs(p)) : Float64[]
    Ms = store ? zeros(n, T) : zeros(0, 0)
    Ps = store ? zeros(n, n, T) : zeros(0, 0, 0)
    for t in 1:T
        t >= 2 && time_update!(m, P, sch, t, N, vel, buf)
        r = p.week_ptr[t]:(p.week_ptr[t+1]-1)
        if predict
            for j in r
                obs_row!(h, N, p.obs_home[j], p.obs_att[j], p.obs_def[j])
                mul!(k, P, h)
                pm[j] = dot(h, m)
                pv[j] = dot(h, k) + R
            end
        end
        for j in r
            obs_row!(h, N, p.obs_home[j], p.obs_att[j], p.obs_def[j])
            mul!(k, P, h)
            S = dot(h, k) + R
            if !(S > 0.0 && isfinite(S))
                # only reachable at absurd θ (a slice bracket far in the tails): the
                # covariance has lost positive-definiteness, so the point has no density
                return (; loglik = -Inf, pred_mean = pm, pred_var = pv, m_filt = Ms, P_filt = Ps)
            end
            v = p.obs_y[j] - dot(h, m)
            ll += -0.5 * (log(2π * S) + v^2 / S)
            axpy!(v / S, k, m)
            BLAS.ger!(-1.0 / S, k, k, P)
        end
        if !isempty(r)
            @inbounds for j in 1:n, i in 1:(j-1)          # re-symmetrise once a week
                a = 0.5 * (P[i, j] + P[j, i])
                P[i, j] = a
                P[j, i] = a
            end
        end
        if store
            Ms[:, t] .= m
            Ps[:, :, t] .= P
        end
    end
    return (; loglik = ll, pred_mean = pm, pred_var = pv, m_filt = Ms, P_filt = Ps)
end

"Transition matrix F (dense) for the backward pass."
function transition_matrix(arm::AbstractArm, N, sch)
    n = state_dim(arm, N)
    F = Matrix{Float64}(I, n, n)
    if has_velocity(arm)
        for i in 1:2N
            F[2+i, 2+2N+i] = 1.0
            F[2+2N+i, 2+2N+i] = sch.φ
        end
    end
    return F
end
function process_cov(arm::AbstractArm, N, sch, t)
    n = state_dim(arm, N)
    Q = zeros(n, n)
    for i in 1:2N
        Q[2+i, 2+i] = sch.D[i, t]
    end
    if has_velocity(arm)
        for i in (3+2N):(2+4N)
            Q[i, i] = sch.vel_var
        end
    end
    return Q
end

"Draw from N(mean, cov) with cov PSD (pivoted Cholesky handles the static and v₁ = 0 directions)."
function draw_psd!(out, rng, mean, cov)
    C = cholesky(Symmetric(cov), RowMaximum(); check = false, tol = 1e-14)
    r = C.rank
    z = randn(rng, r)
    w = C.L[:, 1:r] * z
    out .= mean
    @inbounds for (a, i) in enumerate(C.p)
        out[i] += w[a]
    end
    return out
end

"""
    ffbs(arm, panel, sch, filt, rng) -> Matrix (n × T)

One exact draw of the state path given the stored filter output `filt`
(forward-filtering backward-sampling, Carter & Kohn 1994).
"""
function ffbs(arm::AbstractArm, p::MarketPanel, sch::InnovationSchedule, filt, rng::AbstractRNG)
    N = n_teams(p)
    T = p.n_weeks
    F = transition_matrix(arm, N, sch)
    n = size(filt.m_filt, 1)
    X = zeros(n, T)
    @views draw_psd!(X[:, T], rng, filt.m_filt[:, T], filt.P_filt[:, :, T])
    for t in (T-1):-1:1
        backward_step!(view(X, :, t), view(X, :, t + 1), arm, N, sch, filt, F, t, rng)
    end
    return X
end

function backward_moments(xnext, arm, N, sch, filt, F, t)
    Pt = filt.P_filt[:, :, t]
    mt = filt.m_filt[:, t]
    PF = Pt * F'
    Ppred = Symmetric(F * PF + process_cov(arm, N, sch, t + 1))
    Cp = cholesky(Ppred; check = false)
    J = issuccess(Cp) ? (Cp \ PF')' : PF * pinv(Matrix(Ppred))
    mean = mt .+ J * (xnext .- F * mt)
    cov = Pt .- J * PF'
    return mean, 0.5 .* (cov .+ cov')
end
function backward_step!(xt, xnext, arm, N, sch, filt, F, t, rng)
    mean, cov = backward_moments(xnext, arm, N, sch, filt, F, t)
    draw_psd!(xt, rng, mean, cov)
end

"Rauch–Tung–Striebel smoothed mean: the backward recursion with draws replaced by means."
function smoothed_mean(arm::AbstractArm, p::MarketPanel, sch::InnovationSchedule, filt)
    N = n_teams(p)
    T = p.n_weeks
    F = transition_matrix(arm, N, sch)
    X = copy(filt.m_filt)
    for t in (T-1):-1:1
        mean, _ = backward_moments(X[:, t+1], arm, N, sch, filt, F, t)
        X[:, t] .= mean
    end
    return X
end

"Collapsed log-likelihood p(y | θ, aux) — the target of every θ update."
kalman_loglik(arm::Union{StaticArm, GRW1, MomentumGRW}, p::MarketPanel, θ) =
    run_filter(arm, p, schedule(arm, θ, n_teams(p), p.n_weeks); σ_obs = exp(θ[1])).loglik
kalman_loglik(arm::GRW1Break, p::MarketPanel, θ) =
    run_filter(arm, p, schedule(arm, θ, n_teams(p), p.n_weeks, season_break_weeks(p));
               σ_obs = exp(θ[1])).loglik
kalman_loglik(arm::Union{StochVolGRW, RegimeGRW}, p::MarketPanel, θ, aux) =
    run_filter(arm, p, schedule(arm, θ, n_teams(p), p.n_weeks, aux); σ_obs = exp(θ[1])).loglik

# ==============================================================================
# 4. Samplers
# ==============================================================================

"""
Univariate slice sampler (Neal 2003, stepping-out + shrinkage) on coordinate `j`.
Returns the new θ and its log target. `w` is the initial bracket width.
"""
function slice_coordinate(logf, θ::Vector{Float64}, lf0::Float64, j::Int, w::Float64,
                          rng::AbstractRNG; max_steps::Int = 20)
    logy = lf0 + log(rand(rng))
    x0 = θ[j]
    L = x0 - w * rand(rng)
    R = L + w
    θt = copy(θ)
    fL = (θt[j] = L; logf(θt))
    steps = 0
    while fL > logy && steps < max_steps
        L -= w; θt[j] = L; fL = logf(θt); steps += 1
    end
    fR = (θt[j] = R; logf(θt))
    steps = 0
    while fR > logy && steps < max_steps
        R += w; θt[j] = R; fR = logf(θt); steps += 1
    end
    while true
        x1 = L + rand(rng) * (R - L)
        θt[j] = x1
        f1 = logf(θt)
        if f1 > logy
            return θt, f1
        end
        x1 < x0 ? (L = x1) : (R = x1)
        (R - L) < 1e-10 && return θ, lf0
    end
end

"One sweep of coordinate-wise slice updates of θ under `logf`."
function slice_sweep(logf, θ, lf, widths, rng)
    for j in eachindex(θ)
        θ, lf = slice_coordinate(logf, θ, lf, j, widths[j], rng)
    end
    return θ, lf
end

"""
Elliptical slice sampling (Murray, Adams & MacKay 2010) for x ~ N(0, I) prior
and log-likelihood `loglik`.
"""
function elliptical_slice(x::Vector{Float64}, llx::Float64, loglik, rng::AbstractRNG)
    ν = randn(rng, length(x))
    logy = llx + log(rand(rng))
    a = 2π * rand(rng)
    lo = a - 2π
    hi = a
    while true
        xn = x .* cos(a) .+ ν .* sin(a)
        l = loglik(xn)
        l > logy && return xn, l
        a < 0 ? (lo = a) : (hi = a)
        a = lo + rand(rng) * (hi - lo)
    end
end

"Rating innovations e_{i,t} = x̃_{i,t} − x̃_{i,t−1} (the raw step; for momentum it includes ṽ)."
function innovations(X::AbstractMatrix, N::Int)
    T = size(X, 2)
    E = zeros(2N, T)
    for t in 2:T, i in 1:2N
        E[i, t] = X[2+i, t] - X[2+i, t-1]
    end
    return E
end

"SV: log p(e_{i,2:T} | ξ_i) for one series (whitened ξ, θ fixed)."
function sv_series_loglik(e::AbstractVector, ξ::AbstractVector, hb, c)
    T = length(e)
    sd0 = c.σ_h / sqrt(1 - c.γ^2)
    h = hb + sd0 * ξ[2]
    ll = -h - 0.5 * e[2]^2 * exp(-2h)
    for t in 3:T
        h = hb + c.γ * (h - hb) + c.σ_h * ξ[t]
        ll += -h - 0.5 * e[t]^2 * exp(-2h)
    end
    return ll
end

"Regime: forward-filter backward-sample S_{2:T} for one series; returns (S, n_trans)."
function regime_ffbs!(S::AbstractVector{<:Integer}, e::AbstractVector, σ1, σ2, p11, p22, rng)
    T = length(e)
    α = zeros(2, T)
    P = [p11 1-p11; 1-p22 p22]
    lik(t) = (exp(-0.5 * (e[t] / σ1)^2) / σ1, exp(-0.5 * (e[t] / σ2)^2) / σ2)
    l1, l2 = lik(2)
    α[1, 2] = REGIME_PI0[1] * l1
    α[2, 2] = REGIME_PI0[2] * l2
    α[:, 2] ./= sum(α[:, 2])
    for t in 3:T
        l1, l2 = lik(t)
        a1 = (α[1, t-1] * P[1, 1] + α[2, t-1] * P[2, 1]) * l1
        a2 = (α[1, t-1] * P[1, 2] + α[2, t-1] * P[2, 2]) * l2
        z = a1 + a2
        α[1, t] = a1 / z
        α[2, t] = a2 / z
    end
    S[T] = rand(rng) < α[1, T] ? 1 : 2
    for t in (T-1):-1:2
        w1 = α[1, t] * P[1, S[t+1]]
        w2 = α[2, t] * P[2, S[t+1]]
        S[t] = rand(rng) < w1 / (w1 + w2) ? 1 : 2
    end
    return S
end

function regime_transition_counts(S::AbstractMatrix{<:Integer})
    n = zeros(Int, 2, 2)
    for i in axes(S, 1), t in 3:size(S, 2)
        n[S[i, t-1], S[i, t]] += 1
    end
    return n
end

"""
    ArmFit

Posterior output of one arm: θ draws (`iters × params × chains`, CONSTRAINED),
unconstrained draws, retained path draws (a thinned subset), auxiliary summaries,
and timing.
"""
struct ArmFit
    arm::AbstractArm
    names::Vector{String}
    draws::Array{Float64, 3}          # constrained
    udraws::Array{Float64, 3}         # unconstrained
    paths::Vector{Matrix{Float64}}    # FFBS state paths (n × T), thinned, all chains
    path_theta::Vector{Vector{Float64}}
    aux_mean::Matrix{Float64}         # SV: posterior mean h; regime: P(turbulent); else empty
    seconds::Float64
end

"""
    fit_arm(arm, panel; n_chains, n_warmup, n_samples, n_paths, seed) -> ArmFit

Chains run in parallel on threads (one chain per task, BLAS single-threaded).
Slice widths adapt during warm-up only (2 × running sd), then freeze.
"""
fit_arm(::RegimeGRW, p::MarketPanel; ess_steps::Int = 0, kwargs...) = fit_regime_with_p(p; kwargs...)

function fit_arm(arm::AbstractArm, p::MarketPanel; n_chains::Int = 4, n_warmup::Int = 1000,
                 n_samples::Int = 1000, n_paths::Int = 50, seed::Int = 20260922,
                 ess_steps::Int = 2, thin::Int = 1)
    t0 = time()
    names = param_names(arm)
    K = length(names)
    draws = zeros(n_samples, K, n_chains)
    udraws = zeros(n_samples, K, n_chains)
    per_chain = cld(n_paths, n_chains)
    paths = Vector{Vector{Matrix{Float64}}}(undef, n_chains)
    pth = Vector{Vector{Vector{Float64}}}(undef, n_chains)
    auxs = Vector{Matrix{Float64}}(undef, n_chains)
    tasks = map(1:n_chains) do c
        Threads.@spawn run_chain(arm, p, c, seed, n_warmup, n_samples, per_chain, ess_steps, thin)
    end
    for c in 1:n_chains
        out = fetch(tasks[c])
        udraws[:, :, c] .= out.U
        for s in 1:n_samples
            cc = constrain(arm, out.U[s, :])
            draws[s, :, c] .= collect(values(cc))
        end
        paths[c] = out.paths
        pth[c] = out.path_theta
        auxs[c] = out.aux
    end
    aux = isempty(auxs[1]) ? zeros(0, 0) : reduce(+, auxs) ./ n_chains
    return ArmFit(arm, names, draws, udraws, reduce(vcat, paths), reduce(vcat, pth), aux,
                  time() - t0)
end

"Retained-iteration indices for FFBS paths: `k` evenly spaced over `1:n` (the last one if k == 1)."
path_schedule(n::Int, k::Int) = k <= 1 ? Set([n]) : Set(round.(Int, range(1, n; length = k)))

"Chain state initialisation for the auxiliary process."
init_aux(::Union{StaticArm, GRW1, GRW1Break, MomentumGRW}, N, T, rng) = nothing
init_aux(::StochVolGRW, N, T, rng) = zeros(2N, T)          # ξ = 0: h at its mean
init_aux(::RegimeGRW, N, T, rng) = ones(Int8, 2N, T)       # all calm

# |θ| > 12 on the unconstrained scale is e.g. σ < 6e-6 or logit φ beyond 1 − 6e-6:
# zero prior mass for practical purposes, and where the filter's arithmetic stops
# being trustworthy, so it is excluded outright rather than evaluated.
const THETA_BOUND = 12.0
collapsed_logpost(arm::Union{StaticArm, GRW1, GRW1Break, MomentumGRW}, p, θ, aux) =
    maximum(abs, θ) > THETA_BOUND ? -Inf : kalman_loglik(arm, p, θ) + log_prior(arm, θ)
collapsed_logpost(arm::Union{StochVolGRW, RegimeGRW}, p, θ, aux) =
    maximum(abs, θ) > THETA_BOUND ? -Inf : kalman_loglik(arm, p, θ, aux) + log_prior(arm, θ)

sched_for(arm::Union{StaticArm, GRW1, MomentumGRW}, θ, N, T, aux) = schedule(arm, θ, N, T)
sched_for(arm::GRW1Break, θ, N, T, breaks) = schedule(arm, θ, N, T, breaks)
sched_for(arm::Union{StochVolGRW, RegimeGRW}, θ, N, T, aux) = schedule(arm, θ, N, T, aux)

function run_chain(arm::AbstractArm, p::MarketPanel, c::Int, seed::Int, n_warmup::Int,
                   n_samples::Int, n_keep_paths::Int, ess_steps::Int, thin::Int = 1)
    rng = Xoshiro(seed + 1000c)
    N = n_teams(p)
    T = p.n_weeks
    θ = init_theta(arm, rng)
    aux = arm isa GRW1Break ? season_break_weeks(p) : init_aux(arm, N, T, rng)
    P = (0.9, 0.8)                                  # regime transition (p11, p22)
    widths = fill(1.0, length(θ))
    logf = x -> collapsed_logpost(arm, p, x, aux)
    lf = logf(θ)
    U = zeros(n_samples, length(θ))
    hist = zeros(n_warmup, length(θ))
    keep_at = path_schedule(n_samples, n_keep_paths)
    paths = Matrix{Float64}[]
    path_theta = Vector{Float64}[]
    aux_acc = arm isa Union{StochVolGRW, RegimeGRW} ? zeros(2N, T) : zeros(0, 0)
    needs_paths = arm isa Union{StochVolGRW, RegimeGRW}
    for it in 1:(n_warmup+n_samples*thin)
        logf = x -> collapsed_logpost(arm, p, x, aux)
        lf = logf(θ)                                # aux changed since the last sweep
        θ, lf = slice_sweep(logf, θ, lf, widths, rng)
        if it <= n_warmup
            hist[it, :] .= θ
            if it >= 50 && it % 25 == 0
                sd = vec(std(hist[max(1, it - 199):it, :]; dims = 1))
                widths .= clamp.(3.0 .* sd, 0.02, 3.0)
            end
        end
        s = it > n_warmup && (it - n_warmup) % thin == 0 ? (it - n_warmup) ÷ thin : 0
        keep = s >= 1 && s in keep_at
        if needs_paths || keep
            sch = sched_for(arm, θ, N, T, aux)
            filt = run_filter(arm, p, sch; σ_obs = exp(θ[1]), store = true)
            X = ffbs(arm, p, sch, filt, rng)
            if keep
                push!(paths, X)
                push!(path_theta, copy(θ))
            end
            if needs_paths
                aux, P = update_aux!(arm, aux, P, θ, X, N, T, rng, ess_steps)
                θ, aux = interweave(arm, θ, aux, X, N, rng, widths)
                if s >= 1
                    accumulate_aux!(aux_acc, arm, θ, aux, N)
                end
            end
        end
        if s >= 1
            U[s, :] .= θ
        end
    end
    s_aux = isempty(aux_acc) ? aux_acc : aux_acc ./ n_samples
    Pd = arm isa RegimeGRW ? P : nothing
    return (; U, paths, path_theta, aux = s_aux, P = Pd)
end

"""
Log density of the SV log-volatility paths H (2N × T, columns 2:T) under the AR(1)
prior at θ — the CENTRED parameterisation used by the interweaving step.
"""
function sv_path_logdens(θ, H::AbstractMatrix, N::Int)
    c = constrain(StochVolGRW(), θ)
    T = size(H, 2)
    sd0 = c.σ_h / sqrt(1 - c.γ^2)
    lp = 0.0
    for i in 1:2N
        hb = i <= N ? c.hbar_att : c.hbar_def
        lp += -log(sd0) - 0.5 * ((H[i, 2] - hb) / sd0)^2
        for t in 3:T
            lp += -log(c.σ_h) - 0.5 * ((H[i, t] - hb - c.γ * (H[i, t-1] - hb)) / c.σ_h)^2
        end
    end
    return lp
end

"Invert `sv_logvol`: the whitened ξ that reproduce H at θ."
function sv_whiten(θ, H::AbstractMatrix, N::Int)
    c = constrain(StochVolGRW(), θ)
    T = size(H, 2)
    ξ = zeros(2N, T)
    sd0 = c.σ_h / sqrt(1 - c.γ^2)
    for i in 1:2N
        hb = i <= N ? c.hbar_att : c.hbar_def
        ξ[i, 2] = (H[i, 2] - hb) / sd0
        for t in 3:T
            ξ[i, t] = (H[i, t] - hb - c.γ * (H[i, t-1] - hb)) / c.σ_h
        end
    end
    return ξ
end

"""
Ancillarity-sufficiency interweaving (Yu & Meng 2011) for the volatility block.
The collapsed θ step conditions on the whitened aux (ancillary); this one updates
the volatility hyper-parameters conditional on the CENTRED path (sufficient),
holding σ_obs fixed, then re-expresses the aux at the new θ. The paths themselves
do not move, so the Kalman-level state stays consistent. Returns (θ, aux).
"""
function interweave(arm::StochVolGRW, θ, ξ, X, N, rng, widths)
    H = sv_logvol(constrain(arm, θ), ξ, N)
    logf = z -> maximum(abs, z) > THETA_BOUND ? -Inf : sv_path_logdens(z, H, N) + log_prior(arm, z)
    lf = logf(θ)
    for j in 2:5
        θ, lf = slice_coordinate(logf, θ, lf, j, widths[j], rng)
    end
    return θ, sv_whiten(θ, H, N)
end

function interweave(arm::RegimeGRW, θ, S, X, N, rng, widths)
    E = innovations(X, N)
    T = size(E, 2)
    function logf(z)
        maximum(abs, z) > THETA_BOUND && return -Inf
        c = constrain(arm, z)
        l = 0.0
        for t in 2:T, i in 1:2N
            s1 = i <= N ? c.σ1_att : c.σ1_def
            σ = S[i, t] == 1 ? s1 : s1 * (1 + c.Δ)
            l += -log(σ) - 0.5 * (E[i, t] / σ)^2
        end
        return l + log_prior(arm, z)
    end
    lf = logf(θ)
    for j in 2:4
        θ, lf = slice_coordinate(logf, θ, lf, j, widths[j], rng)
    end
    return θ, S
end

function update_aux!(arm::StochVolGRW, ξ, P, θ, X, N, T, rng, ess_steps)
    c = constrain(arm, θ)
    E = innovations(X, N)
    for i in 1:2N
        hb = i <= N ? c.hbar_att : c.hbar_def
        e = E[i, :]
        xi = ξ[i, 2:T]
        ll = z -> sv_series_loglik(e, vcat(0.0, z), hb, c)
        l0 = ll(xi)
        for _ in 1:ess_steps
            xi, l0 = elliptical_slice(xi, l0, ll, rng)
        end
        ξ[i, 2:T] .= xi
    end
    return ξ, P
end

function update_aux!(arm::RegimeGRW, S, P, θ, X, N, T, rng, ess_steps)
    c = constrain(arm, θ)
    E = innovations(X, N)
    for i in 1:2N
        s1 = i <= N ? c.σ1_att : c.σ1_def
        regime_ffbs!(view(S, i, :), view(E, i, :), s1, s1 * (1 + c.Δ), P[1], P[2], rng)
    end
    n = regime_transition_counts(S)
    (a1, b1), (a2, b2) = REGIME_P_PRIOR
    p11 = rand(rng, Distributions.Beta(a1 + n[1, 1], b1 + n[1, 2]))
    p22 = rand(rng, Distributions.Beta(a2 + n[2, 2], b2 + n[2, 1]))
    return S, (p11, p22)
end

accumulate_aux!(acc, arm::StochVolGRW, θ, ξ, N) = (acc .+= sv_logvol(constrain(arm, θ), ξ, N); acc)
accumulate_aux!(acc, arm::RegimeGRW, θ, S, N) = (acc .+= (S .== 2); acc)

"""
    fit_regime_with_p(panel; kwargs...) -> (fit, p_draws)

The regime arm needs its transition-probability draws kept too; this is `fit_arm`
with a trace of (p11, p22) per retained iteration (iters × 2 × chains).
"""
function fit_regime_with_p(p::MarketPanel; n_chains::Int = 4, n_warmup::Int = 1000,
                           n_samples::Int = 1000, n_paths::Int = 50, seed::Int = 20260922,
                           thin::Int = 1)
    arm = RegimeGRW()
    t0 = time()
    K = length(param_names(arm))
    draws = zeros(n_samples, K + 2, n_chains)
    udraws = zeros(n_samples, K, n_chains)
    per_chain = cld(n_paths, n_chains)
    outs = Vector{Any}(undef, n_chains)
    tasks = map(1:n_chains) do c
        Threads.@spawn run_regime_chain(p, c, seed, n_warmup, n_samples, per_chain, thin)
    end
    for c in 1:n_chains
        outs[c] = fetch(tasks[c])
        udraws[:, :, c] .= outs[c].U
        for s in 1:n_samples
            draws[s, 1:K, c] .= collect(values(constrain(arm, outs[c].U[s, :])))
        end
        draws[:, K+1:K+2, c] .= outs[c].Ptrace
    end
    aux = reduce(+, [o.aux for o in outs]) ./ n_chains
    fit = ArmFit(arm, vcat(param_names(arm), ["p11", "p22"]), draws, udraws,
                 reduce(vcat, [o.paths for o in outs]), reduce(vcat, [o.path_theta for o in outs]),
                 aux, time() - t0)
    return fit
end

function run_regime_chain(p::MarketPanel, c::Int, seed::Int, n_warmup::Int, n_samples::Int,
                          n_keep_paths::Int, thin::Int = 1)
    arm = RegimeGRW()
    rng = Xoshiro(seed + 1000c)
    N = n_teams(p)
    T = p.n_weeks
    θ = init_theta(arm, rng)
    S = init_aux(arm, N, T, rng)
    P = (0.9, 0.8)
    widths = fill(1.0, length(θ))
    U = zeros(n_samples, length(θ))
    Ptrace = zeros(n_samples, 2)
    hist = zeros(n_warmup, length(θ))
    keep_at = path_schedule(n_samples, n_keep_paths)
    paths = Matrix{Float64}[]
    path_theta = Vector{Float64}[]
    acc = zeros(2N, T)
    for it in 1:(n_warmup+n_samples*thin)
        logf = x -> collapsed_logpost(arm, p, x, S)
        θ, _ = slice_sweep(logf, θ, logf(θ), widths, rng)
        if it <= n_warmup
            hist[it, :] .= θ
            if it >= 50 && it % 25 == 0
                sd = vec(std(hist[max(1, it - 199):it, :]; dims = 1))
                widths .= clamp.(3.0 .* sd, 0.02, 3.0)
            end
        end
        sch = schedule(arm, θ, N, T, S)
        filt = run_filter(arm, p, sch; σ_obs = exp(θ[1]), store = true)
        X = ffbs(arm, p, sch, filt, rng)
        S, P = update_aux!(arm, S, P, θ, X, N, T, rng, 0)
        θ, S = interweave(arm, θ, S, X, N, rng, widths)
        s = it > n_warmup && (it - n_warmup) % thin == 0 ? (it - n_warmup) ÷ thin : 0
        if s >= 1
            U[s, :] .= θ
            Ptrace[s, 1] = P[1]
            Ptrace[s, 2] = P[2]
            acc .+= (S .== 2)
            if s in keep_at
                push!(paths, X)
                push!(path_theta, copy(θ))
            end
        end
    end
    return (; U, Ptrace, paths, path_theta, aux = acc ./ n_samples)
end

# ==============================================================================
# 5. Convergence and posterior summaries
# ==============================================================================

"θ draws as an MCMCChains.Chains (constrained scale)."
draws_chains(f::ArmFit) = MCMCChains.Chains(f.draws, f.names)

"""
    convergence_table(fit) -> DataFrame

Posterior mean/sd/quantiles plus R̂ and bulk/tail ESS per parameter, via
MCMCChains' rank-normalised split-R̂ (Vehtari et al. 2021).
"""
function convergence_table(f::ArmFit)
    ch = draws_chains(f)
    ss = DataFrame(MCMCChains.summarystats(ch))
    qs = DataFrame(MCMCChains.quantile(ch; q = [0.05, 0.5, 0.95]))
    out = innerjoin(select(ss, :parameters, :mean, :std, :ess_bulk, :ess_tail, :rhat),
                    qs; on = :parameters)
    out.arm .= arm_name(f.arm)
    return select(out, :arm, :parameters, Not([:arm, :parameters]))
end

"Median θ on the unconstrained scale (the plug-in used by every filter)."
median_theta(f::ArmFit) = [median(vec(f.udraws[:, j, :])) for j in axes(f.udraws, 2)]

"""
    team_paths(fit, panel) -> DataFrame

Posterior mean and 90% band of the CENTRED ratings α = C x̃, β = C x̃ per team and
week, from the retained FFBS path draws.
"""
function team_paths(f::ArmFit, p::MarketPanel)
    N = n_teams(p)
    T = p.n_weeks
    D = length(f.paths)
    A = zeros(D, N, T)
    B = zeros(D, N, T)
    for (d, X) in enumerate(f.paths), t in 1:T
        xa = X[3:(2+N), t]
        xb = X[(3+N):(2+2N), t]
        A[d, :, t] .= xa .- mean(xa)
        B[d, :, t] .= xb .- mean(xb)
    end
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        a = A[:, i, t]; b = B[:, i, t]
        push!(rows, (arm = arm_name(f.arm), team = p.teams[i], week = t,
                     week_start = p.week_start[t],
                     att_mean = mean(a), att_lo = quantile(a, 0.05), att_hi = quantile(a, 0.95),
                     def_mean = mean(b), def_lo = quantile(b, 0.05), def_hi = quantile(b, 0.95)))
    end
    return DataFrame(rows)
end

# ==============================================================================
# 6. One-step-ahead prediction
# ==============================================================================

"""
    onestep_predictions(arm, panel, θ; n_particles, seed) -> (; pred_mean, pred_var, logpd, loglik, ess)

Pre-week predictive for every observation at plug-in θ. Exact Kalman for the
Gaussian arms; RBPF for SV / regime (`logpd` is the log of the particle-mixture
predictive density; `ess` the effective sample size per week).
"""
function onestep_predictions(arm::Union{StaticArm, GRW1, GRW1Break, MomentumGRW}, p::MarketPanel, θ;
                             kwargs...)
    sch = arm isa GRW1Break ? schedule(arm, θ, n_teams(p), p.n_weeks, season_break_weeks(p)) :
                              schedule(arm, θ, n_teams(p), p.n_weeks)
    f = run_filter(arm, p, sch; σ_obs = exp(θ[1]), predict = true)
    lpd = @. -0.5 * (log(2π * f.pred_var) + (p.obs_y - f.pred_mean)^2 / f.pred_var)
    return (; pred_mean = f.pred_mean, pred_var = f.pred_var, logpd = lpd, loglik = f.loglik,
              ess = Float64[])
end

onestep_predictions(arm::Union{StochVolGRW, RegimeGRW}, p::MarketPanel, θ; n_particles::Int = 1000,
                    seed::Int = 7, P = (0.9, 0.8)) =
    rbpf_predictions(arm, p, θ; n_particles = n_particles, seed = seed, P = P)

"Propagate one particle's volatility state into week t; returns its variance column."
function propagate_vol!(state, arm::StochVolGRW, c, N, t, rng, P)
    for i in 1:2N
        hb = i <= N ? c.hbar_att : c.hbar_def
        if t == 2
            state[i] = hb + c.σ_h / sqrt(1 - c.γ^2) * randn(rng)
        else
            state[i] = hb + c.γ * (state[i] - hb) + c.σ_h * randn(rng)
        end
    end
    return exp.(2 .* state)
end
function propagate_vol!(state, arm::RegimeGRW, c, N, t, rng, P)
    d = zeros(2N)
    for i in 1:2N
        if t == 2
            state[i] = rand(rng) < REGIME_PI0[1] ? 1.0 : 2.0
        else
            stay = state[i] == 1.0 ? P[1] : P[2]
            if rand(rng) >= stay
                state[i] = 3.0 - state[i]
            end
        end
        s1 = i <= N ? c.σ1_att : c.σ1_def
        d[i] = (state[i] == 1.0 ? s1 : s1 * (1 + c.Δ))^2
    end
    return d
end

"""
    rbpf_predictions(arm, panel, θ; n_particles, seed, P)

Rao-Blackwellised particle filter: particles carry the volatility state (h or S)
and a full Kalman mean/covariance. Proposal = the volatility transition; weight =
the Kalman predictive likelihood of the week's observations; systematic
resampling when ESS < n/2. `loglik` is the (unbiased-in-likelihood) estimate of
log p(y | θ).
"""
function rbpf_predictions(arm::Union{StochVolGRW, RegimeGRW}, p::MarketPanel, θ;
                          n_particles::Int = 1000, seed::Int = 7, P = (0.9, 0.8))
    N = n_teams(p)
    T = p.n_weeks
    c = constrain(arm, θ)
    R = exp(θ[1])^2
    m0, P0 = initial_state(arm, N)
    n = length(m0)
    Np = n_particles
    ms = [copy(m0) for _ in 1:Np]
    Ps = [copy(P0) for _ in 1:Np]
    vol = [zeros(2N) for _ in 1:Np]
    logw = fill(-log(Np), Np)
    nob = n_obs(p)
    pm = fill(NaN, nob); pv = fill(NaN, nob); lpd = fill(NaN, nob)
    ess = zeros(T)
    ll = 0.0
    nth = Threads.nthreads()   # one work chunk (and one h/k buffer) per thread
    rngs = [Xoshiro(seed + 17k) for k in 1:Np]
    hs = [zeros(n) for _ in 1:nth]
    ks = [zeros(n) for _ in 1:nth]
    for t in 1:T
        r = p.week_ptr[t]:(p.week_ptr[t+1]-1)
        nr = length(r)
        mu = zeros(Np, nr)
        va = zeros(Np, nr)
        linc = zeros(Np)
        chunks = collect(Iterators.partition(1:Np, cld(Np, nth)))
        Threads.@threads for ci in eachindex(chunks)
          h = hs[ci]; k = ks[ci]
          for q in chunks[ci]
            m = ms[q]; Pq = Ps[q]
            if t >= 2
                d = propagate_vol!(vol[q], arm, c, N, t, rngs[q], P)
                @inbounds for i in 1:2N
                    Pq[2+i, 2+i] += d[i]
                end
            end
            for (a, j) in enumerate(r)
                obs_row!(h, N, p.obs_home[j], p.obs_att[j], p.obs_def[j])
                mul!(k, Pq, h)
                mu[q, a] = dot(h, m)
                va[q, a] = dot(h, k) + R
            end
            l = 0.0
            for j in r
                obs_row!(h, N, p.obs_home[j], p.obs_att[j], p.obs_def[j])
                mul!(k, Pq, h)
                S = dot(h, k) + R
                v = p.obs_y[j] - dot(h, m)
                l += -0.5 * (log(2π * S) + v^2 / S)
                axpy!(v / S, k, m)
                BLAS.ger!(-1.0 / S, k, k, Pq)
            end
            if nr > 0
                @inbounds for jj in 1:n, ii in 1:(jj-1)
                    x = 0.5 * (Pq[ii, jj] + Pq[jj, ii])
                    Pq[ii, jj] = x
                    Pq[jj, ii] = x
                end
            end
            linc[q] = l
          end
        end
        # pre-week mixture predictive (weights from before this week)
        w = exp.(logw .- maximum(logw)); w ./= sum(w)
        for (a, j) in enumerate(r)
            pm[j] = sum(w .* mu[:, a])
            pv[j] = sum(w .* (va[:, a] .+ mu[:, a] .^ 2)) - pm[j]^2
            ld = @. log(w) - 0.5 * (log(2π * va[:, a]) + (p.obs_y[j] - mu[:, a])^2 / va[:, a])
            mx = maximum(ld)
            lpd[j] = mx + log(sum(exp.(ld .- mx)))
        end
        # weight update and likelihood increment
        lw = logw .+ linc
        mx = maximum(lw)
        ll += mx + log(sum(exp.(lw .- mx))) - (maximum(logw) + log(sum(exp.(logw .- maximum(logw)))))
        logw = lw .- (mx + log(sum(exp.(lw .- mx))))
        wn = exp.(logw)
        ess[t] = 1 / sum(wn .^ 2)
        if ess[t] < Np / 2
            idx = systematic_resample(wn, rngs[1])
            ms = [copy(ms[i]) for i in idx]
            Ps = [copy(Ps[i]) for i in idx]
            vol = [copy(vol[i]) for i in idx]
            logw = fill(-log(Np), Np)
        end
    end
    return (; pred_mean = pm, pred_var = pv, logpd = lpd, loglik = ll, ess = ess)
end

function systematic_resample(w::Vector{Float64}, rng)
    Np = length(w)
    u = (rand(rng) .+ (0:(Np-1))) ./ Np
    cw = cumsum(w)
    cw[end] = 1.0
    idx = zeros(Int, Np)
    j = 1
    for i in 1:Np
        while u[i] > cw[j]
            j += 1
        end
        idx[i] = j
    end
    return idx
end

"""
    prediction_metrics(panel, pred; mask) -> NamedTuple

RMSE / MAE of the pre-week predictive mean against the observed log λ_mkt, mean
log predictive density, and 90% interval coverage, over the observations in `mask`.
"""
function prediction_metrics(p::MarketPanel, pred; mask = trues(n_obs(p)))
    r = p.obs_y[mask] .- pred.pred_mean[mask]
    z = r ./ sqrt.(pred.pred_var[mask])
    return (; n = count(mask), rmse = sqrt(mean(r .^ 2)), mae = mean(abs.(r)),
              mean_logpd = mean(pred.logpd[mask]), sum_logpd = sum(pred.logpd[mask]),
              cover90 = mean(abs.(z) .<= 1.6448536), sd_z = std(z))
end

# ==============================================================================
# 7. EDA of the GRW1 innovations, and the shock catalogue
# ==============================================================================

"""
    innovation_diagnostics(fit, panel) -> DataFrame

Per retained FFBS path of the rating walk: lag-1 autocorrelation of the weekly
innovations (momentum ⇒ > 0), excess kurtosis of the standardised innovations
(regimes / SV ⇒ > 0) and lag-1 autocorrelation of their squares (volatility
clustering ⇒ > 0). Pooled over teams, IN-SEASON weeks only (a team's innovation
enters only for weeks between its first and last observed fixture of a season).
Posterior draws of the innovations have the prior's properties under a correct
model, so these are posterior-predictive checks of the Arm 1 assumptions.
"""
function innovation_diagnostics(f::ArmFit, p::MarketPanel)
    N = n_teams(p)
    active = active_weeks(p)
    rows = NamedTuple[]
    for (d, X) in enumerate(f.paths)
        E = innovations(X, N)
        for (comp, rr) in (("att", 1:N), ("def", (N+1):2N))
            lag1 = Float64[]; lag0 = Float64[]
            pool = Float64[]
            for i in rr
                ti = i <= N ? i : i - N
                e = [E[i, t] for t in 2:p.n_weeks if active[ti, t] && active[ti, t-1]]
                length(e) < 10 && continue
                e ./= std(e)
                append!(pool, e)
                append!(lag0, e[1:end-1]); append!(lag1, e[2:end])
            end
            ac1 = cor(lag0, lag1)
            ac1sq = cor(lag0 .^ 2, lag1 .^ 2)
            kurt = mean(pool .^ 4) / mean(pool .^ 2)^2 - 3
            push!(rows, (draw = d, component = comp, acf1 = ac1, acf1_sq = ac1sq,
                         excess_kurtosis = kurt))
        end
    end
    return DataFrame(rows)
end

"Weeks inside each team's observed span within each season (N × T)."
function active_weeks(p::MarketPanel)
    N = n_teams(p)
    A = falses(N, p.n_weeks)
    for s in unique(p.obs_season)
        for i in 1:N
            ws = [p.obs_week[j] for j in eachindex(p.obs_y)
                  if p.obs_season[j] == s && (p.obs_att[j] == i)]
            isempty(ws) && continue
            A[i, minimum(ws):maximum(ws)] .= true
        end
    end
    return A
end

"""
    anomaly_catalog(panel, pred; smooth_resid, σ_obs, z_cut = 2.5) -> DataFrame

Two kinds of shock, both flagged at `z_cut`:
* `surprise`  — one-step-ahead standardised residual (the market repriced a team
  between weeks more than the dynamics expected);
* `idio`      — smoothed residual relative to σ_obs (the fixture's price sits off
  the team ratings of its own week: rotation, weather, a cup hangover, …).
"""
function anomaly_catalog(p::MarketPanel, pred, smooth_fit::AbstractVector, σ_obs::Float64;
                         z_cut::Float64 = 2.5)
    z_pre = (p.obs_y .- pred.pred_mean) ./ sqrt.(pred.pred_var)
    z_idio = (p.obs_y .- smooth_fit) ./ σ_obs
    meta = Dict(r.match_id => r for r in eachrow(p.matches))
    rows = NamedTuple[]
    for j in eachindex(p.obs_y)
        (abs(z_pre[j]) >= z_cut || abs(z_idio[j]) >= z_cut) || continue
        r = meta[p.obs_match[j]]
        push!(rows, (match_id = p.obs_match[j], match_date = r.match_date, season = r.season,
                     home_team = r.home_team, away_team = r.away_team,
                     side = p.obs_home[j] == 1.0 ? "home" : "away",
                     attacking_team = p.teams[p.obs_att[j]],
                     log_lambda_mkt = p.obs_y[j], pred_onestep = pred.pred_mean[j],
                     z_surprise = z_pre[j], fit_smoothed = smooth_fit[j], z_idio = z_idio[j]))
    end
    return sort!(DataFrame(rows), :match_date)
end

"Smoothed fitted log-rate per observation from a state path matrix X (n × T)."
function fitted_logrates(p::MarketPanel, X::AbstractMatrix)
    N = n_teams(p)
    h = zeros(size(X, 1))
    out = zeros(n_obs(p))
    for j in eachindex(out)
        obs_row!(h, N, p.obs_home[j], p.obs_att[j], p.obs_def[j])
        out[j] = dot(h, @view X[:, p.obs_week[j]])
    end
    return out
end

"Posterior-mean fitted log-rates over the retained paths (in-sample fit)."
fitted_logrates(p::MarketPanel, f::ArmFit) = mean(fitted_logrates(p, X) for X in f.paths)

# ==============================================================================
# 8. Gates — exactness of the engine on a toy problem
# ==============================================================================

"""
    mid_gates(; seed) -> DataFrame

Independent re-derivation checks (julia guide §7 rungs 1–2) on a toy panel small
enough to write the joint Gaussian of every state and observation down directly:

* G1  Kalman log-likelihood == the batch marginal N(y; H m, H Σ Hᵀ + R)   (GRW1, momentum)
* G2  RTS smoothed mean == the batch posterior mean E[s | y]               (GRW1, momentum)
* G3  RBPF with a degenerate volatility process == the Kalman filter       (SV, regime)
* G4  FFBS draws: sample mean/cov of 4000 draws vs the batch posterior     (GRW1; MC tolerance)
"""
function mid_gates(; seed::Int = 1)
    rng = Xoshiro(seed)
    p = toy_panel(rng)
    rows = NamedTuple[]
    for (arm, θ) in ((GRW1(), [log(0.07), log(0.05), log(0.03)]),
                     (MomentumGRW(), [log(0.07), log(0.05), log(0.03), logit(0.6), log(0.02)]))
        sch = schedule(arm, θ, n_teams(p), p.n_weeks)
        filt = run_filter(arm, p, sch; σ_obs = exp(θ[1]), store = true)
        bm = batch_posterior(arm, p, sch, exp(θ[1]))
        push!(rows, (gate = "G1 loglik $(arm_name(arm))", value = abs(filt.loglik - bm.loglik),
                     tol = 1e-9))
        sm = smoothed_mean(arm, p, sch, filt)
        push!(rows, (gate = "G2 smoothed mean $(arm_name(arm))",
                     value = maximum(abs.(vec(sm) .- bm.post_mean)), tol = 1e-8))
        if arm isa GRW1
            D = 4000
            draws = [vec(ffbs(arm, p, sch, filt, rng)) for _ in 1:D]
            M = reduce(hcat, draws)
            se = sqrt.(max.(diag(bm.post_cov), 1e-300) ./ D)
            zmax = maximum(abs.(vec(mean(M; dims = 2)) .- bm.post_mean) ./ max.(se, 1e-12))
            push!(rows, (gate = "G4 FFBS mean max |z| (4000 draws)", value = zmax, tol = 4.5))
            cerr = maximum(abs.(cov(M; dims = 2) .- bm.post_cov)) / maximum(abs.(bm.post_cov))
            push!(rows, (gate = "G4 FFBS cov max rel err", value = cerr, tol = 0.1))
        end
    end
    # G1/G2 again with a HETEROSCEDASTIC schedule (a random regime path, Δ = 1.5):
    # the SV / regime arms reach the filter only through D, so this covers them.
    let arm = RegimeGRW(), θ = [log(0.07), log(0.05), log(0.03), log(1.5)]
        S = Int8.(rand(rng, 1:2, 2 * n_teams(p), p.n_weeks))
        sch = schedule(arm, θ, n_teams(p), p.n_weeks, S)
        filt = run_filter(arm, p, sch; σ_obs = exp(θ[1]), store = true)
        bm = batch_posterior(arm, p, sch, exp(θ[1]))
        push!(rows, (gate = "G1 loglik regime (random S)", value = abs(filt.loglik - bm.loglik),
                     tol = 1e-9))
        push!(rows, (gate = "G2 smoothed mean regime (random S)",
                     value = maximum(abs.(vec(smoothed_mean(arm, p, sch, filt)) .- bm.post_mean)),
                     tol = 1e-8))
    end
    # G3: degenerate SV (σ_h → 0 so h ≡ h̄) and regime with Δ → 0 reproduce GRW1
    θ1 = [log(0.07), log(0.05), log(0.03)]
    k1 = kalman_loglik(GRW1(), p, θ1)
    sv = rbpf_predictions(StochVolGRW(), p, [log(0.07), log(0.05), log(0.03), logit(0.5), -60.0];
                          n_particles = 16)
    push!(rows, (gate = "G3 RBPF(SV, σ_h→0) vs Kalman", value = abs(sv.loglik - k1), tol = 1e-8))
    rg = rbpf_predictions(RegimeGRW(), p, [log(0.07), log(0.05), log(0.03), -60.0];
                          n_particles = 16)
    push!(rows, (gate = "G3 RBPF(regime, Δ→0) vs Kalman", value = abs(rg.loglik - k1), tol = 1e-8))
    k1p = onestep_predictions(GRW1(), p, θ1)
    push!(rows, (gate = "G3 RBPF(regime) pre-week means vs Kalman",
                 value = maximum(abs.(rg.pred_mean .- k1p.pred_mean)), tol = 1e-10))
    out = DataFrame(rows)
    out.pass = out.value .<= out.tol
    return out
end

"A 5-team, 7-week toy panel with a gap week and a same-week double fixture."
function toy_panel(rng)
    teams = ["A", "B", "C", "D", "E"]
    fixtures = [(1, 1, 2), (1, 3, 4), (2, 5, 1), (2, 2, 3), (4, 4, 5), (4, 1, 3), (4, 2, 4),
                (5, 3, 5), (6, 4, 1), (7, 5, 2), (7, 1, 4)]
    ow = Int[]; oh = Float64[]; oa = Int[]; od = Int[]; oy = Float64[]; om = Int[]
    for (k, (w, h, a)) in enumerate(fixtures)
        push!(ow, w, w); push!(oh, 1.0, 0.0); push!(oa, h, a); push!(od, a, h)
        push!(oy, 0.3 + 0.2randn(rng), 0.1 + 0.2randn(rng)); push!(om, k, k)
    end
    T = 7
    ptr = zeros(Int, T + 1); ptr[1] = 1
    for w in ow
        ptr[w+1] += 1
    end
    cumsum!(ptr, ptr)
    mt = DataFrame(match_id = 1:length(fixtures))
    return MarketPanel(mt, DataFrame(), teams, T, [Date(2024, 1, 1) + Week(t - 1) for t in 1:T],
                       ow, oh, oa, od, oy, om, fill("toy", length(ow)), ptr)
end

"""
Batch joint Gaussian of the stacked states s_{1:T} (prior from the transition
recursion written out directly) and the observations; returns log p(y) and the
posterior mean/cov of vec(s_{1:T}).
"""
function batch_posterior(arm::AbstractArm, p::MarketPanel, sch::InnovationSchedule, σ_obs)
    N = n_teams(p)
    T = p.n_weeks
    m1, P1 = initial_state(arm, N)
    n = length(m1)
    F = transition_matrix(arm, N, sch)
    # prior mean / cov of the stack, by the definition s_t = F s_{t-1} + w_t
    M = zeros(n * T)
    Σ = zeros(n * T, n * T)
    blk(t) = ((t-1)*n+1):(t*n)
    M[blk(1)] .= m1
    Σ[blk(1), blk(1)] .= P1
    for t in 2:T
        M[blk(t)] .= F * M[blk(t - 1)]
        for u in 1:(t-1)
            Σ[blk(t), blk(u)] .= F * Σ[blk(t - 1), blk(u)]
            Σ[blk(u), blk(t)] .= Σ[blk(t), blk(u)]'
        end
        Σ[blk(t), blk(t)] .= F * Σ[blk(t - 1), blk(t - 1)] * F' .+ process_cov(arm, N, sch, t)
    end
    H = zeros(n_obs(p), n * T)
    h = zeros(n)
    for j in 1:n_obs(p)
        obs_row!(h, N, p.obs_home[j], p.obs_att[j], p.obs_def[j])
        H[j, blk(p.obs_week[j])] .= h
    end
    S = Symmetric(H * Σ * H' + σ_obs^2 * I)
    r = p.obs_y .- H * M
    C = cholesky(S)
    ll = -0.5 * (length(r) * log(2π) + logdet(C) + dot(r, C \ r))
    K = Σ * H' / C
    return (; loglik = ll, post_mean = M .+ K * r, post_cov = Σ .- K * H * Σ)
end

end # module
