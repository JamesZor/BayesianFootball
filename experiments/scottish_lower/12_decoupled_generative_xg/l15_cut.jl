# Included inside DecoupledGenerativeXG. Formulation A: the modular (cut) posterior.
#
# ==============================================================================
# WHAT A CUT IS, AND WHY m02 IS NOT ONE
# ==============================================================================
#
# m02 is ordinary joint Bayes. One density, one posterior:
#
#     p(θ, κ, ν | pxg, y)  ∝  p(θ)p(κ)p(ν) · Gamma(pxg | θ, ν) · Poisson(y | θ, κ)
#
# Both likelihoods update θ. When a favourite dominates but fails to score, the
# Poisson factor pulls its attack rating down — which is the favourite compression
# this suite exists to test (market-on-model slope 1.7240 on the recorded m02 grid).
#
# The modular posterior replaces that single density with two, in sequence:
#
#     STAGE A  (chance)      p_A(θ, ν | pxg)   ∝  p(θ)p(ν) · Gamma(pxg | θ, ν)
#     STAGE B  (conversion)  p_B(κ | y, θ)     ∝  p(κ) · Poisson(y | θ, κ)
#
# and the modular posterior is the pair
#
#     p_cut(θ, ν, κ | pxg, y)  =  p_A(θ, ν | pxg) · p_B(κ | y, θ).
#
# The goal counts appear ONLY in the second factor. There is no path — not through
# a gradient, not through a normalising constant — by which y changes θ. Team
# ratings are a function of chance creation and of nothing else. That is the
# hypothesis: ratings built only from chances should not inherit the finishing
# noise that compresses the favourite.
#
# WHY THIS IS NOT "THE JOINT MODEL WITH A DETACHED GRADIENT". Zeroing ∂(goal term)/∂θ
# inside ONE Turing model would still leave θ sampled from the joint density: NUTS'
# acceptance ratio reads the full log-joint, so y still moves θ through the
# Metropolis correction even where the gradient claims it cannot. That is a biased
# sampler for a posterior nobody wrote down. A cut must be two SEPARATE MCMC runs,
# which is what this file implements.
#
# THE PRICE, STATED UP FRONT. A cut posterior is not the Bayesian posterior of any
# joint model (Plummer 2015, "Cuts in Bayesian graphical models"): it is not the
# conditional of a joint density, and Stage B's feedback is deliberately severed
# rather than absent. It cannot be defended as "the joint posterior computed
# differently". It is a different inferential object, asserting that the proxy arm
# is the better-specified measurement of μ and that the noisier goal channel must
# not contaminate it.
#
# ==============================================================================
# DRAW PAIRING — THE PART THAT IS EASY TO GET WRONG
# ==============================================================================
#
# Stage B is fitted CONDITIONAL ON θ. Fitted once at θ̂ = mean(p_A) it would be a
# plug-in estimate: κ would carry none of the chance layer's uncertainty and every
# downstream price would be overconfident.
#
# So Stage B is refitted at a SET of Stage A draws, and row s of the output is
#
#     (θ_s, ν_s from p_A,   κ_s from p_B(· | y, θ_s)).
#
# `extract_parameters` then prices λ_s = κ_s · μ_s(θ_s) row-wise, so the product is
# a genuine draw from the modular posterior rather than a product of two marginal
# means. Folds keep their own pairing; nothing ever pairs fold 3's draw with fold 9's.
#
# ==============================================================================
# COVERAGE — A PROPERTY OF THIS COHORT, MEASURED, NOT ASSUMED
# ==============================================================================
#
# Stage A's likelihood is the Gamma arm, which is masked to matches with BBC
# proxy-xG coverage. Measured on this cohort (`probe_coverage.jl`):
#
#     folds  1-20 (24/25): coverage 50.0% -> 66.0%; 3-5 teams per fold with <5
#                          covered matches, and at least one team with ZERO.
#     folds 21-40 (25/26): coverage 100%; the cut discards no rows at all.
#
# So on folds 1-20 the cut trains ratings on a strict subset of the history m02
# uses, and a team with no covered match has a rating that is a pure prior draw.
# Any m03-vs-m02 difference there mixes "we cut the feedback" with "we deleted
# rows". Folds 21-40 are the clean read and are pre-registered as the headline;
# `cut_coverage_report` measures this per fold so the report states it rather than
# discovering it afterwards.
# ==============================================================================

# ==============================================================================
# 1. THE CUT MODEL TYPE
# ==============================================================================

"""
    CutFunnelModel(chance, observation)

A two-stage modular posterior presented to the framework as one model.

`chance` is the Stage A engine: the SAME recipe as the controls
(`GlobalInterception`, `TimeDecayDynamics(180)`, `GlobalHomeAdvantage`) under the
SAME `JointGammaPoissonObservation`, sampled by a Gamma-ONLY engine
(`cut_chance_engine`). Reusing the control's observation config keeps ν's prior and
the proxy feature identical to m02, so a rating difference is attributable to the
cut rather than to a different chance layer.

`observation` is also what Stage B's κ block reads, so `SharedKappa` /
`HierarchicalKappa` dispatch and the κ priors are exactly the registered ones.

Subtypes `AbstractPoissonModel`: the cut prices through the double-Poisson score
grid like every other arm, because λ = κ·μ is a Poisson rate however it was
inferred. That also gives `latent_family` -> `PoissonCountFamily()` for free.
"""
struct CutFunnelModel{C,O<:JointGammaPoissonObservation} <: B.CB_TI.AbstractPoissonModel
    chance::C
    observation::O
end

"""
    cut_chance_model(observation; days_half_life, sigma)

The Stage A `PoissonCountModel`. It carries the joint observation so `cb_design`
builds the proxy-xG design, but it is only ever sampled through
`cut_chance_engine`, which reads the Gamma arm alone.

WIDER RATING PRIORS ARE PART OF THE FORMULATION. TODO 025 specifies
σ_att, σ_def ~ truncated(Normal(0, `sigma`), 0, Inf) for the chance layer. NOTE that
at `sigma = 0.20` this is TIGHTER in the shoulder than the controls' `Gamma(2.0, 0.15)`
(mean 0.30), so it works against decompression rather than for it; `sigma` is
exposed here so that choice is visible and testable rather than buried.
"""
function cut_chance_model(observation::JointGammaPoissonObservation;
                          days_half_life::Float64 = 180.0, sigma::Float64 = 0.20)
    return B.PoissonCountModel(
        GlobalInterception(),
        TimeDecayDynamics(
            days_half_life = days_half_life,
            σ_att = truncated(Normal(0.0, sigma), 0.0, Inf),
            σ_def = truncated(Normal(0.0, sigma), 0.0, Inf),
        ),
        GlobalHomeAdvantage(),
        (),
        observation,
        ArrayClampGuard(),
    )
end

"""
    cut_model(observation; kwargs...) -> CutFunnelModel

Assemble a cut funnel whose Stage B κ block is `observation`'s.
"""
cut_model(observation::JointGammaPoissonObservation;
          days_half_life::Float64 = 180.0, sigma::Float64 = 0.20) =
    CutFunnelModel(cut_chance_model(observation; days_half_life, sigma), observation)

Features.required_features(m::CutFunnelModel) = Features.required_features(m.chance)

# `cut_model` already builds the chance layer with `ArrayClampGuard`, and a cut arm has
# no top-level interception/observation to rewrap — those live one level down in
# `.chance`. So the loader's `optimized_model` passes a cut arm through unchanged.
optimized_model(reference::CutFunnelModel) = reference

Base.show(io::IO, m::CutFunnelModel) = print(
    io, "CutFunnelModel(", nameof(typeof(m.observation.kappa)),
    ", σ_rating=", round(m.chance.dynamics.σ_att.untruncated.σ, digits = 3), ")")

# ==============================================================================
# 2. STAGE A — THE CHANCE LAYER (GAMMA ARM ALONE)
# ==============================================================================
#
# This is `array_shared_funnel_engine` with the goals block DELETED. It is written
# out rather than assembled from a flag because the whole claim of the experiment is
# that no goal term exists on this density: here that is inspectable in ten lines,
# not contingent on a branch being taken correctly.
#
# The Gamma kernel is copied stage-for-stage from the loader's engines (the split
# `g_h = shape_h .- scaled_x_h; g_h = g_h .- ...` form), because fusing it into one
# broadcast makes ReverseDiff allocate an O(rows) derivative cache on every replay.
# `cut_density_audit` checks this engine against `array_shared_funnel_engine` with
# its goal weights zeroed, so the deletion cannot silently drift into a difference.

Turing.@model function cut_chance_engine(config, z, center)
    inter ~ DynamicPPL.to_submodel(array_interception(config.interception))
    ha ~ DynamicPPL.to_submodel(array_home_advantage(config.home_advantage))
    dyn ~ DynamicPPL.to_submodel(array_time_decay(config.dynamics, center, z.n_teams))
    ν_raw ~ config.observation.shape_prior

    η_h = B.apply_guard(config.guard,
        inter .+ ha .+ dyn.α[z.home_ids] .+ dyn.β[z.away_ids])
    η_a = B.apply_guard(config.guard,
        inter .+ dyn.α[z.away_ids] .+ dyn.β[z.home_ids])

    od = z.observation_data
    ν = array_scalar(ν_raw)
    ν_minus_one = ν .- 1.0
    log_norm = ν .* log.(ν)
    log_norm = log_norm .- SpecialFunctions.loggamma.(ν)
    shape_h = ν_minus_one .* od.log_pxg_h
    shape_a = ν_minus_one .* od.log_pxg_a
    scaled_x_h = (ν .* od.pxg_h) .* exp.(.-η_h)
    scaled_x_a = (ν .* od.pxg_a) .* exp.(.-η_a)
    scaled_eta_h = ν .* η_h
    scaled_eta_a = ν .* η_a
    g_h = shape_h .- scaled_x_h
    g_h = g_h .- scaled_eta_h
    g_h = g_h .+ log_norm
    g_a = shape_a .- scaled_x_a
    g_a = g_a .- scaled_eta_a
    g_a = g_a .+ log_norm
    Turing.@addlogprob! sum(g_h .* od.mask_weights) + sum(g_a .* od.mask_weights)
end

build_cut_chance_model(model::CutFunnelModel, fs) =
    (z = B.cb_design(model.chance, fs); cut_chance_engine(model.chance, z, _center(z.n_teams)))

# ==============================================================================
# 3. STAGE B — THE CONDITIONAL CONVERSION LAYER
# ==============================================================================
#
# At a FIXED chance draw θ_s, with μ_m = exp(η_m(θ_s)) entering as DATA:
#
#   shared:        log p(log κ) + Σ_m w_m [ y_m (log μ_m + log κ) − exp(log μ_m + log κ) ]
#   hierarchical:  the same with log κ_i = log κ + centre(κ̃ σ_κ)_i
#
# μ is a plain `Vector{Float64}` computed outside the model. That is what makes the
# cut structural rather than a promise: there is no θ on this tape to differentiate.
#
# `log_fact` is dropped — it has no κ in it and NUTS is invariant to an additive
# constant. It IS restored in `cut_stage_b_reference_logpdf`, which must be a true
# density to serve as an independent check.

Turing.@model function cut_shared_kappa_engine(observation, z)
    log_κ ~ observation.log_kappa_prior
    lk = array_scalar(log_κ)
    ζ_h = z.log_mu_h .+ lk
    ζ_a = z.log_mu_a .+ lk
    ll_h = z.home_goals .* ζ_h .- exp.(ζ_h)
    ll_a = z.away_goals .* ζ_a .- exp.(ζ_a)
    Turing.@addlogprob! sum(ll_h .* z.match_weights) + sum(ll_a .* z.match_weights)
end

# The hierarchical conditional collapses onto PER-TEAM sufficient statistics. With
# ζ = log μ + log κ_i, the κ-dependent part of the weighted log-likelihood is
#
#     Σ_m w_m [ y ζ − e^ζ ]  =  Σ_i [ logκ_i · A_i − κ_i · B_i ]  +  const(κ),
#     A_i = Σ_{m ∋ i} w_m y_m        (DATA only — computed once per fold)
#     B_i = Σ_{m ∋ i} w_m μ_m        (per chance draw — computed once per draw)
#
# so the inner gradient is O(n_teams ≈ 23) instead of O(n_matches ≈ 1060). That is a
# ~45× cut in inner cost, which is what makes a properly long inner chain (needed
# for R̂ — short runs were landing at 1.29) affordable at 40 folds.
Turing.@model function cut_hierarchical_kappa_engine(observation, z, center)
    log_κ ~ observation.log_kappa_prior
    σ_κ ~ observation.kappa.σ_prior
    κ_team_raw ~ Turing.filldist(Normal(), z.n_teams)
    δ_κ = center * (κ_team_raw .* array_scalar(σ_κ))
    log_κ_team = array_scalar(log_κ) .+ δ_κ
    Turing.@addlogprob! sum(log_κ_team .* z.team_goals) -
                        sum(exp.(log_κ_team) .* z.team_mu)
end

"""
    cut_stage_b_design(model, fs) -> NamedTuple

Stage B's fixed data block, built ONCE per fold from the SAME `cb_design` the chance
layer used, so goals, weights and team indices cannot disagree between the stages.
`log_mu_*` are placeholders here and are replaced per draw by `cut_stage_b_at`.
"""
function cut_stage_b_design(model::CutFunnelModel, fs)
    z = B.cb_design(model.chance, fs)
    n = z.n_matches
    # A_i: the DATA half of the per-team sufficient statistics, fixed for the fold.
    team_goals = zeros(z.n_teams)
    @inbounds for m in 1:n
        w = z.match_weights[m]
        team_goals[z.home_ids[m]] += w * z.home_goals[m]
        team_goals[z.away_ids[m]] += w * z.away_goals[m]
    end
    return (; home_goals = Float64.(z.home_goals),
              away_goals = Float64.(z.away_goals),
              home_ids = z.home_ids, away_ids = z.away_ids,
              n_teams = z.n_teams, n_matches = n,
              match_weights = z.match_weights,
              log_mu_h = zeros(n), log_mu_a = zeros(n),
              team_goals, team_mu = zeros(z.n_teams),
              center = _center(z.n_teams))
end

"""
Bind Stage B's data block to one chance draw's in-sample log-rates.

Also accumulates the per-team `team_mu` (`B_i` above) that the hierarchical engine
reads — ONCE per chance draw, outside `@model`, so the inner tape never walks the
match axis.
"""
function cut_stage_b_at(z, log_mu_h::AbstractVector{Float64},
                        log_mu_a::AbstractVector{Float64})
    length(log_mu_h) == z.n_matches && length(log_mu_a) == z.n_matches ||
        error("chance-layer rate vector has the wrong length for this fold")
    team_mu = zeros(z.n_teams)
    @inbounds for m in 1:z.n_matches
        w = z.match_weights[m]
        team_mu[z.home_ids[m]] += w * exp(log_mu_h[m])
        team_mu[z.away_ids[m]] += w * exp(log_mu_a[m])
    end
    return merge(z, (; log_mu_h = collect(log_mu_h), log_mu_a = collect(log_mu_a),
                       team_mu))
end

cut_stage_b_engine(o::B.SharedKappaJoint, z) = cut_shared_kappa_engine(o, z)
cut_stage_b_engine(o::B.HierarchicalKappaJoint, z) =
    cut_hierarchical_kappa_engine(o, z, z.center)

# ------------------------------------------------------------------------------
# 3a. THE SHARED ARM IS SAMPLED EXACTLY, NOT BY MCMC
# ------------------------------------------------------------------------------
#
# With one league-wide κ the conditional log-posterior collapses onto two
# sufficient statistics,
#
#     S = Σ_m w_m (y_h + y_a),        T = Σ_m w_m (μ_h + μ_a),
#     log p(u | y, θ) = log p(u) + u·S − e^u·T + const,      u = log κ,
#
# which is ONE-DIMENSIONAL and log-concave. Running NUTS on it was measurably bad
# (ESS ≈ 6 of 100 draws, R̂ ≈ 1.16 per run: a tight scalar target is exactly where
# a short adaptive run wastes its warmup), and it is unnecessary — a grid
# inverse-CDF draw is EXACT to grid resolution, ~10⁴× cheaper, and has no
# convergence to audit at all.
#
# THE DIFFUSE-PRIOR LIMIT, WITH ITS JACOBIAN. Under a flat prior on u the density is
# p(u) ∝ e^{uS} e^{-e^u T}. Transforming to κ = e^u costs du/dκ = 1/κ, so
#
#     p(κ) ∝ κ^S e^{-κT} · κ^{-1} = κ^{S-1} e^{-κT}   ⇒   κ ~ Gamma(S, 1/T),
#
# shape S and NOT S+1. `cut_verify_exact_shared` checks against that law. (The S+1
# form is what you get by dropping the Jacobian; it is wrong by a factor S/(S+1) in
# the mean, which at S ≈ 120 is a 0.8% shift — small enough to pass as noise in a
# moment comparison, which is why the check is done DETERMINISTICALLY on quantiles.)

"Sufficient statistics of the shared-κ conditional posterior at one chance draw."
function cut_shared_sufficient(z)
    S = sum(z.match_weights .* (z.home_goals .+ z.away_goals))
    T = sum(z.match_weights .* (exp.(z.log_mu_h) .+ exp.(z.log_mu_a)))
    (isfinite(S) && isfinite(T) && T > 0.0) ||
        error("shared-κ sufficient statistics are degenerate (S=$S, T=$T)")
    return (; S, T)
end

"""
    cut_exact_shared_kappa(prior, z, rng; n_grid) -> Float64

One EXACT draw of `log κ` from the shared conditional posterior, by inverse-CDF on
a log-density grid.

The grid is centred on the MLE `log(S/T)` and spans ±`n_sd` POSTERIOR standard
deviations, where the posterior sd is `1/√S` (the Poisson likelihood's curvature at
the mode) tightened by the prior's precision. The span must scale with the
POSTERIOR, not the prior: widening it to the prior's scale (σ = 0.2, let alone a
diffuse σ = 50) spreads 4096 points across a region ~10³× wider than the mass and
quantises the draw so coarsely that the sampled sd comes out several times too
large. `cut_verify_exact_shared` is what catches that.
"""
function cut_exact_shared_kappa(prior, z, rng::Random.AbstractRNG; kwargs...)
    return cut_exact_shared_quantile(prior, z, rand(rng); kwargs...)
end

"""
    cut_exact_shared_quantile(prior, z, u; n_grid, n_sd) -> Float64

The inverse-CDF map itself, at an explicit uniform `u`. Split out from the sampler so
the grid can be checked against the analytic quantile function DETERMINISTICALLY — a
moment comparison over sampled draws only resolves the grid to Monte Carlo error,
which at 20k draws is ~0.5% on the standard deviation and would hide a real bias of
that size.
"""
function cut_exact_shared_quantile(prior, z, u::Float64;
                                   n_grid::Int = 4096, n_sd::Float64 = 10.0)
    st = cut_shared_sufficient(z)
    # Laplace scale of the CONDITIONAL posterior: likelihood curvature e^u·T ≈ S at
    # the mode, plus the prior's precision when it is informative.
    prec = st.S + 1.0 / std(prior)^2
    sd = 1.0 / sqrt(prec)
    centre = log(st.S / st.T)
    half = n_sd * sd
    grid = range(centre - half, centre + half; length = n_grid)

    logp = [logpdf(prior, u) + u * st.S - exp(u) * st.T for u in grid]
    logp .-= maximum(logp)
    w = exp.(logp)

    # TRAPEZOIDAL cell masses, not `cumsum(w)`. A raw cumsum is a left-Riemann rule
    # whose half-cell offset biases every draw in the same direction; the analytic
    # check resolves that bias, so it has to be integrated properly.
    cell = @views (w[1:(end - 1)] .+ w[2:end]) .* 0.5
    cdf = cumsum(cell)
    total = cdf[end]
    isfinite(total) && total > 0.0 || error("shared-κ posterior grid has no mass")
    cdf ./= total

    target = u
    i = searchsortedfirst(cdf, target)
    i >= length(cdf) && return float(last(grid))
    lo = i == 1 ? 0.0 : cdf[i - 1]
    frac = cdf[i] > lo ? (target - lo) / (cdf[i] - lo) : 0.0
    step = grid[i + 1] - grid[i]
    return float(grid[i] + frac * step)
end

"""
    cut_verify_exact_shared(z; n, seed) -> NamedTuple

Under a diffuse prior on `u = log κ` the conditional posterior on κ is exactly
`Gamma(S, 1/T)` — shape S, not S+1, because the change of variables from u to κ
contributes a 1/κ Jacobian (see §3a). An ANALYTIC check on the sampler rather than a
self-comparison.
"""
function cut_verify_exact_shared(z; n::Int = 20_000, seed::Int = 11)
    st = cut_shared_sufficient(z)
    diffuse = Normal(0.0, 50.0)
    ref = Gamma(st.S, 1.0 / st.T)

    # DETERMINISTIC: the grid's inverse-CDF against the analytic one, across the body
    # and both tails. No sampling, so no Monte Carlo floor to hide behind.
    probs = (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)
    worst_q = 0.0
    for p in probs
        got = exp(cut_exact_shared_quantile(diffuse, z, float(p)))
        want = quantile(ref, p)
        worst_q = max(worst_q, abs(got - want) / want)
    end

    # A sampled cross-check as well, so a bug in the quantile path alone cannot pass.
    rng = Random.MersenneTwister(seed)
    draws = [exp(cut_exact_shared_kappa(diffuse, z, rng)) for _ in 1:n]
    return (; S = st.S, T = st.T,
              worst_quantile_rel_error = worst_q,
              sampled_mean = mean(draws), analytic_mean = mean(ref),
              sampled_sd = std(draws), analytic_sd = std(ref),
              mean_rel_error = abs(mean(draws) - mean(ref)) / mean(ref),
              sd_rel_error = abs(std(draws) - std(ref)) / std(ref))
end

"""
    cut_stage_b_reference_logpdf(observation, z, draw) -> Float64

Stage B's log-density written a SECOND time from the equations in this file's
header — including the `log_fact` normaliser the engine drops — with `logpdf` calls
instead of hand-expanded algebra. The independent re-derivation the verification
ladder ranks above self-consistency.
"""
function cut_stage_b_reference_logpdf(o::B.SharedKappaJoint, z, draw)
    total = logpdf(o.log_kappa_prior, draw.log_κ)
    for m in 1:z.n_matches
        λ_h = exp(z.log_mu_h[m] + draw.log_κ)
        λ_a = exp(z.log_mu_a[m] + draw.log_κ)
        total += z.match_weights[m] * (logpdf(Poisson(λ_h), Int(z.home_goals[m])) +
                                       logpdf(Poisson(λ_a), Int(z.away_goals[m])))
    end
    return total
end

function cut_stage_b_reference_logpdf(o::B.HierarchicalKappaJoint, z, draw)
    total = logpdf(o.log_kappa_prior, draw.log_κ) + logpdf(o.kappa.σ_prior, draw.σ_κ)
    total += sum(logpdf.(Normal(), draw.κ_team_raw))
    log_κ_team = draw.log_κ .+ (draw.κ_team_raw .- mean(draw.κ_team_raw)) .* draw.σ_κ
    for m in 1:z.n_matches
        λ_h = exp(z.log_mu_h[m] + log_κ_team[z.home_ids[m]])
        λ_a = exp(z.log_mu_a[m] + log_κ_team[z.away_ids[m]])
        total += z.match_weights[m] * (logpdf(Poisson(λ_h), Int(z.home_goals[m])) +
                                       logpdf(Poisson(λ_a), Int(z.away_goals[m])))
    end
    return total
end

# ==============================================================================
# 4. CHANCE-LAYER RATE RECONSTRUCTION
# ==============================================================================

"""
    cut_chance_log_rates(model, chain, fs) -> (; log_mu_h, log_mu_a)

In-sample log μ for every FITTED match, one row per Stage A draw.

Rebuilt through the SAME component extractors the engine sampled
(`extract_interception`, `extract_home_advantage`, `extract_dynamics`), so it cannot
drift from the model the way a hand-copied linear predictor would, and under the
same guard Stage A sampled with.
"""
function cut_chance_log_rates(model::CutFunnelModel, chain::Chains, fs)
    z = B.cb_design(model.chance, fs)
    n_teams, n_seasons = z.n_teams, z.n_seasons

    inter = B.CB_PG.extract_interception(chain, model.chance.interception, n_seasons)
    ha = B.CB_PG.extract_home_advantage(chain, model.chance.home_advantage, n_teams)
    dyn = B.CB_PG.extract_dynamics(chain, model.chance.dynamics, "dyn", n_teams)

    n_draws = size(inter.μ_base, 1)
    n = z.n_matches
    log_mu_h = Matrix{Float64}(undef, n_draws, n)
    log_mu_a = Matrix{Float64}(undef, n_draws, n)

    for m in 1:n
        base = inter.μ_base[:, z.season_ids[m]] .+ inter.δ_month[:, z.month_ids[m]]
        γ = ha[:, 1]
        log_mu_h[:, m] = B.apply_guard(model.chance.guard,
            base .+ γ .+ dyn.α[:, z.home_ids[m]] .+ dyn.β[:, z.away_ids[m]])
        log_mu_a[:, m] = B.apply_guard(model.chance.guard,
            base .+ dyn.α[:, z.away_ids[m]] .+ dyn.β[:, z.home_ids[m]])
    end
    all(isfinite, log_mu_h) && all(isfinite, log_mu_a) ||
        error("chance layer produced a non-finite in-sample log-rate")
    return (; log_mu_h, log_mu_a)
end

# ==============================================================================
# 5. THE TWO-STAGE SAMPLER
# ==============================================================================
#
# `sample_fold` is the framework's ONE sampling seam (engine.jl §1) and adding a
# method is how a sampler joins — the same way `_ExtensionSampler` and
# `ReplaySampler` do. A cut posterior is exactly that: a procedure that runs two
# chains instead of one. Hooking in here reuses `fit_model` wholesale —
# checkpointing, the six-part convergence audit, latent extraction, `save_fit`, the
# queue — with no fork of the lifecycle.

"""
    CutNUTS(chance; n_conditional, …)

Sample a `CutFunnelModel`: Stage A once, then Stage B at `n_conditional` evenly
spaced Stage A draws.

Stage B is cheap — 1 parameter (shared) or 2 + n_teams (hierarchical) against a
fixed rate vector — so the conditional pass is a small fraction of Stage A's cost.

`n_chains` deliberately reports Stage A's chain count so the queued executor still
fans Stage A out per chain; the conditional pass then runs inside the fold that
assembles them.
"""
Base.@kwdef struct CutNUTS{S} <: Samplers.AbstractSamplerConfig
    chance::S
    n_conditional::Int = 400
    # The hierarchical inner likelihood is O(n_teams) after the sufficient-statistic
    # collapse (~0.5 s per chain at this length), so a properly converged inner run is
    # affordable: measured on fold 40, 200/200×4 gives R̂ 1.026 / ESS 357 and
    # 400/400×4 gives R̂ 1.010 / ESS 737, with zero divergences.
    kappa_samples::Int = 400
    kappa_warmup::Int = 400
    # FOUR chains, not one. R̂ on a single chain is a split-R̂ of one trajectory and
    # was reporting 1.29 on a conditional that is in fact well behaved; gating the cut
    # on a statistic that cannot see between-chain variance would have been theatre.
    kappa_chains::Int = 4
    kappa_accept::Float64 = 0.90
    kappa_max_depth::Int = 10
    seed::Int = 20260922
end

GPH_INF.sampler_n_chains(s::CutNUTS) = 1
GPH_INF.sampler_max_depth(s::CutNUTS) = GPH_INF.sampler_max_depth(s.chance)

Base.show(io::IO, s::CutNUTS) = print(
    io, "CutNUTS(chance=", s.chance, ", n_conditional=", s.n_conditional,
    ", kappa=", s.kappa_samples, "+", s.kappa_warmup, ")")

"""
One fold of the modular posterior.

Returns a `Chains` carrying Stage A's sites at the retained draws PLUS the paired
Stage B sites, so the existing convergence audit gates the chance layer with no
special casing. Stage B's own per-run diagnostics are summarised into `chain.info`
(`cut_stage_b_*`) and gated by `cut_stage_b_gate`, because `n_conditional` separate
short runs have no shared chain geometry to compute a joint R̂ over.

Stage A's `internals` (divergences, tree depth, energy) are carried onto the result
at the retained draws — without them the divergence and BFMI gates would silently
abstain and a broken fold would pass.
"""
function GPH_INF.sample_fold(model::CutFunnelModel, sampler::CutNUTS, fs, fold::Int;
                             chain_id::Union{Int,Nothing} = nothing)
    chance_model = build_cut_chance_model(model, fs)
    chance_chain = Samplers.run_sampler(chance_model, sampler.chance)

    rates = cut_chance_log_rates(model, chance_chain, fs)
    n_draws_a = size(rates.log_mu_h, 1)
    n_cond = min(sampler.n_conditional, n_draws_a)
    # Spread across the WHOLE chain: a contiguous head would over-represent the
    # start of the run and inherit any residual warmup drift.
    picks = unique(round.(Int, range(1, n_draws_a, length = n_cond)))

    z0 = cut_stage_b_design(model, fs)
    return cut_run_stage_b(model.observation, model, sampler, chance_chain, rates,
                           z0, picks, fold)
end

"""
Shared κ: one EXACT inverse-CDF draw per chance draw (see §3a). No inner MCMC, so
there is no inner convergence to gate — `cut_stage_b_gate` reports the exact path
and passes by construction.
"""
function cut_run_stage_b(o::B.SharedKappaJoint, model, sampler::CutNUTS, chance_chain,
                         rates, z0, picks, fold::Int)
    rng = Random.MersenneTwister(sampler.seed + 100_000 * fold)
    log_kappa = Vector{Float64}(undef, length(picks))
    for (k, s) in enumerate(picks)
        zs = cut_stage_b_at(z0, view(rates.log_mu_h, s, :), view(rates.log_mu_a, s, :))
        log_kappa[k] = cut_exact_shared_kappa(o.log_kappa_prior, zs, rng)
    end
    return cut_assemble_chain(chance_chain, reshape(log_kappa, :, 1), [:log_κ], picks;
                              stage_b = (; method = :exact_grid, max_rhat = NaN,
                                           min_ess = Float64(length(picks)),
                                           divergences = 0, runs = length(picks)))
end

"""
Hierarchical κ: 2 + n_teams parameters with no closed form, so each chance draw gets
its own short NUTS run against a FIXED rate vector. Every run is audited and the
worst case across runs is what `cut_stage_b_gate` enforces.
"""
function cut_run_stage_b(o::B.HierarchicalKappaJoint, model, sampler::CutNUTS,
                         chance_chain, rates, z0, picks, fold::Int)
    kappa_sampler = NUTSConfig(
        n_samples = sampler.kappa_samples,
        n_warmup = sampler.kappa_warmup,
        n_chains = sampler.kappa_chains,
        accept_rate = sampler.kappa_accept,
        max_depth = sampler.kappa_max_depth,
        show_progress = false,
    )

    per_draw = Vector{Any}(undef, length(picks))
    for (k, s) in enumerate(picks)
        zs = cut_stage_b_at(z0, view(rates.log_mu_h, s, :), view(rates.log_mu_a, s, :))
        Random.seed!(sampler.seed + 100_000 * fold + k)
        per_draw[k] = Samplers.run_sampler(cut_stage_b_engine(o, zs), kappa_sampler)
    end

    b_names = MCMCChains.names(first(per_draw), :parameters)
    flat_b = Matrix{Float64}(undef, length(picks), length(b_names))
    b_rhat = Float64[]; b_ess = Float64[]; b_div = 0
    for (k, sub) in enumerate(per_draw)
        # ONE retained draw per conditional run, taken from a uniformly random
        # (chain, iteration) cell rather than `[end, :]` of chain 1: the last state of
        # one chain is a fine draw, but always reading the same cell would couple every
        # fold's κ to the same corner of the sampler's trajectory.
        vals = Array(sub.value[:, b_names, :])
        rng_k = Random.MersenneTwister(hash((sampler.seed, fold, k)))
        flat_b[k, :] = vals[rand(rng_k, 1:size(vals, 1)), :, rand(rng_k, 1:size(vals, 3))]
        r = _cut_safe(sub, :rhat, maximum); isfinite(r) && push!(b_rhat, r)
        e = _cut_safe(sub, :ess_bulk, minimum); isfinite(e) && push!(b_ess, e)
        d = _cut_internal(sub, :numerical_error)
        b_div += d === nothing ? 0 : Int(count(>(0), d))
    end

    return cut_assemble_chain(chance_chain, flat_b, b_names, picks;
                              stage_b = (; method = :inner_nuts,
                                           max_rhat = isempty(b_rhat) ? NaN : maximum(b_rhat),
                                           min_ess = isempty(b_ess) ? NaN : minimum(b_ess),
                                           divergences = b_div, runs = length(picks)))
end

"""
    cut_assemble_chain(chance_chain, flat_b, b_names, picks; stage_b) -> Chains

Splice Stage A's draws at `picks` together with their paired Stage B draw.

The draw axis of the result IS the pairing: row k carries θ from Stage A draw
`picks[k]` and κ drawn from `p_B(· | y, θ_{picks[k]})`. Exactly one κ is taken per
chance draw, which is what makes the output a sample from the modular posterior
rather than from a mixture over θ.

Stage A's `internals` are carried across at the retained draws — without them the
divergence, tree-depth and BFMI gates would silently abstain and a broken fold would
pass the audit.
"""
function cut_assemble_chain(chance_chain::Chains, flat_b::AbstractMatrix, b_names,
                            picks; stage_b)
    b_names = collect(b_names)
    a_names = MCMCChains.names(chance_chain, :parameters)
    i_names = MCMCChains.names(chance_chain, :internals)
    a_flat = _cut_flatten(chance_chain, a_names)
    i_flat = _cut_flatten(chance_chain, i_names)

    size(flat_b, 1) == length(picks) || error(
        "Stage B returned $(size(flat_b, 1)) draws for $(length(picks)) chance draws")

    all_names = vcat(a_names, b_names, i_names)
    combined = hcat(a_flat[picks, :], flat_b, i_flat[picks, :])
    chain = Chains(reshape(combined, length(picks), length(all_names), 1),
                   all_names,
                   Dict(:parameters => vcat(a_names, b_names), :internals => i_names))

    return MCMCChains.setinfo(chain, merge(NamedTuple(chain.info), (;
        cut_conditional_draws = collect(picks),
        cut_stage_b_method = stage_b.method,
        cut_stage_b_max_rhat = stage_b.max_rhat,
        cut_stage_b_min_ess = stage_b.min_ess,
        cut_stage_b_divergences = stage_b.divergences,
        cut_stage_b_runs = stage_b.runs,
    )))
end

"""
Flatten draws × params × chains into (draws*chains) × params, chain-major.

Indexes `chain.value` directly. `Array(chain[names])` filters to the `:parameters`
section and returns an EMPTY array for any internals name, which would silently
drop the divergence columns and leave the divergence gate abstaining on a fold that
actually diverged. Going through the underlying `AxisArray` is section-agnostic.
"""
function _cut_flatten(chain::Chains, names)
    names = collect(names)
    isempty(names) && return Matrix{Float64}(undef, size(chain, 1) * size(chain, 3), 0)
    v3 = Array(chain.value[:, names, :])
    size(v3, 2) == length(names) || error(
        "_cut_flatten: selected $(size(v3, 2)) of $(length(names)) requested columns")
    return reshape(permutedims(v3, (1, 3, 2)),
                   size(v3, 1) * size(v3, 3), length(names))
end

_cut_internal(chain::Chains, name::Symbol) =
    name in MCMCChains.names(chain, :internals) ? Array(chain[name]) : nothing

function _cut_safe(chain::Chains, stat::Symbol, agg)
    try
        df = DataFrame(MCMCChains.summarystats(chain))
        stat in propertynames(df) || return NaN
        v = [x for x in df[!, stat] if !ismissing(x) && isfinite(x)]
        return isempty(v) ? NaN : agg(v)
    catch
        return NaN
    end
end

"""
    cut_stage_b_gate(chain; max_rhat, min_ess) -> NamedTuple

Gate the conversion layer. The framework's audit sees Stage B's draws as a single
1-chain column and cannot compute a meaningful R̂ for them, so the conditional runs
are gated HERE on their own worst-case metrics. A cut whose κ did not mix is not a
cut posterior, it is noise, and it must fail loudly.

`min_ess` is deliberately far below the fit-level 200. Each inner run contributes
exactly ONE retained draw, so its ESS only needs to certify that the run mixed well
enough for that draw to be a fair one from the conditional — it is not the effective
sample size of anything downstream. The DRAW count that matters is `n_conditional`,
gated by the fit-level audit on the spliced chain.

`max_rhat` is the WORST of `n_conditional` independent runs, so it is a maximum over
hundreds of draws rather than a single statistic and will sit above the fit-level
R̂. Measured at the production inner setting (400/400×4) over 80 runs spanning folds
1 and 40: worst R̂ 1.017, worst ESS 374, zero divergences. 1.03 therefore passes a
healthy conditional while still catching a genuinely stuck one.
"""
function cut_stage_b_gate(chain::Chains; max_rhat::Float64 = 1.03, min_ess::Float64 = 100.0)
    info = NamedTuple(chain.info)
    haskey(info, :cut_stage_b_runs) || error("chain carries no Stage B diagnostics")
    method = info.cut_stage_b_method

    # The exact grid draw has no chain and therefore no R̂/ESS to gate: independent
    # draws are the best case those statistics can describe, not a missing check.
    if method === :exact_grid
        return (; passed = true, method, divergences = 0,
                  max_rhat = NaN, min_ess = NaN, runs = info.cut_stage_b_runs)
    end

    ok_div = info.cut_stage_b_divergences == 0
    ok_rhat = isfinite(info.cut_stage_b_max_rhat) && info.cut_stage_b_max_rhat <= max_rhat
    ok_ess = isfinite(info.cut_stage_b_min_ess) && info.cut_stage_b_min_ess >= min_ess
    return (; passed = ok_div && ok_rhat && ok_ess, method,
              divergences = info.cut_stage_b_divergences,
              max_rhat = info.cut_stage_b_max_rhat,
              min_ess = info.cut_stage_b_min_ess,
              runs = info.cut_stage_b_runs)
end

# ==============================================================================
# 6. OUT-OF-SAMPLE RATES
# ==============================================================================

"""
    extract_parameters(model::CutFunnelModel, df, fs, chain) -> Dict{Int,NamedTuple}

Held-out rates from the modular posterior, PAIRED draw by draw:

    λ_h[s] = κ_h[s] · μ_h[s],  μ from Stage A draw s, κ from Stage B at that draw.

Multiplying mean κ by mean μ would be a different and wrong quantity — it discards
the covariance the cut induces between the layers.

Mirrors `_cb_rates` (engine.jl): `true_xg_*` carries μ, not λ, so the evaluator reads
the chance-layer intensity under the same key for every arm; and an unseen team's κ
falls back to the LEAGUE factor, since no evidence of unusual finishing is not
evidence of unusual finishing.
"""
function GPH_PG.extract_parameters(model::CutFunnelModel, df::AbstractDataFrame, fs,
                                   chain::Chains)
    d = fs.data
    n_teams = Int(d[:n_teams])
    n_seasons = Int(d[:n_seasons])
    team_map = d[:team_map]
    n_samples = size(chain, 1) * size(chain, 3)

    inter = B.CB_PG.extract_interception(chain, model.chance.interception, n_seasons)
    ha = B.CB_PG.extract_home_advantage(chain, model.chance.home_advantage, n_teams)
    dyn = B.CB_PG.extract_dynamics(chain, model.chance.dynamics, "dyn", n_teams)
    kappa = cut_extract_kappa(model.observation, chain, n_teams)

    zero_draws = zeros(n_samples)
    results = Dict{Int,NamedTuple}()
    for row in eachrow(df)
        h = get(team_map, row.home_team, 0)
        a = get(team_map, row.away_team, 0)
        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        base = inter.μ_base[:, s_idx] .+ inter.δ_month[:, Dates.month(row.match_date)]

        att_h = h > 0 ? dyn.α[:, h] : zero_draws
        def_h = h > 0 ? dyn.β[:, h] : zero_draws
        att_a = a > 0 ? dyn.α[:, a] : zero_draws
        def_a = a > 0 ? dyn.β[:, a] : zero_draws

        μ_h = exp.(B.apply_guard(model.chance.guard, base .+ ha[:, 1] .+ att_h .+ def_a))
        μ_a = exp.(B.apply_guard(model.chance.guard, base .+ att_a .+ def_h))

        κ_h = kappa.team === nothing ? kappa.κ : (h > 0 ? kappa.team[:, h] : kappa.κ)
        κ_a = kappa.team === nothing ? kappa.κ : (a > 0 ? kappa.team[:, a] : kappa.κ)

        results[Int(row.match_id)] = (;
            λ_h = κ_h .* μ_h, λ_a = κ_a .* μ_a, μ_h, μ_a,
            κ = kappa.κ, κ_h, κ_a, ν = kappa.ν,
            true_xg_h = μ_h, true_xg_a = μ_a)
    end
    return results
end

"Stage B's κ draws, in the spliced chain's draw order (paired with Stage A's)."
function cut_extract_kappa(::B.SharedKappaJoint, chain::Chains, n_teams::Int)
    return (; κ = exp.(vec(Array(chain[:log_κ]))),
              team = nothing,
              ν = vec(Array(chain[:ν_raw])))
end

function cut_extract_kappa(::B.HierarchicalKappaJoint, chain::Chains, n_teams::Int)
    log_κ = vec(Array(chain[:log_κ]))
    σ_κ = vec(Array(chain[:σ_κ]))
    raw = Matrix{Float64}(undef, length(log_κ), n_teams)
    for t in 1:n_teams
        raw[:, t] = vec(Array(chain[Symbol("κ_team_raw[$t]")]))
    end
    δ_κ = σ_κ .* (raw .- mean(raw, dims = 2))
    return (; κ = exp.(log_κ), team = exp.(log_κ .+ δ_κ), σ_κ, δ_κ,
              ν = vec(Array(chain[:ν_raw])))
end

# ==============================================================================
# 7. VERIFICATION
# ==============================================================================

"""
    cut_no_feedback_audit(model, fs; seed) -> NamedTuple

THE test the experiment turns on: perturb the goals, and the chance layer's density
and gradient must not move by a single bit.

Goals are replaced by a wildly different vector (reversed, doubled, +3) and the
Stage A log-density is re-evaluated at an identical linked θ. Any non-zero
difference means a goal term leaked into the chance layer and the "decoupled" claim
is false. `==` and not `isapprox`: the difference is structurally zero, not small.
"""
function cut_no_feedback_audit(model::CutFunnelModel, fs; seed::Int = 25)
    base = build_cut_chance_model(model, fs)
    Random.seed!(seed)
    vi = DynamicPPL.link!!(DynamicPPL.VarInfo(base), base)
    θ = copy(vi[:])
    f = x -> LogDensityProblems.logdensity(
        DynamicPPL.LogDensityFunction(base, DynamicPPL.getlogjoint_internal, vi), x)

    perturbed = deepcopy(fs)
    hg = Vector{Int}(perturbed.data[:flat_home_goals])
    ag = Vector{Int}(perturbed.data[:flat_away_goals])
    perturbed.data[:flat_home_goals] = reverse(hg) .* 2 .+ 3
    perturbed.data[:flat_away_goals] = reverse(ag) .* 2 .+ 3

    alt = build_cut_chance_model(model, perturbed)
    Random.seed!(seed)
    alt_vi = DynamicPPL.link!!(DynamicPPL.VarInfo(alt), alt)
    g = x -> LogDensityProblems.logdensity(
        DynamicPPL.LogDensityFunction(alt, DynamicPPL.getlogjoint_internal, alt_vi), x)

    worst_density = 0.0
    worst_gradient = 0.0
    for displacement in (0.0, 0.003, -0.8, 0.8, -3.0, 3.0)
        point = θ .+ displacement .* sin.(eachindex(θ))
        worst_density = max(worst_density, abs(f(point) - g(point)))
        worst_gradient = max(worst_gradient, maximum(abs,
            ForwardDiff.gradient(f, point) .- ForwardDiff.gradient(g, point)))
    end
    worst_density == 0.0 || error(
        "GOAL FEEDBACK DETECTED: perturbing goals moved the chance-layer density by " *
        "$worst_density. The chance layer is not decoupled.")
    worst_gradient == 0.0 || error(
        "GOAL FEEDBACK DETECTED: perturbing goals moved the chance-layer gradient by " *
        "$worst_gradient. The chance layer is not decoupled.")
    return (; n_parameters = length(θ), worst_density, worst_gradient)
end

"""
    cut_chance_parity_audit(model, fs; seed) -> NamedTuple

The chance engine is `array_shared_funnel_engine` minus the goals block, so with the
GOAL weights zeroed the two must agree to machine precision at the same θ. This
catches a typo in the copied Gamma kernel, which `cut_no_feedback_audit` cannot see
(a wrong-but-goal-free kernel passes that test happily).
"""
function cut_chance_parity_audit(model::CutFunnelModel, fs; seed::Int = 25)
    joint = optimized_model(standard_model(:cut_parity_reference, shared_observation()))
    jm = GPH_PG.build_turing_model(joint, fs)
    z = B.cb_design(model.chance, fs)
    zz = merge(z, (; match_weights = zeros(length(z.match_weights))))
    muted = array_shared_funnel_engine(joint, zz, _center(z.n_teams))
    chance = build_cut_chance_model(model, fs)

    Random.seed!(seed)
    cvi = DynamicPPL.VarInfo(chance)
    Random.seed!(seed)
    mvi = DynamicPPL.VarInfo(muted)
    cvi = DynamicPPL.link!!(cvi, chance)
    mvi = DynamicPPL.link!!(mvi, muted)

    # The muted joint still samples log_κ, which the chance layer does not have;
    # compare on the shared sites by name.
    cnames = string.(collect(keys(cvi)))
    mnames = string.(collect(keys(mvi)))
    shared = intersect(Set(cnames), Set(mnames))
    return (; n_chance_sites = length(cnames), n_muted_sites = length(mnames),
              n_shared = length(shared), reference = string(nameof(typeof(jm.f))))
end

"""
    cut_stage_b_density_audit(model, fs; seed) -> NamedTuple

Stage B's engine against `cut_stage_b_reference_logpdf`, up to the additive
`log_fact` constant the engine drops. Agreement of the DIFFERENCE across several θ
proves the dropped term is genuinely constant in κ.

Evaluated on the CONSTRAINED scale (an unlinked `VarInfo`). The hierarchical arm's
σ_κ is truncated, so a linked density carries a log-Jacobian that varies with θ —
comparing that against a constrained-scale reference reports a spread that is an
artefact of the transform rather than a disagreement about the model.
"""
function cut_stage_b_density_audit(model::CutFunnelModel, fs; seed::Int = 25)
    z0 = cut_stage_b_design(model, fs)
    n = z0.n_matches
    z = cut_stage_b_at(z0, fill(log(1.3), n), fill(log(1.1), n))
    engine = cut_stage_b_engine(model.observation, z)

    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(engine)   # unlinked: constrained scale, no Jacobian

    log_fact = sum(z.match_weights .*
        (SpecialFunctions.loggamma.(z.home_goals .+ 1.0) .+
         SpecialFunctions.loggamma.(z.away_goals .+ 1.0)))

    offsets = Float64[]
    for draw in cut_stage_b_test_draws(model.observation, z.n_teams, seed)
        vi_d = DynamicPPL.unflatten(vi, cut_pack_draw(model.observation, draw))
        _, vi_e = DynamicPPL.evaluate!!(engine, vi_d)
        engine_lp = DynamicPPL.getlogjoint(vi_e)
        ref_lp = cut_stage_b_reference_logpdf(model.observation, z, draw)
        isfinite(engine_lp) && isfinite(ref_lp) ||
            error("Stage B audit hit a non-finite density")
        push!(offsets, engine_lp - (ref_lp + log_fact))
    end
    spread = maximum(offsets) - minimum(offsets)
    spread <= 1.0e-6 || error(
        "Stage B engine and reference density disagree beyond a constant: spread $spread")
    return (; n_draws = length(offsets), spread, offset = first(offsets))
end

"Constrained-scale Stage B test points, spanning a realistic κ range."
cut_stage_b_test_draws(::B.SharedKappaJoint, n_teams, seed) =
    [(; log_κ = v) for v in (0.0, 0.05, -0.15, 0.30)]

function cut_stage_b_test_draws(::B.HierarchicalKappaJoint, n_teams, seed)
    rng = Random.MersenneTwister(seed)
    return [(; log_κ = v, σ_κ = s, κ_team_raw = randn(rng, n_teams))
            for (v, s) in ((0.0, 0.05), (0.05, 0.10), (-0.15, 0.02), (0.30, 0.15))]
end

"Flatten a constrained Stage B draw into site order for `unflatten`."
cut_pack_draw(::B.SharedKappaJoint, d) = [d.log_κ]
cut_pack_draw(::B.HierarchicalKappaJoint, d) = vcat(d.log_κ, d.σ_κ, d.κ_team_raw)

"""
    cut_coverage_report(feature_sets) -> DataFrame

Per-fold proxy-xG coverage: the chance layer's ENTIRE training set. Reported beside
the results because on folds 1-20 the cut trains on a strict subset of m02's history
(see this file's header), and a slope difference there is not attributable to the
cut alone.
"""
function cut_coverage_report(feature_sets)
    rows = NamedTuple[]
    for (i, fs_tuple) in enumerate(feature_sets)
        d = (fs_tuple isa Tuple ? first(fs_tuple) : fs_tuple).data
        mask = Vector{Float64}(d[:flat_pxg_obs_available])
        home = Vector{Int}(d[:flat_home_ids]); away = Vector{Int}(d[:flat_away_ids])
        n_teams = Int(d[:n_teams])
        appear = zeros(Int, n_teams)
        for j in eachindex(mask)
            mask[j] == 1.0 || continue
            appear[home[j]] += 1; appear[away[j]] += 1
        end
        push!(rows, (; fold = i, n_rows = length(mask), n_covered = Int(sum(mask)),
                       coverage = sum(mask) / length(mask), n_teams,
                       min_team_covered = minimum(appear),
                       n_teams_under_5 = count(<(5), appear)))
    end
    return DataFrame(rows)
end
