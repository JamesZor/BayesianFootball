# Market-inverse state-space models — Scottish Lower (TODO 023)

**Verdict.** The market's own team ratings move like a **first-order random walk
with no momentum**, about **0.027–0.029 per week** in log-rate units for attack
and defence alike. Four more elaborate laws were tested against it:

* **Momentum GRW (Arm 2):** adds nothing. φ barely moves from its prior, and
  one-step error is identical to GRW1 to four decimals.
* **Season-boundary jump (control):** adds nothing. The summer is already
  covered by twelve ordinary weekly steps.
* **Stochastic volatility (Arm 3) and 2-state regime switching (Arm 4):** these
  do improve the predictive density: +0.026 and +0.048 nats per observation on
  the honest 25/26 hold-out. Point error does not improve.

Inspecting *where* they spend that extra volatility shows it is mostly
**single-fixture pricing outliers absorbed as spike-and-revert state jumps**.
That is a heavy-tailed *observation* effect, not a latent volatility regime.
Phase 2 should test that directly (§7) before any MS-GARCH work.

> **Phase 2 (2026-09-22):** what the market's supremacy is *made of* is in
> [`PHASE2_FEATURE_ATTRIBUTION.md`](PHASE2_FEATURE_ATTRIBUTION.md)
> ([`r02_market_feature_attribution.jl`](r02_market_feature_attribution.jl)).
> Its main results:
>
> * goal form and proxy-xG form explain about 30% each of market supremacy;
>   wealth explains 4%, the RAPM lineup 3%, and rest nothing;
> * with the features in, the latent team spread falls by 37%;
> * a Student-t observation model gives σ_obs 0.055 with ν ≈ 2.9;
> * two inversion defects were found (T015 totals-only books, thin books), and
>   they correct one Phase 1 anomaly (§6).

All numbers below are measured by
[`r01_market_inverse_runner.jl`](r01_market_inverse_runner.jl) on mcmc-beast
(`/root/BF_market_inverse`, `-t 16`, 2026-09-22). Raw outputs are in
[`results/production/`](results/production/) and the full run log is in
[`results/production_run.log`](results/production_run.log).

---

## 1. Design as run

**Target.** The Betfair (−20, 0] TWA close, de-vigged within market. This is the
same book as TODO 021's supremacy slopes. It is inverted to
(λ_mkt_h, λ_mkt_a) by `Calibration.invert_market_rates` with the default
`MarketInversionConfig` gates.

**Panel.** There are 710 fixtures (tournaments 56/57, seasons 24/25 + 25/26).
**623 were accepted**, giving 1,246 log-rate observations, 22 teams and 91
weekly steps. The 87 refusals are all coverage, not fit quality:

* 77 had no Betfair quotes;
* 10 had only 2 quoted selections;
* 0 failed the SSE gate.

The refusals are listed in `refusals.csv`. Note that the SSE gate is 5e-3 (the
repo default), not the 0.05 written in DESIGN §4.

**Observation model** (DESIGN §2):

```
log λ_home = μ + γ_home + α[h,t] + β[a,t] + ε
log λ_away = μ + α[a,t] + β[h,t] + ε
ε ~ N(0, σ_obs²)
```

Here t is the kick-off week. Identification follows DESIGN §2.2: the latent walk
x̃ is unconstrained and the ratings are its zero-centred projection
(α = (I − 11ᵀ/N) x̃). Every innovation is therefore per-team and independent.

**Arms.** x ∈ {α, β}, with one path per team per component.

| Arm | Law | θ |
|---|---|---|
| a0 static | one rating per team (control) | σ_obs |
| a1 GRW1 | x̃_t = x̃_{t−1} + σ_c ω | σ_obs, σ_att, σ_def |
| a1b GRW1 + break | a1 + extra N(0, σ_break²) in the first week of 25/26 (control) | + σ_break |
| a2 momentum | x̃_t = x̃_{t−1} + ṽ_{t−1} + σ_c ω; ṽ_t = φ ṽ_{t−1} + σ_v η; ṽ_1 = 0 | + φ, σ_v |
| a3 stoch-vol | step sd = exp(h_t), h AR(1) around h̄_c | σ_obs, h̄_att, h̄_def, γ_h, σ_h |
| a4 regime | step sd σ_{c,1} (calm) or σ_{c,1}(1+Δ) (turbulent), Markov P | σ_obs, σ1_att, σ1_def, Δ, p11, p22 |

The priors are in the `log_prior` docstring in the loader. The σ_x prior scale is
DESIGN's HalfNormal(0.10). DESIGN's x₀ ~ N(0, 0.25) is read as a variance, so
the prior sd is 0.5.

**Inference: exact where possible, and no NUTS.**

* **Why no NUTS.** Given the per-team innovation variances, the model is
  linear-Gaussian. So the Kalman filter gives the **exact collapsed
  likelihood** p(y | θ), with μ, γ_home and every rating path integrated out.
  FFBS then draws paths exactly.
* **a0 / a1 / a1b / a2.** Coordinate slice sampling on the collapsed posterior
  of θ.
* **a3 / a4.** Partially collapsed Gibbs, in this order:
  1. θ | aux (collapsed);
  2. paths (FFBS);
  3. per-team volatility: elliptical slice on whitened ξ for SV, FFBS on S for
     the regime arm;
  4. conjugate Beta for P;
  5. an ASIS interweaving step on the volatility hyper-parameters.
* **One-step-ahead prediction.** Exact Kalman for the Gaussian arms. For SV and
  regime, a Rao-Blackwellised particle filter with 20,000 particles, each
  carrying its own Kalman filter.
* **Filtration.** Predictions are **pre-week**: an observation in week t uses
  only data strictly before week t.

**Chains.** Every arm ran 4 chains with 2,000 warm-up iterations and 3,000
retained draws. The thinning differed:

| Arm | Thinning | Sweeps per chain |
|---|---:|---:|
| Gaussian arms (a0, a1, a1b) | 1 | 5,000 |
| a2 momentum | 4 | 14,000 |
| a3 SV, a4 regime | 20 | 62,000 |

## 2. Engine gates

These are independent re-derivations on a toy panel (5 teams, 7 weeks, a gap
week, a same-week double fixture). The reference is the batch joint Gaussian of
every state and observation, written down directly (`mid_gates`,
`engine_gates.csv`). All 11 pass.

| Gate | Value | Tol |
|---|---:|---:|
| Kalman loglik vs batch marginal — GRW1 / momentum / heteroscedastic regime path | 1.6e-13 / 9.6e-14 / 1.1e-14 | 1e-9 |
| RTS smoothed mean vs batch posterior mean — same three | 4.8e-15 / 4.7e-15 / 3.8e-15 | 1e-8 |
| RBPF with a point-mass volatility process vs Kalman (SV, regime) | 3.6e-15 / 3.6e-15 | 1e-8 |
| FFBS 4,000 draws: max \|z\| of mean, max rel. error of covariance | 1.49 / 0.045 | 4.5 / 0.1 |

## 3. Convergence

The gates are R̂ ≤ 1.05 and bulk **and** tail ESS ≥ 200. **22 of 24
parameters pass.** The full table is in `posterior_summary.csv`.

| Arm | max R̂ | min bulk ESS | Verdict |
|---|---:|---:|---|
| a0, a1, a1b | 1.001 | 7,887 | pass |
| a2 momentum | 1.002 | 1,492 | pass |
| a3 stoch-vol | 1.012 | **152** (γ_h, σ_h) | **fails ESS** on γ_h and σ_h; R̂ passes |
| a4 regime | 1.020 | 202 (p22) | pass |

The SV persistence and vol-of-vol parameters mix slowly even with interweaving
and 62k sweeps per chain. Their posteriors are wide anyway (below), and nothing
in the verdict rests on their exact values. The threshold was not relaxed.

## 4. Posterior profiles

Values are medians with [5%, 95%] intervals. Rates are per week, in log-rate
units.

| | a1 GRW1 | a2 momentum | a3 SV | a4 regime |
|---|---|---|---|---|
| σ_obs | 0.093 [0.088, 0.097] | 0.094 | 0.079 [0.073, 0.084] | 0.078 [0.073, 0.083] |
| attack step sd | 0.027 [0.023, 0.031] | 0.017 [0.002, 0.028] | exp(h̄) = 0.015 | calm 0.024 [0.020, 0.028] |
| defence step sd | 0.029 [0.025, 0.034] | 0.020 [0.003, 0.031] | exp(h̄) = 0.022 | calm 0.031 [0.026, 0.036] |
| persistence | — | φ = 0.33 [0.09, 0.63] (prior Beta(2,2)) | γ_h = 0.92 [0.78, 0.98] | p11 = 0.996, p22 = 0.82 |
| other | σ_break = 0.021 [0.002, 0.063] (a1b) | σ_v = 0.013 [0.001, 0.023] | σ_h = 0.38 [0.20, 0.70] | Δ = 11.9 [8.7, 16.5] |

What each arm says:

* **Momentum is not identified.** φ stays near its prior mean. σ_v and σ_att
  trade off against each other, and the momentum arm's likelihood
  (loglik 744.75) equals GRW1's (744.91).
* **The season break is small.** σ_break ≈ 0.02 [0.002, 0.063]. The twelve
  empty summer weeks already carry √12 × 0.027 ≈ 0.09 of drift.
* **Regime durations.** Calm spells last 279 weeks [146, 600]. Turbulence lasts
  5.7 weeks [3.0, 13.0]. The stationary P(turbulent) is 0.020. A turbulent
  step has sd ≈ 0.3–0.4, which is 13× the calm step.
* **Market-rating spread.** The cross-sectional spread of the a1 rating means
  (teams active that season, pooled over both tiers) is 0.14 (attack) and 0.16
  (defence) mid-24/25, and 0.17 and 0.19 mid-25/26.

## 5. Prediction error

**Subsets.** "warm" drops the first 3 weeks of the panel (cold start).
"season-open" is the first 3 weeks of each season. `loglik` is the sequential
log p(y | θ) over the whole panel. For SV and regime it is a particle estimate.
The full table, including the in-season and season-open splits, is in
`prediction_metrics.csv`.

### 10a — θ = full-panel posterior median, states filtered honestly (n = 1,194)

| Arm | RMSE | MAE | mean log pd | 90% cover | in-sample RMSE | loglik |
|---|---:|---:|---:|---:|---:|---:|
| a0 static | 0.1627 | 0.1274 | 0.386 | 0.894 | 0.149 | 442.9 |
| a1 GRW1 | **0.1270** | 0.0960 | 0.646 | 0.920 | 0.076 | 744.9 |
| a1b + break | 0.1270 | 0.0960 | 0.645 | 0.920 | 0.076 | 744.7 |
| a2 momentum | 0.1270 | 0.0960 | 0.645 | 0.918 | 0.078 | 744.8 |
| a3 stoch-vol | 0.1295 | 0.0970 | 0.660 | 0.920 | 0.061 | 767.3 |
| a4 regime | 0.1289 | 0.0961 | **0.677** | 0.930 | 0.061 | **789.7** |

### 10b — honest: θ fitted on 24/25 only, scored on 25/26 only (n = 612)

| Arm | RMSE | MAE | mean log pd | 90% cover |
|---|---:|---:|---:|---:|
| a0 static | 0.1775 | 0.1388 | 0.243 | 0.801 |
| a1 GRW1 | **0.1298** | 0.0954 | 0.620 | 0.917 |
| a1b + break | 0.1306 | 0.0962 | 0.594 | 0.920 |
| a2 momentum | 0.1303 | 0.0958 | 0.616 | 0.915 |
| a3 stoch-vol | 0.1320 | 0.0954 | 0.646 | 0.928 |
| a4 regime | **0.1298** | **0.0943** | **0.668** | 0.931 |

How to read these:

* **Dynamics are worth a lot.** Static → GRW1 cuts honest one-step RMSE by 27%
  (0.178 → 0.130).
* **Beyond GRW1, point error is flat.** Every dynamic arm is within 0.003 RMSE.
  The regime arm's gain is in the **density**: +0.048 nats per observation
  honest, or +29 nats over 612 observations. SV gains +0.026. In other words,
  the extra arms know *when* to be unsure; they do not predict the centre
  better.
* **The honest a1b row is prior-driven.** A single training season contains no
  season boundary, so σ_break sits near its prior and over-widens the 25/26
  opening weeks (log pd 0.20 vs 0.46).
* **One-step error is mostly irreducible here.** The one-step RMSE (0.127) is
  dominated by σ_obs (0.093): the part of each fixture's price that no team
  rating of that week explains.
* **RBPF caveat.** The median particle ESS is healthy (6,000–11,600 of 20,000),
  but it collapses to single digits in a handful of weeks. Those are exactly
  the jump weeks. The SV/regime log-likelihoods are therefore noisy estimates.
  At season-open, both particle arms predict worse than GRW1 (RMSE 0.16–0.18
  vs 0.14 in 10a).

## 6. Where the volatility lives — trajectories and shocks

Figures are in [`results/production/figures/`](results/production/figures/):

* `volatility_calendar.png` — team-mean P(turbulent) and SV σ by week;
* `trajectory_<team>.png` — Hamilton, Inverness CT, Dumbarton and Peterhead,
  all arms overlaid with the GRW1 90% band;
* `insample_fit_a1.png`.

Falkirk and Partick Thistle, named in the brief, are Championship clubs and are
not in tournaments 56/57, so four League One/Two clubs were substituted.

* **Spike and revert, not regimes.** The regime arm's strongest turbulence
  (Hamilton and Kelty defence, weeks 59–61; `regime_shock_weeks.csv`) is one
  fixture: **Kelty v Hamilton, 2025-09-20**. The observed log-rates were
  (0.71, −0.05) against a GRW1 prediction of (−0.03, 0.59), so the market made
  Kelty the favourite. SV and regime model this as a +0.6 jump in Hamilton's
  defence that reverts the next week (see `trajectory_hamilton-academical.png`).
  A persistent regime would not revert. A one-off observation outlier would.
  The sofascore book agrees with Betfair on this fixture, so it is genuine
  pricing of the fixture as listed; a venue change would explain it but is not
  verified.
* **Other turbulent clusters.** These fall at the panel start (Bonnyrigg Rose
  and Edinburgh City, weeks 2–4, again tied to one outlying fixture: Edinburgh
  City v Bonnyrigg Rose, 2024-08-17, idiosyncratic z = +3.3 / −3.7). **Phase 2
  correction:** that fixture has no Betfair 1X2 book; its "market" supremacy is
  the inversion's initial guess ([T015](../../docs/tickets/T015-inversion-accepts-books-without-1x2.md)),
  so this cluster is an artefact, not a market move. The same holds for
  Dumbarton v Inverness CT (2024-10-26) and Elgin City v Clyde (2024-12-17) in the
  anomaly catalogue.
  and in the **last four weeks of 25/26**: Edinburgh City attack, and Queen of
  the South attack and defence (April 2026). That end-of-season cluster is the
  one plausibly genuine regime: dead rubbers, relegation stakes and rotated
  squads.
* **The anomaly catalogue.** `anomaly_catalog.csv` flags 27 observations at
  |z| ≥ 2.5 (24 one-step surprises, 12 idiosyncratic smoothed residuals). That
  is 2.3% of 1,194, against 1.2% expected under Gaussian noise, so the tails
  are heavy. The largest entries are Kelty v Hamilton (z = +5.7 / −4.9) and
  East Kilbride v Edinburgh City, 2026-04-18 (z = −5.1).
* **Data defect found — [T014](../../docs/tickets/T014-betfair-1x2-home-away-swap.md).**
  Cross-checking the Betfair and sofascore 1X2 books over 595 fixtures turned
  up one clean home/away swap in the Betfair book: Montrose v Kelty,
  2025-08-09 (match 14035501). It is in this panel. It is one observation pair
  of 1,246 and immaterial to the conclusions. It was not fixed inline.
* **No momentum in the innovations either.** The Arm 1 check on posterior
  draws of the weekly innovations, pooled over in-season weeks, finds:
  * lag-1 autocorrelation 0.001 (attack) and 0.001 (defence), 90% interval
    [−0.045, 0.041];
  * autocorrelation of squares −0.01;
  * excess kurtosis −0.10.

  This check is **weak**: FFBS draws inherit the GRW1 prior wherever the data
  are thin, and a spike absorbed by σ_obs never reaches the innovations. It
  agrees with the momentum posterior, but it cannot see the tail behaviour
  that §5 does.

## 7. Phase 2 recommendations

In priority order:

1. **Heavy-tailed observations first.** Use a scale mixture
   ε_m ~ N(0, σ_obs²/ω_m) with ω_m ~ Gamma(ν/2, ν/2), i.e. Student-t. Given ω
   the model is still linear-Gaussian, so it drops into this engine as a
   per-observation R: the collapsed Kalman likelihood and FFBS are unchanged,
   and there is one more Gibbs block. If a GRW1 with a t observation matches
   a4's log pd (0.668 honest), the "regime" finding is fully explained by
   outlier fixtures. That is the likely result given §6.
2. **Observation-driven (GARCH-type) volatility instead of SV / MS-GARCH.**
   Let σ²_{i,t} = ω + a·z²_{i,t−1} + b·σ²_{i,t−1}, driven by the team's own
   one-step standardised surprise. The variance is a deterministic function of
   the past, so the **exact Kalman likelihood survives and the particle filter
   disappears**. That removes the ESS collapse in §5 and the slow γ_h/σ_h
   mixing in §3. Only pursue this if (1) leaves a gap. The squared-innovation
   autocorrelation (≈ 0) and the long calm spells (279 weeks) argue that
   clustering is weak.
3. **An end-of-season term.** The one cluster that looks structural is the
   final four weeks of a season. Add a stakes covariate (points to
   promotion/relegation/safety) or a late-season variance inflation. Both fit
   as extra states or per-week D in this engine.
4. **Market covariates as observation weights.** Model R_m = σ_obs²·f(matched
   volume, spread), so thin Betfair books count for less. Implement T014's
   cross-book check first, so the weights are not learned on mirrored books.
5. **Lineup / RAPM synergy.** Add the fixture's lineup differential ΔL_m as an
   observed regressor in the observation equation, with its coefficient as a
   static state. σ_obs = 0.093 is exactly the fixture-level variation that team
   ratings cannot hold, and the obvious candidate for team selection. The size
   of the σ_obs reduction measures how much of the market's fixture-to-fixture
   repricing is the teamsheet.
6. **Feed the goals models.** TODO 021 says the compression is in the team
   latent. This study gives the market's own target dynamics: a no-momentum
   walk with a weekly step of about 0.027–0.029. Richer walk dynamics (momentum,
   fast/slow scales, volatility regimes) are structure the market's ratings do
   not show, so decompression work should target the **level and spread** of the
   ratings rather than their dynamics.

## 8. Caveats

* **θ is plug-in.** Protocol 10a filters the states honestly but estimates the
  3–5 global scales on the full panel. 10b is the honest protocol.
* **The two-tier offset is a gauge direction.** The two tiers are pooled with
  no δ_league offset (DESIGN has none). The tier-level (α+c, β−c) offset is
  identified only through promoted and relegated clubs and the prior. It is
  invisible to every prediction, but individual α/β levels of the two tiers
  are not comparable in absolute terms.
* **Absent clubs drift.** Clubs outside the panel for a season (promoted up or
  relegated out) random-walk under the prior while absent. Their plotted paths
  are masked to active weeks.
* **The regime initial distribution is fixed** at (0.8, 0.2), which keeps P
  conjugate.

## Files

| File | Is |
|---|---|
| `l01_market_inverse_loader.jl` | module `MarketInverseDynamics`: panel, arms, Kalman / FFBS / RTS, samplers, RBPF, diagnostics, gates |
| `r01_market_inverse_runner.jl` | the notebook: sections 1–13 (`MID_SMOKE=1` for a short pass) |
| `results/production/*.csv` | every table above |
| `results/production/figures/*.png` | trajectories, volatility calendar, in-sample fit |
| `results/production/fits.jls` | serialized draws (on the beast only; git-ignored) |
| `r02_market_feature_attribution.jl` | Phase 2 runner: conviction gap, Shapley attribution, feature + Student-t state-space arms |
| `PHASE2_FEATURE_ATTRIBUTION.md` | Phase 2 findings |
| `inputs/fce_fixture_panel_c3bdb53a.csv` | pinned lineup / wealth / travel design from the feature-compression EDA |
| `results/phase2/*.csv`, `results/phase2_run.log` | every Phase 2 table |
