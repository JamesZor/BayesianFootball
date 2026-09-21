# Feature compression & player ratings — Scottish Lower (tournaments 56/57)

**Empirical audit, read-only. Read §0 before quoting a number.**

| | |
|---|---|
| Runner | `current_development/feature_compression_eda/r01_feature_compression_eda.jl` |
| Loader | `current_development/feature_compression_eda/l01_feature_compression_loader.jl` |
| Tables | `current_development/feature_compression_eda/results/*.csv` — one CSV per table below |
| Cohort | 40 walk-forward folds, seasons 24/25 + 25/26, **710 held-out fixtures**, **623** with an invertible closing book, **596** with a complete de-vigged 1X2 |
| Executed | locally on `archpc`, 2026-09-21 (`mcmc-beast` was saturated by the `d09f_grid` run) |

---

## 0. Evidence contract

**Nothing here was sampled.** Two stored posteriors are read from
`mcmc_experiments` **by UUID**, because both names have newer rows:

| Arm | Run UUID | Recipe |
|---|---|---|
| `m12_joint_hybrid_synergy` | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | team time decay + shots-RAPM lineup (bench 0.10) + production wealth |
| `m05_joint_production_wealth` | `ed541a7c-01e2-447e-a771-783517728d47` | identical **minus the lineup pillar** — the attribution control |

Both were later extended to 43 folds; this study uses **folds 1–40 only**, the
canonical 710-fixture cohort. Feature sets are rebuilt from the cached
`ScottishLower` DataStore with the Experiment 06 splitter, so fold *k* here is
fold *k* there, and every RAPM fit (production and sweep) is fit on that fold's
frozen history block (`fit_on = :history`).

**The control is the TIME-DECAY m05, not the GRW m05.** The brief's 0.5515–0.5779
slope belongs to `m05_joint_production_wealth_grw` (`f870dbb7-…`,
`MultiScaleGRW`, a different experiment namespace). The control used here differs
from m12 in exactly one component, which is what makes a difference between them
attributable to that component. Any comparison with the GRW arm in this report is
flagged as out of scope, not reproduced.

**What this is not.** Not a fit, not an evaluation, not a portfolio study. No
proper score, Kelly stake or P&L appears here, and nothing below establishes that
decompressing the model would be profitable. The closing line is a yardstick, not
a target and not a feature.

**Verification.** The four-term decomposition of §1 reproduces the stored latents
exactly: over all 710 fixtures,
`max |Σ terms − log(λ_h/λ_a)| = 1.67e-16`, `mean = 1.05e-17`.

**Market yardstick.** Betfair TWA close over (−20 m, 0 m], proportionally de-vigged
within each (match, market, line), then inverted to `(λ_mkt_h, λ_mkt_a)` by the
production inverter `Calibration.invert_market_rates`. 623 of 710 fixtures invert;
27 of those have no complete 1X2, so probability-scale tables have n = 596.

**Two named fixtures in the brief are outside this cohort.** Hamilton vs Queen of
the South and Ross County vs Cove Rangers are 26/27 slate fixtures; Ross County and
Queen of the South do not appear in tournaments 56/57 in 24/25–25/26 at all. §7
reports the cohort's own heaviest favourites instead.

---

## 1. One scale for both sides

The model's supremacy for a fixture is four scalars and nothing else:

```
η_h − η_a  =  γ  +  (α_h − α_a) + (β_a − β_h)  +  (w_att + w_def)·ΔL  +  2·w_W·ΔW
              HA         team attack/defence        lineup RAPM pillar     production wealth
```

`ΔL` is the bench-weighted lineup differential the pillar multiplies (starters +
0.10 × bench, home − away); `ΔW` is the age-weighted log wealth ratio. The market's
supremacy is `log(λ_mkt_h / λ_mkt_a)`. Both are log-rate differences, so a
regression of one on the other has an interpretable unit slope — unlike a
logit-on-logit regression, whose identity slope is not one.

Three hypotheses were put to the data:

* **H1 — double shrinkage**: ridge `λ = 1000`, then `w_att, w_def ~ N(0, 0.3)`.
* **H2 — variance cannibalisation**: `α/β`, `ΔL`, `ΔW` all measure team quality.
* **H3 — wealth saturation**: `RichardsSigmoid` flattens the wealth tails.

**Result in one line: H2 holds and dominates; H1 and H3 do not survive contact
with the data; and the compression is concentrated in the team latent `α/β`, which
the closing line wants amplified 2.4×.**

---

## 2. The compression, measured (`r01_supremacy_slopes.csv`, `r01_probability_tails.csv`)

OLS of each model quantity on market supremacy, n = 623:

| Quantity | slope | s.e. | R² | sd(quantity) | sd(market) |
|---|---:|---:|---:|---:|---:|
| **m12 total supremacy** | **0.3280** | 0.0142 | 0.463 | 0.2042 | 0.4234 |
| m12 team term `α/β` | 0.1592 | 0.0087 | 0.349 | 0.1141 | — |
| m12 lineup term | 0.1147 | 0.0146 | 0.091 | 0.1613 | — |
| m12 wealth term | 0.0536 | 0.0061 | 0.109 | 0.0687 | — |
| **m05 total supremacy (control)** | **0.3575** | 0.0118 | 0.595 | 0.1962 | 0.4234 |
| m05 team term `α/β` | 0.2981 | 0.0107 | 0.556 | 0.1692 | — |

1. **The model ranks well and scales at a third.** Correlation with the market is
   0.68; the slope is 0.328 and the supremacy spread is 48% of the market's.
2. **Dropping the player pillar does not decompress the time-decay model.** The
   control's spread is 0.196 vs m12's 0.204 and its slope 0.358 vs 0.328 — under
   10%, not the +82% reported for the GRW arm. Whatever produces that GRW jump is
   the dynamics component, not the presence of the lineup pillar.

Probability tails (independent Poisson over the stored goal-rate draws; the goals
arm of the joint observation):

| Cohort | n | market mean | m12 mean | m05 mean |
|---|---:|---:|---:|---:|
| home favourite ≥ 0.60 | 51 | 0.675 | 0.496 | 0.503 |
| home favourite ≥ 0.70 | 16 | 0.767 | 0.511 | 0.530 |
| home favourite ≥ 0.80 | 3 | 0.858 | 0.566 | 0.586 |
| home longshot < 0.15 | 9 | 0.127 | 0.303 | 0.292 |
| **maximum over the cohort** | 710 | — | **0.578** | **0.604** |

The 57.8% maximum reproduces the shrinkage-decompression audit exactly. It is an
observed maximum over this cohort, **not** a support bound — nothing in a Gaussian
team effect forbids a higher probability.

---

## 3. H1 — the two-stage variance loss (`r01_variance_chain.csv`, `r01_prior_vs_posterior.csv`)

### 3.1 The chain, stage by stage (means over 40 folds)

| Stage | Quantity | Value |
|---|---|---:|
| 0 — raw plus-minus | `sd(r_raw)`, all players | 6.218 shots/90 |
| 0 — raw plus-minus | `sd(r_raw)`, ≥ 450 min | 2.785 shots/90 |
| 1 — ridge RAPM (λ = 1000) | `sd(r̂)`, all players | 0.0524 shots/90 |
| 1 — ridge RAPM (λ = 1000) | `sd(r̂)`, ≥ 450 min | 0.0606 shots/90 |
| 1 → aggregation | `sd(ΔL)` on held-out fixtures | 0.366 |
| 2 — Bayesian weight | `sd((w_att + w_def)·ΔL)` | 0.157 log-rate |
| reference | `sd(market supremacy)` | 0.415 log-rate |

Stage 1 removes **99.15%** of the raw dispersion (`σ(r̂)/σ(r_raw) = 0.0085`). On its
own that number proves nothing: the raw plus-minus is noise-dominated (a 90-minute
cameo in a 2-shot swing scores ±2.0). §5 asks whether the removed dispersion was
signal.

### 3.2 The Bayesian prior is not the binding constraint

With `w_att, w_def ~ N(0, 0.3)` independent, the prior alone permits
`sd((w_att + w_def)·ΔL) = √2 · 0.3 · sd(ΔL) = 0.155`. The posterior delivers
**0.157** — the lineup term is realised *at* the prior's own scale.

| Parameter | prior | posterior mean | posterior sd (within fold) | fold 1 | fold 40 |
|---|---|---:|---:|---:|---:|
| `lineup.w_att` | `N(0, 0.3)` | 0.187 | 0.050 | 0.212 | 0.126 |
| `lineup.w_def` | `N(0, 0.3)` | 0.240 | 0.062 | 0.257 | 0.240 |
| `production_wealth.w` (m12) | `truncated(N(0.10, 0.05), 0, ∞)` | 0.105 | 0.036 | 0.077 | 0.130 |
| `production_wealth.w` (m05) | same | 0.115 | 0.038 | 0.088 | 0.135 |

The within-fold posterior sd of `w_att` is **0.050 against a prior sd of 0.300** —
the likelihood is six times sharper than the prior, so the prior is not what pins
`w_att`. The documented 0.212 → 0.126 decline is real, non-monotone, and coincides
with the wealth weight *rising* 0.077 → 0.130: two terms trading one variance,
which is H2, not H1.

The wealth prior is the one prior in the recipe that *is* informative
(`truncated(N(0.10, 0.05), 0, ∞)`, sd 0.047, posterior means 0.105/0.115, i.e.
2.2–2.4 prior sds). It is not binding either — the posterior sits above the prior
mean — but it is the only weight whose prior is of the same order as its
likelihood.

### 3.3 The rating is not just shrunk, it is stale (`r01_history_blocks.csv`)

`fit_on = :history` fits the ridge on the fold's frozen history block — the last
two complete seasons — not on everything before kickoff. Across the 40 folds there
are **exactly 2 distinct history blocks**:

| Block | folds | history matches | last history match | rating age at kickoff |
|---|---:|---:|---|---|
| 1 | 20 | 720 | 2024-05-04 | 91 – 357 days |
| 2 | 20 | 720 | 2025-05-03 | 91 – 357 days |

So a player's rating is **identical for all 20 folds of a season** and is between 3
and 12 months old when it prices a fixture. Within a season the only thing that
moves `ΔL` is the teamsheet. Any in-season improvement, injury, transfer or loss of
form is invisible to the pillar by construction. This is a leak-safety choice, not
a bug — but it bounds how much fresh supremacy the pillar could ever carry, and it
is a far larger effect on the *information* in `ΔL` than the ridge penalty is.

**Verdict on H1 (prior half): not supported.** The ridge half is settled in §5.

---

## 4. H2 — the pillar moves variance, it does not add it

| Quantity | m05 (no pillar) | m12 (pillar) | change |
|---|---:|---:|---:|
| `sd(total supremacy)` | 0.196 | 0.204 | **+4%** |
| `sd(team term α/β)` | 0.169 | 0.114 | **−33%** |
| `sd(lineup term)` | — | 0.161 | — |
| `sd(wealth term)` | 0.075 | 0.069 | −9% |
| posterior `dyn.σ_a` (mean over folds) | 0.092 | 0.053 | **−43%** |
| posterior `dyn.σ_d` (mean over folds) | 0.139 | 0.102 | −27% |
| slope on market supremacy | 0.358 | 0.328 | −8% |

Adding a term whose own spread is 0.161 raises the total by 0.008. The team
parameters absorb the difference — `dyn.σ_a`, the spread of the team attack
effects, is cut by 43%.

The correlations say it more sharply (n = 623):

| Pair | r |
|---|---:|
| `ΔL` vs team contrast **in the model without the pillar** (m05) | **+0.524** |
| `ΔL` vs team contrast **in the model with the pillar** (m12) | **−0.176** |
| `ΔL` vs `ΔW` | +0.180 |
| m12 team contrast vs `ΔW` | +0.004 |
| market supremacy vs m12 total supremacy | +0.680 |

Without the pillar the fitted team effects *carry* the lineup signal (r = +0.52);
with the pillar they turn slightly against it (r = −0.18). The model subtracts from
`α/β` what it has just credited to `ΔL`. Both are estimated from the same goals, so
the likelihood splits one quantity of in-sample variation between them, and the
split is close to zero-sum.

**Verdict on H2: supported, and dominant.** The model's supremacy budget is
conserved near 0.20 log-rate units however many quality features are added, against
a market spread of 0.42.

---

## 5. Ridge λ sensitivity — the penalty is not the problem (`r01_lambda_sweep_pooled.csv`)

Same folds, same aggregation, same fixtures; only λ moves. `cor` and `partial cor`
are against market supremacy, the latter after partialling out the team contrast
and `ΔW`; `ols weight` is the univariate market-implied coefficient on `ΔL`.

| λ | sd(r̂) | sd(ΔL) | cor(ΔL, market) | partial cor | market-implied weight | model's own weight |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.7536 | 5.686 | 0.030 | −0.015 | 0.0022 | — |
| 10 | 0.6907 | 2.572 | 0.198 | 0.175 | 0.0325 | — |
| 50 | 0.3020 | 1.412 | 0.276 | 0.349 | 0.0828 | — |
| 100 | 0.2095 | 1.122 | 0.296 | 0.417 | 0.1118 | — |
| 250 | 0.1283 | 0.807 | 0.306 | 0.470 | 0.1603 | — |
| 500 | 0.0848 | 0.579 | 0.307 | 0.488 | 0.2242 | — |
| **1000 (production)** | **0.0524** | **0.376** | **0.308** | **0.495** | **0.347** | **0.427** |
| 2000 | 0.0301 | 0.222 | 0.312 | 0.497 | 0.5961 | — |

Three readings:

1. **Weak penalties destroy the signal, they do not release it.** At λ = 1 the
   ratings have 33× the production spread and a correlation with market supremacy
   of 0.03 — pure noise. The information appears only as λ rises.
2. **λ = 1000 is on the plateau.** Correlation is flat from λ ≈ 250 (0.306 → 0.312
   at λ = 2000) and the partial correlation likewise (0.470 → 0.497). Lowering λ
   strictly *loses* information; raising it is neutral to marginally positive. The
   scale of `ΔL` is not identifiable from this diagnostic, because the model's
   `w_att`/`w_def` rescale it — which is exactly why the spread argument fails.
3. **The model's weight on `ΔL` is already the market's weight.** At λ = 1000 the
   market-implied univariate coefficient is 0.347 ± 0.043 and the posterior
   `w_att + w_def` is 0.427; in the multivariate fit of §6 the market wants 0.425
   against the model's 0.427 — a ratio of **1.00**.

**Verdict on H1 (ridge half): not supported.** `λ = 1000` is not destroying
usable player differentiation; at lower λ the extra dispersion is noise. The
pillar's *scale* is right; its limitation is staleness (§3.3) and redundancy (§6).

---

## 6. Collinearity and what the market would weight (`r01_design_vif.csv`, `r01_team_absorption.csv`, `r01_market_implied_weights.csv`)

### 6.1 The fixture-level design is *not* ill-conditioned

| Feature | VIF |
|---|---:|
| team `α/β` contrast (m12 posterior mean) | 1.033 |
| lineup `ΔL` | 1.068 |
| production wealth `ΔW` | 1.036 |
| travel `log_dist_z` | 1.001 |

Scaled condition number **1.29**; the largest Belsley condition index is 1.29 and
no index carries > 0.5 of two columns' variance. By the classical diagnostics there
is **no multicollinearity problem at all** — which is the diagnostic trap here: the
fitted team contrast has already been orthogonalised against `ΔL` by the very
cannibalisation of §4. VIF computed on a posterior output cannot see the competition
that produced it.

### 6.2 The collinearity that matters is with team identity

Regressing each fixture-level difference on the signed team-indicator design
(+1 home, −1 away, one column per club) over each fold's training matches:

| Feature | mean R² on team identity | range |
|---|---:|---|
| lineup `ΔL` | **0.890** | 0.874 – 0.922 |
| production wealth `ΔW` | 0.547 | 0.466 – 0.663 |

**89% of the lineup differential is team identity.** A club's RAPM total is nearly a
club constant — which is what one should expect from a rating that is frozen for a
whole season (§3.3) and aggregated over an XI drawn from one squad. Only ~11% of
`ΔL` is the rotation/selection information that `α/β` cannot already express, and it
is that 11% the pillar can contribute without competing with the team effects. The
wealth differential is about half team identity.

### 6.3 What the closing line would weight

OLS of market supremacy on the same four columns (n = 623, R² = 0.590):

| Term | market coefficient | s.e. | t | model's own coefficient | ratio market / model |
|---|---:|---:|---:|---:|---:|
| (intercept — home advantage) | 0.1759 | 0.0118 | 14.9 | γ = 0.124 | 1.42 |
| **team `α/β` contrast** | **2.4341** | 0.0972 | 25.0 | 1 (by construction) | **2.43** |
| lineup `ΔL` | 0.4246 | 0.0300 | 14.2 | `w_att + w_def` = 0.427 | **1.00** |
| production wealth `ΔW` | 0.3329 | 0.0340 | 9.8 | `2·w_W` = 0.210 | 1.58 |
| travel `log_dist_z` | 0.0253 | 0.0137 | 1.8 | 0 (not in m12) | — |

This is the report's central table. Read as "what multiple of each term the closing
line would have used":

* the **team latent is under-weighted by a factor of 2.4** — this is where the
  compression lives;
* the **lineup pillar is weighted correctly** (1.00) — it is neither double-shrunk
  nor free to grow;
* **wealth is under-weighted by ~1.6×**, consistent with its informative prior;
* **home advantage is under-weighted by ~1.4×** (0.124 vs 0.176), a small but
  systematic contribution to favourite underpricing, since the home side is the
  favourite in most heavy-favourite fixtures;
* travel distance adds nothing significant (t = 1.8), supporting its absence from
  m12.

**Caveat.** `sup_team` is a posterior output fitted jointly with the other terms, so
these coefficients are rescaling factors the close would want *given this
decomposition*, not causal effects and not a recipe that can be applied by
multiplying a fitted chain.

---

## 7. Where the favourites are lost (`r01_favourite_attribution.csv`, `r01_supremacy_by_decile.csv`, `r01_counterfactual_rescale.csv`)

The cohort's heaviest market favourites (top 8 of the 15 in the CSV):

| Date | Fixture | market sup | model sup | HA | team | lineup | wealth | P(home) mkt | P(home) m12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04-18 | east-kilbride vs edinburgh-city | 2.013 | 0.541 | 0.116 | 0.183 | −0.040 | 0.282 | 0.917 | 0.561 |
| 2025-10-18 | inverness vs kelty-hearts | 1.584 | 0.633 | 0.138 | 0.094 | 0.309 | 0.092 | 0.855 | 0.568 |
| 2026-04-18 | inverness vs kelty-hearts | 1.473 | 0.656 | 0.116 | 0.307 | 0.219 | 0.014 | 0.804 | 0.568 |
| 2024-08-10 | east-fife vs edinburgh-city | 1.343 | 0.361 | 0.115 | 0.186 | 0.016 | 0.044 | 0.781 | 0.496 |
| 2026-02-07 | inverness vs east-fife | 1.332 | 0.361 | 0.096 | 0.134 | 0.116 | 0.016 | 0.775 | 0.477 |
| 2026-03-17 | inverness vs peterhead | 1.289 | 0.223 | 0.092 | 0.220 | −0.052 | −0.037 | 0.760 | 0.432 |
| 2026-03-07 | inverness vs montrose | 1.252 | 0.651 | 0.087 | 0.176 | 0.375 | 0.013 | 0.760 | 0.578 |
| 2025-11-22 | hamilton-academical vs kelty-hearts | 1.146 | 0.607 | 0.123 | 0.063 | 0.290 | 0.131 | 0.740 | 0.564 |

In the fixture the market rates 0.917, the model's entire supremacy is 0.541, of
which the *team* contrast contributes 0.183 and the lineup −0.040. Attribution by
market decile:

| Decile | n | market sup | model sup | team | lineup | wealth | P(home) mkt | P(home) m12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63 | −0.577 | −0.122 | −0.111 | −0.102 | −0.032 | 0.203 | 0.335 |
| 5 | 62 | 0.146 | 0.115 | −0.018 | −0.006 | 0.014 | 0.409 | 0.408 |
| 8 | 62 | 0.463 | 0.230 | 0.039 | 0.052 | 0.018 | 0.513 | 0.445 |
| 9 | 62 | 0.586 | 0.219 | 0.065 | 0.006 | 0.024 | 0.554 | 0.441 |
| 10 | 63 | 0.922 | 0.375 | 0.124 | 0.087 | 0.041 | 0.659 | 0.489 |

Every term keeps its sign and keeps growing into the tail — nothing saturates —
but all of them grow **too slowly**. Between decile 5 and decile 10 the market moves
0.78 log-rate units and the model 0.26.

**The ceiling is a supremacy-spread property, not a total-rate property.** Holding
each fixture's total rate `μ = √(λ_h λ_a)` fixed and rescaling only the supremacy by
1/0.328 = 3.05:

| Cohort | n | market | m12 | m12 with supremacy ×3.05 |
|---|---:|---:|---:|---:|
| home favourite ≥ 0.70 | 16 | 0.767 | 0.511 | **0.760** |
| maximum over the cohort | 623 | — | 0.578 | **0.900** |

A single scalar rescale of the supremacy reproduces the market's favourite
probabilities almost exactly. This is an arithmetic counterfactual, **not a
proposal**: it is fitted to the close it is compared against, and nothing here shows
it would improve a proper score or a bankroll.

---

## 8. H3 — the wealth transform (`r01_age_curve.csv`, `r01_wealth_transform_contrast.csv`)

**The premise of H3 is wrong about the code.** `RichardsSigmoid(23.0, 0.80, 2.0)` is
an *age*-productivity curve applied to each player's market value inside the squad
sum; the differential itself is a plain log ratio,
`log Σ(value·ϕ(age))_h − log Σ(value·ϕ(age))_a`. There is no sigmoid on the wealth
differential and no saturation of large disparities.

What the curve does:

| age | 16 | 18 | 20 | 23 | 26 | 29 | 33 | 37 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ϕ(age) | 0.061 | 0.134 | 0.288 | 0.707 | 0.958 | 0.996 | 1.000 | 1.000 |

It is a youth discount that is flat from ~26 onward — it does not compress the tails
of a wealth gap between a rich and a poor club; it discounts teenagers.

Contrast with the raw log-sum on the same 623 fixtures:

| Transform | sd | p05 | p95 | max abs | cor(market) | partial cor (given team) | univariate market weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| production wealth (age-weighted) | 0.327 | −0.492 | 0.550 | 1.197 | 0.326 | **0.402** | 0.423 |
| raw log-sum wealth | 0.249 | −0.406 | 0.407 | 0.735 | 0.292 | 0.274 | 0.497 |

Correlation between the two transforms: 0.829. The age-weighted differential has
**31% more spread, 63% wider tails and a higher partial correlation with market
supremacy** than the raw log-sum. The transform is not the compressor — if anything
it is the better of the two.

**Verdict on H3: not supported.** The wealth term is small (sd 0.069 of a 0.204
total) because its *coefficient* is small (`2·w_W = 0.21` against a market-implied
0.33), not because its feature is squashed.

---

## 9. Recommendations for the feature layer

Ordered by expected effect on the measured compression. Each is a hypothesis for
Stage 2, with the measurement that would settle it — none is established here.

1. **Stop looking for the compression in the feature layer; it is in `α/β`.**
   The market wants the team contrast amplified 2.4×, the lineup weight left alone,
   and wealth raised by ~1.6×. The candidates that follow from that are dynamics
   candidates: heavier-tailed team innovations, a hierarchical or longer-memory team
   state, a per-team `σ`, or simply a weaker shrinkage on the team effects. The
   existing Experiment 09 candidate list is pointed at the right layer.
2. **Refresh the RAPM inside the season.** Two distinct rating vectors over 40 folds
   (§3.3) is a design choice, not a statistical necessity: a rating fit on "every
   match strictly before this fold's first fixture" is equally leak-free and up to a
   year fresher. Measure it the way §5 does — correlation and partial correlation of
   `ΔL` with market supremacy — before any refit.
3. **Give the pillar the 11% that is not team identity.** Since `ΔL` is 89% club
   constant, the informative part is the deviation from the club's own baseline.
   Decomposing `ΔL` into a club mean plus a within-club deviation, and giving the
   deviation its own weight, would let the pillar price rotation and absence without
   competing with `α/β` for the club-level variance. The club-mean part can then be
   dropped or explicitly shared with the team effect.
4. **Raise the wealth weight or widen its prior.** `truncated(N(0.10, 0.05), 0, ∞)`
   is the only informative prior in the recipe, the posterior already sits 2.2 prior
   sds above the mean, and the market wants ~1.6× more. A wider prior costs nothing
   in AD terms and is a one-line change.
5. **Do not lower the ridge λ.** §5 shows λ = 1000 is on the plateau and λ ≤ 100
   destroys the signal. If λ is touched at all, the evidence points *up*, and the
   pre-registered WP7 decision rule (split-half reliability plus retrodiction)
   should be re-run before changing the production value.
6. **Consider a home-advantage look.** `γ = 0.124` against a market-implied 0.176
   is a systematic under-statement that affects every home favourite. This overlaps
   TODO 008 (hierarchical HA), which should be measured on this same yardstick.

---

## 10. Limits of this report

* Everything is measured against the **closing line**, which is a benchmark, not
  truth. "The market would weight `α/β` 2.4× more" is a statement about agreement
  with the close, not about out-of-sample accuracy. None of the recommendations is
  validated until a candidate is fitted and scored.
* The market-implied weights of §6.3 are regressions on a posterior output; they
  identify a rescaling, not a causal structure.
* Probability tables use an independent-Poisson grid over the stored goal-rate
  draws, not the production score-grid kernels with the copula/finishing terms.
  They are diagnostics, and they reproduce the audited 57.8% figure exactly.
* The GRW arm (`m05_joint_production_wealth_grw`) is **not** analysed here. The
  +82% decompression attributed to removing the lineup pillar is not reproducible
  on the time-decay pair (§2), so that comparison confounds the dynamics change
  with the pillar change and should be re-run as a one-factor contrast.
* `n = 623` inverted books out of 710; 596 complete 1X2. Fixtures without a book
  are absent from every market-referenced table, not imputed.

## 11. Reproducing

```bash
julia --project -t 6
julia> include("current_development/feature_compression_eda/r01_feature_compression_eda.jl")
```

Needs `BF_DB_URL` (betdb, for the cached DataStore refresh) and
`BF_EXPERIMENTS_DB_URL` or `~/.pgpass` (mcmc_experiments, for the two runs). Wall
time ≈ 35 minutes on `archpc`, dominated by the λ sweep; every table is written to
`current_development/feature_compression_eda/results/`.
