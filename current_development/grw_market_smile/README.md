# Task 015 — MultiScaleGRW with Market Supremacy and Market Smile anchoring

> Status: **complete (2026-09-13).** Converged on all four anchored rungs; **no proper-score gain
> (0 of 84 contrasts significant)**; home-favourite compression narrowed only slightly; on the
> Option B closing line every anchored rung out-returns the baseline (+405% to +588% vs +386%,
> ROI 12.8–15.8% vs 11.7%) with overlapping growth intervals; **at the tradeable T−25 book the smile
> keeps a ROI lead but loses its bankroll lead**, raw and calibrated (paired growth p(better)
> 0.23–0.51); the smile's φ never sizes a stake in Portfolio (T011); fringe trust stays pruned, with
> Under 1.5 / 4.5 the only consistent-sign candidates; the 2026-09-12 card would still have lost,
> mostly less by staking less. Every number in this file is
> copied from a runner report in `results/`.

## The question

On 2026-09-12 the live account `live_scottish_m12_500` priced the Scottish League One/Two card
from Run 67 (TimeDecay hybrid) and closed −£45.89. The model priced every home side into a
0.40–0.43 band while the market priced the favourites at 0.55–0.60, and the account backed the
away sides. Task 008 showed a per-club home advantage moves that band by at most 0.003.

This task asks whether anchoring the Gen 3 team-level GRW model to the closing market — the
Ireland dual pillar — closes that gap **out of sample**, and whether it would have changed the card.

### Hypotheses, written so they can fail

| | claim | refuted if |
|---|---|---|
| H1 | the supremacy pillar improves 1X2 LogLoss over the pure GRW baseline | paired Δ(supremacy_w040 − baseline) on 1X2 has an interval including 0 or above it |
| H2 | the smile adds to supremacy on totals | Δ(smile_w040 − supremacy_w040) on O/U 2.5 includes 0 or is positive |
| H3 | home-favourite compression shrinks | in the Betfair p_home ≥ 0.50 bins the anchored rungs' mean p_model − p_market is no closer to 0 than the baseline's |
| H4 | there is a better weight than Ireland's 0.40 | neither 0.20 nor 0.70 beats 0.40 with an interval excluding 0 |

**The prior to argue against.** The Betfair close already beats every football model on this
panel (0.64182 vs the baseline's 0.64315). A model pulled toward the close can gain LogLoss just by
becoming more like the benchmark, so every arm is also scored *against the close*: beating the
baseline is not beating the market.

## The ladder

Foundation = Task 013 `m05_wealth_grw`, verbatim: `GlobalInterception + MultiScaleGRW +
GlobalHomeAdvantage + ProductionWealthCovariate(SupremacyRole) + JointGammaPoissonObservation`.

| rung | C1 supremacy | C2 smile | run |
|---|---|---|---|
| `m05_joint_grw_baseline` | — | — | Task 013 `b0961bc4-c40c-4dbe-9c05-57df7ae0839e` (pinned; identical recipe, asserted) |
| `m05_joint_grw_supremacy_w040` | 0.40 | — | pending |
| `m05_joint_grw_smile_supremacy_w020` | 0.20 | 0.20 | pending |
| `m05_joint_grw_smile_supremacy_w040` | 0.40 | 0.40 | pending |
| `m05_joint_grw_smile_supremacy_w070` | 0.70 | 0.70 | pending |

```
C1   η_h − η_a                           ~ Normal(log λ̂_h − log λ̂_a, σ_sup)      × w_sup
C2   log κ + log(μ_h + μ_a) + log φ_K    ~ Normal(log Λ̂_K, σ_smile),  K = 0…4    × w_smile
σ_sup, σ_smile ~ truncated(Normal(0.15, 0.10), lower = 0.02)     log φ ~ Normal(0, 0.5)^5
```

λ̂ is the double-Poisson inversion of the closing book (`DoublePoissonMarketFeature`), Λ̂_K the
Poisson rate implied by the de-vigged closing Under K.5 (`MarketSmileFeature(Kmax = 4)`). Both
pillars read **training** fixtures only. κ enters C2 and not C1 because the joint model prices
goals at λ = κμ: a ratio cancels κ, a level does not. O/U prices through `λ_tot·φ(K)`
(`SmileLatents`); 1X2 and BTTS through the (λ_h, λ_a) grid.

## Files

| file | role |
|---|---|
| `l01_loader.jl` | `MarketAnchoredCountModel` wrapper + engine, pillars, ladder, gates, persistence |
| `l02_evaluation.jl` | book, panel, `SmileLatents` restriction, scores, fixture-clustered bootstrap, compression table |
| `l03_slate.jl` | replay-engine helpers for the 2026-09-12 card (carried from Task 008) |
| `r01_smoke.jl` | G0–G6 on folds 1–2 for baseline / supremacy / smile @ 0.40 |
| `r02_production_grid.jl` | 43-fold grid for the four anchored rungs |
| `r04_evaluate.jl` | proper scores, paired Δ, compression, vs the close |
| `r05_slate_repricing.jl` | T−25 counterfactual through the replay engine |
| `r06_portfolio.jl` | Option B closing-line portfolio, 5 arms on one buildable panel, with smile-routing gate |
| `l04_portfolio_calibration.jl` | T−25 book, calibrated smile sources (pooltot / grid), paired slate-growth bootstrap, trust-sweep policies |
| `r07_t25_calibrated_portfolio.jl` | close and T−25, raw vs Option B calibrator, 5 arms, paired contrasts Q1–Q4 |
| `r08_trust_sweep.jl` | fringe trust sweep on smile @0.40 vs baseline, close and T−25 |

## Contracts

* **Why a wrapper.** The composable engine types its config as `ComposableCountModel` and hides
  `obs.log_κ` inside `_observe`. `MarketAnchoredCountModel` calls the builder's own submodels in
  the builder's declaration order; G0 requires its log density with both pillars off to be
  **bit-identical** to the Task 013 builder model.
* **Split.** `GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)` over
  24/25–26/27: 43 folds, of which 1–40 are asserted identical to the canonical 40-fold split. r04
  restricts every arm to the 710-fixture 24/25 + 25/26 panel. Fold 43 priced the live card.
* **Snapshot.** `datastore_ScottishLower.jls` of 2026-09-12 00:40 (before the card).
* **Sampler.** Smoke 4 × (500 + 500); production 4 × (500 + 1,000), δ = 0.80, max depth 10 —
  Task 013's budget, so the pinned baseline is sampled identically. The baseline run's *recorded*
  sampler reads 500 retained draws: Task 013 persisted every 2nd draw and `extend_fit` re-recorded
  the thinned count when it added folds 41–43. `extend_fit` also re-audits every fold from the thinned
  chains (`extension.jl:365`), so the run's own fold-1 audit now counts 4 × 500 = 2,000 transitions.
  r02 therefore checks the sampled budget against Task 013's committed pre-extension report —
  160,000 transitions over 40 folds = 4 × 1,000 per fold — plus the recorded sampler at
  `samples ÷ stride`, the fold-1 re-audit at 2,000, and 2,000 persisted draws per fixture, the same
  as every candidate persists.
* **Persistence (T010).** `PostgresStorage` stores `CountLatents` only and rebuilds every panel as
  one on load. Smile runs are saved with the panel detached and rebuilt from the persisted chains;
  the round-trip gate requires equality field for field. See
  [`docs/tickets/T010`](../../docs/tickets/T010-postgres-storage-refuses-smile-latents.md).
* **Option B on a smile rung** calibrates the grid (1X2/BTTS) but not `λ_tot·φ` (O/U). The raw arms
  are the clean re-pricing comparison.

## Smoke gates (r01)

Folds 1–2, pooled 56/57, store through 2026-09-05. Reports:
`results/smoke/4x500w500s/` (work package budget) and `results/smoke/4x500w1000s/`
(production budget). Verdict at the production budget: **PASS, 3 / 3 rungs.**

### G0 — likelihood parity (identical at both budgets)

| case | fold | base sites | pillar sites | Δ(model − base) | gap to re-derivation (rel) |
|---|---:|---:|---|---|---:|
| both slots empty | 1 | 13 | — | **0.0 exactly** at 4 prior draws | 0 |
| both slots empty | 2 | 17 | — | **0.0 exactly** | 0 |
| supremacy @0.40 | 1 / 2 | 13 / 17 | σ_sup | up to 8.4e3 | 4.5e-16 / 6.0e-16 |
| smile + supremacy @0.40 | 1 / 2 | 13 / 17 | σ_sup, σ_smile, log_φ | up to 4.5e4 | 1.1e-15 / 1.3e-15 |

The first attempt displaced one prior draw by δ·cos(i); under MultiScaleGRW that steps truncated
scales out of support and both models returned −Inf. That was a defect in the gate, not the model;
the gate now uses four independent prior draws.

### G1 — gradient audit

| rung | fold | θ | tape | Δtape | grad ms | alloc B | Δalloc | RD vs FD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 101 | 791 | 0 | 0.086 | 128,752 | 0 | 4.0e-16 |
| baseline | 2 | 161 | 1,365 | 0 | 0.104 | 132,992 | 0 | 5.7e-16 |
| supremacy | 1 | 102 | 811 | +20 | 0.093 | 175,408 | +46,656 | 9.3e-16 |
| supremacy | 2 | 162 | 1,385 | +20 | 0.115 | 181,184 | +48,192 | 3.5e-17 |
| smile + supremacy | 1 | 108 | 843 | +52 | 0.175 | 395,120 | +266,368 | 3.6e-16 |
| smile + supremacy | 2 | 168 | 1,417 | +52 | 0.197 | 407,296 | +274,304 | 1.6e-16 |

The compiled tape is exact at three perturbed points on every row. **The work package's "zero heap
allocations" is not met by any rung, the baseline included** (129–133 KB per call); the smile pillar
costs ~2× gradient time and ~270 KB per call, most likely the `n × 5` matrix broadcast. Recorded, not
tuned away.

### GB — market coverage

The supremacy pillar reads 719 / 720 (fold 1) and 739 / 740 (fold 2) training matches; the smile
pillar reads 705 / 720 and 725 / 740, 641–718 per strike (K = 2 has the most quotes).

### G2–G5 — sampling, latents, persistence

| budget | rung | R̂ | ESS bulk / tail | div | σ_sup | σ_smile | κ | φ(K = 0…4) | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 4 × (500 + 500) | baseline | 1.0125 | 615 / **363** | 0 | — | — | 1.123 | — | **FAIL** (tail ESS) |
| 4 × (500 + 500) | supremacy | 1.0083 | 859 / 914 | 0 | 0.210 | — | 1.117 | — | PASS |
| 4 × (500 + 500) | smile + supremacy | 1.0152 | 409 / 561 | 0 | 0.217 | 0.053 | 1.131 | 0.833/0.965/0.990/1.016/1.057 | PASS |
| 4 × (500 + 1000) | baseline | 1.0059 | 1505 / 1217 | 0 | — | — | 1.124 | — | PASS |
| 4 × (500 + 1000) | supremacy | 1.0037 | 1346 / 1988 | 0 | 0.210 | — | 1.117 | — | PASS |
| 4 × (500 + 1000) | smile + supremacy | 1.0064 | 866 / 1383 | 0 | 0.217 | 0.053 | 1.131 | 0.834/0.965/0.991/1.017/1.058 | PASS |

The work-package smoke budget fails the **baseline** — the recipe Task 013 already ran to 43 folds —
on one tail ESS, 363 < 400. The threshold was not lowered; the gate was re-run at the budget r02
uses, where it passes. Every pillar site (σ_sup, σ_smile, log_φ[1:5]) has R̂ ≤ 1.005 and
ESS ≥ 640 on both folds.

* **G4.** O/U 2.5 through the typed evaluation kernel, the legacy MatchDay row route and the
  reference `mean cdf(Poisson(λ_tot·φ₂), 2)` agree to 6 d.p. on every checked fixture; the smile
  sits +0.0078 above the plain grid at the same λ draws. The smile is anchored hard (σ_smile ≈ 0.053)
  and under-prices totals at K = 0 (φ ≈ 0.83) while leaving K = 2–3 near 1.
* **G5.** Every rung saved and reloaded identically; for the smile rung the `SmileLatents` rebuilt
  from the PostgreSQL chains equal the fitted container field for field (T010 path).
* **G6.** Registers after the second attempt's fix: the wrapper's base model is saved as
  `<name>__base_model`, the full recipe as `<name>_fit` (`save_model` refuses non-builder types —
  recorded in T010).

Smoke run IDs (`smoke_grw_smile`, production budget): baseline `e2152be3-f609-4148-b250-96a624e3966c`,
supremacy `cd886262-38d4-4956-a26a-94cb88706e58`, smile `5ce9ded8-036f-40f5-b878-2452286dbcbc`.

## Production grid (r02)

43 folds (1–40 asserted identical to the 40-fold split), 769 held-out fixtures per run — the same
set as the baseline control, asserted. Store through 2026-09-05, no 2026-09-12 rows. Namespace
`scottish_lower_grw_market_smile`. Report:
`results/r02_production_report_4x500w1000s_*.md`.

Before sampling, r02 proved the pinned baseline (`b0961bc4`) is the same recipe at the same budget:
`string(model)` identical; recorded sampler at 500 = 1,000 ÷ stride 2; Task 013's pre-extension
report 160,000 transitions over 40 folds; fold-1 re-audit 2,000 transitions; 2,000 persisted draws.

| rung | budget | R̂ (fold) | ESS bulk / tail | div | BFMI | wall | gate | run |
|---|---|---:|---:|---:|---:|---:|---|---|
| baseline (Task 013) | 4 × (500 + 1000) | 1.0115 | — | 0 | — | 41 min + ext. | pinned | `b0961bc4-c40c-4dbe-9c05-57df7ae0839e` |
| supremacy @0.40 | 4 × (500 + 1000) | 1.0105 (14) | 814 / 516 | 0 / 172k | 0.683 | 60 min | PASS | `0ee58d18-b7e9-4168-8d78-93887b1a8c26` |
| smile + sup @0.20 | 4 × (500 + 1000) | 1.0139 (34) | 472 / 498 | 0 / 172k | 0.623 | 158 min | PASS | `fcd5e974-9a46-4a10-9828-6b987a5484d6` |
| smile + sup @0.40 | 4 × (500 + 1000) | 1.0200 (39) | 431 / 696 | 0 / 172k | 0.665 | 185 min | PASS | `30620d3e-e4bd-4c05-b1a1-85cefa36b728` |
| smile + sup @0.70 | 4 × (500 + 1000) | 1.0201 (29) | **312** / 633 | 0 / 172k | 0.681 | 266 min | **FAIL** (bulk ESS, fold 29) | not persisted |
| smile + sup @0.70 | 4 × (1000 + 2000), stride 4 | 1.0159 (27) | 504 / 1182 | 0 / 344k | 0.694 | 438 min | PASS | `32d588f1-d666-4112-a7e1-5c9545fbbe3d` |

The @0.70 failure is a draw-count shortfall on one fold — R̂ within gate, no divergences, no
tree-depth saturation — so it is re-run at twice the warmup and draws with every 4th draw persisted,
which keeps its artefact at the same 2,000 draws per fixture as every other arm. Checkpoints are
budget-stamped, so the failed draws cannot be resumed into the re-run. The threshold was not changed.

**The smile costs sampling time.** Wall time grows 60 → 158 → 185 → 266 min from supremacy-only to
smile at 0.20 / 0.40 / 0.70, and minimum bulk ESS falls 814 → 472 → 431 → 312: a stronger anchor
makes the posterior harder to explore, not easier.

### Pillar posteriors (pooled over folds)

| rung | σ_sup median [5%, 95%] | σ_smile median [5%, 95%] | κ | φ(K = 0…4) |
|---|---|---|---:|---|
| supremacy @0.40 | 0.212 [0.194, 0.232] | — | 1.105 | — |
| smile + sup @0.20 | 0.237 [0.213, 0.264] | 0.052 [0.049, 0.055] | 1.116 | 0.844 / 0.976 / 1.001 / 1.026 / 1.069 |
| smile + sup @0.40 | 0.220 [0.202, 0.240] | 0.050 [0.048, 0.053] | 1.115 | 0.843 / 0.976 / 1.001 / 1.026 / 1.069 |
| smile + sup @0.70 (failed run, not persisted) | 0.210 [0.194, 0.227] | 0.049 [0.047, 0.052] | 1.114 | 0.844 / 0.976 / 1.001 / 1.026 / 1.069 |

* **σ_sup ≈ 0.21–0.24 is identified and tight**, ~40% above its prior mean of 0.15: after the GRW
  state and wealth covariate, model and market log-supremacy still disagree by about 0.2 on a typical
  training match. The weight trades against σ_sup — lighter weight, wider σ — as tempering implies.
* **σ_smile ≈ 0.05 at every weight**, a quarter of σ_sup: the model's total intensity follows the
  market's per-strike totals far more closely than its result market follows the market's supremacy.
* **φ does not depend on the weight** (identical to 3 d.p.). Under a Normal anchor log φ_K sits at the
  mean market − model residual at strike K whatever the tempering; the weight changes how hard that
  residual pulls on the team state, not where φ settles. The curve says the Poisson-referenced market
  prices **more 0-goal and fewer 4+-goal outcomes than a Poisson on the model total** — the
  over-dispersion signature — and moves the O/U 2.5 price by +0.0073 P(Under) against the plain grid.

## Out-of-sample scores (r04)

Panel: the 710 walk-forward fixtures of 24/25 + 25/26 (627 carry a Betfair close). Book: de-vigged
Betfair TWA(−20, 0] close. **Reproduction gate passed exactly**: the pinned baseline scores LogLoss
0.64315 / ECE 0.0123 on 2,899 rows, Task 013's published figures. 10,000 fixture-clustered paired
resamples. Report: `results/evaluation/r04_evaluation_report.md`.

### Headline: 0 of 84 contrasts significant

| model | all (1X2 + OU2.5 + BTTS) | 1X2 | OU2.5 | BTTS | OU1.5 | OU3.5 | ECE (all) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.64315 | 0.61558 | 0.68986 | **0.68192** | **0.52070** | 0.61235 | **0.0123** |
| supremacy @0.40 | 0.64135 | 0.61338 | 0.68796 | 0.68235 | 0.52285 | 0.61131 | 0.0195 |
| smile + sup @0.20 | 0.64116 | 0.61245 | 0.68705 | 0.68742 | 0.52784 | 0.60948 | 0.0152 |
| smile + sup @0.40 | **0.64092** | **0.61209** | 0.68698 | 0.68741 | 0.52765 | 0.60897 | 0.0145 |
| smile + sup @0.70 | 0.64097 | 0.61230 | **0.68668** | 0.68734 | 0.52706 | **0.60853** | 0.0198 |
| *Betfair close* | *0.64182* | *0.61312* | *0.68988* | *0.68337* | *0.52728* | *0.61046* | *0.0139* |

LogLoss unless stated. The anchored rungs' point estimates beat the baseline by ~0.002 and edge
under the close; **none of it is distinguishable from zero**, and it is bought with worse
calibration (ECE 0.0123 → 0.0145–0.0198).

### Paired ΔLogLoss, selected contrasts

| contrast | scope | Δ | 95% interval | p(better) |
|---|---|---:|---|---:|
| supremacy @0.40 − baseline | all | −0.00180 | [−0.00440, +0.00082] | 0.909 |
| supremacy @0.40 − baseline | 1X2 | −0.00221 | [−0.00612, +0.00173] | 0.862 |
| smile @0.40 − baseline | all | −0.00223 | [−0.00715, +0.00269] | 0.816 |
| smile @0.40 − baseline | 1X2 | −0.00349 | [−0.00787, +0.00090] | 0.937 |
| smile @0.40 − baseline | OU1.5 | +0.00694 | [−0.00687, +0.02020] | 0.154 |
| smile @0.40 − supremacy @0.40 | OU2.5 | −0.00097 | [−0.01366, +0.01166] | 0.560 |
| smile @0.20 − smile @0.40 | all | +0.00024 | [−0.00069, +0.00117] | 0.300 |
| smile @0.70 − smile @0.40 | all | +0.00005 | [−0.00089, +0.00098] | 0.469 |

The nearest miss is on Brier, smile @0.40 − baseline on 1X2: Δ −0.00167 [−0.00363, +0.00030],
p(better) 0.951. Against the Betfair close every rung's interval straddles zero on every scope.

### Hypotheses

| | verdict | evidence |
|---|---|---|
| H1 supremacy improves 1X2 | **refuted** | Δ −0.00221 [−0.00612, +0.00173] |
| H2 smile adds on totals | **refuted** | smile − supremacy on OU2.5 Δ −0.00097 [−0.01366, +0.01166]; the smile is *worse* on BTTS and OU1.5 point-wise |
| H3 home-favourite compression shrinks | **partly, and small** | see below |
| H4 a weight beats 0.40 | **refuted** | every weight contrast \|Δ\| ≤ 0.0006, intervals ±0.001 |

### Home-favourite compression (1X2 home, by Betfair-close p_home)

| close p_home bin | n | close | home won | baseline gap | supremacy gap | smile @0.40 gap | smile @0.70 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.00, 0.30) | 111 | 0.237 | 0.252 | +0.065 | +0.062 | +0.074 | +0.063 |
| [0.30, 0.40) | 148 | 0.356 | 0.405 | +0.021 | +0.028 | +0.030 | +0.026 |
| [0.40, 0.50) | 178 | 0.448 | 0.494 | −0.018 | −0.010 | −0.017 | −0.016 |
| [0.50, 0.60) | 107 | 0.539 | 0.495 | −0.058 | −0.046 | −0.062 | −0.055 |
| [0.60, 1.00) | 51 | 0.675 | 0.588 | −0.121 | −0.093 | −0.107 | −0.091 |

Gap = mean p_model − p_close. The compression is real and systematic in the baseline — a 25-point
market spread (0.24 → 0.68) becomes a 25-point model spread only from 0.30 → 0.55. The pillars
shrink the top-bin gap by at most 0.03, and the light and moderate smile *widen* the 0.50–0.60 gap.
**But the realised home win rate sits between model and market in both favourite bins** (0.588 against
0.555–0.584 model and 0.675 close; 0.495 against 0.470–0.493 model and 0.539 close): on this panel
the close over-prices home favourites, and the model's "compression" is partly closer to what
happened. n = 51 and 107 — too few to call the market wrong, enough to refuse the premise that
matching it would have been right.

## Closing-line portfolio (r06)

Every arm staked through `MatchDay.option_b_system()` — 1X2 home and Under 2.5 at full trust,
draw, away and Over 1.5 at 1/1.4; 30% fractional Kelly, `SlateDrawdown(8.0)`, `FixedCap(0.25)`,
daily slates, 2% commission — against the de-vigged Betfair TWA(−20, 0] close. Panel: 710
walk-forward → 635 quoted → **632 buildable by every arm** (3 dropped). Only the posterior differs
between rows. Bootstrap B = 4,000. Report: `results/portfolio/r06_portfolio_report.md`.

**Gates.** P1: the baseline reproduces its published Option B row exactly on the same 632
fixtures (+385.78%, ROI 11.68%, 1,247 bets). P2: every staked totals bet of every smile arm
(252–253 bets each) carries `p_model` equal to `mean cdf(Poisson(λ_tot·φ_K), K)` to ≤ 1.8e-15, so
those books were priced through the smile, not the plain grid. P3: the four candidate portfolios
are persisted under `scottish_lower_grw_market_smile` and each reloads with an identical ledger.

> **Correction (r07, 2026-09-13) — read this before the r06 tables.** Portfolio sizes every stake
> from the (λ_h, λ_a) score grid; a smile container's φ reaches only the ledger's reported `p_model`
> (`src/Portfolio/pricing.jl:360–375`, ticket T011). P2 below is therefore a check that the *reported
> price* column is smile-priced — true, and irrelevant to staking. Every smile-arm difference in this
> section comes from the team state λ the pillars moved during fitting, not from the per-strike curve.
> The bullets that say "the smile re-sizes" or "the smile moves stake" describe those λ effects.

### Headline

| model | total return | flat ROI | max DD | Sharpe (ann.) | Calmar | bets (1X2 / totals) | win rate | growth / slate [95%] | P(ROI > 0) |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| baseline | +385.8% | 11.68% | −42.67% | 1.453 | 9.04 | 1,247 (986 / 261) | 35.2% | [−0.0004, +0.0323] | 0.984 |
| supremacy @0.40 | +404.6% | 12.75% | −42.64% | **1.551** | 9.49 | 1,244 (983 / 261) | 36.5% | [+0.0007, +0.0321] | 0.992 |
| smile + sup @0.20 | **+588.4%** | 15.71% | −44.02% | 1.495 | **13.37** | 1,240 (988 / 252) | 34.0% | [−0.0003, +0.0390] | **0.996** |
| smile + sup @0.40 | +545.0% | **15.82%** | −44.03% | 1.516 | 12.38 | 1,237 (984 / 253) | 34.0% | [+0.0004, +0.0376] | 0.995 |
| smile + sup @0.70 | +481.4% | 15.48% | **−42.37%** | 1.530 | 11.36 | 1,231 (978 / 253) | 35.0% | [+0.0006, +0.0354] | 0.993 |

Persisted portfolios: supremacy `1e6b80b0-0b76-4f09-ac15-95a30959446c`, smile @0.20
`b8ea28ef-3373-49d8-baa9-16d2c596e475`, @0.40 `38d3038e-2b34-435a-a5b3-7d098c1f8031`, @0.70
`5cff0ccc-0df1-4f10-8cf9-f952123adba3`.

Every anchored arm out-returns the baseline and lifts ROI by +1.1 (supremacy) to +4.1 points
(smile). **The per-slate growth intervals overlap almost entirely** (baseline [−0.0004, +0.0323]
against smile @0.20 [−0.0003, +0.0390]), so this ranking is descriptive: it is not a paired test of
the difference, and r04 found no proper-score difference to explain one.

### Where the return comes from

| selection | baseline: bets / ROI / stake share | supremacy @0.40 | smile @0.20 | smile @0.40 | smile @0.70 |
|---|---|---|---|---|---|
| 1X2 home | 310 / 20.0% / 33.1% | 339 / 13.9% / 35.8% | 305 / 17.7% / 38.9% | 311 / 17.9% / 38.4% | 313 / 18.5% / 37.8% |
| 1X2 draw | 273 / 3.4% / 9.5% | 250 / 11.1% / 8.6% | 302 / 9.6% / 11.8% | 284 / 8.4% / 11.0% | 274 / 7.8% / 10.0% |
| 1X2 away | 403 / **7.8%** / 33.3% | 394 / 16.6% / 27.4% | 381 / **18.8%** / 34.1% | 389 / **19.4%** / 34.1% | 391 / 17.1% / 34.4% |
| Under 2.5 | 197 / 10.8% / 19.4% | 203 / 9.7% / 23.3% | 195 / 10.7% / 12.4% | 197 / 10.3% / 13.5% | 200 / 11.9% / 14.8% |
| Over 1.5 | 64 / 1.3% / 4.7% | 58 / −0.4% / 4.9% | 57 / −0.7% / 2.9% | 56 / 0.8% / 3.0% | 53 / 2.2% / 3.0% |
| **1X2 total** | 986 / 12.6% / 75.9% | 983 / 14.6% / 71.8% | 988 / 17.0% / 84.8% | 984 / 17.2% / 83.5% | 978 / 16.6% / 82.2% |
| **totals total** | 261 / 8.9% / 24.1% | 261 / 8.0% / 28.2% | 252 / 8.5% / 15.2% | 253 / 8.6% / 16.5% | 253 / 10.2% / 17.8% |

Stake share is the fraction of the arm's total stake. Win rates by market: 1X2 29.4–31.3%, totals
51.4–55.9%, both flat across arms.

* **The gain is on 1X2 away** — the selection the work package set out to remove. Away ROI roughly
  doubles (7.8% → 16.6–19.4%) on a similar number of bets. It is not that the anchored models back
  fewer away sides (381–394 vs 403); they back *different* ones and size them differently.
* **The smile arms move stake out of totals** (through their λ; φ does not size — T011). Their reported Under 2.5 edge halves (5.3 → 3.2 pp mean edge), so
  totals fall from 24% to 15–18% of stake with ROI unchanged, and the freed exposure goes to 1X2.
  The smile does not make the totals book better; it makes it smaller.
* **Home ROI falls** under every anchor (20.0% → 13.9–18.5%): lifting P(home) toward the market
  (r04) buys more, worse home bets, most visibly at supremacy (339 bets, 13.9%).

### Shared vs exclusive bets against the baseline

| candidate | shared bets | shared ROI: candidate vs baseline | candidate-only: bets / ROI | baseline-only: bets / ROI |
|---|---:|---|---|---|
| supremacy @0.40 | 1,074 | 13.10% vs 13.54% | 170 / +7.2% | 173 / **−25.8%** |
| smile @0.20 | 978 | **15.94% vs 13.11%** | 262 / +13.8% | 269 / +4.4% |
| smile @0.40 | 960 | **16.35% vs 11.65%** | 277 / +11.9% | 287 / +11.8% |
| smile @0.70 | 938 | **16.24% vs 13.55%** | 293 / +10.7% | 309 / +3.3% |

Shared bets are the same fixture, selection, price and outcome, so their ROI gap is stake size
alone. The two pillars earn their return differently:

* **Supremacy** sizes the shared bets marginally *worse* and gains entirely by declining 173 bets the
  baseline took that lost 25.8% — it prunes.
* **The smile arms** size the ~950 shared bets better (+2.7 to +4.7 ROI points) — they re-weight. The
  sizing is the grid of their pillar-shifted λ, not φ (T011).

### How to read this

This is the most favourable test the pillars could be given: closing prices, for models whose
likelihood was fitted to closing prices (on training fixtures only, so there is no leakage into
the scored fixtures, but the book is the anchor's own reference). Against it stand r04's null on
every proper score and growth intervals that overlap almost completely. **A +100 to +200 point
bankroll gap on 632 fixtures is within what this contract's variance produces between arms that
score the same**: Task 014 measured +297% to +492% across eight arms with indistinguishable proper
scores on this same panel. The portfolio result is a reason to run the executable-price test, not a
reason to promote.

## Tradeable T−25 book, raw vs calibrated (r07)

Every arm staked through `MatchDay.option_b_system()` in two environments, Task 014 `r06`'s design.
Report: `results/t25_portfolio/r07_t25_portfolio_report.md`.

| environment | book | panel |
|---|---|---:|
| close | de-vigged Betfair TWA(−20, 0] close | 632 |
| t25 | point-in-time T−25 book (median staleness 8 min, p90 51) | 611 (580 inverted by the calibrator) |

**Gates.** T1: the five close/raw rows reproduce r06 exactly (Δ 0.0 pp, 0 bet-count mismatches). T2:
the T−25 panel is Task 014's 611 fixtures and the baseline reproduces its §9 rows — raw +531.78% /
1,124 bets, `t25_inv` +245.85% / 969 bets. T3: every smile raw and `t25_inv_pooltot` ledger's
totals `p_model` equals `λ_tot·φ(K)` to ≤ 1.8e-15.

### The calibrated smile, and why its two definitions give one ledger

`Calibration.calibrate_latents` refuses `SmileLatents` — a calibrated smile is not defined in `src`.
Two definitions were run (`l04` header): **pooltot** (calibrate the grid, rebuild λ_tot from it, keep
φ) and **grid** (calibrate the grid, drop φ). They stake **bit-identical ledgers** on every smile arm
(955 / 947 / 940 shared bets, 0 exclusive, ROI equal to 1e-14) while their recorded mean edge differs
(e.g. 1.869 vs 1.905 pp). The reason is structural: `_finish_book` sizes from
`allocate(allocator, p_grid, R)` over the score grid and `BakerMcHale` re-solves on grid draws, so φ
only ever reaches `p_model` (`src/Portfolio/pricing.jl:240–257, 360–375`). **Raised as ticket T011.**
Consequence for every portfolio number in this README: smile-vs-baseline differences are λ
differences.

### Headline at T−25

| model | variant | return | flat ROI | Sharpe | max DD | bets (1X2 / totals) | growth / slate [95%] |
|---|---|---:|---:|---:|---:|---|---|
| baseline | raw | +531.8% | 14.15% | 1.658 | −41.9% | 1,124 (896 / 228) | [+0.0023, +0.0354] |
| supremacy @0.40 | raw | **+605.6%** | 16.17% | **1.711** | −38.3% | 1,089 (864 / 225) | [+0.0031, +0.0372] |
| smile @0.20 | raw | +542.4% | **16.91%** | 1.342 | −42.9% | 1,111 (902 / 209) | [−0.0024, +0.0409] |
| smile @0.40 | raw | +479.4% | 16.80% | 1.322 | −43.6% | 1,087 (883 / 204) | [−0.0024, +0.0385] |
| smile @0.70 | raw | +381.6% | 15.78% | 1.276 | −41.8% | 1,094 (885 / 209) | [−0.0019, +0.0341] |
| baseline | t25_inv | **+245.9%** | 17.39% | **1.976** | −22.0% | 969 (776 / 193) | [+0.0031, +0.0220] |
| supremacy @0.40 | t25_inv | +219.5% | 17.98% | 1.856 | −19.6% | 963 (772 / 191) | [+0.0026, +0.0214] |
| smile @0.20 | t25_inv (either) | +242.4% | 21.78% | 1.818 | −18.4% | 955 (790 / 165) | [+0.0024, +0.0229] |
| smile @0.40 | t25_inv (either) | +223.7% | **22.10%** | 1.825 | −16.7% | 947 (782 / 165) | [+0.0024, +0.0220] |
| smile @0.70 | t25_inv (either) | +178.8% | 20.37% | 1.697 | **−14.7%** | 940 (767 / 173) | [+0.0018, +0.0200] |

The calibrator treats every arm alike: median weight kept on the model's log-rate 0.274–0.284, market
share 72%, retained posterior log-variance 7.5–8.1%, on 580 fixtures.

### Paired slate-level log growth (B = 10,000)

| question | contrast | ΔROI | Δ log growth / slate [95%] | p(better) |
|---|---|---:|---|---:|
| close reference (r06) | smile @0.20 raw − baseline raw | +4.03 | +0.0035 [−0.0075, +0.0152] | 0.722 |
| | smile @0.40 raw − baseline raw | +4.14 | +0.0028 [−0.0084, +0.0144] | 0.687 |
| **Q1 raw lead at T−25** | smile @0.20 raw − baseline raw | +2.75 | +0.0002 [−0.0112, +0.0119] | 0.513 |
| | smile @0.40 raw − baseline raw | +2.65 | −0.0009 [−0.0121, +0.0107] | 0.441 |
| | smile @0.70 raw − baseline raw | +1.63 | −0.0027 [−0.0139, +0.0088] | 0.325 |
| | supremacy raw − baseline raw | +2.02 | +0.0011 [−0.0072, +0.0094] | 0.600 |
| **Q2 beyond L2** | smile @0.20 cal − baseline cal | +4.39 | −0.0001 [−0.0052, +0.0053] | 0.485 |
| | smile @0.40 cal − baseline cal | +4.71 | −0.0007 [−0.0059, +0.0049] | 0.406 |
| | smile @0.70 cal − baseline cal | +2.98 | −0.0022 [−0.0078, +0.0035] | 0.226 |
| | supremacy cal − baseline cal | +0.59 | −0.0008 [−0.0046, +0.0031] | 0.346 |
| **Q4 pillar instead of L2** | smile @0.20 raw − baseline cal | −0.48 | +0.0063 [−0.0077, +0.0201] | 0.814 |
| | smile @0.40 raw − baseline cal | −0.59 | +0.0052 [−0.0082, +0.0186] | 0.783 |

(Q3, calibrated grid − baseline calibrated, is identical to Q2 by T011.)

### Answers

* **Does smile @0.20–0.40 keep its lead at T−25?** Its **ROI** lead survives, smaller (+2.7 points
  raw, against +4.0 at the close). Its **bankroll** lead does not: +542% / +479% against the
  baseline's +532%, Sharpe 1.32–1.34 against 1.66, paired growth p(better) 0.51 and 0.44. The best raw
  arm at T−25 is supremacy (+606%, Sharpe 1.71, p 0.60), not a smile.
* **Does the fit-time smile add value beyond the L2 calibrator?** The φ curve cannot — it never sizes
  a bet (T011). A smile-*fitted* model, calibrated, has the highest ROI (+4.4 / +4.7 points) and the
  lowest drawdown (−16.7% to −18.4% against −22.0%), because it stakes less: fewer bets and totals at
  16–17% of stake against the baseline's 30%. It does **not** grow the bankroll faster (p(better)
  0.49 / 0.41 / 0.23) and its Sharpe is lower (1.82 against 1.98). That is a different risk
  position, not extra edge.
* **Can the pillar replace the calibrator?** No. A raw smile arm matches the calibrated baseline's ROI
  (−0.5 points) at roughly double its drawdown (−43% against −22%); its higher growth (p 0.78–0.81) is
  the larger raw stake, not a better price.

Where the calibrated smile ROI comes from: 1X2 ROI 21.1–22.8% against the calibrated baseline's
15.7%, on 83–84% of stake; totals ROI 16.7–19.4% against 21.4%. The same pattern as r06 — the anchored
λ backs a better 1X2 book and a smaller, worse totals book.

## Trust-pruning sweep (r08)

The EDA (`eda/README.md`) prunes every fringe selection to trust 0 after measuring Jensen tail
inflation on deep Unders (Under 0.5: −30.7% ROI) and capacity cannibalisation. r08 re-asks whether
the smile model can re-open them. Arms: smile @0.40 and the baseline as control. Environments: close
and raw T−25. Policy: Option B plus one fringe line at a time at the second tier (1/1.4), then unions;
risk, cap, filter and grouping unchanged. Report: `results/trust_sweep/r08_trust_sweep_report.md`.

**Not comparable with the EDA's figures**, which come from the TimeDecay hybrids m12/m13 under
`SlateDrawdown(23)` with a 20% cap; the baseline arm is the like-for-like control.

**Gates.** S1: close/P0 reproduces r06 for both arms exactly. S2: all 732 (close) and 539 (T−25)
staked totals bets of the smile arm under `+all_fringe` carry `p_model = λ_tot·φ(K)` to ≤ 2.7e-15.
**S0 is a finding, not a pass:** Option B's book extended with O/U 4.5 at trust 0 does **not** stake
Option B's ledger — return moves −16.9 pp (baseline close), −20.3 (smile close), −22.3 (baseline T−25),
−6.6 (smile T−25), by one or two bets. A zero-trust market still enters the joint solve. Every sweep
row is therefore measured against **P0 on the same extended book**, so a row differs from its reference
in the trust table only.

### Two premises the sweep corrects

1. **φ₀ < 1 raises P(Under 0.5), it does not lower it.** P(N = 0) = e^{−Λ(0)} and φ₀ = 0.844 lowers
   Λ(0). The smile's staked Under 0.5 price is *higher* than the grid's (mean p_model 0.100 vs 0.084 at
   the close, 0.100 vs 0.089 at T−25) and further from what happened (realised 0.038 / 0.000). The
   market's curve prices more 0-0 than a Poisson on the total, but on this panel the market itself
   over-prices 0-0 (close p_market 0.059–0.067 against realised 0.027–0.038).
2. **φ does not size stakes (T011).** The smile arm's fringe bets are sized from its λ grid, so the
   sweep cannot test "the smile repriced the tail"; it tests "does the smile-fitted λ change the fringe
   decision". Paired smile − baseline growth has p(better) between 0.42 and 0.82 in all 24
   policy × environment cells: it does not.

### Δ terminal return vs P0 on the extended book (pp), and the added bets' own ROI

| line | baseline close | smile close | baseline T−25 | smile T−25 | added bets | added ROI |
|---|---:|---:|---:|---:|---|---|
| + Under 0.5 | −57.1 | −17.4 | −58.6 | −22.6 | 81 / 53 / 37 / 25 | −63.6 / −41.8 / −89.5 / **−100.0** |
| + Under 1.5 | **+119.3** | +22.5 | +10.2 | +9.0 | 105 / 93 / 76 / 61 | +37.7 / +15.8 / +3.9 / +4.5 |
| + Under 3.5 | +20.9 | +31.7 | −82.9 | −34.5 | 99 / 88 / 67 / 83 | +2.4 / +2.4 / −15.0 / −11.9 |
| + Under 4.5 | +56.2 | +21.8 | +13.9 | +41.5 | 28 / 31 / 23 / 28 | +18.8 / +21.8 / +10.4 / +16.4 |
| + Over 2.5 | −22.6 | +30.5 | +52.6 | +18.4 | 104 / 73 / 116 / 87 | +0.9 / +19.6 / +10.8 / +10.2 |
| + Over 3.5 | −23.0 | −18.1 | −88.7 | +5.8 | 120 / 95 / 75 / 42 | +2.9 / −5.0 / −23.7 / +8.1 |
| + Over 4.5 | −57.7 | +70.3 | −0.3 | +19.3 | 54 / 47 / 21 / 11 | −30.8 / +93.7 / −9.3 / +131.4 |
| + BTTS yes | −28.9 | +12.5 | −53.8 | −55.6 | 52 / 62 / 73 / 70 | +2.0 / +2.1 / −12.4 / −27.5 |
| + BTTS no | −21.3 | −10.4 | +81.1 | +44.2 | 69 / 53 / 82 / 59 | −7.1 / −7.7 / +13.0 / +14.3 |
| + all Unders | +118.9 | +37.2 | −129.4 | −32.5 | 313 / 265 / 203 / 197 | +8.5 / +6.8 / −8.9 / −4.5 |
| + all fringe | −69.8 | +89.5 | **−159.2** | −14.4 | 712 / 595 / 570 / 466 | +1.7 / +8.5 / −3.3 / −2.4 |

Added-bet columns are in the order baseline close / smile close / baseline T−25 / smile T−25. Over 4.5
is a lottery ticket (win rate 5–13% on 11–54 bets): its ±100% ROI cells are a handful of results.

### Capacity

Under Option B (`SlateDrawdown(8)`, cap 0.25) cannibalisation is mild. Any single line leaves the core
basket at 0.96–1.00× its Option B stake with core ROI within ±0.6 points; the unions take it to
0.93–0.98× (all Unders) and 0.83–0.95× (all fringe), where fringe bets absorb 17–28% of stake. The
return lost to the fringe is the fringe's own P&L, not a damaged core.

### Reading

* **Robust across both arms and both books:** Under 0.5 loses in all four cells (−42% to −100% on its
  own bets). The EDA's pruning of it stands, and the smile does not change that — by premise 1 it
  would make it worse if it could.
* **Consistently positive in all four cells, on thin samples:** Under 1.5 (ROI +3.9 to +37.7%, 61–105
  bets) and Under 4.5 (+10.4 to +21.8%, 23–31 bets at a 93–96% strike rate).
* **Sign-unstable, i.e. noise at this sample size:** Under 3.5, Over 2.5, Over 3.5, Over 4.5 and both
  BTTS directions change sign between the close and T−25, or between the two arms.
* **Unions are worse at the tradeable book:** all fringe −159 pp (baseline) and −14 pp (smile) at T−25.

Recommendation for the trust table: keep the EDA's pruning as the default. If an exploration budget
is wanted, Under 1.5 and Under 4.5 at the second tier are the only additions with a consistent sign in
every cell. They are candidates for a prospective paper trial, not a promotion; nothing here depends
on the smile.

## 2026-09-12 counterfactual (r05)

Card: 9 played fixtures (Ross County v Hamilton, no score, removed). as_of 2026-09-12 13:35 UTC
(T−25); every arm on Fold 43, every fixture covered, the Option B calibrator inverted 9 / 9 books.
Priced through the replay engine; nothing written to either ledger. Report:
`results/slate_20260912/r05_slate_report.md`.

**Reproduction.** The live ledger reads 11 legs from Run 67 / Fold 43, realised −£45.89 (summed
from `paper_settlements`), 2% commission. `live_optB` re-prices all 11 live legs plus 4 more, with max
|Δrisk| £3.64 and max |Δp_model| 0.023 — the same figures Task 008 measured, from the same cause
(no provisional XI existed at T−25). The team-level rungs read no lineup, so none of that gap
touches them.

### What moved: P(home) at T−25

| fixture | score | book | live (optB) | baseline raw | supremacy raw | smile @0.40 raw | smile @0.70 raw | smile @0.40 optB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Queen of the South v East Fife | 1-1 | 0.392 | 0.401 | 0.440 | 0.488 | 0.495 | 0.506 | 0.427 |
| East Kilbride v Peterhead | 3-1 | 0.560 | 0.494 | 0.478 | 0.532 | 0.526 | 0.544 | 0.555 |
| Montrose v Cove Rangers | 2-1 | 0.477 | 0.465 | 0.369 | 0.403 | 0.405 | 0.421 | 0.460 |
| Airdrieonians v Alloa | 1-0 | 0.411 | 0.416 | 0.323 | 0.367 | 0.366 | 0.376 | 0.405 |
| Edinburgh City v Stirling Albion | 7-3 | 0.541 | 0.494 | 0.476 | 0.493 | 0.478 | 0.480 | 0.526 |
| Clyde v Kelty Hearts | 4-1 | 0.514 | 0.478 | 0.546 | 0.526 | 0.534 | 0.526 | 0.525 |
| The Spartans v Forfar | 5-1 | 0.567 | 0.512 | 0.460 | 0.494 | 0.493 | 0.508 | 0.543 |
| Annan Athletic v Elgin | 2-0 | 0.372 | 0.396 | 0.352 | 0.383 | 0.401 | 0.407 | 0.385 |
| Stranraer v Dumbarton | 3-1 | 0.550 | 0.504 | 0.475 | 0.461 | 0.452 | 0.471 | 0.517 |

"Book" is the de-vigged 1X2 quote at as_of. Two facts correct the work package's framing. The live
arm's calibrated P(home) spanned **0.40–0.51**, not a flat 40–43%, and the book's favourites sat at
**0.54–0.57**, not 55–60% across the card.

**Did the supremacy pillar lift the favourites toward the market?** Partly. From baseline to
supremacy the raw P(home) rises on East Kilbride (+0.054), Airdrie (+0.044), The Spartans (+0.034),
Montrose (+0.034) and Edinburgh City (+0.017). But it moves **away** from the book on two fixtures:
Stranraer falls (−0.014 against a 0.550 book), and Queen of the South rises +0.048 **past** a
0.392 book to 0.488–0.506. No raw anchored arm reaches the book on any favourite.

### Stake sheets, settled at full fill

| arm | legs | risk | net P&L | loss per £ risk | away legs | away net | home legs | home net | Under 2.5 legs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| live_optB (Run 67) | 15 | £73.78 | −£54.52 | 0.74 | 7 | −£28.17 | 1 | +£4.46 | 6 |
| base_raw | 17 | £125.00 | −£98.27 | 0.79 | 7 | −£56.86 | 2 | +£0.06 | 4 |
| base_optB | 11 | £60.07 | −£53.56 | 0.89 | 6 | −£24.76 | 1 | +£1.36 | 4 |
| sup040_raw | 15 | £125.00 | −£102.00 | 0.82 | 6 | −£43.27 | 2 | −£19.07 | 5 |
| sup040_optB | 10 | £42.44 | −£39.40 | 0.93 | 5 | −£13.89 | 1 | −£3.36 | 4 |
| smile020_raw | 18 | £124.74 | −£91.04 | 0.73 | 6 | −£46.39 | 3 | −£10.46 | 5 |
| smile020_optB | 11 | £39.47 | −£34.20 | 0.87 | 5 | −£14.55 | 1 | −£2.87 | 5 |
| smile040_raw | 17 | £123.94 | −£78.85 | 0.64 | 6 | −£40.60 | 3 | −£11.64 | 6 |
| smile040_optB | 11 | £36.18 | −£28.88 | 0.80 | 5 | −£12.64 | 1 | −£4.08 | 5 |
| smile070_raw | 15 | £124.04 | −£69.40 | 0.56 | 5 | −£33.50 | 2 | −£15.68 | 6 |
| smile070_optB | 10 | £31.41 | −£21.36 | 0.68 | 5 | −£10.05 | 1 | −£5.30 | 4 |
| live ledger (realised fills) | 11 | £55.73 | **−£45.89** | 0.82 | 6 | | 0 | | 5 |

Full-fill settlement assumes every leg filled at planned risk; the live account filled some legs
partially, so the ledger row is not directly comparable with the re-priced rows.

**Did anchoring eliminate the losing away bets?** No. Every anchored Option B arm still backs five
away sides — Montrose, The Spartans, Stranraer, Airdrie, Edinburgh City — and all five lost. Anchoring
removed East Kilbride away, the East Kilbride draw and Clyde away from the live sheet, and shrank the
survivors (away risk £28.17 → £10.05–£13.89).

**Did it add anything?** One new losing leg in every anchored arm: **Queen of the South home**
(P(home) 0.42–0.51 against a 0.392 book, 1-1). In the raw arms it becomes the largest leg on the
sheet (£20.61 at supremacy, £25.11 at smile @0.70).

**Counterfactual P&L.** Under Option B the anchored arms lose −£21.36 to −£39.40 against −£54.52 for
the live arm at full fill. **Most of that is staking less, not choosing better**: risk falls from £73.78
to £31.41–£42.44, and per pound at risk only smile @0.70 (0.68) beats the live arm (0.74); supremacy
(0.93) and smile @0.20 (0.87) lose *more* per pound than it. Among the raw arms the stronger smiles
do lose less per pound (0.64 / 0.56 vs the baseline's 0.79), driven by one winning Annan Under 2.5
leg the smile sized up (£10.53, +£14.86).

What the card shows, beyond the pillars: every arm, anchored or not, took the Under 2.5 on a card that
averaged 4.1 goals (37 in 9 fixtures; The Spartans 5-1 Under is the largest single leg in 7 of the 8
sheets inspected leg by leg — smile @0.70 raw's largest is Queen of the South home), and backed away
sides on a Saturday when 8 of 9 home sides won and the ninth drew. Nine fixtures cannot separate a model
defect from a card; `r04` is the evidence, and it says the pillars change little.

## Conclusions

1. **The market-anchored GRW is mechanically sound.** Its log density is the Task 013 model's to the
   bit with the pillars off and the stated equations to 1e-15 with them on; gradients are exact; all
   four rungs converge over 43 folds with 0 divergences (the @0.70 rung needed twice the draws).
2. **It does not improve out-of-sample proper scores.** 0 of 84 paired contrasts significant. The
   pooled LogLoss leans ~0.002 better than the baseline and ~0.001 under the Betfair close, with
   intervals of ±0.005, and calibration worsens (ECE 0.0123 → 0.0145–0.0198).
3. **The smile adds nothing measurable over supremacy, and the weight does not matter.** φ is the same
   curve at 0.20, 0.40 and 0.70; weight contrasts are |Δ| ≤ 0.0006.
4. **It narrows home-favourite compression only slightly** (top-bin gap −0.121 → −0.091), and on this
   panel the close over-prices home favourites — the realised home win rate lies between model and
   market in both favourite bins.
5. **On the Option B closing line it out-returns the baseline, descriptively.** +404.6% (supremacy)
   to +588.4% (smile @0.20) against +385.8%, flat ROI 12.75–15.82% against 11.68%, Sharpe 1.50–1.55
   against 1.45, drawdown unchanged. The gain sits on 1X2 away bets and on re-sizing shared bets;
   per-slate growth intervals overlap almost entirely, and Task 014 saw a comparable spread
   (+297% to +492%) between arms with indistinguishable proper scores on this panel.
6. **At the tradeable T−25 book the bankroll lead is gone.** Raw: smile @0.20 / 0.40 +542% / +479%
   against the baseline's +532% (ROI +2.7 points, Sharpe 1.32–1.34 against 1.66, paired p(better)
   0.51 / 0.44); supremacy is the best raw arm (+606%). Calibrated with Option B: the smile-fitted arms
   have the highest ROI (+4.4 / +4.7 points) and the lowest drawdown, on less exposure, and do not grow
   faster (p 0.49 / 0.41) — a risk position, not extra edge beyond L2.
7. **The smile curve never sizes a bet.** Portfolio stakes off the (λ_h, λ_a) grid; φ reaches only the
   reported `p_model` (T011). Every portfolio difference attributed to a smile arm is its λ.
8. **Fringe trust stays pruned.** Under 0.5 loses in every cell, and φ₀ < 1 would raise its price, not
   lower it. Under 1.5 and Under 4.5 are the only additions positive in all four arm × book cells, on
   thin samples; everything else changes sign between the close and T−25. Separately, a zero-trust
   market is not inert in Option B's solve (−6.6 to −22.3 pp).
9. **It would not have prevented the 2026-09-12 losses.** Five of the losing away legs survive in every
   anchored sheet; one new losing home leg (Queen of the South) appears; the smaller counterfactual
   loss comes mainly from smaller stakes.
10. **Infrastructure findings, ticketed rather than patched:** T011 (Portfolio sizes smile containers
   off the grid) and `PostgresStorage` cannot persist
   `SmileLatents` and `save_model` refuses non-builder model types (T010); `extend_fit` re-audits
   existing folds from thinned chains, so an extended run's recorded diagnostics describe the
   persisted draws, not the sampled ones (noted here; worth a ticket if anyone reads budgets back).

### Recommendation

Do not promote a market-anchored GRW rung. The executable test has now been run (r07): at T−25 the
smile arms keep a ROI lead raw and calibrated, but not a growth lead, and after calibration the
lead is lower exposure and lower drawdown rather than faster bankroll growth. Market information
belongs in the **L2 calibration tier**, which already delivers the drawdown cut (−42% → −22%) on the
plain baseline.

Two things would have to change before a smile rung could be re-tested as a *betting* model:
**T011** (make φ reach the allocator, or refuse smile containers in Portfolio), and a smile
calibration defined in `src` (`calibrate_latents` for `SmileLatents`). Until then a smile arm in any
portfolio is a λ arm with a smile-priced `p_model` column. For the trust table, keep the EDA's pruning;
Under 1.5 and Under 4.5 at the second tier are the only candidates for a prospective paper trial. The Under 2.5 and away-underdog exposure on the
2026-09-12 card is a staking question that no pillar here changed.

## Reproduction

```bash
# mcmc-beast, /root/BF_grw_market_smile (rsync of this worktree, cache of 2026-09-12 00:40)
julia --project -t 16 current_development/grw_market_smile/r01_smoke.jl
julia --project -t 16 current_development/grw_market_smile/r02_production_grid.jl
# the @0.70 rung failed bulk ESS at the production budget; re-run at twice the draws, same persisted count
R02_MODELS=m05_joint_grw_smile_supremacy_w070 R02_WARMUP=1000 R02_SAMPLES=2000 R02_STRIDE=4 \
  julia --project -t 16 current_development/grw_market_smile/r02_production_grid.jl
julia --project -t 16 current_development/grw_market_smile/r04_evaluate.jl
# r05 reads betdb: BF_DB_URL must be EXPORTED (T009), and from mcmc-beast archpc's LAN address
# (192.168.1.88) times out — reach it over Tailscale. Neither line prints the credential.
set -a; . ./.env; set +a
export BF_DB_URL="${BF_DB_URL/192.168.1.88/100.124.38.117}"
julia --project -t 16 current_development/grw_market_smile/r05_slate_repricing.jl
julia --project -t 16 current_development/grw_market_smile/r06_portfolio.jl
julia --project -t 16 current_development/grw_market_smile/r07_t25_calibrated_portfolio.jl
julia --project -t 16 current_development/grw_market_smile/r08_trust_sweep.jl
```
