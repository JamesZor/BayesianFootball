# Negative Binomial + Linear Proxy-xG Form Covariate — Scottish Lower Decompression

**TODO [024](../../../todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md)** ·
namespace `scottish_lower_decompression` · **COMPLETED — 2026-09-22**

## Decision

**The direct proxy-xG form term materially decompresses the model, but not enough,
and it is not a production replacement.** The market-on-model supremacy slope
falls from **1.7240** for the joint Gamma-Poisson control to **1.3023** for the
candidate. That is a 58.1% closure of the distance from the joint control to the
ideal 1.0, but it misses the pre-specified **0.85–1.15** target. Mean model
probability on the 18 market favourites rises from 51.93% to 56.48%, still far
below the close's 76.25%.

The coefficient is identified in the expected band: the equal-fold mean of the 40
posterior means is **0.6379** (range **0.5493–0.7742**), and the average posterior
probability that it is positive is **0.99998**. Moving proxy xG into the linear
predictor therefore works in the intended direction. It does not recover all of
the market's scale.

The cost is material. The candidate loses **0.00103 aggregate LogLoss** to the
joint control, is worse on O/U 2.5 and BTTS, and returns **+83.3%** versus
+128.0% for the joint control and +139.3% for the Poisson control on the common
622-fixture portfolio panel. It is best of the three on 1X2 LogLoss, RPS and
aggregate ECE, but none of the paired LogLoss differences is resolved from zero.
Retain the prototype as evidence that fast form belongs outside the shrunk latent;
do not promote this exact Negative-Binomial formulation.

| Arm | Market on model slope ↓ | Favourite P(win) | Overall LogLoss ↓ | 1X2 LogLoss ↓ | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m01_poisson_time_decay` | 2.5288 | 48.62% | 0.646809 | 0.620351 | +139.3% | 1.204 | −22.84% |
| `m02_joint_gamma_poisson` | 1.7240 | 51.93% | **0.643730** | 0.617436 | **+128.0%** | **1.270** | −22.43% |
| `m03_negbin_pxg_covariate` | **1.3023** | **56.48%** | 0.644755 | **0.616690** | +83.3% | 1.057 | **−22.09%** |
| Betfair close | 1.0000 | 76.25% | 0.641816 | 0.613118 | — | — | — |

Bold compares models only. Return rankings are descriptive historical backtests,
not evidence of executable future profit.

## Formulation and filtration

The candidate implements the work package's coefficient parameterisation exactly:

\[
\eta_h = \mu + \gamma_h + \alpha_h + \beta_a + \tfrac12 w_{pxg}\,\Delta pxg,
\qquad
\eta_a = \mu + \alpha_a + \beta_h - \tfrac12 w_{pxg}\,\Delta pxg,
\]
\[
y_h \sim \operatorname{NegBin}(\exp\eta_h,r),\qquad
y_a \sim \operatorname{NegBin}(\exp\eta_a,r),\qquad
w_{pxg}\sim\mathcal N(0.60,0.20^2).
\]

`PxGFeature` supplies `Δpxg = (att_h + def_a) − (att_a + def_h)` from BBC
live-text commentary only (`fallback = :none`). It uses exponential match-time
decay with a 16-match half-life, a three-match league prior, and a two-match
minimum. The feature walk emits every same-day card before updating from any
fixture on that card, so neither the fixture itself nor a same-slot result can
enter its design. The custom covariate stores half of this column; consequently
`pxg_form.w` is the full effect on log-rate supremacy rather than a side-level
coefficient.

The three arms hold `GlobalInterception`, `TimeDecayDynamics(180)`,
`GlobalHomeAdvantage`, the splitter, sampler and panel fixed. `m02` adds the
canonical masked Gamma proxy-xG observation arm; `m03` instead adds the direct
form covariate and a global Negative-Binomial dispersion. The candidate's mean
`r` averaged over folds is **24.71** (fold-mean range 22.84–27.99), consistent
with Experiment 02's mild overdispersion.

The Phase-2 report's 1.07 pure-goals slope referred to its GRW posterior. This
work package explicitly prescribed a **TimeDecay(180)** control, whose slope here
is 2.5288; these numbers are not contradictory because they are different
team-dynamics models.

## Stage 0 — deterministic and AD verification

Executed on `mcmc-beast`, 16 pinned Julia threads and one BLAS thread.
`test_decompression.jl` passed **14/14 assertions**, including future-outcome and
same-day perturbation checks.

The optimized prototype and the production composable reference have identical
sampled-site layouts and prior initializations. At linked-space displacements
0, 0.003, ±0.8 and ±3:

| Fold | Parameters | Tape instructions | Warmed gradient | Replay allocation | Max density difference | Max gradient relative error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52 | 232 | 0.167 ms | **0 B** | 2.27e−13 | 3.03e−13 |
| 20 | 56 | 232 | 0.245 ms | **0 B** | 3.64e−12 | 2.87e−12 |
| 40 | 52 | 232 | 0.243 ms | **0 B** | 2.27e−13 | 1.22e−12 |

The density differences are floating-point re-association from scalar lifting,
not posterior changes; ForwardDiff, fresh ReverseDiff, the compiled tape and the
reference agree to the errors shown. Replay is allocation-free, although the
0.17–0.25 ms NegBin gradient is slower than the AD guide's aspirational 0.1 ms
Poisson-scale bar.

Feature preflight used **1,109 commentary observations and zero shot-count or goal
fallbacks**. Candidate fitted-row availability was 340/720, 676/1,060 and
1,034/1,060 on folds 1, 20 and 40 respectively; unavailable history contributes
the exact linear zero.

Evidence: [`verification/preflight/`](verification/preflight/).

## Stage 1 — smoke gate

Folds 1, 20 and 40 used four chains × (400 warmup + 400 retained), target
acceptance 0.90. All three arms passed the six-part convergence audit, latent
extraction, database fit round-trip, score partition check, common-panel portfolio
persistence, and identical re-pricing after reload.

| Arm | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Min BFMI | Smoke run UUID |
|---|---:|---:|---:|---:|---:|---|
| `m01` | 1.0150 | 535 | 674 | 0/4,800 | 0.712 | `abfa41e5-15f1-4060-9c4a-c59d53725ed3` |
| `m02` | 1.0190 | 427 | 436 | 0/4,800 | 0.617 | `1542e230-7e34-4b59-a5ed-e71746a8b66a` |
| `m03` | 1.0142 | 754 | 618 | 0/4,800 | 0.776 | `4715736d-6f53-4e7c-9696-9d1f6974a318` |

Candidate `w_pxg` means were **0.7804 / 0.5518 / 0.6579**; all lie in the required
0.40–0.80 band. Their 5th percentiles were 0.5933 / 0.3359 / 0.4195 and every
posterior draw was positive to reported precision.

The production score grid is intentionally truncated to 0–11 goals per side. Its
market partitions equal the grid's retained mass, rather than a mathematically
impossible exact unit mass for an infinite-support count law. Worst partition
error was 1.52e−13 for `m03`; worst omitted tail mass was 0.0316%. Thus the
literal “sum to 1 within 1e−12” wording is replaced by the repository's stronger
implementation-relevant invariant: every market is a coherent partition of the
same retained score tensor, with omitted mass reported rather than hidden.

The common smoke portfolio contained 44/50 fixtures. Evidence:
[`verification/stage1/`](verification/stage1/).

## Stage 2 — 40-fold production grid

All arms completed the canonical 40 folds / 710 unique OOS fixtures with four
chains × (800 warmup + 800 retained). All 3,200 retained draws per fixture were
persisted; no thinning was used.

| Arm | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Min BFMI | Fit wall | Model-run UUID |
|---|---:|---:|---:|---:|---:|---:|---|
| `m01` | 1.01005 | 943.6 | 786.6 | 0/128,000 | 0.632 | 2.62 min | `90a8c7bc-b55c-4ef2-8e65-07cdaefa2b29` |
| `m02` | 1.00787 | 748.4 | 852.8 | 0/128,000 | 0.570 | 5.60 min | `6d9970c8-df28-4c2a-a67d-a5741153e708` |
| `m03` | 1.01005 | 1,189.2 | 1,257.6 | 0/128,000 | 0.760 | 6.34 min | `f866b2bc-d87b-42d0-8c36-8c8d623d1178` |

All specified R-hat≤1.05 and ESS≥200 gates pass. The stricter advisory R-hat≤1.01
is missed by `m01` and `m03` by about 5e−5. Fits and relational latents round-trip
from PostgreSQL namespace `scottish_lower_decompression`. Evidence:
[`verification/production/`](verification/production/).

## Stage 3 — matched evaluation

### Coverage and decompression

All arms carry 710 forecasts. Proper scores cover **2,899 selections / 627
fixtures**. Market inversion accepts **623** and reports 87 refusals. The portfolio
uses **622 common fixtures**; 75 forecasts lack closing quotes and 13 have no
usable selection. Filters are reported separately because scoring, inversion and
tradability answer different questions.

Supremacy is `E[log λ_h − log λ_a]`. The headline slope is OLS of market
supremacy on model supremacy **with an intercept**; above 1 means compression.

| Arm | Slope | Intercept | R² | Model-on-market slope | Favourite model / market / realized |
|---|---:|---:|---:|---:|---:|
| `m01` | 2.5288 | −0.1833 | 0.5106 | 0.2019 | 48.62% / 76.25% / 72.22% |
| `m02` | 1.7240 | −0.0344 | 0.6003 | 0.3482 | 51.93% / 76.25% / 72.22% |
| `m03` | **1.3023** | +0.0029 | **0.6228** | **0.4783** | **56.48%** / 76.25% / 72.22% |

The covariate closes 0.4217 slope units versus the joint control but remains
0.3023 above ideal. Only 18 quoted home/away selections meet the 0.70 favourite
threshold, so the tail percentages are diagnostic rather than precise estimates.

### Proper scores

| Metric | `m01` | `m02` | `m03` | Betfair close |
|---|---:|---:|---:|---:|
| All-selection LogLoss | 0.646809 | **0.643730** | 0.644755 | 0.641816 |
| 1X2 LogLoss | 0.620351 | 0.617436 | **0.616690** | 0.613118 |
| O/U 2.5 LogLoss | 0.689973 | **0.687000** | 0.689826 | 0.689878 |
| BTTS LogLoss | 0.687569 | **0.683436** | 0.689510 | 0.683371 |
| All Brier | 0.227653 | **0.226196** | 0.226715 | 0.225293 |
| All ECE | 0.012151 | 0.013617 | **0.010909** | 0.013907 |
| 1X2 RPS | 0.227006 | 0.225070 | **0.224412** | 0.211104 |
| Goal-count CRPS | 0.630578 | **0.627766** | 0.629746 | — |

The candidate's aggregate LogLoss difference is −0.002054 versus `m01`
(95% paired fixture-bootstrap interval **[−0.005381, +0.001499]**) and +0.001026
versus `m02` (**[−0.002116, +0.004067]**). Its 1X2 differences are −0.003661
versus `m01` and −0.000746 versus `m02`; both intervals also cross zero. No
predictive superiority claim is supported.

### Portfolio

Identical `BookSpec(1X2, OU2.5, BakerMcHale)` and
`PolicySpec(FlatTrust(0.25), SlateDrawdown(20), FixedCap(0.25))` were applied to
the common 622-fixture panel.

| Arm | Bets | Return | Flat ROI | Sharpe | Max drawdown | Portfolio UUID |
|---|---:|---:|---:|---:|---:|---|
| `m01` | 1,323 | **+139.32%** | **12.96%** | 1.204 | −22.84% | `3d5f5290-a80a-4e68-9024-73829d341b59` |
| `m02` | 1,311 | +127.97% | 12.56% | **1.270** | −22.43% | `ee7bb72e-53b2-4459-a7a4-374699632641` |
| `m03` | 1,291 | +83.32% | 10.53% | 1.057 | **−22.09%** | `6324d2b0-d39b-4cd7-86a1-94a18898a02c` |

The candidate allocates 44.0% of summed stake fraction at odds ≥4.0, versus 49.3%
for `m02` and 55.2% for `m01`; decompression reduces long-shot allocation but does
not improve historical growth. No paired portfolio uncertainty interval was run.

## Limitations

- The prior centre (0.60) is informed by the same league's market attribution
  study. Posterior positivity is strong, but the result is not external validation.
- `w_pxg` summaries are per-fold posteriors over expanding histories. Their
  equal-fold average is descriptive, not one pooled posterior.
- The shot-xG cell table is history-fitted per fold and carries no team identity;
  the commentary form walk itself is strictly pre-match.
- The fixed 12×12 score tensor omits at most 0.280% on one production candidate
  draw. Tiny score differences should not be read below this numerical scale
  without a larger-grid sensitivity analysis.
- Closing prices assume availability, not fills, liquidity or market impact.
- The experiment changes both the observation law (joint Poisson → single-arm
  NegBin) and the route by which proxy xG enters. It answers whether the proposed
  complete candidate works, not a pure one-factor ablation between those changes.

## Reproduction

```bash
cd /root/BF_negbin_pxg_covariate
set -a; . ./.env; set +a                 # do not print credentials
J=/root/.juliaup/bin/julia
D=experiments/scottish_lower/11_decompression_pxg_covariate
$J --project -t 2  --startup-file=no "$D/test_decompression.jl"
$J --project -t 16 --startup-file=no "$D/r00_preflight.jl"
PXG_PREPARE_ONLY=true $J --project -t 16 --startup-file=no "$D/r10_smoke.jl"
$J --project -t 16 --startup-file=no "$D/r10_smoke.jl"
PXG_PREPARE_ONLY=true $J --project -t 16 --startup-file=no "$D/r20_production_grid.jl"
$J --project -t 16 --startup-file=no "$D/r20_production_grid.jl"
$J --project -t 16 --startup-file=no "$D/r30_evaluation.jl"
python "$D/verification/verify_results.py"
./scripts/todo.sh check
```

Source fingerprint for every accepted run:
`271a0aaa11965df52e158d63f2c23da5411f917bdb67e3bd54cce6e86742dc03`.
Exact CSV evidence is committed under [`verification/`](verification/); binary
fits, checkpoints and manifests remain under `results/` on `mcmc-beast`.
