# Momentum MultiScale GRW — Scottish Lower Phase 1

**TODO [022](../../../todos/022_prototype_momentum_multiscale_grw_dynamics.md)** ·
namespace `scottish_lower_momentum_grw` · **COMPLETED — 2026-09-22**

## Decision

**Phase 1 is technically successful but does not justify replacing first-order
GRW.** Momentum modestly decompresses supremacy (slope **0.3880 → 0.4338**) and
market-favourite probabilities (**55.01% → 56.43%**), but falls well short of the
market's **76.25%**. Aggregate LogLoss is effectively tied: improvement **0.000078**,
with a paired 95% bootstrap interval spanning zero. Momentum slightly worsens 1X2
LogLoss/RPS and earns less in this backtest (**+172.4% vs +197.1%**), while taking
**2.30×** as long to fit. Keep the component as a validated prototype; do not
promote it to the live model on these results.

| Arm | Supremacy slope | Favourite P(win) | Overall LogLoss ↓ | 1X2 LogLoss ↓ | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| TimeDecay (`m01`) | 0.2017 | 48.59% | 0.646849 | 0.620292 | +139.79% | 1.209 | −22.84% |
| First-order GRW (`m02`) | 0.3880 | 55.01% | 0.644518 | **0.616840** | **+197.10%** | **1.653** | −20.73% |
| Momentum GRW (`m03`) | **0.4338** | **56.43%** | **0.644441** | 0.617172 | +172.41% | 1.568 | **−20.21%** |
| Betfair close | 1.0000 (reference) | 76.25% | 0.641816 | 0.613118 | — | — | — |

These columns have different, explicitly audited coverage: 710 forecast fixtures,
627 scored fixtures, 623 accepted market-rate inversions, 622 common tradeable
fixtures, and 18 quoted favourites. LogLoss is the framework's **selection-level
binary cross-entropy**, including on 1X2; it is not categorical match-outcome
LogLoss. Full denominators, uncertainty and portfolio conventions are below.

## Scientific question

Does damped team velocity improve favourite-tail forecasts and proper scores
relative to pure-Poisson TimeDecay(180) and first-order MultiScaleGRW? The matched
cohort is 40 pooled-56/57 match-biweek folds, seasons 24/25–25/26, 710 fixtures.
No player, wealth, proxy-xG or smile terms are included.

See [DESIGN.md](DESIGN.md) for equations, priors, boundary conditions, stationarity
analysis and AD implementation. Velocity is stable for phi < 1; the level retains
a unit root. First-order GRW has zero expected increments, not a level that resets
to zero. Momentum is a hypothesis, **not a guarantee of decompression**.

User-approved forecasting: zero velocity at the target-season boundary;
conditional-mean next-biweek states, without future innovation noise, matching
the first-order control's convention. The terminal unobserved velocity innovation
is integrated out; K≤1 folds have no velocity parameters.

## Stage 0 — measured verification (2026-09-21)

Executed on `mcmc-beast`, `/root/BF_momentum_grw`, 16 pinned threads, BLAS=1.
`test_momentum.jl`: **202/202 assertions passed** (including the strict-zero audit adapter regression). Tests cover independent scalar
recurrences, sigma_v=0 nesting, phi endpoints, no-target/single-target shapes,
centering, multi-chain draw ordering, Turing-returned states, OOS reconstruction
and compiled/ForwardDiff gradient parity. Isolated momentum replay: **0 B**.

Full **linked-space** gradient replay, warmed minimum of 100 calls:

| Arm | Fold | Parameters | Tape instructions | Gradient ms | Allocated B |
|---|---:|---:|---:|---:|---:|
| TimeDecay | 1 | 50 | 196 | 0.0370 | 0 |
| TimeDecay | 20 | 54 | 196 | 0.0506 | 0 |
| TimeDecay | 40 | 50 | 196 | 0.0509 | 0 |
| First-order GRW | 1 | 98 | 610 | 0.0532 | 0 |
| First-order GRW | 20 | 1058 | 8284 | 0.3220 | 0 |
| First-order GRW | 40 | 974 | 7644 | 0.3007 | 0 |
| Momentum GRW | 1 | 98 | 610 | 0.0531 | 0 |
| Momentum GRW | 20 | 1962 | 15580 | 0.5662 | 0 |
| Momentum GRW | 40 | 1806 | 14364 | 0.5247 | 0 |

Original and optimized engines have identical sampled-site layouts and prior
initializations. Density discrepancies ≤ **4.7e-10**; gradient relative errors ≤
**3.6e-15**, comparing compiled, fresh ReverseDiff, ForwardDiff, and the original
engine, at linked-space displacements 0, 0.003, ±0.8 and ±3. No allocation
threshold was waived. Compilation/setup allocations are not replay allocations.

The source controls allocate 35–52 KB per replay. The prototype removes scalar
broadcast scratch, `fill` scratch, and scalarizing keyword centering without
changing priors, sample sites, weights or clamp limits. All modifications remain
local; [T002](../../../docs/tickets/T002-scalar-taped-likelihood.md) records the
shared-engine findings. The custom scalar-lift instruction is version-sensitive.

Prepare-only smoke preflight passed: exact 40/710 cohort, ordered filtration on
folds 1/20/40, all nine AD gates, canonical recipe registration and run-hash
lookup. No matching completed smoke runs existed at that preflight.

## Stage 1 — smoke protocol

Three arms, each 4 chains × (400 warmup + 400 retained), target acceptance 0.90,
max tree depth 10. Native queue, at most 16 concurrent chain tasks. Gates:

- zero divergences, max R-hat ≤1.05, bulk/tail ESS ≥200;
- six-part audit also requires BFMI ≥0.30 and tree-depth saturation ≤5%;
- exact OOS coverage, finite positive rate draws and coherent market partitions;
- fit/chain/latent database round-trip; portfolio persistence and identical
  re-priced bet ledger after loading the fit;
- report phi/sigma_v posterior summaries against their priors (identification is
  measured, not presumed; no velocity posterior exists on fold 1).

**User-approved score-grid clarification:** production scores truncate each side
to 0–11 goals without renormalization. Check all 1X2/OU2.5/BTTS partitions against
`cdf(Poisson(lambda_h),11)*cdf(Poisson(lambda_a),11)` within **1e-12** on every draw;
report omitted tail mass separately. This is not a claim that truncated mass is 1.

Only a passing, source-matched smoke certificate permits production. Failed fits
may be persisted for diagnostics, but cannot enter portfolio promotion.

### Measured Stage 1 outcome — PASS

All nine arm/fold combinations passed; 50 OOS fixtures, 1,600 draws per fixture.
All full-tape AD gates stayed allocation-free. No convergence metric abstained.

| Arm | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Min BFMI |
|---|---:|---:|---:|---:|---:|
| TimeDecay | 1.0118 | 552.8 | 578.4 | 0/4800 | 0.682 |
| First-order GRW | 1.0252 | 399.0 | 533.8 | 0/4800 | 0.668 |
| Momentum GRW | 1.0209 | 450.4 | 359.9 | 0/4800 | 0.749 |

No tree-depth saturation. The advisory R-hat≤1.01 bar was not met by every fold;
the specified acceptance bar remains ≤1.05. Maximum score-partition error was
1.78e-15; worst omitted tail mass was 0.000510 / 0.000384 / 0.001766 respectively.

**Audit correction, not resampling:** the initial runner mistakenly passed zero
to the shared audit's strict `<` divergence threshold, causing `0 < 0` failures.
It now uses the smallest positive Float64, plus an explicit zero-count check.
`r12_reaudit_smoke.jl` re-audited the exact original chains, preserved the original
UUIDs, and proved unchanged draw arrays. Both zero- and one-divergence cases are
regression-tested. No convergence requirement was relaxed.

**User-approved common portfolio panel:** 44/50 fixtures. Five lacked closing
quotes (`12477130`, `12476630`, `12476458`, `12476570`, `14032714`); `12473327`
had no usable selection. Every arm had exactly these refusals, with no construction
errors. All 50 remain in latent/grid checks. Fit/chain/latent database round-trips,
portfolio persistence and re-priced ledger equality passed for every arm on the
same 44-fixture panel. Production likewise uses a reported common tradeable panel rather
than silently excluding different fixtures per arm.

| Arm | Accepted smoke run UUID | Portfolio UUID |
|---|---|---|
| TimeDecay | `a05fb858-8033-4d4f-a805-509c5b5daab4` | `0adf69d9-9bc5-41ba-84b5-40b2bceb9e53` |
| First-order GRW | `02c1d10a-515f-4592-ba97-c895f8b38895` | `b2ace66e-57ca-4466-b9b6-68a307ce01f8` |
| Momentum GRW | `112bb865-c0e5-470a-b53a-619909367ced` | `9a1fcfde-fb0b-4f2f-9cdc-9ee6d4af4b83` |

Certificate source: `6869de6297b580cdc7aaf3052ba31c81c1facc0bc826d495e770bfbbb4a73c6f`.
Auditable CSVs, including original UUID lineage, are in [verification/stage1/](verification/stage1/).

Momentum persistence is **weakly identified**: posterior phi SDs 0.199–0.219 versus
prior SD 0.224. Attack sigma_v means are 0.00935/0.01182 on folds 20/40 versus prior
0.015; defence means 0.01064/0.01024 versus prior 0.012. On this small smoke panel,
supremacy slopes are 0.249/0.358/0.388. Only **three** quoted favourites meet ≥0.70;
their model probabilities are 0.518/0.511/0.521 versus market 0.750. These are
mechanical diagnostics, not evidence of a full-cohort performance win.

## Stage 2 — production protocol

The all-fold prepare-only run passed: 40 folds, 710 fixtures, source-matched smoke
certificate, canonical recipes and no existing completed production matches.
Budget stays 4×(800 warmup + 800 retained), 16 concurrent chain tasks.

**User-approved storage contract:** all 3,200 retained draws per fold are audited
and saved in full local fits; PostgreSQL stores every fourth draw (800 per fold),
with latents reconstructed from those exact draws. The same stride applies to all
arms. Full-draw diagnostics are retained. This avoids the current hex-encoded
single-artifact limit; the momentum smoke artifact alone is 49 MB. Stored recipe
descriptions explicitly record the persistence stride. Stage 3 compares the
matched persisted panels. Storage preflight passed on all three smoke fits:
reconstructed thinned rates equal the exact original draw columns; full-draw
diagnostics are preserved.

**Stage 2 completed, exit 0**, sampled at commit `a6cca1bc` on beast. All three
arms produced 40 folds / 710 fixtures; 128,000 retained draws per arm were audited
before thinning. The production manifest and all fit/chain/latent PostgreSQL
round-trips passed. No convergence gate abstained.

| Arm | Max R-hat | Min bulk ESS | Min tail ESS | Divergences (retained) | Min BFMI | Depth-cap rate | Fit wall min |
|---|---:|---:|---:|---:|---:|---:|---:|
| TimeDecay | 1.01178 | 1032.3 | 978.0 | 0/128000 | 0.643 | 0.0000% | 1.75 |
| First-order GRW | 1.01469 | 780.2 | 590.4 | 0/128000 | 0.687 | 0.0320% | 23.14 |
| Momentum GRW | 1.01357 | 831.4 | 832.4 | 0/128000 | 0.660 | 0.5930% | 53.27 |

All specified gates pass; the advisory R-hat≤1.01 threshold is not universally
met. Fit wall times exclude the subsequent persistence/evaluation stages.

### Production lineage

| Arm | Model-run UUID | Portfolio-run UUID |
|---|---|---|
| TimeDecay | `33d85b4a-e929-4738-8125-706e0dc26de1` | `4a0f91fb-bad4-47dd-b789-2631cb1feb25` |
| First-order GRW | `f8da493d-db2c-42d5-85e2-0f19d0107b1d` | `24473994-fa82-4949-b884-51b3fe50c374` |
| Momentum GRW | `3e06683b-de96-431f-843d-f98619d9fc13` | `beee109b-3947-4c19-ba15-80bb061e2d68` |

Recipe hashes, exact source identity, full-fit paths, filtration and diagnostics:
[verification/production/](verification/production/). Full local fits and the
production manifest remain under
`results/production/6869de6297b580cdc7aaf3052ba31c81c1facc0bc826d495e770bfbbb4a73c6f/`
and `results/production_manifest.jls`, respectively, on beast. Both fit and
portfolio UUIDs address `mcmc_experiments`, not the operational `betdb`.

## Stage 3 — measured benchmark (2026-09-22, exit 0)

`r30_momentum_evaluation.jl` loaded the pinned production UUIDs; it did not refit
models or regenerate latent panels. The following tables use the matched **800
persisted draws per fixture**, while convergence above uses all 3,200 retained
training draws. Exact CSVs: [verification/evaluation/](verification/evaluation/).

### Coverage and market contract

The close is the de-vigged Betfair **TWA (−20, 0] minutes**. Every arm has the same
710 OOS fixture IDs and scores exactly the same quoted selections/outcomes:

| Quantity | Coverage |
|---|---:|
| Forecasts / goal-count CRPS | 710 fixtures |
| Headline proper scores | 2,899 selections across 627 fixtures |
| 1X2 scores | 1,785 selections / 595 complete fixtures |
| O/U 2.5 scores | 758 selections / 379 fixtures |
| BTTS scores | 356 selections / 178 fixtures |
| Supremacy comparison | 623 accepted market-rate inversions |
| Favourite tail (home or away close ≥0.70) | 18 fixtures; independent of inversion acceptance |
| Portfolio comparison | 622 common tradeable fixtures |

The 87 inversion refusals are 77 fixtures with no quoted selections and 10 with
only two selections (minimum three required). Portfolio refusals are identical
across arms: **75 missing closing quotes + 13 without a usable selection**.
No portfolio construction errors or identity/unplayed exclusions were accepted.
All refused IDs/reasons and the common portfolio panel are committed. Scoring,
inversion and tradability filters answer different questions; their counts are
not interchangeable. The remaining 83 forecast fixtures are unscored against
closing markets, not removed from the latent panel.

### Proper scores and calibration

Lower is better. ECE uses 10 bins. RPS is ordered 1X2 only. CRPS is the existing
framework's **Poisson-at-posterior-mean-rate marginal** score, averaged over home
and away, not a posterior-mixture CDF score. Selection odds alone do not supply a
unique goal-count distribution, so no direct market CRPS is claimed.

| Metric | TimeDecay | First-order GRW | Momentum GRW | Betfair close |
|---|---:|---:|---:|---:|
| All-selection LogLoss | 0.646849 | 0.644518 | **0.644441** | 0.641816 |
| 1X2 LogLoss | 0.620292 | **0.616840** | 0.617172 | 0.613118 |
| O/U 2.5 LogLoss | 0.690225 | 0.687756 | **0.687506** | 0.689878 |
| BTTS LogLoss | **0.687655** | 0.691234 | 0.689472 | 0.683371 |
| All-selection Brier | 0.227674 | 0.226628 | **0.226598** | 0.225293 |
| All-selection ECE | **0.011616** | 0.017258 | 0.014241 | 0.013907 |
| 1X2 ECE | 0.017192 | **0.017058** | 0.018560 | 0.018581 |
| O/U 2.5 ECE | 0.015489 | 0.019562 | **0.014196** | 0.018299 |
| BTTS ECE | **0.005490** | 0.037676 | 0.030921 | 0.030044 |
| 1X2 RPS | 0.226944 | **0.225012** | 0.225142 | 0.211104 |
| Goal CRPS (home/away average) | 0.630635 | 0.626921 | **0.626824** | — |
| Home-goal CRPS | 0.640403 | 0.638382 | **0.637307** | — |
| Away-goal CRPS | 0.620867 | **0.615460** | 0.616340 | — |

Bold identifies the best **model**, not necessarily a win over the market. Momentum
improves aggregate/OU/BTTS point scores relative to first-order GRW, but worsens
1X2 LogLoss, 1X2 ECE and RPS. Its aggregate LogLoss gain is only **0.00007755**.
The close remains better on aggregate LogLoss, Brier, RPS and 1X2/BTTS LogLoss;
the GRWs' OU point-score advantage is not statistically established below.

### Paired uncertainty

4,000 paired **fixture-clustered** bootstrap resamples, seed 22, preserving all
selections within each sampled fixture. Delta is momentum minus comparator;
negative favours momentum. These intervals describe this scored cohort, not
parameter-posterior intervals or a guarantee of future performance.

| Comparator / scope | Δ LogLoss | 95% bootstrap interval | Fraction Δ<0 |
|---|---:|---:|---:|
| First-order / all | −0.000078 | [−0.001181, +0.001071] | 54.0% |
| First-order / 1X2 | +0.000332 | [−0.000943, +0.001611] | 30.5% |
| First-order / O/U 2.5 | −0.000250 | [−0.002473, +0.002012] | 59.1% |
| First-order / BTTS | −0.001762 | [−0.004207, +0.000647] | 92.0% |
| TimeDecay / all | −0.002409 | [−0.007300, +0.002547] | 81.3% |
| Market / all | +0.002625 | [−0.004722, +0.009917] | 23.1% |

Every listed interval crosses zero. The full 28-comparison table, including
per-market comparisons against TimeDecay and the close, is retained in
`paired_logloss.csv`. Treat the aggregate momentum/first-order difference as a
near tie, not a predictive win.

### Decompression and favourite tail

Supremacy is `E[log(lambda_h) − log(lambda_a)]`; slope/intercept are OLS against
log-rate supremacy from accepted market inversions, with an intercept fitted.

| Diagnostic | TimeDecay | First-order GRW | Momentum GRW |
|---|---:|---:|---:|
| Supremacy slope | 0.2017 | 0.3880 | 0.4338 |
| Supremacy intercept | 0.1084 | 0.0706 | 0.0624 |
| R² versus close | 0.5119 | 0.4141 | 0.4415 |
| Mean favourite P(win), 18 fixtures | 48.59% | 55.01% | 56.43% |
| Same favourites: market / realized | 76.25% / 72.22% | 76.25% / 72.22% | 76.25% / 72.22% |

Momentum increases the first-order slope by **0.04581** (11.8% relative) and the
favourite mean by **1.42 percentage points**. The favourite market gap is still
**19.82 pp**. Thirteen of 18 favourites won; that small sample does not support
strong tail-calibration conclusions. Directional decompression occurs, but the
hypothesized leap into the market's heavy-favourite regime does not.

### Momentum identification

Equal-weight averages of per-fold posterior summaries, **not a pooled posterior
or 36 independent datasets**. There are 36 active-velocity folds; the first two
folds of each season omit the velocity block (K≤1).

| Side / parameter | Prior mean / SD | Mean of posterior means | Range of fold means | Mean posterior SD |
|---|---:|---:|---:|---:|
| Attack phi | 0.500 / 0.2236 | 0.4759 | 0.4223–0.5137 | 0.2179 |
| Defence phi | 0.500 / 0.2236 | 0.5014 | 0.4421–0.5701 | 0.2211 |
| Attack sigma_v | 0.015 / 0.01061 | 0.01306 | 0.00912–0.01572 | 0.00888 |
| Defence sigma_v | 0.012 / 0.00849 | 0.01226 | 0.01014–0.01498 | 0.00829 |

Persistence remains largely prior-driven. Attack velocity scale is somewhat
shrunk; defence velocity scale is almost unchanged from its prior. More latent
states and extra runtime have not produced strong evidence of identified,
high-persistence momentum.

### Portfolio and capital allocation

Identical `BookSpec(1X2, OU2.5, BakerMcHale)` and
`PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`, with default
`DeArb`, Kelly log-utility allocation, daily slates and **2% per-bet commission**.
Book completeness rules are unchanged. Quoted backtest period: 2024-08-03 through
2026-04-25. Portfolio artifacts and re-priced ledgers round-trip identically for
all three production fits on the common 622-fixture panel.

| Metric | TimeDecay | First-order GRW | Momentum GRW |
|---|---:|---:|---:|
| Compounded total return | +139.79% | **+197.10%** | +172.41% |
| Annualized Sharpe (slate log returns) | 1.2086 | **1.6532** | 1.5682 |
| Maximum drawdown (signed) | −22.84% | −20.73% | **−20.21%** |
| Reported flat ROI¹ | 12.97% | **14.53%** | 13.95% |
| Equal-unit-stake ROI² | 3.95% | 4.28% | **5.17%** |
| Bets | 1314 | 1331 | 1326 |
| Capital fraction at odds ≥4.0³ | 55.18% | 38.43% | 37.22% |
| Capital fraction at odds ≤1.8³ | 0.00% | 3.78% | 4.39% |

¹ The API's `flat_roi_pct` is `100*sum(pnl)/sum(stake)` using fractions of each
slate's opening bankroll; it is stake-weighted/path-normalized, not equal stakes.
² Computed separately as `100*mean(payoff)` over each arm's selected bet ledger.
³ Shares of summed **stake fractions**, not shares of cumulative currency turnover.

Momentum reduces extreme-underdog allocation by only **1.21 pp** versus first-order
GRW and raises short-odds allocation by **0.61 pp**. Return is **24.68 pp lower**,
Sharpe **0.085 lower**, with drawdown only **0.52 pp shallower**. The better equal-
stake ROI does not translate into better compounded sizing performance. No paired
portfolio significance interval was computed; these are descriptive historical
backtests, not executable-return promises. Closing-price availability is not
proof of fills, capacity or future edge.

### Numerical limitations and research conclusion

The retained-mass gate passes to ≤**1.89e-15**, but the fixed 0–11 score grid does
**not** conserve unit mass. Maximum omitted mass on a single draw:

| Arm | Persisted 800-draw panel | Full 3200-draw panel |
|---|---:|---:|
| TimeDecay | 0.0809% | 0.0809% |
| First-order GRW | 1.0017% | 1.2524% |
| Momentum GRW | 2.2716% | **5.3125%** |

These are worst-case draw losses, not posterior-average losses; no average-tail
claim is made. They are material enough to warrant a larger/adaptive-grid
sensitivity check before interpreting tiny proper-score differences or promoting
the model. The accepted gate checks consistent partitions of the retained mass,
not negligible truncation.

Other boundaries: conditional-mean state forecasts omit future process noise;
velocity resets each target season; the calibration tail contains only 18 fixtures;
market coverage is incomplete; evaluation uses explicitly thinned draws; and this
is a pure-Poisson experiment, not a test of the live joint/player-lineup model.
No prior or model was retuned after seeing these evaluation outcomes.

**Sign-off:** the scoped prototype, smoke/production gates, persistence, scoring,
decompression diagnostics and matched portfolio benchmark are complete. The
breakthrough hypothesis is not supported at the required practical magnitude.
Keep first-order GRW as the preferred control for this comparison; retain the
momentum prototype for research. Any future process-noise, grid-size, prior or
joint-observation variant needs its own frozen protocol and validation cohort.

## Reproduction

```bash
cd /root/BF_momentum_grw
# Load environment without printing credentials.
set -a; . ./.env; set +a
J=/root/.juliaup/bin/julia
D=experiments/scottish_lower/10_momentum_multiscale_grw
$J --project -t 16 --startup-file=no "$D/test_momentum.jl"
$J --project -t 16 --startup-file=no "$D/r00_momentum_preflight.jl"
MMG_PREPARE_ONLY=true $J --project -t 16 --startup-file=no "$D/r10_momentum_smoke.jl"
$J --project -t 16 --startup-file=no "$D/r10_momentum_smoke.jl"
# Existing completed recipes are loaded rather than sampled again.
$J --project -t 16 --startup-file=no "$D/r20_momentum_production_grid.jl"
$J --project -t 16 --startup-file=no "$D/r30_momentum_evaluation.jl"
# Read-only CSV audit; no Julia packages, database or sampling needed:
python "$D/verification/verify_results.py"
./scripts/todo.sh check
```

CSV evidence is committed under `verification/{stage1,production,evaluation}/`.
Runtime artifacts are under `results/{smoke,production,evaluation}/<source SHA256>/`;
recipes/checkpoint directories include source identity. Stage 2 and Stage 3 both
finished with exit 0; Stage 3 ran at commit `168d90c4`. The read-only CSV sign-off
audit verifies matching scores/outcomes/market probabilities, cohort sizes,
convergence summaries, common portfolio exclusions and ledger arithmetic. The `MomentumGRW` module must be included before loading
prototype fits from PostgreSQL. DataStore caches and binary fits are not committed.
