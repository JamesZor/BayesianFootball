# 05 — Quality/Style Goal Model vs Market Model (TODO 040)

## Question

Does rotating team strength into quality (supremacy) and style (intensity), with the style/quality
step ratio r set from the market, improve 40-fold Scottish League One/Two goal forecasts over the
independent attack/defence MultiScaleGRW? How close do the goal arms come to the same quality/style
model fitted to the market itself?

## Arms

| Arm | Dynamics | Likelihood | Role |
|---|---|---|---|
| `control_grw` | `MultiScaleGRW()` | Poisson | **control** |
| `control_td` | `TimeDecayDynamics(180)` | Poisson | secondary control |
| `qs_market_r` | `QualityStyleGRW()`; r₀, rₖ ~ LogNormal(log 0.3, 0.35), rₛ ~ LogNormal(log 0.4, 0.6) | Poisson | candidate |
| `qs_weak_r` | `QualityStyleGRW`; r₀, rₛ, rₖ ~ LogNormal(0, 1) | Poisson | candidate |
| `market_c0` | Phase C C0 Gaussian Kalman QS model on inverted Betfair closes, refitted per fold | — | market arm (not a builder model) |

The goal arms use `GlobalInterception`, `GlobalHomeAdvantage` and the W2 sampler (4 × (500 + 1000),
acceptance 0.65). `QualityStyleGRW` uses α = q + s and β = s − q. Its per-side scales
(τ₀ ~ Gamma(2, 0.08), τₛ ~ Gamma(2, 0.0425), τₖ ~ Gamma(2, 0.0135)) have prior means equal to
the averages of the control's attack/defence prior means. The QS arms differ from each other only in r.

Scope `lower`: tournaments 56/57, two history seasons, `match_biweek`, targets 24/25 + 25/26. That
gives 40 walk-forward folds and 710 held-out fixtures.

## Decision rule

The classification follows the work package. On 1X2 LogLoss (primary) and O/U 2.5 LogLoss (secondary),
an arm is "better" if the 8-week block-bootstrap 90% interval of the paired difference lies wholly
below 0, and "worse" if wholly above. This is an experiment, not a promotion: no ROI or staking.

## Result (2026-10-09, scorecard v1.2)

**No promotion.** Neither QS arm improves 1X2 LogLoss over `control_grw`. Both are slightly worse
(+0.00159 and +0.00107), classed "worse" on the prescribed noncircular block interval. That call does
not survive a fixture-clustered or circular-block interval; see the caveat in [REPORT.md](REPORT.md).
What QS does buy is **decompression**: the compression slope falls from 1.19 to 1.01 (market r) and
1.05 (weak r). ECE over all markets also falls, from 0.020 to 0.013. Goals barely move r off the
market prior. Under a weak prior they pull r₀ and rₛ toward the market's ~0.3–0.5, but cannot
identify rₖ.

The market C0 arm has the best point 1X2 LogLoss (0.61343, vs 0.61312 for the close itself) and the
smallest transition bias. Its gap to every goal arm is not detectable.

| Arm | LL all | LL 1X2 | Δ 1X2 vs control [block 90%] | LL OU2.5 | Compression | Run |
|---|---:|---:|---|---:|---:|---|
| `control_grw` (ctl) | 0.64452 | 0.61678 | 0 | 0.68786 | 1.192 | `a036d22a` |
| `control_td` | 0.64685 | 0.62046 | +0.00368 [−0.00103, +0.00715] (sign-reversed pair) | 0.68983 | 2.782 | `1dccb320` |
| `qs_market_r` | 0.64520 | 0.61837 | +0.00159 [+0.00042, +0.00369] worse | 0.68708 | **1.007** | `b18ae74b` |
| `qs_weak_r` | 0.64472 | 0.61785 | +0.00107 [+0.00032, +0.00287] worse | 0.68682 | 1.051 | `21f2a9f9` |
| `market_c0` | **0.64240** | **0.61343** | −0.00335 [−0.00702, +0.00434] | 0.68859 | 1.023 | files |
| market close | 0.64182 | 0.61312 | — | 0.68988 | — | — |

Full tables, goal log-score channels, posterior r by fold and the bootstrap caveat are in
[REPORT.md](REPORT.md). The machine-readable results are in `results/`.

## Caveats

- **qs_weak_r's minimum tail ESS is 355**, below the engine's 400 review gate. Its R̂ is 1.0092
  with 0 divergences. The work package reruns only folds with R̂ > 1.05, so the run stands. Its saved
  config carries `convergence:FAIL`, which also hides it from `find_completed_run`.
- **The market arm is not a builder run.** Its per-fold C0 fits and latents are files under
  `/root/BF_runs/qs_experiment_out/market_grid/`. It is scored through the harness by
  `scoring_adapter.jl`, which matches the harness to |Δ| = 0.0 on a W1 run. Its harness rows are
  not written to `harness_scores`, because its UUID is not an `mcmc_experiments` run.
- **Market panel.** It uses four seasons (22/23–25/26), because goal folds 1–20 train on 22/23 + 23/24.
  The B2 rules reproduce the published 24/25 + 25/26 counts exactly (710/595/78/517). 517 of the
  710 targets have a full book. All 710 are forecast pre-week from earlier closes only.
- **Block bootstrap.** The prescribed noncircular 8-week scheme is off-centre on this panel by up
  to 0.0058. A circular-block sensitivity is reported alongside it; it does not replace the prescribed class.
- **Leaderboard.** Goal-arm grid scores are written to `harness_scores`. The shared
  `experiments/LEADERBOARD.md`, `LEADERBOARD.csv` and `REGISTER.md` were regenerated through
  `scripts/leaderboard.jl`; all four goal UUIDs are present. The market arm remains file-based.

## Reproduction

Pinned snapshot `.cache/datastore_ScottishLower.jls`, SHA256
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`. Everything ran on mcmc-beast in
owned `pi_qsx_*` REPLs from detached checkouts of `exp/qs-goal-vs-market`.

- Phase 1 tests: `test/test_quality_style_grw.jl` (373/373) and the tape-allocation case (0 B).
- Phase 2: `r02_phase2_checks.jl` (panel counts, fold leakage audit, adapter proof).
- Smokes: `r03_goal_smoke.jl` and `r03_market_smoke.jl`, at `2060056f`.
- Grids: `r04_goal_grid.jl` (`-t 14`, cores 0–13, seeds 4001–4004 in priority order) and
  `r04_market_grid.jl` (`-t 2`, then resumed at `-t 16`), at `e6685ceb`.
- Scoring: `r05_score.jl` (+ `r05_score_r_only.jl`, + `r05_block_check.jl`), at `81c77a42`+;
  `r05_figures.jl` renders the frozen posterior-r CSV.
- Reproduction: fresh-REPL `r06_reproduce.jl` at `ac04b173`: all six refits exact.
- Final record: `r07_finalize.jl` verifies persisted scores, upserts the completed decision and
  regenerates the shared leaderboard/register without resampling.

Logs are in `/root/BF_runs/logs/qs_experiment/`. Byte-for-byte refit checks are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
