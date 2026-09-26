# Experiment harness Phase 1 report

Date: 2026-09-26  
Task: [TODO 030](../../todos/030_shared_experiment_harness_recipes_gates_runner_scorecard_leaderboard.md)  
Scope: Phase 1 only

## Result

Phase 1 is complete. `BayesianFootball.Harness` now owns the versioned scorer, transition
cohorts, additive score/register persistence, and leaderboard projection. The W0 backfill
loaded and scored all 19 pinned runs without sampling.

Implementation commits:

- `3f09a1eb` — scorer, cohorts, store, scripts, seeds, and offline tests.
- `0426d681` — prototype/old-artifact loader support used for the W0 backfill.
- `0a792a7f` — common-panel refusal and stable register ordering.
- `ef6aa517` — Julia 1.12 latest-world entry for dynamically included prototype types.

## Acceptance evidence

### 1. r04 reproduction

Command (on `mcmc-beast`, detached tmux session `harness_p1`, checkout `0426d681`):

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 \
  scripts/score_runs.jl repro_runs.csv
```

It scored 4/4 runs and wrote 2,240 rows. For both `all` and `1X2`, every requested
LogLoss, Brier, and ECE below is bit-for-bit equal to
`current_development/grw_pyramid_cups/results/evaluation/r04_proper_scores.csv` (absolute
error `0.0`, tolerance `1e-9`).

| model | all LL | all Brier | all ECE | 1X2 LL | 1X2 Brier | 1X2 ECE |
|---|---:|---:|---:|---:|---:|---:|
| g1 | 0.6464436634968842 | 0.2275066275801324 | 0.012213458158100012 | 0.6193889755524499 | 0.2145546923020737 | 0.02235801173976079 |
| g2 | 0.6461676791106672 | 0.2273692266192647 | 0.018058234380821655 | 0.6163358652442303 | 0.21312617014395407 | 0.016664171710867974 |
| g3 | 0.6438500016918945 | 0.22624733706490774 | 0.009529611988978368 | 0.6171812581927506 | 0.21360241066435334 | 0.011644677879366258 |
| m12_td | 0.6433697999906764 | 0.22604583756864916 | 0.00995616723169428 | 0.6163628377808433 | 0.21311380224484566 | 0.015540487807250399 |

The v1 `Calibration.invert_market_rates` slope also agrees closely with r04's independent
1X2-only inversion; an alternative v1 definition is not needed.

| model | harness v1 | r04 | difference |
|---|---:|---:|---:|
| g1 | 1.0993304510 | 1.0995700477 | -0.000240 |
| g2 | 1.2517049012 | 1.2519395093 | -0.000235 |
| g3 | 1.1046918867 | 1.1049022477 | -0.000210 |
| m12_td | 1.5697463987 | 1.5700508231 | -0.000304 |

All differences are far inside the required `±0.02`.

### 2. W0 complete

Command (same host/session/SHA):

```bash
julia --project -t 16 scripts/seed_register.jl
julia --project -t 16 scripts/score_runs.jl experiments/scotland/W0_runs.csv
```

Result: **19/19 runs loaded, 10,640 rows written** (`19 × 560`) to
`harness_scores`, with all eight transition subset labels populated. No run had a panel
different from the control's 710 fixtures. Generated views are committed as
[`experiments/LEADERBOARD.csv`](../../experiments/LEADERBOARD.csv) and
[`experiments/LEADERBOARD.md`](../../experiments/LEADERBOARD.md).

W0 headline diagnostic (bias is transitioning-club model minus close win probability on
`transition_any_first20`, percentage points):

| model | compression slope | transition bias pp |
|---|---:|---:|
| m12_td | 1.570 | +1.793 |
| m05_joint_td | 1.870 | +1.212 |
| m13_joint_td | 1.488 | +1.858 |
| m12_grw | 1.328 | +2.366 |
| m05_joint_grw | 1.243 | +2.738 |
| m00_baseline_grw | 1.190 | +2.539 |
| g1_grw_all_spfl | 1.099 | +1.404 |
| g2_grw_all_spfl_cups | 1.252 | +1.825 |
| g3_grw_joint_all_spfl_cups | 1.105 | +2.024 |
| s10_m01_td_poisson | 2.787 | +0.823 |
| s10_m02_grw_poisson | 1.186 | +2.524 |
| s10_m03_momentum_grw | 1.126 | +2.512 |
| s11_m01_td_poisson | 2.773 | +0.815 |
| s11_m02_td_joint | 1.931 | +1.304 |
| s11_m03_negbin_pxg | 1.485 | +1.769 |
| s12_m01_td_poisson | 2.781 | +0.803 |
| s12_m02_td_joint | 1.928 | +1.307 |
| s12_m03_funnel_shared | 2.232 | +1.420 |
| s12_m04_funnel_hier | 2.336 | +1.408 |

These are descriptive cohort diagnostics, not causal estimates. The promoted and relegated
biases often have opposite signs; the direction-specific tables in `LEADERBOARD.md` should
be read instead of treating `:any` as one homogeneous population.

### 3. Register

`scripts/seed_register.jl` seeded **23 rows** into `harness_experiments`: suites 01–13
(including an explicit absent/not-run row for 09), TODOs 021–029, and the pyramid/cups
study. [`experiments/REGISTER.md`](../../experiments/REGISTER.md) contains question,
dimension, status, decision, run addresses, and source record for every row.

### 4. Tests

Standalone:

```text
julia --project -t 8 -e 'using Test, BayesianFootball, DataFrames, Dates;
include("test/harness_scoring_tests.jl")'

Harness transition cohorts                 9/9 pass
Harness fixture-clustered bootstrap         3/3 pass
Harness compression slope convention        3/3 pass
Harness per-subset score row counts         5/5 pass
Total                                      20/20 pass
```

Parallel runner:

```text
julia --project -t 8 test/run_parallel_tests.jl
19 / 20 suites passed
```

The sole failure was the pre-existing T007 signature in `features_tests.jl`:
`UndefVarError: SplitClockProbe not defined in Main`. The newly added
`harness_scoring_tests.jl` passed. This is no worse than baseline and no threshold or test
was weakened.

### 5. REVIEW.md discrepancies

#### (a) Suites 01/02 versus unified

Two `m00_baseline` UUIDs exist. Run `013af743-853d-481a-abc7-03a233fc8420` is explicitly
`synthetic-no-mcmc` (zero-second duration); run
`2722f7e2-0ee6-4040-95cc-55420800b1c3` is the later real ReverseDiff rerun. Both have 40
folds / 710 relational latents. Harness v1 gives:

| artefact | harness-v1 LogLoss | Brier | ECE |
|---|---:|---:|---:|
| synthetic `013af743` | 0.6545486902608844 | 0.23118818193789797 | 0.030086551277384337 |
| real `2722f7e2` | 0.646785678933513 | 0.22763803149663206 | 0.012169447072339284 |

The synthetic row reproduces the unified report exactly. That report was generated on
2026-09-03; the real rerun was created on 2026-09-10, so it could not have been the report's
input. Suite 01's 0.6603 came from its original local MCMC fit, which was not the artefact
that the later sync exposed under the unified report.

There is a large scorer/input change. Suite 01's
`r20_train_5models_2426_unified.jl` calls `evaluate_predictions(fit, ds)` against
`ds.odds`, whereas harness v1 uses the de-vigged Betfair TWA(−20,0] frame. On the **same
real UUID `2722f7e2`**, the old call scores model LogLoss **0.6603526199679975** and market
LogLoss **0.6568108034651682** over 4,955 rows—reproducing suite 01's 0.6603/0.6568—while
harness v1 scores the model 0.646785678933513 over the 2,899-row Betfair panel. Thus the old-versus-
current scorer disagreement is real and material. The exact unified 0.6545 row has an
additional confound: that report used the synthetic fallback, not the later real rerun.

#### (b) TimeDecay Poisson 0.6545 versus 0.6468

The main cause is now identifiable: unified's 0.6545 row was the deterministic
`synthetic-no-mcmc` fallback, while suites 10–12 are real posterior fits. The later real
Gen-1 rerun itself scores 0.646785678933513, matching the three later controls and ruling out the
model name as the explanation.

| run | UUID | kind / guard | NUTS accept | harness-v1 LL | config hash |
|---|---|---|---:|---:|---|
| unified input | `013af743-853d-481a-abc7-03a233fc8420` | synthetic / `ClampGuard` | nominal 0.65 | 0.654549 | `417a7fd6…` |
| later Gen-1 rerun | `2722f7e2-0ee6-4040-95cc-55420800b1c3` | real / `ClampGuard` | 0.65 | 0.646786 | `a5cfff7b…` |
| suite 10 | `33d85b4a-e929-4738-8125-706e0dc26de1` | real / `ArrayClampGuard` | 0.90 | 0.64685 | `3581efb6…` |
| suite 11 | `90a8c7bc-b55c-4ef2-8e65-07cdaefa2b29` | real / `ClampGuard` | 0.90 | 0.64681 | `48834ee7…` |
| suite 12 | `de7fa956-87e8-418f-afb4-61ce01cb9f7d` | real / `ArrayClampGuard` | 0.90 | 0.64679 | `561cea95…` |

All real runs use the same `GroupedCVConfig` (24/25 + 25/26, two history seasons,
`match_biweek`) and hold 40 folds / 710 latents. They were produced at different dirty Git
commits; suite 10 also records `persistence_stride=4`. Their nominal config hashes differ,
but their scores agree within 0.00006. It would therefore be wrong to attribute the old
0.0077 gap to target acceptance or the guard adapter; it is the synthetic fallback row that
made the unified control incomparable.

### 6. Repository checks

```text
./scripts/todo.sh check
OK: 31 task(s); metadata, template and registry agree; AGENTS.md 19673 bytes (< 22000).

git diff --check
(no output)
```

TODO 030 remains `ACTIVE`, as requested, with dated `@pi` work-log entries.

## W0 findings

- The target LogLoss leader is `m05_joint_td` (0.64299), but its 95% paired interval versus
  m12_td crosses zero. The top nine rows through `s10_m03_momentum_grw` also have intervals
  crossing zero.
- GRW sharply decompresses the pure Poisson controls (about 2.78 to 1.19), consistent with
  the prior suite result. Cup pooling gives g3 a near-ideal 1.105 slope, but no target
  LogLoss win over the joint lower-tier controls.
- A single aggregate transition bias hides direction: for example, several models are
  negative on relegated-into-L1 and positive on promoted-into-L1. Phase 2 selection should
  use the pre-registered direction-specific cohorts.

## Open questions for Phase 2

1. Confirm that scoped target and monitor folds preserve these exact 710 target IDs before
   using cross-tier monitor scores for selection.
2. Decide whether transition bias should also be reported fixture-weighted when both clubs
   transitioned; v1 deliberately averages transitioning-club observations.
3. Replace synthetic historical registry rows with explicitly identified real reruns before
   any Phase-2 cross-suite selection; never resolve a report by run name alone.
4. Keep the v1 slope definition fixed; the r04 cross-check is close enough that no dual
   production definition is warranted.

## Blocked

None.
