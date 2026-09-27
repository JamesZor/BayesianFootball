# Wave 2 Phase B — scoring handoff (2026-09-27)

## Scope and provenance

- Item 0: `8effca72c60a51284985dbd7be1bd3ddae552ede`, pushed before subsequent scoring changes. The empty-target GRW summer innovation is centred **per draw across teams**, with the fold/side RNG seed retained. Synthetic serialized-chain reload parity and per-draw zero sums pass (pyramid suite 94/94 on this checkout; item-0 run 94/94).
- Core/CLI/register: `af725704cb7a0cde40e4e088330d91b570f133e0`, pushed. W1 target-market RPS correction: `5e72e8488469d79a48cb59bf1a97a232adc4dc60`, pushed. Finalist-panel/schema documentation: `25a3b3f48f36ec837f300c376fbf3e636861fbf1`; legacy priced-panel backfill: `151e8af6119a10c2501b77bd9de6a7929b3e5109`. This report's last edit is on the subsequent branch tip.
- No sampling or grid job was started or interrupted. Before remote validation, checked `pgrep -af '[r]un_candidates.jl'` on `root@mcmc-beast` (no job). Both re-scores used immutable `scripts/beast_checkout.sh <sha>`, `nice -n 19 julia --project -t 4`, a credentials-loaded environment **without printing it**, and logs outside checkout under `/root/BF_runs/logs/<sha>/`.
- Manager update: W2 smoke/grid rows were removed because of the engine allocation bug; there is no W2 run validation claimed here. Another branch owns the engine/checks/smoke fix.

## Implemented

1. **Panel.** `score_runs` and `score_fits` accept `target_seasons` and `expected_fixtures`; CLI accepts `--target-seasons` and `--expected-fixtures N|none`, defaulting to W1's 24/25,25/26 and 710 in the CLI and `score_runs`. `grid` uses `candidate.scope.target_seasons` and the known W1/W2 fixture-count gates (710/1070); other scopes have no assumed fixture count. v1.2 rows carry `panel`; the additive schema migration backfills historical target panels from **each run's target/all fixture count** (627 priced W1 fixtures map to its 710-fixture target panel; also recognizes 710/1070, otherwise `unknown`), then derives historical finalist seasons from that run's grid target row and keeps the finalist's own buildable count. The leaderboard groups and ranks by panel and labels each grid, cohort and MAP section. No v1.1 rows were deleted or overwritten.
2. **Per-row controls.** Optional CSV `control` values resolve to a CSV label or run UUID and group each candidate with its own base before calling `score_runs`; the old CSV with one `role=control` and no `control` column follows the original path. Paired deltas retain the `control_run_id`.
3. **(g).** For every transition subset, `transition_bias_pp` now carries fixed-seed, B=10,000 club-season-clustered percentile 95% limits. The point estimate retains v1.1's original per-cohort iteration and `mean` order (confirmed below). `delta_abs_bias_vs_control` resamples paired (club, season) units and computes `100 × (|mean arm bias| − |mean base bias|)` at each resample; identical arms have an exactly zero point and degenerate interval.
4. **(i).** `score_runs.jl` uses `ScottishPyramid` so stored 54/55 monitor latents can be scored. Target contexts use the old 56/57 odds universe: Evaluation's `market_rps` traverses all odds rows, so a pooled context would otherwise change an old W1 target metric by 0.01107116 even though target model scores were unchanged. That discrepancy was found in the first 12-run parity re-score and corrected in the second.
5. **(h).** `write_experiment!` uses an atomic ON-CONFLICT set union of run IDs, and an additive UUID-keyed `run_commits` JSONB map. `run_candidates.jl --only` supplies per-fit **original** commit provenance, including resumed/portfolio fits. A real DB test made two sequential upserts with separate IDs/commit maps, asserted both survived, and deleted only the synthetic register row: `REGISTER_MERGE_PASS`.
6. **Finalist continuity.** v1.2 portfolio score rows now explicitly name their smaller common buildable W1 panel, rather than failing the v1.2 store's explicit-panel requirement.

## Validation commands and outcomes

Local: `julia --project -e 'include("test/harness_scoring_tests.jl")'` (43/43), `include("test/harness_runner_tests.jl")` (65/65), `include("test/harness_portfolio_tests.jl")` (27/27), and `include("test/pyramid_tier_tests.jl")` (94/94). Synthetic tests cover wider clustered than fixture-resampled CIs for duplicated fixtures, fixed-seed reproducibility, bit-identical point order, degenerate/sign-correct paired delta, two distinct control groups in one CSV, two-panel leaderboard tables, target-subset invariance under a larger datastore, and sequential register merges. `julia --project -t 8 test/run_parallel_tests.jl`: 23/24 suites passed; the only failure is the known T007 `features_tests.jl` missing `SplitClockProbe` when isolated.

Remote W1 command at `5e72e848`: `nice -n 19 julia --project -t 4 scripts/score_runs.jl experiments/scotland/03_dynamics_scope_matrix/W1_runs.csv --target-seasons 24/25,25/26 --expected-fixtures 710` → **12/12 runs, 8,388 v1.2 rows**; log `/root/BF_runs/logs/5e72e8488469d79a48cb59bf1a97a232adc4dc60/w1_rescore.log`. The bounded DB comparison joined old and new rows by run UUID, subset, market, metric and control UUID: **7,848 shared rows**, 12 runs, max absolute value difference **0.0 for every shared metric** (including transition bias). Explicit maxima: brier 0; compression_slope 0; delta_logloss_vs_control 0; ece 0; logloss 0; market_brier 0; market_ece 0; market_logloss 0; market_mce 0; market_rps 0; mce 0; model_on_market_slope 0; rps 0; transition_bias_pp 0. Log: `/root/BF_runs/logs/5e72e8488469d79a48cb59bf1a97a232adc4dc60/parity.log`. The old table also has historical extra control comparisons not in this invocation; the new table adds 120 delta-absolute-bias rows and monitor rows. The final schema backfill from checkout `151e8af6` printed `BACKFILL_PASS`: all 26,256 v1.1 rows have a non-null, non-unknown panel. Grid/screen rows carry the 710-target W1 label; 60 finalist rows carry W1 seasons and their own smaller buildable panel sizes (632 or 611). Log: `/root/BF_runs/logs/151e8af6119a10c2501b77bd9de6a7929b3e5109/backfill.log`. All **120** new transition-bias and **120** new delta-absolute-bias rows have finite 95% endpoints.

### W1 first-20 transition bias, percentage points [clustered 95% CI]

| Cell | Relegated into L1 | Promoted into L1 | Monitor 54 LL | Monitor 55 LL |
|---|---:|---:|---:|---:|
| grw_lower_joint | −0.309 [−4.339, +4.860] | +4.132 [+2.665, +6.456] | — | — |
| grw_lower_poisson | −1.966 [−9.157, +8.252] | +1.874 [+0.314, +3.465] | — | — |
| grw_spfl_cups_joint | −2.602 [−5.050, +2.055] | +4.584 [+1.856, +8.923] | —¹ | —¹ |
| grw_spfl_cups_poisson | −5.452 [−12.332, +0.233] | +3.189 [+0.987, +6.272] | —¹ | —¹ |
| grw_spfl_joint | −5.084 [−10.625, +0.387] | +5.872 [+4.146, +8.401] | 0.64392 | 0.64603 |
| grw_spfl_poisson | −7.804 [−16.845, −1.316] | +5.222 [+3.366, +7.352] | —¹ | —¹ |
| td_lower_joint | −3.899 [−5.864, −2.151] | +4.797 [+3.562, +6.109] | — | — |
| td_lower_poisson | −5.744 [−8.593, −3.937] | +3.005 [+1.762, +4.248] | — | — |
| td_spfl_cups_joint | −7.541 [−11.829, −1.259] | +5.842 [+4.267, +7.935] | 0.64452 | 0.64277 |
| td_spfl_cups_poisson | −8.675 [−10.935, −6.670] | +4.115 [+2.572, +5.855] | 0.64488 | 0.64990 |
| td_spfl_joint | −8.733 [−14.059, −2.806] | +6.358 [+5.223, +7.797] | 0.64330 | 0.64542 |
| td_spfl_poisson | −9.166 [−11.867, −7.216] | +4.145 [+2.864, +5.426] | 0.64559 | 0.64999 |

¹ The three reused pre-harness GRW artifacts do **not** contain persisted 54/55 posterior latents; choosing a larger scoring datastore cannot manufacture forecasts for those fixtures. Five other pooled cells yield 10 new monitor LogLoss rows (v1.1 had zero). These three cannot meet the literal “every pooled W1 cell has monitor rows” acceptance criterion without regenerating forecasts from their original chains or rerunning fits, neither authorised while shared-grid rules apply.

## Blocked

**Literal all-pooled-monitor requirement remains blocked by the three reused legacy GRW fits** listed with ¹ above. Their historical artifact/prediction coverage, not the new scorecard or `ScottishPyramid` data, is the missing dependency. A follow-up may project their persisted fold chains onto 54/55 fixtures if the needed provenance is recoverable, or rerun the three fits with monitor fixtures in scope after explicit manager approval; no fabricated monitor numbers should be inserted. Phase B code, W1 paired metrics and five pooled monitor cells are otherwise validated. The final all-panel/W2 leaderboard cannot be validated until the manager reruns the W2 grid after the separate engine fix.
