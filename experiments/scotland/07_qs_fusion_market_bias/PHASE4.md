# Phase 4: scoring, leakage audit and record (PASS)

This phase used saved fits only: no sampling, no reference refits, and no score writes. The plan was frozen in DECISIONS_LOG before any score was computed. Each step ran on the beast in a fresh 16-thread REPL, with cores pinned and BLAS set to 1 thread.

| step | commit | wall | result |
|---|---|---|---|
| `phase4_gate.sh a1 score leakage` | `4f7ee7bc` | 03:57–04:02 UTC | score PASS (150 s), leakage PASS |
| `phase4_gate.sh a1_record record` | `9cf7d0de` | 04:03–04:04 UTC | record PASS |

## Score step (`r06_score.jl`)

**Inputs.** The three wave-3 grids, taken from `results/phase3/RUNS.csv`, which equals the receipts. Each fit was checked against its candidate's model and sampler and has 40 folds. The two wave-2 references were loaded by UUID.

**Reference parity** (`results/phase4/reference_parity.csv`): 10/10 checks pass.
- qs_marketobs and qs_joint harness rows: 598 each, max difference 0.
- Observation frames for qs_marketobs, qs_joint and market_close: 2899 rows each, difference 0.
- Goal fixture scores: 710 each, difference 0.
- Common-panel means: ≤ 2.2e-16, within the 1e-14 tolerance.

**Common 1X2 panel.** 595 fixtures / 1785 selections, with identical keys for all six arms. The harness's own RPS population is 596 fixtures for the models and 1528 for the close, so it is recorded but not used (`score_panel_audit.csv`).

**Intervals** (`paired_intervals.csv`): 60 rows, 8 pairs × 6 market metrics plus 4 non-close pairs × 3 goal channels.
- Each row carries a noncircular 8-week 90% interval (the classification), a circular 90% interval and a clustered 95% interval.
- Clustered and block point estimates are equal (≤ 1e-12) on every row.
- 6 rows have a circular class that disagrees with the noncircular one. None is in the primary pair.
- Centring drift is at most 0.00564 (noncircular) and 0.00063 (circular).

**Posteriors:**
- `posterior_bias_by_fold.csv`: 3 arms × 40 folds; ν, log κ, σ_obs, plus κ_D, δ_D, δ_M on the bias arms.
- `posterior_level_joint_by_fold.csv`: 80 rows, giving the δ_M–log κ correlation and the sum δ_M + log κ.
- `posterior_r_by_fold.csv`.
- `posterior_bias_by_fold.svg`.

## Leakage step (`r10_leakage_audit.jl`)

The log line reads: `LEAKAGE_AUDIT_PASS folds=120 target_overlap=0 at_or_after_cutoff=0 own_close_provenance=PASS common_fixtures=595 saved_predictions_exact=PASS`.
- 1430 provenance rows (1107 full books), with 0 cross-match rows and 0 post-kickoff ticks.
- Repricing the saved predictions reproduces them exactly. LogLoss gaps are ≤ 6.7e-16.

## Record step (`r08_record.jl`)

- Phase 3 = Phase 4 score parity is exact for all three arms: 664 rows / 9960 cells each, max difference 0 (`phase3_phase4_score_parity.csv`).
- Register row `scottish_lower_qs_wave3_2426` was written with status `scored` and the reviewed decision.
- `results/phase4/leaderboard.csv` was written, and the canonical boards were regenerated. Additions only: 3 rows in `experiments/LEADERBOARD.csv`, 18 lines in `LEADERBOARD.md` and 1 register row. Existing rows are unchanged.
- The beast checkout was restored to clean afterwards.

## Outcome

There is no promotion; see REPORT.md. Flags carried forward:
- control fold-40 tail ESS 330.7, whose bias posteriors are typical;
- qs_nobias recovery 21/27 (Phase 1);
- the circular contradictions listed in REPORT.md.

Evidence:
- `results/phase4/`;
- `results/leakage_*.csv`;
- `results/gate_logs/phase4_launcher_a1.log`;
- `results/gate_logs/phase4_launcher_a1_record.log`.

Next is Phase 5: refit one fold per arm from the frozen seed in a fresh REPL.
