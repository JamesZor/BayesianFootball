# Published Scottish regression gate — PASS

2026-10-07, beast owned REPLs `%272` (B2) and `%273` (C8), branch source `205cf4e2f6ec519ac31b9a53ead79358861a8bd8`, 16 Julia threads, BLAS=1. Pinned **published** ScottishLower snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` was used for this gate, **not** the later fresh QS snapshot. C1 was not run. The full existing `r03_covariance_ladder.jl` reached `R03_DONE`, all eight R2/R5/R6/R7 protocol fits passed convergence, and the existing `r05_fast_rungs.jl` reached `R05_FAST_RUNGS_DONE`, all six C0/H1/H2 protocol fits passed convergence. No prior, seed, budget, price window, or population was changed.

Compared with the tracked reference using `cmp -s` on the beast (no rounding):

| Reference | Byte-identical rerun CSVs |
|---|---|
| `results/B2/` | `ladder_summary_b2.csv`, `onestep_metrics_b2.csv`, `onestep_fixture_b2.csv`, `rotation_quality_style.csv`, `paired_vs_r2.csv` |
| `results/C/v8_fast_1/` | `parameter_posteriors_c.csv`, `onestep_metrics_c.csv`, `onestep_fixture_c.csv`, `paired_c0_vs_r6.csv` |

The C8 rerun used a fresh output directory outside the published result tree (`/root/BF_runs/market_model_qs/results_qs_regression_c0`); B2 reran in its beast worktree's tracked result directory and the five checked scientific CSVs remained byte-identical. Source/logs: `/root/BF_runs/logs/market_model_qs/regression.log` and `regression_c8.log`. Other B2/C8 outputs are not claimed byte-identical by this narrow gate; runtime and SHA-bearing convergence metadata are excluded.
