# Claude handover — Phase 3 PASS

**QBT_HANDOVER** at the Phase 3 boundary. Phase 4 (README.md + REPORT.md) has not started.

## Gate and cross-checks

| Check | Result |
|---|---|
| m12 × T−25m × historical_exp07 | 2.515204600600047 / 1127 bets / 99 slates / 608 books: **bit-identical** |
| m12 × T−25m × canonical (Phase 0 `8018dacf`) | 2.773493294562897 / 1132: exact |
| m12 × T−25m × option_b_live (Phase 0 `72cd5496`) | 2.207736087761667 / 1172: exact |
| Snapshot books vs committed Phase 0 books | all 4 equal |
| DB audit, 10 tables, both runner attempts | unchanged |
| Cold-load ScenarioKelly smoke, fresh REPL | 51/51 PASS |

## Grid

- 5 models × 4 snapshots (T−25m/90, T−6h/90, T−24h/90, T−24h/360) × 5 policies = **100/100 COMPUTED**, 0 build errors.
- Nothing is NOT_AVAILABLE. C0's saved latents are draw-level (512 = 128 θ × 4 predictive state draws), so its BakerMcHale and scenario cells are real.
- Every per-draw grid reproduced its book's `p_grid` exactly. Witness = model | run | fold | layout. No day spans two folds for any model.
- Scenario Kelly: 1,435 daily solves, all converged. KKT > 1e-3 occurs only on days where the budget binds (`kkt_residual` tolerance artifact; follow-up).
- Uncertainty: 8-week noncircular moving blocks within season, 999 reps, seed 20261009, 90% intervals. These are not centred on the point estimate when edge weeks differ (wave 2 noted the same). Paired m12 differences have 0 unmatched days.

## Files

- Runner: `l02_phase3_grid.jl` + `r02_phase3_grid.jl` (`4f1d3d43`). Log: `logs/phase3_grid.log`; beast copy `/root/BF_runs/logs/qs_backtest/phase3_grid.log`.
- `results/phase3/phase3_cells.csv` (one row per cell, including growth CI and paired m12 difference), `phase3_markets.csv`, `phase3_daily.csv`, `phase3_bootstrap_growth.csv`, `phase3_paired_vs_m12.csv`, `phase3_scenario_days.csv`, `phase3_bets.csv.gz` (45,191 bets with CLV), `phase3_gate.csv`, `phase3_db_*.csv`.

## Ownership

- Owned beast REPL `jl-claude_qsbt_p3` is closed. Beast checkout `/root/BF_runs/qs_backtest_phase0` is detached at the runner commit. Untracked Phase 2 CSV collision archived (sha-identical) in `/root/BF_runs/qsbt_phase2_native_artifacts_13677357/`.
- Unrelated panes untouched.

## Not yet claimed

No profitability or ranking claim is made here. That belongs to Phase 4, with the in-sample caveats: 2 seasons, about 99 slates, and policies partly chosen on these seasons in experiment 07.
