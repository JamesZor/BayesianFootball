# 26/27 paper trade progress (TODO 045, Claude CLI)

Brief: `experiments/claude_qs_2627_paper_brief.md`. Sentinels: P27_HANDOVER / P27_BLOCKED / P27_DONE.

## Phase 0: setup, data audit, gates, plan (DONE; accepted by the manager)
- [x] Folder, TODO 045 (README row), DECISIONS_LOG (rulings 1–9 + D1–D13), PROGRESS.
- [x] Beast checkout `/root/BF_runs/qs_2627` (detached worktree of `/root/BayesianFootball`; Manifest symlinked,
  pinned cache and `.env` copied from `qs_wave5`), `jl` REPL `claude_p27`.
- [x] Runner `l00_p27.jl` + `r00_phase0.jl` (head `aace7998`), log `logs/phase0.log`, outputs `results/phase0/`.
  Static check PASS; DB audit unchanged (10 tables).
- [x] DataStore rebuilt read-only (`f407a1b4…`, 2,049 matches, last 2026-09-19); 79/79 fixtures played.
- [x] Order book pinned (`2adb1271…`). Coverage by match day: **79 / 70 / 60 at T−25m** (napkin §13.1
  reproduced); 57 with 1X2 + OU2.5, 49 with all canonical markets, 50 with 1X2 LTP (none on 08-01).
- [x] Odds frames back / mid / LTP (744 / 744 / 418 rows) and the mid TWA close (60 fixtures).
- [x] Gates: inversion PASS (max |Δ log λ| = 0.0, 1,430 rows); engine PASS bit-identical (3 cells);
  qs positions 1–40 equal 40/40, filtration PASS (41–44); m12 inventory: **DB folds miss 09-05** → in-memory
  extension planned (positions 1–40 equal, filtration PASS).
- [x] Sampling plan in the P27_HANDOVER message (qs 4 folds; m12 4 folds needs manager go).

## Phase 1: walk-forward predictions (DONE; awaiting manager review)
- [x] Loader `l01_sampling.jl`, `l02_predictions.jl`; runners `r01_extend_qs.jl`, `r02_extend_m12.jl` (head `21b75564`),
  `r03_phase1_predictions.jl` (head `1e0f4ca5`). Logs `logs/extend_qs.log`, `logs/extend_m12.log`,
  `logs/phase1_predictions.log`; outputs `results/phase1/`. DB audit unchanged around all three.
- [x] qs extension (positions 41–44, 1.5 min): R̂ ≤ 1.0115, tail ESS ≥ 768, 0 divergences; no rerun.
- [x] m12 extension (positions 41–44, 0.5 min): R̂ ≤ 1.0084, tail ESS ≥ 873, 0 divergences; no rerun.
  Played XI from `ds.lineups` (same builder rule as 1–40; D17). DB-fold check on 69: mean |Δp| 0.0004.
- [x] Sampled-fold filtration PASS (8 folds). Per-fixture probabilities (79 × 2) with fold and cut-off.
- [x] 1X2 log loss vs the mid close (60 fixtures): qs − close +0.0028 [−0.012, +0.017]; m12 − close +0.034
  [+0.008, +0.062]; transition vs other split in `phase1_logloss.csv`.

## Beast
- Checkout: `/root/BF_runs/qs_2627`. Logs: `/root/BF_runs/logs/qs_2627/`. Outputs: `/root/BF_runs/qs_2627/out/`
  (`datastore_ScottishLower_2627.jls`, `orderbook_2627.jls`).
- REPL: `jl` session `claude_p27` (tmux `jl-claude_p27`).
