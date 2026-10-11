# 26/27 paper trade progress (TODO 045, Claude CLI)

Brief: `experiments/claude_qs_2627_paper_brief.md`. Sentinels: P27_HANDOVER / P27_BLOCKED / P27_DONE.

## Phase 0: setup, data audit, gates, plan (DONE; awaiting manager approval of the sampling jobs)
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

## Phase 1: walk-forward predictions (NOT STARTED)

## Beast
- Checkout: `/root/BF_runs/qs_2627`. Logs: `/root/BF_runs/logs/qs_2627/`. Outputs: `/root/BF_runs/qs_2627/out/`
  (`datastore_ScottishLower_2627.jls`, `orderbook_2627.jls`).
- REPL: `jl` session `claude_p27` (tmux `jl-claude_p27`).
