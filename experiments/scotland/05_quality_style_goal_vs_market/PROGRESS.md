# Quality/style goal vs market — progress

Updated: 2026-10-09 UTC. Owner: pi. Branch: `exp/qs-goal-vs-market`.

## Current phase

Phase 4 — **GRID RUNNING at `e6685ceb`** (manager approved 2026-10-09 with order control_grw → qs_market_r → qs_weak_r → control_td, market concurrent, 20-min polls).

| Arm | Status | Started (UTC) | Hours | run_id |
|---|---|---|---|---|
| control_grw | running | 13:09 | | |
| qs_market_r | queued | | | |
| qs_weak_r | queued | | | |
| control_td | queued | | | |
| market (C0) | running, fold 1/40 done (234 s, identical to smoke) | 13:10 | | n/a (files) |

Phase 3 — smokes DONE at `2060056f`.

### Smoke summary (`logs/phase3_smoke.log`)

Goal arms: harness smoke, 2 folds × 2 chains × (200+200); all **6/6 hard checks pass** (gradient, tape 0 B, filtration, latents, score grid, fit parity). Convergence is a review flag at smoke budget.

| Arm | wall s | max R̂ | min ESS bulk/tail | div | ms/leapfrog | smoke run_id |
|---|---|---|---|---|---|---|
| control_grw | 115.6 | 1.048 | 69 / 98 | 0 | 1.24 | `3575d788` |
| control_td | 20.1 | 1.029 | 121 / 138 | 0 | 0.28 | `e66bcbf4` |
| qs_market_r | 87.6 | 1.062 (review) | 106 / 52 | 0 | 1.23 | `0a060fb6` |
| qs_weak_r | 55.6 | 1.077 (review) | 66 / 37 | 0 | 1.34 | `e2f48631` |

All four flag `sampling_performance` review (GC share ~31% on fold 40, as for the control; not QS-specific). Market arm, full C0 budget 4×(2000+3000): folds 1/21/40 R̂ 1.0012/1.0004/1.0005, ESS bulk ≥ 6230, gates pass; fit 81.5/70.3/103.8 s, forecast <1 s.

ETA (W2 sampler 4×(500+1000), 40 folds, per prior W1/W2 experience: TD cell ~10 min, lower GRW cell ~90–150 min): control_td ~0.2 h; control_grw, qs_market_r, qs_weak_r ~1.5–2.5 h each; market arm 40 × ~86 s ≈ 1 h sequential. Total ≈ 6–9 h on the beast, one grid at a time.

Phase 2 PASS at `ef8d0d8e`: B2 counts 710/595/78/517 reproduced; 40 folds leak-free, market training 509–824 fixtures, 710 held-out (517 quoted); adapter vs harness max |Δ| **0.0** over 664 rows (log `/root/BF_runs/logs/qs_experiment/phase2_checks.log`, outputs `/root/BF_runs/qs_experiment_out/`). Phase 1 accepted: fresh-REPL strengthened QS test **373/373** at `69d18589` (45s), plus zero-allocation gradient 0 B (QS 2/2), MultiScaleGRW, builder and harness all passed on the new component. No smoke or grid has started. Manager accepted Phase 0 with excluded C2-pending 29/39. Beast isolated detached checkout at `a4729e8e`; `.env` and `Manifest.toml` linked, pinned snapshot installed with SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` and mtime `2026-09-25 12:57:15.480765468 UTC`.

## Done

- On checkout `903ed008`, `test/test_quality_style_grw.jl` passed 98/98; tape allocation matrix 150/150 plus QS zero-allocation **2/2 at 0 B**; baseline `test_multiscale_grw.jl` 124/124, builder suite all testsets, harness 179/179. Strengthened test at `69d18589` checks **every θ coordinate** at three random perturbed points on a real 40-fold input (ForwardDiff, compiled/fresh ReverseDiff, central finite differences ≤ 1e-6) and independently recomputes OOS σ_q,K and σ_s,K; fresh-REPL **373/373**. Logs: `/root/BF_runs/logs/qs_experiment/phase1_quality_style.log` and `phase1_ad_full.log`.
- On isolated beast checkout `a4729e8e`, each prescribed entry point ran in a fresh `pi_qsx_*` Julia REPL. `test_multiscale_grw.jl` 124/124; tape allocation 150/150; builder and harness passed; market t01 269/269, t02 131/131, t03 131/131, t04 92/92, t05 fast 17/17, reports 22/22, workflow 85/85, preflight 11/11, pooled full-book deterministic 305/305; t06 passed. Logs: `/root/BF_runs/logs/qs_experiment/{phase0_repl,tape,builder,harness,t01,t02,t03,t04,t05_fast,t05_reports,t05_workflow,t05_preflight,t05_pooled,t06}.log`.
- Manager's explicit Phase 0 ruling: `t05_pooled_tests.jl` includes `t05_c2_pending_tests.jl` at line 145, a labelled, deliberately reported **non-acceptance** set. Its 29 passes / 10 known failures (one integrated derivative, nine archived thin-book Gate 1) are expected on every branch. C1 and C2 are parked by the human. Full-book 305/305 and all other required assertions passed; Phase 0 passes. No C2 tests changed.
- Created and claimed TODO 040. Verified local branch was clean before task edits, beast reachable, existing beast repository belongs to another workstream; created isolated detached checkout under `/root/BF_runs/qs_experiment`.
- Source snapshot `/root/BF_runs/market_model_b/.cache/datastore_ScottishLower.jls`: SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` (published snapshot).

## Next action

Poll every 20 min. Per arm: record hours, run_id, R̂/divergences; refit any goal fold with R̂ > 1.05 once, seeded. Watch goal-arm speed; pause the market session if it slows the goal arm > ~20%. Then Phase 5 scoring.

## Open beast sessions

- `pi_qsx_goal_grid` pane `%338`: `-t 14`, cores 0–13, `r04_goal_grid.jl`, log `/root/BF_runs/logs/qs_experiment/phase4_goal_grid.log`.
- `pi_qsx_market_grid` pane `%339`: `-t 2`, cores 14–15, `r04_market_grid.jl`, log `phase4_market_grid.log`, outputs `/root/BF_runs/qs_experiment_out/market_grid/`.
- `pi_qsx_phase2` closed. No other sessions touched.
