# Quality/style goal vs market — progress

Updated: 2026-10-09 UTC. Owner: pi. Branch: `exp/qs-goal-vs-market`.

## Current phase

Phase 6 — **reproduction running** (`r06_reproduce.jl`, pane `%341`). Phase 5 DONE: scoring at `81c77a42`+, report in `REPORT.md`, card in `README.md`, tables in `results/`. Goal-arm grid scores are written to `harness_scores`; the market arm's rows are not (file-based run). The shared leaderboard is not regenerated.

Phase 4 — DONE (was: **GRID RUNNING at `e6685ceb`**) (manager approved 2026-10-09 with order control_grw → qs_market_r → qs_weak_r → control_td, market concurrent, 20-min polls).

| Arm | Status | Started (UTC) | Hours | run_id |
|---|---|---|---|---|
| control_grw | **done**, hard PASS; max R̂ 1.0117, min ESS bulk 650, 6 div (3.75e-5); 1X2 LL 0.61678 | 13:09 | 0.26 | `a036d22a-ff32-404c-b801-5f928d8a89f4` |
| qs_market_r | **done**, hard PASS; max R̂ 1.0100, ESS 951, 2 div (1.25e-5); 1X2 LL 0.61837 | 13:25 | 0.27 | `b18ae74b-9bc1-4cfa-b363-a640131adb2d` |
| qs_weak_r | **done**, hard PASS; max R̂ 1.0092, ESS 820, 0 div; 1X2 LL 0.61785 | 13:41 | 0.26 | `21f2a9f9-b96f-4034-97de-767704a9d54a` |
| control_td | **done**, hard PASS; max R̂ 1.0072, ESS 1079, 13 div (8.1e-5); 1X2 LL 0.62046 | 13:57 | 0.03 | `1dccb320-526c-4700-9258-134788a636ef` |
| market (C0) | folds 1–16 done on 2 threads, all gates pass; goal grids finished, so restarted at 14:3x on 16 threads (session `%340`), resuming from fold 17 (fold 17 was in flight, empty dir, refitted from scratch with the same seeds) | 13:10 | | n/a (files) |

All goal folds R̂ ≤ 1.0117 (max over parameters, all folds) → **no R̂ > 1.05 reruns needed**. Divergences ≤ 8.1e-5 ≪ 0.1%. Goal grids ran ~0.26 h per GRW arm (the ETA was based on older, slower runs). The 1X2 LogLoss above is the harness `target`/`1X2` score printed by the grid runner, not a paired comparison; the paired intervals come in Phase 5.

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

Wait for `REPRODUCTION_DONE` in `/root/BF_runs/logs/qs_experiment/phase6_repro.log`. Write `REPRODUCIBILITY.md`, close TODO 040 (or mark the leaderboard step for the manager), tidy beast sessions, then print the sentinel.

## Open beast sessions

- `pi_qsx_repro` pane `%341`: fresh `-t 16` REPL, `r06_reproduce.jl`, log `/root/BF_runs/logs/qs_experiment/phase6_repro.log`, outputs `/root/BF_runs/qs_experiment_out/repro/`.
- All earlier `pi_qsx_*` sessions closed.
- `pi_qsx_phase2` closed. No other sessions touched.
