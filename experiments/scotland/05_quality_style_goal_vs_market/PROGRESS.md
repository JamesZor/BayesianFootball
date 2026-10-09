# Quality/style goal vs market — progress

Updated: 2026-10-09 UTC. Owner: pi. Branch: `exp/qs-goal-vs-market`.

## Current phase

Phase 3 — **IN PROGRESS** (smokes). Phase 2 PASS at `ef8d0d8e`: B2 counts 710/595/78/517 reproduced; 40 folds leak-free, market training 509–824 fixtures, 710 held-out (517 quoted); adapter vs harness max |Δ| **0.0** over 664 rows (log `/root/BF_runs/logs/qs_experiment/phase2_checks.log`, outputs `/root/BF_runs/qs_experiment_out/`). Phase 1 accepted: fresh-REPL strengthened QS test **373/373** at `69d18589` (45s), plus zero-allocation gradient 0 B (QS 2/2), MultiScaleGRW, builder and harness all passed on the new component. No smoke or grid has started. Manager accepted Phase 0 with excluded C2-pending 29/39. Beast isolated detached checkout at `a4729e8e`; `.env` and `Manifest.toml` linked, pinned snapshot installed with SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` and mtime `2026-09-25 12:57:15.480765468 UTC`.

## Done

- On checkout `903ed008`, `test/test_quality_style_grw.jl` passed 98/98; tape allocation matrix 150/150 plus QS zero-allocation **2/2 at 0 B**; baseline `test_multiscale_grw.jl` 124/124, builder suite all testsets, harness 179/179. Strengthened test at `69d18589` checks **every θ coordinate** at three random perturbed points on a real 40-fold input (ForwardDiff, compiled/fresh ReverseDiff, central finite differences ≤ 1e-6) and independently recomputes OOS σ_q,K and σ_s,K; fresh-REPL **373/373**. Logs: `/root/BF_runs/logs/qs_experiment/phase1_quality_style.log` and `phase1_ad_full.log`.
- On isolated beast checkout `a4729e8e`, each prescribed entry point ran in a fresh `pi_qsx_*` Julia REPL. `test_multiscale_grw.jl` 124/124; tape allocation 150/150; builder and harness passed; market t01 269/269, t02 131/131, t03 131/131, t04 92/92, t05 fast 17/17, reports 22/22, workflow 85/85, preflight 11/11, pooled full-book deterministic 305/305; t06 passed. Logs: `/root/BF_runs/logs/qs_experiment/{phase0_repl,tape,builder,harness,t01,t02,t03,t04,t05_fast,t05_reports,t05_workflow,t05_preflight,t05_pooled,t06}.log`.
- Manager's explicit Phase 0 ruling: `t05_pooled_tests.jl` includes `t05_c2_pending_tests.jl` at line 145, a labelled, deliberately reported **non-acceptance** set. Its 29 passes / 10 known failures (one integrated derivative, nine archived thin-book Gate 1) are expected on every branch. C1 and C2 are parked by the human. Full-book 305/305 and all other required assertions passed; Phase 0 passes. No C2 tests changed.
- Created and claimed TODO 040. Verified local branch was clean before task edits, beast reachable, existing beast repository belongs to another workstream; created isolated detached checkout under `/root/BF_runs/qs_experiment`.
- Source snapshot `/root/BF_runs/market_model_b/.cache/datastore_ScottishLower.jls`: SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` (published snapshot).

## Next action

Phase 3: goal smokes (`r03_goal_smoke.jl`, 4 arms) then market smoke on folds 1, 21, 40 (`r03_market_smoke.jl`), sequentially in `pi_qsx_phase2`. Then summarise wall time, ETA, R̂, ESS, divergences and ask the manager before any grid.

## Open beast sessions

`pi_qsx_phase2` pane `%337` (warm REPL; log `/root/BF_runs/logs/qs_experiment/phase2_checks.log`). No other sessions touched.
