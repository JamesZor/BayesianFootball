# Quality/style goal vs market — progress

Updated: 2026-10-09 UTC. Owner: pi. Branch: `exp/qs-goal-vs-market`.

## Current phase

Phase 1 — **IN PROGRESS**. Manager accepted Phase 0 with the explicitly excluded C2-pending child testset (29/39); every acceptance assertion passed. No smoke or grid has started. Beast isolated detached checkout at `a4729e8e`; `.env` and `Manifest.toml` linked, pinned snapshot installed with SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` and mtime `2026-09-25 12:57:15.480765468 UTC`.

## Done

- On isolated beast checkout `a4729e8e`, each prescribed entry point ran in a fresh `pi_qsx_*` Julia REPL. `test_multiscale_grw.jl` 124/124; tape allocation 150/150; builder and harness passed; market t01 269/269, t02 131/131, t03 131/131, t04 92/92, t05 fast 17/17, reports 22/22, workflow 85/85, preflight 11/11, pooled full-book deterministic 305/305; t06 passed. Logs: `/root/BF_runs/logs/qs_experiment/{phase0_repl,tape,builder,harness,t01,t02,t03,t04,t05_fast,t05_reports,t05_workflow,t05_preflight,t05_pooled,t06}.log`.
- Manager's explicit Phase 0 ruling: `t05_pooled_tests.jl` includes `t05_c2_pending_tests.jl` at line 145, a labelled, deliberately reported **non-acceptance** set. Its 29 passes / 10 known failures (one integrated derivative, nine archived thin-book Gate 1) are expected on every branch. C1 and C2 are parked by the human. Full-book 305/305 and all other required assertions passed; Phase 0 passes. No C2 tests changed.
- Created and claimed TODO 040. Verified local branch was clean before task edits, beast reachable, existing beast repository belongs to another workstream; created isolated detached checkout under `/root/BF_runs/qs_experiment`.
- Source snapshot `/root/BF_runs/market_model_b/.cache/datastore_ScottishLower.jls`: SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` (published snapshot).

## Next action

Implement `QualityStyleGRW` with identical state geometry to `MultiScaleGRW`, builder seams, posterior q/s reconstruction and OOS micro-step forecast; prove algebra, geometry, AD and zero-allocation in new tests. Run all four Phase 0 source tests plus the new file on the beast and stop on any acceptance failure.

## Open beast sessions

None; `pi_qsx_gate0` `%319` closed, `pi_qsx_gate_driver` `%320` exited, all child sessions exited. No other sessions touched.
