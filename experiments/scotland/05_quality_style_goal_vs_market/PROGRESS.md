# Quality/style goal vs market — progress

Updated: 2026-10-09 UTC. Owner: pi. Branch: `exp/qs-goal-vs-market`.

## Current phase

Phase 0 — **BLOCKED** at the specified stop gate; no component implementation, smoke, or grid has started. Beast isolated detached checkout at `a4729e8e`; `.env` and `Manifest.toml` linked, pinned snapshot installed with SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` and mtime `2026-09-25 12:57:15.480765468 UTC`.

## Done

- On isolated beast checkout `a4729e8e`, each prescribed entry point ran in a fresh `pi_qsx_*` Julia REPL. `test_multiscale_grw.jl` 124/124; tape allocation 150/150; builder and harness passed; market t01 269/269, t02 131/131, t03 131/131, t04 92/92, t05 fast 17/17, reports 22/22, workflow 85/85, preflight 11/11, pooled full-book deterministic 305/305; t06 passed. Logs: `/root/BF_runs/logs/qs_experiment/{phase0_repl,tape,builder,harness,t01,t02,t03,t04,t05_fast,t05_reports,t05_workflow,t05_preflight,t05_pooled,t06}.log`.
- **Failure:** `t05_pooled_tests.jl` includes `t05_c2_pending_tests.jl` at line 145. Its own `@testset` has 29 passes, **10 failures** (one integrated derivative, nine archived thin-book Gate 1); it deliberately catches `TestSetException` and prints `C2_PENDING_REPORTED: thin-book failures retained; excluded from C4 acceptance`. The original driver emitted a misleading PASS marker because the exception was swallowed. The driver is now corrected to flag this marker as failure. The work package says to stop if anything fails on the merged base; thus Phase 1 is not started.
- Created and claimed TODO 040. Verified local branch was clean before task edits, beast reachable, existing beast repository belongs to another workstream; created isolated detached checkout under `/root/BF_runs/qs_experiment`.
- Source snapshot `/root/BF_runs/market_model_b/.cache/datastore_ScottishLower.jls`: SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` (published snapshot).

## Next action

Manager decision required: either explicitly exclude the pre-existing C2-pending checks from the Phase 0 acceptance gate (they are intentionally excluded from C4 in the source), or fix them outside this package. Do not start Phase 1 before that decision.

## Open beast sessions

None; `pi_qsx_gate0` `%319` closed, `pi_qsx_gate_driver` `%320` exited, all child sessions exited. No other sessions touched.
