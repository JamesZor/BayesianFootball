# Wave 2 — decisions

## 2026-10-09 — scope and Phase 0

- Continue `exp/qs-goal-vs-market`; implement directly. No changes beyond the work package, no ROI or staking, no refit of wave-1 references.
- All Julia runs in fresh owned beast REPLs (`pi_qsx2_*`), 16 threads, pinned cores, BLAS 1; tests run sequentially to avoid contention. Use `root@mcmc-beast` (bare ssh selects nonexistent remote james).
- Reuse wave-1 Phase 0 entry points, adding explicit QS and MultiScaleGRW tests. Only the labelled C2-pending child set (29/39, 10 known failures) is excluded by manager ruling; no other failure is excluded.
- Coverage uses the actual MatchProxyXGFeature(k=25, fallback=:none) extractor, with the fold's history IDs for its shot-cell fit, not merely nonmissing raw columns. Report training and held-out coverage by tournament for all 40 folds. Held-out coverage audit is descriptive only; no held-out measurements enter fitting.
- Snapshot pin is unchanged: SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.
