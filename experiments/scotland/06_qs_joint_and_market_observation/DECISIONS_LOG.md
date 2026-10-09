# Wave 2 — decisions

## 2026-10-09 — scope and Phase 0

- Continue `exp/qs-goal-vs-market`; implement directly. No changes beyond the work package, no ROI or staking, no refit of wave-1 references.
- All Julia runs in fresh owned beast REPLs (`pi_qsx2_*`), 16 threads, pinned cores, BLAS 1; tests run sequentially to avoid contention. Use `root@mcmc-beast` (bare ssh selects nonexistent remote james).
- Reuse wave-1 Phase 0 entry points, adding explicit QS and MultiScaleGRW tests. Only the labelled C2-pending child set (29/39, 10 known failures) is excluded by manager ruling; no other failure is excluded.
- Coverage uses the actual MatchProxyXGFeature(k=25, fallback=:none) extractor, with the fold's history IDs for its shot-cell fit, not merely nonmissing raw columns. Report training and held-out coverage by tournament for all 40 folds. Held-out coverage audit is descriptive only; no held-out measurements enter fitting.
- Snapshot pin is unchanged: SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC`.

## 2026-10-09 — Phase 1 frozen implementation choices (before any new fit)

- Derivation: `l05_pooled.jl:c0_parameters` maps σ_obs into log coordinates. The actual C0 prior is `MID.log_prior(::FullBookRung, theta)` in included `l05_fullbook_engine.jl:30`: normalised HalfNormal(0.20), plus the log-coordinate Jacobian. Use `truncated(Normal(0,0.20),0,Inf)` on σ_obs; DynamicPPL supplies the linked-space Jacobian. Do not infer a prior from the C0 initial value 0.07.
- Observe full-book log rates only, with no goals likelihood. Precompute binary mask × dynamics weights; reduce weighted quadratic residuals before sampled scalar arithmetic, to avoid fused-broadcast allocations. Outcome grid and latent rates follow Poisson unchanged; σ_obs is fit-time measurement noise, not predictive goal dispersion.
- Feature accepts a table and aligns by match ID. Refuse duplicates and nonfinite covered rows; uncovered rows get zero dummies. The experiment creates a sorted frozen CSV from the unchanged wave-1 inversion panel, checks its SHA256 before injection, and retains the snapshot pin. src contains no prototype imports.
- AD and synthetic recovery use real fold 40 (not fold 1, whose absent micro states cannot test rₖ). AD checks every coordinate at three perturbed points in linked space, compiled/fresh ReverseDiff, ForwardDiff and central finite differences, unchanged 1e-6 threshold.
- Synthetic recovery is predeclared at smoke budget 2×(200 warmup+200 retained), accept .65, depth 10. Data seeds 20261014/20261015 and sampler seeds 20261016/20261017 for GRW/QS. σ_obs=.07; GRW attack scales [.12,.06,.03], defence [.20,.11,.024]; QS τ=[.16,.085,.027], r=[.4,.4,.3]. Preserve real full-book mask; all 14 scale/ratio intervals reported to CSV, including any miss. No seed or truth tuning after a result.

## 2026-10-09 — stop on Phase 1 preparation failure

- At source 351b8ddd the unchanged wave-1 provenance guard rejects the saved panel (`inversion provenance differs`). Its manifest includes `Main.QSMarketArm.MarketModel.MarketModelConfig`; the new loader nests the module under `Main.Wave2MarketTable`. Retain this failure, do not change the guard or reference manifest, and stop for manager review. No second attempt, synthetic fit or later phase.
- Preserve unverified implementation for review rather than claiming tests passed. Snapshot and reference rows untouched; all owned idle panes closed. Proposed original-namespace reuse or independent unchanged panel preparation requires manager authorisation.
