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

## 2026-10-09 — manager ruling: option 1 authorised

- Claude authorises original-namespace reuse: include wave-1 market_arm.jl at top level as Main.QSMarketArm and have Wave2MarketTable call that module. The provenance guard and saved wave-1 manifest remain exactly unchanged. No independent inversion or guard bypass.
- Add exact value-level parity: all table match IDs match the prepared wave-1 rates; full-book flags match membership in its accepted full-book panel; covered home/away logs equal panel.obs_y exactly, uncovered rows carry zero dummies. Verify target counts 710/595/78/517 and re-check parity after frozen CSV round-trip. Record its SHA256 here and in PROGRESS.md once generated.
- Resume Phase 1 with unchanged priors, seeds, truths and thresholds; preserve the earlier failed attempt. Use new attempt name prepare_v2, never overwrite its evidence.
- Authorised preparation PASS at 4a71c02e (36.2634 s include wall): table 1,430 rows / 1,107 full books over four seasons; exact value parity, including CSV round-trip. Target counts exactly 710/595/78/517. Frozen table SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`, committed as results/market_rates.csv + market_rates.sha256. Saved wave-1 manifest SHA256 unchanged before/after: `57941606a195010fe86cbfda0564d61f456379d05d0e8988dce3bb2edd600fb1`. No reference refit or guard/manifest edit.

## 2026-10-09 — stop on component gate at ec1a4137

- Fresh MarketRateObservation entry point FAIL, 5533/5905 passed, 372 failed. One standalone likelihood assertion, 370 central finite-difference assertions (GRW 65, QS 305), one frozen recovery miss (GRW dyn.β.σₖ). Compiled/fresh ReverseDiff and ForwardDiff comparisons pass but do not supersede the failed hard checks. The unconditional MARKET_AD PASS prints are not evidence of passing AD.
- Julia Test runs all child sets before throwing; recovery fits therefore executed within this failed entry point. All 14 intervals preserved in results/synthetic_recovery.csv; 13 cover. Do not tune truths/seeds, widen intervals or relax tolerance. No second attempt or fix.
- Launcher stopped before allocation/builder/harness gates. Phase 1 not accepted; no smoke/grid/scoring. Table and saved manifest SHA256 rechecked unchanged. Close only owned idle sessions after preserving evidence; await manager review (PHASE1_BLOCKED.md).

## 2026-10-09 — manager amendments to Phase 1

1. Likelihood: diagnose before fixing. Report DynamicPPL prior and likelihood separately, linked/unlinked densities and bijector Jacobian, and the independent expected Normal density. Fix the wrong side only after this evidence. Acceptance remains absolute 1e-12, all Normal constants included.
2. AD: tighten compiled ReverseDiff / fresh ReverseDiff / ForwardDiff comparisons to relative 1e-10. Replace the former single-step FD gate with convergence evidence on every previously failing coordinate plus a fixed random sample of 50 remaining coordinates per model. Relative h=1e-3,1e-4,1e-5,1e-6 times max(1,abs(x)); show second-order decrease until rounding floor. Richardson from h=1e-4 and 5e-5 must agree with AD to relative 1e-8. Manager rationale: independent exact AD paths agree; market curvature differs from Poisson. No other thresholds changed.
3. Recovery: retain the first seed's 14 intervals and run two further predeclared seeds per arm, same truths and smoke budget. Gate: no parameter misses all three seeds; pooled coverage >=80%. Manager notes 0.9^14≈0.23, so demanding simultaneous first-seed coverage was not calibrated. No seed tuning; report all 42 intervals.
- Additional seeds frozen before execution: GRW data/sampler (20261024,20261026), (20261034,20261036); QS (20261025,20261027), (20261035,20261037). Initial seeds and first-seed CSV untouched. Fixed FD sample seed 20261020 (GRW) / 20261021 (QS); use the union of failed coordinates across the original three perturbed points, plus 50 others, and audit them at those same points.
- Resume only diagnosis now, then amended component gates, allocation/builder/harness, candidates and smoke. Ask manager before grid. Prior failure remains recorded; use new attempt names.

## 2026-10-09 — likelihood diagnosis BEFORE correction (bd924940)

- Separate DynamicPPL components at sigma=.12: returned-only _observe prior=1.2036465597893726, likelihood=0, unlinked joint=1.2036465597893726. It returns the scalar likelihood but intentionally does not accumulate it (engine.jl:130–134 contract).
- Production seam uses ll ~ to_submodel(_observe(...), false); @addlogprob! ll (engine.jl:534–538). This has prior=1.2036465597893726, likelihood=2.5623083423195876, unlinked joint=3.7659549021089602. Independent weighted Normal formula, original two-logpdf expected value and pure helper all equal 2.5623083423195876 exactly, constants included.
- Linked coordinate log(.12)=-2.120263536200091. The production seam's linked internal density=1.6456913659088697; its model-space joint=3.7659549021089607. Difference is log(sigma), the inverse-bijector Jacobian. Likelihood differs only by 4.4e-16 between coordinate spaces. Separate components in results/likelihood_audit.csv.
- Diagnosis: the failing test's expected value was correct, but it compared a returned-only submodel's density to the complete production density. Correct the test by mirroring the production accumulation seam; do not alter production source or the 1e-12 acceptance rule.
- FD implementation declared before rerun: replay the original Float64 single-step check only to recover the failed-coordinate set (must reproduce counts 65/305). Record Float64 step curves too. Evaluate the convergence/Richardson audit with 128-bit BigFloat arithmetic and an independent high-precision ForwardDiff reference to avoid subtraction/AD-reference rounding floors. Compare high-precision ForwardDiff with the Float64 exact paths at relative 1e-10. Report both precisions, all selected coordinates, all four steps, observed order/floor, Richardson error at 1e-8. No step selection, dropped coordinates or seed changes based on the new results. Second-order order band 1.5–2.5 applies only while errors exceed a roundoff estimate; otherwise label the floor explicitly.
