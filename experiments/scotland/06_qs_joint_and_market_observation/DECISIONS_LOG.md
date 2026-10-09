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

## 2026-10-09 — Phase 1 acceptance under manager amendments

- Fresh entry points at 153239a7 PASS: component 3787/3787; legacy tapes 150/150, QS 2/2, both market-observation dynamics 8/8 (0 B); builder 106/106; harness 179/179. Complete 16:47:05 UTC. No production source correction required: only the standalone test's accumulation seam was wrong.
- Exact AD errors <=4.2e-16 at tightened 1e-10. Historical FD totals 65/305 reproduced; failed-coordinate unions 33/112 plus 50 others per model all tested at 3 points. Resolved high-precision orders ~2; all 249/486 Richardson checks pass (maximum 4.1924e-16/4.2649e-18). All curves, floor estimates and coordinate manifests retained, including Float64 evidence.
- Recovery 39/42 =92.86%, 0/14 parameters miss all three seeds. First-seed CSV unchanged; two additional declared seeds per arm, every interval saved in synthetic_recovery_three_seeds.csv. No tuning, widened intervals or omitted parameters.
- PHASE1.md records all gates and paths. Frozen table and saved manifest SHA256 rechecked unchanged after gates. Owned panes closed. Proceed to Phase 2 smoke; manager approval still required before grid.

## 2026-10-09 — Phase 2 frozen recipes and launch

- Declare grw_joint (sole control), qs_joint, grw_marketobs, qs_marketobs; same wave-1 lower scope and W2 sampler. All use GlobalInterception/GlobalHomeAdvantage. Joint observation exactly experiment 04 no-league-delta specification; all QS r priors weak LogNormal(0,1). Market sigma prior/table unchanged.
- Fresh beast REPL per arm, sequential order as above. Harness.smoke on pinned ScottishLower (generic CLI loads a different ScottishPyramid cache, so do not use it). Standard folds 1/40, 2x(200+200), standard 16-chain 50+50 performance probe. Root seeds 20261040–43 set before each harness invocation, no smoke seed tuning.
- Register canonical model/splitter/sampler/fit templates before sampling; check completed grid config hashes and stop rather than duplicate a completed fit. Harness creates smoke nonce internally and persists the exact invocation fit; preserve immutable UUID/check receipts.
- Six required hard checks must all be present/pass; database parity must not abstain. Convergence and performance review flags are reported, not hidden or treated as passing diagnostics. Report Rhat, bulk/tail ESS, divergence rate, wall and largest-fold probe ETA.
- ETA extrapolation: probe sampling window x(40x4x1500)/(16x100); largest fold, excludes grid/scoring overhead, not a guarantee. No grid code executed or authorised; ask manager after all four hard smoke passes.
- Initial Phase 2 commit hit a transient index.lock. The compound shell command (not set -e locally) continued and started launcher %372 on e86a03fe, which had no phase2_gate.sh; no Julia, smoke checks or sampling ran. Lock disappeared without removal; no git process remained. Close owned idle launcher, retain phase2_launcher0_no_source.log, and retry the commit once with set -e before any Julia. No scientific gate/seed rerun.

## 2026-10-09 — Phase 2 stop before smoke checks (73147ce0)

- First grw_joint entry point fails at candidates.jl:5: include undefined in Main.Wave2Candidates (bare Module constructor does not define the include helper). No candidate build, datastore preflight, registration, smoke hard check, model sampling or DB write occurred; other three arms not started. No UUID/convergence/ETA available.
- Stop/report per package. Proposed one-line Base.include(@__MODULE__, ...) correction, preserving Main.QSMarketArm and all frozen inputs/priors/seeds/thresholds. No source correction or retry yet; await manager authorisation. Keep previous launch evidence and use new names after approval.
- Phase 1 remains accepted under all three amendments, evidence e86a03fe; no Phase 1 rerun required. PHASE2_BLOCKED.md records precise error and action. Owned idle panes closed; no grid.

## 2026-10-09 — Manager authorises loader fix and fresh smoke retry

- Explicit ruling via confirmation: apply one-line Base.include(@__MODULE__, ...) in candidates.jl and retry smoke. Grid approval remains separate. No scientific gate had started at failure; no input/seed/prior/threshold change.
- Preserve original phase2_smoke_grw_joint.* and launch logs; retry with phase2_smoke_v2_* evidence/session names and new phase2_launcher_v2.log. Same fresh-REPL serial arm order, pinned snapshot, registrations and smoke budgets.
- Correct README-only shape-prior typo (Normal(4,1) -> Normal(4,1.5)); executable prior already exactly matched experiment 04 and remains unchanged.

## 2026-10-09 — Phase 2 persistence parity stop and read-only diagnosis

- V2 source ac30c4d7: grw_joint and qs_joint all 6 hard checks pass, saved UUIDs e169aa8e-2302-4c9c-9e39-6c3b1e93218b and 032bd044-1f10-4e71-9876-f586a04eaa48. Short-smoke Rhat maxima 1.05414/1.07449, ESS minima 182.89/69.66 and 133.14/79.94, zero divergences; review flags retained. Probe ETAs 1.937/1.955 h.
- grw_marketobs passes 5 hard checks then fit_parity FAIL (FitConfig changed on reload), saved fit UUID 8ff19fe2-8a83-4a50-b964-8b8ac617e316. Sampler completion is not a smoke pass. Rhat 1.13787, ESS 16.10/57.47, zero divergences, performance review (GC 36.2%, GC-stall 56.4%, efficiency 4.741); ETA 4.219 h. qs_marketobs not started; no grid.
- Read-only audit source 58665779 confirms all 1430 dictionary key/value tuples unchanged after in-memory and actual DB round trips. _structural_equal descends into Dict hash-table internals, so returns false despite value equality; full serialization-only config clone reproduces, non-model fields unchanged. No chain/latent parity claim after config short-circuit.
- Stop/report; no parity fix, sampling retry or threshold/input/seed change. Proposed exact per-key recursive AbstractDict comparison with positive and mutation-negative regressions requires manager review. Also request eligibility policy: current has_passing_smoke rejects any historical failed hard row even after a passing retry; retain history, do not delete/relabel it.
- PHASE2_PARITY_BLOCKED.md is current stop evidence; raw/CSV/probe receipts retained. Table/manifest digests rechecked unchanged. All owned panes closed, TODO BLOCKED pending manager parity/eligibility ruling.

## 2026-10-09 — Manager ruling: array-backed feature, no harness changes

- Authorised fix solely inside the new component: MarketRatesFeature stores sorted match_ids::Vector{Int}, aligned Float64 log-rate vectors and full_book::Vector{Bool}, with searchsortedfirst extraction. No Dict in the feature; preserve masking, duplicate/nonfinite checks and frozen table values. src/harness remains untouched.
- Record AbstractDict structural-comparator gap as TODO 041 follow-up, not a fix in this package. Retain every historic failure; no row deletion/relabel or eligibility override.
- Additional explicit confirmation authorises component-local canonical identity: harness recipe_hash only hashes model display/scope; its generic observation display ignores feature fields. Therefore arrays alone would NOT change the failed recipe hash. New MarketRateObservation display includes versioned canonical content SHA256 and actual sigma prior, computed from the current arrays, not an attempt suffix/cached digest. This gives the revised feature a truthful new recipe identity without changing harness.
- Full test_market_rate_observation.jl and tape_allocation_tests.jl entry points must pass in fresh 16-thread pinned beast REPLs, BLAS 1. All exact-AD/FD/likelihood thresholds, priors, truths, budgets and declared seeds unchanged. New unit checks cover sorted alignment, missing IDs/empty input, whole-config structural serialization parity, and digest sensitivity to rates/flags/IDs/prior.
- Revalidation output is phase1_arrays/ rather than overwriting original evidence. Rerun ALL three declared recovery seeds per arm for the new feature (including first seed) at the same budget, retaining the old first-seed CSV unchanged. Report all 42 new intervals; no tuning.
- Then ONLY grw_marketobs and qs_marketobs fresh smokes (new phase2_smoke_arrays_* logs). Joint hard smoke passes at ac30c4d7 stand. Clearly retain previous GRW market smoke Rhat 1.13787 / bulk ESS 16.10 at 2x(200+200), 4.219 h ETA when asking BEFORE grid. Full four-arm ETA required; no grid authorised.
- Fresh array feature validation at b006743f PASS, complete 18:13:39 UTC: component 3831/3831, tapes 160/160 (0 B). Exact AD max 4.1099e-16; 735 Richardson checks pass. All three recovery seeds rerun: 39/42, zero all-seed misses; every one of the 42 intervals exactly reproduces previous evidence. Full report ARRAY_FEATURE_REVALIDATION.md. Table/manifest SHA256 unchanged. Commit evidence before market-only smoke retry; phase2_arrays preserves original failure artifacts.

## 2026-10-09 — Market-only smoke retry PASS; approval before grid

- Source d0d933ae: both market arms 6/6 hard PASS, including real PostgreSQL fit_parity. UUIDs GRW 30ad2ca7-ccd4-4cc0-9973-81142b67dcb3 / QS 3ec74b0f-ee29-4bef-9ed9-dea5ceb8bba5. No joint rerun; prior ac30c4d7 receipts stand. Original dictionary failure rows/UUID retained, no relabel or bypass.
- GRW short smoke diagnostics exactly retain the original warning: Rhat 1.13787, bulk/tail 16.10/57.47, 0 divergences. Original 4.219 h probe preserved; new 4.199 h. QS Rhat 1.06250, bulk/tail 67.14/74.33, 0 divergences. Both convergence statuses fail; these are hard-check passes, not convergence passes. Performance review flagged (non-tape GC/efficiency), 0 B tapes remain.
- Four-arm probe ETA 1.9374+1.9550+4.1991+4.2278 = 12.3192 h (~12 h 19 min) before setup/scoring/checkpoint overhead and diagnostic reruns. PHASE2.md and four_arm_smoke_summary.csv carry all flags, exact values, UUIDs and source commits.
- Additional read-only fresh REPL %387 at d0d933ae verifies unchanged has_passing_smoke true for all four current recipes. No sampling/DB writes; exact audit source and receipt saved. Original table/manifest SHA256 unchanged. All owned panes closed.
- Stop at approval gate. TODO 041 BLOCKED only on manager grid approval; no grid code, scores, ROI/staking or reference refits. Ask with full ETA and explicit GRW 1.14/bulk16 warning; preserve prescribed grid budget/order and convergence rerun policy.

## 2026-10-09 — Explicit grid approval and seeds frozen before sampling

- Manager authorises the frozen four-arm grid, with all short-smoke flags and full 12.3192 h ETA disclosed. No budget/prior/data/threshold tuning.
- Experimental durable queue uses unchanged Training.sample_fold, 16 concurrent tasks, fixed per-chain task-local seeds: 202700000 + 100000*arm + 100*fold + 10*attempt + chain (arm 1–4 in approved order, fold 1–40, attempt 0/1, chain 1–4). Postprocessing seed 202700000+100000*arm+9000+attempt. Frozen BEFORE first grid; same queue can refit a single fold byte-for-byte independent of scheduling/resume.
- Reason for local queue: core fit_model writes checkpoints only AFTER the whole pending queue returns. Atomic complete-fold landing avoids losing a multi-hour in-flight run; no core/harness modification. Manifest refuses changed recipe/sampler/seeds; mock-chain persistence/resume/seed tests precede real sampling.
- Unchanged Harness.grid consumes the complete checkpoints, audits latents/coverage/parity and persists/scorers as usual. Primary UUID retained. If any Rhat>1.05, a separately named __rhat_rerun full fit keeps unaffected initial folds and refits each bad fold ONCE with attempt-1 seeds. Always select that rerun, even if worse; never third attempt, prior tuning or favourable subset. Preserve selection and both diagnostics/UUIDs. Final divergence <=0.1% / finite-Rhat gate; tail ESS reviewed against 400.
- Grid shell is sequential fresh REPL per arm; immutable attempt-labelled logs and receipt/checkpoint resume. No scoring adapter/references changed.

## 2026-10-09 — Phase 3 scoring stop: experimental wrapper omits control

- Mock-chain queue validation PASS before sampling; source 8f285a6a grid starts 19:07:51 UTC. GRW completes 19:41:28 UTC, UUID 98447840-e95c-420f-92b4-2db6545506a3, all40 folds, Rhat1.01008, bulk641.78/tail324.47, zero divergences/reruns; one tail400 review flag retained. Include2008.6993s.
- QS all40 folds persisted and four hard checks (filtration/latents/coverage/DB parity) PASS. Then Harness.grid scoring fails expected exactly one role=:control; found0 at20:14:20 UTC, include1962.5365s. No market arm started. Cause: grid_stage accepts control=nothing, but run never forwards the GRW reference for role=:candidate. Core scorer correctly requires a control; do not bypass it.
- Read-only diagnostic audit in retained %392 finds completed QS UUID 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea, Rhat1.00937, bulk832.35/tail761.02, zero divergences/reruns due, convergence review PASS. Exact CSV/check-row evidence in results/phase3; error/logs/audit script in results/gate_logs. All80 atomic checkpoints retained. No refit, scoring retry or DB mutation from the audit.
- Stop/report per package, TODO BLOCKED. Proposed manager-reviewed experiment-only l05/r05 fix: explicit RunRef for final grw_joint UUID from its summary CSV, forwarded to primary/rerun grid_stage for non-control arms with preflight/regression. Resume via fresh attempt label using saved receipt/DB fit/checkpoints, never resample joint fits. No src/harness edit, science/seed/budget/threshold change or favourable selection.
- All owned panes closed after evidence audit: launcher%390, successfulGRW%391, failedQS%392. Other sessions untouched. PHASE3_BLOCKED.md/PROGRESS/HANDOVER are current continuation state; Phase3+ incomplete.

## 2026-10-09 — Authorised explicit-control correction and resume

- Manager authorises ONLY l05/r05 wrapper correction plus offline forwarding regression. Explicit approved final GRW UUID98447840-e95c-420f-92b4-2db6545506a3 must be owned by this experiment, completed and contain all40 folds before candidate sampling. Both primary and prescribed rerun receive its RunRef. All scientific settings unchanged.
- Skip standalone GRW stage completely: no refit/resampling/new GRW run. QS must resume persisted UUID2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea, fail closed if the completed recipe query does not resolve to this UUID, and finish scoring/receipt without sampling. Then market arms in original order under fresh attempt label/fresh REPLs.
- Clarification requested because unchanged Harness.grid recomputes control bundle when pairing candidates. Manager explicitly ALLOWS this internal deterministic paired-control scoring, but requires every recomputed GRW score row equal its existing row exactly or within1e-12 for floats. Wrapper snapshots existing score rows before each stage, compares every field after core scoring/upsert, writes comparison CSV and refuses a receipt on any mismatch. Offline regression includes perturbation failures. No src/harness edits.
- Original GRW tail ESS324.47 review flag remains in report; no Rhat rerun warranted. Earlier scoring failure evidence retained. Wrapper at081ba5a6 offline regression PASS57/57 in fresh %393:8ownership+7scorecomparison+42forwarding/negative checks. No MCMC/datastore/DB in this regression.
- V2 launcher%394 skips standalone GRW. FreshQS%395 resumes exact2a5eee90, no sampling, completes scoring/receipt20:32:29UTC; all664GRW score rows/9960fields EXACT, maxdiff0. All80joint checkpoint SHA256 files unchanged. CONTROL_FORWARDING_REVALIDATION.md / exact baseline and comparison CSVs committed. %393/%395 closed after PASS. FreshGRWmarket%396 starts20:32:29UTC, thenQSmarket; source remains081ba5a6 while grid runs.

## 2026-10-09 — Phase3 complete; review flags preserved

- GRWmarket%396 completes21:18:36UTC UUIDe47a71a7-52eb-4aa5-b747-e64099ea9977; QSmarket%397 completes22:05:57UTC UUID81fdf817-f82d-4f66-9bfd-5f456c4fe29f. All4×40initial folds Rhat<=1.05, zero divergences; no prescribed reruns due, all selection ledgersfalse. Both market internalGRW score comparisons664rows/9960fields EXACT maxdiff0, matching QS resume evidence.
- Fresh read-only DB audit%398 checks exactly4completed named runs/no duplicate or new joints, all160folds/640chains/1000retained per chain,16hard passes. All80joint checkpoints remain hash-identical. Original GRWjoint tail324.47 and GRWmarket tail339.34/bulk395.44 review flags retained: tail ESS never authorises extra sampling here. PHASE3.md and exact CSV/logs committed.
- All owned panes closed including launcher%394 and final audit%398. Source/sampler/seed/prior/input/threshold/reference unchanged; src/harness untouched. Phase4 full report/bootstraps/leaderboard and Phase5 fresh frozen-seed byte reproduction remain; task IN_PROGRESS. No predictive decision or full package completion claimed.

## 2026-10-09 — Phase4 saved-fit scoring

- Adapt wave1 r05_score.jl with explicit final wave2 UUIDs and saved wave1 control_grw/qs_weak_r/C0. No sampling or database writes in scoring. Preserve all prescribed pairs, seeds, panel and weights. Reference independent harness metrics and fixture goal scores must match saved wave1 exactly before reporting. Include unchanged r05_block_check/r05_goal_cluster as additional reuse/parity evidence.

## 2026-10-09 — Authorised Phase4 syntax retry

- Phase4 v1 at f9b6cdd7 stops at r06_score.jl:209: inherited extra closing parenthesis after the new clustered-hi field. Partial harness/reference parity CSVs and failed log retained under phase4_v1; no sampling or DB writes. Manager explicitly authorises deleting that single parenthesis in wave2 only and fresh-REPL retry with frozen settings unchanged. Wave1 source/outputs stay untouched.
- Read-only git provenance follow-up underway: current line blame 54a22ee7; historical classify(...) ending had three parentheses legitimately (function+tuple+push). New scalar clustered.hi retained all three, introducing the defect. Base result CSVs committed b14e98b9; clustered columns added via standalone r05_goal_cluster.jl and committed d16a5800. Check source/log provenance before final conclusion.

- Completed read-only wave1 provenance finding: defect introduced54a22ee7, not initial48832cc6 (classify(...) then had valid three closes). Base scoring source727af1d4: checkout13:15:40UTC, harness output13:16:39UTC, section7 subsequently failed on absent micro site; repaired r-only source81c77a42/7f3ff52b, then base CSVs committedb14e98b9. Clustered goal columns produced via standalone r05_goal_cluster.jl at54a22ee7 (CSV14:15:34UTC), committedd16a5800. Current full scorer cannot reproduce final outputs end-to-end; out-of-package follow-up recorded in TODO041. Evidence: results/gate_logs/wave1_source_reflog_mtimes.txt and wave1_scoring_log_excerpt.txt. Wave1 files unchanged.

## 2026-10-09 — Phase4 scoring/intervals accepted

- Fresh v2 scorer PASS after authorised one-character syntax fix; exact independent harness-reference metrics and all710 fixture goal-reference rows per saved arm, no sampling/DB writes. Goal-cluster reuse confirms all30pairs, byte-identical interval CSV. All56market and30goal comparisons have noncircular/circular/cluster paths; figure236fold-r rows and120sigma rows. Native SVG uses unchanged weak-r prior median1.
- All primary noncircular1X2 intervals cross zero; circular flags both market arms versusGRWjoint better. Noncircular drift0.004564; preserve classification and contradictions. No promotion. GRWESS flags retained. Source/result trace tables, full observations and REPORT committed at phase boundary; canonical board/DB score equality audit next, then only four frozen-seed onefold reproductions.

## 2026-10-09 — Phase5 frozen reproduction PASS

- Sourcee7abd34e, fresh REPLs%403–%406, onefold1 each in original order, frozen attempt0 chain/post seeds and4×(500+1000). All8posterior chain/latent Float64 payload comparisons byte-identical/SHA256equal,maxdiff0. No retries, reference refits, new DB fits or source/sampler changes. All160 original grid checkpoint digests unchanged; launcherexit0. All owned panes closed.
- REPRODUCIBILITY.md and phase5CSV/logs record exact UUIDs/seeds/counts/source. Final fresh read-only original-grid/ownership/budget audit and completed register materialisation next; task remains IN_PROGRESS until verification/checks recorded. No more sampling authorised.

## 2026-10-10 — Manager-required leakage audit PASS

- Read-only r10_leakage_audit.jl in beast%407: all80folds have zero active market observations in held-out targets or with kickoff >= first target kickoff; zero held-out likelihood rows of any kind. Saved-model and persisted-fold target equality checked. No fits, inversion optimisation, DB writes or scientific changes.
- All1430frozen rows match saved own-close rates/flags; reconstructing closes with all other matches removed is exact. Own-book KL objectives equal frozen objectives,maxgap0. Source dependency inspection confirms per-match-only inversion and no future-match information.
- Both arms' saved-latent probabilities equal committed prediction rows;595common fixtures/1785selections recompute LogLoss0.6117542636747713 and0.6119173567756958 exactly. No promotion decision unchanged.
- Initial audit failed only on Symbol/runtime versus string/CSV prediction keys, after provenance and40zero-count folds passed. Normalised key types only; final audit PASS. Both logs preserved. All owned panes closed. TODO041 remains IN_PROGRESS pending final registry materialisation/task closure.
