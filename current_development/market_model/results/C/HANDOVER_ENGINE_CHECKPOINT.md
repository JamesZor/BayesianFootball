# Phase C revision 4 — implementation checkpoint handover (2026-10-06)

## State: full-book continuation authorised, still incomplete

Worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`,
branch `pi/market-model-phase-c`. **Manager revision 4 already authorises
C0/C1/H1/H2 on full books only. Do not ask again for that scope approval.**
C2 thin pooling is deferred; no thresholds, spreads, grid or data are changed.
The prior v3 handover is preserved in HANDOVER_V3.md, not current instructions.

**No C0/C1/H1/H2 posterior fit, sampler, sampling, synthetic parameter recovery,
forecast evaluation, learned n/sigma_u, shrinkage, ratings/HA posterior figures
or production reproduction exists.** Deterministic state engines now exist
and pass their necessary gates. r05 is still a preflight, not production.
TODO 039 is IN_PROGRESS; all owned Julia panes `%45`, `%46`, `%47` are closed.
No beast checkout/session was created. Do not operate unrelated panes.

Read AGENTS and Julia/runner guides, manager revisions 1–4, stream README/DESIGN,
PHASE_C_REPORT.md and REPRODUCIBILITY.md. Previous scientific reports were read
in this session; reports and outputs A/B/B2/B3 remain untouched. Sentinels are
PHASEC4_DONE / PHASEC4_BLOCKED / PHASEC4_HANDOVER, alone on their line.
Definition of done remains all authorised rungs, gates, recovery, converged
beast fits, measures/figures and two-run byte identity; **not met yet**.

## What was implemented and proved

`l05_fullbook_engine.jl` is included INSIDE `PooledMarket` by l05_pooled.jl.
It adds:

- `FullBookRung(:C0/:C1/:H1/:H2) <: MID.AbstractArm`, correct physical names,
  init centres and new `MID.log_prior` methods. C0/H1/H2 coordinates are
  log(sigma_obs), log(sigma_q), log(sigma_s), plus raw kappa for H2.
  C1 coordinates are log(sigma_q), log(sigma_s), log(sigma_u), log(n).
  HalfNormal priors have full normalisers/Jacobians. Log n is exactly
  Normal(log(1000),1.5); **do not apply MID.THETA_BOUND to it**.
- `fullbook_initial`, `fullbook_process`, `fullbook_design`: same raw
  alpha/beta initial prior and zero-sum design as B2, static mu/gamma,
  ordinary weekly q/s innovations. Q's same-team diagonals are q²+s²,
  off-diagonal s²−q²; no summer jump or common mu shock.
- H1 adds one static state gamma_def, negative on away scoring. **Manager
  explicitly clarified its prior in this session:** gamma_att N(.15,.25²),
  gamma_def N(0,.25²), independent. `gamma_def_sd=0.0` pins the latter for
  its C0 reduction gate. This pin is a gate only, not the fitted prior.
- H2 adds kappa times the centred HOME team's q to its own scoring row,
  with kappa N(0,.5²). kappa=0 reduces exactly to C0.
- `fullbook_markets(ds,p,config)` asserts fixture/target order, unique IDs,
  full-book type and config exclusions, returns markets in PANEL order.
- `fullbook_filter(a,p,theta; markets,store,predict)` exact joint Gaussian
  C0/H1/H2; approximate C1. Forecast all fixtures pre-week, THEN update
  in inherited week/date/ID order. C1 adds ONLY the returned marginal from
  existing `laplace_update`; no additional Kalman likelihood or lost peak.
  It uses production density for raw marginal/FD derivatives and identical
  AD algebra for the third tensor, unchanged from passing v3 full-book code.
- `conditional_state_update`: a=H m, S=H P H'+sigma_u²I,
  K=P H'/S, m+=K(b−a), P+=K(V−S)K'. Integrates u once, no inverse of P.
  Singular-state, information/moment identity, PSD/symmetry tests pass.
- Stored `f.factors` retain fixture IDs, weeks, H, z/R/constant, a/S,
  theta posterior moments and raw/clipped curvature diagnostics.
  `frozen_book_factor` defines exp(constant)*Normal(z;theta,R), choosing
  the constant to match the original returned marginal at its prediction.
  R+sigma_u²I observes H*x. This frozen surrogate is Gaussian; the original
  nonlinear model is not. Do NOT remove constants from batch/sampling checks.
- `fullbook_smoothing(f)` RTS on stored Gaussian moments, approximate C1.
  `fullbook_batch` independently assembles original-side joint Gaussian
  covariance over weeks; C1 accepts the FROZEN factors. No globally exact
  Dirichlet batch claim.
- `fullbook_engine_gates()` includes all 34 B2/B/TODO 023 gates plus 19 new
  reductions, independent batch/RTS and scalar-eigen/full-2D checks.

`laplace_gate` now defaults `kinds=("full",)`; it no longer constructs thin
prediction centres in that scope. Explicit kinds are diagnostic-only C2.
**No change to laplace_update/marginal/third tensor/integrator algebra.**

## Verified artifacts and exact marker meanings

Fresh final-source laptop `%47`: t05 **209/209** necessary assertions,
`T05_C4_DETERMINISTIC_DONE`, then separate **C2-pending 29 pass/10 fail/39**,
`C2_PENDING_REPORTED`. The unchanged integrated derivative discrepancy remains
5.405838441375254e-5 (limit 1e-6); nine archived thin Gate 1 setting flags fail.
The pending tests execute/report, excluded by manager scope. No new numerical
variation or thin quadrature rerun was made. **Not T05_DONE**: parameter recovery
is unbuilt. `t05_fullbook_engine_tests.jl` is included inside acceptance;
`t05_c2_pending_tests.jl` is included AFTER it, with its own labelled exception
reporting. Do not move non-C2 failures into that exception boundary.

Fresh `%46` regressions t04 **92/92**, t03 **131/131**, t02 **131/131**, all
markers. t03 uses temporary A outputs. Logs are in results/C:
LAPTOP_C4_FIRST_TESTS.txt, LAPTOP_C4_REGRESSIONS.txt,
LAPTOP_C4_FINAL_PREFLIGHT.txt. First development t05 failed its log-prior identity
once: MID's HalfNormal helper omits constants. New priors now include those
constants, fixing discrepancy 5.53723404048801 with unchanged priors/thresholds.

`r05_pooled.jl` fresh-source preflight completes:

1. Pinned cache check/load.
2. Full Gate 1, **12/12 settings / 360 fixture-setting rows**, same first 30,
   same n/spreads/offsets. Max median .003980333011027959 nats, max p95
   .0051705855019272395, max mean/SD .0009284497092163062, max SD relative
   .010184310576972222. All unchanged thresholds pass.
3. **53/53** state gates. C0 matched R6 error 3.552713678800501e-14;
   H1/H2 reductions 0.0; scalar eigen errors <=6.661338147750939e-16.
4. All four FIXED-parameter filters/smoothers run on 517 fixtures.
   C0/H1 theta=log([.07,.03,.01]); H2 also kappa=0;
   C1 theta=log([.03,.01,.06,1000]). All finite and positive smoothed
   minimum eigenvalues. **These are not fits.** C1 warmed single filter
   .320236339 s locally; use this as a warning of slice runtime, not a
   posterior/sampling timing claim.
5. `C05_C4_ENGINE_PREFLIGHT_DONE`, **NOT R05_DONE**.

New outputs only `results/C/v4_preflight/` (six CSVs). Historical root C CSVs
remain v3 partial/all-types failures, copied to v3_gate/. Earlier reports,
reproduction notes and handover have V3 archives; v1/v2 preserved too.
Current source/scientific hashes identify a checkpoint, not two production runs.
An auxiliary DuckDB historical-row join failed twice on reserved identifiers
and was stopped; no new cross-version row/byte-identity comparison is claimed.

## Exact next work, in order

1. Implement a dedicated full-book slice chain/fit wrapper, using `FullBookRung`
   names/priors and `fullbook_filter` likelihood. `MID.slice_sweep` and CM chain
   adaptation mechanics are references. Physical draws: exp positive coordinates,
   H2 kappa unchanged. ArmFit can carry this arm. **Do not reuse fitted R6 chains,
   its rho/scale prior, or its whole-target ±12 guard.** Support/overflow handling
   must be explicit; mode/PSD failures must not be silently converted to posterior
   rejections. Adapt widths during warmup only, prescribed 4×(2000+3000), thin 1.
   Freeze a fresh seed manifest; prototype fit serialisation requires loaders.
2. Add full-book synthetic panel generation with known q/s/u/n and
   Dirichlet(n*q) market draws on the same grid, then full parameter recovery
   within each 90% interval. Run sampling ONLY on beast. Mixed-type recovery is
   C2 deferred; do not sneak thin books into this recovery. Deterministic tests
   of the generator/population/normalisers first. Complete t05's recovery evidence
   and engine gates before production promotion; current necessary tests aren't
   full acceptance.
3. Before beast work read remote guide (already read in this session), check
   connectivity/load, unused session/checkout names and existing Manifest/cache.
   Use `/root/BF_runs/market_model_c`, session `pi_mm_c`, returned pane ID only,
   `-t 16`, core pinning/BLAS=1, logs `/root/BF_runs/logs/market_model_c/`.
   Copy pinned cache preserving metadata, no refresh. No DB writes. All existing
   remote panes belong to others; do not operate them. No session to resume.
4. Train each C0/C1/H1/H2 in 10a full and 10b config.honest_train only. For 10b
   restrict panel and its markets together by fixture ID, never positional reuse
   of full-panel market vectors. Convergence every parameter/rung/protocol:
   Rhat<=1.05, bulk/tail ESS>=200. Failed gates block inference promotion.
5. Evaluation on SAME full-book isolated targets, B2 schema/axes/subsets. 10a is
   retrospective; 10b only scores honest_test, with all forecasts pre-week.
   C0/H1/H2 predictive noise=sigma_obs; C1 theta prediction=H P H'+sigma_u²I.
   Own fixture book must not set its forecast mean or add an evaluation-book
   variance. Paired sum of supremacy+level marginal logpd versus C0 + fixture SE.
   Report measured gap to R6: matched likelihood does not make priors/fits equal.
6. Add descriptive smoothing, fixture theta=structure+u shrinkage and noise/n
   summaries, q/s paths/ratios for named transition clubs. C1 approximate RTS
   and frozen factors are ready, but theta/u smoothing extraction is NOT done.
   Do not mistake H*smoothed_state for theta including u. Derive joint Gaussian
   conditional u moments for frozen factors, with independent toy checks.
7. H1 gamma_att/gamma_def are STATIC STATES, not sampled theta columns. Report
   their posterior with state uncertainty (and hyperparameter mixing as required),
   not merely a median-theta RTS point. H2 kappa is sampled. Reference the requested
   Ridall p.1/3 convention only after checking the library source. No new HA
   posterior was estimated here.
8. Runner needs actual model/seed/config, recovery/training, convergence,
   inference/evaluation/output sections. Gate 1 must be regenerated in the full
   production run too. Root production tables should be distinct from historical
   v3 diagnostics; preserve evidence before replacing anything. Final R05_DONE
   only after all tables/figures and gates. Two fresh beast runs at frozen source
   and seeds must give byte-identical scientific CSVs. Existing hashes do NOT
   satisfy reproduction.
9. Update report/README and tracker honestly. C2 remains deferred, with four
   human options already included unselected. No Phase D or later score-grid
   Phase C2. TODO 039 is not closable while later authorised work is unfinished.

## Commands / guardrails

Use REPRODUCIBILITY.md for fresh REPL commands and current hash checks.
Pinned cache SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
load max_age_hours=10^6. No one-shot Julia, package/data/grid/threshold changes,
`src/` changes or DB writes. Never force-push, merge/rebase or touch stash.
Only push origin pi/market-model-phase-c. Preserve historical evidence and
close only owned panes. Before commit/handoff: git diff --check, todo.sh check,
current source/scientific hashes. Keep work log and matching index in sync.
