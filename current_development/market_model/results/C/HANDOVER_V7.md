# Phase C revision7 — exact-case PASS; live recovery HANDOVER (2026-10-06)

## Definition of done / scope / PRIMARY ACTION

Same worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`,
branch `pi/market-model-phase-c`. C0/C1/H1/H2 full books remains AUTHORISED;
no renewed scope approval is needed. C2 remains separately deferred.

Done = all authorised gates/regressions/recovery/converged fits, measures/figures
and two fresh byte-identical beast production runs. **NOT MET.** No completed
recovery fit/interval/convergence or production fit/conclusion yet. This is
HANDOVER, not DONE or a new numerical BLOCKED checkpoint.

**FIRST: monitor the EXISTING recovery in beast pane %265. DO NOT relaunch it,
kill it, checkout/edit running code, change seeds/budgets or start another grid.**
This session deliberately transfers ownership of that active pane rather than
killing the only prescribed attempt. All laptop panes %58–%61 are CLOSED.
No unrelated session/pane was operated. No other owned recovery process exists.

Frozen solver/recovery source **6335316f4e93ec91495341fa2f6960a64ce772a9**.
New workflow preparation is committed later but has NOT been loaded into that
running REPL. Previous revision6 report/repro/handover/source hashes preserved
with `_V6` names. Prior-phase/scientific checkpoints remain unchanged.

Read AGENTS, Julia/runner guides, manager revisions1–7 under experiments/,
stream README/DESIGN, current PHASE_C_REPORT.md and REPRODUCIBILITY.md before
changes. Revision7 explicitly supersedes revision6's zero-motion review STOP.
No scientific setting or verification threshold changed.

## Revision7 implementation / exact and fresh gates PASS

Armijo-accepted step is a stall iff actual movement infinity norm<1e-14 OR
Float64 target equals current target. With pre-polishdelta<=1e-9, reuse existing
<=3 undamped derivative-only polish from current x. Same stop/gradient/decrement
acceptance/error rules. Count `zero_motion_polished`, with separate step totals.
No iteration/line-search floor/eigenvalue/derivative dispatch/density/normaliser/
clipping/marginal/mean-correction/quadrature/prior/seed/grid/population changes.

New regression captures fixture12476686's exact prediction through the FROZEN
revision6 forward prefix at seed4961 initial coordinate. Recorded mode
[0.11192915227756556,2.14396416446582], gradient
[1.8404678883143788e-6,4.9462900923558095e-5] matches exactly.
From BOTH recorded mode and original prediction: mode gap0.0 (limit1e-8),
marginal gap0.0 nats (limit1e-9);1 zero-motion polish step,delta2.287082719929954e-26,
reference residual7.286615755219827e-12. Inputs/results in v7_newton/.
Old fixture12476625 still passes unchanged limits:2.0039525594484076e-13 mode,
1.1574741165532032e-11 marginal,1 polished step,delta1.50342686280365e-24.
Reference method/tolerances unchanged; no selected replacement seed/fixture.

Fresh %58/final fresh %61 t05 **305/305**,T05_C7_DETERMINISTIC_DONE, NOT T05_DONE.
C2-pending remains29pass/10fail/39, executed/reported separately and excluded.
%58 t04 **92/92**,t03/t02 **131/131**,all markers; A uses temporary outputs.
Real full Gate1 **12/12**,360 rows,state53/53,four fixed517-fixture filters/smoothers.
`v7_preflight/` accounting1597decrement,11zero-motion polished,27steps/max3,
max accepteddelta9.96108898035961e-13. These are NOT posterior fits or complete
production runs. Exact-regression accounting2polished+2zero-motion,4steps/max1.
Logs named LAPTOP_C7_*; no old-output identity claim under changed termination.

## ACTIVE recovery — ownership and precise monitoring

Checkout `/root/BF_runs/market_model_c`, detached6335316f; own session pi_mm_c,
**pane %265**, Julia1.12.4,16 threads/core pinning/BLAS1, existing Manifest.
Pinned cache c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
max_age_hours10^6; no DB writes/package/cache changes. Generation3962,
chains4961–4964,truth(.03,.01,.06,1000),517 fixtures/91 weeks/full line mix,
2000warmup+3000retained per chain,thin1. ALL FOUR reported warmup200 at the
recorded checkpoint; no numerical abort yet. Progress is slow: monitor coarse
intervals, not tight loops. It has NOT passed recovery/convergence.

```bash
ssh root@mcmc-beast 'tmux capture-pane -t %265 -p -J -S -100'
```

Original log `/root/BF_runs/logs/market_model_c/6335316f_recovery1.log`.
Latest saved checkpoint `results/C/BEAST_C7_RECOVERY_IN_PROGRESS.txt`.
Historical launch include(r05_fullbook_recovery.jl) — **do not send again**.
Sampler progress still says `C6`; this is unchanged logging text, not old solver.
Expected success marker R05_FULLBOOK_RECOVERY_DONE, not R05_DONE.

Outputs `results/C/v7_recovery/`:53 passing engine gates and generated binary
exist. NO C1_recovery.jls/interval/convergence table was present at checkpoint.
Accounting writes in finally on completion/error after @sync waits for peers;
no partial accepted-mode counts were invented/copied. Binary SHA256
**de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1**, byte-identical
with v4/v6 panels, data identity only. Only engine CSV/captures copied to laptop.

On failure: wait for all cancellations; expand `Base.current_exceptions()` /
CompositeException to retain ROOT error and coordinate; preserve accounting,
panel/log/CSV; STOP without another numerical variation or sampler retry.
On success: recheck ALL parameters Rhat<=1.05,bulk/tailESS>=200 and all four
truths within their90% intervals. File existence/progress alone is not acceptance.

Remote historical preservation: v6_recovery/v6_diagnostic moved INTACT to
`/root/BF_runs/logs/market_model_c/04e67fa8_outputs_preserved/` before checkout,
including binary (hash verified). Root aborted-run accounting copied/compared to
`04e67fa8_accounting_preserved/newton_termination.csv` before tracked-baseline
restoration. Original v4/c254 preservation paths remain as HANDOVER_V6 records.
No reset/delete/stash; do not overwrite/delete those evidence directories.

## Prepared C0/C1 next stage (not launched)

New `l05_fullbook_workflow.jl` / `l05_fullbook_reports.jl` and
`t05_fullbook_workflow_tests.jl`: final fresh %60/%61 **85/85**,
T05_WORKFLOW_PREPARATION_DONE. Independent augmented joint-Gaussian theta/u
mean/cov gates, forecast parity with matchedR6 and whole-week perturbation
filtration, fixture-pair guards, schema smoke/recovery-input refusal pass.
Mock IID hyperdraws are not sampled recovery/production posteriors.
One early harness failure removed an unscored home axis rather than supremacy;
corrected index, not a numerical threshold. First log preserved.

`r05_fullbook_core.jl` parsed but NOT executed. It is a readable C0/C1 STAGE,
not complete C0/C1/H1/H2 production: regenerated Gate1/state gates, frozen
recovery panel hash plus retained-draw convergence/interval checks before any
fit, frozen seeds/budgets,10b first then10a, accepted B2-compatible forecasts,
paired-vs-C0/fixtureSE, conditional theta smoothing and shrinkage/local noise/
parameter ratios/conditional q/s paths. C1 forecast variance is structure+u,
without own-book posterior mean or extra evaluation-book variance.

Only AFTER recovery/tasks stop: safely fetch/check out the latest permitted
branch source, preserving newly untracked recovery outputs before checkout.
Start a fresh own beast16-thread persistent REPL, then include core runner.
Defaults C05_RECOVERY_INPUT=v7_recovery/, C05_CORE_OUT=v7_core_1/; output must be
fresh. Do not accidentally re-run recovery or interpret R05_FULLBOOK_CORE_DONE
as complete acceptance. r05_pooled.jl remains a deterministic preflight.

Remaining priority: C0/C1 both protocols/honest10b measures and MEASURED paired
R6 gap, then H1/H2 (static H1 gamma-state uncertainty + hyperparameter mixing;
H2 sampledkappa), then figures/full production workflow/second reproduction.
Helpers' conditional q/s bands exclude hyperparameter mixing and local book
noise is plug-in, not posterior quantiles. Nonlinear smoothed total, all required
shrinkage/noise/n/HA/ratings figures, source-checked Ridall citation, and complete
r05 production workflow/reproduction still need finishing. No learned parameter
or forecast-benefit conclusion exists. C2/later score-grid phase/PhaseD deferred.

## Guardrails / finish checks

No one-shot Julia,src/package/data/grid/scientific-threshold/DB changes,
force-push/merge/rebase/stash or unrelated panes. Only push origin
pi/market-model-phase-c. Before commit/handoff:diff--check,todo.sh check and
source/scientific hashes. Tracker and matching README row together.
Current manifests cover deterministic/preparation/in-progress evidence only,
NOT two successful byte-identical complete production runs.

PHASEC7_HANDOVER
