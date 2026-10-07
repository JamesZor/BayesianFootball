# Phase C revision 7 — exact-case PASS; recovery IN PROGRESS (2026-10-06)

**Not complete; not numerically blocked at this checkpoint.** The prescribed
C1 recovery is still running at frozen statistical source
`6335316f4e93ec91495341fa2f6960a64ce772a9`. No completed recovery fit, retained
interval, convergence result, production fit or forecast conclusion exists yet.
C0/C1/H1/H2 full books remains authorised; C2 remains separately deferred.
Revision6 report/reproduction/handover/source hashes are preserved with `_V6`
names; prior scientific evidence remains untouched.

## 1. Authorised solver change and exact verification

Revision7 manager brief: `experiments/pi_market_model_phaseC_v7_stall.md`.
An Armijo-accepted candidate is a stall when its movement infinity norm is
**<1e-14 OR its Float64 target equals the current target**. From the current
point, reuse revision6's <=3 undamped derivative-only Newton polish, only when
pre-polish decrement <=1e-9. Final gradient norm cannot exceed pre-polish norm;
final decrement must still pass <=1e-9. Errors propagate. The existing decrement
<=1e-12, step <=1e-10, search floor, likelihood clipping, derivative dispatch,
raw marginal precision, mean correction, quadrature, priors, seeds and population
are unchanged. Accepted-step stalls are counted as `zero_motion_polished`;
polish-step totals are now separated by termination type.

Sources: `v7_newton/newton_regression.csv`, `zero_motion_regression.csv` and
respective serialized-input CSVs. The tighter half-Newton reference and both
verification tolerances are unchanged.

| Fixture / frozen chain coordinate | Mode gap | Marginal gap, nats | Termination | Steps | Accepted decrement |
|---|---:|---:|---|---:|---:|
| 12476625 / seed4964 warmup8 | 2.0039525594484076e-13 | 1.1574741165532032e-11 | polished | 1 | 1.50342686280365e-24 |
| 12476686 / seed4961 initial target | 0.0 | 0.0 | zero_motion_polished | 1 | 2.287082719929954e-26 |

Limits remain **1e-8** mode and **1e-9 nats** marginal. New case is tested from
both the recorded stalled mode and its original prediction; both comparisons
are 0.0. Frozen revision6 forward prefix reconstructs the exact original
prediction covariance, not a revision7-selected fixture/prediction. Recorded
gradient matches `[1.8404678883143788e-6,4.9462900923558095e-5]` exactly.
Reference residual is 7.286615755219827e-12. The old case still reproduces its
original gradient exactly. Regression accounting:2 polished +2 zero-motion
polished,4 total steps,max1; failed/mock solves are excluded from these counts.

## 2. Fresh laptop evidence

Julia1.12.1,8 threads,BLAS1, pinned cache/existing Manifest; owned panes only.
- `%58`: t05 **305/305**, `T05_C7_DETERMINISTIC_DONE`; t04 **92/92**,
  t03/t02 **131/131**, all markers. A regression writes temporary outputs.
- Full real Gate1 regenerated: **12/12** settings,360 fixture-setting rows;
  state gates **53/53**, four fixed517-fixture filters/smoothers complete.
- `v7_preflight/newton_termination.csv`:1597 decrement,11 zero-motion polished,
  27 total polish steps,max3; max accepted decrement9.96108898035961e-13.
- C2-pending separately reports unchanged **29 pass /10 fail /39**, explicitly
  excluded, not passed or newly promoted.
- Fresh `%60`: workflow preparation **85/85**, including independent augmented
  joint-Gaussian theta/u mean/covariance gates and pre-week filtration checks.
  Core runner parsed, NOT executed. Final fresh `%61`:305/305 t05 and85/85
  workflow, both markers; C2 unchanged/excluded.

Logs: `LAPTOP_C7_T05.txt`, `LAPTOP_C7_REGRESSIONS_PREFLIGHT.txt`,
`LAPTOP_C7_WORKFLOW_FINAL.txt`, `LAPTOP_C7_FINAL_TESTS.txt`.
Earlier workflow harness attempt:70pass/1fail because the negative pairing test
removed a home-rate row, not a scored supremacy/level row. Corrected to remove
supremacy;71/71 then84/84 then final85/85. No numerical gate/tolerance changed.
The first failed output is preserved in `LAPTOP_C7_WORKFLOW_FIRST.txt`.

## 3. Prescribed beast recovery: live, no promotion

Dedicated `/root/BF_runs/market_model_c`, detached6335316f; owned pane **%265**.
Julia1.12.4,16 threads/core pinning/BLAS1, existing Manifest, pinned cache and
max_age_hours10^6. Frozen generation3962, chains4961–4964, truth(.03,.01,.06,1000),
517 fixtures/91 weeks/full market mix and4x(2000+3000),thin1. All four chains have
reported warmup200 at the latest recorded checkpoint, with no numerical abort.
**Warmup progress is not convergence, parameter recovery or retained draws.**
Do not restart/include recovery again or change/check out code while it runs.

Original log: `/root/BF_runs/logs/market_model_c/6335316f_recovery1.log`.
Local capture: `BEAST_C7_RECOVERY_IN_PROGRESS.txt`. Only53 passing recovery
engine gates have been copied back into `v7_recovery/`; no partial intervals or
acceptance accounting are invented. Run-local/root accounting writes on completion
or propagated failure, after @sync waits for every chain.

Generated binary SHA256 remains
`de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1`, byte-identical
with revision4/6. This is data identity, not fit or production reproduction.
Historical v6_recovery/v6_diagnostic directories were moved INTACT before source
checkout to `/root/BF_runs/logs/market_model_c/04e67fa8_outputs_preserved/`, including
that binary. Root aborted-run accounting was copied/compared into
`04e67fa8_accounting_preserved/` before restoring the tracked baseline for checkout.
No reset/delete/stash, cache/data/Manifest/.env sync or unrelated pane operation.

## 4. Prepared next stage (not sampled or scientifically accepted)

New loader-only `l05_fullbook_workflow.jl` / `l05_fullbook_reports.jl` add:
- conditional theta=structure+u smoothing under frozen C1 factors, checked
  against an independent augmented joint Gaussian (mean/covariance for theta/u);
- B2-compatible median-theta pre-week forecasts: C1 uses H*P*H'+sigma_u²*I,
  without own-book means/Hessians or an extra evaluation-book variance;
- paired sum of supremacy+level marginal logpd and fixture SE;
- physical hyperdraw summaries/ratios, conditional q/s paths, local plug-in book
  noise, full-book shrinkage diagnostics, and CSV schema helpers.

`r05_fullbook_core.jl` is the PREPARED C0/C1 stage in readable sections. It
regenerates full Gate1/state gates, requires the frozen recovery panel hash and
rechecks all retained recovery convergence/interval gates before any fit, then
fits both protocols at frozen budgets/seeds (10b first) and writes accepted
measures. It refuses non-beast/fewer threads/non-fresh output. **Not executed.**
Its marker is `R05_FULLBOOK_CORE_DONE`, not R05_DONE/PHASEC7_DONE.
The test's IID mock hyperdraws test schema only, NOT posterior recovery/fit.
Conditional path bands exclude hyperparameter mixing; plug-in noise is not
reported as posterior quantiles. H1 static-gamma mixing, HA posterior figures,
nonlinear smoothed log-total, Ridall source check, figures and full top-to-bottom
production/reproduction remain work to finish. `r05_pooled.jl` still preflight.

## 5. Definition of done / priority

Definition of done: all authorised gates/regressions/recovery/converged fits,
measures/figures and two fresh byte-identical beast production runs. **Not met.**
First monitor the existing recovery. On any supported failure, preserve expanded
exceptions/coordinate/accounting and stop; no retry/threshold/seed/prior change.
After recovery passes: C0/C1 both protocols/honest10b measures, H1/H2, complete
measures/figures/production workflow, second reproduction. Keep all-parameter
Rhat<=1.05 and bulk/tailESS>=200, unchanged scoring population and fixture pairing.

C2 thin pooling remains blocked/deferred exactly as revision4: wide-spread thin
Laplace accuracy and integrated derivatives remain unvalidated. Human options
(exact thin quadrature, exact NUTS, actual-spread gate, restricted types) are
unselected. No thin books enter these fits; no later score-grid C2 or PhaseD.

Local panes close at handoff. The ACTIVE beast pane is explicitly transferred to
the next session rather than killing the only prescribed recovery attempt.
Source/scientific hashes identify deterministic and IN-PROGRESS evidence only,
not two complete production runs. Tracker/diff/hash checks precede final commit.
