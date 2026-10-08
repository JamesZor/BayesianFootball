# Phase C revision6 — exact-case PASS, recovery iteration-limit BLOCKED (2026-10-06)

**Not complete.** Revision6's prescribed polish fixes the exact revision4/5
stalled case at unchanged tolerances. Prescribed recovery then fails a NEW
Newton100-iteration limit. Stop without another solver variation or sampler
retry. Full-book C0/C1/H1/H2 remains authorised; C2 is separately deferred.

Primary statistical source **d9e6c0030d6c56e2325a2448b0e2525971b121a0**.
Observer-only diagnostic source **04e67fa85b50d4793c33d7c8e5c1a0782786d1db**;
no production equations or termination decisions change in that commit.
Revision6 brief copied unchanged into experiments/. Revision5 evidence is
archived with `_V5` names; all previous scientific artifacts remain unchanged.

## 1. Authorised implementation

Only on Armijo exhaustion withδ<=1e-9: up to3 undamped, derivative-only Newton
steps, stopping at step infinity norm<=1e-12 or nondecreasing gradient norm.
Final norm cannot exceed its pre-polish norm and finalδ must pass the existing
1e-9 gate; otherwise errors propagate, never -Inf/fallback. Record `polished`,
total steps and max steps. Existing search floor/raw precision/density/derivatives/
likelihood clipping/marginal/mean correction/quadrature/priors/population unchanged.
All scientific and verification tolerances, budgets and seeds stay unchanged.

## 2. Exact-case regression and fresh laptop gates PASS

Source: `v6_newton/newton_regression.csv`, full fixture12476625 at the unchanged
seed4964 warmup8 coordinate, generation3962. Original forward-prefix replay
exactly reproduces the handover gradient. The same tighter half-Newton reference
and both verification gates are retained.

| Measure | Observed | Limit |
|---|---:|---:|
| Mode infinity-norm difference | 2.0039525594484076e-13 | 1e-8 |
| Laplace marginal difference, nats | 1.1574741165532032e-11 | 1e-9 |
| Accepted decrement | 1.50342686280365e-24 | 1e-9 |
| Undamped polish steps | 1 | 3 |
| Tighter-reference gradient residual | 5.340616837656853e-12 | 1e-10 diagnostic |

All pass. Regression accounting records2 actual polished solves,2 total steps,
max1; unit mocks reset before these counters.

Owned laptop `%56`, Julia1.12.1,8 threads/BLAS1, pinned cache/existing Manifest:
- t05 **289/289**, T05_C6_DETERMINISTIC_DONE, not complete recovery acceptance;
- separate C2-pending unchanged29pass/10fail/39, explicitly excluded;
- t04 **92/92**, t03/t02 **131/131**, all markers; A regression uses temporary outputs;
- real full Gate1 **12/12**,360 rows, state gates **53/53**;
- all four517-fixture fixed filters/smoothers complete, not posterior fits.

`v6_preflight/newton_termination.csv`:1597decrement,11polished,25 total steps,
max3, maximum acceptedδ9.827548787014079e-13. Other types0; gate passes.
New preflight output is versioned separately; no old-output byte identity is
claimed under the changed termination. Logs: LAPTOP_C6_T05.txt and
LAPTOP_C6_REGRESSIONS_PREFLIGHT.txt. Pane now closed.

## 3. Prescribed beast recovery and new blocker

Owned `%264`, dedicated `/root/BF_runs/market_model_c`, frozen d9e6c003,
Julia1.12.4,16 threads, core pinning/BLAS1, existing Manifest symlink, pinned
cache/max_age_hours10^6. No package/data/grid/DB changes. Recovery uses the
same generation3962, chain seeds4961–4964, truth(.03,.01,.06,1000),517-fixture/
91-week schedule, full market line mix, initial prior and2000+3000 budget.

Root chain **4961** fails its INITIAL target (iteration0):
`joint mode did not converge in 100 iterations`, supported coordinate
`[-4.3667389598945885,-4.699137880221636,-3.0309405303003616,7.090705824646739]`.
Other chains cooperatively cancel at warmup1. Expanded CompositeException
confirms one root and three peer cancellations. @sync finishes every task.
**No retained iteration, fit binary, recovery interval or convergence table.**

Generation binary is byte-identical to both revision4 panels, SHA256
`de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1`.
This is generated-data identity, not successful fit/production reproduction.
The recovery directory contains only the generated binary,53 passing deterministic
engine gates and termination accounting. Canonical root accounting is identical
to the run-local CSV:1696decrement,1polished,1 total/max polish step,
maximum acceptedδ9.743716301157053e-13. The failed mode is not an acceptance.
These partial aborted-run counts can depend on cancellation scheduling and are
not evidence of converged chains or two complete byte-identical runs.

### Deterministic observer-only trace (no solver remedy)

After recovery stopped, `l05_newton_diagnostics.jl` / `r05_newton_diagnostic.jl`
trace ONE full filter on the SAME serialized panel at that exact coordinate.
A derivative observer returns exactly the original derivatives into unchanged
production joint_mode and records x/g/H/decrement/f(x). No sampler is restarted,
no solver/budget/tolerance/seed is changed, and failed filters remain failed.

Source: `v6_diagnostic/newton_mode_trace_summary.csv`, `newton_mode_trace.csv`,
`newton_trace_activity.csv`; full fixture **12476686**:

| Final / activity measure | Observed |
|---|---:|
| Derivative calls (limit) | 100 |
| Gradient norm | 4.9497130117024364e-5 |
| Gradient components | 1.8404678883143788e-6,4.9462900923558095e-5 |
| Decrement | 1.035864027765691e-12 |
| Unscaled Newton step infinity norm | 4.189252610998754e-8 |
| Actual movement at last call | 0.0 |
| Raw bracket min precision eigenvalue | 965.3501213272148 |
| First zero-movement call | 7 |
| Zero-movement calls (7–100) | 94 |

Calls6–100 have identical x,g,H,decrement and target value. The decrement is just
above1e-12; the step exceeds1e-10. Armijo backtracking accepts a rounded,
zero-motion candidate, never exhausting the line search. Thus revision6's
polishing branch is never entered, and the100-iteration limit eventually errors.
This is zero-motion/rounded Armijo acceptance, not non-PD curvature. The observer
prefix accepts146decrement modes; the failed fixture is excluded from counts.

**Stop:** recognising zero-motion accepted steps as stalls needs explicit review.
No such patch, raised iteration limit, relaxed decrement/verification threshold,
fixture exclusion, seed/prior/grid change or recovery retry was attempted.
The exact old-case regression remains passed; it does not certify all supported
synthetic likelihood/prediction cases.

## 4. Provenance / cleanup

Original v4_ad output was moved intact before checkout to
`/root/BF_runs/logs/market_model_c/9de98548_recovery_outputs/`, preserving its
binary/hash and avoiding untracked-file overwrite. Nothing was deleted/reset.
First-attempt v4_recovery/ remains in the checkout. Remote now detached04e67fa8;
only diagnostic code was loaded after the failed run, primary solver still d9e6c003.
Remote root accounting is a preserved tracked modification; untracked recovery/
diagnostic directories and ignored binaries are retained. No active session.

Both owned `%56` and beast `%264` are CLOSED. No unrelated pane operation.
Remote log `/root/BF_runs/logs/market_model_c/d9e6c003_recovery1.log` includes
run/error expansion/diagnostic. Local normalised captures:
BEAST_C6_RECOVERY_ATTEMPT.txt, BEAST_C6_RECOVERY_AND_DIAGNOSTIC.txt.
All returned artifacts are CSVs/captures only; no cache/data/.env/Manifest sync.

## 5. Definition of done / remaining priorities

Definition of done: all authorised gates/regression/recovery/converged fits,
measures/figures and two fresh byte-identical beast production runs. **Not met.**
C1 recovery is blocked before completion. C0/C1 both protocols/honest measures,
H1/H2, conditional theta=structure+u smoothing/toy checks, posterior noise/n/q/s
paths/ratios, static H1 gamma-state uncertainty plus hyperparameter mixing,
sampled H2 kappa, source-checked Ridall citation, actual production runner and
two-run reproduction remain unfinished. No learned parameter, recovery/convergence,
forecast-benefit, shrinkage/HA/ratings posterior figure or production claim exists.
Review zero-motion stall recognition first; then retain revision5's priority order.

## 6. C2 deferred (unchanged)

**C2 (thin-book pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads.**
Integrated1X2 derivatives remain unvalidated. Human options remain unselected:

| Option | What | Cost | Caveat |
|---|---|---|---|
| (i) exact moment matching | thin exact-likelihood quadrature moments/normaliser | slower, feasible on beast | independent order-stability gate |
| (ii) exact NUTS | original DESIGN route on all books | hours, funnel geometry | no Kalman collapse |
| (iii) actual-spread gate | re-specify settings at C1 posterior spreads | cheap | human setting/threshold decision |
| (iv) restricted thin types | e.g. O/U-only | cheap | human data-population decision |

No src/package/data/grid/scientific-threshold/DB changes, PhaseD/later score-grid
C2, force-push/merge/rebase/stash or unrelated pane operations. Hashes identify
blocked evidence, not completed fit/production reproduction.
