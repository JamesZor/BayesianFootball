# TODO 039 — Phase C revision 5: exact-mode regression BLOCKED (2026-10-06)

**Not complete.** The manager-authorised Newton termination rule is implemented,
but its mandatory exact-stall mode and Laplace-marginal verification fails.
No recovery sampler was restarted and no posterior fit was promoted. Full-book
C0/C1/H1/H2 remains authorised; C2 remains separately deferred.

Statistical source **120ac906922e7be7ea175ba9763920be4b875085**, identified by
`SOURCE_SHA256SUMS.txt`. Base was `cd1a3a3f` (revision5 brief merged).
Revision4 report/repro/handover and hashes are preserved as
`PHASE_C_V4_REPORT.md`, `REPRODUCIBILITY_V4.md`, `HANDOVER_V4.md`,
`SOURCE_V4_SHA256SUMS.txt`. Earlier checkpoints and A/B/B2/B3 are unchanged.

## 1. Authorised implementation

Only termination changes in `l05_pooled.jl:joint_mode`:

- Compute unscaled Newton direction with the existing search eigenvalue floor
  1e-6; decrement is g' H^-1 g / 2.
- Accept `decrement` at <=1e-12 nats, or `step` at infinity norm <=1e-10.
- On Armijo exhaustion accept `stalled_converged` only at decrement <=1e-9;
  otherwise propagate the numerical error. Every acceptance enforces finite,
  nonnegative decrement <=1e-9. No posterior rejection/fallback is substituted.
- Return the original raw precision. Density, prior, derivative dispatch,
  likelihood clipping, Laplace marginal, skewness correction, quadrature,
  scientific thresholds, data, seed schedule and grid remain unchanged.

Thread-safe accounting records the three counts and maximum accepted decrement.
Counts/max commute across chains. Recovery fitting writes accounting in `finally`
after `@sync` has waited for cancelled peers; the runner also writes the canonical
`results/C/newton_termination.csv`. The real preflight now uses `v5_preflight/`
and writes accounting even on failure. Neither runner was executed in this session.

Optional `mode_solver`/`mode_audit` hooks support **test-only** replay of original
termination to capture the exact stalled book before any subsequent state update.
Production defaults to the authorised solver. The legacy diagnostic is not an
alternative sampler and does not accept or promote its stalled mode.

## 2. Exact frozen-case evidence and blocker

`t05_newton_regression_tests.jl` regenerates the full unchanged 517-fixture panel,
seed3962, truth(.03,.01,.06,1000), then replays the original forward prefix at
seed4964's recorded warmup8 coordinate:

`[-3.549527585137839,-4.460929121755582,-2.821549571263347,7.9567722491577495]`.

Its first stalled book is **12476625**. Original gradient is exactly the handover's
`[-1.5699131339808048e-5,3.249019587192592e-5]`; the gradient-match assertion passes.
The rule is tested from the same prediction and exact synthetic book.

The independent tighter path starts at the accepted mode, uses analytic Gaussian
prior derivatives plus exact AD likelihood derivatives, and applies half Newton
steps until the unscaled displacement is <=1e-13. It never uses Armijo density
comparisons. This is verification, not a production numerical change.

Source: `v5_newton/newton_regression.csv` (unrounded quantities):

| Measure | Observed | Requirement | Result |
|---|---:|---:|---|
| Accepted decrement | 3.382117013018548e-12 | <=1e-9 | Pass |
| Termination | stalled_converged | recorded | Recorded |
| Mode infinity-norm difference | 4.0159674463691175e-7 | <=1e-8 | **Fail** |
| Laplace marginal absolute difference, nats | 6.658597637709818e-7 | <=1e-9 | **Fail** |
| Tighter-path gradient infinity norm | 7.670308832530282e-12 | <=1e-10 diagnostic | Pass |
| Raw/differentiable density algebra difference at reference | -1.1368683772161603e-13 | measured | Negligible here |
| Accepted-minus-tighter density | 8.773426429797837e-12 | measured | Roundoff-sized |
| Accepted-minus-tighter logdet precision | 1.3317370690657526e-6 | measured | Dominates marginal gap |

Accepted mode is `[-0.6022593822678554,2.18268468005322]`; tighter mode is
`[-0.6022597838646,2.1826846941964675]`. The CSV also records n, prediction and
covariance. `newton_regression_book.csv` saves every selection/logp of this
identified book; no books were selected or changed to make recovery pass.

**Diagnosis:** roundoff-level objective gain does not establish the prescribed
coordinate or marginal agreement. The decrement controls a quadratic objective
gain locally; precision/logdet can change to first order in displacement. Here
almost the whole marginal difference comes from the logdet term. The claimed
implication from a <=1e-9-nat decrement to <=1e-9-nat Laplace-marginal agreement
is not supported by this exact case. The reference is itself well resolved and
its primal-density algebra agrees; this is not a missing/NaN mode diagnosis.

Focused and fresh aggregate tests fail these same two checks with identical
numbers. **Stop without a third numerical variation.** Do not change the rule,
polish accepted modes in production, relax either verification gate, change the
seed/prior/data/grid or start recovery until the manager reviews this incompatibility.

## 3. Fresh verification and accounting

Owned laptop panes `%54` (focused development) and `%55` (fresh aggregate),
Julia1.12.1, 8 threads, BLAS1, existing Manifest and pinned cache. Both now closed.
No beast pane was created. Cache SHA256:
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.

| Check | Result |
|---|---|
| Focused exact-case regression | 5 pass / 2 fail / 7; original development log |
| Fresh aggregate t05 | **282 pass / 2 fail / 284**, no acceptance marker |
| Inherited full-book checks within aggregate | All original270 pass; state gates53/53 |
| New stopping/accounting unit checks | 6/6 (all termination types and fail-loud branch) |
| New exact-case regression within aggregate | 6 pass / 2 fail / 8; added reference residual check |
| Separate C2-pending | 29 pass / 10 fail / 39, explicitly excluded; no C2 investigation |
| t04 | 92/92, T04_DONE |
| t03 | 131/131, T03_DONE; A regression uses temporary outputs |
| t02 | 131/131, T02_DONE |

Logs: `LAPTOP_C5_NEWTON_DEVELOPMENT.txt`, `LAPTOP_C5_FINAL_T05.txt`,
`LAPTOP_C5_FINAL_TESTS.txt`. The same-expression `include`/`finally` REPL command
initially hit Julia1.12 world age when calling the newly loaded accounting helper.
A separate command, after include returned, wrote both accounting CSVs correctly;
that harness failure and correction are retained. No solver rerun was needed.

Canonical and `v5_newton/newton_termination.csv` record **two** actual regression
solver calls: decrement0, step0, stalled_converged2, maximum accepted decrement
3.382117013018548e-12, gate_pass=true. Unit mock counts were reset before the
regression. These are labelled `exact_regression`, **not real/recovery/production**
counts; the legacy prefix does not use the new solver or enter these counts.

Revision4's full Gate1 12/12 and six byte-identical real preflight CSVs remain
historical. They were **not regenerated** under revision5. Source/scientific
hashes and tracker/diff checks identify this blocked checkpoint, not completed fits.

## 4. Definition of done / priority status

Definition of done: all authorised full-book rungs and gates, exact-failure
regression, recovery/converged fits, prescribed measures/figures, and two fresh
byte-identical beast production runs. **Not met.**

| Priority / acceptance item | Status |
|---|---|
| Authorised stopping rule/accounting | Implemented; accounting verified |
| Mandatory exact-stall mode/marginal regression | **BLOCKED** |
| C1 recovery | Not restarted; revision4 aborted evidence retained |
| C0/C1 10a/10b fits and honest measures | Not completed |
| H1/H2 10a/10b fits and uncertainty-integrated HA measures | Not completed |
| Conditional theta=structure+u smoothing, noise/n, q/s paths and figures | Not completed |
| Production runner, full Gate1 regeneration, two byte-identical runs | Not completed |

No learned n/sigma_u, recovery interval, convergence conclusion, forecast benefit,
shrinkage, posterior HA/ratings figure or production reproducibility claim exists.
Ridall source pages remain unchecked; no new library citation is claimed.

## 5. C2 deferred to the human (unchanged)

**C2 (thin-book pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads.**
The integrated1X2 derivative check remains unvalidated. Original evidence and
settings stay unchanged. These human options remain unselected:

| Option | What | Cost | Caveat |
|---|---|---|---|
| (i) exact moment matching | thin-book exact likelihood quadrature moments/normaliser | slower sampling, feasible on beast | independent order-stability gate |
| (ii) NUTS on exact likelihood | original DESIGN route, all books | hours, funnel geometry | no Kalman collapse |
| (iii) gate at actual prediction spread | re-specify settings at C1 posterior spreads | cheap | human setting/threshold decision |
| (iv) restrict thin types | e.g. O/U-only | cheap | human data-population decision |

No src/package/data/grid/scientific-threshold changes, DB writes, PhaseD/later
score-grid C2, force-push/merge/rebase/stash, or unrelated pane operation. Beast
checkout/logs/binaries were inspected read-only and remain at revision4 source.
