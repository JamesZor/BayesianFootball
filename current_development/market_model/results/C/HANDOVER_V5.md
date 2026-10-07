# Phase C revision 5 — exact-stall accuracy BLOCKED (2026-10-06)

## State and stop boundary

Worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`,
branch `pi/market-model-phase-c`. **Full-book C0/C1/H1/H2 remains authorised;
do not ask again for scope approval. C2 remains separately deferred.**

Revision5 Newton termination/accounting is implemented at statistical source
**120ac906922e7be7ea175ba9763920be4b875085**, frozen in SOURCE_SHA256SUMS.txt.
Subsequent docs-only commit does not change that implementation.
**The required exact-failure regression fails twice identically. No third
numerical variation or recovery launch was attempted.** TODO039 is BLOCKED on
termination/accuracy compatibility, not scope approval. Read the numbers before
rerunning. Do not silently add a production polishing/fallback step, weaken the
verification gates, change seeds/prior/data/grid or select fixtures to pass.

Definition of done: authorised gates/regression/recovery/converged fits, measures,
figures and two fresh byte-identical full beast production runs. **Not met.**
No completed recovery/production fit, retained recovery intervals, convergence,
forecast evaluation, learned n/sigma_u, shrinkage/HA/ratings posterior figures
or production reproduction exists. r05_pooled remains a deterministic preflight.

Owned laptop `%54` (focused) and `%55` (fresh aggregate/regressions) are CLOSED.
No beast pane was created. All previous owned laptop/beast panes remain closed.
No running session/recovery to resume; do not operate unrelated panes.

Read AGENTS, Julia/runner guides, manager revisions1–5 under `experiments/`,
stream README/DESIGN, PHASE_C_REPORT.md and REPRODUCIBILITY.md before changes.
Revision4 documents and source hashes are preserved as *_V4*; engine checkpoint
and V1/V2/V3 evidence and all A/B/B2/B3 artifacts remain unchanged.

## Implementation / accounting

`joint_mode` now computes δ=g' H^-1 g/2 using the EXISTING1e-6 search eigenvalue
floor and unscaled direction. Accepts `decrement` atδ<=1e-12, `step` at
infinity-norm<=1e-10, or `stalled_converged` at Armijo exhaustion only ifδ<=1e-9.
All accepted modes enforce finite0<=δ<=1e-9; otherwise errors propagate.
Raw precision, exact density/normalisers, prior, derivatives, likelihood clipping,
Laplace marginal, skewness correction and quadrature are unchanged.

Thread-safe counts/max; reset before runs, finally-write after all chain tasks
finish/cancel. `fit_fullbook` optionally writes run-local accounting; recovery
runner passes the output directory and writes the root canonical CSV too.
Real preflight writes into fresh `v5_preflight/`, preserving historical v4.
Neither runner was executed in this session.

Test-only `mode_solver`/`mode_audit` hooks let the old termination replay retain
its exact original forward prefix until FIRST stall, before a state update.
The legacy diagnostic never enters sampling/promotes a stalled mode. New test
`t05_newton_regression_tests.jl` is within **accepted-boundary** t05, not pending C2.

## Exact blocker and evidence

Frozen generation3962, chain4964 warmup8 coordinate from revision4:
`[-3.549527585137839,-4.460929121755582,-2.821549571263347,7.9567722491577495]`.
Original replay identifies full fixture **12476625** and exactly reproduces
`[-1.5699131339808048e-5,3.249019587192592e-5]` gradient.

Authorised rule from the SAME prediction/book accepts `stalled_converged`:

- δ **3.382117013018548e-12** nats, passing1e-9 accounting gate;
- accepted mode `[-0.6022593822678554,2.18268468005322]`;
- tighter damped Newton `[-0.6022597838646,2.1826846941964675]`;
- mode infinity-norm difference **4.0159674463691175e-7**, limit1e-8: **FAIL**;
- Laplace marginal difference **6.658597637709818e-7** nats, limit1e-9: **FAIL**;
- tighter gradient residual **7.670308832530282e-12**, diagnostic limit1e-10: pass;
- raw/differentiable density discrepancy **-1.1368683772161603e-13**;
- logdet precision shift **1.3317370690657526e-6**, density difference only
  **8.773426429797837e-12**. The logdet shift dominates marginal error.

The tighter path starts at the accepted point, uses analytic Gaussian prior
and exact AD likelihood derivatives, half Newton steps until displacement<=1e-13,
no Armijo comparisons. It is a verification path ONLY. Small decrement measures
quadratic objective gain, not first-order coordinate/Hessian/logdet displacement;
the manager's sufficient-accuracy implication does not hold for this exact case.

`v5_newton/newton_regression.csv` records all numbers, prediction/covariance/n.
`newton_regression_book.csv` records every market selection/logp. Root and
`v5_newton/newton_termination.csv` record TWO actual regression solves:
decrement0/step0/stalled_converged2, maxδ above, gate_pass=true. These are
`exact_regression`, NOT recovery/production; unit mocks are excluded by reset.

Focused `%54`: 5pass/2fail/7. Fresh `%55` aggregate: **282pass/2fail/284**;
all original270 and new stopping/accounting6 pass, exact regression6pass/2fail.
Same two numerical failures, identical values. **No third numerical variation.**
Fresh t04 **92/92**, t03 **131/131**, t02 **131/131**, their markers. Separate
C2-pending unchanged **29pass/10fail/39**, C2_PENDING_REPORTED. No T05_DONE,
T05_C5_DETERMINISTIC_DONE, R05_DONE or recovery-DONE marker.

Logs: LAPTOP_C5_NEWTON_DEVELOPMENT.txt, LAPTOP_C5_FINAL_T05.txt,
LAPTOP_C5_FINAL_TESTS.txt. Initial REPL accounting command after same-expression
include hit Julia1.12 world age; separate post-include command writes both CSVs
correctly, without rerunning the solver. Logs retain that harness failure.

Revision4 real Gate1/preflight remains historical; **not regenerated at revision5**.
No fresh two-run scientific identity claimed. Hashes describe blocked evidence only.

## Remote unchanged

Read-only beast status/load check: `/root/BF_runs/market_model_c` remains detached
at **9de98548**, existing Manifest/cache unchanged, historical untracked recovery
outputs preserved. No checkout, remote source edit, rsync, reset/delete or new pane.
Original logs/binary locations/hash are in HANDOVER_V4.md and REPRODUCIBILITY_V4.md.
The two synthetic binary hashes remain
`de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1` (not fits).

## Next dependency / priority work

**Review stopping/accuracy compatibility first; don't rerun a third solver variation.**
Any production polishing/Armijo change beyond termination needs explicit review.
Do not waive the1e-8 mode/1e-9 marginal regression merely because Gate1's broader
0.01-nat budget is much looser. Full-book scope is already approved.

After a reviewed remedy passes exact-case regression: C1 recovery; C0/C1 both
protocol fits plus honest10b measures; H1/H2; measures/figures/report and second
fresh byte-identical beast run. All prior outstanding work is still required:
all-parameter Rhat<=1.05, bulk/tailESS>=200; B2-compatible pre-week scoring with
no own-book forecast mean or extra evaluation-book variance; sum of supremacy+
level marginal logpd paired against C0 with fixtureSE; measuredR6gap;
conditional theta=structure+u smoothing/toy gates; noise/n/shrinkage and q/s
paths/ratios; H1 static gamma-state uncertainty plus hyperparameter mixing;
H2 sampled kappa; check Ridall source before citation; actual production runner
sections and full Gate1 regeneration. C2 stays deferred, not silently dropped.

Pinned cache SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
max_age_hours10^6. No one-shot Julia, package/data/grid/scientific-threshold/src/DB
changes, PhaseD/later score-grid C2, force-push/merge/rebase/stash. Only permitted
push is origin pi/market-model-phase-c. Before handoff/commit: diff --check,
todo.sh check, source/scientific hash verification; tracker and README row together.

PHASEC5_BLOCKED
