# Phase C revision6 — exact-case PASS, recovery iteration-limit BLOCKED (2026-10-06)

**Not complete.** Manager revision6 explicitly remedies revision5's inconsistent
acceptance rule. Brief is copied unchanged into experiments/. All verification
and scientific tolerances/settings/seeds remain unchanged. Full-book C0/C1/H1/H2
remains authorised, C2 separately deferred. Revision5 report/repro/handover and
source hashes are preserved with `_V5` names; all earlier evidence is unchanged.

## Recovery stop boundary

At frozen source d9e6c003, owned beast %264, prescribed recovery fails its
initial chain4961 target: joint mode did not converge in100 iterations.
Peers cancelled at warmup1; no retained fit or interval exists. Root coordinate
[-4.3667389598945885,-4.699137880221636,-3.0309405303003616,7.090705824646739].
Stop without another solver variation or changed tolerances. An observer-only
trace of this exact supported coordinate will report fixture/g/decrement; it is
not a sampler restart or remedy. Generation binary is byte-identical to revision4.

## Implementation

Only on an Armijo stall withδ<=1e-9: take up to3 undamped, derivative-only Newton
steps, stopping at step infinity norm<=1e-12 or nondecreasing gradient norm.
Final gradient norm must not exceed the pre-polish norm; final acceptedδ must
still pass the existing1e-9 gate. Otherwise errors propagate, never -Inf/fallback.
Record solves as `polished` and count total/max polish steps. Existing search
floor, raw precision, density, derivatives, likelihood clipping, marginal,
mean correction, quadrature, priors and population remain unchanged.

## Fresh exact-case verification

Source: `v6_newton/newton_regression.csv`; fixture12476625 at the unchanged
seed4964 warmup8 coordinate, generation3962. Original forward-prefix replay
still exactly reproduces the revision4 handover gradient. The same tighter
half-Newton reference is used; its tolerance and both acceptance gates are unchanged.

| Measure | Result | Limit |
|---|---:|---:|
| Mode infinity-norm difference | 2.0039525594484076e-13 | 1e-8 |
| Laplace marginal difference, nats | 1.1574741165532032e-11 | 1e-9 |
| Accepted decrement | 1.50342686280365e-24 | 1e-9 |
| Polish steps | 1 | 3 |
| Tighter-reference gradient residual | 5.340616837656853e-12 | 1e-10 diagnostic |

All pass. `v6_newton/newton_termination.csv` records2 actual regression solves,
both polished in1 step. Unit mocks were reset before these counters.

Owned laptop `%56`, Julia1.12.1,8 threads/BLAS1, pinned cache and existing Manifest:
- t05 **289/289**, T05_C6_DETERMINISTIC_DONE, not complete recovery acceptance;
- C2-pending unchanged29pass/10fail/39, excluded and explicitly reported;
- t04 **92/92**, t03/t02 **131/131**, their markers;
- real full Gate1 **12/12**,360 fixture-setting rows; state gates **53/53**;
- all four517-fixture fixed filters/smoothers complete, not posterior fits.

New real preflight artifacts are under `v6_preflight/`, with termination accounting.
Historical v4/v5 evidence is not replaced or claimed identical under the changed
termination rule. Logs: LAPTOP_C6_T05.txt and LAPTOP_C6_REGRESSIONS_PREFLIGHT.txt.
Scientific source is frozen by SOURCE_SHA256SUMS.txt before prescribed beast
recovery. No new recovery fit or posterior scientific claim exists at this checkpoint.

## Remaining priorities / definition of done

Definition of done remains all authorised gates/recovery/converged fits, prescribed
measures/figures and two fresh byte-identical full beast production runs: **not met**.
Next: C1 recovery; then C0/C1 both protocols with honest10b measures; H1/H2;
measures/figures/report and second reproduction run. Production training/evaluation,
conditional theta=structure+u smoothing, uncertainty-integrated static H1 gamma
states, sampled H2 kappa, noise/n/q/s summaries and source-checked Ridall citation
remain outstanding. No retained recovery intervals, learned parameters, convergence,
forecast benefit, posterior figures or production reproduction is claimed.

**C2 (thin-book pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads.**
Its unchanged human options (quadrature moment matching, exact NUTS, revised spread
gate, or restricted thin types) remain unselected and are tabulated in the README.
No src/package/data/grid/scientific-threshold/DB changes, PhaseD/later score-grid C2,
force-push/merge/rebase/stash or unrelated pane operations.
