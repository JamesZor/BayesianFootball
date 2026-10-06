# TODO 039 — Phase C revision 2 BLOCKED at posterior-mean gate (2026-10-06)

**No pooled fit was launched or promoted.** The corrected joint-mode expansion
passes the log-marginal gate on the first 30 full books, but fails revision 2's
posterior-mean accuracy requirement. This is a blocker report, not Phase C
completion. Statistical implementation: `7b6c7659`.

**Double Poisson for C; C2 later.** The authorised pooled rung C2 is distinct
from the later score-grid Phase C2. Neither that later phase nor Phase D was
attempted. No thresholds, data, packages, `src/` files or databases were changed.
The first-attempt report is preserved verbatim in `PHASE_C_V1_REPORT.md`; its
preflight source and CSVs remain unchanged.

## 1. Revision 2 and what was implemented

Revision 1 used the wrong KL orientation for the Dirichlet large-n limit,
compared a projected 1X2 kernel with an unprojected exact likelihood, and assumed
a fixed totals direction/rank. Revision 2 corrects these choices:

- Exact Dirichlet/Beta density, all normalisers, unchanged production grid.
- Newton joint mode of likelihood times Gaussian prediction; likelihood score
  and observed Hessian at that mode. Phase A's KL(p||q) inversion stays the
  **scoring target**, not the expansion point.
- Joint-mode Laplace integral, including its density peak and determinant;
  information-form posterior update with the prescribed eigenvalue clipping.
- The manager's logged totals/BTTS-only decision: retain the likelihood exactly.
  O/U constrains a curved total contour; BTTS+O/U retains its actual weak
  supremacy information rather than being forced to rank one.
- A flat-measure level-integrated 1X2 likelihood is implemented, but **not
  validated or promoted**. Thin-book production checks were not reached.

`../../l05_pooled.jl` documents the integral derivation and implements the
update and exact-density quadrature. `../../r05_pooled.jl` exposes configuration,
pinned data, revised Gate 1 and the fail-fast promotion decision.
`../../t05_pooled_tests.jl` contains necessary deterministic unit checks only;
it is **not the complete synthetic-recovery/acceptance suite**. No sampling,
RTS production smoother, forecast metrics, HA fits or production recovery
pipeline was built after the accuracy blocker became clear.

## 2. Fixed gate evidence

Sources: `laplace_gate.csv`, `laplace_gate_fixture.csv`, `engine_gates_c.csv`.
The first **30 full books in match-ID order**, with no selection on errors,
were evaluated at all **12** prescribed settings: n=250/1000/4000, equal-side
prediction SD=0.05/0.20, and offsets 0/+0.10 to both log-rate means. Centres
are their Phase A isolated rates, explicitly permitted by revision 2. There
are **360** fixture-setting rows. These are fixed n values, not a posterior.

Thresholds: median absolute marginal error <=0.01 nats; p95 <=0.05 nats;
each posterior mean within 0.05 of its exact posterior SD, and each SD within
5% relative error. The mean statistic is max over the two original log-rate
coordinates of |approximate mean - exact mean| / exact SD. No averaging across
fixtures conceals a failing book.

| n | Prediction SD | Offset | Median absolute marginal error | p95 | Maximum mean error / SD | Mean failures / 30 |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.05 | 0 | 0.000356 | 0.000446 | 0.016751 | 0 |
| 250 | 0.05 | 0.10 | 0.000403 | 0.000472 | 0.015890 | 0 |
| 250 | 0.20 | 0 | 0.003980 | 0.004940 | **0.091744** | **12** |
| 250 | 0.20 | 0.10 | 0.003881 | 0.005171 | **0.089202** | **11** |
| 1000 | 0.05 | 0 | 0.000461 | 0.000542 | 0.026054 | 0 |
| 1000 | 0.05 | 0.10 | 0.000453 | 0.000533 | 0.024996 | 0 |
| 1000 | 0.20 | 0 | 0.001262 | 0.002044 | **0.055696** | **2** |
| 1000 | 0.20 | 0.10 | 0.001256 | 0.002041 | **0.055150** | **2** |
| 4000 | 0.05 | 0 | 0.000251 | 0.000321 | 0.023112 | 0 |
| 4000 | 0.05 | 0.10 | 0.000244 | 0.000339 | 0.022466 | 0 |
| 4000 | 0.20 | 0 | 0.000334 | 0.000573 | 0.029351 | 0 |
| 4000 | 0.20 | 0.10 | 0.000333 | 0.000573 | 0.029271 | 0 |

All marginal and SD checks pass on this tested type; **four settings fail the
mean gate**. Maximum relative SD error is **0.010184310576972222** (about 1.02%).
The 48 scalar Gate 1 entries in `engine_gates_c.csv` have **44 passes / 4 failures**.
This file contains **only Gate 1**, not unexecuted pooled-engine gates 2–5.

### Quadrature resolution and a concrete witness

Every integrand evaluation uses the exact Dirichlet/Beta density times the
Gaussian prediction, divided by the mode-centred Gaussian proposal. The
proposal changes coordinates; it does not approximate the integrand.
Independent Gauss–Hermite orders **32/64** give maximum differences:

- log marginal: **2.6929569685307797e-12 nats**;
- posterior mean / exact SD: **3.069430883859357e-12**;
- relative posterior SD: **1.0662581928500003e-12**.

The largest mean error is fixture **12476452**, n=250, SD=0.20, no offset:

| Coordinate | Laplace mean | Exact quadrature mean | Exact posterior SD |
|---|---:|---:|---:|
| log lambda_h | 0.6542372458730691 | 0.6525518797137508 | 0.04871981120155758 |
| log lambda_a | -0.08086135763246184 | -0.08831025597956597 | 0.08119227881292301 |

Its away-side normalised mean error is **0.09174392511223031**, versus the
fixed limit 0.05. The mean-mode difference is only
**2.2982440950336525e-10**. Its likelihood precision minimum eigenvalue is
**129.6217272253511**; the minimum across all tested rows is
**51.10001386984887**. Thus clipping is inactive on these books, and the
marginal's raw-curvature versus clipped-update distinction cannot explain
this failure.

**Diagnosis:** a Gaussian expanded at the joint mode approximates the total
integral very well, yet its mean is the mode rather than the mean of the
slightly skewed exact posterior. Accurate normalising constants do not imply
accurate first moments. No second relinearisation was used: the update already
returns the converged joint mode (maximum mean-mode displacement across this
panel is **1.879189764110445e-7**). Re-expanding at that same mean is a fixed
point; it does not integrate the higher-order skewness that shifts the exact
mean. No posterior-moment correction or alternate likelihood was silently added.

### Scope of the stopped gate

`book_inventory_c.csv` retains availability, not an invented passing subset:

| Type | Available | Tested |
|---|---:|---:|
| Full | 517 | 30 |
| 1X2-only | 78 | 0 |
| O/U-only | 25 | 0 |
| BTTS+O/U | 10 | 0 |
| BTTS-only | 1 | 0 |

The clean runner stops after the failed full-book type as the brief instructs.
No claims are made about the thin-book marginal, moment or level-truncation
gates. Those remain prerequisites even if the full-book blocker is addressed.

## 3. Verification and acceptance status

Fresh owned laptop pane **%42**, Julia **1.12.1**, Distributions **0.25.126**,
8 threads, BLAS=1; pinned cache SHA matches the brief. Source `7b6c7659`.
`LAPTOP_REVISION2.txt` records:

- necessary revision-2 update tests **24/24**, **8.8 s**, `T05_UPDATE_DONE`;
- t04 **92/92**, **32.4 s**, `T04_DONE`;
- t03 **131/131**, **40.0 s**, `T03_DONE` (also `R01_DONE` in temporary output);
- t02 **131/131**, **58.8 s**, `T02_DONE`;
- fresh gate runner ends at **C05_LIKELIHOOD_BLOCKED**, not `R05_DONE`.

The unit checks cover independently known Gaussian integrals/moments in 1-D
and 2-D, exact Beta identity, the reversed leading large-n KL limit, a synthetic
narrow-spread total book, quality/style covariance algebra and an existing R6
batch likelihood. The prior-phase tests retain their existing gates/counts.
They do **not** prove a complete pooled engine or posterior parameter recovery.

| Brief acceptance item | Status |
|---|---|
| Full t05 and t02–t04 in fresh laptop REPL | **Incomplete.** Necessary t05 checks and prior suites pass; 20-book Hessian, thin-book checks and synthetic parameter recovery are not completed. |
| All revised gates | **Not met.** Full-book posterior mean gate fails. Thin-book accuracy/projection/pooled-engine gates not promoted. Prior B2/B/TODO 023 gates pass in regression tests. |
| Convergence everywhere | **Not run.** No C0/C1/C2/H1/H2 or synthetic fits. |
| Every table and figure | **Not met.** Four new gate/inventory CSVs exist. No ladder/one-step/paired/shrinkage/convergence/team-path/HA tables or C figures. |
| Fresh beast R05_DONE, second byte-identical production run | **Not run.** No beast checkout/session; no production reproducibility claim. |
| README and DESIGN mapping | README updated for revision 2 and the blocker. DESIGN §5 unchanged; the briefs replace its old Turing/static smoke/MAP-fold route with quality/style GRW and approximate collapsed slice inference. |
| TODO 039 and tracker check | Dated @pi claim/blocker entries; status BLOCKED, not closed. `todo.sh check` and `git diff --check` pass at handoff. |

## 4. Plain words: what can and cannot be concluded

- **Isolated versus pooled:** not fitted; no evidence yet about changes to full
  or thin-book rates or the share of deviations retained.
- **Honest forecasts:** no C0/C1/C2 forecast comparison exists. No pooling gain
  or loss can be claimed from this local accuracy diagnostic.
- **Book worth:** n is not estimated. Its posterior, implied book noise and
  ratio to sigma_u remain unknown.
- **1X2-only totals:** no pooled totals exist; the expected approximately +0.10
  change cannot be checked.
- **T015 supremacy:** no pooled supremacy exists; no correction of its
  isolated-inversion artefact has been demonstrated.
- **Ratings and HA:** no transition-club paths or H1/H2 posterior evidence.

Choices were in scope: mathematical gate before expensive inference; exact
existing book/grid calls; deterministic first-ID selection; independent
quadrature orders; fixed required n/spreads/offsets; clipping retained and
reported separately; fail fast by book type. No failing fixtures/settings were
removed and no threshold or prior was tuned to turn a red gate green.

## 5. Brief/design issues and development record

DESIGN §4.4 still incorrectly calls KL(p||q) the large-n Dirichlet limit; the
correct orientation is KL(q||p). Its old “totals only” statement is superseded
by the manager's explicit revision-2 decision. Its Turing/MAP/NUTS engine and
fold-refit checklist are not the new brief's collapsed-slice route. These
contradictions are documented, not edited outside the authorised contract.

Revision 2's O/U Hessian test at the **joint mode** also needs care. Even for
an untruncated grid, log L=f(T), T=exp(h)+exp(a), implies
Hessian(log L)=f''(T) grad(T)grad(T)' + f'(T) diag(exp(theta)).
A tangent vector annihilates the first term, not necessarily the second.
At the likelihood's own maximum f'=0, but at a prior-regularised joint mode
f' generally is not zero. A curved flat contour therefore does not require
a near-zero *observed Hessian* tangent eigenvalue at that joint mode. This
additional test-contract concern was not used to excuse the full-book failure.

Development pane %41 first passed the 24 unit checks. An initial runner call
exposed an incorrect assumption about B2 CSV column names (corrected after
reading the schema). A finite-difference Newton termination check was then
made displacement-aware to avoid iterations at density roundoff resolution.
The full-book rows already demonstrated the moment failure. Before fail-fast
was added, attempting the level-integrated 1X2 step exposed a noisy nested
quadrature/finite-difference line search; that code is unvalidated and must not
be treated as a passed 1X2 implementation. No further optimisation variations
were tried. The fresh fail-fast run reproduces the full-book blocker without
entering that stage. These development errors did not change any scientific
threshold, book, centre, n value or spread.

## Blocked / next action

The manager must decide how to meet the **unchanged posterior-mean accuracy
requirement** beyond a Gaussian at the joint mode. A higher-order moment
correction or quadrature-moment update would change the approximation and needs
an amended brief; no such substitute is authorised or implemented here.
A second relinearisation at the current Gaussian mean does not address this
mode-versus-mean discrepancy. Thin-book quadrature/moment tests and the rest
of Phase C remain unfinished. No claims transfer to production until those
gates and full acceptance checks pass.

All owned REPL panes (%41/%42) are closed. No other sessions were operated.
The work is committed and pushed only to `pi/market-model-phase-c` for review.
