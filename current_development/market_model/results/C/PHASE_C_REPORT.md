# TODO 039 — Phase C revision 3 BLOCKED (2026-10-06)

**No pooled sampling or inference promotion.** The authorised cubic correction
fixes the full-book mean gap, but newly tested thin books fail the unchanged
accuracy gates. The level-integrated 1X2 derivative check also fails twice;
work stops rather than trying a third variation. **The all-types Gate 1 attempt
is incomplete, not passed:** four 2-D types completed; 1X2-only did not.

Implementation: `8591099b`. The manager's addendum was read and copied unchanged
to `experiments/pi_market_model_phaseC_v3_addendum.md`.
Double Poisson for C; later score-grid Phase C2 and Phase D remain unauthorised.
Revision-1 evidence is unchanged. Revision-2 report/notes are preserved as
`PHASE_C_V2_REPORT.md` / `REPRODUCIBILITY_V2.md`, with its four gate CSVs in
`v2_gate/`. No `src/`, packages, data, databases, thresholds, n settings,
prediction spreads or book subsets were changed.

## 1. Authorised mean correction

`l05_pooled.jl` implements the specified contraction:

```
Sigma = (-Hessian(log L + log N) at mode)^-1
shift_i = 0.5 sum_jkl Sigma_ij T_jkl Sigma_kl
corrected_mean = mode + shift
```

T is the third derivative of **log L only**. The loader docstring derives the
formula by expanding the cubic exponential and applying Wick's fourth-moment
identity; the three symmetric contractions turn 1/6 into 1/2. The odd cubic
term has zero Gaussian expectation, so the revision-2 marginal is unchanged.
The corrected Gaussian's information vector is precision times corrected mean;
subtract prior information to define the book pseudo-observation. This avoids
shrinking the mean correction a second time. J, clipping and covariance stay
as before. No quadrature-moment replacement or alternate likelihood was added.

ForwardDiff differentiates the identical renormalised cutoff-10 double-Poisson
algebra. Production's common exponential factors cancel in its normalisation;
a generic recurrence permits third derivatives without a `src/` change. Tests
on the first 20 full books verify probability/density parity, Hessian finite
differences and AD versus central differences of the AD Hessian. The raw
production density remains the 2-D integration/marginal reference. This is
AD-compatible evaluation of the same grid, not a new grid or tail convention.

The 1X2 level integral was attempted with differentiable quadrature. Its
current adaptive Gauss-Kronrod implementation calls the **already installed**
`Distributions.quadgk`, with no new package. Integration bounds are [-8,4],
endpoint diagnostics retained; orders labelled 32/64 select Gauss orders 7/15
for that inner integral. The outer 2-D quadrature remains Gauss-Hermite 32/64.
The integrated third-derivative consistency check remains red; this code is
**unvalidated**, not a completed supremacy-only likelihood pipeline.

## 2. Measured gate results

Sources: `laplace_gate.csv`, `laplace_gate_fixture.csv`, `book_inventory_c.csv`.
All prescribed n=250/1000/4000, equal-side prediction SD=0.05/0.20, offsets
0/+0.10 were retained. Full-book centres are Phase A isolated rates. Books
without an isolated rate use the documented fixed-parameter C0 pre-week
prediction, at B2 R6 quality/style rotation medians, updating only on full-book
isolated rates. This is a realistic fixed-parameter centre, **not a newly fitted
C0 posterior**. Phase A inversions remain the future scoring targets.

| Type | Available | Completed books | Completed settings | Failed settings |
|---|---:|---:|---:|---:|
| Full | 517 | first 30 | 12 | 0 |
| O/U-only | 25 | all 25 | 12 | 3 |
| BTTS+O/U | 10 | all 10 | 12 | 6 |
| BTTS-only | 1 | 1 | 12 | 0 |
| 1X2-only | 78 | incomplete | incomplete | not assessed |

Selection is match-ID order, not error-dependent. There are **792** completed
fixture-setting rows. `engine_gates_c.csv` records the four scalar marginal/mean/
SD checks for each of the **48** completed settings: **167/192 pass, 25 fail**.
It does not include unexecuted pooled-engine/convergence gates or invent passing
1X2 rows. After the attempted runner aborted, this table was regenerated from
its already-written summary CSV; it is not evidence of top-to-bottom success.
The inventory's 1X2 tested=0 denotes no completed type batch written, not a
claim that no 1X2 evaluations were attempted.

### Full books: correction succeeds, marginal/covariance unchanged

Across all 12 settings the maximum mean error / exact posterior SD is
**0.0009284497092163062**, about **0.093%** (limit 5%), down from revision 2's
9.17% worst error. Maximum SD relative error remains
**0.010184310576972222**, about 1.02%. All full-book settings now pass.

Joining all **360** full-book rows with `v2_gate/laplace_gate_fixture.csv` on
fixture/n/spread/offset gives maximum change in the saved log marginal **0.0**.
The largest full-book median absolute marginal error is still
**0.003980333011027959 nats**; largest p95 is **0.0051705855019272395**.
Thus the mean was corrected without replacing the passing marginal or covariance.

### Stable thin-book failures already block promotion

The following no-offset SD=0.20 settings fail. All numbers are copied from
`laplace_gate.csv`; the offset cases are retained there too.

| Type | n | Median absolute marginal error | p95 | Maximum mean error / SD | Maximum relative SD error |
|---|---:|---:|---:|---:|---:|
| O/U-only | 250 | 0.002794 | 0.006559 | 0.005291 | **0.058978** |
| O/U-only | 1000 | 0.003853 | 0.007461 | 0.005733 | **0.068271** |
| O/U-only | 4000 | 0.004542 | 0.008859 | 0.010854 | **0.062199** |
| BTTS+O/U | 250 | **0.020609** | 0.039342 | **0.058362** | **0.148782** |
| BTTS+O/U | 1000 | **0.038844** | **0.096802** | **0.096360** | **0.218257** |
| BTTS+O/U | 4000 | **0.073372** | **0.161919** | **0.299363** | **0.493985** |

Unchanged limits are median <=0.01 nats, p95 <=0.05 nats, and mean/SD errors
<=0.05 per fixture. The **n=250** witnesses are well resolved independently:

- O/U-only, SD=.20, no offset: maximum order-32/64 marginal difference
  **2.2578857006294584e-8 nats**, mean/SD order difference
  **1.827533694049216e-7**, relative SD order difference
  **7.289597666604308e-7**.
- BTTS+O/U, same setting: maximum marginal order difference
  **1.7352243730783812e-7 nats**, mean/SD difference
  **1.6853123070507956e-6**, SD difference **8.079214087408637e-6**.

These resolution errors are far below the gate failures. High-n thin-book
quadrature is less stable: O/U-only n=4000 has a maximum marginal order
difference **0.0029324648804998077 nats**. No claim of complete high-n
quadrature resolution is made. The well-resolved n=250 failures alone suffice
to stop; unresolved settings were not removed or substituted.

**O/U witness:** fixture **14035540**, n=250, SD=.20, no offset has relative
SD error **0.058978186176263714**, versus 0.05. Its raw/unclipped Gaussian
SD error is **0.03898563692037171** and likelihood minimum eigenvalue
**-1.2364093919265855**. The stipulated clipping is active. A curved flat
constant-total contour need not give a positive observed Hessian at a
prior-regularised mode; clipping changes its local posterior covariance.
Revision 3 changes means only and cannot cure that covariance discrepancy.
No unclipped substitute was promoted.

**BTTS+O/U witness:** fixture **14035709**, n=250, SD=.20, no offset has
marginal error **-0.048158097212724016 nats**, mean error/SD
**0.05836204173970596**, and SD relative error **0.14878198870219494**.
Its minimum J eigenvalue is **40.28420140644717**, so clipping is inactive.
The raw SD error is the same. Symmetric BTTS+totals prices constrain the
sum/product of the rates and permit swapped solutions. Weak, curved and
potentially multimodal supremacy information is not reliably summarised by
one mode, its Hessian and one cubic correction. This geometry explains why
full books can pass while these thin books fail; it is not a reason to force
them rank one or change the likelihood.

## 3. 1X2-only is incomplete: numerical check fails twice

The first all-types gate attempt completed all four 2-D types, then the
level-integrated 1X2 Newton line search stalled with reported score
**-0.06441733688240525**. No complete 1X2 batch was written. This is a
numerical failure, not a measured passing or failing 1X2 moment distribution.

The moving Gaussian level node mesh was then replaced once with adaptive
Gauss-Kronrod. The mandated AD-versus-finite-difference check on the
level-integrated likelihood, using fixture **12473328**'s 1X2 market, failed:

| Attempt | Relative third-derivative discrepancy | Test tolerance |
|---|---:|---:|
| Adaptive quadrature | 5.405790669565064e-5 | 1e-6 |
| Adaptive quadrature, all-jet error norm | 5.405838441375254e-5 | 1e-6 |

Source: `integrated_derivative_check_c3.csv`; full output is in
`LAPTOP_REVISION3_DEVELOPMENT.txt`. The second attempt explicitly controls all
nested dual-number components, not only integrand values, but does not resolve
the discrepancy. The test fails in the same way twice, so **no third variation
or additional solver attempt was made**. The precise integrated-derivative
numerical cause remains unresolved; do not treat the AD third jet as validated.
Neither a larger test tolerance nor discarded thin books was used to proceed.

## 4. Tests and acceptance — explicitly incomplete

Owned laptop pane **%43**, Julia 1.12.1, existing pinned environment, 8 threads,
BLAS=1; closed after capture. No beast session or sampling was created.

The preliminary Gaussian-level quadrature version passed **130/130** necessary
unit assertions after its endpoint range was widened from 2.5 to 4.0. That
version subsequently stalled in the real 1X2 gate. The **current adaptive
implementation** has **129 passes / 1 failure / 130 assertions**: all 20-book
production parity/Hessian/third-tensor checks pass, but integrated third-jet
consistency fails. These development tests are not a fresh complete pooled
acceptance suite, and no T05_DONE or R05_DONE is claimed.

| Acceptance item | Evidence/status |
|---|---|
| Full t05 and t02–t04 in fresh laptop REPL | **Not met.** Current t05 necessary checks fail 1/130; full recovery suite not built. Prior suites were not rerun for revision 3 (revision-2 counts remain in its archived report). |
| All Gate 1 types, all other gates | **Not met.** Four types completed, thin-book accuracy fails; 1X2 incomplete. No pooled batch/projection gates promoted. |
| Convergence | **Not run.** No C0/C1/C2/H1/H2 or synthetic posterior fits. |
| Every table/figure | **Not met.** Gate diagnostics only; no forecast, shrinkage, concentration, convergence, team paths, HA tables or C figures. |
| Fresh beast R05_DONE and byte-identical second run | **Not run.** No production reproducibility claim. |
| README / DESIGN | README records revision 3; DESIGN §5 unchanged. The briefs supersede its older Turing/MAP/NUTS route; no unauthorised alternative fitted. |
| TODO 039 / checks | Dated @pi claim/blocker entries, status BLOCKED. Tracker and diff checks pass at handoff. |

## 5. Plain words and stop

The new correction does what was intended for full books. It does **not** make
one Gaussian reliable for every thin book. Means are not the only problem:
O/U curvature/clipping affects variance, and BTTS+totals has a nonlinear
supremacy shape with poor marginal and variance approximation. The supremacy-
only integrated 1X2 likelihood still has an unresolved numerical derivative
check. Production inference must not start with those requirements red.

There are still **no pooled rates, honest forecast comparisons, learned n or
sigma_u, shrinkage shares, corrected 1X2 totals, T015 pooled supremacy, ratings
or HA posterior conclusions**. No approximately +0.10 thin-total shift or
pooling forecast gain can be claimed from these local diagnostics.

The manager must decide a revised approximation that meets the unchanged
thin-book marginal/covariance gates and resolve the integrated derivative
check. Revision 3 authorises a mean correction only; no covariance replacement,
multimode sum, projection, threshold change or likelihood substitution was
silently made here. The full all-types gate and all subsequent Phase C
acceptance remain unfinished. All owned sessions are closed; only the
permitted branch is committed/pushed for review.
