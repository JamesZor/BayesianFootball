# Phase C, revision 4: run the full-book rungs; defer thin-book pooling (C2) to the human

Manager decision, 2026-10-06 night. **Scope is reduced, not expanded. No threshold is changed.**

**Evidence** (`results/C/PHASE_C_REPORT.md`, `laplace_gate.csv`, commit `ce9b727d`):
- **Full books:** the joint-mode Laplace with the revision 3 skewness-corrected mean **passes Gate 1
  at all 12 settings** (marginal, mean and SD).
- **Thin books fail only at prediction spread 0.20**, all of them on curved likelihood ridges:
  - BTTS + O/U: median 0.02–0.07 nats; SD error up to 49%;
  - O/U-only: SD error 5.9–6.8%;
  - the 1X2-only level-integrated derivative check is incomplete.
- At spread 0.05 every type passes.

## What to run (all of it on full books only; no thin books anywhere)

| Rung | As specified in |
|---|---|
| C0 two-stage control (quality/style GRW, Gaussian σ_obs, isolated rates) | revision 1 §2.3 |
| **C1** pooled measurement model, Laplace book likelihood with learned n and σ_u, full books only | revisions 1–3 (exact Dirichlet, joint-mode Laplace marginal, skewness-corrected mean, information-form update) |
| H1 split home advantage; H2 quality-linked home advantage (two-stage, against C0) | revision 1 §2.4 (the human's request) |

**Gates:** revision 1 gates 2–5 for these rungs, and revision 2/3 Gate 1 **for full books only**
(it already passes; regenerate it in the production run). **Thresholds are unchanged.**

**Measures:** revision 1 §3, measures 1, 3 and 4 for C0, C1, H1 and H2, scored on the 517 full
books. **Measure 2 (shrinkage by book type) applies to full books only.** The 1X2-only and
totals-only special cases are C2 and are deferred.

**Tests:** move the thin-book checks (the 1X2-integrated derivative and the thin-book Gate 1 cases)
into a clearly labelled `C2-pending` testset. It runs and reports, but it is excluded from
acceptance and marked as such. Every other t05 check, and t02–t04, must pass.

**C2 is deferred, not dropped.** Put a section in the report and the README: "C2 (thin-book
pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads".
Include this table of options for the human, without choosing:

| Option | What | Cost | Caveat |
|---|---|---|---|
| (i) exact moment matching (assumed-density filtering) | per-book quadrature of the exact likelihood × prediction: mean, covariance and normaliser, for the ~114 thin books only | slower sampling; feasible on the beast with parallel chains | exact up to quadrature order; needs its own order-stability gate |
| (ii) NUTS on the exact Dirichlet likelihood | the original DESIGN §4.4 route: all books, no approximation | hours of sampling; harder geometry (σ, n, u funnels) | exact; no Kalman collapse |
| (iii) gate at the realistic spread only | re-specify Gate 1 at the C1 posterior's actual prediction spreads | cheap | **a threshold or setting change: the human's call** |
| (iv) restrict thin-book types | e.g. O/U-only books only (they nearly pass), dropping the ~11 BTTS + O/U books | cheap | changes which data are used: the human's call |

**Sentinels:** `PHASEC4_DONE` / `PHASEC4_BLOCKED` / `PHASEC4_HANDOVER`.

**Reproduction:** both full beast runs are byte-identical, as revision 1 §5 requires, restricted to
the implemented rungs.
