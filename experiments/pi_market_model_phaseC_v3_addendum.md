# Phase C, revision 3 addendum: a skewness-corrected mean (manager, 2026-10-06 night)

This supersedes revision 2's posterior-**mean** treatment only. Everything else in revisions 1
and 2 stands, and **every threshold is unchanged**.

**Status after revision 2** (`5c325578`/`7bb4b99d`), first 30 full books:
- the marginal passes everywhere, with median error ≤ 0.004 nats;
- the SD error is ≤ 1.02%;
- **the posterior-mean gate fails** at n = 250 and spread 0.20 (max 9.17% of SD, 12/30 books)
  and at n = 1000 and spread 0.20 (5.6%, 2/30).

This is the Laplace mode-versus-mean gap from posterior skewness. The fix is to improve the
method, not to relax the gate.

**The change: a second-order (skewness-corrected) Laplace mean.** Let f(θ) = log L_m(θ) +
log N(θ; a_m, S_m), with mode θ*, Σ = (−∇²f(θ*))⁻¹ and third derivatives T_ijk = ∂³f(θ*). Use

```
E[θ_i] ≈ θ*_i + ½ Σ_j Σ_kl Σ_ij T_jkl Σ_kl
```

This is the standard first-order correction of the mode toward the mean. It needs the third
derivatives of f only: the Gaussian part has none, so T comes from log L_m alone. Get them by
AD (ForwardDiff over the existing Hessian) or by central differences of the Hessian, and test
both against each other.
- Use the corrected mean as the pseudo-observation's centre in the information-form update.
  Keep J_m and Σ as before (the SD gate already passes).
- 1X2-only books: the 1-D version on d.
- **The log-marginal stays the joint-mode Laplace of revision 2.** It passes; don't change it.
- Document the formula, with its derivation or a standard reference, in the `l05` docstring.

**Then:**
1. Rerun the **full** Gate 1, at the same settings and thresholds, and on **all** book types:
   full, totals/BTTS-only (O/U-only and BTTS + O/U separately) and 1X2-only (level-integrated).
   Revision 2 stopped at full books; the thin types are untested.
2. **If everything passes,** continue Phase C as revisions 1 and 2 specify: the rungs, the gates,
   sampling on the beast, the measures, a second byte-identical run, the report and the README.
3. **If any type still fails,** stop with the numbers and a diagnosis. Don't relax thresholds,
   prediction spreads, n or the books.

Sentinels for this continuation: **`PHASEC3_DONE` / `PHASEC3_BLOCKED` / `PHASEC3_HANDOVER`**,
printed on their own line. The handover rule at ~75% context still applies.
