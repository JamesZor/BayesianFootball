# Phase C, revision 5: the joint-mode Newton stopping rule (manager, 2026-10-06 ~04:20)

This is the manager's explicit go-ahead for the numerical fix that the revision 4 blocker
(`414598c4`, `results/C/HANDOVER.md`) asked for. **It changes solver termination only. No
scientific gate, threshold, seed, prior, data, grid or book selection changes.**

**Diagnosis.** In the C1 recovery the per-book joint-mode Newton fails its line search where
‖∇f‖ ≈ 3×10⁻⁵, that is, at the optimum. That is roundoff stalling Armijo, not a missing mode.

**The rule**, for f(θ) = log L_m(θ) + log N(θ; a, S) and its observed Hessian
H = −∇²f (PD after the existing eigenvalue clipping):

1. Converged when the **Newton decrement** δ = ½·gᵀH⁻¹g ≤ 1e-12 nats (the predicted log-density
   gain of a full Newton step), or ‖Δθ‖∞ ≤ 1e-10.
2. If the line search cannot find an Armijo step but **δ ≤ 1e-9**, accept θ as the mode, and
   count it as `stalled_converged`.
3. Otherwise raise an error exactly as now: no silent −Inf, no fallback.

**Accounting** for every real or recovery run: counts of each termination type (`decrement`,
`step`, `stalled_converged`), and the maximum δ at acceptance. Write them to
`results/C/newton_termination.csv`.

**Verification:**
- On the exact failing case (seed 4964, the θ recorded in the handover), re-solve the stalled
  books with a tighter path, for example BigFloat, or a damped Newton started from the accepted
  point. Show that the accepted mode agrees to ‖·‖∞ ≤ 1e-8 and that the Laplace marginal agrees to
  ≤ 1e-9 nats. Add it to t05 as a regression test.
- **Gate:** δ at acceptance ≤ 1e-9 always. 1e-9 nats is seven orders of magnitude inside Gate 1's
  0.01-nat marginal budget.

**Then continue** revision 4 to completion:
1. C1 synthetic recovery;
2. C0/C1/H1/H2 beast fits in both protocols;
3. the measures and figures;
4. the report, README and two byte-identical runs.

If anything else blocks, stop with numbers. Sentinels are `PHASEC5_DONE` / `PHASEC5_BLOCKED` /
`PHASEC5_HANDOVER`, alone on their line.

**If time or quota runs short,** prioritise in this order:
1. C1 recovery;
2. C0 and C1 fits with the honest 10b measures;
3. H1 and H2;
4. the second reproduction run.

Hand over cleanly rather than half-finishing a step.
