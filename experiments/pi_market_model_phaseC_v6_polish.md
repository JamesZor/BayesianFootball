# Phase C, revision 6: polish stalled modes with undamped Newton steps (manager, ~04:35)

**Tolerances stay exactly as revision 5 set them:** mode ≤ 1e-8, marginal ≤ 1e-9 nats against the
high-precision re-solve.

**Why the gaps appeared.** Revision 5 accepted a stalled point at δ ≤ 1e-9, and that still allows a
mode error of order √(2δ/λ_min) ≈ 1e-6. That was the manager's inconsistency. The observed gaps
(4e-7 mode, 7e-7 nats marginal) follow from it.

**The fix keeps every tolerance.** The roundoff that stalls Armijo is in the f comparisons, not in
the gradient or Hessian. So when the line search stalls and δ ≤ 1e-9:
- take up to **3 undamped Newton steps**, θ ← θ + H⁻¹g;
- accept once ‖Δθ‖∞ ≤ 1e-12, or ‖g‖ stops decreasing;
- record the counts as `polished` in `newton_termination.csv`.

If after polishing ‖g‖ still exceeds its pre-polish value, or δ > 1e-9, raise an error exactly as
now.

Re-run the exact-case regression at the unchanged tolerances.
- **If it passes,** continue revision 4/5 in revision 5's priority order.
- **If it fails,** stop with the numbers. **Don't** change the tolerances. The manager or the
  human decides.

Sentinels: `PHASEC6_DONE` / `PHASEC6_BLOCKED` / `PHASEC6_HANDOVER`.
