# Phase C, revision 7: zero-motion steps count as stalls (manager, ~04:57)

**Tolerances are unchanged.** Revision 6 passed its exact-case regression (t05 289/289). Recovery
then failed at fixture 12476686:
- Armijo **accepted** 94 steps that left θ unchanged (zero motion);
- polishing never triggered, because it fires only on line-search *failure*;
- the solver hit the 100-iteration limit at δ = 1.04e-12, with a Newton step of 4.2e-8.

**The rule:** an accepted step with ‖θ_new − θ‖∞ < 1e-14, or with f(θ_new) == f(θ) in Float64,
**is a stall.**
- Apply revision 6's polish: up to 3 undamped Newton steps, the same acceptance rules, and the
  same error if they fail.
- Record such cases as `zero_motion_polished` in `newton_termination.csv`.
- Add fixture 12476686 at the recorded θ as a t05 regression.

Then continue in revision 5's priority order. Sentinels: `PHASEC7_DONE` / `PHASEC7_BLOCKED` /
`PHASEC7_HANDOVER`.
