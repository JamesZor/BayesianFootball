# Work package — TODO 039 Phase C, revision 2: the pooled model with an exact-likelihood Laplace update

**This supersedes §2.2 and gate 1 of `pi_market_model_phaseC_prompt.md` (revision 1).** Everything
else in revision 1 still applies: rules §0, reading list §1, fixed decisions §2.1, rungs
C0/C1/C2 §2.3, home-advantage side rungs H1/H2 §2.4, gates 2–5 and the measures of §3, tests §4,
acceptance §5 and the report §6. Read revision 1 first, then this.

Revision 1 was blocked at preflight (`results/C/PHASE_C_REPORT.md`, commit `614a2676`) for three
correct reasons. All three were errors in the manager's brief:
1. A Dirichlet(n·q) density at p expands as **−n·KL(q‖p)**, plus lower-order terms. Phase A's
   KL(p‖q) centre and its Hessian are therefore not the Dirichlet's.
2. Projecting a 1X2 likelihood to supremacy changes the density, so a gate against the
   **unprojected** book likelihood fails by construction.
3. Over/under constrains λ_h + λ_a, which is a **curved** contour in log-rate space. BTTS plus
   over/under books identify both rates.

You are a **fresh session on the same branch and worktree** (`pi/market-model-phase-c`,
`/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`). Your preflight code
`l05_laplace_preflight.jl` / `r05_laplace_preflight.jl` / `t05_laplace_preflight_tests.jl` and its
CSVs are useful: reuse the exact-Dirichlet density and the quadrature. Keep the preflight files
as the record of the block, and build the production files `l05_pooled.jl`, `r05_pooled.jl` and
`t05_pooled_tests.jl` beside them. **Night mode** still applies; sentinels are
`PHASEC_DONE` / `PHASEC_BLOCKED` / `PHASEC_HANDOVER`.

## 2.2 (revised) The book likelihood and its update

**Exact book likelihood.** Use the full Dirichlet/Beta density with all normalisers, on the
unchanged double-Poisson grid:

```
L_m(θ; n) = Π_{k ∈ markets of m} Dirichlet(p_{m,k}; n·q_{m,k}(θ))      θ = (log λ_h, log λ_a)
```

Phase A's isolated rates stay the **scoring targets** for every rung. They are no longer the
expansion point.

**What each kind of book contributes:**

| Book | Likelihood used |
|---|---|
| Full books (1X2 + ≥ 1 other market) | L_m(θ) on all its markets |
| Totals/BTTS-only books (no complete 1X2) | L_m(θ) on its markets, **exactly as is**. An O/U-only likelihood is flat along its constant-total curve, so it carries no supremacy information without any projection. BTTS + O/U books carry the (weak) supremacy information their prices actually contain. **Manager decision (logged for the human):** "informs the total only" in revision 1 was a linear approximation of that intent; the exact likelihood is the faithful version. |
| 1X2-only books (supremacy only) | the **level-integrated** likelihood L̃_m(d) = ∫ L_m(d, ℓ) dℓ, with d = log λ_h − log λ_a, ℓ = (log λ_h + log λ_a)/2 and flat measure on ℓ. Compute it by 1-D quadrature over ℓ on a grid wide enough that the integrand is negligible at both ends; report the truncation. This is the exact definition of "informs supremacy only". |

**Per-book update (a Laplace / iterated-EKF step at the joint mode):**
1. At fixture m's week, the state predicts θ_m ~ N(a_m, S_m), with S_m = H P_pred Hᵀ + σ_u² I.
2. Find θ*_m = argmax [log L_m(θ) + log N(θ; a_m, S_m)] by Newton with line search, from a_m.
3. Take J_m = −∇² log L_m(θ*_m) by AD or finite differences, and clip its eigenvalues at
   ε = 10⁻⁸ so flat directions carry no information.
4. **Equivalent Gaussian pseudo-observation:** z_m = θ*_m + J_m⁻¹ ∇ log L_m(θ*_m), with noise
   J_m⁻¹, as an observation of θ_m. Equivalently, use information form: add J_m to the
   precision, which avoids inverting the clipped J.
5. **1X2-only books:** the same in 1-D on d, with the prediction marginal N(cᵀa_m, cᵀS_m c) for
   c = (1, −1).
6. **The book's log marginal (collapsed-likelihood contribution),** with k = 2, or 1 for
   1X2-only books, and the outer log det taken of −∇² of the bracket at θ*:

   ```
   log L_m(θ*) + log N(θ*; a_m, S_m) + (k/2) log 2π − ½ log det(J_m + S_m⁻¹)
   ```

   Derive it in the loader docstring.
7. Optionally iterate (relinearise once at the updated mean) **if** the gate shows it's needed.
   Say whether you did.

**Smoothing:** RTS on the Gaussian pseudo-observations from the forward pass, labelled as
approximate.

**Inference:**
- θ = (σ_q, σ_s, σ_u, n), with the revision 1 priors;
- collapsed (approximate) likelihood plus slice sampling, 4 × (2,000 + 3,000);
- C0 is unchanged: exact Gaussian on the isolated rates.

## Gate 1 (revised): is the Laplace update accurate?

For ≥ 30 books of **each** type (full, 1X2-only, O/U-only, BTTS + O/U, as available), taken in
match-ID order with no selection:
- compare the step 6 log marginal with **exact 2-D quadrature** of L_m(θ)·N(θ; a, S) (1-D of
  L̃_m(d)·N for 1X2-only books);
- n ∈ {250, 1000, 4000};
- prediction centred at **realistic** a: the C0 one-step prediction for that fixture, or the
  isolated rate where it exists, plus a 0.10 offset case;
- S with equal-side SD ∈ {0.05, 0.20}.

**Thresholds are unchanged:** median |error| ≤ 0.01 nats and p95 ≤ 0.05 per book type and
setting. Report quadrature-order stability as before. If a type still fails, print
`PHASEC_BLOCKED` with the numbers and a diagnosis, as you did. Don't loosen anything.

Also add: **the Laplace update reproduces the exact posterior mean and variance of θ_m** from
quadrature, to ≤ 5% relative error of the SD, on the same books.

## Tests (additions to revision 1 §4)

- The Dirichlet log-density's large-n expansion: −n·KL(q‖p) + ((S−1)/2) log n + …, checked
  numerically (your preflight test can move here).
- The level-integrated 1X2 likelihood: quadrature-converged, and it has no dependence on ℓ.
- O/U-only books: J has one near-zero eigenvalue **along the local tangent of the
  constant-total curve at θ*** (not along a fixed direction); a BTTS + O/U book has rank 2.
- A synthetic panel with books drawn from Dirichlet(n·q): the full pipeline recovers
  (σ_q, σ_s, σ_u, n) within their 90% intervals, as revision 1 required.

## Reporting additions

In the report and the README, include a short "revision 2" note:
- what was wrong in revision 1 and how the update now works;
- that Phase A's KL(p‖q) inversion stays the scoring target;
- the manager's logged decision on totals/BTTS-only books.
