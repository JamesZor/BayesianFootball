# Stage 0 — damped local-linear trend

## Contract

Phase 1 retains the minimal Poisson likelihood, global intercept and global home
advantage. The canonical split is `GroupedCVConfig`: pooled tournaments 56/57,
two history seasons, match-biweek steps, 24/25 and 25/26. A micro step is **not** an
individual fixture. Historical seasons have one state each; the target season has
one state per observed biweek. Output has exactly `n_history + n_target` columns.

The original macro/initial/micro scale priors are unchanged. Add independent
attack/defence velocity processes, with persistence pooled across teams within each
side:

- attack and defence persistence: `Beta(2,2)`;
- attack velocity innovation SD: `Gamma(2,0.0075)` (shape/scale);
- defence velocity innovation SD: `Gamma(2,0.006)`;
- velocity innovations: independent standard Normal;
- target-season boundary velocity: exactly zero.

Resetting velocity at the season boundary avoids treating a season-long gap as
one biweek. Macro levels still carry across historical seasons. These choices are
part of the prototype, not inferred empirical findings.

## Derivation and stability

For either attack or defence, before cross-team centering:

```
a_k = a_(k-1) + v_(k-1) + sigma_a * epsilon_k
v_k = phi * v_(k-1) + sigma_v * eta_k
v_0 = 0
```

The transition matrix is `[[1,1],[0,phi]]`, with eigenvalues **1 and phi**.
For `0 <= phi < 1`, velocity is stable and asymptotically stationary, with
variance `sigma_v^2/(1-phi^2)`. At finite k its variance is
`sigma_v^2*(1-phi^(2k))/(1-phi^2)`. Starting at zero is not a stationary initial
distribution. The **level is nonstationary**; damping velocity does not remove
its unit root. Centering removes the common league-level direction, not the unit
roots in team contrasts.

Eliminating velocity gives

```
a_k - (1+phi)*a_(k-1) + phi*a_(k-2)
    = sigma_v*eta_(k-1) + sigma_a*(epsilon_k - phi*epsilon_(k-1)).
```

This is not an AR(2) with independent residuals when `sigma_a != 0`.
Kinematic second-difference acceleration is the boundary `phi=1, sigma_a=0`:
velocity itself becomes a random walk and level variance grows cubically with
horizon. That undamped alternative is not sampled in Phase 1.

At `phi=0`, velocity adds lagged independent noise; this is **not** the original
GRW unless `sigma_v=0`. The exact nested first-order limit is `sigma_v=0`.
A prior on innovations centered at zero does not pull an already attained GRW
level back to zero each match. Momentum changes expected *increments*, not that
fact. Better favourite probabilities remain a testable hypothesis, not a theorem.

## Vector representation and identifiability

Let `Z[j] = eta_j`, and define a constant lag table and mask for observed K:

```
H[j,k] = 1(j<k) * phi^(k-1-j),  j=1,...,K-1; k=1,...,K.
```

Then the velocity contribution to position is
`(sigma_v * Z) * H * target_accumulator` (teams in rows).
The accumulator is the unchanged GRW linear map. This polynomial has no division
by `1-phi`, so remains numerically well-behaved near 1. Constant exponent zero
uses a data-only specialization to avoid the generic power adjoint's `0/0` at
phi=0. No sampled-value branch, mutable recurrence, or runtime loop is on the
model tape. Centering is multiplication by `I - 11'/n_teams`.

Only `eta_1,...,eta_(K-1)` affect observed positions. `eta_K` is omitted rather
than sampled as an unidentifiable terminal nuisance. K=0 and K=1 omit the whole
velocity block, including phi and sigma_v. K=2 observes sigma_v but cannot identify
phi from positions; the latter remains prior-driven. At longer horizons sigma_a,
sigma_v and phi can still trade off. Smoke diagnostics must report this honestly.

## Held-out forecasting — approved by user on 2026-09-21

Use conditional-mean states, matching the existing first-order GRW convention:

```
a_forecast = a_K + E[v_K | retained latent draws]
E[v_K | retained latent draws] = sigma_v * sum_j phi^(K-j) * eta_j
```

The omitted terminal innovation has mean zero. Both position and velocity are
centered across the same teams. At K=0 or K=1 forecast velocity is zero. Unknown
teams retain the existing zero team-effect fallback, and global home advantage
remains global. Forecast horizon is the next biweek, not elapsed calendar days.
No future outcomes are consulted and no future process noise is injected. Thus
these rate draws are conditional-state forecasts, not a full state-transition
posterior predictive distribution. Integrating process noise is a separate,
matched-control experiment, not an unnoticed change to this comparison.

## AD optimization boundary

All changes remain inside this prototype module; no `src/` methods are replaced.
`ArrayClampGuard` selects a minimal, mathematically equivalent Poisson engine
through dispatch on a prototype-owned type. Priors, site names/order, clamp
limits, time weights and extraction are unchanged.

1. `array_scalar` lifts scalar sites into a one-element `TrackedArray`. Its local
   ReverseDiff instruction preallocates the output and accumulates its adjoint
   back to the scalar. No shared/global scratch; each tape owns its buffer.
2. Array-only broadcasts select ReverseDiff's cached `∇broadcast` instead of the
   allocating scalar-argument `tracker_∇broadcast` path. Clamp bounds are arrays.
3. Matrix centering avoids the installed ReverseDiff's unvectorized
   `mean(...; dims=1)` path. This can change final rounding, so full log densities
   are checked to `1e-9`, not represented as universally bit-identical.
4. The optimized global intercept/home submodels avoid `fill` replay allocations.
5. `models(optimized=false)` retains the ordinary engine for deterministic parity.

The scalar-lift instruction uses ReverseDiff internals and is version-sensitive;
rerun the complete AD gates after upgrades. `test_momentum.jl` tests the independent
recurrence, synthetic chain reconstruction, Turing returned states, the nested
limit, both boundaries and ForwardDiff gradients. `r00_momentum_preflight.jl`
requires zero warmed allocations and identical parameter layouts, checks linked
log-density parity, and compares compiled, fresh ReverseDiff and ForwardDiff at
small and warmup-scale displacements. No sampling gate is implied by these tests.
