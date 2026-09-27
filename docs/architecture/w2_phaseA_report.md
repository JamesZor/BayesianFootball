# Wave 2 Phase A build report

**TODO:** 031  
**Branch:** `feat/w2-tier-components`  
**Validated implementation SHA:** `248ec796`  
**Date:** 2026-09-27

## Scope and outcome

Phase A implements the opt-in model surface required by the Wave 2 contract. It does
not run smokes, grids, NUTS, or other posterior inference.

- `MultiScaleGRW(target_season_step=true)` adds one summer macro innovation before
  the first target-season state. An empty target block receives a deterministic,
  fold-seeded posterior-predictive summer innovation during OOS extraction.
- `PyramidTiers` supplies the first-window or full-pyramid tier anchor, optional
  `Beta(1.5, 4)` carry, optional `Gamma(2, 0.08)` jump scale, fixed 0.48/0.52
  attack/defence split, and the Old Firm pseudo-tier.
- `FixedTransitionPriors` implements all three B1-literal transitions from the
  contract.
- `CompetitionKappa` adds zero-sum tier intercept deltas (scale 0.10) and
  match-competition log-kappa deltas (scale 0.25); senior cups share one level.
- `experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl` declares all
  15 cells on the frozen 23/24–25/26 panel.

All options are off by default. Optional model behavior is selected by concrete type
or `Val` dispatch, and the new `@model` bodies contain no runtime branches, loops,
mutation, `view`, `findall`, or `isnan`.

## Compatibility and leakage controls

- The historical nine-field `MultiScaleGRW` layout is unchanged. The opt-in form is
  represented by a separate internal concrete type returned by
  `MultiScaleGRW(target_season_step=true)`, preserving old serialized fit artifacts.
- A persisted W1 GRW fit (`decb01be-f214-47bb-bc7b-bcc309026bda`) loaded successfully
  on the validated SHA. Log: `/root/BF_runs/logs/248ec796/w1_grw_load.log`.
- All 12 W1 candidate recipe hashes exactly match persisted `harness_checks` rows.
  Representative pinned hashes are also unit-tested:
  - `td_lower_joint`: `722ed88a7efe8c82eb013f789f4d41936fbcea597fa8b9d8cef3f620d062e9fc`
  - `grw_lower_joint`: `d373b3393d5079129fa7bed773b28e23cf2c5b67a2904b9ea189f9d37a55f8a6`
- `Features.create_features` now places the fold's declared target season in the
  feature data before pyramid extraction. A regression test adds target+1 schedule
  rows and proves every pyramid feature remains unchanged.
- Full-pyramid extraction requires `Data.apply_scope` metadata instead of silently
  degrading to the scoped tier table.
- The W2 `m12_td_ref` was compared with persisted run
  `132df5c2-c742-4e95-8693-3aeb2b2cbaef`: interception, dynamics, home advantage,
  both covariates, guard, joint feature, shape prior, log-kappa prior, and shared-kappa
  mode are field-for-field equal. The persisted config hash remains
  `a11db174f91861a795c150bbdfb6f34b5814516f0a179c153ebec912eed5670d`.

## Synthetic and regression verification

`test/pyramid_tier_tests.jl` covers accumulator geometry, empty-target OOS behavior,
feature reductions, cup ties, anchor semantics, target+1 leakage, all fixed transition
priors, default-off identity, pinned W1 hashes, compiled ReverseDiff/ForwardDiff
agreement, real-chain extraction algebra, league-delta rates, and OOS reload parity.

Results on the laptop:

- `test/pyramid_tier_tests.jl`: **90/90**
- `test/test_multiscale_grw.jl`: **124/124**
- `test/builder_tests.jl`: all testsets pass
- `test/scoped_splitter_tests.jl`: all testsets pass
- canonical sequential `test/runtests.jl`: **4429/4429** in 7m31s
- parallel runner: **23/24**; only the pre-existing T007 hidden dependency in
  `features_tests.jl` (`SplitClockProbe` undefined) fails, exactly as documented in
  `docs/tickets/T007-parallel-feature-test-hidden-dependency.md`.

The canonical sequential runner and parallel runner both include the new suite.

## Beast instantiation audit

The pushed SHA `248ec796` was checked out with `scripts/beast_checkout.sh` into
`/root/BF_runs/248ec796`. The audit used 16 Julia threads, pinned cores, one BLAS
thread, and performed no posterior sampling. For every candidate it instantiated the
real scoped model on fold 1 and the first 24/25 fold (`source_fold=21`,
`n_target_steps=0`), compiled a ReverseDiff tape, evaluated finite gradients, exercised
OOS extraction from a two-draw prior-only chain, and checked serialize/reload equality.

Log: `/root/BF_runs/logs/248ec796/phaseA_instantiate.log`

| Candidate | Parameters | Tape instructions | First compile (s) | Gradient (ms) |
|---|---:|---:|---:|---:|
| `td_base` | 92 | 191 | 7.144 | 0.130 |
| `td_a2` | 94 | 220 | 2.934 | 0.154 |
| `td_a2_carry` | 95 | 228 | 2.231 | 0.163 |
| `td_a2_carry_jump` | 136 | 264 | 2.147 | 0.209 |
| `td_ldelta` | 101 | 219 | 2.129 | 0.157 |
| `td_a2_carry_jump_ldelta` | 145 | 292 | 1.872 | 0.231 |
| `grw_base` | 180 | 1255 | 2.680 | 0.160 |
| `grw_step` | 180 | 1255 | 0.410 | 0.168 |
| `grw_step_a2_carry_jump` | 224 | 1328 | 1.017 | 0.246 |
| `grw_step_ldelta` | 189 | 1283 | 0.833 | 0.190 |
| `grw_step_a2_carry_jump_ldelta` | 233 | 1356 | 1.028 | 0.270 |
| `td_lower_base` | 52 | 191 | 0.001 | 0.066 |
| `td_lower_a2full_carry_jump` | 96 | 264 | 0.001 | 0.105 |
| `td_lower_b1lit` | 76 | 222 | 2.347 | 0.078 |
| `m12_td_ref` | 55 | 232 | 2.234 | 0.077 |

The same 15 candidates also passed the first-24/25 empty-target fold; the complete
30-row telemetry is in the external log. Compile times after the first matching Julia
specialization are warm-cache timings and should not be interpreted as candidate
fit-time comparisons.

## Decision

Phase A is ready for the manager acceptance rerun. No smoke or grid was launched.
Wave 2 Phase B and all sampling remain out of scope.
