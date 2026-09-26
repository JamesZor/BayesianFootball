# 00 — Harness Validation: g2 Reproduction

## Question

Does the unified Phase 3 experiment harness reproduce the historical `g2_grw_all_spfl_cups`
fit (`a6f62436-ec8a-461d-8bd5-dc1861a2daaa`) within stochastic tolerance under the same model
recipe, data scope, and sampler budget?

## Candidate

- `g2_harness_repro`:
  - Model: `GlobalInterception` + `MultiScaleGRW()` + `GlobalHomeAdvantage` + `PoissonObservation`
  - Scope: `spfl_cups` (train 54–57 + `:senior_spfl_ties`, held-out target 56/57, no monitor)
  - Sampler: `QueuedNUTSConfig(1000, 500, 4 chains, accept_rate = 0.80, max_depth = 10)`

## Acceptance Decision Rules

1. Convergence audit passes or is triaged.
2. Paired fixture-clustered bootstrap ΔLogLoss vs g2 (`a6f62436`) on the 710 target fixtures has 95% CI covering 0 and |Δ| < 0.001.
3. Market-on-model compression slope within ±0.05 of g2's 1.2517.
4. Git commit recorded on `runs` table equals the checkout SHA.
