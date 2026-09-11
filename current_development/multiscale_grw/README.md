# Task 007 — MultiScaleGRW prototype

This directory prototypes a non-centred two-speed Gaussian random walk for Scottish
Lower team attack and defence. History advances by one macro innovation per season;
the target season advances by one micro innovation per observed match-biweek. Each
state is zero-centred over teams, and held-out fixtures use the final state visible at
the fold cutoff.

## Contract

- Data: pooled tournaments 56/57, target seasons 24/25 and 25/26.
- Split: 40 match-biweek walk-forward folds, 710 held-out fixtures.
- AD: compiled ReverseDiff tape; replay allocations are measured. The installed
  TimeDecay control allocates 43,888 bytes in the identical harness, so literal
  zero allocation remains an unmet performance target rather than a claimed pass.
- Sampler: four chains, 800 warmup + 800 retained, target acceptance 0.90.
- Promotion: R̂ ≤ 1.01, bulk/tail ESS ≥ 400, divergence rate < 0.1%, BFMI ≥ 0.30.
- Control: otherwise-matched `TimeDecayDynamics(days_half_life = 180.0)` fits loaded
  from the experiment database and required to cover the identical OOS fixture IDs.

## Two-fold preflight

| Model | Fold | Parameters | Tape instructions | Gradient ms | Alloc bytes | R̂ max | ESS min | Divergences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | 1 | 98 | 735 | 0.053 | 35440 | 1.0091 | 558 | 0 |
| `m00_baseline_grw` | 2 | 158 | 1309 | 0.071 | 36608 | 1.0104 | 691 | 0 |
| `m05_production_wealth_grw` | 1 | 99 | 751 | 0.058 | 35440 | 1.0128 | 566 | 0 |
| `m05_production_wealth_grw` | 2 | 159 | 1325 | 0.085 | 36608 | 1.0132 | 453 | 0 |
| `m05_joint_production_wealth_grw` | 1 | 101 | 791 | 0.120 | 128848 | 1.0075 | 681 | 0 |
| `m05_joint_production_wealth_grw` | 2 | 161 | 1365 | 0.108 | 133088 | 1.0154 | 582 | 0 |


## Persisted production runs

| Model | Folds | OOS | R̂ max | ESS min | Divergences | Wall min | Run UUID |
|---|---:|---:|---:|---:|---:|---:|---|
| `m00_baseline_grw` | 40 | 710 | 1.0067 | 1662 | 0 | 1.1 | `f64a00a2-34a0-4f31-8c58-c093c92d54b7` |
| `m05_production_wealth_grw` | 40 | 710 | 1.0073 | 1909 | 0 | 60.7 | `b2d8036d-8fbd-45f9-92b5-cc7675926232` |
| `m05_joint_production_wealth_grw` | 40 | 710 | 1.0069 | 1086 | 0 | 90.9 | `f870dbb7-9df0-4dae-a84a-cf570cf8113e` |


## Proper scores versus TimeDecayDynamics

Negative deltas favour MultiScaleGRW.

| GRW candidate | TimeDecay control | LogLoss | Δ LogLoss | Brier | Δ Brier | RPS | Δ RPS | CRPS | Δ CRPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | `m00_baseline` | 0.65866 | -0.00170 | 0.23337 | -0.00079 | 0.22537 | -0.00189 | 0.62701 | -0.00368 |
| `m05_production_wealth_grw` | `m05_production_wealth` | 0.65825 | -0.00148 | 0.23318 | -0.00069 | 0.22493 | -0.00129 | 0.62667 | -0.00254 |
| `m05_joint_production_wealth_grw` | `m05_joint_production_wealth` | 0.65681 | -0.00031 | 0.23250 | -0.00011 | 0.22534 | -0.00023 | 0.62660 | -0.00050 |


## Runtime comparison

The ratio is `TimeDecay wall time / MultiScaleGRW wall time`; values above one mean
MultiScaleGRW completed faster despite its larger latent state.

| GRW candidate | GRW wall min | TimeDecay wall min | TimeDecay / GRW |
|---|---:|---:|---:|
| `m00_baseline_grw` | 1.1 | 2.2 | 1.94× |
| `m05_production_wealth_grw` | 60.7 | 2.0 | 0.03× |
| `m05_joint_production_wealth_grw` | 90.9 | 5.0 | 0.06× |


## Phase 2

completed and persisted at 2026-09-11T11:42:56.294; strict convergence and score comparison passed

## Reproduction

```bash
# On mcmc-beast, from /root/BF_multiscale_grw
L01_STAGE=preflight /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
L01_STAGE=phase1   /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
L01_STAGE=phase2   /root/.juliaup/bin/julia --project -t 16 current_development/multiscale_grw/r01_runner.jl
```

PostgreSQL namespace: `scottish_lower_multiscale_grw_2426`.
