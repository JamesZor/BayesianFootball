# Momentum MultiScale GRW Dynamics (Scottish Lower Phase 1)

> **Experiment**: `experiments/scottish_lower/10_momentum_multiscale_grw/`  
> **TODO**: [022](../../todos/022_prototype_momentum_multiscale_grw_dynamics.md)  
> **Target**: Scottish Lower League Football (Tournaments 56 & 57, 40 folds, 710 fixtures, seasons 24/25 + 25/26)  
> **Status**: IN_PROGRESS  

## 1. Executive Summary

This experiment investigates whether introducing **2nd-order / momentum dynamics** into team state-space random walks breaks through the Bayesian shrinkage compression ceiling on Scottish Lower football.

Standard 1st-order `MultiScaleGRW` suffers from memoryless shrinkage: every step resets prior expectation to zero, causing net team supremacy slopes of ~0.39 vs Betfair close, and underpricing heavy favourites at 55% (vs market 76%).

Momentum GRW models team ability $\alpha_t$ with an autoregressive velocity term $v_t$:
$$\begin{aligned}
\alpha_t &= \alpha_{t-1} + v_{t-1} + \sigma_\alpha \epsilon_{\alpha, t} \\
v_t &= \phi v_{t-1} + \sigma_v \epsilon_{v, t}
\end{aligned}$$
allowing dominant teams on consistent form to build positive momentum and separate directionally into heavy-favourite territory without adding unguided isotropic noise.

## 2. Benchmark Arms

| Arm | Dynamics | Observation | Purpose |
|---|---|---|---|
| `m01_poisson_time_decay` | `TimeDecayDynamics(180.0)` | `PoissonObservation()` | Control 1 (Traditional time decay) |
| `m02_poisson_grw_1st_order` | `MultiScaleGRW()` | `PoissonObservation()` | Control 2 (1st-order random walk) |
| `m03_poisson_momentum_grw` | `MomentumMultiScaleGRW()` | `PoissonObservation()` | Candidate (2nd-order momentum walk) |

## 3. Headline Results

*(To be populated by Pi / GPT-6 Astra upon completion of Stage 3)*

| Arm | Supremacy Slope vs Close | $R^2$ vs Close | Win Prob on Market Favourites $\ge 0.70$ | 1X2 LogLoss | O/U 2.5 LogLoss | Total Return | Max DD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m01_poisson_time_decay` | — | — | — | — | — | — | — | — |
| `m02_poisson_grw_1st_order` | — | — | — | — | — | — | — | — |
| `m03_poisson_momentum_grw` | — | — | — | — | — | — | — | — |

## 4. Verification & Reproduction

```bash
# On mcmc-beast (/root/BF_momentum_grw)
julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl
julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r20_momentum_production_grid.jl
julia --project -t 16 experiments/scottish_lower/10_momentum_multiscale_grw/r30_momentum_evaluation.jl
```
