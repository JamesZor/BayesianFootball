# 026 — Prototype Two-Tier Fast-Slow Latent State-Space Dynamics

| Field | Value |
|---|---|
| ID | 026 |
| Title | Prototype Two-Tier Fast-Slow Latent State-Space Dynamics |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-22 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`, `todos/021_prototype_fast_slow_grw_rate_pooling_and_decompression.md` |

## Context & Problem Statement

Standard single-frequency dynamic models face an unavoidable dilemma:
- A conservative step volatility ($\sigma \approx 0.02$/week) maintains stable long-term team ratings and robust portfolio Kelly growth, but causes severe favourite under-scaling (compression slope 1.41–1.66) because it refuses to adapt quickly to hot streaks.
- An aggressive step volatility ($\sigma \ge 0.08$/week) tracks hot streaks, but introduces erratic swings, destroys long-term proper scores, and causes severe drawdowns during mean-reverting shocks.

Phase 2 of Market-Inverse Dynamics (`current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`) revealed that market prices operate on two simultaneous timescales: a persistent baseline ability (where market ratings change by only $\sim 0.027$/week) combined with an acute sensitivity to fast rolling form (which contributes 28.6% of market supremacy variance).

Option 3 formulates a structural Two-Tier Fast-Slow State-Space architecture directly inside the Bayesian latent dynamics:
$$\log \lambda_{h, t} = \mu + \gamma_{\text{home}} + (\alpha_{\text{slow}, h, t} + \alpha_{\text{fast}, h, t}) + (\beta_{\text{slow}, a, t} + \beta_{\text{fast}, a, t})$$
1. **Slow Baseline State ($\theta_{\text{slow}}$)**:
   $$\theta_{\text{slow}, i, t} = \theta_{\text{slow}, i, t-1} + \sigma_{\text{slow}} \omega_{\text{slow}, i, t}, \quad \sigma_{\text{slow}} \sim \text{HalfNormal}(0.02)$$
   Anchors persistent structural quality, squad depth, and wage tier.
2. **Fast Form State ($\theta_{\text{fast}}$)**:
   $$\theta_{\text{fast}, i, t} = \rho_{\text{fast}} \theta_{\text{fast}, i, t-1} + \sigma_{\text{fast}} \omega_{\text{fast}, i, t}, \quad \rho_{\text{fast}} \in [0.60, 0.85], \quad \sigma_{\text{fast}} \sim \text{HalfNormal}(0.08)$$
   Captures transient tactical confidence, form runs, and momentum, naturally mean-reverting back to zero.

Both proxy xG and match outcomes inform the state updates. When a dominant favourite enters a high-performance run, $\theta_{\text{fast}}$ expands rapidly, allowing the model to match market favourite conviction while preserving long-term calibration.

Primary scope: Scottish Lower (tournaments 56/57, seasons 24/25 + 25/26, 40-fold walk-forward grid, 710 matches).

## Acceptance Criteria

- [ ] **Stage 0 (Mathematical Formulation & State-Space Compilation)**:
  - Implement two-tier state-space dynamics in `current_development/two_tier_state_space/`.
  - Ensure zero allocations in ReverseDiff compiled gradient tapes for the joint slow-fast trajectory.
  - Implement sum-to-zero identifiability constraints ($\sum_i \alpha_{\text{slow}, i} = 0$, $\sum_i \alpha_{\text{fast}, i} = 0$) at each time step.
- [ ] **Stage 1 (Smoke Gate, Folds 1/20/40)**:
  - 4 chains $\times$ 400 warmup + 400 draws on folds 1, 20, 40.
  - Zero NUTS divergences; $\hat{R} \le 1.05$; bulk/tail ESS $\ge 200$.
  - Verify posterior identification separates $\sigma_{\text{slow}}$ from $\sigma_{\text{fast}}$ without multimodality.
- [ ] **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
  - Benchmark against `m01_poisson_time_decay`, `m02_poisson_grw_1st_order`, and `m03_joint_gamma_poisson`.
  - Persist runs to PostgreSQL `mcmc_experiments`.
- [ ] **Stage 3 (Evaluation, Scaling & Portfolio)**:
  - Measure supremacy slope vs Betfair close ($y = \beta x$); verify target decompression.
  - Compute LogLoss, CRPS, RPS, ECE on 1X2 and totals.
  - Portfolio backtest under standard Scottish Lower policy.
- [ ] **Stage 4 (Findings Report)**:
  - Deliver findings report in `experiments/scottish_lower/13_two_tier_fast_slow_grw/README.md`.

## Ideas & Candidate Solutions

- **Kalman / FFBS vs Turing NUTS**: Since the transition equations are linear-Gaussian, the slow and fast states can be solved analytically via 2-tier Kalman filtering / FFBS conditional on static hyperparameters $(\rho_{\text{fast}}, \sigma_{\text{slow}}, \sigma_{\text{fast}})$, achieving orders-of-magnitude faster sampling than full MCMC.
- **Form Feature Injection vs State-Space**: Compare state-space fast AR(1) against explicit rolling form covariates (Option 1). State-space is generative and smooth; covariates are simpler and have no identifiability trade-offs.

## Work Log & Progress

- [2026-09-22 @antigravity] Formulated task specification based on multi-frequency market dynamics findings. Set status BACKLOG.

## Verification & Findings

*(To be filled upon completion of experimental stages)*
