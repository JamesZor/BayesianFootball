# 023 — Prototype market-inverse state-space and dynamic GRW volatility models

| Field | Value |
|---|---|
| ID | 023 |
| Title | Prototype market-inverse state-space and dynamic GRW volatility models |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-22 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `current_development/market_inverse_dynamics/` |

## Context & Problem Statement

Standard Bayesian football models fit dynamic team ratings on low-information discrete goal counts ($0, 1, 2, \dots$). While informative, match scores have high Poisson variance and slow updates. In contrast, betting markets price continuous, high-information consensus beliefs that evolve match-by-match.

By inverting the pre-match and closing odds into latent Poisson log-rates via Nelder-Mead on `Features.DoublePoissonMarketFeature` (`Calibration.invert_market_rates`), we obtain continuous match-level target intensities $(\lambda_{\text{mkt}, h}, \lambda_{\text{mkt}, a})$.

This research workpackage investigates how market-implied team attack and defence ratings evolve over time under alternative Gaussian Random Walk (GRW) and state-space volatility specifications. Specifically, we want to measure:
1. Do market-implied team abilities follow standard 1st-order random walks, or do they exhibit persistent momentum (2nd-order damped velocity)?
2. Is team ability volatility ($\sigma$) constant, or does it vary over time (Stochastic Volatility)?
3. Do teams experience discrete regime shifts (Markov-switching between stable and turbulent/high-volatility periods)?
4. Can we detect market anomalies, rapid repricing events (e.g. manager changes, injury crises), and structural shocks from the filtered states?

Primary focus is Scottish Lower (tournaments 56/57, seasons 24/25 + 25/26, 710 fixtures), with optional secondary expansion to Scottish Premiership or English tournaments.

## Acceptance Criteria

- [ ] **Stage 0 (Mathematical Formulation & Extraction Pipeline)**:
  - Formulate the structural state-space decomposition for market log-intensities:
    $$\log \lambda_{\text{mkt}, h, t} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h, t} + \beta_{\text{def}, a, t} + \epsilon_{h, t}$$
    $$\log \lambda_{\text{mkt}, a, t} = \mu + \alpha_{\text{att}, a, t} + \beta_{\text{def}, h, t} + \epsilon_{a, t}$$
    with Gaussian observation noise $\epsilon \sim \mathcal{N}(0, \sigma_{\text{obs}}^2)$.
  - Pipeline extracts and caches accepted `MarketRateFit` targets across the 710 Scottish Lower fixtures using `Calibration.invert_market_rates`.
- [ ] **Stage 1 (Model Architecture & Candidate Implementations)**:
  - Implement and benchmark four dynamic arms in `current_development/market_inverse_dynamics/`:
    - **Arm 1: 1st-Order GRW (Standard Random Walk)**: Constant step volatility $\sigma$.
    - **Arm 2: 2nd-Order Momentum GRW**: Local-linear trend with damped velocity $v_t$ ($\phi \in [0, 1)$).
    - **Arm 3: Stochastic Volatility GRW**: Time-varying step volatility $\log \sigma_t = \gamma \log \sigma_{t-1} + \sigma_\sigma \xi_t$.
    - **Arm 4: 2-State Regime-Switching GRW**: Discrete latent states (Low $\sigma$ vs High $\sigma$) driven by Markov transition probabilities.
- [ ] **Stage 2 (MCMC / Variational Fitting & Diagnostics)**:
  - Verify convergence (0 divergences, $\hat{R} \le 1.05$, ESS $\ge 200$) or stable MAP/Kalman/particle filter estimation.
  - Profile parameter persistence ($\phi$), volatility scales ($\sigma, \sigma_t$), and regime duration posteriors.
- [ ] **Stage 3 (Evaluation & Metrics)**:
  - In-sample and out-of-sample log-rate prediction error (RMSE, MAE, Log-Likelihood) predicting next-match market rates.
  - Team form trajectory comparisons (visualizing attack/defence paths across Scottish Lower teams).
  - Shock/anomaly detection catalog (identifying fixtures with large market residual shifts).
- [ ] **Stage 4 (Findings Report & Phase 2 Roadmap)**:
  - Deliver findings in `current_development/market_inverse_dynamics/README.md`.
  - Propose Phase 2 extensions: MS-GARCH / conditional shock clustering, market-line covariates, and player lineup ratings.

## Ideas & Candidate Solutions

- **State-space vs MCMC**: Because the observation equation is linear-Gaussian conditional on the latent path, Arms 1 and 2 admit exact Kalman filtering / Forward-Filtering Backward-Sampling (FFBS) for extreme speed, while Arms 3 and 4 can be sampled via Turing.jl NUTS or particle MCMC.
- **Centering & Identification**: Team attack and defence must sum to zero ($\sum_i \alpha_{i, t} = 0$, $\sum_i \beta_{i, t} = 0$) at each time step to prevent drift against the global intercept $\mu$.
- **Time Indexing**: Group fixtures by matchday bi-weeks (matching the L1 convention) or continuous calendar time.

## Work Log & Progress

- [2026-09-22 @antigravity] Structured task specification via interactive `/grill-me` alignment. Provisioned dedicated worktree `.worktrees/BayesianFootball-market-inverse` on branch `feat/market-inverse-grw-dynamics`. Claimed for `@claude` in tmux session `agent_claude_market_inverse`.

## Verification & Findings

Not run yet.
