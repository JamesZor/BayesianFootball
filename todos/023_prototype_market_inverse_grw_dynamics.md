# 023 — Prototype market-inverse state-space and dynamic GRW volatility models

| Field | Value |
|---|---|
| ID | 023 |
| Title | Prototype market-inverse state-space and dynamic GRW volatility models |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-22 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `current_development/market_inverse_dynamics/` (README = findings), `docs/tickets/T014-betfair-1x2-home-away-swap.md`, f51522dd |

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

- [x] **Stage 0 (Mathematical Formulation & Extraction Pipeline)**:
  - Formulate the structural state-space decomposition for market log-intensities:
    $$\log \lambda_{\text{mkt}, h, t} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h, t} + \beta_{\text{def}, a, t} + \epsilon_{h, t}$$
    $$\log \lambda_{\text{mkt}, a, t} = \mu + \alpha_{\text{att}, a, t} + \beta_{\text{def}, h, t} + \epsilon_{a, t}$$
    with Gaussian observation noise $\epsilon \sim \mathcal{N}(0, \sigma_{\text{obs}}^2)$.
  - Pipeline extracts and caches accepted `MarketRateFit` targets across the 710 Scottish Lower fixtures using `Calibration.invert_market_rates`.
- [x] **Stage 1 (Model Architecture & Candidate Implementations)**:
  - Implement and benchmark four dynamic arms in `current_development/market_inverse_dynamics/`:
    - **Arm 1: 1st-Order GRW (Standard Random Walk)**: Constant step volatility $\sigma$.
    - **Arm 2: 2nd-Order Momentum GRW**: Local-linear trend with damped velocity $v_t$ ($\phi \in [0, 1)$).
    - **Arm 3: Stochastic Volatility GRW**: Time-varying step volatility $\log \sigma_t = \gamma \log \sigma_{t-1} + \sigma_\sigma \xi_t$.
    - **Arm 4: 2-State Regime-Switching GRW**: Discrete latent states (Low $\sigma$ vs High $\sigma$) driven by Markov transition probabilities.
- [ ] **Stage 2 (22/24 parameters pass; SV `gamma_h`/`sigma_h` bulk ESS 152 < 200 — see Verification) (MCMC / Variational Fitting & Diagnostics)**:
  - Verify convergence (0 divergences, $\hat{R} \le 1.05$, ESS $\ge 200$) or stable MAP/Kalman/particle filter estimation.
  - Profile parameter persistence ($\phi$), volatility scales ($\sigma, \sigma_t$), and regime duration posteriors.
- [x] **Stage 3 (Evaluation & Metrics)**:
  - In-sample and out-of-sample log-rate prediction error (RMSE, MAE, Log-Likelihood) predicting next-match market rates.
  - Team form trajectory comparisons (visualizing attack/defence paths across Scottish Lower teams).
  - Shock/anomaly detection catalog (identifying fixtures with large market residual shifts).
- [x] **Stage 4 (Findings Report & Phase 2 Roadmap)**:
  - Deliver findings in `current_development/market_inverse_dynamics/README.md`.
  - Propose Phase 2 extensions: MS-GARCH / conditional shock clustering, market-line covariates, and player lineup ratings.

## Ideas & Candidate Solutions

- **State-space vs MCMC**: Because the observation equation is linear-Gaussian conditional on the latent path, Arms 1 and 2 admit exact Kalman filtering / Forward-Filtering Backward-Sampling (FFBS) for extreme speed, while Arms 3 and 4 can be sampled via Turing.jl NUTS or particle MCMC.
- **Centering & Identification**: Team attack and defence must sum to zero ($\sum_i \alpha_{i, t} = 0$, $\sum_i \beta_{i, t} = 0$) at each time step to prevent drift against the global intercept $\mu$.
- **Time Indexing**: Group fixtures by matchday bi-weeks (matching the L1 convention) or continuous calendar time.

## Work Log & Progress

- [2026-09-22 @antigravity] Structured task specification via interactive `/grill-me` alignment. Provisioned dedicated worktree `.worktrees/BayesianFootball-market-inverse` on branch `feat/market-inverse-grw-dynamics`. Claimed for `@claude` in tmux session `agent_claude_market_inverse`.
- [2026-09-22 @claude] Picked up. kaimon down → compute on mcmc-beast in a dedicated checkout `/root/BF_market_inverse` (tmux `market_inverse`, `-t 16`, Manifest copied from `BF_fast_slow_grw` to keep Distributions 0.25.126). Stage 0: panel = 710 fixtures → 623 accepted inversions (87 refused, `refusals.csv`), 22 teams, 91 weekly steps, 1,246 log-rate observations; target = Betfair (−20, 0] TWA close, same book as TODO 021.
- [2026-09-22 @claude] Stage 1 design choice: no NUTS. The observation model is linear-Gaussian given the per-team innovation variances, so the loader uses an exact Kalman filter (collapsed likelihood, μ/γ_home/paths integrated out) + FFBS. Arms 1/2 (+ static control): coordinate slice sampling on the collapsed posterior of θ. Arms 3/4: partially-collapsed Gibbs (θ | aux collapsed → FFBS paths → elliptical-slice SV / FFBS regimes → conjugate P). One-step-ahead: exact Kalman / Rao-Blackwellised particle filter. Engine gates (toy panel vs the batch joint Gaussian): Kalman loglik to 1.6e-13, RTS mean to 5e-15, heteroscedastic schedule to 1e-14, RBPF degenerate limit = Kalman to 4e-15, FFBS moments within MC error.
- [2026-09-22 @claude] Smoke fits flagged the regime arm finding a turbulent state ~10× the calm scale with short bursts; added control arm `a1b_grw1_break` (GRW1 + one-off season-boundary jump) so SV/regime gains can be separated from the summer repricing.
- [2026-09-22 @claude] Production run (`results/production_run.log`, ~75 min on the beast, all arms concurrently; 4 × (2,000 + 3,000) draws, thin 1/4/20 for Gaussian/momentum/SV+regime; RBPF 20,000 particles). Findings + Phase 2 roadmap in `current_development/market_inverse_dynamics/README.md`. Cross-book check found one Betfair home/away swap (match 14035501) → ticket T014, not fixed inline.

## Verification & Findings

Measured 2026-09-22, `r01_market_inverse_runner.jl`, 623 accepted fixtures / 1,246 log-rate observations:

- **Engine gates** (toy panel vs batch joint Gaussian): 11/11 pass — Kalman loglik ≤ 1.6e-13, RTS mean ≤ 4.8e-15, degenerate RBPF = Kalman to 3.6e-15, FFBS moments within MC error.
- **Convergence**: 22/24 parameters pass R̂ ≤ 1.05 and bulk+tail ESS ≥ 200 (max R̂ 1.020). **Fails:** a3 SV `gamma_h`, `sigma_h` bulk ESS 152 (R̂ 1.011). Threshold not relaxed; a longer SV run or the observation-driven variant (README §7.2) would close it.
- **Honest one-step (θ from 24/25, scored on 25/26, n = 612)**: RMSE static 0.178 → GRW1 0.130; momentum 0.130, season-break 0.131, SV 0.132, regime 0.130. Mean log pd: GRW1 0.620, SV 0.646, regime 0.668.
- **Verdict**: no-momentum GRW1 at ~0.027–0.029/week describes the market's ratings; SV/regime density gains come from spike-and-revert absorption of single-fixture outliers (e.g. Kelty v Hamilton 2025-09-20) — next step is a Student-t observation model (README §7.1).
