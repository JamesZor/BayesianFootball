# Claude Task Prompt: Market-Inverse State-Space & Dynamic GRW Volatility Models

**Task Reference**: [`todos/023_prototype_market_inverse_grw_dynamics.md`](../../todos/023_prototype_market_inverse_grw_dynamics.md)  
**Design Document**: [`current_development/market_inverse_dynamics/DESIGN.md`](DESIGN.md)  
**Branch**: `feat/market-inverse-grw-dynamics`  
**Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-market-inverse`  
**Tmux Session**: `agent_claude_market_inverse`

---

## 1. Objective & Background

Traditional Bayesian football models update team ratings on discrete match goal counts ($0, 1, 2, \dots$), which have high Poisson variance and slow updates. Betting markets, however, price continuous, high-information consensus expectations reflecting the aggregate information of all participants.

By inverting the devigged market closing odds back to Poisson rates $(\lambda_{\text{mkt}, h}, \lambda_{\text{mkt}, a})$ using `Calibration.invert_market_rates` (Nelder-Mead on `Features.DoublePoissonMarketFeature`), we obtain continuous match-level target intensities.

Your task is to conduct an exploratory research and EDA study to determine what dynamic state-space model structure best expresses the changes in team abilities and feature dynamics implied by the market over time.

---

## 2. Core Model Architecture

Decompose the market-implied rates in log space:
$$\log \lambda_{\text{mkt}, h, m} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h(m), t(m)} + \beta_{\text{def}, a(m), t(m)} + \epsilon_{h, m}$$
$$\log \lambda_{\text{mkt}, a, m} = \mu + \alpha_{\text{att}, a(m), t(m)} + \beta_{\text{def}, h(m), t(m)} + \epsilon_{a, m}$$
with Gaussian observation noise $\epsilon \sim \mathcal{N}(0, \sigma_{\text{obs}}^2)$.

At each time step $t$, team ratings must sum to zero across teams ($\sum_i \alpha_{i, t} = 0$, $\sum_i \beta_{i, t} = 0$).

---

## 3. Dynamic Model Specifications (The 4 Arms)

Implement and compare four dynamic formulations for $\alpha_{\text{att}, i, t}$ and $\beta_{\text{def}, i, t}$:

1. **Arm 1: 1st-Order GRW (Constant Volatility)**:
   $$x_{i, t} = x_{i, t-1} + \sigma_x \cdot \omega_{i, t}$$
2. **Arm 2: 2nd-Order Momentum GRW (Damped Velocity)**:
   $$x_{i, t} = x_{i, t-1} + v_{i, t-1} + \sigma_x \cdot \omega_{i, t}, \quad v_{i, t} = \phi_x v_{i, t-1} + \sigma_{v, x} \cdot \eta_{i, t}$$
3. **Arm 3: Stochastic Volatility GRW (Time-Varying $\sigma_t$)**:
   $$x_{i, t} = x_{i, t-1} + \sigma_{x, i, t} \cdot \omega_{i, t}, \quad \log \sigma_{x, i, t} = \bar{h} + \gamma (\log \sigma_{x, i, t-1} - \bar{h}) + \sigma_h \cdot \xi_{i, t}$$
4. **Arm 4: 2-State Regime-Switching GRW**:
   $$x_{i, t} = x_{i, t-1} + \sigma_{x, S_{i, t}} \cdot \omega_{i, t}, \quad S_{i, t} \in \{\text{Calm}, \text{Turbulent}\}$$
   with Markov transition matrix $\mathbf{P}$.

---

## 4. Benchmark Cohort & Pipeline

- **Dataset**: Scottish Lower (tournaments 56/57, seasons 24/25 + 25/26, 710 fixtures).
- **Extraction**: Extract closing market rates using `Calibration.invert_market_rates(ds.odds; match_ids = panel)`.
- **Prototype Structure**: Follow repository prototype standard:
  - Loader: `current_development/market_inverse_dynamics/l01_market_inverse_loader.jl` (types, data structures, state-space equations, sampler definitions).
  - Runner: `current_development/market_inverse_dynamics/r01_market_inverse_runner.jl` (clean execution notebook with numbered sections: Packages, Config, Data Inversion, Model Fitting, Diagnostics, Trajectory Evaluation, Anomaly Detection).

---

## 5. Key Outputs & Evaluation

1. **Prediction Error**: Next-match market log-rate RMSE, MAE, and log-likelihood.
2. **Parameter Posteriors**: Inferred scales for $\mu, \gamma_{\text{home}}, \sigma_x$, momentum persistence $\phi$, and regime durations.
3. **Form Trajectory Plots**: Visual comparison of team paths (e.g. Falkirk, Hamilton, Partick Thistle).
4. **Anomaly / Shock Detection**: Catalog of fixtures with large market pricing residuals ($> 2.5 \sigma$).
5. **Phase 2 Recommendations**: Specific proposals for MS-GARCH conditional shock clustering, market-spread covariates, and player lineup/RAPM ratings.

---

## 6. Repository Standards & Safety Rules

- Read [`AGENTS.md`](../../AGENTS.md) and [`docs/guides/julia_coding_context_for_agents.md`](../../docs/guides/julia_coding_context_for_agents.md) before writing Julia.
- Always use multi-threaded pinning (`pinthreads(:cores)`, `LinearAlgebra.BLAS.set_num_threads(1)`).
- Keep work logged in `todos/023_prototype_market_inverse_grw_dynamics.md` and verify `./scripts/todo.sh check` is green before commit.
