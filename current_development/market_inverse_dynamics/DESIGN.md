# Market-Inverse State-Space & Dynamic GRW Volatility Models

**Task Reference**: [TODO 023](../../todos/023_prototype_market_inverse_grw_dynamics.md)  
**Assigned**: `@claude`  
**Branch**: `feat/market-inverse-grw-dynamics`  
**Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-market-inverse`

---

## 1. Executive Summary & Research Motivation

In traditional Bayesian football models, team rating trajectories ($\alpha_{i, t}, \beta_{i, t}$) are fitted on low-information discrete goal counts ($0, 1, 2, \dots$). While essential for actual match settlement, goal observations suffer from high Poisson noise, rare-event latency, and sample-size constraints.

Betting markets, by contrast, price continuous, high-information consensus expectations reflecting the aggregate information of all market participants (lineups, tactics, injuries, motivation, weather). 

By inverting the devigged market closing odds back to Poisson rates $(\lambda_{\text{mkt}, h}, \lambda_{\text{mkt}, a})$ using `Calibration.invert_market_rates` (Nelder-Mead on `Features.DoublePoissonMarketFeature`), we obtain continuous, match-level target intensities.

This research package formulates and benchmarks a family of **state-space dynamic models fitted directly to market log-intensities**. Our goal is to extract:
1. True market-perceived team attack and defence ratings over time.
2. Market belief volatility ($\sigma$) and whether team ability updates follow standard random walks or directional momentum (velocity).
3. Evidence for **Stochastic Volatility** (time-varying uncertainty around team form) and **Regime Switching** (calm periods vs turbulent injury/manager crises).
4. Structural shocks and market anomalies (detecting dates of rapid market repricing).

---

## 2. Mathematical Formulation

### 2.1 Market Observation Model

For match $m$ between home team $h(m)$ and away team $a(m)$ at time $t(m)$:
$$\log \lambda_{\text{mkt}, h, m} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h(m), t(m)} + \beta_{\text{def}, a(m), t(m)} + \epsilon_{h, m}$$
$$\log \lambda_{\text{mkt}, a, m} = \mu + \alpha_{\text{att}, a(m), t(m)} + \beta_{\text{def}, h(m), t(m)} + \epsilon_{a, m}$$

where:
- $\mu \in \mathbb{R}$ is the global league baseline log-intensity.
- $\gamma_{\text{home}} \in \mathbb{R}$ is the league home advantage.
- $\alpha_{\text{att}, i, t}$ is team $i$'s attack strength at time $t$.
- $\beta_{\text{def}, i, t}$ is team $i$'s defence weakness at time $t$ (positive = concedes more).
- $\epsilon_{h, m}, \epsilon_{a, m} \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma_{\text{obs}}^2)$ is Gaussian observation noise reflecting idiosyncratic match-level market pricing residual.

### 2.2 Identification Constraints

At each discrete time step $t$, team ratings must sum to zero:
$$\sum_{i=1}^{N_{\text{teams}}} \alpha_{\text{att}, i, t} = 0, \quad \sum_{i=1}^{N_{\text{teams}}} \beta_{\text{def}, i, t} = 0$$
This is strictly enforced via orthogonal projection / zero-centering:
$$\alpha_{t} = \tilde{\alpha}_t - \frac{1}{N}\sum_i \tilde{\alpha}_{i, t}$$

---

## 3. Dynamic Model Family (The 4 Arms)

Let $x_{i, t} \in \{\alpha_{\text{att}, i, t}, \beta_{\text{def}, i, t}\}$.

### Arm 1: 1st-Order Gaussian Random Walk (Constant Volatility)
Standard random walk with time-invariant innovation standard deviation:
$$x_{i, t} = x_{i, t-1} + \sigma_x \cdot \omega_{i, t}, \quad \omega_{i, t} \sim \mathcal{N}(0, 1)$$
- Priors: $\sigma_x \sim \text{HalfNormal}(0.10)$, $x_{i, 0} \sim \mathcal{N}(0, 0.25)$.

### Arm 2: 2nd-Order Momentum GRW (Damped Velocity)
Team ability has persistent velocity $v_{i, t}$, capturing continuous directional form:
$$x_{i, t} = x_{i, t-1} + v_{i, t-1} + \sigma_x \cdot \omega_{i, t}$$
$$v_{i, t} = \phi_x v_{i, t-1} + \sigma_{v, x} \cdot \eta_{i, t}$$
where:
- $\phi_x \in [0, 1)$ is momentum persistence.
- $\sigma_{v, x}$ is velocity innovation scale.
- Boundary condition: $v_{i, 0} = 0$.

### Arm 3: Stochastic Volatility GRW (Time-Varying $\sigma_t$)
The volatility of team ability updates is itself a dynamic process:
$$x_{i, t} = x_{i, t-1} + \sigma_{x, i, t} \cdot \omega_{i, t}$$
$$h_{i, t} \equiv \log \sigma_{x, i, t} = \bar{h} + \gamma (h_{i, t-1} - \bar{h}) + \sigma_h \cdot \xi_{i, t}$$
- High $h_t$ indicates a period of rapid tactical or personnel changes (high uncertainty).
- Low $h_t$ indicates settled, predictable team form.

### Arm 4: 2-State Regime-Switching GRW
Team updates switch between two discrete volatility regimes ($S_{i, t} \in \{1, 2\}$):
$$x_{i, t} = x_{i, t-1} + \sigma_{x, S_{i, t}} \cdot \omega_{i, t}$$
- Regime 1 (Calm): $\sigma_1$ (small step updates).
- Regime 2 (Turbulent): $\sigma_2 = \sigma_1 \times (1 + \Delta_\sigma)$ where $\Delta_\sigma > 0$ (large repricing updates, e.g. after manager departure or key injuries).
- Transition matrix $\mathbf{P} = \begin{pmatrix} p_{11} & 1-p_{11} \\ 1-p_{22} & p_{22} \end{pmatrix}$.

---

## 4. Benchmark Cohort & Data Pipeline

- **Primary Dataset**: Scottish Lower (tournaments 56 & 57, Scottish Championship / League 1).
- **Target Seasons**: 2024/25 & 2025/26 (40 matchday bi-weeks, 710 fixtures).
- **Market Data**: Closing Betfair 1X2 and Over/Under 2.5 odds extracted via `Data.load_datastore_cached(Data.ScottishLower())`.
- **Target Extraction**: Pre-compute and cache $(\lambda_{\text{mkt}, h}, \lambda_{\text{mkt}, a})$ using `Calibration.invert_market_rates(odds; match_ids = panel)`.
- **Exclusions**: Any fixtures failing the standard 4 inversion gates (`sse <= 0.05`, `selections >= 4`, etc.) are recorded in `refusals.csv`.

---

## 5. Evaluation Protocol & Diagnostics

1. **Prediction Error on Market Rates**:
   - Out-of-sample one-step-ahead log-rate RMSE, MAE, and predictive log-likelihood.
2. **Volatility & Form Profiles**:
   - Posterior distributions of baseline rate $\mu$, home advantage $\gamma_{\text{home}}$, innovation scales $\sigma$, velocity persistence $\phi$, and regime occupancy.
3. **Trajectory Visualizations**:
   - Comparative plots of team trajectories across the four arms (e.g. tracking Falkirk, Hamilton, Partick Thistle).
4. **Market Shock & Anomaly Catalog**:
   - Identify matchdays where $|\log \lambda_{\text{mkt}} - \hat{\log \lambda}| > 2.5 \sigma_{\text{obs}}$ to isolate real-world external shocks.

---

## 6. Phase 2 Roadmap (Post Initial Results)

Upon completing the core 4-arm benchmark:
1. **MS-GARCH Dynamics**: Formulate conditional shock clustering where market surprise directly drives subsequent volatility.
2. **Market Covariates**: Incorporate market spread and volume as uncertainty weights.
3. **Player Lineup / RAPM Synergy**: Integrate teamsheet ratings as dynamic covariates within the market rate decomposition.
