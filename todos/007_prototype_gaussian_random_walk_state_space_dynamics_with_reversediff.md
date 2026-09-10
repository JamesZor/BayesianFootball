# 007 — Prototype Gaussian Random Walk State Space Dynamics with ReverseDiff

| Field | Value |
|---|---|
| ID | 007 |
| Title | Prototype Gaussian Random Walk State Space Dynamics with ReverseDiff |
| Status | IN_PROGRESS |
| Priority | P2 |
| Assignee | pi |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [src/models/pregame/components/dynamics/team_level/multiscale.jl](../src/models/pregame/components/dynamics/team_level/multiscale.jl); [current_development/multiscale_grw/](../current_development/multiscale_grw/); `feat/multiscale-grw-dynamics` |

## Context & Problem Statement

Historically, team strength evolution in BayesianFootball.jl relied on exponential decay with a fixed half-life (`TimeDecayDynamics(days_half_life = 180.0)`) or static season ratings. True latent state-space Gaussian Random Walk (GRW) or AR(1) dynamics—where each team has time-varying latent attack/defence states $\alpha_{i,t} \sim \mathcal{N}(\alpha_{i,t-1}, \sigma_a^2)$ across $T$ match weeks—were previously discarded due to prohibitive MCMC execution times under ForwardDiff (where Dual-number overhead on 1,000+ latent parameters scaled quadratically or exhausted L3 cache).

With ReverseDiff tape compilation fully operational (producing 24× speedups and 5–8 minute 40-fold grids on `mcmc-beast`), high-dimensional latent state models are now computationally tractable. A single reverse sweep over the compiled tape evaluates the gradient of 1,000+ latent random-walk innovations in milliseconds.

We need to prototype, benchmark, and evaluate a true latent state-space GRW component against the current fixed half-life standard on Scottish Lower.

## Acceptance Criteria

- [ ] Implement a `GaussianRandomWalkDynamics` component in `src/models/pregame/components/dynamics.jl` conforming to `AbstractDynamicsComponent`.
- [ ] Implement a non-centered parameterization ($\alpha_{i,t} = \alpha_{i,0} + \sigma_a \sum_{k=1}^t z_{i,k}$, with $z_{i,k} \sim \mathcal{N}(0, 1)$) to prevent Neal's funnel geometry divergences in NUTS.
- [ ] Support either weekly discretization or continuous-time Brownian increments scaled by $\sqrt{\Delta t_{\text{days}}}$.
- [ ] Ensure AD-safety and zero-allocation execution inside the likelihood tape under ReverseDiff.
- [ ] Preflight on 2-fold CV to verify tape compilation, gradient validity, and absence of NUTS divergences.
- [ ] Execute a full 40-fold walk-forward grid on Scottish Lower (`scottish_lower_2426`) on `mcmc-beast` (-t 16).
- [ ] Persist run artifacts to PostgreSQL `mcmc_experiments` with `ad_backend = 'reversediff'`.
- [ ] Benchmark out-of-sample proper scores (LogLoss, CRPS, Brier, RPS) and sampling times against `TimeDecayDynamics(180.0)`.

## Ideas & Candidate Solutions

- **Non-Centered vs Centered Parameterization**: In sparse leagues with weekly observations, centered parameterizations ($\alpha_t \sim \mathcal{N}(\alpha_{t-1}, \sigma)$) suffer severe funnel geometry as $\sigma \to 0$. Non-centered formulation using unit-normal innovations $z_{i,t}$ is mandatory for robust NUTS sampling.
- **Discretization Strategy**:
  - *Weekly step grid*: Matches are mapped to calendar match-weeks $w \in \{1, \dots, W\}$. Vectorized over teams, easily formatted as a 2D matrix of innovations `z[team, week]`.
  - *Continuous-time Brownian motion*: Innovation variance scales as $\sigma^2 \Delta t_{ij}$ where $\Delta t$ is days elapsed since the team's prior match. Captures mid-week vs weekend scheduling but introduces ragged indexing.
- **Mean-Reversion (Ornstein-Uhlenbeck / AR(1))**: A pure random walk can diffuse to unphysical extremes over multi-year periods. Adding a mild mean-reversion parameter $\rho \in (0.90, 0.99)$ shrinks ratings back toward the league baseline:
  $$\alpha_{i,t} = \rho \alpha_{i,t-1} + \sigma \sqrt{1 - \rho^2} z_{i,t}$$
- **Innovation Scale Hyperpriors**: Prior scale on weekly drift $\sigma_a, \sigma_d \sim \text{HalfNormal}(0.05)$ to ensure team strength changes smoothly week-to-week rather than overfitting single match noise.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task in BACKLOG following ReverseDiff 24× speedup breakthrough and user proposal. Outlined mathematical formulation and acceptance criteria.
- [2026-09-10 @pi] Claimed in session `pi_solo:1`, worktree `/home/james/bet_project/.worktrees/BayesianFootball-grw-dynamics` on branch `feat/multiscale-grw-dynamics`; remote compute target `/root/BF_multiscale_grw` on `mcmc-beast`. Design locked via `/grill-me`: revive `MultiScaleGRW` in `current_development/multiscale_grw/` (l01_loader.jl + r01_runner.jl); Phase 1 (Poisson m00/m05) -> 2-fold preflight -> 40-fold grid on beast -> benchmark vs TimeDecayDynamics; Phase 2 (Two-arm Joint Gamma-Poisson m05) overnight.

## Verification & Findings

Not run yet. Record commands, pass/fail, wall time, convergence diagnostics (R̂, ESS, divergences), and comparative proper scores vs TimeDecayDynamics baseline.
