# Momentum MultiScale GRW Dynamics (Scottish Lower POC)

> **Work Package**: TODO 022 — Prototype Momentum MultiScale GRW Dynamics  
> **Target**: Scottish Lower League Football (Tournaments 56 & 57, 40-fold walk-forward cohort, seasons 24/25 + 25/26, 710 fixtures)  
> **Harness**: Pi Solo Agent (`openai-codex/gpt-6-astra` with `--thinking high`) in tmux session `agent_pi_momentum_grw`  
> **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-momentum-grw`  
> **Branch**: `feat/scottish-lower-momentum-grw`  
> **Compute Node**: `mcmc-beast` (AMD Ryzen 9 32-core node)  

---

## 1. Executive Context & Motivation

Scottish Lower football models (`m12_joint_hybrid_synergy`, `m05_joint_production_wealth_grw`, and standard `MultiScaleGRW`) suffer from severe **Bayesian shrinkage compression**:
- In small 10-team leagues, standard hierarchical zero-mean Gaussian shrinkage on team attack/defense innovations aggressively regularises quality differences towards zero.
- As a result, net team supremacy ($\log(\lambda_h / \lambda_a)$) has an empirical slope of only **0.316–0.399** against de-vigged Betfair closing lines (where market slope is 1.00).
- When closing odds imply a dominant team has a 75–85% win probability (e.g. East Kilbride, Inverness), standard GRW models max out at **55–57%**.
- This compression creates phantom Kelly edges on high odds ($\ge 4.0$), forcing excessive bankroll allocation onto underdogs.

### Grounded Learnings from TODO 021 (Fast-Slow GRW Benchmark)
TODO 021 conclusively demonstrated:
1. **Goal Likelihood Pins Step Variance**: Widening Gaussian priors by $2.5\times$ only shifted posterior $\sigma_0$ from $0.192 \to 0.209$. The Poisson likelihood on low-scoring match goals strongly constrains independent step variance.
2. **Forced Spread Adds Orthogonal Noise**: Forcing $\sigma_0 \approx 0.48$ (`m04`) added unguided isotropic variance, dropping $R^2$ vs the close from $0.44 \to 0.34$ and worsening 1X2 log loss.
3. **Draw Mixtures are Bounded by Their Arms**: A posterior draw mixture ($P_{\text{mix}} = (1-\rho)P_1 + \rho P_2$) cannot evaluate beyond its constituent arms.

### The Breakthrough Hypothesis: 2nd-Order / Momentum GRW ("The Second Term")
In the standard 1st-order random walk (`MultiScaleGRW`):
$$\alpha_t = \alpha_{t-1} + \Delta_t, \quad \Delta_t \sim \mathcal{N}(0, \sigma^2)$$
Every step $\Delta_t$ is independent. Even after a 5-match winning streak, the prior expectation for the next step is strictly **zero**. The model resets to zero-mean shrinkage at every fixture.

By introducing **form momentum / velocity ($v_t$)**:
$$\begin{aligned}
\alpha_t &= \alpha_{t-1} + v_{t-1} + \sigma_\alpha \epsilon_{\alpha, t} && \text{(Position / Rating)} \\
v_t &= \phi v_{t-1} + \sigma_v \epsilon_{v, t} && \text{(Velocity / Form Momentum)}
\end{aligned}$$
- Runaway, dominant teams accumulate positive velocity over consecutive games ($v_t > 0$).
- At the next fixture, $\mathbb{E}[\alpha_t \mid \alpha_{t-1}, v_{t-1}] = \alpha_{t-1} + v_{t-1} > \alpha_{t-1}$.
- **Momentum directionally accelerates genuine favourites into the high-supremacy regime ($> 70\%$) without inflating isotropic noise on the rest of the league!**

---

## 2. Experimental Scope & 3-Arm Benchmark

To establish an unconfounded comparison, Phase 1 evaluates **pure minimal Poisson models** (zero complex covariates: no player lineups, no squad wealth, no smiles).

All models assemble via `CountModelBuilder`:
- Observation: `PoissonObservation()`
- Interception: `GlobalInterception()`
- Home Advantage: `GlobalHomeAdvantage()`
- Dynamics: (Varies across arms)

### The 3 Arms:
1. **`m01_poisson_time_decay` (Control 1)**:
   - Traditional exponential time-decay dynamics (`TimeDecayDynamics(days_half_life = 180.0)`).
2. **`m02_poisson_grw_1st_order` (Control 2)**:
   - 1st-order `MultiScaleGRW()` baseline (Task 013 / TODO 021 handrail).
3. **`m03_poisson_momentum_grw` (Candidate)**:
   - 2nd-order / Momentum MultiScale GRW architecture designed in Stage 0.
   - (Optional `m04_poisson_accel_grw`: 2nd-difference kinematic acceleration if you benchmark both variants).

---

## 3. Agent Operating Rules (CRITICAL)

- **SOLO AGENT EXECUTION**: You are operating as a solo agent with `openai-codex/gpt-6-astra` and `--thinking high`. **Do NOT invoke subagents.** Execute all mathematical research, code editing, command execution, and verification directly.
- **Deep Mathematical Thinking**: Leverage your high reasoning capabilities to derive the discrete state-space recursions, verify stationarity and damping conditions ($\phi \in [0, 1)$), and ensure seamless integration with the multiscale hierarchy.
- **ReverseDiff AD Performance & Zero Allocations**:
  - The model runs under compiled ReverseDiff gradient tapes.
  - Zero heap allocations in the inner likelihood loop.
  - Unroll state-space recursions using linear vector operations or `cumsum` rather than dynamic loops with mutating arrays.
  - Set `BLAS.set_num_threads(1)` and `ThreadPinning.pinthreads(:cores)`.
- **Database Separation**:
  - `betdb` operational data is on `archpc:5433` (via `BF_DB_URL`).
  - `mcmc_experiments` is on `mcmc-beast:5432` (via `BF_EXPERIMENTS_DB_URL`).
  - Save completed production fits to PostgreSQL namespace `scottish_lower_momentum_grw`.

---

## 4. Execution Stages

### Stage 0: Mathematical Research & Architecture Design
- Study `src/models/pregame/components/dynamics/team_level/multiscale.jl`.
- Formulate `MomentumMultiScaleGRW`:
  - Macro transitions: season-to-season level shifts.
  - Micro transitions: in-season fixture steps with autoregressive velocity $v_t$.
  - Prior specifications for $\sigma_\alpha, \sigma_v$, and persistence $\phi$ (e.g. $\phi \sim \text{Beta}(2, 2)$ or truncated $\mathcal{N}(0.7, 0.15)$).
- Implement `l10_momentum_grw_loader.jl` with Turing model function and latent reconstruction logic.

### Stage 1: Smoke Gate (Folds 1, 20, 40)
- File: `r10_momentum_smoke.jl`.
- Budget: 4 chains $\times$ (400 warmup + 400 samples).
- Gates:
  - G1: ReverseDiff gradient tape compiles and evaluates without errors.
  - G2: 0 NUTS divergences across all smoke chains.
  - G3: Max $\hat{R} \le 1.05$ (advisory $\le 1.01$).
  - G4: Bulk & tail ESS $\ge 200$.
  - G5: Score-grid book sums to $1.0$ within machine precision ($\le 1e-12$).
  - G6: Verification of momentum parameter posterior: confirm whether $\phi$ and $\sigma_v$ are identified and whether favourite supremacy expands.

### Stage 2: 40-Fold Walk-Forward Production Grid on Beast
- File: `r20_momentum_production_grid.jl`.
- Budget: 40 folds $\times$ 4 chains $\times$ (800 warmup + 800 samples) = 3,200 draws per fold (128,000 draws per arm).
- Compute node: Run on `mcmc-beast` in a dedicated background tmux session.
- Persist runs to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_momentum_grw`.

### Stage 3: Unified Evaluation & Portfolio Backtest
- File: `r30_momentum_evaluation.jl`.
- Proper scores vs de-vigged Betfair closing lines:
  - 1X2 LogLoss, Over/Under 2.5 LogLoss, BTTS LogLoss.
  - CRPS, RPS, ECE.
- Decompression Diagnostics:
  - Supremacy slope vs Betfair closing supremacy (target: move slope from ~0.39 closer to 1.00).
  - Favourite tail win-probability calibration on matches where Betfair close $\ge 0.70$ (target: expand beyond 57% towards market average 76%).
  - Capital allocation at extreme odds ($\ge 4.0$ vs $\le 1.8$).
- Portfolio Backtest:
  - Policy: `BookSpec(1X2, OU2.5, BakerMcHale)`, `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`.
  - Headline metrics: Total Return, Annualized Sharpe, Max Drawdown, Flat ROI.

### Stage 4: Documentation & Findings
- Write comprehensive results in `README.md`.
- Compare `m01_time_decay`, `m02_grw_1st_order`, and `m03_momentum_grw`.
- Sign off `todos/022_prototype_momentum_multiscale_grw_dynamics.md` and verify `./scripts/todo.sh check`.
