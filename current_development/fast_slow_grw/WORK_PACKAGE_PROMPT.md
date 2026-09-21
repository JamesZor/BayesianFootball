# Fast and Slow GRW Models & Geometric Rate Pooling (Scottish Lower POC)

> **Work Package**: TODO 021 — Prototype Fast-Slow GRW Rate Pooling and Decompression  
> **Target**: Scottish Lower League Football (Tournaments 56 & 57, 40-fold walk-forward cohort, seasons 24/25 + 25/26, 710 fixtures)  
> **Harness**: Claude CLI Agent in tmux session `agent_claude_fast_slow_grw`  
> **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-fast-slow-grw`  
> **Branch**: `feat/scottish-lower-fast-slow-grw-poc`  
> **Compute Node**: `mcmc-beast` (High-performance AMD Ryzen 9 32-core node)  

---

## 1. Executive Context & Motivation

Scottish Lower football models in this repository (`m12_joint_hybrid_synergy`, `m05_joint_production_wealth_grw`) suffer from **Bayesian shrinkage compression**:
In small 10-team leagues, standard hierarchical zero-mean Gaussian shrinkage on team attack/defense innovations aggressively regularises quality differences towards zero. As a result, net team supremacy ($\log(\lambda_h / \lambda_a)$) has an empirical slope of only **0.316–0.328** against de-vigged Betfair closing lines. When closing odds imply a dominant team has a 75–85% win probability (e.g. East Kilbride, Inverness), the model win probability maxes out at **57.8%**. This underconfidence forces residual probability mass onto extreme longshots, generating phantom Kelly edges on high odds ($\ge 4.0$) that cause portfolio drawdowns.

### The Grounded Diagnosis (Claude's Feature EDA)
A comprehensive read-only empirical audit across all 40 folds (`FEATURE_COMPRESSION_EDA_REPORT.md`) recently established that:
1. **Player ratings are NOT compressed**: The market closing lines weight lineup RAPM ($\Delta L$) at **$1.00\times$** the model's posterior weight (exact market parity). Ridge penalty $\lambda = 1000$ is optimal.
2. **Wealth sigmoid is NOT compressed**: `RichardsSigmoid` is an age-discount curve, not a wealth compressor.
3. **The compression is 100% in the team latent $\alpha/\beta$**: Regressing market supremacy on model components shows the market wants the team latent $\alpha/\beta$ amplified by **$2.43\times$**!

### The Solution: "Handrailing and Aiming Off" with Fast & Slow Models
The user proposed a dual-model architecture based on the navigation technique of **"handrailing and aiming off"**:
* In low-visibility navigation, you deliberately "aim off" toward a known linear handrail (a road, river, or coastline) so that upon hitting the handrail, you know with certainty which direction to turn to reach your checkpoint.
* **The Slow / Tight Model (The Handrail)**: High shrinkage, long memory, tight Gaussian priors on team innovations. It never overreacts to short streaks, small-sample flukes, or red cards. It protects your bankroll and bounds tail risk, but it systematically "aims off" by shrinking favourites to 50–58%.
* **The Fast / Loose Model (The Scout / Innovator)**: Relaxed shrinkage ($2.5\times$ prior scale on team variance $\sigma_0$) or heavy-tailed Student-$t$ innovations (`TDist(4.0)`). It reflects genuine market separation, allowing favourites to reach 75–85% win probabilities. On its own, it would be too volatile on noisy small-sample matches.
* **The Combination via Geometric Rate Pooling**: Because we know the exact direction the tight model aims off (inward toward parity), we blend their posterior log-rates:
  $$\log \lambda_{i, \text{blend}} = (1-w)\log \lambda_{i, \text{tight}} + w \log \lambda_{i, \text{loose}}$$
  $$\lambda_{i, \text{blend}} = (\lambda_{i, \text{tight}})^{1-w} \cdot (\lambda_{i, \text{loose}})^w$$

### Why Geometric Rate Pooling ($\lambda$-space)?
Unlike averaging probabilities ($P = (1-w)P_1 + w P_2$), which breaks derivative coherence across 1X2, Totals, and BTTS:
1. The blended $(\lambda_{h, \text{blend}}, \lambda_{a, \text{blend}})$ feeds directly into our bivariate score-grid kernels (`SmileScoreGrid`, bivariate Poisson, Frank copula). 1X2, Over/Under, and BTTS remain exact marginal partitions of a single 12×12 joint score-line tensor.
2. It linearly expands supremacy in log-rate space:
   $$\text{Supremacy}_{\text{blend}} = (1-w)\text{Supremacy}_{\text{tight}} + w \cdot \text{Supremacy}_{\text{loose}}$$

---

## 2. Experimental Scope & Candidate Models

To establish an unconfounded proof-of-concept, we start with **pure minimal Poisson GRW models** (zero complex covariates: no player lineups, no squad wealth, no smiles).

All models assemble via `CountModelBuilder`:
- Observation: `PoissonObservation()`
- Interception: `GlobalInterception()`
- Home Advantage: `GlobalHomeAdvantage()`
- Dynamics: `MultiScaleGRW(...)`

### Model Candidates:
1. **`m01_poisson_grw_tight` (Baseline Handrail)**:
   Standard `MultiScaleGRW` priors:
   - $z_0, z_s, z_k \sim \mathcal{N}(0, 1)$
   - $\alpha_{\sigma_0} \sim \text{Gamma}(2, 0.06)$, $\beta_{\sigma_0} \sim \text{Gamma}(2, 0.10)$
   - $\alpha_{\sigma_s} \sim \text{Gamma}(2, 0.03)$, $\beta_{\sigma_s} \sim \text{Gamma}(2, 0.055)$
   - $\alpha_{\sigma_k} \sim \text{Gamma}(2, 0.015)$, $\beta_{\sigma_k} \sim \text{Gamma}(2, 0.012)$

2. **`m02_poisson_grw_loose_var` (Loose Candidate A: Variance Scaling)**:
   Scale up baseline team spread priors by $\sim 2.5\times$ to directly match the market's implied $2.43\times$ requirement:
   - $\alpha_{\sigma_0} \sim \text{Gamma}(2, 0.15)$ ($2.5\times$ of 0.06)
   - $\beta_{\sigma_0} \sim \text{Gamma}(2, 0.25)$ ($2.5\times$ of 0.10)
   - Step variances $\sigma_s, \sigma_k$ optionally widened or kept moderate.

3. **`m03_poisson_grw_loose_tdist` (Loose Candidate B: Heavy-Tailed Student-$t$)**:
   Heavy-tailed innovations on initial team spread and season transitions:
   - $z_0 \sim \text{TDist}(4.0)$
   - $z_s \sim \text{TDist}(4.0)$
   - Allows genuine outlier teams (dominant favourites and helpless cellar teams) to separate without forcing Gaussian tails.

---

## 3. Staged Execution Protocol

### Stage 1: Smoke / POC Validation (3–5 Folds)
- Run `r01_fast_slow_smoke.jl` on folds 1, 20, 40 (or 1–3).
- **Verification Gates**:
  - G1: AD tape builds cleanly with ReverseDiff.
  - G2: MCMC sampling finishes with 0 divergences.
  - G3: Gelman-Rubin $\hat{R} \le 1.05$ across all parameters.
  - G4: Bulk ESS $\ge 200$ (target $\ge 300$).
  - G5: Checkpoint serialization round-trip (`.jls` and `mcmc_experiments`).
  - G6: **Supremacy Slope Check**: Verify that `m02` and `m03` expand the supremacy slope beyond `m01` on these folds.
  - G7: Verify that `blend_rates(tight_fit, loose_fit; w=0.4)` executes and produces valid score grids.

### Stage 2: Full 40-Fold Walk-Forward Grid on `mcmc-beast`
- Run `r02_fast_slow_production_grid.jl` on `mcmc-beast` across all 40 folds of seasons 24/25 and 25/26 (710 fixtures).
- Store checkpoints to disk: `results/production/<model_name>/checkpoints_4x800w800s/fold_<k>.jls`.
- Record run metadata to PostgreSQL `mcmc_experiments`.

### Stage 3: Geometric Rate Pooling & Headline Benchmark
- Run `r03_fast_slow_evaluation_and_blend.jl`.
- Sweep rate pooling weights $w \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$.
- For each blend (and pure models $w=0, w=1$), evaluate against de-vigged Betfair closing odds and run `simulate_portfolio` with production policy (`BookSpec(1X2, OU2.5, BakerMcHale)`, `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`).
- Report the **6 Headline Metrics**:
  1. **Supremacy Slope** (OLS of model supremacy on market supremacy)
  2. **Capital on Odds $\ge 4.0$** (%)
  3. **Capital on Odds $\le 1.8$** (%)
  4. **Max Drawdown** (%)
  5. **Annual Sharpe Ratio**
  6. **Flat Staking ROI** (%)
- Also compute an auxiliary comparison against linear probability pooling ($P_{\text{blend}} = (1-w)P_{\text{tight}} + w P_{\text{loose}}$) to verify that rate pooling avoids probability distortions on totals and BTTS.

### Stage 4: Documentation & Findings
- Produce `FAST_SLOW_GRW_REPORT.md` documenting:
  - Table of the 6 headline metrics across all weights $w$.
  - Comparison of Loose Candidate A (variance scaling) vs Loose Candidate B (Student-$t$).
  - Probability tail distribution: win probability on heavy favourites (market $\ge 0.70$) vs baseline.
  - Derivative market consistency (1X2, Over/Under 2.5, BTTS).
  - Recommendations for Phase 2 (Two-Arm Joint Gamma-Poisson GRW).

---

## 4. Environment & Execution Guardrails

1. **Working Directory & Worktree**:
   - Worktree path: `/home/james/bet_project/.worktrees/BayesianFootball-fast-slow-grw`
   - Branch: `feat/scottish-lower-fast-slow-grw-poc`
2. **Threads & BLAS**:
   - `using ThreadPinning; pinthreads(:cores)`
   - `using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)`
3. **Database Guardrails**:
   - Operational data: PostgreSQL `betdb` on `archpc:5433` (read via `ENV["BF_DB_URL"]`).
   - Experiment results: PostgreSQL `mcmc_experiments` on `mcmc-beast:5432` (`Training.PostgresStorage("fast_slow_grw_scottish_lower")`).
   - Never print raw database passwords or credentials.
4. **Remote Execution on `mcmc-beast`**:
   - `mcmc-beast` is currently free and idle.
   - Run production sampling inside a tmux session on `root@mcmc-beast` or use git pull / push between `archpc` and `mcmc-beast` as documented in `docs/setup/agy_remote_execution_guide.md`.
