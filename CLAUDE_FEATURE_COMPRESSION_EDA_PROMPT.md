# Autonomous Execution Brief: Feature Compression & Player Ratings EDA

> [!CAUTION]
> **CRITICAL HARD CONSTRAINT: DO NOT USE `mcmc-beast`!**
> Do NOT ssh to `mcmc-beast`, do NOT launch jobs on `mcmc-beast`, and do NOT connect to `mcmc-beast`.
> `mcmc-beast` is currently 100% saturated with an active 40-fold MCMC production grid (`d09f_grid`).
> All EDA scripts, Julia executions, Ridge sweeps, regressions, and calculations MUST run 100% LOCALLY on this machine (`archpc`).
> Local machine has ample CPU/RAM, Julia, and cached data at `.cache/datastore_ScottishLower.jls`.


**Target Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-feature-compression-eda`
**Branch**: `feat/scottish-lower-feature-compression-eda`
**Target League**: Strictly Scottish Lower only (tournaments 56 and 57, seasons 24/25 + 25/26, 40 walk-forward folds, 710 fixtures)
**Deliverable**: Comprehensive empirical report `FEATURE_COMPRESSION_EDA_REPORT.md` + standalone reproducible Julia runner script.

---

## 1. Problem Statement & Background

In production models on Scottish Lower football, we observe severe **Bayesian shrinkage compression and favorite underpricing**:
- **Baseline `m12_joint_hybrid_synergy`**: Employs static time decay, `PlayerLineupPillar` (`:shots_rapm`, `w_bench = 0.10`), and `ProductionWealthCovariate` (`RichardsSigmoid`). Its empirical supremacy responsiveness slope is **0.3164** (vs 1.000 market parity), it commits **48.88%** of capital to longshots ($\ge 4.0$) and only **0.14%** to favorites ($\le 1.8$), and its maximum home win probability is capped at **57.8%** even when market odds imply 67–71% win probabilities.
- **Baseline `m05_joint_production_wealth_grw`**: When the `PlayerLineupPillar` was omitted, the supremacy slope jumped from **0.3164 to 0.5515–0.5779** (+82% decompression!) and maximum home win probability jumped from **57.8% to 70.08%**.
- However, even without player ratings, models still plateau at slope ~0.55–0.58.

### Core Hypotheses to Investigate
1. **Double Shrinkage in Player Ratings**:
   - Step 1: In `src/features/extractors/shots_plus_minus.jl`, RAPM player ratings are regularized via Ridge regression with $\lambda = 1000.0$. In a 10-team league with short seasons, this aggressively shrinks player ratings toward 0.
   - Step 2: When those already-shrunk ratings enter the Bayesian model (`src/models/pregame/builder/player_dynamics.jl`), they receive another zero-mean shrinkage prior ($w_{\text{att}}, w_{\text{def}} \sim \mathcal{N}(0, 0.3)$). In `EDA_REPORT.md`, posterior mean $w_{\text{att}}$ collapsed from $0.212$ to $0.100$ over time.
2. **Variance Cannibalization / Multi-Collinearity**:
   - Team Attack/Defense ($\alpha, \beta$), Squad Wealth ($\Delta W$), and Lineup RAPM all measure team quality.
   - When all three enter a log-linear predictor with zero-mean priors, they divide the variance, causing all coefficients to shrink simultaneously.
3. **Non-linear Saturation in Wealth**:
   - `RichardsSigmoid(23.0, 0.80, 2.0)` flattens at the tails, compressing large wealth disparities between dominant and minnow clubs.

---

## 2. Scope & Research Methodology

You are empowered to autonomously design and execute the statistical analysis. You should implement a reproducible runner (e.g. `current_development/feature_compression_eda/r01_feature_compression_eda.jl`) and produce `FEATURE_COMPRESSION_EDA_REPORT.md`.

### Recommended Statistical Tests & Diagnostics:
1. **Two-Stage Variance Loss Tracking**:
   - Quantify signal variance at each stage:
     $$\text{Var}(\text{raw player stats}) \longrightarrow \text{Var}(\text{Ridge RAPM } \hat{r}_i) \longrightarrow \text{Var}(w_{\text{att}} \cdot \text{Lineup})$$
   - Measure what percentage of true player differentiation is destroyed by Ridge $\lambda = 1000$ vs Bayesian prior shrinkage.
2. **Ridge $\lambda$ Sensitivity Sweep**:
   - Sweep $\lambda \in [10, 50, 100, 500, 1000]$ on Scottish Lower match text/shot commentary.
   - Measure player spread $\sigma(\hat{r})$ and team-level aggregated lineup spread $\sigma(\text{Lineup}_h - \text{Lineup}_a)$.
3. **Multi-Collinearity & Design Matrix Diagnostics**:
   - Compute correlation matrices and Variance Inflation Factors (VIF) between:
     - Team latent ratings ($\alpha_h - \alpha_a$)
     - Lineup difference ($\text{Lineup}_h - \text{Lineup}_a$)
     - Squad wealth difference ($\text{Wealth}_h - \text{Wealth}_a$)
     - Travel distance
   - Quantify condition numbers and variance decomposition proportions.
4. **Supremacy & Probability Attribution**:
   - On heavy favorite matches (e.g. Hamilton vs Queen of the South, Ross County vs Cove), decompose the net log-rate $\eta_h - \eta_a$ into constituent feature contributions.
   - Show how much each feature is muting or amplifying the favorite supremacy.
5. **Wealth Sigmoid vs Log-Wealth Contrast**:
   - Compare `RichardsSigmoid` vs raw log-ratio wealth `LogSumWealthFeature` in terms of tail separation for top vs bottom tier clubs.

---

## 3. Engineering & Environment Guidelines

- Work strictly in `/home/james/bet_project/.worktrees/BayesianFootball-feature-compression-eda`.
- Environment and caching: `.env` is present, `.cache` is symlinked to the main datastore cache (`datastore_ScottishLower.jls` is already present).
- Always use `using ThreadPinning; pinthreads(:cores)` and `LinearAlgebra.BLAS.set_num_threads(1)` in Julia scripts.
- Never hardcode or print database credentials.
- When done, format your findings in `FEATURE_COMPRESSION_EDA_REPORT.md` with clear tables, variance ratios, VIF values, and concrete recommendations for reforming the feature layer.
