# WORK PACKAGE: Bayesian Shrinkage Decompression & Favorite Underpricing for Scottish Lower
# Target Model: openai-codex/gpt-6-astra (thinking: high)
# Working Branch: feat/scottish-lower-shrinkage-decompression
# Working Worktree: /home/james/bet_project/.worktrees/BayesianFootball-shrinkage-decompression
# Working Directory: experiments/scottish_lower/09_bayesian_shrinkage_decompression/
# Remote Compute Node: root@mcmc-beast (32 threads, 64 GB RAM, PostgreSQL mcmc_experiments:5432)
# Operational Database: BF_DB_URL (betdb on archpc:5433)
# Tracking TODO: todos/015_resolve_bayesian_shrinkage_compression_and_favorite_underpricing_in_scottish_low.md

> **Agent Role**: Principal Bayesian Statistician & MCMC Inference Researcher  
> **Target Model**: `openai-codex/gpt-6-astra` (Reasoning: High, Context: 272K)  
> **Mission**: Prove and quantify the Bayesian shrinkage compression / probability ceiling in current Scottish Lower models (`m12_joint_hybrid_synergy` and `m05_joint_production_wealth_grw`), formulate and implement hierarchical decompression mechanisms, and benchmark them across the 40-fold walk-forward grid on `mcmc-beast`.

---

## 1. Executive Context & Problem Statement

### A. The Operational Failure Mode (2026-09-19 Live Slate)
In live MatchDay operations on the 2026-09-19 Scottish Lower slate (tournaments 56 & 57):
1. **The ~60% Probability Ceiling**:
   - In matches with heavy market favorites:
     - **Ross County vs Cove Rangers**: Betfair closing implied probability was **67.0%** for Ross County. The production model `m12_joint_hybrid_synergy` predicted only **48.2%** for Ross County and **29.7%** for Cove (market 17.5%).
     - **Hamilton vs Queen of the South**: Betfair closing implied probability was **71.0%** for Hamilton. The model predicted only **54.2%** for Hamilton and **25.2%** for Queen of the South (market 13.5%).
   - Across the entire historical dataset, the model rarely predicts a win probability $> 60\%$, regardless of the objective talent mismatch between the clubs.
2. **Phantom Kelly Edges on Extreme Underdogs**:
   - Because the model severely underprices the favorites, its residual probability mass is forced onto draws and underdogs.
   - On Cove Rangers, the model saw a **+23.8% edge** at odds of 5.70. On Queen of the South, it saw a **+14.4% edge** at odds of 7.40.
   - Under fractional Kelly staking, these two bets alone consumed **46% of all staked capital** (£20.35 of £43.53 total slate budget). Both lost, contributing directly to negative slate ROI (-14.0% on m12, -18.2% on m05).
   - In raw (uncalibrated) execution, this longshot phantom edge was even worse, staking >£80 across 20 bets and losing -£17.72.

### B. Mathematical Root Cause: Triple Shrinkage Compression
Inspection of posterior parameters for Fold 43 of `m12_joint_hybrid_synergy` revealed the exact structural bottleneck:
1. **Shrinkage Priors on Small Samples**:
   - In a 10-team league with 36 matches per season, zero-mean Gaussian priors on team attack/defense innovations shrink team ratings aggressively toward the league mean (0.0).
   - Posterior dynamic variances are tiny: $\sigma_a \approx 0.041$, $\sigma_d \approx 0.054$.
   - Outfield player RAPM weights are constrained: $w_{\text{att}} \approx 0.101$, $w_{\text{def}} \approx 0.188$.
   - Squad wealth weight is $w_{\text{wealth}} \approx 0.151$.
   - **Theoretical Supremacy Upper Bound**: The maximum conceivable net log-intensity difference between the richest squad and the poorest squad in the posterior is:
     $$\Delta \eta_{\max} = \eta_h - \eta_a \approx \text{HA} (0.24) + \Delta \alpha (0.15) + \Delta \text{wealth} (0.22) + \Delta \text{RAPM} (0.18) \approx 0.79$$
     In a Poisson/NegBin bivariate grid, a log-rate gap of 0.79 produces a maximum home win probability of approximately **58% to 61%**. The model is mathematically incapable of representing dominant 70%+ favorites.
2. **Richards Sigmoid Wealth Saturation**:
   - `ProductionWealthCovariate` applies a sigmoid transform to squad market values:
     $$S(v) = \frac{1}{(1 + \nu e^{-\beta (v - v_0)})^{1/\nu}}$$
     This squashes extreme financial disparities between top clubs (e.g., Premiership-budget relegated clubs) and part-time Lower clubs into a narrow $[-1, 1]$ interval.
3. **Calibrator Tail Distortion**:
   - The Layer 2 calibrator `InverseGaussianLaw(w_base = 0.25, sigma = 0.35)` sets pooling weight $w(\Delta) \to 1.0$ as $|\Delta| \to \infty$.
   - When a massive model-market discrepancy occurs on a longshot, the calibrator assumes the model possesses pure structural alpha and refuses to shrink toward the market, unleashing maximum Kelly stakes on extreme underdogs.

---

## 2. Infrastructure, Topography & Conventions

### A. Environment Topography
| Role | Host | Details |
|---|---|---|
| **Local Workstation** | `archpc` | Where Pi agent runs in tmux (`agent_pi_shrinkage_decompression`), worktree `/home/james/bet_project/.worktrees/BayesianFootball-shrinkage-decompression` |
| **Compute Server** | `mcmc-beast` | Dedicated MCMC node (32 CPU cores, 64 GB RAM), reachable via `ssh root@mcmc-beast`. Repository clone at `/root/BayesianFootball`. Runs Julia with `/root/.juliaup/bin/julia --project -t 32`. |
| **Operational DB** | `archpc:5433` | `BF_DB_URL` (`betdb`: `sofascore`, `bbc`, `betfair`, `betfair_live`, `paper_runbook`). Reached from Julia via `Data.load_datastore_sql` / `load_datastore_cached`. |
| **Experiment DB** | `mcmc-beast:5432` | `BF_EXPERIMENTS_DB_URL` (`mcmc_experiments`). Local to `mcmc-beast`. Stores `configs`, `runs`, `fold_results`, `match_latents`, `fit_artifacts`, and `portfolio_bets`. |

### B. Julia Language Traps & Guidelines (from `AGENTS.md`)
1. **ReverseDiff AD Performance**:
   - Precompile gradient tapes using `ReverseDiff.compile(ReverseDiff.GradientTape(f, x))`.
   - Never introduce dynamically sized vectors, runtime type instability, or dictionary lookups inside the model log-density.
   - Vectorised operations (`@.` or broadcasting) must use preallocated buffers or statically sized arrays.
2. **Experiment Database Integrity**:
   - Always access experiment results via `Training.PostgresStorage("scottish_lower")` or direct SQL on `mcmc_experiments`.
   - Do NOT modify schema migrations or existing experiment records.
3. **Tmux Protocol for Remote Execution**:
   - Any long-running MCMC run (> 2 minutes) on `mcmc-beast` **MUST** be launched inside a detached `tmux` session (e.g. `ssh root@mcmc-beast "tmux new-session -d -s r09_decompression 'cd /root/BayesianFootball && ...'"`).
   - Never leave a foreground SSH command running that could break on network timeouts.
   - Monitor via `tmux capture-pane -pt <session_name> -S -50` or log tailing.

---

## 3. The 3-Stage Research Plan

### STAGE 1: Latents & Calibration EDA (`r09_eda_latent_compression.jl`)
**Objective**: Empirically and mathematically prove shrinkage compression across all 43 historical folds for `m12_joint_hybrid_synergy` and `m05_joint_production_wealth_grw`.

1. **Extract Out-of-Sample Predictions vs Closing Market**:
   - Connect to `mcmc_experiments` and load the out-of-sample predictions (`match_latents` and evaluated market probabilities) for all 43 folds (seasons 24/25 and 25/26, 710 matches).
   - Join with closing Betfair implied probabilities for 1X2 and O/U 2.5.
2. **Compute Empirical Diagnostics**:
   - **Supremacy Scatter & Slope**: Regress model supremacy $(\log \lambda_h - \log \lambda_a)$ against market supremacy $(\text{logit}(P_{\text{mkt, home}}) - \text{logit}(P_{\text{mkt, away}}))$. Quantify the slope $\beta_{\text{supremacy}}$ (expected $\beta < 0.65$, confirming severe compression).
   - **Probability Saturation / Ceiling**: Plot binned market win probabilities ($[0.0, 0.1], \dots, [0.7, 1.0]$) vs model win probabilities. Pinpoint the exact empirical ceiling (e.g. $P_{\text{model}} \le 0.58$ when $P_{\text{mkt}} \ge 0.70$).
   - **Parameter Posterior Analysis across 43 Folds**:
     - Plot fold-by-fold trajectories and distributions for $\sigma_a$, $\sigma_d$, $\kappa$, $w_{\text{wealth}}$, $w_{\text{att}}$, $w_{\text{def}}$.
     - Check whether dynamic variances $\sigma$ are monotonically shrinking over time as match sample size increases.
   - **Kelly Phantom Edge Accumulation**:
     - Correlate model edge $(P_{\text{model}} - P_{\text{mkt}})$ with market odds.
     - Calculate the cumulative Kelly stake allocation that was routed to selections at odds $\ge 4.0$ vs odds $\le 1.80$.
3. **Deliverable**:
   - Script: `r09_eda_latent_compression.jl`
   - Scientific Report: `EDA_REPORT.md` with markdown tables, regression coefficients, and empirical saturation curves.

---

### STAGE 2: Model Formulation & Candidate Decompression Mechanics (`l09_decompression_models.jl`)
**Objective**: Build composable Turing model components that structurally relax the supremacy ceiling without destroying convergence.

Prioritize and formulate the four mechanisms selected by the user:

#### Candidate 1: Team-Level Dynamic Heterogeneity (`m01_hierarchical_sigma` & `m01b_hierarchical_ha`)
- **Heterogeneous Team GRW Volatilities ($\sigma_{i, a}, \sigma_{i, d}$)**:
  - Instead of a single pooled $\sigma_a, \sigma_d$ for all 10 teams, allow team-specific random walk variance:
    $$\log \sigma_{i, a} = \mu_{\sigma, a} + \tau_{\sigma, a} \cdot z_{i, a}, \quad z_{i, a} \sim \text{Normal}(0, 1)$$
    $$\log \sigma_{i, d} = \mu_{\sigma, d} + \tau_{\sigma, d} \cdot z_{i, d}, \quad z_{i, d} \sim \text{Normal}(0, 1)$$
    $$\mu_\sigma \sim \text{Normal}(\log 0.05, 0.5), \quad \tau_\sigma \sim \text{HalfNormal}(0.30)$$
  - This allows volatile, high-turnover or newly relegated dominant clubs to move dynamically without being clamped by stable mid-table clubs.
- **Hierarchical Home Advantage ($h_i$)**:
  - Incorporate team-specific pitch/stadium home advantage: $h_i = h_{\text{global}} + \sigma_h \cdot z_{h, i}$, zero-sum centered ($\sum z_{h, i} = 0$).

#### Candidate 2: Hierarchical Finishing Factor ($\kappa_{\text{team}}$) (`m02_hierarchical_kappa`)
- **Two-Arm Team-Specific Conversion**:
  - In the Two-Arm Joint Gamma-Poisson observation:
    - Arm 1 (proxy xG): $\text{pxg}_s \sim \text{Gamma}(\nu, \mu_s / \nu)$
    - Arm 2 (goals): $y_s \sim \text{Poisson}(\kappa_{t(s)} \cdot \mu_s)$
  - Hierarchical conversion factor:
    $$\log \kappa_i = \log \kappa_{\text{global}} + \delta_{\kappa}[i]$$
    $$\delta_{\kappa} = \sigma_{\kappa} \cdot (z_{\kappa} - \bar{z}_{\kappa}), \quad z_{\kappa} \sim \text{Normal}(0, 1), \quad \sigma_{\kappa} \sim \text{HalfNormal}(0.10)$$
  - Clinical, dominant teams with superior strikers (e.g. Premiership loans) sustain $\kappa_i > 1.0$, while struggling sides have $\kappa_i < 0.85$.

#### Candidate 3: Uncompressed Financial & Pedigree Covariates (`m03_uncompressed_wealth`)
- **Relaxing the Richards Sigmoid**:
  - Replace the squashed $[-1, 1]$ sigmoid with:
    1. **Log-Ratio Wealth**: $\Delta \log W = \log(\text{wealth}_h + \epsilon) - \log(\text{wealth}_a + \epsilon)$.
    2. **Pedigree Indicator**: Binary or categorical indicator for clubs relegated from the Scottish Premiership within the last 2 seasons:
       $$\eta_{\text{pedigree}} = \gamma_{\text{rel}} \cdot (\mathbb{I}_{\text{home relegated}} - \mathbb{I}_{\text{away relegated}})$$
  - Prior: $\gamma_{\text{rel}} \sim \text{Normal}(0.25, 0.15)$. This injects direct supremacy separation for clubs that carry Premiership-caliber squads.

#### Candidate 4: Fatter-Tailed Dynamic Innovations (`m04_student_t_dynamics`)
- **Student-$t$ Random Walk Innovations**:
  - Gaussian random walks penalize large sudden talent shifts quadratically ($\Delta^2 / (2\sigma^2)$), driving strong shrinkage toward zero.
  - Implement Student-$t$ distributed innovations with fixed or estimated degrees of freedom ($\nu \in [4, 7]$):
    $$\Delta \alpha_{t, i} \sim \text{StudentT}(\nu, 0, \sigma_a)$$
  - This allows rare, massive quality divergence between clubs without inflating the variance of the other teams.

#### Candidate 5: Unified Decompressed Synergy (`m05_decompressed_synergy`)
- Combine Candidate 1 (hierarchical $\sigma_i$), Candidate 2 (hierarchical $\kappa_i$), Candidate 3 (uncompressed wealth + pedigree), and Candidate 4 into a single composite architecture.

---

### STAGE 3: Verification Ladder & 40-Fold Grid Execution

#### A. Single-Fold Smoke Ladder (`r09_smoke.jl`)
Before executing any multi-fold grid, each candidate must pass the strict 8-Gate Verification Ladder on Fold 43:
- **G1 (Tape Compilation)**: ReverseDiff gradient tape compiles in $< 200\text{ms}$ with zero runtime allocations.
- **G2 (Sampler Stability)**: NUTS sampler (4 chains $\times$ 800 warmup $\times$ 800 retained draws) completes without crashing or stuck steps.
- **G3 (Convergence Audit)**:
  - $\max \hat{R} \le 1.05$ across all latent parameters.
  - $\min \text{bulk-ESS} \ge 300$, $\min \text{tail-ESS} \ge 200$.
  - Divergences $\le 0.1\%$ of post-warmup draws (ideally 0).
- **G4 (Identification)**: Zero-sum constraints verified ($\sum \alpha_i = 0$, $\sum \beta_i = 0$, $\sum \delta_{\kappa, i} = 0$).
- **G5 (Supremacy Decompression)**: Verify that candidate maximum attainable supremacy on Fold 43 exceeds $1.10$ log-rate difference (allowing $> 68\%$ win probabilities).
- **G6 (Latent Extraction)**: `CountLatents` correctly extracts positive, finite posterior draws for every match.
- **G7 (Score Grid Pricing)**: `SmileScoreGrid` and `ScoreGrid` kernels compute valid probability simplices ($P_{1X2} = 1.0$, $P_{O/U} = 1.0$).
- **G8 (Storage Roundtrip)**: `save_fit` and `load_fit` serialize and deserialize identically via PostgreSQL `mcmc_experiments`.

#### B. 40-Fold Walk-Forward Grid on `mcmc-beast` (`r09_production_grid.jl`)
1. Sync repository to `mcmc-beast`:
   ```bash
   git push origin feat/scottish-lower-shrinkage-decompression
   ssh root@mcmc-beast "cd /root/BayesianFootball && git pull origin feat/scottish-lower-shrinkage-decompression"
   ```
2. Launch production grid in persistent tmux session:
   ```bash
   ssh root@mcmc-beast "tmux new-session -d -s r09_decompression 'cd /root/BayesianFootball && export BF_EXPERIMENTS_DB_URL=postgresql://postgres:football_mcmc_secure@localhost:5432/mcmc_experiments && /root/.juliaup/bin/julia --project -t 32 experiments/scottish_lower/09_bayesian_shrinkage_decompression/r09_production_grid.jl | tee experiments/scottish_lower/09_bayesian_shrinkage_decompression/results/production_grid.log'"
   ```
3. Monitor progress safely without polling:
   - Check status periodically or tail the log:
     ```bash
     ssh root@mcmc-beast "tmux capture-pane -pt r09_decompression -S -30"
     ```

#### C. Proper Scoring & Portfolio Benchmarking (`r09_evaluate.jl` & `r09_portfolio.jl`)
1. **Out-of-Sample Proper Scores**:
   - Compare against canonical benchmarks:
     - `m12_joint_hybrid_synergy`: LogLoss **0.6568**, ECE **0.0100**
     - Betfair Closing Line: LogLoss **0.6568**
   - Specifically measure **Tail Brier Score** for $P_{\text{mkt}} > 0.60$ and $P_{\text{mkt}} < 0.15$. Did decompression eliminate tail underconfidence without increasing false positives?
2. **Portfolio Simulation**:
   - Run standard `BookSpec` (1X2 + O/U 2.5) with `PolicySpec(trust = FlatTrust(0.25), risk = SlateDrawdown(20.0))`.
   - Measure:
     - Total P&L & ROI.
     - Maximum Drawdown.
     - Longshot capital allocation ratio (proportion of bankroll staked at odds $> 4.0$).
     - Realized Sharpe ratio.

---

## 4. File Structure & Deliverables

All artifacts must be created in:
`experiments/scottish_lower/09_bayesian_shrinkage_decompression/`

```
experiments/scottish_lower/09_bayesian_shrinkage_decompression/
├── WORK_PACKAGE_PROMPT.md            <-- This document
├── EDA_REPORT.md                     <-- Mathematical & empirical audit of shrinkage compression
├── l09_decompression_models.jl       <-- Turing model components & builders
├── r09_eda_latent_compression.jl     <-- Stage 1 EDA script
├── r09_smoke.jl                      <-- Stage 2 Verification Ladder (G1-G8)
├── r09_production_grid.jl            <-- Stage 3 40-Fold parallel MCMC grid runner
├── r09_evaluate.jl                   <-- Proper score & tail calibration evaluator
├── r09_portfolio.jl                  <-- Zero-alloc portfolio simulation runner
├── results/                          <-- Logs, manifests, and score CSVs
└── README.md                         <-- Summary, candidate comparisons, and final conclusions
```

---

## 5. Success Criteria & Definition of Done

1. **Empirical Proof**: `EDA_REPORT.md` conclusively proves the presence, magnitude, and financial consequence of shrinkage compression in `m12` and `m05`.
2. **Decompression Validated**: At least one candidate model successfully predicts home win probabilities $> 68\%$ for heavy favorites where justified, with an empirical supremacy slope $\beta_{\text{supremacy}} \ge 0.85$.
3. **Verification Ladder Passed**: All candidate models pass G1 through G8 with $\hat{R} < 1.05$ and 0 divergences.
4. **Predictive Performance**: Out-of-sample 1X2 LogLoss matches or beats Betfair closing line ($\le 0.6568$), with a statistically significant reduction in Tail Brier Score.
5. **Portfolio Capital Discipline**: Phantom edges on extreme longshots are eliminated or curtailed, reducing drawdown and improving realized Sharpe ratio.
6. **TODO Registry Complete**: `todos/015_resolve_bayesian_shrinkage_compression_and_favorite_underpricing_in_scottish_low.md` updated with full work logs, benchmark tables, and verified run IDs.
