# Work Package: Cross-Tier Scottish Pyramid and Informative Priors Time-Decay Models

**Tracking:** [TODO 028](../../../todos/028_cross_tier_scottish_pyramid_and_informative_priors_time_decay_models.md)  
**Assigned Agent:** Pi (`openai-codex/gpt-6-astra`)  
**Host Target:** `mcmc-beast` (64 cores, 256GB RAM, local `mcmc_experiments` Postgres)  
**Branch:** `feat/scotland-cross-tier-models` in `/home/james/bet_project/.worktrees/BayesianFootball-scotland-cross-tier`  
**Execution Namespace:** `scotland_cross_tier_time_decay`

---

## 1. Executive Context & Motivation

In live MatchDay operations on Saturday 2026-09-19, the production Scottish Lower model (`m12_joint_hybrid_synergy`) placed a massive £7.00 paper bet on **Cove Rangers at decimal odds 7.20** against **Ross County** (Match `16362442`), claiming a **+23.82 pp edge** (model win probability 37.40% vs Betfair implied probability 13.57%). Cove lost 0–3. Across the slate, £16.29 in filled risk was lost on transitioning underdogs and draws.

### The Root Cause: Information Asymmetry & Cold-Start Blindfold
- The production Scottish Lower models (`m05`, `m12`) train strictly on **Tournaments 56 (League 1) and 57 (League 2)** in isolation, over a 2-season rolling window under zero-mean Gaussian shrinkage priors ($\alpha_i, \beta_i \sim \mathcal{N}(0, \sigma^2)$).
- When a club like **Ross County** drops into League 1 after competing in the Premiership (Tournament 54), they have **zero matches** in the League 1/2 training dataset. The model assigns them the default prior: $\alpha_0 = 0$—literally assuming they are an ordinary, average League 1 team.
- To the model: *"Ross County is an average League 1 team playing away at Cove; their win probability is only 37%."*
- To the market: *"Ross County has a Premiership squad and budget; their win probability is 67% (odds 1.45)."*
- To our fractional Kelly staking engine: *"Cove at 7.20 is an enormous bargain (+23.8 pp edge); stake maximum slate capital on Cove!"*
- Empirical EDA (TODO 027) confirmed that relegated clubs post a **+0.26 to +0.40 goal differential** in their first 10 matches (Ross County posted **+2.57 goal diff / match** in 26/27).
- Furthermore, **all 4 SPFL leagues already exist in Postgres (`betdb.sofascore.events`)** with 4,068 normal-time fixtures from 2021 to 2026. The 2-league isolation was an artificial model scoping choice.

---

## 2. Mission & Mandate

Your mission is to formulate, implement, smoke-test, train, and benchmark two competing paradigms to solve the transition cold-start problem, comparing them directly against the baseline Scottish Lower control:

1. **Option A (All-SPFL 4-Tier Model)**: Expand the training scope to all 4 SPFL leagues (`ScottishAll`: Premiership 54, Championship 55, League 1 56, League 2 57). Because clubs promote and relegate every season, continuous ratings naturally link the divisions over 3 seasons.
   - **A1 (`m01_all_spfl_league_offsets`)**: Shared continuous ratings with time decay (180d) and zero-sum league scoring offsets $\delta_{\text{league}}$.
   - **A2 (`m02_all_spfl_hierarchical_tiers`)**: Explicit hierarchical tier-strength steps $\tau_{\text{tier}}$ ($\tau_4=0, \tau_r = \sum_{j=r}^3 d_j$, $d_j \sim \text{HalfNormal}(s_d)$).
2. **Option B (Informative Cold-Start Priors for Scottish Lower)**: Keep training scoped to Scottish Lower (56, 57), but eliminate the flat $\alpha_0 = 0$ blindfold when newly transitioned clubs enter.
   - **B1 (`m03_prior_structural_offset`)**: Structural Relegation/Promotion Prior Offset (relegated clubs initialized at $\alpha_0 \sim \mathcal{N}(+\mu_{\text{relegated}}, \sigma_0^2)$ with $\mu_{\text{relegated}} \approx +0.80$ to $+1.0$).
   - **B2 (`m04_prior_market_derived`)**: Prior-season closing market supremacy mapped to the initial-state prior mean $\mu_0 = m_{\text{sup}} / 2$.
3. **Baseline Control (`m00_control`)**: Standard Scottish Lower time-decay model (Tournaments 56 & 57 with flat $\mathcal{N}(0, \sigma^2)$ priors).

### Strict Architectural Boundaries
- **Time-Decay Dynamics Only**: Use `TimeDecayDynamics(days_half_life = 180.0)`.
- **STRICTLY NO Gaussian Random Walk (GRW) dynamics in this phase**: GRW sampling takes significantly longer and will be evaluated in a separate follow-up phase. Keep the dynamics strictly time decay.
- **Staged Likelihood Progression**:
  - **Stage 1 (Team-level Poisson)**: First isolate and prove the cross-tier mechanics under pure Poisson goals likelihood.
  - **Stage 2 (Team-level Two-Arm Joint)**: Evaluate the winning tier/prior candidates under the two-arm `JointGammaPoissonObservation` (BBC proxy xG + goals).
  - **Stage 3 (Production Extension)**: Add `PlayerLineupPillar` (RAPM lineups) to the champion architecture.

---

## 3. The 4-Tier Evaluation Scorecard

To determine the winning paradigm and production recommendation, all candidate models must be evaluated and scored against the baseline control across 4 distinct scorecard dimensions:

### Dimension 1: Out-of-Sample Proper Scores (710 Scottish Lower Fixtures)
Strict apples-to-apples evaluation across the standard 40-fold walk-forward grid for seasons 24/25 and 25/26 (Tournaments 56 & 57):
- **LogLoss**: Overall proper log score, reported separately for:
  - 1X2 market (home/draw/away)
  - Over/Under 2.5 goals line
- **RPS (Ranked Probability Score)** and **Brier Score**.
- **ECE (Expected Calibration Error)**.
- **GLM Edge & Calibration Slope**: Regress outcome indicators on model probabilities to evaluate whether the 1.72 calibration slope / favourite compression is resolved.

### Dimension 2: Live 26/27 Slate Pricing & Underdog Edge Elimination
Re-price the active 26/27 season, specifically auditing the **2026-09-19 live slate**:
- **Ross County vs Cove Rangers**: Verify that Ross County's win probability increases from ~37% to realistic market levels (60–70%), and the spurious +23.8 pp edge on Cove @ 7.20 is completely eliminated.
- **Hamilton vs Queen of the South**: Audit the pricing of Hamilton as heavy favourite.
- Verify that no toxic Kelly stakes are allocated to longshot underdogs facing relegated heavyweights.

### Dimension 3: Transition Fixture Subgroup Calibration
Isolate all fixtures in the evaluation panel where either club is in its **first 10–20 matches** following promotion or relegation:
- Measure LogLoss, Brier, and mean model-minus-market delta specifically on this transition subgroup.
- Prove that the cold-start learning lag has been eliminated.

### Dimension 4: Portfolio Backtest & Risk Profile
Simulate full zero-allocation portfolio execution under standard production parameters:
- **BookSpec**: 1X2 + Over/Under 2.5, Baker-McHale shrinkage.
- **PolicySpec**: Flat trust 0.25–0.30, SlateDrawdown limit 20.0, 25% max cap, 2% net Betfair commission.
- **Metrics**: Annualized Sharpe, Calmar, Sortino, Total Bankroll Growth (ROI), and Maximum Drawdown.

---

## 4. Execution Protocol & The 7 Verification Gates

Follow the repository's strict verification ladder before launching full production grids:

### Step 1: DataStore & Segment Setup
In `src/Data/fetchers/segments.jl`, define the pyramid segment:
```julia
struct ScottishAll <: DataTournemantSegment end
tournament_ids(::ScottishAll) = [54, 55, 56, 57]
```
Ensure `load_datastore_sql` correctly loads matches, odds, lineups, and BBC commentary across all 4 tournaments.

### Step 2: Loader Implementation (`l01_cross_tier_loader.jl`)
Define:
- Model structs and Turing `@model` engines for `m00_control`, `m01_all_spfl_league_offsets`, `m02_all_spfl_hierarchical_tiers`, `m03_prior_structural_offset`, and `m04_prior_market_derived`.
- Clean dispatch for `Features.required_features`.
- Pre-compiled ReverseDiff gradient tapes with zero runtime allocations.

### Step 3: Automated Smoke Test (`r01_smoke_test.jl`)
Assert all **Seven Verification Gates** on 1 fold locally or on `mcmc-beast` with `@testset`:
1. **Gate 1**: Compiled ReverseDiff gradient tape builds and replays under 0.05 ms with zero heap allocation.
2. **Gate 2**: NUTS sampling completes 2 chains × 100 samples with 0 crashes.
3. **Gate 3**: Six-part convergence audit passes (`ConvergenceThresholds(max_rhat=1.05, min_ess=100.0, max_divergence_rate=eps(), min_bfmi=0.30, max_treedepth_rate=0.05)`).
4. **Gate 4**: Posterior latent extraction (`CountLatents`) succeeds.
5. **Gate 5**: `SmileScoreGrid` generation and pricing (1X2, totals, BTTS).
6. **Gate 6**: `save_fit`/`load_fit` PostgreSQL round-trip parity.
7. **Gate 7**: Portfolio simulation executes and persists.

### Step 4: 40-Fold Walk-Forward Production Grid (`r02_production_grid.jl`)
Run on `mcmc-beast`:
- `QueuedExecution` pinning 16–32 threads across cores.
- `CVConfig(target_seasons = ["24/25", "25/26"], window_seasons = 2)`.
- `NUTSConfig(n_samples = 1_000, n_chains = 4, target_accept = 0.85)`.
- Persist all runs to Postgres under namespace `scotland_cross_tier_time_decay`.

### Step 5: Evaluation & Comparison (`r03_evaluate_and_compare.jl`)
Generate the complete comparison table across all 710 Scottish Lower fixtures: LogLoss, 1X2 LogLoss, O/U 2.5 LogLoss, RPS, Brier, ECE, and GLM calibration slopes.

### Step 6: Transition & 26/27 Live Slate Audit (`r04_audit_transition_and_2627_slate.jl`)
Evaluate on the 26/27 season and specifically price the 2026-09-19 slate. Compare the new posteriors against the historical `m12` posteriors on Ross County vs Cove Rangers.

### Step 7: Portfolio Simulation (`r05_portfolio_backtest.jl`)
Execute fractional Kelly portfolio backtests on all candidate models.

### Step 8: Synthesis & Report (`README.md`)
Deliver a comprehensive, publication-grade research report in this directory summarizing the methodology, numerical leaderboard, transition audits, and final architectural recommendation.

---

## 5. Summary of Suite Artifacts to Deliver

```
experiments/scotland/01_time_decay_cross_tier_and_priors/
├── WORK_PACKAGE_PROMPT.md                # This document
├── l01_cross_tier_loader.jl              # Model definitions, loaders, and compiled tapes
├── r01_smoke_test.jl                     # 7-gate smoke test runner
├── r02_production_grid.jl                # 40-fold walk-forward MCMC runner on mcmc-beast
├── r03_evaluate_and_compare.jl           # Proper scores, 1X2 & O/U LogLoss, GLM edge
├── r04_audit_transition_and_2627_slate.jl# Transition subgroup & 26/27 live slate re-pricing
├── r05_portfolio_backtest.jl             # Fractional Kelly backtesting & drawdown curves
├── results/                              # Manifest, CSV score tables, and diagnostics
└── README.md                             # Comprehensive institutional research report
```
