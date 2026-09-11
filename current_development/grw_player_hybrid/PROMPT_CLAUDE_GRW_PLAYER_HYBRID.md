# Work Package: Fusing MultiScaleGRW with Player Lineup Dynamics (Task 013)

## Objective & Executive Summary

Extend Gaussian Random Walk (GRW) latent state-space dynamics from team-level to team + player level features, upgrading the Scottish Lower leagues (tournaments 56 & 57) Generation 4 production architecture (`m12_joint_hybrid_synergy`) to two-speed GRW state space.

### The Opportunity:
1. **Gen 4 Production Architecture (`m12_joint_hybrid_synergy`)**:
   - Composes team-level dynamics with an announced XI `PlayerLineupPillar(rating=:shots_rapm, w_bench=0.10, fit_on=:history)`, `ProductionWealthCovariate(role=SupremacyRole())`, and `JointGammaPoissonObservation()`.
   - Result: ECE **0.0100** vs Betfair close 0.0139, out-of-sample bankroll growth **+136.6%**, annual Sharpe **1.416**.
   - Limitation: Relied on exponential time-decay (`TimeDecayDynamics(180.0)`), which assumes a constant half-life reversion to league mean and struggles during rapid regime shifts or early-season promotions/relegations.
2. **MultiScaleGRW Latent State Space (Task 007)**:
   - Formulates latent team attack and defence as an unconstrained two-speed Gaussian Random Walk:
     - Slow drift: $\sigma_{\text{slow}} \sim \text{TruncatedNormal}(0, 0.08)$ (persistent season-long quality drift)
     - Fast shock: $\sigma_{\text{fast}} \sim \text{TruncatedNormal}(0, 0.25)$ with mean-reversion $\phi \sim \text{Beta}(8, 2)$ (transient form/injury fluctuations)
   - On the 2026/27 season opening slates at T−25 1-minute order books (`betfair_live.order_book_1m`), `m05_joint_grw` delivered **+25.59% ROI** (+18.01% bankroll growth, £500 → £590) with **−0.13% max drawdown**, while `m12_hybrid_td` achieved +13.69% ROI (+10.39% growth, £500 → £552).
3. **The Goal**:
   - Fuse `MultiScaleGRW` with `PlayerLineupPillar` to create `m12_joint_hybrid_synergy_grw`.
   - Validate on the canonical 40-fold walk-forward grid (24/25 + 25/26, 710 matches), extend to 2026/27 opening slates (Folds 41–43), and evaluate under Task 012 Portfolio Attribution standards.

---

## Workspace & Host Environment

- **Remote Compute Host**: `mcmc-beast` (16 physical cores, 32 threads, 128GB RAM).
- **Working Directory**: `/root/BF_grw_player_hybrid`
- **Git Branch**: `feat/grw-player-lineup-hybrid`
- **Julia Environment**: Julia 1.12.4 in `/root/.juliaup/bin/julia`.
- **Manifest**: Pinned to `Distributions` v0.25.126 (`git-tree-sha1 = "96f76dcd6cc75cf8eb49109123868499d413f526"`). Do NOT run `Pkg.update("Distributions")` as v0.25.127 breaks ReverseDiff tape compilation.
- **Database Services**:
  - `mcmc_experiments`: PostgreSQL on `localhost:5432` (`postgres_experiments_data`).
  - `betdb`: PostgreSQL on `archpc:5433` (accessible over LAN/Tailscale, configured in `.env`).

---

## 4-Model Ablation Ladder

To isolate the individual contributions of GRW dynamics, wealth covariates, player lineup RAPM, and proxy xG, run a clean 4-model ladder:

| Model Key | Dynamics | Covariates & Pillars | Observation Likelihood | Purpose |
|---|---|---|---|---|
| `m00_baseline_grw` | `MultiScaleGRW()` | `GlobalInterception()`, `GlobalHomeAdvantage()` | `PoissonObservation()` | Pure GRW team baseline |
| `m05_wealth_grw` | `MultiScaleGRW()` | `GlobalInterception()`, `GlobalHomeAdvantage()`, `ProductionWealthCovariate(role=SupremacyRole())` | `JointGammaPoissonObservation()` | Team-level GRW benchmark (reproduces Task 007) |
| `m10_lineup_grw` | `MultiScaleGRW()` | `GlobalInterception()`, `GlobalHomeAdvantage()`, `PlayerLineupPillar(rating=:shots_rapm, aggregation=BenchWeightedPlayerAggregation(w_bench=0.10), fit_on=:history)` | `PoissonObservation()` | Pure player-lineup + GRW interaction |
| `m12_joint_hybrid_synergy_grw` | `MultiScaleGRW()` | `GlobalInterception()`, `GlobalHomeAdvantage()`, `ProductionWealthCovariate(role=SupremacyRole())`, `PlayerLineupPillar(rating=:shots_rapm, aggregation=BenchWeightedPlayerAggregation(w_bench=0.10), fit_on=:history)` | `JointGammaPoissonObservation()` | **Full hybrid production candidate** |

**Benchmark Controls (Existing Fits in `mcmc_experiments`)**:
- `m05_joint_td_raw`: Team TimeDecay + ProductionWealth + JointGammaPoisson.
- `m12_hybrid_td_raw`: Gen 4 Production Hybrid (TimeDecay 180d + PlayerLineup + Wealth + JointGammaPoisson).

---

## Composable Model Builder Formulation

In `src/models/pregame/builder/grw_dynamics.jl`, `MultiScaleGRW` is registered with the builder.
Build the full hybrid model as:

```julia
using BayesianFootball

model = CountModelBuilder(:m12_joint_hybrid_synergy_grw) |>
    add(GlobalInterception()) |>
    add(MultiScaleGRW()) |>
    add(GlobalHomeAdvantage()) |>
    add(PlayerLineupPillar(rating = :shots_rapm,
                           aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
                           fit_on = :history)) |>
    add(ProductionWealthCovariate(role = SupremacyRole())) |>
    add(JointGammaPoissonObservation()) |>
    build
```

### Critical AD & Tape-Safety Rules
1. Zero heap allocations inside the Turing model tape.
2. `PlayerLineupPillar` generates out-of-sample RAPM differentials during feature extraction before model evaluation.
3. In `MultiScaleGRW`, dynamic states `alpha_slow`, `alpha_fast`, `beta_slow`, `beta_fast` are tracked per team across match time indices $1 \dots T$. Ensure `_cb_oos_dynamics` projects the final step latent into out-of-sample fixtures correctly.

---

## Step-by-Step Execution Plan

### Step 1: Smoke Test Gate (`r01_smoke.jl`)
Create `current_development/grw_player_hybrid/r01_smoke.jl`:
- Target: 2 folds (`target_seasons = ["24/25"]`, Folds 1–2).
- Sampler: `NUTSConfig(n_samples = 100, n_warmup = 50, n_chains = 2)`.
- Test all 4 models: `m00_baseline_grw`, `m05_wealth_grw`, `m10_lineup_grw`, `m12_joint_hybrid_synergy_grw`.
- Verification gates:
  - Gradient tape compiles with ReverseDiff (`CompiledTape`).
  - 0 divergences across all chains.
  - $\hat{R} < 1.05$ across all sites.
  - Posterior latents (`CountLatents`) extract with valid non-NaN means and variances.
  - Save and reload round-trip via `PostgresStorage("smoke_grw_player")`.

### Step 2: 40-Fold Walk-Forward Production Grid (`r02_production_grid.jl`)
Create `current_development/grw_player_hybrid/r02_production_grid.jl`:
- Target: Canonical 40-fold walk-forward grid (seasons 2024/25 + 2025/26, 710 matches).
- Splitter: `Data.CVConfig(target_seasons = ["24/25", "25/26"], window_seasons = 3)`.
- Sampler: `QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, target_accept = 0.80, max_depth = 10)`.
- Execution: `QueuedExecution()` flattening 40 folds × 4 chains across 16 beast cores.
- Storage: `PostgresStorage("scottish_lower_grw_player_hybrid")`.
- Save all fits and verify 6-part convergence audit for every fold.

### Step 3: Extend to 2026/27 Opening Slates (Folds 41–43)
Using `extend_fit`:
- Extend all 4 fitted models to cover the 2026/27 opening slates (matches through September 2026).
- Verify that `extend_fit` appends the new fold results and updates `CountLatents`.

### Step 4: Out-of-Sample Evaluation (`r03_evaluate.jl`)
Create `current_development/grw_player_hybrid/r03_evaluate.jl`:
- Compute proper scores on 24/25 + 25/26 out-of-sample matches (710 matches, 2,899 market observations):
  - 1X2 LogLoss, RPS, Brier score, ECE.
  - Over/Under 2.5 LogLoss, Brier score, ECE.
  - BTTS LogLoss, Brier score, ECE.
- Compare against:
  - Betfair Closing Line benchmark.
  - `m05_joint_td_raw` (Team TimeDecay).
  - `m12_hybrid_td_raw` (Gen 4 Production Hybrid).
- Run paired 10,000-sample bootstrap for LogLoss significance ($\Delta \text{LogLoss}$ and 95% CI).

### Step 5: Portfolio Attribution (Task 012) & Order Book Backtest (`r04_portfolio_attribution.jl`, `r05_t25_backtest.jl`)
1. **Closing Line Simulation (`r04_portfolio_attribution.jl`)**:
   - Apply Option B production policy:
     - `spec = BookSpec(markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]), shrink = BakerMcHale())`
     - `policy = PolicySpec(trust = TieredTrust(base = 0.25), risk = SlateDrawdown(8.0), cap = FixedCap(0.20))`
   - Standardise Task 012 Attribution Metrics:
     - **Capture Ratio**: Realised P&L on common bets between `m12_hybrid_grw` and `m12_hybrid_td`.
     - **Sizing on Shared Bets**: Average stake size when both models agree vs when one model has higher conviction.
     - **Disjoint Bets Analysis**: P&L and strike rate of bets taken *exclusively* by `m12_hybrid_grw` vs *exclusively* by `m12_hybrid_td`.
2. **2026/27 T−25 1-Minute Order Book Backtest (`r05_t25_backtest.jl`)**:
   - Evaluate on `betfair_live.order_book_1m` at T−25 (13:35 UTC) across the 4 opening slates of 2026/27 (Aug 1, Aug 8, Aug 15, Sep 5) starting from £500 bankroll.
   - Test both `TouchOnly` and `LadderSweep` fill execution.
   - Benchmark `m12_joint_hybrid_synergy_grw` directly against `m05_joint_grw_raw` (£590–£600) and `m12_hybrid_td_raw` (£552).

### Step 6: Documentation & Registry Update
- Update `todos/013_fuse_multiscalegrw_with_player_lineup_dynamics.md` with:
  - Work log entries.
  - Tabulated evaluation scores (LogLoss, ECE, RPS).
  - Portfolio performance metrics (Return, ROI, Sharpe, Drawdown, Capture Ratio).
  - Artifact paths and database run IDs.
- Run `./scripts/todo.sh check` to verify metadata consistency.
- Submit git commit and create PR for `feat/grw-player-lineup-hybrid`.

---

## Key Contacts & References
- Task 007 Reference: `current_development/multiscale_grw/` and `todos/007_prototype_gaussian_random_walk_state_space_dynamics_with_reversediff.md`.
- Gen 4 Hybrid Reference: `experiments/scottish_lower/06_joint_player_lineup_fusion/README.md`.
- 2026/27 T−25 Backtest Reference: `current_development/match_day_inference/results/REPORT_T25_GRW_2627.md` and `r13_t25_grw_backtest.jl`.
- Coding Standards: `AGENTS.md` and `docs/prototype_runner_style_guide.md`.
