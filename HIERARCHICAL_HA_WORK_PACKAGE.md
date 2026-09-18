# Work Package: Hierarchical Team Home Advantage & Scottish Lower Model Extension (Task 008 Phase 1)

## 1. Executive Summary & Objective

Investigate, benchmark, and validate **Hierarchical Team Home Advantage** (`HierarchicalTeamHomeAdvantage`) across the canonical Scottish Lower production architectures to resolve the systemic away-underdog bias observed during the 2026-09-12 live MatchDay execution.

### Background & The 2026-09-12 Post-Mortem:
On Saturday 2026-09-12, the live MatchDay operator priced and executed the Scottish Lower slate (Scottish League One & Two, tournaments 56 & 57) using `m12_joint_hybrid_synergy` (Run 67):
- Account `live_scottish_m12_500` started with £500.00 and finished at £454.11 (-£45.89 net P&L, -9.18% return).
- **Structural Failure Mode**: The production model relies on a single scalar `GlobalHomeAdvantage` ($\gamma = 0.1591 \pm 0.0403$, implying a flat $+17.2\%$ home intensity across all 18 grounds). 
- In reality, **6 of the 9 fixtures were hosted on synthetic 3G/4G artificial turf** (Airdrieonians, East Kilbride, Montrose, Spartans, Edinburgh City, Clyde).
- The Betfair market priced these home teams as decisive favourites (55–65% win probability). The flat model estimated them at only ~48–50%, interpreting the gap as an away-underdog bargain and placing **6 away bets** (e.g. East Fife, Stranraer, Peterhead, Stirling Albion, Forfar Athletic, Bonnyrigg Rose).
- **All 6 away bets lost**, while the turf home sides dominated.

### Two-Phase Strategy (Task 008):
- **Phase 1 (This Work Package)**: Evaluate the existing `HierarchicalTeamHomeAdvantage()` component across three production candidate tiers. Test whether allowing each club's ground effect $\gamma_i = \gamma_{\text{base}} + \sigma_\gamma \cdot z_i$ to vary with partial pooling captures the ground heterogeneity (particularly artificial turf vs grass), evaluates out-of-sample proper scores, and reverses the away-underdog bias on the 2026-09-12 slate.
- **Phase 2 (Subsequent)**: Explicit contextual covariates (`is_synthetic_pitch` and `travel_distance` interaction).

---

## 2. Workspace & Execution Environment

- **Development Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-hierarchical-home-advantage`
- **Git Branch**: `feat/hierarchical-home-advantage` (branched from `feat/grw-player-lineup-hybrid`)
- **Tmux Session**: `claude_hier_ha`
- **Remote Compute Node**: `mcmc-beast` (AMD Ryzen 9, 16 physical cores, 32 threads, 128GB RAM) for 40-fold walk-forward grid.
- **Local Host (`archpc`)**: 8-16 threads for smoke testing and evaluation.
- **Julia Environment**: Julia 1.12.4 (`julia --project`). Always pin cores:
  ```julia
  using ThreadPinning, LinearAlgebra
  pinthreads(:cores)
  LinearAlgebra.BLAS.set_num_threads(1)
  ```
- **Manifest Invariant**: `Distributions` is strictly pinned to `v0.25.126`. **DO NOT run `Pkg.update()`** or upgrade Distributions (v0.25.127 breaks ReverseDiff tape compilation).
- **Database Services**:
  - `BF_DB_URL` (operational data & paper ledgers): `archpc:5433` (configured via `.env`).
  - `BF_EXPERIMENTS_DB_URL` (MCMC runs, fold results, latents): `localhost:5432` / `mcmc-beast:5432`.
  - Storage experiment name: `PostgresStorage("scottish_lower_hierarchical_ha")` (smoke: `PostgresStorage("smoke_hier_ha")`).

---

## 3. The 3-Model Candidate Ladder

Benchmark `HierarchicalTeamHomeAdvantage()` against the existing flat `GlobalHomeAdvantage()` controls across three production model tiers:

| Model Key | Dynamics | Home Advantage | Covariates & Pillars | Likelihood | Purpose |
|---|---|---|---|---|---|
| `m05_joint_production_wealth_hier_ha` | `TimeDecayDynamics(days_half_life = 180.0)` | `HierarchicalTeamHomeAdvantage()` | `GlobalInterception()`, `ProductionWealthCovariate(role = SupremacyRole())` | `JointGammaPoissonObservation()` | Gen 3 team-level benchmark with team HA |
| `m12_joint_hybrid_synergy_hier_ha` | `TimeDecayDynamics(days_half_life = 180.0)` | `HierarchicalTeamHomeAdvantage()` | `GlobalInterception()`, `ProductionWealthCovariate(role = SupremacyRole())`, `PlayerLineupPillar(rating = :shots_rapm, aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10), fit_on = :history)` | `JointGammaPoissonObservation()` | **Gen 4 TimeDecay production candidate** |
| `m12_joint_hybrid_synergy_grw_hier_ha` | `MultiScaleGRW()` | `HierarchicalTeamHomeAdvantage()` | `GlobalInterception()`, `ProductionWealthCovariate(role = SupremacyRole())`, `PlayerLineupPillar(rating = :shots_rapm, aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10), fit_on = :history)` | `JointGammaPoissonObservation()` | **Gen 4 state-space GRW production candidate** |

### Benchmark Controls (Existing Runs in `mcmc_experiments`):
- `m05_joint_td_raw` / `m05_joint_production_wealth`
- `m12_hybrid_td_raw` / `m12_joint_hybrid_synergy` (Experiment 06, Run 67 in `scottish_lower_joint_player_2426`)
- `m12_joint_hybrid_synergy_grw` (Task 013 in `scottish_lower_grw_player_hybrid`)

---

## 4. Mathematical Formulation & Composable Builder

`HierarchicalTeamHomeAdvantage` is already implemented in `src/models/pregame/components/home_advantage.jl`:
```julia
Base.@kwdef struct HierarchicalTeamHomeAdvantage <: AbstractHomeAdvantageConfig
    γ_base::ContinuousUnivariateDistribution = Normal(0.2, 0.2)
    σ_γ::ContinuousUnivariateDistribution = truncated(Normal(0, 0.1), lower=0.0)
end

@model function build_home_advantage(config::HierarchicalTeamHomeAdvantage, n_teams::Int)
    γ_base ~ config.γ_base
    σ_γ ~ config.σ_γ
    γ_team_raw ~ filldist(Normal(0, 1), n_teams) 
    return γ_base .+ (γ_team_raw .* σ_γ)
end
```
Non-centered parameterisation: $z_i \sim \mathcal{N}(0, 1)$, $\gamma_i = \gamma_{\text{base}} + \sigma_\gamma \cdot z_i$.

### Composable Builder Assembly:
```julia
using BayesianFootball

# Example for m12_joint_hybrid_synergy_hier_ha
model = CountModelBuilder(:m12_joint_hybrid_synergy_hier_ha) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = 180.0)) |>
    add(HierarchicalTeamHomeAdvantage()) |>
    add(PlayerLineupPillar(rating = :shots_rapm,
                           aggregation = BenchWeightedPlayerAggregation(w_bench = 0.10),
                           fit_on = :history)) |>
    add(ProductionWealthCovariate(role = SupremacyRole())) |>
    add(JointGammaPoissonObservation()) |>
    build
```

---

## 5. Implementation Roadmap & Prototype Layout

Create prototypes in `current_development/hierarchical_home_advantage/`:

### 1. `l01_loader.jl` (or `l01_models.jl`)
- Define `HierHAConfig` holding all experiment hyperparameters.
- Define builders for the 3 candidate models.
- Note: Use canonical `GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)` for exact comparability with Experiment 06 and Task 013.

### 2. `r01_smoke.jl` (Smoke Verification Gate)
- Target: 2 folds (`target_seasons = ["24/25"]`, Folds 1–2).
- Sampler: `QueuedNUTSConfig(n_samples = 400, n_warmup = 400, n_chains = 4, target_accept = 0.80)` (or 1000s/500w/4c) to satisfy:
  - Zero heap allocations inside ReverseDiff gradient tape (`CompiledTape`).
  - Compiled tape gradient matches ForwardDiff within $\le 10^{-6}$.
  - 0 divergences across all chains.
  - $\hat{R} \le 1.05$ across all sites (including `ha.γ_base`, `ha.σ_γ`, `ha.γ_team_raw[*]`).
  - Valid `CountLatents` extracted.
  - Round-trip save and load through `PostgresStorage("smoke_hier_ha")` reproduces identical parameters and latents.

### 3. `r02_production_grid.jl` (40-Fold Walk-Forward Grid)
- Splitter: 40 walk-forward folds across 24/25 and 25/26 (710 matches).
- Sampler: `QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, target_accept = 0.80, max_depth = 10)`.
- Storage: `PostgresStorage("scottish_lower_hierarchical_ha")`.
- Save all fits and verify the 6-part convergence audit for every fold.

### 4. `r04_evaluate.jl` (Out-of-Sample Proper Scores)
- Evaluate proper scores on 710 out-of-sample matches:
  - 1X2 LogLoss, RPS, Brier, ECE.
  - Over/Under 2.5 LogLoss, Brier, ECE.
  - BTTS LogLoss, Brier, ECE.
- Compare against Betfair Closing Line and the 3 control models.
- Paired 10,000 bootstrap for $\Delta\text{LogLoss}$ and 95% confidence intervals.
- Extract posterior distributions for ground-level home advantage $\gamma_i$ for all 18 clubs:
  - Compare estimated $\gamma_i$ for turf clubs (Airdrieonians, East Kilbride, Montrose, Spartans, Edinburgh City, Clyde, Alloa, Cove Rangers) vs grass clubs (Peterhead, Stranraer, Elgin, Dumbarton, etc.).
  - Check whether $\sigma_\gamma$ posterior excludes zero.

### 5. `r05_slate_repricing.jl` (2026-09-12 Counterfactual Slate Re-Pricing)
- Load the 2026-09-12 slate fixtures and T-25 order book (`betfair_live.order_book_1m`).
- Re-price the card using the newly fitted `m12_joint_hybrid_synergy_hier_ha` (or extended Fold 43).
- Compare model probabilities $P(\text{Home}), P(\text{Draw}), P(\text{Away})$ between flat `GlobalHomeAdvantage` and `HierarchicalTeamHomeAdvantage`.
- Generate the counterfactual Option B stake sheet:
  - Did the model eliminate or scale down the 6 losing away bets?
  - Did it take home bets or under bets on the artificial turf grounds?
  - Compute counterfactual settled P&L vs the realised -£45.89 loss.

### 6. `README.md` & Task Tracking
- Document all findings, parameter tables, proper score deltas, and counterfactual tearsheet in `current_development/hierarchical_home_advantage/README.md`.
- Update `todos/008_hierarchical_scottish_pitch_type_and_match_timing_home_advantage.md` with work log, run IDs, and verification results.
- Verify with `./scripts/todo.sh check`.

---

## 6. Key References

- Home Advantage Definition: `src/models/pregame/components/home_advantage.jl`
- Builder Home Advantage Integration: `src/models/pregame/builder/builder.jl` and `engine.jl`
- MatchDay 2026-09-12 Live Slate Runner: `current_development/match_day_inference/r11_live_slate_20260912.jl`
- MatchDay 2026-09-12 Audit: `current_development/match_day_inference/r14_audit.jl`
- Gen 4 Hybrid Benchmark Reference: `current_development/grw_player_hybrid/`
- Style & Coding Standards: `AGENTS.md` and `docs/prototype_runner_style_guide.md`
