# Work Package: Prototyping JointGammaNegBinObservation for Totals and BTTS Calibration (Task 014)

## Objective & Theoretical Motivation

Prototype and evaluate `JointGammaNegBinObservation` across the canonical 40-fold Scottish Lower walk-forward grid (tournaments 56/57, 710 matches), testing whether replacing the conditional Poisson goal likelihood with a `RobustNegativeBinomial` distribution improves predictive proper scores (LogLoss, Brier, ECE) on **Totals (Over/Under 1.5, 2.5, 3.5)** and **Both Teams To Score (BTTS)**.

### The Statistical Context
1. **Empirical Overdispersion**: Raw Scottish Lower match scorelines show marginal overdispersion ($\text{Var}(Y) > \text{Mean}(Y)$).
2. **Poisson in Hierarchical Models**: Latent team rate variation in `MultiScaleGRW` already creates marginal overdispersion. Experiment 02 (`02_negbin_2426_grid`) estimated residual overdispersion at $\hat{r} \approx 26.0\text{--}26.5$ (~5% extra variance over Poisson), which had negligible effect on 1X2 LogLoss ($\Delta = +0.0001$).
3. **The Totals / BTTS Hypothesis**: While 1X2 probabilities are sums of scoreline diagonals that average out minor tail shifts, **Totals and BTTS are highly sensitive to score distribution shape**. A Negative Binomial distribution shifts probability mass:
   - Higher mass at 0 goals (zero-inflation).
   - Lower mass at 1 and 2 goals.
   - Fatter right tail at 4+ goals (blowouts).
4. **The Goal**: Build a two-arm observation likelihood combining Gamma proxy-xG with `RobustNegativeBinomial` goals:
   $$\text{Arm 1 (Proxy xG)}: \quad \text{pxg} \sim \text{Gamma}(\nu, \mu/\nu)$$
   $$\text{Arm 2 (Goals)}: \quad y \sim \text{RobustNegativeBinomial}(r, \kappa \cdot \mu)$$
   with $r = \exp(\text{clamp}(\log r, -10, 10))$ under `GlobalDispersion(log_r ~ Normal(3.1, 0.4))`.

---

## Workspace & Host Environment

- **Remote Compute Host**: `mcmc-beast` (16 cores, 32 threads, 128GB RAM).
- **Working Directory**: `/root/BF_grw_joint_negbin`
- **Local Directory**: `/home/james/bet_project/.worktrees/BayesianFootball-grw-joint-negbin`
- **Git Branch**: `feat/grw-joint-negbin-observation` (branched off `feat/grw-player-lineup-hybrid`).
- **Julia Environment**: Julia 1.12.4 in `/root/.juliaup/bin/julia`.
- **Manifest**: Locked to `Distributions` v0.25.126. Do NOT run `Pkg.update("Distributions")`.
- **Databases**:
  - `mcmc_experiments`: PostgreSQL on `localhost:5432` (`mcmc_experiments_postgres` container).
  - `betdb`: PostgreSQL on `archpc:5433` (configured in `.env`).

---

## 4-Model Ladder & Direct Benchmark Controls

Evaluate the Negative Binomial ladder against the exact Task 013 Poisson counterparts:

| NegBin Model Key | Dynamics | Covariates & Pillars | Observation Likelihood | Direct Poisson Control (Task 013) |
|---|---|---|---|---|
| `m00_baseline_grw_negbin` | `MultiScaleGRW()` | Interception + HomeAdv | `NegativeBinomialObservation(GlobalDispersion())` | `m00_baseline_grw` (`158d2a80…`) |
| `m05_wealth_grw_negbin` | `MultiScaleGRW()` | Interception + HomeAdv + ProductionWealth | `JointGammaNegBinObservation(GlobalDispersion())` | `m05_wealth_grw` (`b0961bc4…`) |
| `m10_lineup_grw_negbin` | `MultiScaleGRW()` | Interception + HomeAdv + ShotsRAPM Lineup | `NegativeBinomialObservation(GlobalDispersion())` | `m10_lineup_grw` (`b13c8fb9…`) |
| `m12_joint_hybrid_synergy_negbin` | `MultiScaleGRW()` | Interception + HomeAdv + Wealth + Lineup | `JointGammaNegBinObservation(GlobalDispersion())` | `m12_joint_hybrid_synergy_grw` (`3a9a4c7e…`) |

---

## Architectural Implementation Plan

### 1. `JointGammaNegBinObservation` Struct & Builder Interface
In `current_development/grw_joint_negbin/l01_loader.jl`:
Define `JointGammaNegBinObservation`:
```julia
Base.@kwdef struct JointGammaNegBinObservation{D<:AbstractDispersionConfig, F<:AbstractFeatureConfig, S<:ContinuousUnivariateDistribution, K<:ContinuousUnivariateDistribution} <: AbstractObservationConfig
    dispersion::D = GlobalDispersion(log_r = Normal(3.1, 0.4))
    feature::F = MatchProxyXGFeature(k = 25.0, fallback = :none)
    shape_prior::S = truncated(Normal(4.0, 1.5), 0.5, Inf)
    log_kappa_prior::K = Normal(0.0, 0.2)
end
```
Wire the observation methods:
- `observation_family(::JointGammaNegBinObservation) = :negbin`
- Observation submodel `_observe(...)`:
  - Samples `log_r ~ dispersion.log_r`, clamps to `[-10.0, 10.0]`, $r = \exp(\log r)$.
  - Arm 1: `pxg ~ Gamma(ν, μ / ν)`.
  - Arm 2: `goals ~ RobustNegativeBinomial(r, κ .* μ)`.
- Latent extraction: extracts both $\lambda_h, \lambda_a$ and dispersion $r$ for score grid generation.

### 2. Score Grid Generation
Ensure that score grid evaluation uses the `RobustNegativeBinomial` PMF on the 12×12 bivariate score grid:
$$P(y_h = i, y_a = j) = \text{RobustNegativeBinomial}(i; r, \lambda_h) \times \text{RobustNegativeBinomial}(j; r, \lambda_a)$$
Compute probabilities for:
- 1X2 (Home, Draw, Away)
- Over/Under 1.5, Over/Under 2.5, Over/Under 3.5, Over/Under 4.5
- BTTS (Both Teams To Score: Yes/No)

---

## Step-by-Step Execution Workflow

### Step 1: Smoke Test Gate (`r01_smoke.jl`)
- 2 folds of Scottish Lower (`GroupedCVConfig`).
- Test all 4 models at sampler `4 × (500 warmup + 1000 retained)`.
- Verification gates:
  - ReverseDiff tape compilation: zero heap allocations inside the gradient tape, matches ForwardDiff to 1e-15.
  - 0 divergences across all chains.
  - $\hat{R} \le 1.05$, bulk/tail ESS $\ge 400$.
  - Exact `save_fit` $\to$ `load_fit` round-trip in `PostgresStorage("smoke_grw_joint_negbin")`.

### Step 2: 40-Fold Walk-Forward Production Grid (`r02_production_grid.jl`)
- Canonical 40-fold walk-forward grid (seasons 2024/25 + 2025/26, 710 matches).
- Split: `gph_splitter(["24/25", "25/26"])` (`GroupedCVConfig`).
- Sampler: `QueuedNUTSConfig(1000 retained, 500 warmup, 4 chains, δ = 0.80, max_depth = 10)`.
- Run on `mcmc-beast` with 16 pinned threads.
- Storage: `PostgresStorage("scottish_lower_grw_joint_negbin")`.
- Save thinned draws (every 2nd draw) and audit convergence across all 4,000 draws.

### Step 3: Proper Scores & Paired Bootstrap Evaluation (`r04_evaluate.jl`)
Evaluate on the 710 held-out matches against de-vigged Betfair closing odds:
1. **Multi-Market Scopes**:
   - 1X2: LogLoss, Brier, RPS, ECE.
   - Totals: Over/Under 1.5, 2.5, 3.5 LogLoss, Brier, ECE.
   - BTTS: LogLoss, Brier, ECE.
2. **Paired Bootstrap ($\Delta\text{LogLoss}$)**:
   - 10,000 fixture-clustered bootstrap resamples comparing each NegBin model directly against its Task 013 Poisson counterpart:
     - `m00_baseline_grw_negbin` vs `m00_baseline_grw`
     - `m05_wealth_grw_negbin` vs `m05_wealth_grw`
     - `m10_lineup_grw_negbin` vs `m10_lineup_grw`
     - `m12_joint_hybrid_synergy_negbin` vs `m12_joint_hybrid_synergy_grw`
3. **Core Research Question**: Does `JointGammaNegBinObservation` produce a statistically significant reduction in LogLoss or ECE on Over/Under 2.5/3.5 or BTTS?

### Step 4: Option B Portfolio Simulation & Attribution (`r05_portfolio.jl`)
- Simulate standard `MatchDay.option_b_system()` on de-vigged Betfair closing odds (1X2 + O/U 2.5).
- Compute Task 012 attribution metrics:
  - Capture Ratio
  - Sizing on shared bets
  - Disjoint/exclusive bet returns

### Step 5: (Optional) Hierarchical Dispersion Extension
- If `GlobalDispersion` demonstrates significant alpha or improved totals calibration, explore whether `HomeAwayDispersion` ($r_h, r_a$) or tournament-level hierarchical dispersion yields further gains.

### Step 6: Documentation & Registry Update
- Update `todos/014_prototype_jointgammanegbinobservation_for_totals_and_btts_calibration.md` with:
  - Run UUIDs, convergence tables, proper scores across 1X2/Totals/BTTS, and portfolio metrics.
- Ensure `./scripts/todo.sh check` passes.
- Commit all findings and open PR.

---

## References & Baseline Artifacts
- Task 013 Reference & Control UUIDs: `current_development/grw_player_hybrid/README.md` and `todos/013_fuse_multiscalegrw_with_player_lineup_dynamics.md`.
- Experiment 02 NegBin Findings: `experiments/scottish_lower/02_negbin_2426_grid/README.md`.
- Coding Standards: `AGENTS.md` and ReverseDiff AD safety guidelines.
