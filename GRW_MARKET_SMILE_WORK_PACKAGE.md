# Work Package: MultiScaleGRW with Market Smile and Supremacy Anchoring (Task 015)

## 1. Executive Summary & Objective

Extend **MultiScaleGRW** (Gaussian Random Walk state-space dynamics) with **Market Supremacy** and **Market Smile** anchoring pillars on the Scottish Lower leagues (tournaments 56 & 57).

### The Opportunity & Problem Statement:
1. **The Live Supremacy Compression Deficit**:
   On 2026-09-12, the live MatchDay account `live_scottish_m12_500` priced the Scottish Lower card from Run 67 and lost -£45.89 (-9.18%). The model estimated home win probabilities into a flat 40%–43% band, whereas Betfair closing lines priced home favourites (e.g. The Spartans @ 1.70, East Kilbride @ 2.00) at 55%–60%. This 13–16 percentage-point gap led the model to see artificial value on away underdogs, placing 6 losing away bets.
2. **Prior Ireland Market Anchoring Experience**:
   In the Ireland leagues (tournaments 79 & 718), a dual market anchoring mechanism was built:
   - **Market Supremacy**: Likelihood penalty $\text{Normal}(\log \lambda_h - \log \lambda_a, \sigma_{\text{sup}})$ anchoring model supremacy to closing de-vigged market supremacy $m_h - m_a$.
   - **Market Smile**: Likelihood penalty $\text{Normal}(\log(\lambda_h + \lambda_a) + \log \phi(K), \sigma_{\text{smile}})$ anchoring total intensity to closing market Under lines across strikes $K \in \{0, 1, 2, 3, 4\}$.
3. **Why GRW Unlocks What TimeDecay Could Not**:
   When previously tested on Scottish Lower with `TimeDecay`, the smile model failed to improve over baseline because TimeDecay's exponential mean-reversion with a fixed half-life fought directly against the market priors.
   `MultiScaleGRW` formulates team attack and defence as a two-speed random walk ($\sigma_{\text{slow}}, \sigma_{\text{fast}}$). Without a rigid decay anchor, GRW has the temporal flexibility to track market-implied form trajectories without distorting long-term fundamental ratings.

---

## 2. Workspace & Execution Environment

- **Development Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-grw-market-smile`
- **Git Branch**: `feat/grw-market-smile` (branched from `feat/grw-player-lineup-hybrid`)
- **Tmux Session**: `claude_grw_smile`
- **Remote Compute Node**: `mcmc-beast` (AMD Ryzen 9, 16 physical cores, 32 threads, 128GB RAM) for 40-fold walk-forward grid.
- **Local Host (`archpc`)**: 8-16 threads for smoke testing and evaluation.
- **Julia Environment**: Julia 1.12.4 (`julia --project`). Always pin cores:
  ```julia
  using ThreadPinning, LinearAlgebra
  pinthreads(:cores)
  LinearAlgebra.BLAS.set_num_threads(1)
  ```
- **Manifest Invariant**: `Distributions` is strictly pinned to `v0.25.126`. **DO NOT run `Pkg.update()`**.
- **Database Services**:
  - `BF_DB_URL` (operational data & paper ledgers): `archpc:5433` (configured via `.env`). Note: Export `BF_DB_URL` explicitly before launching runners (per Ticket T009).
  - `BF_EXPERIMENTS_DB_URL` (MCMC runs, fold results, latents): `localhost:5432` / `mcmc-beast:5432`.
  - Storage experiment name: `PostgresStorage("scottish_lower_grw_market_smile")` (smoke: `PostgresStorage("smoke_grw_smile")`).

---

## 3. The 3-Model Ablation Ladder

Foundation: **Gen 3 Team-Level Joint Model** (`MultiScaleGRW` + `GlobalInterception` + `GlobalHomeAdvantage` + `ProductionWealthCovariate(role=SupremacyRole)` + `JointGammaPoissonObservation`).

| Model Key | Dynamics | Market Supremacy Pillar | Market Smile Pillar | Likelihood | Purpose |
|---|---|---|---|---|---|
| `m05_joint_grw_baseline` | `MultiScaleGRW()` | None | None | `JointGammaPoisson` | Pure football GRW control |
| `m05_joint_grw_supremacy` | `MultiScaleGRW()` | $\text{Normal}(m_{\text{sup}}, \sigma_{\text{sup}})$ | None | `JointGammaPoisson` | Isolates 1X2 market supremacy pull |
| `m05_joint_grw_smile_supremacy` | `MultiScaleGRW()` | $\text{Normal}(m_{\text{sup}}, \sigma_{\text{sup}})$ | $\text{Normal}(\log\Lambda_K, \sigma_{\text{smile}})$ | `JointGammaPoisson` | **Full dual-pillar market anchor** |

### Weight Tuning Grid:
Evaluate the dual-pillar candidate under 3 market gravity regimes:
1. **Light**: `supremacy_weight = 0.20, smile_weight = 0.20`
2. **Moderate**: `supremacy_weight = 0.40, smile_weight = 0.40` (Ireland default)
3. **Strong**: `supremacy_weight = 0.70, smile_weight = 0.70`

---

## 4. Mathematical Formulation & Architecture

### Likelihood Components:
1. **Two-Arm Football Likelihood**:
   - Proxy xG (Gamma arm): $\text{pxg}_s \sim \text{Gamma}(\nu, \mu_s / \nu)$ on matches with BBC commentary.
   - Goals (Poisson arm): $y_s \sim \text{Poisson}(\kappa \cdot \mu_s)$ on all matches.
2. **Pillar C1: Market Supremacy**:
   - Model supremacy: $s_{\text{model}} = \log \lambda_h - \log \lambda_a$
   - Market supremacy: $s_{\text{mkt}} = \text{logit}(p_{\text{fair, home}}) - \text{logit}(p_{\text{fair, away}})$ (or $\log \lambda_h^{\text{mkt}} - \log \lambda_a^{\text{mkt}}$)
   - Likelihood: $\text{Turing.@addlogprob!} \;\; w_{\text{sup}} \sum \log \mathcal{N}(s_{\text{model}} \mid s_{\text{mkt}}, \sigma_{\text{sup}}) \cdot \text{mask} \cdot \text{weight}$
3. **Pillar C2: Local-Intensity Market Smile**:
   - Total intensity: $\log \lambda_{\text{tot}} = \log(\lambda_h + \lambda_a)$
   - Model per-strike intensity: $\log \Lambda_K = \log \lambda_{\text{tot}} + \log \phi_K$
   - Feature: `MarketSmileFeature(Kmax = 4)` extracting `flat_smile_logΛ` and `flat_smile_mask` from de-vigged Under lines ($K \in \{0, 1, 2, 3, 4\}$).
   - Likelihood: $\text{Turing.@addlogprob!} \;\; w_{\text{sml}} \sum \log \mathcal{N}(\log \Lambda_K \mid \text{smile\_log}\Lambda_K, \sigma_{\text{smile}}) \cdot \text{mask} \cdot \text{weight}$
   - Priors:
     - $\sigma_{\text{sup}} \sim \text{truncated}(\text{Normal}(0.15, 0.10), \text{lower}=0.02)$
     - $\sigma_{\text{smile}} \sim \text{truncated}(\text{Normal}(0.15, 0.10), \text{lower}=0.02)$
     - $\log \phi \sim \text{filldist}(\text{Normal}(0.0, 0.50), n_K)$

### Pricing:
- 1X2 and BTTS price off the standard score grid.
- Over/Under totals price through `SmileScoreGrid` via `Poisson(\lambda_{\text{tot}} \cdot \phi(K))`, matching `src/predictions/score_computation/smile_poisson.jl`.

---

## 5. Implementation Roadmap & Prototype Layout

Create prototypes in `current_development/grw_market_smile/`:

### 1. `l01_loader.jl`
- Feature extraction: include `MarketSmileFeature(Kmax=4)` and Market Supremacy extractor.
- Model definitions for the 3 ladder tiers.
- Splitter: canonical `GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)`.

### 2. `r01_smoke.jl` (Smoke Verification Gate)
- Target: Folds 1–2 of Scottish Lower (`Data.ScottishLower()`).
- Sampler: `QueuedNUTSConfig(n_samples = 500, n_warmup = 500, n_chains = 4, target_accept = 0.80)`.
- Verification gates:
  - Compiled ReverseDiff gradient tape matches ForwardDiff ($\le 10^{-6}$).
  - 0 divergences across all chains.
  - $\hat{R} \le 1.05$, ESS $\ge 400$ across all parameters (including `log_φ`, `σ_sup`, `σ_smile`).
  - Correct extraction of `SmileLatents`.
  - Save and load round-trip via `PostgresStorage("smoke_grw_smile")` reproduces exact parameters.

### 3. `r02_production_grid.jl` (40-Fold Production Grid)
- 40 walk-forward folds across 24/25 and 25/26 (710 matches).
- Sampler: `QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, target_accept = 0.80, max_depth = 10)`.
- Execution: `QueuedExecution()` on `mcmc-beast` (-t 16).
- Storage: `PostgresStorage("scottish_lower_grw_market_smile")`.
- 6-part convergence audit on every fold.

### 4. `r04_evaluate.jl` (Out-of-Sample Proper Scores)
- 710 walk-forward fixtures (627 carrying Betfair closing odds, 2,899 market bets).
- Proper scores: LogLoss, Brier, RPS, ECE across 1X2, Over/Under 2.5, and BTTS.
- Paired 10,000 bootstrap for $\Delta\text{LogLoss}$ vs pure GRW control and Betfair closing line.
- Contrast performance across the weight grid (0.20 vs 0.40 vs 0.70).

### 5. `r05_slate_repricing.jl` (2026-09-12 Slate Re-Pricing)
- Re-price the 2026-09-12 slate (at T-25 with Fold-43 posterior).
- Inspect home favourite win probabilities: Did market supremacy pull home win probabilities from ~41% up toward ~55%–60%?
- Did the model eliminate or scale down the 6 losing away bets on artificial turf?
- Compute counterfactual settled P&L vs the realised -£45.89 live loss.

### 6. `README.md` & Task Tracking
- Document all findings, parameter tables, score comparisons, and slate tearsheet in `current_development/grw_market_smile/README.md`.
- Update `todos/015_prototype_multiscalegrw_with_market_smile_and_supremacy_anchoring.md` with run IDs and tables.
- Verify with `./scripts/todo.sh check`.

---

## 6. Key References

- Ireland Smile Engine: `src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl`
- Smile Feature Extractor: `src/features/extractors/market_extractors.jl`
- MultiScaleGRW Reference: `current_development/multiscale_grw/` and `src/models/pregame/components/dynamics/multiscale_grw.jl`
- Smile Score Grids: `src/predictions/score_computation/smile_poisson.jl` and `src/evaluation/pricing.jl`
- MatchDay 2026-09-12 Re-Pricing Template: `current_development/hierarchical_home_advantage/r05_slate_repricing.jl`
- Coding & Style Standards: `AGENTS.md` and `docs/prototype_runner_style_guide.md`
