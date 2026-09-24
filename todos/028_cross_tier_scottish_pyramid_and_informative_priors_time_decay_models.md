# 028 — Cross-Tier Scottish Pyramid and Informative Priors Time-Decay Models

| Field | Value |
|---|---|
| ID | 028 |
| Title | Cross-Tier Scottish Pyramid and Informative Priors Time-Decay Models |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-24 |
| Updated | 2026-09-24 |
| Related Files / Commits / PRs | `experiments/scotland/01_time_decay_cross_tier_and_priors/` |

## Context & Problem Statement

Current production Scottish Lower models (`m05`, `m12`) train strictly on Tournaments 56 (League 1) and 57 (League 2) in isolation. Under flat shrinkage priors ($\alpha_i \sim \mathcal{N}(0, \sigma^2)$), newly transitioned clubs dropping from higher tiers (e.g., Ross County, Hamilton, Dunfermline) enter with zero historical data and an assumed average talent level ($\alpha = 0$).

In live operations (e.g. 2026-09-19), this manufactured a false +23.8 pp model edge on extreme underdogs (Cove Rangers @ 7.20 vs Ross County @ 1.45), drawing 46% of slate capital and losing. Empirical EDA (TODO 027) confirmed that relegated clubs average +0.26 to +0.40 goal differentials in their first 10 matches (Ross County +2.57 in 26/27), which the market prices at 65–70% win probability while isolated lower-league models price at ~37%.

This task designs, trains, and benchmarks candidate solutions across the Scottish football pyramid using time-decay dynamics (excluding Gaussian Random Walk dynamics for now):
- **Option A (All-SPFL 4-Tier Model)**: Pools all 4 SPFL leagues (Premiership 54, Championship 55, League 1 56, League 2 57) so relegated clubs retain their attack/defence ratings across transitions. Tests both shared continuous ratings with zero-sum league scoring offsets $\delta_{\text{league}}$ (Option A1) and ordered hierarchical tier steps $\tau_{\text{tier}}$ (Option A2).
- **Option B (Informative Cold-Start Priors for Scottish Lower)**: Retains fast lower-league training (56, 57) but initializes new entrants with informative priors based on structural relegation offsets $\mu_{\text{relegated}}$ (Option B1) or prior-season closing market supremacy (Option B2).
- **Baseline Control**: Standard Scottish Lower time-decay model (`ScottishLower` [56, 57] baseline, 180-day half-life).

## Acceptance Criteria

- [ ] Dedicated experiment directory established at `experiments/scotland/01_time_decay_cross_tier_and_priors/`.
- [ ] Staged likelihood progression implemented in `l01_cross_tier_loader.jl`:
  1. Pure team-level time decay with Poisson goals likelihood.
  2. Pure team-level time decay with two-arm Joint Gamma-Poisson likelihood (`JointGammaPoissonObservation`).
- [ ] Model variants implemented and passing ReverseDiff zero-allocation tape compilation:
  - `m00_control`: Lower-league baseline (Tournaments 56, 57; flat priors).
  - `m01_all_spfl_league_offsets` (Option A1): All 4 tiers (54, 55, 56, 57) with continuous ratings and zero-sum league offsets $\delta_{\text{league}}$.
  - `m02_all_spfl_hierarchical_tiers` (Option A2): All 4 tiers with ordered tier steps $\tau_{\text{tier}}$.
  - `m03_prior_structural_offset` (Option B1): Lower tiers (56, 57) with structural cold-start prior $\mu_{\text{relegated}}$.
  - `m04_prior_market_derived` (Option B2): Lower tiers (56, 57) with closing market supremacy mapped to initial state prior.
- [ ] Smoke test runner `r01_smoke_test.jl` passing all 7 verification gates on 1 fold.
- [ ] 40-fold walk-forward production grid `r02_production_grid.jl` executed on `mcmc-beast` via QueuedExecution.
- [ ] Out-of-sample evaluation in `r03_evaluate_and_compare.jl` covering the 710 Scottish Lower fixtures (seasons 24/25 & 25/26):
  - LogLoss (overall, 1X2 market, and Over/Under 2.5 line).
  - RPS, Brier, and ECE.
  - GLM edge and calibration slope (evaluating the 1.72 compression).
- [ ] Live 26/27 slate and transition audit in `r04_audit_transition_and_2627_slate.jl`:
  - Re-evaluating the 2026-09-19 slate (Cove vs Ross County, Hamilton vs Queen of the South).
  - Verifying elimination of the false underdog edge on Cove @ 7.20.
  - Subgroup calibration on all transition fixtures (first 10–20 matches of promoted/relegated clubs).
- [ ] Portfolio backtesting profile in `r05_portfolio_backtest.jl` (Sharpe, Calmar, ROI, and max drawdown under fractional Kelly and Baker-McHale shrinkage).
- [ ] Phase 2 extension: Adding `PlayerLineupPillar` (RAPM lineups) to the winning architecture.
- [ ] Comprehensive institutional research report delivered in `README.md`.

## Ideas & Candidate Solutions

- **Option A1 (Zero-sum league offsets)**: Additive $\delta_{\text{league}}$ per division captures scoring rate heterogeneity (e.g. higher goal rates in League 1 than Premiership) while team attack/defence ratings are shared across the whole SPFL. Continuity across promotion/relegation links the leagues over 3 seasons.
- **Option A2 (Hierarchical tier steps)**: Reference tier $\tau_4 = 0$, $\tau_r = \sum_{j=r}^3 d_j$ with $d_j \sim \text{HalfNormal}(s_d)$. Enforces that higher tiers have systematically higher latent quality.
- **Option B1 (Structural offset)**: When a club appears in the training window with fewer than 5 matches and came from Championship/Premiership, assign prior mean $\alpha_0 \sim \mathcal{N}(+\mu_{\text{relegated}}, \sigma_0^2)$ with $\mu_{\text{relegated}} \approx +0.80$ to $+1.0$.
- **Option B2 (Market-derived initial state)**: Extract de-vigged closing line supremacy from the club's last 5 matches in the higher division and set $\alpha_0 = m_{\text{sup}} / 2$.

## Work Log & Progress

- [2026-09-24 @antigravity] Conducted /grill-me session with human user. Agreed on scope: All-SPFL (Option A: A1 and A2) vs Informative Priors (Option B: B1 and B2) vs Baseline Control. Staged likelihoods: Poisson goals first, then Joint Gamma-Poisson. Excluded GRW for now. Created dedicated worktree `/home/james/bet_project/.worktrees/BayesianFootball-scotland-cross-tier` on branch `feat/scotland-cross-tier-models`. Allocated TODO 028.

## Verification & Findings

Not run yet. Record R-hat, ESS, LogLoss, 1X2 and O/U 2.5 proper scores, calibration slope, 26/27 Ross County slate pricing, transition subgroup metrics, and portfolio Sharpe.
