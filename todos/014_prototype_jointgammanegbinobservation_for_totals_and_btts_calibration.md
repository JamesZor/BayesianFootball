# 014 — Prototype JointGammaNegBinObservation for Totals and BTTS Calibration

| Field | Value |
|---|---|
| ID | 014 |
| Title | Prototype JointGammaNegBinObservation for Totals and BTTS Calibration |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-11 |
| Updated | 2026-09-11 |
| Related Files / Commits / PRs | `src/models/pregame/components/observations/`, `src/MyDistributions/negative_binomial.jl`, `src/models/pregame/builder/components.jl`, `feat/grw-joint-negbin-observation` |

## Context & Problem Statement

In exploratory data analysis of Scottish Lower leagues (tournaments 56/57), goal counts display marginal overdispersion ($\text{Var}(Y) > \text{Mean}(Y)$). While hierarchical models with dynamic team state spaces (`MultiScaleGRW`) absorb substantial variance through latent rate variation, the conditional Poisson assumption on goals ($y \sim \text{Poisson}(\kappa \cdot \mu)$) in `JointGammaPoissonObservation` imposes equidispersion conditional on $\mu$.

In Experiment 02 (`02_negbin_2426_grid`), single-arm Negative Binomial models yielded $\hat{r} \approx 26.0\text{--}26.5$, showing mild residual overdispersion. While 1X2 LogLoss was essentially neutral ($\Delta = +0.0001$), Negative Binomial has distinct theoretical implications for scoreline distributions:
1. Higher probability mass at 0 goals (zero-inflation effect).
2. Lower probability mass at 1 and 2 goals.
3. Fatter right tail on 4+ goals (blowouts).

The hypothesis is that replacing the Poisson goal likelihood in the two-arm joint architecture with a `RobustNegativeBinomial` likelihood (`JointGammaNegBinObservation`) will improve tail pricing, specifically sharpening predictive proper scores (LogLoss, Brier, ECE) on Totals (Over/Under 1.5, 2.5, 3.5) and Both Teams To Score (BTTS).

Constraints:
- ReverseDiff tape compilation safety: zero allocations inside the gradient tape.
- Must execute on the canonical 40-fold walk-forward grid (24/25 + 25/26, 710 matches).
- Evaluate 4-model ladder (`m00`, `m05`, `m10`, `m12`) directly against their Task 013 Poisson counterparts.
- If `GlobalDispersion` demonstrates significant proper score improvement on totals, prototype hierarchical tournament/home-away dispersion.

## Acceptance Criteria

- [ ] **Component Implementation**: Implement and wire `JointGammaNegBinObservation` in `current_development/grw_joint_negbin/` (and builder components) with Arm 1 Gamma proxy xG and Arm 2 `RobustNegativeBinomial(r, \kappa \cdot \mu)` using `GlobalDispersion(log_r ~ Normal(3.1, 0.4))`.
- [ ] **Score Grid Integration**: Ensure score-grid kernels evaluate bivariate distributions using `RobustNegativeBinomial` PMF on 12×12 grids.
- [ ] **Smoke Gate (Folds 1–2)**: 2-fold smoke test on `m00_baseline_grw_negbin`, `m05_wealth_grw_negbin`, `m10_lineup_grw_negbin`, `m12_joint_hybrid_synergy_negbin`. Verify gradient tape compilation with ReverseDiff, 0 divergences, R̂ < 1.05, bulk/tail ESS > 400.
- [ ] **40-Fold Walk-Forward Grid**: Train all 4 models across Folds 1–40 (seasons 24/25 + 25/26, 710 matches) on `mcmc-beast` with `QueuedNUTSConfig`.
- [ ] **Expanded Totals & BTTS Evaluation**: Compute proper scores across 1X2, Over/Under 1.5, Over/Under 2.5, Over/Under 3.5, and BTTS. Run 10,000-sample paired bootstrap significance vs Task 013 Poisson counterparts.
- [ ] **Portfolio Backtesting**: Run standard Option B closing-line portfolio simulation and attribution.
- [ ] **Hierarchical Extension (Conditional)**: If global dispersion shows statistically significant edge on totals, evaluate hierarchical dispersion.

## Ideas & Candidate Solutions

- **Ablation Grid**:
  - `m00_baseline_grw_negbin`: Interception + GRW + HomeAdv + NegBin.
  - `m05_wealth_grw_negbin`: Interception + GRW + HomeAdv + ProductionWealth + JointGammaNegBin.
  - `m10_lineup_grw_negbin`: Interception + GRW + HomeAdv + LineupPillar + NegBin.
  - `m12_joint_hybrid_synergy_negbin`: Interception + GRW + HomeAdv + ProductionWealth + LineupPillar + JointGammaNegBin.
- **Direct Controls**:
  - Task 013 runs: `m00_baseline_grw`, `m05_wealth_grw`, `m10_lineup_grw`, `m12_joint_hybrid_synergy_grw` (all persisted in `mcmc_experiments`).

## Work Log & Progress

- [2026-09-11 @antigravity] Completed `/grill-me` alignment interview with trader. Defined `JointGammaNegBinObservation` specification, 4-model ladder, expanded totals/BTTS evaluation scope, and Task 014 creation.
- [2026-09-11 @antigravity] Branched `feat/grw-joint-negbin-observation` off `feat/grw-player-lineup-hybrid`. Prepared worktrees on archpc and mcmc-beast. Handing off to Claude CLI agent.

## Verification & Findings

- To be recorded upon completion of smoke test, 40-fold grid, and totals evaluation.
