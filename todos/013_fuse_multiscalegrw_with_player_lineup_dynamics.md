# 013 — Fuse MultiScaleGRW with Player Lineup Dynamics

| Field | Value |
|---|---|
| ID | 013 |
| Title | Fuse MultiScaleGRW with Player Lineup Dynamics |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-11 |
| Updated | 2026-09-11 |
| Related Files / Commits / PRs | `src/models/pregame/components/dynamics/team_level/multiscale.jl`, `src/models/pregame/builder/grw_dynamics.jl`, `current_development/match_day_inference/r13_t25_grw_backtest.jl`, `feat/grw-player-lineup-hybrid` |

## Context & Problem Statement

In Scottish Lower leagues (tournaments 56/57), two independent modeling innovations produced large out-of-sample improvements:
1. **Gen 4 Hybrid Architecture (`m12_joint_hybrid_synergy`)**: Composes a team-level `TimeDecayDynamics(180.0)` baseline with an announced XI `PlayerLineupPillar(rating=:shots_rapm, w_bench=0.10)` and `JointGammaPoissonObservation()`, achieving an ECE of 0.0100 (vs Betfair close 0.0139) and +136.6% bankroll growth.
2. **MultiScaleGRW Latent State-Space (Task 007)**: Replaces exponential time decay with an unconstrained two-speed Gaussian Random Walk (persistent slow drift $\sigma_{\text{slow}} \sim \text{TruncatedNormal}(0, 0.08)$ + volatile fast shock $\sigma_{\text{fast}} \sim \text{TruncatedNormal}(0, 0.25)$ with mean-reversion $\phi \sim \text{Beta}(8, 2)$). On 2026/27 opening slates at T−25, MultiScaleGRW achieved +25.59% ROI / +18.01% bankroll growth vs +13.69% ROI / +10.39% bankroll growth for `m12_hybrid_td`.

However, MultiScaleGRW has hitherto only been evaluated at the team level (`m05_joint_grw`). The hypothesis is that fusing MultiScaleGRW team dynamics with announced XI player lineup dynamics (`PlayerLineupPillar`) will combine the regime-adaptive team strength tracking of GRW with tactical XI adjustment, yielding superior calibration, lower LogLoss, and higher Sharpe.

Constraints:
- ReverseDiff tape compilation safety: zero allocations inside the gradient tape.
- Must execute on the 40-fold walk-forward cross-validation grid (24/25 + 25/26, 710 matches) with extension to 2026/27 opening slates (Folds 41–43).
- Must adhere to the mandatory 2-fold smoke gate before full 40-fold dispatch.
- Attribution via Task 012 standards (Capture Ratio, Shared-Bet Sizing, Disjoint Edge).

## Acceptance Criteria

- [ ] **Architecture Implementation**: Ensure `PlayerLineupPillar` composes cleanly with `MultiScaleGRW` in `CountModelBuilder` without tape compilation failure or latent extraction dimension mismatch.
- [ ] **Smoke Gate (Folds 1–2)**: Run 2-fold smoke test on `m00_baseline_grw`, `m05_wealth_grw`, `m10_lineup_grw`, and `m12_joint_hybrid_synergy_grw`. Verify gradient tape compilation, 0 divergences, R̂ < 1.05, bulk/tail ESS > 400.
- [ ] **40-Fold Walk-Forward Grid**: Train the 4 models across Folds 1–40 (2024/25 and 2025/26, 710 matches) on `mcmc-beast` using queued NUTS execution (`QueuedNUTSConfig(1000, 500, 4)`).
- [ ] **2026/27 Opening Slates Extension**: Extend all 4 models to Folds 41–43 using `extend_fit`.
- [ ] **Evaluation**: Compute LogLoss, RPS, Brier, ECE, and paired bootstrap significance vs Betfair closing odds and vs `m12_joint_hybrid_synergy` (TimeDecay baseline).
- [ ] **Portfolio Backtesting & Attribution (Task 012)**: Run Option B portfolio simulation (`TieredTrust`, `FractionalKelly 0.3`, `SlateDrawdown 8.0`) on Betfair closing and T−25 1-minute order book (`betfair_live.order_book_1m`). Quantify Capture Ratio, shared-bet sizing, and disjoint edge.
- [ ] **Code Quality & Upstream Merge**: All code clean, documented, tests passing (`test/test_multiscale_grw.jl` + player lineup tests), merged to `main` via PR.

## Ideas & Candidate Solutions

- **4-Model Ablation Grid**:
  - `m00_baseline_grw`: Intercept + HomeAdvantage + MultiScaleGRW + Poisson.
  - `m05_wealth_grw`: Intercept + HomeAdvantage + MultiScaleGRW + ProductionWealth + JointGammaPoisson (team-level benchmark).
  - `m10_lineup_grw`: Intercept + HomeAdvantage + MultiScaleGRW + PlayerLineupPillar (shots_rapm, w_bench=0.10) + Poisson.
  - `m12_joint_hybrid_synergy_grw`: Intercept + HomeAdvantage + MultiScaleGRW + ProductionWealth + PlayerLineupPillar + JointGammaPoisson (full hybrid).
- **Benchmark Controls**:
  - `m05_joint_td_raw` (team TimeDecay)
  - `m12_hybrid_td_raw` (Gen 4 production hybrid TimeDecay)

## Work Log & Progress

- [2026-09-11 @antigravity] Conducted `/grill-me` alignment interview with human trader. Finalised ablation ladder, evaluation scope, and Option B portfolio attribution requirements.
- [2026-09-11 @antigravity] Evaluated prototype MultiScaleGRW on 2026/27 opening slates at T−25 order book: +25.59% ROI vs +13.69% for `m12`.
- [2026-09-11 @antigravity] Created Task 013, branched `feat/grw-player-lineup-hybrid` off `feat/multiscale-grw-dynamics`, locked `Manifest.toml` to `Distributions` v0.25.126. Handing off to Claude CLI agent on `mcmc-beast`.

- [2026-09-11 @claude] Built `current_development/grw_player_hybrid/` (`l01_loader.jl` inference, `l02_evaluation.jl` scoring/portfolio/attribution, runners `r01`–`r06`). Recipes are Exp 06 `l60` verbatim except `TimeDecayDynamics(180)` → `MultiScaleGRW()`; split is the canonical `GroupedCVConfig` (the prompt's `CVConfig(window_seasons = 3)` would not produce the 40-fold/710 grid the controls were scored on). The prompt's `PlayerLineupPillar(rating = :shots_rapm, fit_on = :history)` shorthand maps to `ShotsPlusMinusFeature(λ = 1000, half_life_days = 730, fit_on = :history)` + `BenchWeightedPlayerAggregation(0.10)`.
- [2026-09-11 @claude] Smoke gate (folds 1–2). At the work-package budget (2 × (50 + 100)): 0 divergences but R̂ 1.05–1.13 and ESS 17–51 — a budget artefact, not a geometry one. At 4 × (400 + 400): all gates but ESS (m00 348, m12 343). At the production sampler 4 × (500 + 1000): **PASS 4/4** — R̂ ≤ 1.0124, min ESS 886, 0 divergences, RD ≡ FD to 1e-15 on both GRW branches, latents finite, Postgres round-trip exact. Tape sizes for m00/m05 match Task 007 instruction-for-instruction; the lineup pillar adds 2 parameters and 25 tape instructions. Reports under `current_development/grw_player_hybrid/results/smoke/`.
- [2026-09-11 @claude] Launched r02 40-fold production grid on mcmc-beast (tmux `grw_player_r02`), namespace `scottish_lower_grw_player_hybrid`.

## Verification & Findings

- To be recorded upon completion of smoke test, 40-fold grid, and 2026/27 T−25 backtest.
