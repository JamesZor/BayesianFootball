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
| Related Files / Commits / PRs | [current_development/grw_player_hybrid/](../current_development/grw_player_hybrid/README.md); `src/models/pregame/components/dynamics/team_level/multiscale.jl`; `src/models/pregame/builder/grw_dynamics.jl`; `current_development/match_day_inference/l10_t25_backtest.jl`; `feat/grw-player-lineup-hybrid`; PR #30 |

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

- [x] **Architecture Implementation**: `PlayerLineupPillar` composes with `MultiScaleGRW` in `CountModelBuilder` with no `src/` change — tape compiles on both GRW branches (no-target fold 1, target fold 2), RD ≡ FD to 1e-15, 2 extra parameters and 25 extra tape instructions over `m05_wealth_grw`; latents extract at 710 × 2,000 with no dimension mismatch.
- [x] **Smoke Gate (Folds 1–2)**: PASS 4/4 at the production sampler (R̂ ≤ 1.0124, ESS ≥ 886, 0 divergences, latents, Postgres round-trip). At the work-package budget 2 × (50 + 100) R̂ was 1.05–1.13 and ESS ≥ 400 is unreachable in 200 draws — recorded, see README §2.
- [x] **40-Fold Walk-Forward Grid**: 4/4 pass the six-part audit, 0 divergences / 640,000, R̂ ≤ 1.0149, ESS ≥ 487; `QueuedNUTSConfig(1000, 500 warmup, 4 chains, δ 0.80)`. Strict R̂ ≤ 1.01 (advisory) holds for m10 only.
- [x] **2026/27 Opening Slates Extension**: folds 41–43 via `extend_fit`, 769 fixtures (59 new), new folds R̂ ≤ 1.0147, 0 divergences; m00 / m10 new-fold ESS 320 / 342 at the matched 500-draw budget (run-level flag false), m05 / m12 pass.
- [x] **Evaluation**: LogLoss, Brier, RPS, ECE per market vs Betfair close and vs TimeDecay controls, 10,000-resample fixture-clustered paired bootstrap. Controls reproduce their published scores exactly.
- [x] **Portfolio Backtesting & Attribution (Task 012)**: Option B (`MatchDay.option_b_system()`) at the Betfair close (632 fixtures) with capture ratio, shared-bet sizing, disjoint sets; T−25 order book (TouchOnly + LadderSweep) with the six published benchmark tracks reproduced to the penny.
- [ ] **Code Quality & Upstream Merge**: `test/test_multiscale_grw.jl` 124/124 and `test/test_player_lineup_dynamics.jl` 149/149 on mcmc-beast; PR opened from `feat/grw-player-lineup-hybrid` — merge to `main` pending review.

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

- [2026-09-11 @claude] r02 production grid PASS 4/4 in 2 h 36 min (28 / 41 / 39 / 48 min per model). Artefacts persist every 2nd of 4,000 draws (a full artefact would exceed the 1 GB bytea field); audit uses all draws.
- [2026-09-11 @claude] r03 extended all four runs to 43 folds in place (~1.4 min each). r04 scored the 710-fixture panel; r05 simulated Option B at the close and persisted the four ladder portfolios (ledgers reload identically); r06 ran the T−25 order-book backtest. Verdict: do not promote the GRW hybrid over production `m12` — see Verification.

## Verification & Findings

Full write-up with every table: [`current_development/grw_player_hybrid/README.md`](../current_development/grw_player_hybrid/README.md).

**Runs** (`mcmc_experiments`, namespace `scottish_lower_grw_player_hybrid`, 43 folds after extension):

| Model | Run UUID | R̂ max (40-fold) | ESS min | Divergences | Closing-line portfolio UUID |
|---|---|---:|---:|---:|---|
| `m00_baseline_grw` | `158d2a80-7ea3-4d6c-b3ab-be62bcf1bc11` | 1.0105 | 785 | 0 | `ec5bf23e-c500-4253-a7fb-9e952d732eba` |
| `m05_wealth_grw` | `b0961bc4-c40c-4dbe-9c05-57df7ae0839e` | 1.0101 | 821 | 0 | `3982ffb9-1d1c-482c-92d0-995b6bca6d8e` |
| `m10_lineup_grw` | `b13c8fb9-ce34-4210-aa3f-9d2ed493c286` | 1.0093 | 487 | 0 | `f988117b-78cc-46fa-bc30-913659bafdf8` |
| `m12_joint_hybrid_synergy_grw` | `3a9a4c7e-378b-45d0-a2d2-c8b69b46786b` | 1.0149 | 678 | 0 | `c2aedffe-6e77-4f29-a79b-b8c688ae33bc` |

**Proper scores** (710 fixtures, 2,899 selections, Betfair TWA close):

| Model | LogLoss | RPS | ECE |
|---|---:|---:|---:|
| `m05_joint_td_raw` | 0.64299 | 0.22415 | 0.0149 |
| `m05_wealth_grw` | 0.64315 | 0.22383 | 0.0123 |
| `m12_hybrid_td_raw` (production) | 0.64337 | 0.22447 | 0.0100 |
| `m00_baseline_grw` | 0.64433 | 0.22492 | 0.0184 |
| `m12_joint_hybrid_synergy_grw` | 0.64437 | 0.22493 | **0.0086** |
| `m10_lineup_grw` | 0.64561 | 0.22631 | 0.0092 |
| Betfair close | 0.64182 | 0.21110 | 0.0139 |

ΔLL GRW hybrid − TD hybrid = +0.00100 [−0.00299, +0.00497]; lineup on GRW + joint +0.00122 [−0.00073, +0.00322]; GRW vs TD with the joint arm +0.00016 [−0.00368, +0.00403]. Nothing is significant, including every model against the close.

**Portfolio** (Option B):

| Model | Close: return / Sharpe / MDD / capture | T−25 TouchOnly | T−25 LadderSweep |
|---|---|---:|---:|
| `m12_joint_hybrid_synergy_grw` | +351.9% / 1.309 / −52.6% / 1.043 | £557.71 (ROI 16.7%) | £595.53 |
| `m12_hybrid_td_raw` | +606.5% / 1.487 / −42.4% / 0.949 | £551.97 (13.7%) | £575.94 |
| `m05_wealth_grw` | +385.8% / 1.453 / −42.7% / 1.080 | £588.43 (25.2%) | £600.33 |
| `m05_joint_grw_raw` | +404.3% / 1.498 / −42.8% / 1.094 | £590.05 (25.6%) | £600.22 |

Attribution, GRW hybrid vs TD hybrid at the close: 1,062 shared bets (71%) where GRW's ROI is higher (13.41% vs 12.92%) and sizing ΔPnL is −0.032; the gap is selectivity — TD's 240 exclusive bets returned +29.7%, GRW's 191 returned −11.9%.

**Conclusions.** (1) The GRW state and the proxy-xG joint likelihood are substitutes, confirming Task 007. (2) The lineup pillar buys calibration (ECE −30 to −50%) but not LogLoss or bankroll, on either dynamics — Exp 06's finding reproduced on GRW state. (3) Do not promote `m12_joint_hybrid_synergy_grw` over production `m12`; the strongest GRW arm for staking remains team-level `m05` joint.
