# 011 — Fuse Goal Decomposition with Proxy xG and Player Lineup Dynamics

| Field | Value |
|---|---|
| ID | 011 |
| Title | Fuse Goal Decomposition with Proxy xG and Player Lineup Dynamics |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | `experiments/scottish_lower/08_goal_decomposition/`, `todos/002_complete_experiment_08_goal_decomposition_production_grid.md`, `src/models/pregame/components/lineup/` |

## Context & Problem Statement

Experiment 08 (`08_goal_decomposition`) evaluated decomposing team goals into three distinct event processes: regular open-play goals, penalty attempts/conversions, and own-goal receipts. The completed 40-fold evaluation (`fullcomparison_2026-09-11.md`) showed that all three decomposed arms (`m01`, `m02`, `m03`) lagged the production champion `m12_joint_hybrid_synergy` by **+0.0062 LogLoss** ($95\%$ CI strictly excluding zero), suffered **15× worse Over/Under 2.5 calibration** (ECE $0.0016 \to 0.025$), and incurred deep drawdowns ($-32.7\%$ vs $-19.6\%$).

The post-mortem identified that Experiment 08 was evaluated in **complete isolation from the repository's primary alpha engines**:
1. **No Proxy-xG Arm**: It discarded the two-arm joint Gamma-Poisson observation (`JointGammaPoissonObservation`), which extracts continuous signal from ~25 BBC shot events per match.
2. **No Player Lineups**: It omitted `PlayerLineupPillar`, which injects announced T−60 starter and bench RAPM ratings.
3. **No Squad Wealth**: It omitted `ProductionWealthCovariate` (Transfermarkt market valuations).
4. **Sparsity & Convex Distortion**: Estimating team penalty and own-goal parameters from rare counts (penalties occur ~0.25/match, own goals ~0.05/match) created extreme parameter variance, while recombining intensities via additive exponentiation distorted the marginal totals distribution.

This task investigates whether event decomposition can be made viable and competitive by fusing it with continuous proxy-xG, player-level lineup dynamics, and squad wealth, or whether player-level features subsume incident decomposition altogether.

## Acceptance Criteria

- [ ] **Data & Feature Pipeline Audit**: Verify availability of decomposed proxy-xG features (open-play pxG vs set-piece/penalty pxG) from BBC commentary event tags alongside team lineups.
- [ ] **Mathematical Formulation**: Formulate an AD-safe, ReverseDiff-compilable hybrid model combining decomposed event likelihoods with `PlayerLineupPillar` and `ProductionWealthCovariate`.
- [ ] **Totals Calibration Guard**: Implement an explicit score-grid or totals-smile formulation (e.g. `SmileScoreGrid` or totals intensity coupling) to prevent Jensen convex distortion from degrading Over/Under 2.5 calibration.
- [ ] **Fast 2-Fold Preflight**: Pass 2-fold CV smoke tests with zero ReverseDiff tape allocations, zero divergences, R̂ ≤ 1.01, and ESS ≥ 400.
- [ ] **40-Fold Walk-Forward Production Grid**: Run full Scottish Lower 24/25 + 25/26 grid on `mcmc-beast` under queued execution.
- [ ] **Like-for-Like Benchmark vs `m12`**: Evaluate on the canonical 710 OOS fixtures across proper scores (LogLoss, Brier, RPS, CRPS, ECE) and the equal-book Betfair Kelly portfolio (CAGR, growth per slate, Hurdle $G$, Calmar, max drawdown).
- [ ] **Scientific Verdict**: Formally determine whether incident decomposition adds marginal value over `m12_joint_hybrid_synergy` when provided identical player and proxy-xG information.

## Ideas & Candidate Solutions

### Candidate A: Hybrid Lineup Decomposition (Goals Decomposed + Player Lineups + Wealth)
- Attach `PlayerLineupPillar` directly to the open-play goal intensity $\eta_{\text{regular}}$, and `ProductionWealthCovariate` to team supremacy.
- Leave penalties and own goals to hierarchical team/referee baselines.
- *Pros*: Simple extension within the existing `CountModelBuilder` DSL.
- *Cons*: Open-play goals remain sparse without proxy xG; may not overcome the ~0.006 LogLoss deficit.

### Candidate B: Decomposed Two-Arm Joint Likelihood (Decomposed Goals + Decomposed pxG)
- Decompose **both** observation arms:
  - Open-play: $\text{pxG}_{\text{open}} \sim \text{Gamma}(\nu_1, \mu_{\text{open}}/\nu_1)$ and $y_{\text{open}} \sim \text{Poisson}(\kappa_1 \cdot \mu_{\text{open}})$
  - Penalties: $\text{pxG}_{\text{pen}} \sim \text{Gamma}(\nu_2, \mu_{\text{pen}}/\nu_2)$ and $y_{\text{pen}} \sim \text{Binomial}(n, k)$
- Pair with `PlayerLineupPillar` and `ProductionWealthCovariate`.
- *Pros*: Directly eliminates the event count sparsity trap by providing continuous shot-level density for each goal type.
- *Cons*: Requires feature extraction to cleanly partition BBC commentary into open-play vs set-piece shot events.

### Candidate C: Player-Level Incident Attribution (Penalties at Player Level)
- Rather than estimating team penalty drawing/conceding skills, assign penalty expectation to the designated penalty taker announced in the XI.
- *Pros*: Grounded in physical reality (penalty conversion is a player skill, not a team constant).
- *Cons*: Small sample sizes for individual penalty takers in lower divisions; requires historical penalty taker tracking.

## Work Log & Progress

- [2026-09-10 @antigravity] Task created and scoped in response to Experiment 08 portfolio and evaluation results. Identified proxy-xG omission, player-lineup omission, and totals calibration collapse as primary root causes to resolve.

## Verification & Findings

- [2026-09-10 @antigravity] Benchmark baseline established from Experiment 08 full comparison (`results/fullcomparison_2026-09-11.md`):
  - Current champion `m12_joint_hybrid_synergy`: LogLoss **0.64337**, ECE **0.0094**, CAGR **61.80%**, Hurdle $G$ **3.20 bps**, MDD **−19.63%**.
  - Isolated decomposed baseline `m01_decomposed_baseline`: LogLoss **0.64928**, ECE **0.0213**, CAGR **50.29%**, Hurdle $G$ **1.48 bps**, MDD **−32.74%**.
  - Target for Todo 011: Challenger must beat LogLoss 0.64337 and maintain ECE ≤ 0.0100 on the identical 710 canonical fixtures.
