# 028 — Cross-Tier Scottish Pyramid and Informative Priors Time-Decay Models

| Field | Value |
|---|---|
| ID | 028 |
| Title | Cross-Tier Scottish Pyramid and Informative Priors Time-Decay Models |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-24 |
| Updated | 2026-09-27 |
| Related Files / Commits / PRs | `experiments/scotland/01_time_decay_cross_tier_and_priors/` |

## Context & Problem Statement

Current production Scottish Lower models (`m05`, `m12`) train strictly on Tournaments 56 (League 1) and 57 (League 2) in isolation. Under flat shrinkage priors ($\alpha_i \sim \mathcal{N}(0, \sigma^2)$), newly transitioned clubs dropping from higher tiers (e.g., Ross County, Hamilton, Dunfermline) enter with zero historical data and an assumed average talent level ($\alpha = 0$).

In live operations (e.g. 2026-09-19), this manufactured a false +23.8 pp model edge on extreme underdogs (Cove Rangers @ 7.20 vs Ross County @ 1.45), drawing 46% of slate capital and losing. Empirical EDA (TODO 027) confirmed that relegated clubs average +0.26 to +0.40 goal differentials in their first 10 matches (Ross County +2.57 in 26/27), which the market prices at 65–70% win probability while isolated lower-league models price at ~37%.

This task designs, trains, and benchmarks candidate solutions across the Scottish football pyramid using time-decay dynamics (excluding Gaussian Random Walk dynamics for now):
- **Option A (All-SPFL 4-Tier Model)**: Pools all 4 SPFL leagues (Premiership 54, Championship 55, League 1 56, League 2 57) so relegated clubs retain their attack/defence ratings across transitions. Tests both shared continuous ratings with zero-sum league scoring offsets $\delta_{\text{league}}$ (Option A1) and ordered hierarchical tier steps $\tau_{\text{tier}}$ (Option A2).
- **Option B (Informative Cold-Start Priors for Scottish Lower)**: Retains fast lower-league training (56, 57) but initializes new entrants with informative priors based on structural relegation offsets $\mu_{\text{relegated}}$ (Option B1) or prior-season closing market supremacy (Option B2).
- **Baseline Control**: Standard Scottish Lower time-decay model (`ScottishLower` [56, 57] baseline, 180-day half-life).

## Acceptance Criteria

- [x] Dedicated experiment directory established at `experiments/scotland/01_time_decay_cross_tier_and_priors/`.
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

- [2026-09-24 @pi] Claimed Stage 1 implementation: added `ScottishAll`; developing five Poisson candidates and strict seven-gate smoke runner. User approved isolated rsync execution (no commit/push), A2 first-observed training-window tier anchors, B1 no promoted offset, and B2 last-five prior-season higher-tier market-implied goal difference. Production and later likelihood stages remain gated.

- [2026-09-24 @pi] Completed remote five-candidate Stage 1 smoke; all candidates sampled 2×100 and round-tripped CountLatents through PostgreSQL. Corrected reporter/getter/audit-namespace/A2-extraction bugs and reran the corrected suite. All candidates still fail genuine tape allocation and convergence gates; B1/B2 also fail grid tail mass. Portfolio gates abstain, and production prepare-only correctly refuses absent promotion manifests. BLOCKED on these gates; no production MCMC or Stage 2/3 launched. Results and UUIDs: `experiments/scotland/01_time_decay_cross_tier_and_priors/results/STAGE1_SMOKE_FINDINGS.md`.

- [2026-09-24 @pi] User approved a separate longer one-fold pilot at 4×1,000 draws, 500 adaptation and acceptance **0.65** (overriding the work-package's 0.85 proposal). Completed all five on mcmc-beast: all pass R-hat/ESS; m00/B1/B2 pass all convergence checks, A1/A2 retain 22/1 divergences. All pilot Fits pass PostgreSQL parity; sampler/filtration/persistence assertions 55/55. Tape/grid blockers remain and no production/portfolio was launched. Recorded five new UUIDs in `results/LONGER_PILOT_FINDINGS.md`; prepared production recipe corrected to 0.65.

- [2026-09-26 @claude] Re-scoped with the human: the remaining work moves to TODO 031 (TimeDecay and GRW side by side across lower / SPFL / SPFL + cups × Poisson / joint) on the shared harness of TODO 030, which reclassifies the tape-allocation and grid-mass gates as warnings. A1/A2/B1/B2 become TODO 031 wave-2 arms; B1 uses TODO 029's +0.12 relegated prior, not +0.90. Stays BLOCKED until 031 lands, then it is closed as superseded.

- [2026-09-27 @claude] **Closed as SUPERSEDED** (status COMPLETED is the tracker's only closed state; this task's own acceptance criteria were *not* met here). With the human's agreement, its arms are redefined in TODO 031 wave 2 (`experiments/scotland/04_pyramid_tier_transition_arms/README.md`):
  - A1 → league deltas on the intercept and log κ;
  - A2 → the anchored tier component with carry κ and jump η;
  - B1 → a lower-scope B1-literal arm (TODO 029 priors), subsumed by A2 + carry + jump in pooled scopes;
  - B2 → deferred to TODO 031 W4.

  The hand-written `l01_cross_tier_loader.jl` engines are not reused.

## Verification & Findings

- [2026-09-24] Local `test/data_tests.jl`: **55/55 passed**, including ScottishAll routing (Julia 1.12.1, `--project -t 8`). Standalone routing check: 4/4 passed.
- [2026-09-24] Remote fresh ScottishAll SQL load passed: 4,324 unique matches in four tournaments, no duplicate IDs, no checked-domain orphan IDs; per-tier odds/lineup/BBC coverage recorded in `results/data_preflight.md`.
- [2026-09-24] Corrected remote smoke `20260924_235900`: all five pass G2/G4/G6; all fail G1/G3; m00/A1/A2 pass G5, B1/B2 fail its grid-mass floor; G7 blocked and not run. Replay allocations 64,152–135,832 bytes; max R-hat 1.0537–1.0993; tail ESS 24.6–73.1. All five have zero divergences. Exact audit JSON and five persisted diagnostic Fit UUIDs are in the findings report.
- [2026-09-24] Focused loader tests: **15/15 passed**, including two-chain A2 extraction regression. Production sampler API: **7/7 passed**; prepare-only promotion guard refuses the failed smoke, with no grid sampling.
- [2026-09-24] Longer pilot `20260925_001951` (remote timestamp): max R-hat **1.00375–1.00839**, min bulk ESS **1,271–1,513**, min tail ESS **1,465–1,878**. m00/B1/B2 converge; A1/A2 fail zero-divergence gating (22/4,000 and 1/4,000). All five yield 20×4,000 CountLatents draws and exact DB parity. This does not retrospectively pass the 2×100 smoke.
- Full-grid scores, live slate audit, portfolio results, zero-allocation tapes and complete seven-gate success remain **unfulfilled**. Remaining blockers: A1/A2 divergences, tape allocation/latency, and finite-grid tail coverage. Thresholds were not relaxed.
