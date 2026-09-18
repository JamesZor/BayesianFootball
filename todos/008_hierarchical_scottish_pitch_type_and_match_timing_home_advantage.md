# 008 — Hierarchical Scottish Pitch Type and Match Timing Home Advantage

| Field | Value |
|---|---|
| ID | 008 |
| Title | Hierarchical Scottish Pitch Type and Match Timing Home Advantage |
| Status | COMPLETED |
| Priority | P2 |
| Assignee | claude |
| Created | 2026-09-10 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | [src/models/pregame/components/home_advantage.jl](../src/models/pregame/components/home_advantage.jl); [src/models/pregame/components/interfaces.jl](../src/models/pregame/components/interfaces.jl); [src/features/extractors/](../src/features/extractors/); [experiments/scottish_lower/](../experiments/scottish_lower/) |

## Context & Problem Statement

Current pregame models use a single global home advantage parameter $\gamma_{\text{home}} \sim \mathcal{N}(0.15, 0.05)$ across all Scottish Lower matches. In Scottish Championship, League 1, and League 2, ground conditions and schedules are notoriously heterogeneous:
1. **Pitch Surface (Synthetic Turf vs Natural Grass)**: Multiple clubs play on 3G/4G artificial turf (e.g. Alloa Athletic, Falkirk, Queen of the South, Hamilton Academical, Airdrieonians, Cove Rangers), while others play on grass. Grass-based squads visiting synthetic grounds face altered ball bounce, pace, and unfamiliar footing, creating asymmetric home advantage.
2. **Match Timing & Recovery**: Friday night fixtures (e.g., BBC Alba/BBC Scotland broadcasts), midweek Tuesday night cross-country journeys (e.g. Stranraer traveling to Peterhead or Elgin) where semi-professional squads travel after work, versus standard Saturday 3pm kickoffs.

With ReverseDiff's fast gradient tape, we can replace the crude scalar $\gamma_{\text{home}}$ with a hierarchical contextual home advantage pillar that pools ground-specific and situational effects with Bayesian shrinkage.

## Acceptance Criteria

- [x] Add stadium pitch surface metadata (`is_synthetic_pitch` boolean) and match scheduling metadata (day of week, evening kickoff flag, rest days) to Scottish Lower match feature extraction.
- [x] Implement `HierarchicalContextualHomeAdvantage` in `src/models/pregame/components/home_advantage.jl`: *(built as a prototype in `current_development/hierarchical_home_advantage/l04_contextual_loader.jl` — `HierarchicalTeamHomeAdvantage` + `ContextualCovariate` terms — not in `src`, since it did not earn promotion)*
  $$\gamma_{ij} = \gamma_{\text{base}} + \beta_{\text{turf\_asym}} \cdot (\text{turf}_i \land \neg \text{turf}_j) + \beta_{\text{midweek}} \cdot \text{is\_midweek} + u_i$$
  with ground random effects $u_i \sim \mathcal{N}(0, \sigma_{\text{stadium}}^2)$.
- [x] Ensure non-centered parameterization for $u_i$ to prevent NUTS geometry bottlenecks.
- [ ] Implement zero-allocation feature tensor extraction compatible with compiled ReverseDiff tapes. *(Not met literally: the design vectors are precomputed and tape-safe, but every model on this ReverseDiff stack allocates ~130 KB per gradient; contextual terms add +512–544 B, the same 16 B/club as Phase 1.)*
- [x] Smoke test on 2-fold CV to verify tape compilation and posterior parameter recovery.
- [x] Execute full 40-fold walk-forward grid on `mcmc-beast` (-t 16).
- [x] Audit posterior credible intervals: verify whether $\beta_{\text{turf\_asym}}$ Credible Interval excludes zero.
- [x] Benchmark out-of-sample proper scores (LogLoss, Brier, RPS) against the baseline `GlobalHomeAdvantage`.

## Ideas & Candidate Solutions

- **Surface Asymmetry Directionality**:
  - Asymmetric indicator: $(\text{turf}_i == 1 \land \text{turf}_j == 0)$, testing whether grass teams visiting turf grounds suffer extra penalty.
  - Symmetrical interaction: Also test whether turf teams visiting grass grounds suffer an equivalent penalty.
- **Semi-Pro Midweek Travel Interaction**:
  - In Scottish Leagues 1 and 2, players are semi-professional. A Tuesday evening away trip involving $>100\text{ km}$ of travel represents severe fatigue. Include an interaction feature: $\text{midweek} \times \log(1 + d_{ij})$.
- **Stadium-Level Random Intercepts**:
  - Individual stadium random effects $u_i \sim \mathcal{N}(0, \sigma_{\text{stadium}}^2)$ with a tight hyperprior $\sigma_{\text{stadium}} \sim \text{HalfNormal}(0.05)$ to prevent small-sample overfitting (each club only plays ~18 home games per season).
- **Friday Night Broadcast Effect**:
  - Isolated indicator for televised Friday evening fixtures, which often have higher attendance and different pre-match preparation routines.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task in BACKLOG following user proposal and Scottish Lower domain analysis. Defined surface asymmetry and scheduling specifications.
- [2026-09-12 @antigravity] Claimed task for `claude_hier_ha` in worktree `feat/hierarchical-home-advantage`. Following post-match settlement analysis of the 2026-09-12 slate (where 6 artificial-turf home teams won and the model suffered an away underdog bias), structured task into two phases:
  - **Phase 1 (Team-Level Random Effects)**: Benchmark existing `HierarchicalTeamHomeAdvantage()` across `m05_joint_production_wealth`, `m12_joint_hybrid_synergy`, and `m12_joint_hybrid_synergy_grw` on 2-fold smoke and 40-fold walk-forward grid; perform counterfactual re-pricing of 2026-09-12 slate.
  - **Phase 2 (Contextual Pitch & Travel)**: Implement `HierarchicalContextualHomeAdvantage` with `is_synthetic_pitch` and `travel_distance` covariates.
- [2026-09-12 @claude] Phase 1 prototypes in `current_development/hierarchical_home_advantage/` (`l01_loader.jl`, `l02_evaluation.jl`, `l03_slate.jl`, `r01_smoke.jl`–`r05_slate_repricing.jl`). Each candidate is its flat-HA control with only the HA slot changed; non-HA recipes, splitter and machinery are Task 013's by include. Running on `mcmc-beast` from `/root/BF_hier_ha` (rsync, tmux `hier_ha`); the slate/extension checkout is `/root/BF_hier_ha_slate` (2026-09-12 cache).
- [2026-09-12 @claude] r01 smoke, folds 1–2, 4 × (400 + 400): FAIL 0/3 on gates stricter than the work package's — G1 Δalloc(hier − flat) = +512/+544 B (16 B per club; flat baselines 129–182 KB, so a literal zero-allocation tape is unreachable for every production model) and G3 ESS < 400 on m12 TD (tail 251, non-HA site) and m12 GRW (bulk 361; `ha.σ_γ` tail 285). Tape == ForwardDiff (≤ 6.7e-16), 0 divergences, R̂ ≤ 1.0146, latents and round-trip all passed. Runs 3a3e1509, 50bd541b, 59f31574 in `smoke_hier_ha`.
- [2026-09-12 @claude] User chose a re-smoke at the production budget 4 × (500 + 1000): all sampling gates pass — 0 divergences, R̂ ≤ 1.0087 (ha ≤ 1.0050), ESS bulk ≥ 878 / tail ≥ 610 — G1 Δalloc still fails as disclosed. Runs 6ced7c3c, 3b748983, 550a0d00 in `smoke_hier_ha`. r02 40-fold grid launched on that basis.
- [2026-09-12 @claude] r02 40-fold grid PASS 3/3 in `scottish_lower_hierarchical_ha` (4 × 500+1000, δ 0.80; 710 OOS each): m05 TD `6117c711-a9f9-4539-a36c-40d6ddb6593c` (R̂ 1.0081, ESS 921/652, 4 div, 6.7 min); m12 TD `87c1052d-a181-434c-a965-3c8c801f4142` (R̂ 1.0097, ESS 1071/835, 0 div, 8.5 min); m12 GRW `744cf6bd-0441-462c-8bd1-2a5169d70b4b` (R̂ 1.0106, ESS 716/439, 0 div, 48 min). σ_γ identified under GRW on every fold (fold 40 median 0.087 [0.048, 0.131], P(σ<0.02) 0.006) but pulled toward zero under both TimeDecay models (fold 40 median 0.035–0.037, P(σ<0.02) 0.28–0.30 vs prior 0.159). r04 evaluation, r03 Fold-43 extension and r05 re-pricing launched as one chain.
- [2026-09-12 @claude] r04 (710 fixtures / 2,899 scored rows, B = 10,000; control reproduced 0.64337 / 0.0100): no proper-score gain. ΔLogLoss(hier − flat) m05 −0.00027 [−0.00092, +0.00038], m12 TD −0.00004 [−0.00055, +0.00046], m12 GRW +0.00065 [−0.00084, +0.00215]; 1X2/OU2.5/BTTS all null. On turf-home fixtures m05 (+0.00095 [+0.00005, +0.00181]) and GRW (+0.00271 [+0.00004, +0.00525]) are significantly WORSE. P(mean γ_turf > mean γ_grass) at fold 40: 0.48 / 0.52 / 0.13. Calibration improves on TD (m12 ECE 0.0100 → 0.0068, m05 0.0149 → 0.0114) and worsens on GRW totals/BTTS. Surface cuts excluding the one turf-home T003 fixture added to r04 for a re-run; the work package's turf list is unverified (Phase 2 item).
- [2026-09-12 @claude] r04 re-run with surface cuts excluding T003: turf-home cut unchanged (813 rows / 178 fixtures — the turf-home T003 fixture, East Kilbride v Spartans, has no Betfair close and was never scored), so the significant turf-home deficits (m05 +0.00095, GRW +0.00271) are not a T003 artefact; grass-home gains shrink toward zero without the two fold-1 T003 fixtures. H2/H4 remain conditional on the unverified turf list only. r03 extended m12 TD hier (`87c1052d…`) to 43 folds / 769 OOS (new folds R̂ 1.0101, ESS 782, 0 div, 1.6 min; its closing CSV write failed on a missing results dir — fixed with `mkpath`, fold data persisted). r05 queued.
- [2026-09-12 @claude] T003 extended: the composable engine prices an unmapped home club at γ = 0 under a hierarchical HA but γ_global under the flat one (`engine.jl:629`), biasing the A/B. 3 of 710 grid fixtures; r04 reports every contrast with and without them.

- [2026-09-12 @claude] r05 2026-09-12 T−25 counterfactual (replay engine, Fold 43 both arms). Live ledger read: 11 legs, realised −£45.89 (matches). `flat_optB` re-price contains all 11 live legs (+4), max |Δrisk| £3.64 — residual is lineups (0/9 provisional XIs at T−25 vs the live BBC source). Hierarchical HA moved P(home) by ≤ 0.003 on every fixture; `hier_optB` backs the same 7 away legs (6 on turf grounds) as `flat_optB` and drops only a £1.04 draw leg. Full-fill P&L −£54.26 vs −£54.52. Raw model P(home) is 0.40–0.43 across all nine fixtures, a gap to the market far larger than any HA slot movement. Raised T009 (`.env` loaded at precompile time; `BF_DB_URL` unset outside an exporting shell) after r05's first launch failed on it.
- [2026-09-18 @antigravity] Phase 2 activated via /grill-me. Verified Scottish Lower pitch surface registry with `is_synthetic_pitch` in `scottish_stadium_geocodes.csv` (17 synthetic, 14 grass). Authored `WORK_PACKAGE_PHASE_2_TURF_TIMING.md` covering asymmetric turf advantage, general turf scoring intensity, and midweek schedule fatigue. Tasked Claude agent in tmux session `claude_hier_ha`.

- [2026-09-18 @claude] Phase 2 executed on `mcmc-beast` (smoke too — beast idle; standing no-archpc-MCMC rule). `l04_contextual_loader.jl`: γ_base + u_i via `HierarchicalTeamHomeAdvantage` (work-package priors) plus scalar `ContextualCovariate{K}` terms through the builder's covariate contract, with a new `HomeOnlyRole`; no engine change. Registry spot-checked against public sources; one dated override (Dumbarton turf from 2026/27); Falkirk turf throughout (2023 install replaced an older artificial surface); notes-field commas quoted in the CSV. Rest days are league-only (betdb has no cup fixtures).
- [2026-09-18 @claude] r06 smoke folds 1–2, 4 × (500+1000): PASS 3/3 (RD==FD ≤ 7.1e-16, 0 div, R̂ ≤ 1.0047, ESS ≥ 1059). r07 40-fold grid in `scottish_lower_contextual_ha`: turf_asym `d20ff61d…` PASS; turf_dual attempt 1 FAILED tail ESS 327 (fold 29, non-HA site), not persisted, resampled → `a50e1171…` PASS; contextual `37feea2c…` PASS. r09 rung 5 `m12_joint_hybrid_contextual` (full contextual set, 43 folds) `991e4991…` PASS.
- [2026-09-18 @claude] r08 (710 fixtures, B = 10,000; controls reproduced r04): no ΔLogLoss gain on any rung, scope or surface/timing cut (1 of 105 cells nominally significant, on a rung without the term it cuts on). H1 fails (P(β_asym>0) 0.80–0.88 at fold 40 vs prior 0.84), H2 fails (β_pace ≈ 0; raw 2.71 vs 2.67 goals turf/grass), H3 fails (P(β_mid>0) 0.62 < prior 0.84), H4 fails. r09 slate: P(home) moves ≤ 0.011; same 7 away legs (6 on turf); −£52.61 vs −£54.52 full fill. H5 fails. Recommendation: close without promotion. Write-up: `current_development/hierarchical_home_advantage/README.md` §Phase 2.

## Verification & Findings

Phase 1 (existing `HierarchicalTeamHomeAdvantage`) — complete. Full write-up:
[`current_development/hierarchical_home_advantage/README.md`](../current_development/hierarchical_home_advantage/README.md).

| step | command (mcmc-beast) | result | wall |
|---|---|---|---|
| smoke 4 × (400+400) | `julia --project -t 16 …/r01_smoke.jl` | FAIL 0/3 — G1 Δalloc +16 B/club; G3 ESS on m12 TD / GRW | ~5 min |
| smoke 4 × (500+1000) | `R01_SAMPLES=1000 R01_WARMUP=500 R01_CHAINS=4 julia … r01_smoke.jl` | sampling gates PASS 3/3; G1 Δalloc still fails (structural, disclosed) | ~8 min |
| 40-fold grid | `julia --project -t 16 …/r02_production_grid.jl` | PASS 3/3 — runs `6117c711…`, `87c1052d…`, `744cf6bd…` | 6.7 / 8.5 / 48 min |
| evaluation | `julia --project -t 16 …/r04_evaluate.jl` | control reproduced 0.64337 / 0.0100; no ΔLogLoss gain; turf-home significantly worse for m05 and GRW | ~8 min |
| Fold 43 | `julia --project -t 16 …/r03_extend_2627.jl` (from `/root/BF_hier_ha_slate`) | 43 folds / 769 OOS; new folds R̂ 1.0101 | 1.6 min |
| slate | `…/r05_slate_repricing.jl` with `BF_DB_URL` exported | P(home) Δ ≤ 0.003; same away legs; −£54.26 vs −£54.52 | ~5 min |

Posterior and score findings:

* **σ_γ** (fold 40): m05 TD 0.037 [0.004, 0.097], m12 TD 0.035 [0.004, 0.094] — boundary mass 0.28–0.30 above the prior's 0.159, i.e. pulled toward zero; m12 GRW 0.087 [0.048, 0.131], boundary mass 0.006 — identified.
* **Turf − grass mean γ** (fold 40): P(turf > grass) 0.48 / 0.52 / 0.13. The work package's turf list is unverified and should be replaced by a dated `is_synthetic_pitch` feature before Phase 2 tests β_turf.
* **ΔLogLoss vs GlobalHomeAdvantage twin**: m05 −0.00027 [−0.00092, +0.00038]; m12 TD −0.00004 [−0.00055, +0.00046]; m12 GRW +0.00065 [−0.00084, +0.00215]. Turf-home fixtures: m05 +0.00095 [+0.00005, +0.00181], GRW +0.00271 [+0.00004, +0.00525] (unaffected by T003).
* **ECE**: m12 TD 0.0100 → 0.0068, m05 0.0149 → 0.0114; m12 GRW 0.0086 → 0.0111, worse on O/U 2.5 and BTTS.
* **β_turf / β_timing**: not estimated — Phase 2 scope.

Recommendation: do not promote a `_hier_ha` model on predictive grounds; if the TD calibration gain matters, run an Option B portfolio simulation against `m12_hybrid_td_raw` first. Tickets touched: T003 (extended), T009 (raised).

Phase 2 (contextual turf / timing HA) — complete, see README §Phase 2. No rung improves
LogLoss (m05 contextual +0.00001 [−0.00094, +0.00096]; m12 contextual +0.00004
[−0.00094, +0.00103]); turf asymmetry, turf pace and midweek coefficients are prior-dominated;
the 2026-09-12 away legs are unchanged. Do not promote.
