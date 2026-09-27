# 031 — Scottish pyramid dynamics x scope x observation matrix

| Field | Value |
|---|---|
| ID | 031 |
| Title | Scottish pyramid dynamics x scope x observation matrix |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-26 |
| Updated | 2026-09-27 |
| Related Files / Commits / PRs | TODO 030 (dependency), TODO 028 (superseded), TODO 029, `current_development/grw_pyramid_cups/`, `experiments/scotland/` |

## Context & Problem Statement

Scottish Lower models compress team strength against the market (market-on-model slope
2.53 for pure-Poisson TimeDecay, 1.57 for the live `m12_td`), which mis-priced relegated
clubs such as Ross County. The evidence so far:

- TODO 029 / `grw_pyramid_cups`: pooling the four SPFL leagues alone does not help the
  League One/Two book. Adding SPFL-vs-SPFL cup bridges is the only effect that clears zero
  on 1X2 (−0.0031). The joint GRW on leagues + cups (g3) is the least compressed joint model
  measured (slope 1.10) at a log-loss tied with the live models.
- TODO 028: the time-decay cross-tier arms were never scored on the grid. They were
  blocked by performance gates that TODO 030 reclassifies.

Decision (human, 2026-09-26): run TimeDecay and GRW side by side, across data scopes and
observations, on the shared harness. Select on the 56/57 walk-forward book, and report
per-league performance against the Betfair close for 54/55 as a monitor.

## Acceptance Criteria

- [x] Blocked until TODO 030's harness passes its g1/g2/g3 validation (TODO 030 completed 2026-09-27).
- [x] Twelve recipes: dynamics {`TimeDecayDynamics(180)`, `MultiScaleGRW`} × scope
      {lower 56/57, SPFL 54–57, SPFL + senior cup ties} × observation {Poisson,
      `JointGammaPoissonObservation`}; no wealth or lineup terms in wave 1.
- [x] Smoke then 40-fold grid for all twelve on `mcmc-beast`; gate reports recorded; any
      `review` flag triaged in the Work Log.
- [ ] (partial: target, t56/t57, transition and Δ vs control and m12_td done; 54/55 monitor
      rows and transition-bias CIs missing — see Work Log) Grid scorecard (`scotland_default_v1`) for all twelve: target, per-tier (incl.
      54/55 monitor for pooled scopes), transition and cold-start subsets; paired bootstrap
      vs champion; factorial read-out (dynamics / scope / observation main effects).
- [ ] Wave 2 (spec agreed 2026-09-27, see the decision block in the Work Log and
      `experiments/scotland/04_pyramid_tier_transition_arms/README.md`): 15 cells on the
      23/24–25/26 panel. Pyramid tier component (anchor, carry κ, jump η, B1-literal mode),
      league deltas on intercept and log κ (replacing A1), a `MultiScaleGRW` target-season macro
      step. Bases `td_spfl_cups_joint` and `grw_spfl_cups_joint` with `td_lower_joint` and
      `m12_td` re-run; judged on the two-gate rule. `PlayerLineupPillar` + wealth (and B2) move to W4.
- [ ] Finalist stage (T−25 calibrated portfolio) for the top candidates vs `m12_td`.
- [ ] 2026-09-19 slate re-priced (Cove v Ross County, Hamilton v QoS) from the finalists
      with a fold trained to the card (see the card-injection procedure).
- [ ] Findings README under `experiments/scotland/NN_.../` and rows on the leaderboard;
      TODO 028 closed as superseded with a pointer here.

## Ideas & Candidate Solutions

- About 18–22 h of beast time for wave 1 at the observed 80–110 min per pooled arm; queue
  overnight in two batches.
- Existing runs (`m00_baseline_grw` f64a00a2, g1 f00ec78a, g2 a6f62436, g3 9babf9e9) are
  cross-checks for the harness re-runs, not substitutes for them.

## Work Log & Progress

- [2026-09-26 @claude] Scoped with the human; blocked on TODO 030.
- [2026-09-27 @claude] **Wave 1 run.** With the human's agreement, 6 cells reuse existing runs (g1/g2/g3,
  s12 m01/m02, multiscale GRW m00; control `td_lower_joint` = s12_m02 `97c7a3d9`) and 6 are new.
  Smokes from `ade6ee9a`: 6/6 pass all 5 hard checks. Grids from `5307d039` (branch `run/w1-grid`;
  the only change is CONTROL → UUID): 6/6 pass the hard gates and the convergence review
  (R̂ ≤ 1.0124, divergences 0–16/160k), 5–94 min per cell. `grw_spfl_joint` was saved at stride 2
  after Postgres dropped the stride-1 write (1 GB field limit). In-process grid scoring failed in
  every cell (`UndefVarError: DecoupledGenerativeXG` loading the s12 control) *after* save, so no
  data was lost. All 12 were scored with `scripts/score_runs.jl` (`W1_runs.csv`, 12/12;
  `W1_vs_m12td_runs.csv`, 7/7).
- [2026-09-27 @claude] **W1 result: no promotion.** No cell meets the non-inferiority margin
  (best `td_spfl_cups_joint` −0.00084 [−0.00441, +0.00277] vs control; −0.00046 vs m12_td; it is
  top of the leaderboard at 0.64291 with ECE 0.0080). Joint beats Poisson in 6/6 pairs. GRW
  compression is 1.16 vs TD 1.92. Pooling worsens the promoted/relegated bias and cups partly
  repair it. The W2 base is a human decision; the README recommends both `*_spfl_cups_joint`
  cells, with g3 re-run under the harness recipe. Full tables:
  `experiments/scotland/03_dynamics_scope_matrix/README.md`.
- [2026-09-27 @claude] Harness follow-ups (from TODO 030 acceptance and W1):
  (a) `run_candidates.jl` grid loop has no try/catch;
  (b) `grid()` resolves a string control by name, so use a UUID;
  (c) `Harness.grid()` includes only the pyramid prototype loader; reuse `score_runs.jl`'s loader table;
  (d) the MAP screen *inflates* non-centred scales; fix the "collapse" wording in LEADERBOARD and
      consider a fixed-scale MAP screen;
  (e) `harness_checks.at` is CEST labelled UTC;
  (f) two `harness_clean_smoke` throwaway runs await a human decision;
  (g) `transition_bias_pp` has no bootstrap CI, which the promotion rule needs;
  (h) per-candidate `--only` invocations overwrite the `harness_experiments` row (run_ids = the last
      candidate only, metadata from that commit); corrected by hand for W1;
  (i) `score_runs.jl` uses the `ScottishLower` datastore, so the 54/55 monitor subsets are never scored.
- [2026-09-27 @claude] W2 blocker → [TODO 034](034_zero_allocation_compiled_gradients_in_the_builder_engine.md)
  (branch `fix/zero-alloc-engine`): the builder engine now replays compiled tapes at 0 B/gradient
  (was 423 KB for `td_base`); 16-thread throughput 2.97× `td_base`, 1.80× `grw_step_a2_carry_jump_ldelta`;
  recipe hashes unchanged. Smoke adds a hard `tape_allocation` check (now required by
  `has_passing_smoke`) and a review `sampling_performance` probe; `td_base` and
  `grw_step_a2_carry_jump_ldelta` re-smoked and pass all hard checks. Report:
  `docs/architecture/zero_alloc_engine_report.md`.
- [2026-09-27 @claude] **Decision — Wave 2 spec (grill-me session with the human).** Contract:
  `experiments/scotland/04_pyramid_tier_transition_arms/README.md`. Agreed points:
  1. **Panel:** target 23/24–25/26 (frozen). W1's panel held only 3 relegated / 4 promoted / 1 entered
     club; the new one holds 5 / 6 / 2. 26/27 to date is the W2.5 live check. Every base is re-run.
  2. **Bases:** `td_spfl_cups_joint` and `grw_spfl_cups_joint`, each the paired control for its own
     arms. `td_lower_joint` is re-run as the lower-scope control, and `m12_td` as a non-gating reference.
  3. **A2 = one tier component.**
     - Tier sets the shrinkage target: anchor = first tier in the fold window (full-pyramid tier table
       in the lower scope).
     - Carry κ of the tier change: κ ≡ 0 or κ ~ Beta(1.5, 4). The current-season additive form (κ = 1)
       is rejected, because it contradicts TODO 029's +0.28.
     - One step s ~ TN(0.47, 0.16); fixed 0.48/0.52 α/β split; Old Firm pseudo-tier o ~ TN(1.15, 0.10).
  4. **`MultiScaleGRW` gap:** the target season is entered by a micro (σₖ) step only; there is no
     macro step at the summer boundary. Fix = opt-in `target_season_step`, run as its own arm.
  5. **Jump:** per-change η ~ N(0, σ_j²), σ_j ~ Gamma(2, 0.08), persisting after the change, on α
     and β.
  6. **League δ** (the human's proposal): zero-sum deltas on the intercept and log κ, fixed
     N(0, 0.10); log κ amended after step 0, see below. Replaces A1. Dropped if the W1 residual goals/(κμ) by league is within ±3%.
  7. **B1:** no pooled B1. Lower scope gets A2 (full-pyramid anchor) + carry + jump, plus a B1-literal
     arm (TODO 029 table incl. promoted). B2 is deferred to W4.
  8. **Cells:** 15 (full ladder on TD, combinations only on GRW).
  9. **Decision rule:**
     - Gate 1: ΔLL vs own base, CI upper < +0.002.
     - Gate 2: club-season-clustered bias CI; |bias| falls on both relegated and promoted into L1
       (first 20), and the Δ|bias| CI excludes 0 on at least one.
  10. **Harness Phase B**, built while the grids run: (g), scorecard v1.2 + panel labels, (i), (h).
      No MAP screen.
  11. **Build Phase A:** pi Sol, straight into `src/` as opt-in components, on branch
      `feat/w2-tier-components`, with a 7-part test bar.
  12. **Sampler:** as W1; one re-run at acceptance 0.80 if a cell fails convergence review.
  13. **Pre-authorised launches:** the TD batch auto-launches after 5/5 smokes; the GRW batch
      launches overnight after the TD read-out, with cells 9 and 11 cut if A2 is inert on TD.
  14. **Order after W2:** W2.5 slate re-price → W3 (conditional) → W4 lineup + wealth (+B2) → W5
      T−25 portfolio.
  15. **Housekeeping:** close 028; new TODOs for the fixed-scale MAP screen and the T55 Betfair
      backfill; delete the `harness_clean_smoke` runs after showing the rows; open the harness PR
      after #43 merges.
- [2026-09-27 @claude] **Step 0 (league-δ gate), and an amendment agreed with the human.**
  - Measured goals ÷ pxG by league on 23/24–25/26: 54 0.849, 55 1.149, 56 1.231, 57 1.237
    (pooled 1.080; ±4% CIs). The gate passes, so the league-δ cells stay.
  - Cause: pxG measurement. Premiership commentary logs 25.4 attempts per match (7.7 blocked)
    against 18–19 (2.8–3.9 blocked) below it; SofaScore shows BBC under-recording blocks in T55.
  - **Amendment:** δ on log κ is keyed by the **match's competition** (54/55/56/57/cups, shared by
    both sides), prior N(0, 0.25). This replaces "attacking side's tier, N(0, 0.10)", which would
    bias the tier step through cup ties. The intercept δ is unchanged.
  - W1 pooled joint cells carried this mismatch.
- [2026-09-27 @pi] **Wave 2 Phase A complete.** Built the opt-in target-season GRW
  boundary step, pyramid anchor/carry/jump/B1 components, competition-aware joint
  deltas, and all 15 declarative W2 cells on `feat/w2-tier-components`. Laptop
  verification: focused suites green and canonical `test/runtests.jl` 4429/4429;
  the parallel runner remains 23/24 only because of known T007. Beast SHA `248ec796`
  compiled finite ReverseDiff gradients and exercised OOS extraction/reload parity
  for all 15 candidates on fold 1 and first-24/25 (30/30); no smoke, grid or posterior
  sampling was run. W1 hashes are unchanged and a persisted W1 GRW fit loads under the
  compatibility-preserving layout. Evidence and telemetry:
  `docs/architecture/w2_phaseA_report.md`.
- [2026-09-27 @claude] **W2 Phase A accepted** at `c50db345` (pi, `feat/w2-tier-components`).
  - **Manager rerun on a clean beast checkout:** pyramid tiers 90/90, GRW 124/124, builder 108/108,
    splitter 48/48, harness runner 55/55, scoring 29/29, portfolio 25/25.
  - **Recipe hashes:** 12/12 W1 hashes are identical to `harness_checks`; the 15 W2 hashes are unique
    and new.
  - **Instantiation:** all 15 cells build on 2 folds each; gradients 0.07–0.37 ms.
  - **Human decision:** season-opening GRW folds integrate the summer step in the forecast.
  - **Findings:**
    - (a) The forecast boundary innovation is not centred across teams. This is Phase B item 0 and
      must land before the GRW grids.
    - (b) The lower-scope full-pyramid arm samples prior-only events and an Old Firm offset that no
      lower-scope match informs. Harmless; kept.

- [2026-09-27 @pi] **Wave 2 Phase B scoring implementation and W1 validation.**
  Centred the season-opening GRW innovation (`8effca72`); scorecard v1.2 now
  carries a panel and clustered transition-bias/paired absolute-bias intervals,
  monitor scoring, per-row controls, and atomic per-run register merging. W1
  re-score: 12/12, 8,388 v1.2 rows, all 7,848 common old/new metric rows
  numerically identical (max absolute difference 0 for every metric); 120 bias
  and 120 paired absolute-bias rows have CIs. Of eight pooled W1 cells, only
  five persisted monitor latents; the three reused pre-harness GRW artifacts
  have no monitor rows to score. See `docs/architecture/w2_phaseB_report.md`.
- [2026-09-27 @claude] **W2 Phase B accepted** at `fdc472ec` (pi, Sol 6.0).
  - **Manager rerun on a clean beast checkout:** harness scoring 43/43, runner 65/65, portfolio
    27/27, pyramid tiers 94/94, splitter 48/48.
  - **DB check:** 7,848 shared v1.1/v1.2 W1 rows are identical (max |Δ| 0.0, no NaN mismatch); every
    `harness_scores` row carries a panel (W1 710; finalists 632/611); the club-season-clustered
    bias CIs are present; the W1 register row lists 12/12 runs.
  - **Limitation accepted (manager decision):** the reused pre-harness W1 GRW runs g1
    `f00ec78a`, g2 `a6f62436` and g3 `9babf9e9` have no persisted 54/55 latents, so they have no
    monitor rows. They are not refitted, and W2's fresh grids persist monitor latents.
  - **Incident, same day:**
    - The W2 TD grid ran at 40–50% sampler CPU. Cause: the builder engine allocated 432 KB per
      compiled gradient (`clamp` guard + a fused Gamma-arm broadcast with tracked scalars →
      ReverseDiff `tracker_∇broadcast`), a GC storm at 16 threads.
    - It had been fixed before, but only in prototypes (`l10`/`l11`).
    - With the human, all W2 runs and rows were deleted (17 runs, 1,476 score rows, 139 checks) and
      the grids await `fix/zero-alloc-engine` (Claude CLI, TODO 034).
    - The grid-panel coverage fix is on `run/w2-grid` `1a8f4a03`; Phase B carries the general
      version.
- [2026-09-27 @claude] **Harness follow-up (j): an unchanged recipe cannot be re-smoked.**
  - Cause: `Training.save_fit` deduplicates on `configs.config_hash`, which excludes `time:` tags,
    so it returns the *existing* run's ID without saving the new fit. The smoke's `fit_parity` then
    compares the new in-memory fit with the old reloaded one and fails ("FitConfig changed on
    reload", the `time:` tag differs). A failed hard smoke record then permanently blocks that
    recipe's grid.
  - Hit on the first post-merge `td_base` smoke at `a76a65df`, because the zero-alloc agent had
    already smoked it for evidence at `6f146ae2`.
  - Resolved for W2 by deleting the two evidence smoke runs and all 33 W2 check rows, then
    re-smoking every recipe once.
  - Proper fix, to be scheduled: smoke parity must detect a pre-existing `configs` row (skip or
    compare against the stored fit explicitly), or smokes must carry a per-run nonce tag.

## Verification & Findings

- Wave 1 (2026-09-27): 12/12 cells scored (scorecard v1.1, 710 target fixtures). Results,
  main effects and caveats: `experiments/scotland/03_dynamics_scope_matrix/README.md`; rows in
  `experiments/LEADERBOARD.md`; register row `scottish_dynamics_scope_matrix_w1` = completed.
