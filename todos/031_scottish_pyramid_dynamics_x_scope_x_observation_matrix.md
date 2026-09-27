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
- [ ] Wave 2 on the best wave-1 cell: TODO 028's A1 league offsets / A2 tier steps, and a
      B1 relegated prior at the TODO 029 estimate (α0 ~ N(+0.12, 0.20²), β0 ~ N(−0.16, 0.26²),
      not +0.90); then `PlayerLineupPillar` + wealth.
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

## Verification & Findings

- Wave 1 (2026-09-27): 12/12 cells scored (scorecard v1.1, 710 target fixtures). Results,
  main effects and caveats: `experiments/scotland/03_dynamics_scope_matrix/README.md`; rows in
  `experiments/LEADERBOARD.md`; register row `scottish_dynamics_scope_matrix_w1` = completed.
