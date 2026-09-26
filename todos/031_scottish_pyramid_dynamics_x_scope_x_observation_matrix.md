# 031 — Scottish pyramid dynamics x scope x observation matrix

| Field | Value |
|---|---|
| ID | 031 |
| Title | Scottish pyramid dynamics x scope x observation matrix |
| Status | BLOCKED |
| Priority | P1 |
| Assignee | unassigned |
| Created | 2026-09-26 |
| Updated | 2026-09-26 |
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

- [ ] Blocked until TODO 030's harness passes its g1/g2/g3 validation.
- [ ] Twelve recipes: dynamics {`TimeDecayDynamics(180)`, `MultiScaleGRW`} × scope
      {lower 56/57, SPFL 54–57, SPFL + senior cup ties} × observation {Poisson,
      `JointGammaPoissonObservation`}; no wealth or lineup terms in wave 1.
- [ ] Smoke then 40-fold grid for all twelve on `mcmc-beast`; gate reports recorded; any
      `review` flag triaged in the Work Log.
- [ ] Grid scorecard (`scotland_default_v1`) for all twelve: target, per-tier (incl.
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

## Verification & Findings

Not run yet.
