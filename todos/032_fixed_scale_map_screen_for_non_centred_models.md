# 032 — Fixed-scale MAP screen for non-centred models

| Field | Value |
|---|---|
| ID | 032 |
| Title | Fixed-scale MAP screen for non-centred models |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-27 |
| Updated | 2026-09-27 |
| Related Files / Commits / PRs | TODO 030 (harness `screen` stage), TODO 031 follow-up (d), `src/harness/stages.jl` |

## Context & Problem Statement

The harness `screen` stage (TODO 030) runs MAP as a cheap pre-NUTS ranking. On non-centred
models (`MultiScaleGRW`, `TimeDecayDynamics`) MAP **inflates** the learned scale parameters 5–50×
vs NUTS, not collapses them. Measured 2026-09-27 on fold 20, lower scope:
- GRW α.σₖ 0.93 vs 0.017;
- TimeDecay σ_a 0.63 vs 0.11.

The ratings come out over-dispersed, so the screen's slopes and LogLoss are invalid for any model
with learned scales. `experiments/LEADERBOARD.md`'s warning wrongly says "collapse". TODO 031 W2
runs without a screen stage.

## Acceptance Criteria

- [ ] Fix the LEADERBOARD/register warning wording ("inflate", not "collapse").
- [ ] A screen mode that fixes every scale parameter at reference posterior medians (from a named
      NUTS run) and optimises only the location/innovation parameters.
- [ ] On W1 cells, the fixed-scale screen ranks cells consistently with the NUTS grid (rank
      correlation reported), or the screen is marked `limited` for learned-scale models.

## Ideas & Candidate Solutions

- Fix scales at NUTS posterior medians from the control run of the same recipe family.
- Alternative: Laplace or short ADVI instead of MAP.

## Work Log & Progress

- [2026-09-27 @claude] Created from TODO 031 follow-ups, agreed with the human in the W2 grill-me session.

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.
