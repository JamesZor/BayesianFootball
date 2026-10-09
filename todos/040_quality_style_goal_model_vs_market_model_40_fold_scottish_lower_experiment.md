# 040 — Quality/style goal model vs market model: 40-fold Scottish Lower experiment

| Field | Value |
|---|---|
| ID | 040 |
| Title | Quality/style goal model vs market model: 40-fold Scottish Lower experiment |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-09 |
| Updated | 2026-10-09 |
| Related Files / Commits / PRs | — |

## Context & Problem Statement

Execute `experiments/pi_qs_goal_vs_market_prompt.md` phases 0–6: validate the merged base, implement and verify QualityStyleGRW, compare 40-fold Scottish Lower goal and C0 market arms on a pinned snapshot without staking or ROI. No grid before passing smoke and manager approval.

## Acceptance Criteria

- [ ] Phase 0 merged-base tests pass on mcmc-beast.
- [ ] QualityStyleGRW builder, OOS, reconstruction, AD parity and zero-allocation tests pass.
- [ ] Market arm and score adapter validated on matching 40-fold fixtures.
- [ ] All four goal arms and market arm pass smoke; manager approves full grid.
- [ ] 40-fold grid, paired-bootstrap scoring, report and leaderboard completed without ROI.
- [ ] Fresh-REPL frozen-seed reproduction recorded with byte-for-byte comparison.

## Ideas & Candidate Solutions

Follow the phase gates, fixed priors and seeds in the work package. Record decisions in `experiments/scotland/05_quality_style_goal_vs_market/DECISIONS_LOG.md`; operational state in `PROGRESS.md`.

## Work Log & Progress

- [2026-10-09 @pi] Claimed in pi worktree `/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`; beginning Phase 0 branch-health gate on mcmc-beast. No experiment scores yet.

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.
