# 041 — Wave 2: QS on joint goals+xG and GRW/QS with market-rate observation

| Field | Value |
|---|---|
| ID | 041 |
| Title | Wave 2: QS on joint goals+xG and GRW/QS with market-rate observation |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-09 |
| Updated | 2026-10-09 |
| Related Files / Commits / PRs | experiments/pi_qs_joint_marketobs_prompt.md; experiments/scotland/06_qs_joint_and_market_observation/; TODO 040 |

## Context & Problem Statement

Compare QS versus MultiScaleGRW on joint goals + proxy-xG, and both dynamics observing inverted market rates versus saved wave-1 C0. Same pinned snapshot, 40-fold lower panel and scoring; no ROI/staking. Claude manages; human scope fixed 2026-10-09. Work package is authoritative.

## Acceptance Criteria

- [ ] Phase 0 fresh beast branch-health tests and per-fold proxy-xG coverage recorded.
- [ ] MarketRateObservation wiring, masked likelihood, real-fold AD, 0 B tapes and synthetic recovery evidenced.
- [ ] Four W2 candidates pass smoke hard gates; manager approval before grid.
- [ ] Ordered 40-fold grids with checkpoints, prescribed convergence reruns and diagnostics.
- [ ] Report/CSV/leaderboard with all prescribed pairs, bootstraps and posterior parameters; saved wave-1 references unchanged.
- [ ] Fresh-REPL frozen-seed fold reproduction for each arm, byte comparison recorded.
- [ ] Phase-boundary commits/pushes, progress/decisions current; todo check and diff check pass.

## Ideas & Candidate Solutions

- Reuse wave-1 panel, splitter and scorer without refitting references. Audit proxy-xG through MatchProxyXGFeature with k=25, fallback=:none and fold-specific history fit IDs.
- Inject a stream-independent rates feature; src has no prototype dependency. Use binary-masked Normal log-rate observations and existing Poisson prediction seam.
- Derive sigma prior from C0 before implementation; freeze priors before scoring. No threshold relaxation or score-driven tuning.

## Work Log & Progress

- [2026-10-09 @pi] Claimed in session 01a12134-5a93-7644-aa40-b5294e798206, worktree /home/james/bet_project/.worktrees/BayesianFootball-qs-experiment, branch exp/qs-goal-vs-market. Phase 0 started; fresh beast tests and coverage next. Read-only betdb; Julia exclusively beast.

## Verification & Findings

- Initial branch clean at 9a86f7b1. Beast detached checkout clean at d16a5800; .env and Manifest symlinks intact.
- Beast snapshot SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4; original mtime 2026-09-25 12:57:15.480765468 UTC intact.
- Phase 0 gates not yet run. Evidence will live under experiments/scotland/06_qs_joint_and_market_observation/logs/.
