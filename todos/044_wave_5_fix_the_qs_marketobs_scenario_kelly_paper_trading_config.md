# 044 — Wave 5: fix the qs_marketobs + scenario Kelly paper-trading config for 26/27

| Field | Value |
|---|---|
| ID | 044 |
| Title | Wave 5: fix the qs_marketobs + scenario Kelly paper-trading config for 26/27 |
| Status | IN_PROGRESS |
| Priority | P2 |
| Assignee | claude |
| Created | 2026-10-10 |
| Updated | 2026-10-10 |
| Related Files / Commits / PRs | `experiments/claude_qs_wave5_config_brief.md` (`5c868bc8`); `experiments/scotland/09_qs_marketobs_wave5_config/`; wave 4 = TODO 043 |

## Context & Problem Statement

Wave 4 (TODO 043) found scenario Kelly helps qs_marketobs at T−25m but left the basket and λ at
the canonical production values (λ 23, H/D/A + U2.5), which were chosen with m12. Wave 5 picks
the basket and λ on 24/25 only (λ matched to m12 canonical's 24/25 MDD), freezes them in
FREEZE.md, then scores the frozen config on 25/26 against m12 × canonical × λ 23 and writes the
paper-trading config for 26/27. No fits, no DB writes, src frozen. Julia on mcmc-beast only.

## Acceptance Criteria

- [ ] Phase 0: wave-5 runner reproduces bit-identical m12 × canonical × T−25m (2.773493294562897 / 1132) and qs × scenario_kelly × T−25m (2.154029744590488 / 952); per-season MDDs match to 0.01 pp; DB audit unchanged.
- [ ] Phase 1: per-direction table, λ curves for B1–B4 (+ explored baskets), ruling-4 sensitivity, FREEZE.md committed and pushed before any 25/26 wave-5 number exists.
- [ ] Phase 2: 25/26 (and 24/25) scoring of frozen, m12, fallback and every candidate; T−6h rows; ruling-8 decision; REPORT.md and PAPER_TRADING_CONFIG.md with traceable numbers; CSVs committed.
- [ ] DB audit unchanged around every runner; beast panes closed; everything pushed.

## Ideas & Candidate Solutions

- Baskets filter legs before the joint solve (ruling 4). The joint scenario matrix is built once per day over every canonical leg; a basket solves on a column subset (same draws, same scorelines), so baskets and λ values are cheap to sweep. λ only changes the overlay, so each basket is solved once and simulated at 12 λ.

## Work Log & Progress

- [2026-10-10 @claude] Claimed as the wave-5 Claude CLI implementer in worktree `.worktrees/BayesianFootball-qs-experiment`, branch `exp/qs-goal-vs-market`; manager is the launching Claude session. Phase 0 started.
- [2026-10-10 @claude] Phase 0 PASS (`21f3514f`, beast pane %458): both wave-4 T−25m cells bit-identical, per-season MDDs within 0.01 pp, 24/25-only runs equal the full runs' first season, basket subset equivalence holds, seal refuses 25/26, DB audit unchanged. Stopped with W5_HANDOVER.

## Verification & Findings

Not run yet.
