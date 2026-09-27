# 033 — Backfill Championship Betfair archive for the T55 monitor

| Field | Value |
|---|---|
| ID | 033 |
| Title | Backfill Championship Betfair archive for the T55 monitor |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-27 |
| Updated | 2026-09-27 |
| Related Files / Commits / PRs | TODO 031 (54/55 monitor scoring, follow-up i), betdb `betfair.match_meta` |

## Context & Problem Statement

TODO 031 W2 scores the 54/55 monitor subsets for pooled scopes. betdb has a verified Betfair
match for only **479 of 925** Championship (T55) fixtures in 21/22–26/27 (Premiership 956/1032;
TODO 029 table). Most of the T55 monitor falls back to missing or reduced markets, so per-tier
Championship scores are thin and possibly biased by which matches were archived.

## Acceptance Criteria

- [ ] Audit why T55 matches fail Betfair matching (`betfair.match_meta.status` / `error_type`).
- [ ] Backfill the resolvable ones; report T55 verified coverage per season before and after.
- [ ] Refresh the `ScottishAll` datastore cache and re-score the TODO 031 monitor rows if coverage rises materially.

## Ideas & Candidate Solutions

- Check the name normalisation and search strategy for Championship clubs first (`search_strategy_used`, `normalized_name`).

## Work Log & Progress

- [2026-09-27 @claude] Created from TODO 031 follow-ups, agreed with the human in the W2 grill-me session.

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.
