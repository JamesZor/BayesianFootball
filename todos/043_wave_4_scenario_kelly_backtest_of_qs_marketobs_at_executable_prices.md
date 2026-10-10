# 043 — Wave 4: scenario-Kelly backtest of qs_marketobs at executable prices

| Field | Value |
|---|---|
| ID | 043 |
| Title | Wave 4: scenario-Kelly backtest of qs_marketobs at executable prices |
| Status | IN_PROGRESS |
| Priority | P2 |
| Assignee | pi |
| Created | 2026-10-10 |
| Updated | 2026-10-10 |
| Related Files / Commits / PRs | `experiments/pi_qs_marketobs_backtest_prompt.md`; `experiments/scotland/08_qs_marketobs_backtest/` |

## Context & Problem Statement

Backtest five saved models against T−25m/T−6h/T−24h Betfair archive prices after commission. Claude manages the package. Work phase by phase, stopping at every boundary or failed gate. No new fits; databases read-only; Julia only on mcmc-beast.

## Acceptance Criteria

- [ ] Phase 0: exact production inventory, m12 experiment 07 wealth/bet parity, three snapshot coverage reports.
- [ ] Phase 1 allocator audit with source line references; Phase 2 scenario allocator tested or unnecessary.
- [ ] All five-model × snapshot × policy cells computed or NOT_AVAILABLE with reasons, block-bootstrap intervals and paired m12 differences.
- [ ] Reports answer all five brief questions with numbers traceable to committed CSVs.
- [ ] Follow-ups recorded, branch pushed, owned beast panes closed, PHASEBT_DONE.

## Ideas & Candidate Solutions

Freeze policy before viewing results. Distinguish canonical MatchDay defaults from the newer Option B stack and experiment 07's historical book settings. Do not force parity by tuning.

## Work Log & Progress

- [2026-10-10 @pi] Claimed Phase 0 in `BayesianFootball-qs-experiment`, branch `exp/qs-goal-vs-market`. Recorded exact canonical defaults and experiment 07 historical target before running any reproduction; Option B is a distinct configuration. Next: test parity on beast and stop at failure or boundary.

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.
