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
- [2026-10-10 @pi] STOP at Phase 0 read-only-session assertion on corrected startup (`9249fbcd`). No fit loaded or numerical backtest computed. Startup world-age failure preserved separately. Dependency: Claude authorises verified read-only connection setup and resolves canonical/Option B/historical policy discrepancy. Owned beast pane `%451` closed; reports/logs handed over. No acceptance criterion closed.

- [2026-10-10 @pi] Manager authorised Phase 0 resume: static read-path verification plus before/after table-wide DB audit replaces session-default assertion; actual SHOW value logged once, no PGOPTIONS debugging. Canonical alone targets 1127 bets / 2.5152 ±0.00005. Both canonical and option_b_live inventories reported; raw probabilities, no fitted calibration; plain_kelly 0.25 retained. Updated loader to audit even failure exits and to deserialize cache with no SQL fallback. Static gate passes, four injected negative fixtures rejected.

## Verification & Findings

- Beast `julia --project -t 16 -i`, pinned cores, BLAS 1; include `r00_phase0_inventory.jl` in owned pane `%451`.
- Attempt 1 `8c4fccec`: startup MethodError/world-age warning, no gate reached; corrected module include once.
- Attempt 2 `9249fbcd`: FAIL `assert_readonly` (`SHOW default_transaction_read_only`), before m12 load_fit. No parity/coverage/wealth/CLV or bootstrap evidence.
- Historical target 2.5152 unit wealth / 1127 bets is committed experiment 07 evidence, NOT a fresh reproduction.
- No fits, source allocator changes, or DB writes. m12 UUID remains `132df5c2-c742-4e95-8693-3aeb2b2cbaef` (not loaded).
- Evidence and resume requirements: `experiments/scotland/08_qs_marketobs_backtest/{REPORT,HANDOVER,PROGRESS,DECISIONS_LOG}.md`, committed attempt logs and execution-status CSV.
- `scripts/todo.sh check` and `git diff --check` passed before the source commits; final checks recorded in handoff commit. Owned session closed and verified absent.
