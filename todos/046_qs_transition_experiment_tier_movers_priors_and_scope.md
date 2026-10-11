# 046 — QS transition experiment: tier movers, market-informed priors and scope

| Field | Value |
|---|---|
| ID | 046 |
| Title | QS transition experiment: tier movers, market-informed priors and scope |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi (stages), claude (manager) |
| Created | 2026-10-11 |
| Updated | 2026-10-11 |
| Related Files / Commits / PRs | `experiments/scotland/11_qs_transition/DESIGN.md`; continues TODO 031 W2 (`04_pyramid_tier_transition_arms`); evidence from TODO 045 |

## Context & Problem Statement

QualityStyleGRW (qs_marketobs) misprices clubs that change tier. In the 26/27 paper trade (TODO 045),
the 37 bets on transition-club fixtures returned −39%, against +22% on the rest. W2's A2 tier term
fixed most of this bias for the TimeDecay and GRW models, but it was never applied to QS, and its
live check (W2.5) never ran. The aim is to diagnose the cause (cold start, tier scale, no summer step,
lag) and then test the fixes (a per-club market prior, transition shifts, A2 on QS, a summer step,
pooled scope with cups, weekly refits). See DESIGN.md.

## Acceptance Criteria

- [ ] S0 research report, EDA questions and prior forms.
- [ ] S1 EDA answers DESIGN §6 and proposes a prior form; James signs off.
- [ ] S2–S4 arms judged on W2's gate (transition bias plus log-loss non-inferiority) vs qs_marketobs.
- [ ] S5 backtest of the winner, and the one-time 26/27 test.

## Ideas & Candidate Solutions

See DESIGN.md §3 (F1–F7).

## Work Log & Progress

- [2026-10-11 @claude] Scope agreed with James over three AskUserQuestion rounds. DESIGN.md written.
  The S0 research brief (gpt-6-astra) is drafted.
