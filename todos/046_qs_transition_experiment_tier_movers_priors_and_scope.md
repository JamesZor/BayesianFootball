# 046 — QS transition experiment: tier movers, market-informed priors and scope

| Field | Value |
|---|---|
| ID | 046 |
| Title | QS transition experiment: tier movers, market-informed priors and scope |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-10-11 |
| Updated | 2026-10-12 |
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
- [2026-10-12 @pi] S2 blocked under manager D12: beast Tailscale SSH still unavailable at deadline. Pre-src legacy byte identity, W2 panel/input parity and control/QS+A2 normal smoke hard gates passed; convergence review failures remain. Opt-in entrant/summer source at `e2e956cf` is ungated. No full grids. Continuation: `experiments/scotland/11_qs_transition/build/HANDOVER.md`.

- [2026-10-12 @pi] S2 resumed after beast recovery: fresh owned REPL, synced `ba33e752` including source `e2e956cf`. New/legacy units 419/419, synthetic and real-fold exact AD/FD/0-byte tapes, post-src chain/latent bit identity, endpoint overlap0 and fold1→2 prior persistence PASS. Three-seed recovery running; entrant/summer normal smokes next. F6 deferred; no grids.

- [2026-10-12 @pi] S2 build complete (F6 explicitly deferred for James): all src gates PASS, all four normal two-fold latest smoke hard checks PASS. Recovery86/90 intervals, no all-three misses; post-src legacy byte identity preserved. REPORT_S2 includes UUIDs, Rhat/ESS/divergences, gate commands and22.37h four-arm timing projection. Convergence/performance review failures retained. Read-only S3 audit archives sole stale control failed check832; manager must approve targeted cleanup before control grid. No DB cleanup, grids, pushes or merges; owned s2 stopped.

## Verification & Findings

S2 verified: `experiments/scotland/11_qs_transition/REPORT_S2.md`, `build/gate_commands.csv` and evidence CSVs/logs. All source gates and four-arm hard smokes pass; F6 deferred by resume scope. S3 readiness still needs manager review and authorised narrow cleanup of stale control check832; no cleanup performed. S3–S5 acceptance criteria remain open; task stays IN_PROGRESS.
