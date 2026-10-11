# 045 — 26/27 paper trade of the frozen qs_marketobs scenario-Kelly config

| Field | Value |
|---|---|
| ID | 045 |
| Title | 26/27 paper trade of the frozen qs_marketobs scenario-Kelly config |
| Status | IN_PROGRESS |
| Priority | P2 |
| Assignee | claude |
| Created | 2026-10-11 |
| Updated | 2026-10-11 |
| Related Files / Commits / PRs | Brief `experiments/claude_qs_2627_paper_brief.md` (`57cf058e`); folder `experiments/scotland/10_qs_marketobs_2627_paper/`; frozen config `experiments/scotland/09_qs_marketobs_wave5_config/FREEZE.md` (`04fdba71`); TODO 044 |

## Context & Problem Statement

Wave 5 (TODO 044) froze a paper-trading config: qs_marketobs (QS weak-r + `MarketRateObservation`)
× `ScenarioKelly` on basket B1 (1X2 H/D/A + U2.5), TieredTrust, `SlateDrawdown(12)`, at T−25m.
It has only been scored on 24/25–25/26 backtest prices. This task replays that config, without
tuning, on the 79 finished 26/27 League One/Two matches (2026-08-01 → 2026-09-19): walk-forward
qs_marketobs predictions without look-ahead (in-memory fold extension), 26/27 market-rate
observations from the `betfair_live` order book, and a paper trade at the executable best back
price beside the fallback (λ 23) and m12 canonical. No DB writes; src frozen.

## Acceptance Criteria

- [ ] Gates pass: market-rate inversion reproduction (ruling 3), engine reproduction of the wave-5
      frozen 25/26 cell (ruling 9), m12 fold inventory and filtration (ruling 4), qs extension
      positions 1–40 reproduce the persisted held-out sets; DB audit unchanged around every runner.
- [ ] No sampled fold trains on a fixture on or after its first held-out day; filtration CSV committed.
- [ ] Per-fixture held-out probabilities (qs, m12) and 1X2 log loss vs the mid close, split
      transition clubs vs rest.
- [ ] The 9 cells (frozen / fallback / m12 × back / mid / LTP), fill check, drawdown watch,
      `PAPER_LEDGER.csv` and `REPORT.md` committed with traceable numbers.
- [ ] Follow-ups recorded (ingestion gap after 09-19; src/MatchDay integration).

## Ideas & Candidate Solutions

- Reuse wave-4/5 loaders by `include`; copy only what must change (logged in the folder's DECISIONS_LOG).
- Order-book handling from the CLV napkin (`proto/clv-napkin` @ `1c215df4`, `l04_live_orderbook.jl`).

## Work Log & Progress

- [2026-10-11 @claude] Claimed (Claude CLI implementer, worktree `.worktrees/BayesianFootball-qs-experiment`,
  branch `exp/qs-goal-vs-market`; manager pane %114). Phase 0 started.
- [2026-10-11 @claude] Phase 0 done (runner `aace7998`, `results/phase0/`, `logs/phase0.log`). Coverage 79/70/60
  at T−25m. Inversion gate PASS (Δ = 0). Engine gate PASS bit-identical. qs plan: positions 41–44, 1–40 equal,
  filtration PASS. m12 DB folds miss the 09-05 round, so an in-memory m12 extension is proposed. DB audit
  unchanged. P27_HANDOVER; waiting for approval of the sampling jobs.
- [2026-10-11 @claude] Manager approved both jobs and D12. Phase 1 done: qs and m12 in-memory positions 41–44
  (R̂ ≤ 1.0115, 0 divergences, no reruns), sampled filtration PASS, m12 played-XI rule documented (D17),
  m12 DB check mean |Δp| 0.0004, log loss vs mid close in `results/phase1/`. P27_HANDOVER.

## Verification & Findings

Not run yet.
