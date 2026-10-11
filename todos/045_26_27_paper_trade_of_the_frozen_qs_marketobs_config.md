# 045 — 26/27 paper trade of the frozen qs_marketobs scenario-Kelly config

| Field | Value |
|---|---|
| ID | 045 |
| Title | 26/27 paper trade of the frozen qs_marketobs scenario-Kelly config |
| Status | COMPLETED |
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

- [x] Gates pass: market-rate inversion reproduction (ruling 3), engine reproduction of the wave-5
      frozen 25/26 cell (ruling 9), m12 fold inventory and filtration (ruling 4), qs extension
      positions 1–40 reproduce the persisted held-out sets; DB audit unchanged around every runner.
- [x] No sampled fold trains on a fixture on or after its first held-out day; filtration CSV committed.
- [x] Per-fixture held-out probabilities (qs, m12) and 1X2 log loss vs the mid close, split
      transition clubs vs rest.
- [x] The 9 cells (frozen / fallback / m12 × back / mid / LTP), fill check, drawdown watch,
      `PAPER_LEDGER.csv` and `REPORT.md` committed with traceable numbers.
- [x] Follow-ups recorded (ingestion gap after 09-19; src/MatchDay integration).

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
- [2026-10-11 @claude] Manager accepted Phase 1 (manager now pane %113). Phase 2 done: `l03_paper.jl` + `r04_phase2.jl`
  (`d8544a0a`, `d61c4dfc`), engine gate re-run bit-identical, 9 cells, fill check, drawdown watch, `PAPER_LEDGER.csv`,
  `REPORT.md`. DB audit unchanged. P27_DONE.

## Verification & Findings

- Gates: inversion PASS (Δ = 0 on 1,430 rows); engine PASS bit-identical in Phase 0 and again at the Phase 2 head
  (1.6241209548752367 / 494); positions 1–40 equal (qs, m12); sampled filtration PASS on 8 folds; DB audit
  unchanged around every runner. See `experiments/scotland/10_qs_marketobs_2627_paper/REPORT.md`.
- Coverage 79 / 70 / 60 at T−25m; 8 match days with a book (7 for LTP: none on 08-01); no book on 08-22 or 08-29.
- Predictions: qs R̂ ≤ 1.0115, m12 R̂ ≤ 1.0084, 0 divergences. 1X2 log loss − mid close: qs +0.003 [−0.012, +0.017],
  m12 +0.034 [+0.008, +0.062]; both worse on transition clubs (qs +0.030, m12 +0.083).
- Frozen B1 λ 12 at the back price: wealth 0.9639 [0.704, 1.319], g −4.6×10⁻³ [−43.8, +34.6], MDD −10.56 %
  (watch −20.39 % not breached), 91 bets, CLV −0.69 pp vs the mid close. Fallback 0.9834 (MDD −5.69 %);
  m12 canonical 0.8670 (MDD −14.95 %). Mid: 0.968 / 0.984 / 0.879. LTP: 0.988 / 0.998 / 0.865.
- Fills at £1,000: 29.7 % of frozen bets exceed the best-level size, 5.5 % exceed 3 levels; 54 % of the money
  fills at the best level. At £200: 6.6 % / 0 %.
- 8 match days cannot rank anything; every interval spans zero and contains wave 5's 25/26 g (9.70×10⁻³).

## Follow-ups

- **Ingestion gap after 09-19 and the src/MatchDay integration** (new TODO): betfair_live order book and results
  stop at 2026-09-19 (and rounds 08-22 / 08-29 have no book). Restore ingestion, then build the live path: a weekly
  order-book market-rate inversion feed, live qs_marketobs fold extension per round, the ScenarioKelly day-problem
  adapter and B1 leg filter in src, SlateDrawdown bookkeeping, and stakes at the executable back price.
- Continue the paper trade on later rounds with this replay (same frozen config, back price primary); re-read the
  drawdown watch (−20.39 %) and the fill check as the bankroll path lengthens.
- Transition clubs: qs is worse than the close on them in 26/27 (and the frozen bets on them lost). A review of
  the cold-start / tier-change handling is a candidate for the next re-selection, not a config change now.
- The backtests never paid the spread: re-score wave-5 cells at an executable price when a historical order book
  exists, or carry the ≈ −0.7 pp back-price CLV cost into the next selection.
- m12's held-out XI is the played XI (confirmed lineups); a forecast-XI variant would remove that caveat.
