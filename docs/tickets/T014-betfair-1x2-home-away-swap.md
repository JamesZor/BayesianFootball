# T014 — Betfair 1X2 book has home/away swapped on at least one Scottish Lower fixture; no QA invariant compares it with the sofascore book

| Field | Value |
|---|---|
| Severity | low |
| Area | `ds.betfair_odds` (betfair → sofascore match mapping), `Data.summarize_odds` consumers |
| Status | open |
| Raised | 2026-09-22 |

## Evidence

Found during TODO 023 (`current_development/market_inverse_dynamics/`). The
de-vigged Betfair (−20, 0] TWA close (`closing_book`, identical to
`fast_slow_grw/l01::fsg_closing_book`) was compared, fixture by fixture, with the
de-vigged sofascore close `ds.odds.prob_fair_close` over the 595 Scottish Lower
24/25 + 25/26 fixtures that have both 1X2 books. Median |Δp_home| + |Δp_away| is
0.039. One fixture matches far better with the sides exchanged:

| match_id | date | home | away | Betfair p_home / p_away | sofascore p_home / p_away | direct Δ | swapped Δ |
|---|---|---|---|---|---|---|---|
| 14035501 | 2025-08-09 | montrose | kelty-hearts-fc | 0.555 / 0.199 | 0.217 / 0.507 | 0.647 | 0.066 |

(`east-fife v peterhead`, 14035642, is flagged by the same rule but with direct
0.157 vs swapped 0.127 — a disagreement, not a clean swap.)

The inverted close for 14035501 is λ_mkt = (1.72, 0.93); every model trained or
evaluated against the Betfair close since TODO 021 sees Montrose as the
favourite in a fixture the rest of the market priced the other way.

Two large market "shocks" that looked like swaps in the TODO 023 anomaly
catalogue — kelty-hearts v hamilton (2025-09-20) and edinburgh-city v
bonnyrigg-rose (2024-08-17) — are **not** this defect: sofascore agrees with
Betfair on both.

## Root cause

Not established. Candidates: the selection → side mapping for one Betfair
market (runner names matched to the wrong sofascore side), or a market whose
runners are listed away-first. The check that would have caught it — the two
independent books disagreeing by more than a swap — does not exist anywhere in
the L0 QA stage.

## Reproduction

`/root/BF_market_inverse/swapcheck_scratch.jl` on mcmc-beast (scratch, not
committed): build both 1X2 books wide, then
`d_direct = |bf_home − so_home| + |bf_away − so_away|`,
`d_swapped = |bf_home − so_away| + |bf_away − so_home|`, flag
`d_swapped < d_direct && d_direct > 0.15`.

## Blast radius

One fixture of 623 in the TODO 023 panel (negligible for its conclusions). Any
backtest priced off the Betfair close stakes this fixture against a mirrored
book. Other segments are unchecked.

## Proposed fix

1. Add a Fetch→Process→QA invariant: for every fixture with both books, flag
   `d_swapped < d_direct` with `d_direct > 0.15`, and refuse or repair the Betfair side.
2. Trace 14035501's Betfair market id and runner order to find the mapping fault.

## Acceptance criteria

- The QA stage reports cross-book swaps for every segment.
- 14035501 either carries corrected sides or is excluded with a stated reason.

## Scope guard

Do not "fix" by overwriting Betfair prices with sofascore prices; the Betfair
close is the trading book.
