# Tickets

Defects and scoped pieces of work that are **deliberately not being fixed inline**,
because doing so would derail the conversation that found them.

Each ticket is a self-contained brief: a fresh Claude session should be able to open
one file, understand the problem, reproduce it, fix it, and know when it is done —
without reading the conversation that raised it.

## How to use this

**Raising one.** Copy the structure of an existing ticket. It must contain: evidence,
root cause with `file:line`, a reproduction, blast radius, proposed fix with
trade-offs, acceptance criteria, and an explicit scope guard saying what NOT to touch.
Add a row below.

**Working one.** Start a fresh session, point it at the ticket file, and let it work
only that ticket. Update `Status` here when it lands.

**Status values:** `open` · `in progress` · `blocked` · `done` · `wontfix`

## Open

| ID | Title | Severity | Area | Status | Raised |
|---|---|---|---|---|---|
| [T002](T002-scalar-taped-likelihood.md) | Engine likelihoods taped scalar-by-scalar (~20x AD work); `view` defeats vectorisation and NegBin crashes on the fast path | medium | `src/models/pregame/engines/`, `src/MyDistributions/` | open | 2026-08-26 |
| [T003](T003-home-advantage-population-fallback.md) | Unmapped teams silently lose home advantage at extraction (λ_h 0.849x); 28 call sites | medium | `src/models/pregame/engines/`, `src/models/pregame/components/home_advantage.jl` | open | 2026-08-26 |
| [T004](T004-1x2-grading-disagrees-with-score.md) | `is_winner` contradicts the recorded score on 3 fixtures (2-2 draws with no 1X2 winner); no QA invariant catches it | low | `ds.odds` grading | open | 2026-08-26 |
| [T006](T006-scottish-lower-arm-include-guards.md) | Arm 02/03/04 loaders re-include a shared loader (guard tests a name that never existed) and call `subset` unqualified | low | `current_development/scottish_lower/` | open | 2026-08-28 |
| [T007](T007-parallel-feature-test-hidden-dependency.md) | `features_tests.jl` depends on a probe defined only by an earlier sequential include, so the parallel suite fails | low | `test/` | open | 2026-08-29 |
| [T011](T011-portfolio-sizes-smile-latents-off-the-grid.md) | Portfolio prices a smile container's totals through φ (`p_model`) but sizes every stake off the plain (λ_h, λ_a) grid, so φ never reaches a bet | medium | `src/Portfolio/pricing.jl` | open | 2026-09-13 |
| [T015](T015-inversion-accepts-books-without-1x2.md) | `invert_market_rates` accepts totals-only books (27 of 623 Scottish Lower fixtures); their home/away split is the optimiser's initial guess (+0.4 supremacy), not the market | medium | `src/Calibration/rate_pool.jl` | open | 2026-09-22 |
| [T014](T014-betfair-1x2-home-away-swap.md) | Betfair 1X2 book has home/away swapped on ≥ 1 Scottish Lower fixture (14035501); no QA invariant compares it with the sofascore book | low | `ds.betfair_odds` mapping | open | 2026-09-22 |
| [T010](T010-postgres-storage-refuses-smile-latents.md) | `PostgresStorage` refuses `SmileLatents` on save and rebuilds every panel as `CountLatents` on load, dropping φ | medium | `src/training/inference/db_storage.jl` | open | 2026-09-12 |

## Closed

| ID | Resolution | Closed |
|---|---|---|
| [T005](T005-betfair-summariser-drops-90pc.md) | Closing selections are retained with nullable opening fields; the wider default window and coverage warning prevent silent fixture loss | 2026-09-19 |
| [T012](T012-zero-trust-market-reprices-the-portfolio.md) | Opt-in BookSpec trust excises wholly zero-trust markets before payoff, Kelly and shrinkage geometry while preserving legacy defaults | 2026-09-18 |
| [T001](T001-pooled-tournament-clock.md) | Pooled tournament groups use a shared calendar clock with strict kickoff safety | 2026-08-25 |
| [T011](T011-portfolio-sizes-smile-latents-off-the-grid.md) | Smile CDFs reweight score-grid anti-diagonals, so pricing, Kelly and shrinkage consume one joint tensor | 2026-09-18 |
| [T013](T013-multiscalegrw-missing-builder-dynamics-dispatch.md) | MultiScaleGRW builder dynamics extractors graduated to src/ and covered by regression tests | 2026-09-18 |
| [T008](T008-multilevel-fill-price-accounting.md) | V2 sweeps budget exact child liability; settlement uses fill cashflows and market-net commission | 2026-09-18 |
