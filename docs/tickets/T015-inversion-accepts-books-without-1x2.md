# T015 — `invert_market_rates` accepts totals-only books, whose home/away split is the optimiser's initial guess, not the market

| Field | Value |
|---|---|
| Severity | medium |
| Area | `src/Calibration/rate_pool.jl` (`invert_market_rates`, `MarketInversionConfig`), every consumer of `lambda_mkt_h/a` |
| Status | open |
| Raised | 2026-09-22 |

## Evidence

Found in TODO 023 Phase 2 (`current_development/market_inverse_dynamics/`,
`results/phase2/fixture_features.csv`). Of the 623 Scottish Lower 24/25 + 25/26
fixtures the default `MarketInversionConfig` accepts, **27 have no 1X2 market in
the Betfair (−20, 0] TWA book** — only Over/Under (and sometimes BTTS) lines.
For all 27 the inverted log-rate supremacy log λ_h − log λ_a is **positive and
between 0.36 and 0.63** (mean |Δ| 0.49 vs 0.36 for books with a 1X2).
`get_initial_guess(::DoublePoissonMarketFeature) = [log(1.5), log(1.0)]`
(`src/features/market_inverse_utils.jl:144`), a supremacy of 0.405.

A totals-only book constrains λ_h + λ_a but is (almost) symmetric in the split, so
Nelder-Mead walks along the total-goals ridge and stops near the initial ratio.
Every one of those fixtures therefore reads "home favourite by ~0.5" whatever
the teams.

In TODO 023 these fixtures surfaced as "market shocks" in Phase 1 (e.g.
edinburgh-city v bonnyrigg-rose 2024-08-17, dumbarton v inverness-ct
2024-10-26, elgin-city v clyde 2024-12-17) and as the most down-weighted
fixtures under the Phase 2 Student-t model. Relatedly, 44% of accepted books with
exactly 3 selections are down-weighted (ω < 0.5) against ~7% of books with ≥ 5.

## Root cause

`MarketInversionConfig` gates on `min_targets = 3`, SSE, convergence and rate
bounds only. None of these asks whether the quoted markets IDENTIFY both rates:
three Over/Under selections pass `min_targets` and fit with SSE ≈ 0 because the
split is a free direction.

## Blast radius

* `GenerativeRateCalibrator` / `calibrate_fit`: a totals-only fixture is pooled
  towards a home-favoured split the market never priced, shifting 1X2 and every
  derivative price for that fixture.
* Every supremacy-slope / compression number measured against the inverted close
  (TODO 021, feature-compression EDA, TODO 023) includes ~4% of fixtures whose
  "market supremacy" is an artefact.

## Reproduction

`r02_market_feature_attribution.jl` §4 prints
`book quality: 27 fixtures without 1X2, 532 of 623 well identified`; the rows are
`fixture_features.csv` with `has_1x2 == false`.

## Proposed fix

Add an identification gate to `MarketInversionConfig`, e.g.
`require_result_market = true`: refuse (with its own reason string) any book with
no 1X2 / Asian-handicap / correct-score market, i.e. nothing that splits the goals
between the sides. Optionally also report `n_targets == 3` books separately in
`inversion_refusals` so thin books are visible.

## Acceptance criteria

- Totals-only books are refused with a distinct reason and fall back to the
  identity path in `calibrate_fit`.
- The TODO 021 / TODO 023 supremacy slopes are re-measured with the gate on.

## Scope guard

Do not replace the initial guess with a "neutral" one (e.g. equal rates) as the
fix: that only moves the artefact from +0.4 to 0.
