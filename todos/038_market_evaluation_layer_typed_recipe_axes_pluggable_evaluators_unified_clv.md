# 038 — Market evaluation layer: typed recipe axes, pluggable evaluators, unified CLV

| Field | Value |
|---|---|
| ID | 038 |
| Title | Market evaluation layer: typed recipe axes, pluggable evaluators, unified CLV |
| Status | ACTIVE |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-28 |
| Updated | 2026-09-28 |
| Related Files / Commits / PRs | TODO 036 (merge first), TODO 031 (W5), TODO 030; brief `experiments/claude_market_eval_prompt.md`; review brief `experiments/claude_market_eval_review_brief.md`; branch `feat/market-eval` |

## Context & Problem Statement

The project's open question is whether the models have an **exploitable edge against the
market**, not whether they are more accurate than it. On the W2 panel, every model's target
LogLoss (0.636–0.640) is behind the Betfair close (0.632).

The harness portfolio stage (`src/harness/portfolio.jl`) hard-codes two whole configurations,
`close_option_b` and `t25_calibrated`. Each bundles pricing, calibration, execution instant,
staking and metrics into one path, and it reports **no CLV**. That leaves the backtest's +17% ROI
at T−25 impossible to judge.

CLV code exists in two places, and the two use different definitions:
- `Calibration.bet_clv`/`clv_summary`: `100·(odds_taken/odds_close_final − 1)`, against the raw
  close;
- `MatchDay.clv_for_order`: `p_close_devigged − 1/effective_odds`, with VWAP fills and lays.

The paper ledger's `clv_audit` table is empty.

## Acceptance Criteria

Design agreed with the human in a grill-me session on 2026-09-28. The Julia type-dispatch design
is mandatory.

- [ ] `src/Portfolio/evaluation/`: typed axes composed in a parametric `MarketRecipe{P,I,S,M}`.
  - Pricing: `RawPosterior`, `Calibrated{C<:AbstractCalibrator}`.
  - Book instant: `MinutesBefore`, `Close`.
  - Staking: `FlatStake`, `SystemStaking{S<:PortfolioSystem}`.
  - Market set: `Data.MarketConfig`.
  - Plus `market_recipe_hash`, a `BetLedger`, and `evaluate(::AbstractMarketEvaluator, ledger, ctx)`
    that returns long score rows.
  - Evaluators: `CLVEvaluator`, `EdgeQuality`, `SlateGrowth`, `ROIRisk`.
  - Behaviour is selected by multiple dispatch only; no Symbol switches.
- [ ] **Unified CLV.** One canonical implementation reports `clv_pp`, `clv_pct` and `beat_close`,
      with the close reference as a dispatch type (`DeviggedClose()` / `RawClose()`).
  - `Calibration.bet_clv`/`clv_summary` and `MatchDay.clv_for_order` become thin wrappers with
    **bit-identical parity tests** against their current outputs.
- [ ] Harness: `src/harness/market.jl` orchestrates recipes over `RunRef`s and writes
      `harness_scores` with `stage = "market"` and an additive `market_recipe_hash` column. A new
      `harness_market_recipes` table holds each recipe's full typed spec.
  - [ ] Today's `close_option_b` and `t25_calibrated` containers re-expressed as recipes reproduce
        the existing finalist numbers **bit-identically**.
- [ ] Two bet sets: a flat edge-threshold ledger (with CLV by edge bucket), and the configured
      system's bets (stake-weighted CLV).
- [ ] First grid (after the human's go):
  - runs: W0 `m12` and `m05` (632/611 panels), W0 `m00` control, and the W2 co-finalists plus
    bases plus `m12_td_ref` (1,070 panel);
  - recipes: {raw, calibrated} × {T−25, close} × {flat, Option-B system} × all markets.

  Panels are never mixed.
- [ ] Paper ledger: `clv_audit` backfilled for the settled paper bets through the unified CLV
      (a betdb write, after the human's go). Automatic CLV at settlement is a follow-up TODO.
- [ ] Built by Claude CLI (Opus 5.5) on `feat/market-eval` after TODO 036 merges, reviewed by a
      separate Claude CLI session, with at most 2 fix rounds.

## Ideas & Candidate Solutions

- The kill criterion is **not pre-registered** (human decision). The report must therefore show
  every slice (market × league × edge bucket × instant × pricing), with multiplicity caveats.
- Rejected:
  - a trait/Symbol single-struct design;
  - putting domain types in the harness;
  - a separate `market_scores` table.
- Later evaluators: capacity/fill modelling, lineup-time instants, automatic MatchDay CLV.

## Work Log & Progress

- [2026-09-28 @claude] Created. The design was agreed with the human (grill-me, 3 rounds). Build
  waits for the TODO 036 merge.

## Verification & Findings

Not run yet.
