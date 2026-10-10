# Ruling 3 — exact experiment 07 reproduction cell

**Cell:** m12 / raw / B0_canonical / prod. Gate: 1127 bets and normalised terminal
wealth 2.5152 ±0.00005. This is a validation reference, not another Phase 3 policy.

## Sources and reconstruction

All runner/loader references below are under
`experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/`.

| Fact | Exact source lines |
|---|---|
| m12 run UUID `132df5c2-c742-4e95-8693-3aeb2b2cbaef` | `l07_calibrated_trust_loader.jl:94–99` |
| 24/25 + 25/26 restriction, T−25, staleness 90 | loader `109–115`, data restriction `339–353` |
| B0: canonical tiers, tradeable book, not flat | `r07_optimal_portfolio_comparison.jl:177–186` |
| Market list comes from l2_tradeable_markets | runner `220`; `src/Calibration/diagnostics.jl:112–118`: 1X2, OU1.5, OU2.5, OU3.5, BTTS, in that order |
| Kelly fraction 0.30, λ23, cap0.25, commission0.02 | loader `131–134` |
| DeArb, KellyLogUtility, FractionalKelly, budget0.99, minimum0.001 | loader `194–200` |
| DailySlate, SlateDrawdown and FixedCap | loader `209–211`; runner `315` |
| Home/Under2.5 tier1; Draw/Away tier2 | loader `259–261` |
| Weight=t1/ratio^(tier−1), default zero | loader `273–284` |
| prod t1=0.35, ratio=1.4, λ23/cap0.25 | runner `268` |
| Construction and simulation call | runner `312–326` |
| Raw identity law w=1 | loader `146–163`; `src/Calibration/rate_pool.jl:358–367` copies each rate when shift=false |
| Default £1000 bankroll, reporting-scale only | `src/Portfolio/simulation.jl:56–61,79–80` |
| Committed headline and ladder | `results/optimal_portfolio_comparison.csv:1–2`: 99 slates, 1127 bets, return151.52046006000467%, unit wealth2.515204600600047 |

The reconstruction uses raw saved latents directly, equivalent to the historical
identity control. It does not fit/apply a market calibration container or invert rates.
Unspecified execution defaults remain max selection0.50, barrier1e-6, complete markets;
policy default KeepAll, no book-level trust excision. The runner is not included because
it would execute the unrelated full sweep; only this cell's exact constructors are used.
The committed CSV cell is read and its unique identity/ladder/headline asserted before use.
The new daily output retains the historical £1000 scale and adds unit-bankroll columns.

## Separate contemporary canonical arm

Today's serve-console stack prices the extra OU0.5 line, uses BakerMcHale and minimum
0.0001 instead of FractionalKelly0.30/minimum0.001. Its fixed measurement at source
`8018dacf` remains **2.773493294562897 / 1132 bets**. This is no longer judged against
the historical target. It is retained without rerunning it. The complete policy bundle
is different; the individual contributions of shrink/minimum/market geometry are not
isolated by this two-stack comparison.

## Input identity and failure comparison

Use the same saved UUID/cache/710 IDs as canonical; assert exact equality of the fixture
universe and rebuilt T−25 book against the committed Phase 0 inputs. New output lives in
`results/phase0_historical/`, preserving previous canonical results/audits/logs.

Inventory of all 17 committed experiment 07 CSVs found no daily/slate output for this
exact runner-3 cell. The sole dated per-bet ledger, `forensic_bet_ledger.csv`, has m12/raw
**1643 bets on 99 days**, but it is FlatTrust(1.0) on the 13-direction forensic book
(`r07_line_forensics_calibrated.jl:79–80,146–163`). If the historical gate fails, a
per-day comparison to this ledger must be labelled non-equivalent; it cannot reveal
the exact target cell's first divergent slate. No same-cell reference will be invented.
