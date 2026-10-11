# 26/27 paper trade of the frozen qs_marketobs scenario-Kelly config (TODO 045)

## Summary

1. **Scope.** 79 finished 26/27 League One/Two matches (2026-08-01 → 09-19), predicted walk-forward with no look-ahead. Bets were placed on the 60 matches with a T−25m order book, over 8 match days. All gates passed, and the DB audit was unchanged around every runner.
2. **The frozen config** (B1, λ 12) at the executable best back price: wealth **0.964** [0.70, 1.32], g −4.6×10⁻³/day [−44, +35]×10⁻³, MDD **−10.6 %** (watch −20.39 %, not breached). 91 bets on 50 matches, money ROI −5.3 % [−49, +44].
3. **The fallback** (λ 23): 0.983 at the back price, MDD −5.7 %. **m12 canonical**: 0.867, MDD −15.0 %, 124 bets.
4. **At the mid and at LTP**, the frozen config ended at 0.968 and 0.988, the fallback at 0.984 and 0.998, and m12 at 0.879 and 0.865. No cell breached the drawdown watch.
5. **CLV against the mid close** follows the price used: −0.69 pp [−0.88, −0.40] at the back, +0.05 at the mid, +0.39 [+0.22, +0.67] at LTP, about the same for every model. So the back price pays roughly the half-spread, and any edge lies inside the spread (as in the CLV napkin).
6. **Fills.** At a £1,000 notional, 30 % of the frozen bets exceed the best-level back size and 5 % exceed 3 levels; capped at the best level, 54 % of the money fills. At £200, the figures are 7 % and 0 %.
7. **Log loss vs the mid close.** qs +0.003 [−0.012, +0.017] and m12 +0.034 [+0.008, +0.062]. Both are worse on the transition clubs, and the frozen config's transition-club bets lost (ROI −39 % vs +22 % elsewhere; descriptive only).
8. **Every interval spans zero growth** and also contains wave 5's 25/26 growth (9.7×10⁻³/day). **8 match days cannot rank anything:** this replay neither confirms nor refutes the frozen config.

## 1. Data and coverage

| Item | Value | Source |
|---|---|---|
| DataStore (read-only rebuild) | 2,049 matches, last 2026-09-19, sha `f407a1b4…` | `logs/phase0.log` |
| 26/27 fixtures played | 79 / 79 (tournaments 56, 57) | `results/phase0/phase0_fixtures.csv` |
| With any Betfair market | 70 | `phase0_coverage.csv` |
| With a T−25m 1X2 book (two-sided, ≤ 15 min old, complete, mid overround in [0.90, 1.10]) | **60** | `phase0_coverage.csv` |
| 1X2 + OU2.5 / all canonical markets / 1X2 LTP | 57 / 49 / 50 | `phase0_coverage.csv` |
| Order book (pinned) | `betfair_live.order_book_1m`, sha `2adb1271…` | `logs/phase0.log` |
| Odds frames, back / mid / LTP | 744 / 744 / 418 rows on 60 / 60 / 50 fixtures | `phase0_odds_{back,mid,ltp}.csv` |
| Mid TWA close, (−20, 0] min | 60 fixtures | `phase0_close_mid.csv` |

Coverage by match day: a book exists on every day except **08-22** (10 played, no T−25m book) and **08-29** (9 played, no market at all). **LTP is missing on 08-01** (10 books, no last-traded prices). The 8 days with books are 08-01, 08-08, 08-15, 08-28, 09-05, 09-12, 09-15 and 09-19; 08-28 and 09-15 each have one match. The LTP cells have 7 days.

**Gates** (`results/phase0/`, `results/phase2/phase2_engine_gate.csv`):
- **Inversion gate (ruling 3): PASS.** The tolerance, 1e-12, was stated before the run; the result was max |Δ log λ| = 0.0 on 1,430 rows of the 24/25–25/26 committed table.
- **Engine gate (ruling 9): PASS bit-identically.** It was run in Phase 0 and again at the Phase 2 head: frozen 1.6241209548752367 / 494, fallback 1.3208208815934512 / 496, m12 1.327330130248303 / 557.
- **Positions 1–40** reproduce the persisted held-out sets for qs and m12 (40/40).
- **m12 inventory (ruling 4):** the DB folds miss the 09-05 round, so m12 was extended in memory, as approved.
- **DB audit:** 10 tables, unchanged around every runner (r00–r04; r04 ran twice, and the first run stopped on a column-name clash after its audit).

**26/27 market-rate observations** (`phase0_market_rates_2627.csv`): 56 fixtures have a full book. Four were refused for KL > 0.01, and 19 had no book; those contribute no observation.

## 2. Predictions (Phase 1)

Both models were extended in memory by four biweekly walk-forward folds (positions 41–44: 20 / 19 / 20 / 20 held-out fixtures; training cut-offs 04-25 / 08-08 / 08-22 / 09-05). Each fold trains only on days strictly before its first held-out day; the sampled filtration passes on all 8 folds (`results/phase1/phase1_*_filtration_sampled.csv`).

| Model | Sampler | Max R̂ (folds 41–44) | Min tail ESS | Divergences | Reruns |
|---|---|---|---|---|---|
| qs_marketobs (`81fdf817`) | 4 × (500 + 1000) | 1.0056 / 1.0072 / 1.0061 / 1.0115 | 768 | 0 | none |
| m12 (`132df5c2`) | 4 × (800 + 800) | 1.0061 / 1.0084 / 1.0044 / 1.0053 | 873 | 0 | none |

- **qs market observations** per fold: 517 / 534 / 543 / 554, of which 0 / 17 / 26 / 37 are 26/27 order-book rates.
- **m12 in-memory vs DB folds** (69 shared fixtures, check only): mean |Δp| 0.0004 (max 0.0027).
- **m12 uses the played XI** (D17).

**1X2 per-selection log loss vs the mid close** (60 fixtures; slate-clustered 95 % intervals; `phase1_logloss.csv`):

| Group | Close | qs | qs − close | m12 | m12 − close |
|---|---|---|---|---|---|
| All 60 | 0.5576 | 0.5604 | +0.0028 [−0.0117, +0.0172] | 0.5912 | +0.0336 [+0.0079, +0.0625] |
| Transition clubs (22) | 0.4918 | 0.5221 | +0.0303 [+0.0054, +0.0527] | 0.5745 | +0.0827 [+0.0262, +0.1507] |
| Other (38) | 0.5957 | 0.5826 | −0.0132 [−0.0258, −0.0016] | 0.6008 | +0.0051 [−0.0243, +0.0384] |

On all 79 fixtures, qs scores 0.5661 and m12 0.5941. qs is close to the market overall. On the transition clubs (airdrieonians, east-kilbride, kelty-hearts-fc, ross-county; 28 fixtures), qs is worse than the close; elsewhere it is better, but only just clear of zero.

## 3. Paper trade: all 9 cells

The setup:
- one continuous bankroll from 1.0 per cell, MDD measured on that path including the opening point;
- 2 % commission on net winnings;
- each price cell decides and settles at its own price (D12);
- CLV is measured against the mid TWA close;
- intervals are slate-clustered 95 % (8 days; 7 for LTP).

Sources: `results/phase2/phase2_summary.csv`, `phase2_uncertainty.csv`, `phase2_daily.csv`, `phase2_bets.csv`.

| Price | Cell | Wealth [95 %] | g ×10⁻³/day [95 %] | MDD % | Bets (fixtures) | Money ROI % [95 %] | CLV pp [95 %] | Beat close % | Mean exposure |
|---|---|---|---|---|---|---|---|---|---|
| **back** | **frozen B1 λ12** | **0.9639** [0.704, 1.319] | −4.6 [−43.8, +34.6] | **−10.56** | 91 (50) | −5.3 [−49.0, +44.0] | −0.69 [−0.88, −0.40] | 22.0 | 0.086 |
| back | fallback B1 λ23 | 0.9834 [0.828, 1.169] | −2.1 [−23.6, +19.5] | −5.69 | 91 (50) | −4.3 [−48.0, +42.9] | −0.69 [−0.88, −0.40] | 22.0 | 0.049 |
| back | m12 canonical λ23 | 0.8670 [0.709, 1.052] | −17.8 [−43.1, +6.3] | −14.95 | 124 (54) | −17.3 [−47.5, +6.8] | −0.68 [−0.81, −0.45] | 21.8 | 0.099 |
| mid | frozen B1 λ12 | 0.9677 [0.721, 1.304] | −4.1 [−40.9, +33.2] | −9.52 | 118 (56) | −3.8 [−37.4, +30.6] | +0.05 [−0.12, +0.31] | 52.5 | 0.108 |
| mid | fallback B1 λ23 | 0.9837 [0.837, 1.157] | −2.1 [−22.2, +18.2] | −5.11 | 118 (56) | −3.4 [−36.6, +30.9] | +0.05 [−0.12, +0.31] | 52.5 | 0.060 |
| mid | m12 canonical λ23 | 0.8790 [0.704, 1.057] | −16.1 [−43.9, +6.9] | −15.66 | 144 (60) | −13.1 [−41.3, +7.0] | +0.17 [−0.03, +0.45] | 61.1 | 0.116 |
| ltp | frozen B1 λ12 | 0.9879 [0.754, 1.282] | −1.7 [−40.2, +35.5] | −8.50 | 100 (47) | −1.8 [−39.4, +37.4] | +0.39 [+0.22, +0.67] | 66.0 | 0.098 |
| ltp | fallback B1 λ23 | 0.9983 [0.862, 1.155] | −0.2 [−21.2, +20.6] | −4.56 | 100 (47) | −0.4 [−38.0, +37.4] | +0.39 [+0.22, +0.67] | 66.0 | 0.056 |
| ltp | m12 canonical λ23 | 0.8649 [0.677, 1.055] | −20.7 [−55.7, +7.7] | −17.87 | 114 (48) | −16.4 [−48.8, +7.4] | +0.34 [+0.18, +0.58] | 66.7 | 0.119 |

**Drawdown watch** (`phase2_drawdown_watch.csv`): the frozen back-price cell's MDD is −10.56 % against the −20.39 % watch, so it was **not breached**. No cell breached; the deepest is m12 at LTP (−17.87 %).

**Frozen back-price daily path** (`phase2_daily.csv`): bankroll after each day.

| Day | 08-01 | 08-08 | 08-15 | 08-28 | 09-05 | 09-12 | 09-15 | 09-19 |
|---|---|---|---|---|---|---|---|---|
| Bets | 13 | 12 | 13 | 2 | 18 | 14 | 3 | 16 |
| Slate P&L % | −0.46 | −10.14 | +7.82 | −1.39 | +7.38 | +1.15 | −2.15 | −4.63 |
| Bankroll | 0.9954 | 0.8944 | 0.9644 | 0.9510 | 1.0211 | 1.0328 | 1.0106 | 0.9639 |

- The SlateDrawdown risk scale was active on half the days.
- The 0.25 FixedCap never bound.
- All bets are in 1X2 or Under 2.5, as basket B1 requires.

**By market** (`phase2_markets.csv`, frozen back): 1X2 had 73 bets, ROI −8.9 %, CLV −0.59 pp; U2.5 had 18 bets, ROI +15.7 %, CLV −1.08 pp.

**By selection** (`PAPER_LEDGER.csv`): home 32 bets, 14 won, ROI +14.5 %; away 22 bets, 5 won, ROI −40.3 %; draw 19 bets, 4 won, ROI −87.4 %; U2.5 18 bets, 6 won, ROI +15.7 %.

**Transition clubs** (descriptive, no interval): 37 of the 91 frozen back bets were on transition-club fixtures. Those bets had ROI −39.2 % and CLV −0.95 pp, against +22.3 % and −0.51 pp on the other 54. This matches the Phase 1 finding that qs mis-prices the promoted and relegated clubs.

**Bet sets differ by price** (`phase2_bet_sets.csv`, D12/D25). Frozen and fallback share one solve, so their bets are identical and only the stakes differ.

| Cell | Back | Mid | LTP | Back ∩ mid | Back ∩ LTP | In all three | Mid not back | LTP not back |
|---|---|---|---|---|---|---|---|---|
| frozen / fallback | 91 | 118 | 100 | 90 | 74 | 73 | 28 | 26 |
| m12 canonical | 124 | 144 | 114 | 121 | 98 | 95 | 23 | 16 |

Paying the spread removes about a quarter of the mid-price bets.

**The LTP cells have no 08-01 round:** LTP was not recorded that day, so 10 booked fixtures are missing from every LTP cell, which start on 08-08. On 08-01 the frozen config made −0.46 % at the back price and +0.38 % at the mid. LTP therefore lost one of 8 days, and its numbers are not on the same days as the back and mid cells.

**Settlement:** back-price bets settle at the quoted back price (the overround is above 1, so no de-arb). At the mid and LTP, 62 of 118 and 45 of 100 frozen bets were settled below the quote by `DeArb` (D22).

**Paired daily log growth** (`phase2_paired.csv`; shown only for its width):
- frozen − m12: +13.2×10⁻³ [−19.1, +46.1] (back), +12.0 [−26.5, +48.7] (mid), +19.0 [−27.6, +62.8] (LTP);
- frozen − fallback: −2.5×10⁻³ [−20.2, +14.9] at the back.
- Every interval spans zero. **No ranking is drawn.**

## 4. Fills (ruling 7)

Frozen config, back price. The money stake is the stake fraction × the paper bankroll × the notional. Source: `results/phase2/phase2_fill.csv`; the per-bet flags are in `PAPER_LEDGER.csv`.

| Notional | Bets > best-level size | Bets > 3-level depth | Median / max stake | Wealth uncapped | Capped at best level: wealth (money filled) | Capped at 3 levels: wealth (money filled) |
|---|---|---|---|---|---|---|
| £1,000 | **29.7 %** (27 / 91) | **5.5 %** (5 / 91) | £3.3 / £60.1 | 0.9639 | 0.9684 (54 %) | 0.9318 (92 %) |
| £200 | 6.6 % (6 / 91) | 0.0 % | £0.7 / £12.0 | 0.9639 | 1.0131 (92 %) | 0.9639 (100 %) |

- **Book depth:** League One/Two order books are thin at T−25m. The median best-level back size is £5–12 on the 1X2 and U2.5 runners (`phase0_entry_book.csv`). At £1,000, the stakes above the best level hold almost half the money.
- **Capped wealth rises at £1,000:** capping at the best level raises wealth from 0.964 to 0.968, because the capped bets were net losers. The direction is noise; the size of the shortfall is the finding.
- **The capped paths are approximate (D23):**
  - the policy's stake fractions are not re-solved on the capped path;
  - the 3-level cap is priced at the best level, which is optimistic;
  - sizes come from one snapshot ≤ 15 min old, and the queue and other bettors are ignored.

## 5. Comparison with the wave-5 backtest

Sources: `results/phase2/phase2_wave5_reference.csv` (from wave 5's `phase2_summary.csv`) and the 26/27 cells above.

| | 24/25 (wave-5 selection, LTP) | 25/26 (wave-5 evaluation, LTP) | 26/27 back | 26/27 mid | 26/27 LTP |
|---|---|---|---|---|---|
| Frozen: g ×10⁻³/day | 17.52 | 9.70 | −4.6 [−43.8, +34.6] | −4.1 | −1.7 |
| Frozen: MDD % | −13.07 | −19.26 | −10.56 | −9.52 | −8.50 |
| Frozen: bets / day | 10.8 | 9.9 | 11.4 | 14.8 | 14.3 |
| Frozen: money ROI % | 22.6 | 12.6 | −5.3 | −3.8 | −1.8 |
| Frozen: CLV pp | +0.21 | +0.29 | −0.69 | +0.05 | +0.39 |
| Frozen: mean exposure | 0.076 | 0.073 | 0.086 | 0.108 | 0.098 |
| m12: g ×10⁻³/day | 15.04 | 5.66 | −17.8 | −16.1 | −20.7 |
| Days | 49 | 50 | 8 | 8 | 7 |

- **Growth:** the 26/27 frozen g interval at the back price, [−43.8, +34.6]×10⁻³, contains the 25/26 value of 9.70×10⁻³. 8 days cannot separate them.
- **Activity:** bets per day and exposure are similar to the backtest, so the config behaves as frozen on the new season.
- **CLV is not like-for-like:**
  - the backtest measured T−25m LTP against the odds_history LTP close;
  - here the close is the order-book mid TWA.
  - At LTP the sign and size agree (+0.39 vs +0.29 pp). At the executable back price the spread turns this negative (−0.69 pp), the same −0.6 pp the CLV napkin found on this order book.
- **The backtests never paid the spread.** Wave 5's 25/26 result is a mid-or-LTP number. Its executable equivalent would carry about this −0.7 pp CLV cost.

## 6. Caveats

1. **8 match days, two of them single-fixture midweek days, and 91 frozen bets.** Every growth, ROI and wealth interval spans zero and the backtest value. Nothing here ranks configs or validates the frozen one.
2. **Market-rate source change.** The 24/25–25/26 observations are odds_history LTP closes; the 26/27 ones are order-book mid TWA closes (same inversion code; the gate passed on the old seasons only). By fold 44, 37 of 554 qs observations come from the new source.
3. **Transition clubs.** Four clubs changed tier, across 28 fixtures. qs is worse than the close on their 22 closed fixtures. The frozen bets on them lost (ROI −39 %), which is descriptive only.
4. **Ingestion gap after 09-19.** The order book and results stop at 2026-09-19. Two rounds also have no book: 08-22 (10 matches) and 08-29 (9 matches).
5. **LTP is missing on 08-01,** so the LTP cells start on 08-08 with 7 days.
6. **The order-book close is not the odds_history close.** CLV and log loss here are against the de-vigged mid TWA over (−20, 0] min, which is not comparable one-to-one with wave 4/5's CLV.
7. **m12 uses the played XI** (the confirmed starting rows) for its held-out fixtures. This is the same rule as its persisted folds; the XI is usually public about 60 min before the off, but it is not a forecast.
8. **The 26/27 folds train without 22/23–23/24,** because the splitter's scope covers 24/25 + 25/26 + 26/27. The folds are biweekly, not one per round:
   - fold 41 predicts 08-01 and 08-08 from data through 2026-04-25, so 08-08 does not see 08-01;
   - fold 42 predicts 08-15 and 08-22 from data through 08-08;
   - fold 43 predicts 08-28 → 09-05 from data through 08-22, so **09-05 is predicted from data through 08-22**;
   - fold 44 predicts 09-12 → 09-19 from data through 09-05.
   A live weekly refit would see one more round on half the days.
9. **The fill check is an approximation** (D23): it uses one snapshot of displayed size, keeps fractions un-re-solved, and prices the depth cap at the best level.
10. **Settlement at the mid and LTP uses DeArb prices** below the quote on about half the bets (D22). The back cell is the only executable one.

## Files

- `PAPER_LEDGER.csv`: the frozen config at the back price, one row per bet (91). It records the back price and the settled price, back size, 3-level depth, lay / mid / LTP, model and market probabilities, opening bankroll, stake fraction and £ at a £1,000 notional, the fill flags, the result, P&L and CLV.
- `results/phase2/`:
  - `phase2_summary.csv`: 9 cells with intervals;
  - `phase2_uncertainty.csv`, `phase2_paired.csv`;
  - `phase2_daily.csv`: the per-day ledger behind each bankroll path;
  - `phase2_bets.csv`: per-bet rows for all 9 cells, with book sizes and CLV;
  - `phase2_markets.csv`, `phase2_bet_sets.csv`, `phase2_books.csv`, `phase2_fill.csv`, `phase2_drawdown_watch.csv`, `phase2_solve_diag.csv`, `phase2_wave5_reference.csv`, `phase2_engine_gate.csv`;
  - the DB audit CSVs.
- Code: `l00_p27.jl` (inputs, replay), `l01_sampling.jl` (extension), `l02_predictions.jl`, `l03_paper.jl`, runners `r00`–`r04`. Logs: `logs/phase0.log`, `logs/extend_{qs,m12}.log`, `logs/phase1_predictions.log`, `logs/phase2.log`.
