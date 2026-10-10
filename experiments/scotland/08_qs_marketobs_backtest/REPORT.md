# 08 — qs_marketobs backtest at archive prices: report (Phase 4)

## Summary

- **Q1: qs_marketobs does not beat m12 at T−25m or T−6h after the 2% commission.** At T−25m it trails in all 5 policies (canonical wealth 1.857 vs 2.773, d = −4.05 ×10⁻³/day [−11.95, 1.27]); at T−6h it also trails in all 5, and 2 of 5 intervals exclude 0.
- It leads m12 only in the low-coverage T−24h subsets, under every policy, and no committed interval excludes 0: canonical T−24h/90 d = +1.64 [−1.57, 4.25] (78 vs 85 bets, 32 slates); T−24h/360 d = +3.37 [−0.26, 4.74] (270 vs 296 bets, 68 slates).
- **Q2: no sign that the edge is larger early.** On the same fixtures, qs_marketobs's T−25m bets return more per stake than its early bets: 43.7% (84 bets) vs 27.5% (78) on the T−24h/90 fixtures, 22.8% (480) vs 13.6% (323) on the T−6h fixtures.
- Its T−24h lead over m12 is already there at T−25m on the same fixtures (ROI 36.3% vs 17.7%, 308 vs 340 bets), so it is a property of that fixture subset, not of the timing.
- **T−24h/360 prices are up to 6 hours old** (median 111 min, p95 335 min before T−24h). They are last-traded prices that could not have been executed at the T−24h bet time.
- **Q3: the trust tiers add nothing consistent.** At T−25m, canonical − plain_kelly runs from −1.42 (market_c0) to +1.36 (m12) ×10⁻³/day and every interval spans 0; for qs_marketobs it is −0.08 [−2.85, 1.35].
- **Q4: scenario Kelly helps the market-informed models at T−25m only.** Scenario − canonical: qs_marketobs +1.50 [0.14, 2.66], fusion +1.17, market_c0 +0.89 (all exclude 0); m12 −0.79 [−1.47, 1.06]. m12 stays first (2.565 vs 2.154). Early slates are small, so it changes little there.
- **Q5: CLV is positive in all 100 cells** (+0.17 to +1.05 pp; 99 of 100 intervals exclude 0). At T−25m it ranks the models opposite to wealth: m12 has the lowest CLV (0.19 pp) and the highest wealth.
- **In-sample, with wide intervals.** The data are 2 seasons and 99 slates at T−25m (32 at T−24h/90). Experiment 07 chose these policies on the same seasons using m12, which favours m12. Prices are last-traded, not executable.
- At T−25m the median 90% interval for the paired m12 difference is 13.2 ×10⁻³/day wide, against a median |d| of 3.2. Only 8 of 40 policy contrasts exclude 0, against about 4 expected by chance.

Units: g = mean daily log growth of the unit bankroll; d = paired difference in g against m12 on the same days. Both are ×10⁻³ per slate day, with 90% block-bootstrap intervals (8-week noncircular blocks within season, 999 reps). All wealth, ROI and growth figures are net of Betfair's 2% commission on net winnings (`PerBetCommission(0.02)`). CLV ignores commission.

## What ran

- **Models.** Saved out-of-sample fits only, with no refits; the 710 held-out fixtures of 24/25 and 25/26.

  | Label | Source |
  |---|---|
  | qs_marketobs | `81fdf817` |
  | m12 (live) | `132df5c2` |
  | grw_joint | `98447840` |
  | fusion_qs_nobias | `5afdfeb3` |
  | market_c0 | `market_latents.jls`, 512 draws per fixture (128 θ × 4 state draws) |

- **Snapshots.** Betfair last-traded prices with a maximum staleness. T−6h and both T−24h settings are low-coverage subsets, never pooled with T−25m.

  | Snapshot | Max staleness | Fixtures with complete 1X2 (of 710) |
  |---|---|---|
  | T−25m | 90 min | 545 |
  | T−6h | 90 min | 182 |
  | T−24h | 90 min | 47 |
  | T−24h | 360 min | 170 |

- **Policies.** These were frozen before any grid number; see [DECISIONS_LOG.md](DECISIONS_LOG.md).
  - **canonical:** the serve-console stack. Trust Home and Under 2.5 at 0.35, Draw and Away at 0.25, everything else 0. BakerMcHale shrink, λ 23, slate cap 0.25.
  - **scenario_kelly:** the canonical policy with the per-match Kelly solve replaced by one joint daily ScenarioKelly solve (1000 joint-scoreline scenarios from shared posterior draws).
  - **plain_kelly:** 1X2, OU2.5 and BTTS at trust 1, FractionalKelly 0.25, λ 23.
  - **option_b_live:** FractionalKelly 0.30, trust 1 and 1/1.4 including Over 1.5, λ 28.
  - **historical_exp07:** the experiment 07 reproduction book.
- **Gate.** m12 × T−25m × historical_exp07 = 2.515204600600047 / 1127 bets, bit-identical to experiment 07. Canonical and option_b_live also reproduce Phase 0 exactly. The DB audit was unchanged on all 10 tables.
- **CLV.** The close is the Betfair time-weighted average over the last 20 minutes, complete markets only, with multiplicative de-vig. `clv_pp = 100·(p_close − 1/odds)`.

## Results per snapshot

Each table has 25 cells: 5 policies × 5 models. Wealth is the final unit bankroll. ROI is experiment 07's convention: Σ P&L fraction / Σ stake fraction. Turnover is Σ stake fractions. MDD is the maximum drawdown. Source: [`results/phase3/phase3_cells.csv`](results/phase3/phase3_cells.csv).

### T−25m (90 min staleness): 99 slates, 611 priced fixtures

|      Policy      |      Model       | Wealth |  g ×10⁻³/day [90%]  |    d vs m12 [90%]     | Bets | ROI % | Turnover | MDD % | CLV pp | Beat % |
|------------------|------------------|--------|---------------------|-----------------------|-----:|-------|----------|-------|--------|--------|
| canonical        | m12              | 2.773  | 10.30 [5.01, 19.59] | —                     | 1132 | 18.5  | 6.1      | -16.2 | 0.19   | 56     |
| canonical        | qs_marketobs     | 1.857  | 6.25 [2.75, 10.63]  | -4.05 [-11.95, 1.27]  | 984  | 17.2  | 3.9      | -12.6 | 0.35   | 58     |
| canonical        | fusion_qs_nobias | 1.669  | 5.17 [1.99, 8.95]   | -5.13 [-13.60, 0.16]  | 981  | 16.0  | 3.5      | -12.1 | 0.36   | 58     |
| canonical        | grw_joint        | 2.259  | 8.23 [5.17, 13.11]  | -2.07 [-8.83, 3.46]   | 1031 | 17.7  | 5.0      | -19.2 | 0.21   | 55     |
| canonical        | market_c0        | 1.371  | 3.19 [0.56, 8.73]   | -7.12 [-14.68, -0.81] | 974  | 9.6   | 3.8      | -20.6 | 0.39   | 60     |
| scenario_kelly   | m12              | 2.565  | 9.51 [4.08, 19.67]  | —                     | 991  | 19.9  | 5.3      | -22.5 | 0.17   | 55     |
| scenario_kelly   | qs_marketobs     | 2.154  | 7.75 [4.38, 11.84]  | -1.76 [-11.07, 3.48]  | 952  | 22.1  | 3.7      | -10.5 | 0.29   | 56     |
| scenario_kelly   | fusion_qs_nobias | 1.873  | 6.34 [3.37, 9.64]   | -3.18 [-12.91, 2.02]  | 941  | 20.2  | 3.3      | -9.5  | 0.32   | 57     |
| scenario_kelly   | grw_joint        | 2.223  | 8.07 [5.47, 12.37]  | -1.45 [-9.68, 3.95]   | 905  | 20.1  | 4.3      | -15.4 | 0.20   | 55     |
| scenario_kelly   | market_c0        | 1.497  | 4.07 [1.20, 9.33]   | -5.44 [-14.14, 0.88]  | 923  | 12.8  | 3.5      | -17.9 | 0.34   | 58     |
| plain_kelly      | m12              | 2.425  | 8.95 [4.33, 18.74]  | —                     | 1368 | 12.6  | 7.9      | -20.6 | 0.17   | 55     |
| plain_kelly      | qs_marketobs     | 1.873  | 6.34 [2.96, 12.12]  | -2.61 [-11.19, 2.36]  | 1308 | 11.9  | 5.9      | -18.0 | 0.30   | 57     |
| plain_kelly      | fusion_qs_nobias | 1.749  | 5.65 [2.48, 10.84]  | -3.30 [-11.91, 1.69]  | 1296 | 11.5  | 5.4      | -18.2 | 0.31   | 57     |
| plain_kelly      | grw_joint        | 2.212  | 8.02 [4.84, 14.27]  | -0.93 [-7.05, 3.46]   | 1369 | 11.1  | 7.9      | -20.4 | 0.19   | 55     |
| plain_kelly      | market_c0        | 1.577  | 4.60 [1.86, 9.81]   | -4.34 [-12.57, 1.01]  | 1302 | 8.8   | 5.8      | -18.8 | 0.36   | 59     |
| option_b_live    | m12              | 2.208  | 8.00 [4.24, 15.57]  | —                     | 1172 | 14.9  | 5.8      | -13.9 | 0.18   | 55     |
| option_b_live    | qs_marketobs     | 1.715  | 5.45 [2.81, 9.13]   | -2.55 [-9.13, 1.43]   | 1059 | 14.2  | 4.1      | -11.2 | 0.34   | 58     |
| option_b_live    | fusion_qs_nobias | 1.574  | 4.58 [2.20, 7.83]   | -3.42 [-10.25, 0.70]  | 1051 | 13.1  | 3.7      | -11.9 | 0.34   | 58     |
| option_b_live    | grw_joint        | 1.901  | 6.49 [3.93, 11.08]  | -1.51 [-6.39, 2.17]   | 1121 | 13.0  | 5.3      | -17.9 | 0.17   | 54     |
| option_b_live    | market_c0        | 1.428  | 3.60 [1.29, 7.38]   | -4.40 [-10.87, -0.14] | 1050 | 9.8   | 4.0      | -16.3 | 0.39   | 59     |
| historical_exp07 | m12              | 2.515  | 9.32 [4.72, 18.61]  | —                     | 1127 | 15.0  | 6.8      | -16.1 | 0.19   | 55     |
| historical_exp07 | qs_marketobs     | 1.836  | 6.14 [3.13, 10.46]  | -3.18 [-11.19, 1.60]  | 1020 | 14.1  | 4.7      | -13.1 | 0.34   | 58     |
| historical_exp07 | fusion_qs_nobias | 1.667  | 5.16 [2.38, 8.99]   | -4.16 [-12.33, 0.85]  | 1009 | 13.0  | 4.3      | -14.3 | 0.35   | 58     |
| historical_exp07 | grw_joint        | 2.028  | 7.14 [4.19, 12.75]  | -2.18 [-7.95, 2.27]   | 1070 | 12.9  | 6.0      | -20.7 | 0.17   | 54     |
| historical_exp07 | market_c0        | 1.499  | 4.09 [1.40, 8.53]   | -5.23 [-13.01, -0.05] | 1003 | 9.8   | 4.6      | -18.6 | 0.40   | 59     |

### T−6h (90 min staleness): 86–88 slates, 295 priced fixtures (low coverage)

|      Policy      |      Model       | Wealth | g ×10⁻³/day [90%]  |    d vs m12 [90%]    | Bets | ROI % | Turnover | MDD % | CLV pp | Beat % |
|------------------|------------------|--------|--------------------|----------------------|-----:|-------|----------|-------|--------|--------|
| canonical        | m12              | 1.654  | 5.72 [1.96, 9.89]  | —                    | 351  | 26.3  | 2.0      | -6.8  | 0.54   | 58     |
| canonical        | qs_marketobs     | 1.207  | 2.14 [0.11, 4.60]  | -3.58 [-7.79, 0.34]  | 323  | 13.6  | 1.5      | -11.1 | 0.70   | 60     |
| canonical        | fusion_qs_nobias | 1.140  | 1.48 [-0.89, 3.76] | -4.23 [-8.40, -0.24] | 318  | 10.8  | 1.3      | -11.5 | 0.91   | 63     |
| canonical        | grw_joint        | 1.294  | 2.92 [-0.17, 5.60] | -2.79 [-6.53, 0.53]  | 317  | 16.4  | 1.7      | -11.2 | 0.49   | 56     |
| canonical        | market_c0        | 1.201  | 2.09 [0.15, 4.90]  | -3.63 [-7.35, 0.44]  | 324  | 12.9  | 1.5      | -9.8  | 0.86   | 62     |
| scenario_kelly   | m12              | 1.707  | 6.08 [2.11, 10.26] | —                    | 353  | 29.3  | 1.9      | -7.5  | 0.51   | 57     |
| scenario_kelly   | qs_marketobs     | 1.215  | 2.21 [0.48, 4.62]  | -3.87 [-7.86, 0.23]  | 328  | 14.6  | 1.4      | -10.2 | 0.62   | 60     |
| scenario_kelly   | fusion_qs_nobias | 1.155  | 1.64 [-0.60, 3.99] | -4.44 [-8.47, -0.41] | 322  | 12.3  | 1.3      | -10.5 | 0.71   | 61     |
| scenario_kelly   | grw_joint        | 1.248  | 2.52 [-0.19, 5.13] | -3.56 [-7.15, -0.24] | 320  | 14.8  | 1.6      | -10.1 | 0.50   | 57     |
| scenario_kelly   | market_c0        | 1.214  | 2.21 [0.15, 5.01]  | -3.87 [-7.74, 0.27]  | 333  | 13.8  | 1.5      | -10.3 | 0.83   | 62     |
| plain_kelly      | m12              | 1.595  | 5.43 [1.66, 10.50] | —                    | 436  | 19.3  | 2.6      | -7.9  | 0.55   | 58     |
| plain_kelly      | qs_marketobs     | 1.134  | 1.46 [-0.84, 4.37] | -3.96 [-8.53, 0.30]  | 416  | 7.1   | 2.0      | -12.9 | 0.67   | 60     |
| plain_kelly      | fusion_qs_nobias | 1.091  | 1.01 [-1.53, 3.74] | -4.42 [-9.09, -0.45] | 412  | 5.5   | 1.8      | -12.3 | 0.86   | 62     |
| plain_kelly      | grw_joint        | 1.297  | 3.02 [-0.47, 6.40] | -2.41 [-6.72, 0.81]  | 434  | 10.6  | 2.7      | -14.3 | 0.57   | 57     |
| plain_kelly      | market_c0        | 1.134  | 1.46 [-0.90, 4.55] | -3.96 [-8.47, 0.20]  | 414  | 6.8   | 2.1      | -12.1 | 0.83   | 62     |
| option_b_live    | m12              | 1.528  | 4.82 [1.89, 8.37]  | —                    | 373  | 23.9  | 1.9      | -5.6  | 0.49   | 57     |
| option_b_live    | qs_marketobs     | 1.174  | 1.82 [0.02, 4.01]  | -2.99 [-6.23, -0.08] | 350  | 11.9  | 1.4      | -10.0 | 0.66   | 59     |
| option_b_live    | fusion_qs_nobias | 1.129  | 1.38 [-0.59, 3.31] | -3.44 [-6.73, -0.72] | 343  | 10.0  | 1.3      | -9.7  | 0.79   | 61     |
| option_b_live    | grw_joint        | 1.283  | 2.83 [0.25, 5.44]  | -1.99 [-4.73, 0.56]  | 348  | 15.4  | 1.7      | -8.7  | 0.54   | 56     |
| option_b_live    | market_c0        | 1.204  | 2.11 [0.33, 4.45]  | -2.71 [-5.58, 0.15]  | 353  | 13.4  | 1.5      | -7.3  | 0.75   | 61     |
| historical_exp07 | m12              | 1.661  | 5.77 [2.27, 10.00] | —                    | 368  | 24.5  | 2.2      | -6.6  | 0.50   | 57     |
| historical_exp07 | qs_marketobs     | 1.193  | 2.00 [-0.15, 4.53] | -3.76 [-7.59, -0.27] | 341  | 11.5  | 1.7      | -11.9 | 0.66   | 59     |
| historical_exp07 | fusion_qs_nobias | 1.149  | 1.58 [-0.68, 3.77] | -4.18 [-8.12, -0.94] | 334  | 10.0  | 1.5      | -11.4 | 0.79   | 61     |
| historical_exp07 | grw_joint        | 1.370  | 3.58 [0.67, 6.53]  | -2.19 [-5.45, 0.79]  | 343  | 17.1  | 2.0      | -9.4  | 0.54   | 56     |
| historical_exp07 | market_c0        | 1.230  | 2.35 [0.28, 5.05]  | -3.41 [-6.85, -0.00] | 343  | 12.9  | 1.7      | -8.7  | 0.75   | 61     |

### T−24h (90 min staleness): 31–32 slates, 53 priced fixtures (low coverage)

|      Policy      |      Model       | Wealth | g ×10⁻³/day [90%]  |   d vs m12 [90%]    | Bets | ROI % | Turnover | MDD % | CLV pp | Beat % |
|------------------|------------------|--------|--------------------|---------------------|-----:|-------|----------|-------|--------|--------|
| canonical        | m12              | 1.052  | 1.58 [-0.82, 5.50] | —                   | 85   | 11.5  | 0.5      | -10.5 | 0.64   | 62     |
| canonical        | qs_marketobs     | 1.109  | 3.23 [2.23, 5.33]  | +1.64 [-1.57, 4.25] | 78   | 27.5  | 0.4      | -4.2  | 0.86   | 64     |
| canonical        | fusion_qs_nobias | 1.082  | 2.47 [1.35, 4.13]  | +0.88 [-2.61, 3.29] | 83   | 24.2  | 0.3      | -4.0  | 0.84   | 62     |
| canonical        | grw_joint        | 1.054  | 1.63 [-0.79, 5.80] | +0.05 [-1.15, 1.24] | 80   | 14.1  | 0.4      | -7.7  | 0.67   | 59     |
| canonical        | market_c0        | 1.090  | 2.70 [1.63, 5.55]  | +1.12 [-2.71, 5.42] | 70   | 25.5  | 0.4      | -4.3  | 0.87   | 62     |
| scenario_kelly   | m12              | 1.041  | 1.26 [-0.93, 5.73] | —                   | 81   | 9.9   | 0.5      | -11.4 | 0.58   | 60     |
| scenario_kelly   | qs_marketobs     | 1.133  | 3.91 [2.84, 5.82]  | +2.65 [-0.72, 4.68] | 81   | 33.3  | 0.4      | -3.5  | 0.43   | 57     |
| scenario_kelly   | fusion_qs_nobias | 1.089  | 2.66 [1.56, 4.20]  | +1.40 [-2.44, 3.36] | 86   | 25.1  | 0.4      | -3.6  | 0.59   | 61     |
| scenario_kelly   | grw_joint        | 1.065  | 1.98 [-0.96, 6.68] | +0.72 [-0.85, 1.73] | 81   | 16.7  | 0.4      | -7.2  | 0.71   | 59     |
| scenario_kelly   | market_c0        | 1.092  | 2.75 [1.67, 5.57]  | +1.49 [-2.82, 5.80] | 69   | 27.2  | 0.3      | -3.8  | 0.75   | 62     |
| plain_kelly      | m12              | 1.056  | 1.76 [-1.84, 5.53] | —                   | 90   | 10.9  | 0.6      | -9.3  | 0.66   | 61     |
| plain_kelly      | qs_marketobs     | 1.130  | 3.95 [2.73, 5.59]  | +2.19 [-1.12, 5.36] | 84   | 27.8  | 0.5      | -4.3  | 0.88   | 63     |
| plain_kelly      | fusion_qs_nobias | 1.089  | 2.75 [1.18, 3.88]  | +0.99 [-2.71, 3.87] | 87   | 21.4  | 0.4      | -4.0  | 0.83   | 62     |
| plain_kelly      | grw_joint        | 1.093  | 2.88 [-1.26, 6.10] | +1.12 [-0.49, 1.59] | 88   | 17.7  | 0.5      | -6.9  | 0.62   | 60     |
| plain_kelly      | market_c0        | 1.122  | 3.70 [2.32, 5.53]  | +1.94 [-2.49, 6.57] | 81   | 28.0  | 0.4      | -4.3  | 0.95   | 61     |
| option_b_live    | m12              | 1.051  | 1.57 [-0.62, 4.94] | —                   | 87   | 12.4  | 0.4      | -8.7  | 0.65   | 62     |
| option_b_live    | qs_marketobs     | 1.097  | 2.91 [2.12, 4.63]  | +1.34 [-1.44, 3.82] | 83   | 26.8  | 0.4      | -3.2  | 0.77   | 62     |
| option_b_live    | fusion_qs_nobias | 1.068  | 2.04 [1.01, 3.54]  | +0.48 [-2.67, 2.71] | 84   | 20.9  | 0.3      | -3.3  | 0.82   | 63     |
| option_b_live    | grw_joint        | 1.061  | 1.86 [-0.31, 5.26] | +0.29 [-0.72, 1.16] | 85   | 16.4  | 0.4      | -6.3  | 0.58   | 60     |
| option_b_live    | market_c0        | 1.092  | 2.74 [1.52, 4.57]  | +1.17 [-2.54, 4.55] | 79   | 27.4  | 0.3      | -3.3  | 0.79   | 60     |
| historical_exp07 | m12              | 1.061  | 1.85 [-0.73, 5.83] | —                   | 87   | 12.6  | 0.5      | -10.4 | 0.65   | 62     |
| historical_exp07 | qs_marketobs     | 1.114  | 3.38 [2.43, 5.47]  | +1.53 [-1.72, 4.41] | 83   | 26.4  | 0.4      | -3.9  | 0.77   | 62     |
| historical_exp07 | fusion_qs_nobias | 1.079  | 2.36 [1.12, 4.15]  | +0.52 [-3.22, 3.13] | 84   | 20.5  | 0.4      | -4.0  | 0.82   | 63     |
| historical_exp07 | grw_joint        | 1.067  | 2.04 [-0.43, 5.91] | +0.19 [-1.01, 1.19] | 85   | 15.5  | 0.5      | -7.5  | 0.58   | 60     |
| historical_exp07 | market_c0        | 1.108  | 3.21 [1.76, 5.44]  | +1.37 [-2.98, 5.30] | 79   | 27.2  | 0.4      | -4.0  | 0.79   | 60     |

### T−24h (360 min staleness): 67–68 slates, 190 priced fixtures (low coverage, not executable)

These prices were last traded up to 6 hours before T−24h, so they are prices nobody could have hit at the bet time. Median staleness is 111 min and p95 is 335 min ([coverage](results/phase0_historical/phase0_snapshot_overview.csv)). Read this table as a sensitivity on fixture coverage, not as a strategy.

|      Policy      |      Model       | Wealth | g ×10⁻³/day [90%]  |   d vs m12 [90%]    | Bets | ROI % | Turnover | MDD % | CLV pp | Beat % |
|------------------|------------------|--------|--------------------|---------------------|-----:|-------|----------|-------|--------|--------|
| canonical        | m12              | 1.163  | 2.22 [-0.05, 7.63] | —                   | 296  | 10.3  | 1.7      | -17.5 | 0.49   | 60     |
| canonical        | qs_marketobs     | 1.462  | 5.58 [3.55, 9.40]  | +3.37 [-0.26, 4.74] | 270  | 27.6  | 1.4      | -6.5  | 0.94   | 62     |
| canonical        | fusion_qs_nobias | 1.421  | 5.17 [3.11, 9.17]  | +2.95 [-0.45, 4.27] | 268  | 29.4  | 1.2      | -5.7  | 1.05   | 64     |
| canonical        | grw_joint        | 1.293  | 3.78 [1.57, 9.92]  | +1.56 [-1.27, 3.91] | 265  | 20.3  | 1.4      | -10.7 | 0.74   | 61     |
| canonical        | market_c0        | 1.312  | 4.00 [2.26, 8.28]  | +1.78 [-1.31, 3.71] | 259  | 21.8  | 1.3      | -6.5  | 1.01   | 64     |
| scenario_kelly   | m12              | 1.200  | 2.68 [0.77, 8.12]  | —                   | 299  | 12.9  | 1.6      | -15.5 | 0.47   | 59     |
| scenario_kelly   | qs_marketobs     | 1.483  | 5.80 [4.21, 9.25]  | +3.11 [-0.78, 4.67] | 275  | 29.0  | 1.4      | -5.1  | 0.75   | 60     |
| scenario_kelly   | fusion_qs_nobias | 1.404  | 4.99 [3.29, 8.77]  | +2.30 [-1.15, 3.64] | 270  | 28.9  | 1.2      | -5.3  | 0.91   | 62     |
| scenario_kelly   | grw_joint        | 1.310  | 3.97 [1.66, 10.29] | +1.29 [-1.74, 3.67] | 271  | 21.7  | 1.3      | -10.4 | 0.67   | 59     |
| scenario_kelly   | market_c0        | 1.335  | 4.25 [2.65, 8.92]  | +1.57 [-1.50, 3.38] | 254  | 23.4  | 1.3      | -6.1  | 0.98   | 63     |
| plain_kelly      | m12              | 1.140  | 1.96 [-0.13, 7.39] | —                   | 310  | 8.0   | 1.9      | -17.3 | 0.52   | 60     |
| plain_kelly      | qs_marketobs     | 1.443  | 5.47 [3.05, 9.47]  | +3.51 [-0.54, 5.09] | 289  | 22.6  | 1.7      | -7.0  | 0.91   | 62     |
| plain_kelly      | fusion_qs_nobias | 1.378  | 4.79 [2.49, 8.50]  | +2.83 [-0.83, 4.07] | 292  | 22.2  | 1.5      | -6.2  | 1.02   | 64     |
| plain_kelly      | grw_joint        | 1.332  | 4.27 [2.28, 10.05] | +2.31 [-0.21, 4.42] | 303  | 17.4  | 1.7      | -10.0 | 0.64   | 60     |
| plain_kelly      | market_c0        | 1.309  | 4.01 [1.77, 8.42]  | +2.06 [-1.30, 3.86] | 288  | 18.0  | 1.6      | -7.0  | 0.91   | 62     |
| option_b_live    | m12              | 1.143  | 1.96 [0.06, 6.57]  | —                   | 300  | 9.9   | 1.5      | -14.8 | 0.51   | 60     |
| option_b_live    | qs_marketobs     | 1.346  | 4.37 [2.56, 7.89]  | +2.40 [-0.57, 3.78] | 286  | 22.9  | 1.3      | -5.4  | 0.87   | 62     |
| option_b_live    | fusion_qs_nobias | 1.304  | 3.91 [2.18, 7.30]  | +1.94 [-0.80, 3.11] | 287  | 23.0  | 1.2      | -4.9  | 1.01   | 64     |
| option_b_live    | grw_joint        | 1.238  | 3.14 [1.60, 7.90]  | +1.17 [-0.76, 2.89] | 293  | 16.9  | 1.3      | -9.3  | 0.63   | 60     |
| option_b_live    | market_c0        | 1.243  | 3.20 [1.60, 6.92]  | +1.24 [-1.28, 2.69] | 281  | 18.3  | 1.2      | -5.6  | 0.87   | 62     |
| historical_exp07 | m12              | 1.178  | 2.41 [0.11, 7.97]  | —                   | 298  | 10.3  | 1.8      | -17.2 | 0.51   | 60     |
| historical_exp07 | qs_marketobs     | 1.425  | 5.21 [3.18, 9.24]  | +2.80 [-0.77, 4.41] | 284  | 23.4  | 1.6      | -6.5  | 0.86   | 62     |
| historical_exp07 | fusion_qs_nobias | 1.374  | 4.67 [2.64, 8.52]  | +2.26 [-1.05, 3.63] | 285  | 23.5  | 1.4      | -5.9  | 1.00   | 64     |
| historical_exp07 | grw_joint        | 1.293  | 3.78 [1.94, 9.42]  | +1.36 [-0.91, 3.38] | 293  | 17.3  | 1.6      | -10.9 | 0.63   | 60     |
| historical_exp07 | market_c0        | 1.303  | 3.89 [1.98, 8.10]  | +1.48 [-1.60, 3.24] | 279  | 19.0  | 1.5      | -6.6  | 0.87   | 62     |

## Q1 — Does qs_marketobs beat m12 after commission?

**No at T−25m or T−6h. It is ahead at T−24h, but only on few bets, and not decisively.**

- **T−25m:** qs_marketobs trails m12 under every policy.
  - Point d is −1.76 (scenario_kelly) to −4.05 (canonical) ×10⁻³/day, and all five intervals include 0.
  - market_c0 is the only model with intervals that exclude 0, all in m12's favour, in 3 of 5 policies. Canonical is −7.12 [−14.68, −0.81].
  - grw_joint is the closest challenger: canonical 2.259, d −2.07.
- **T−6h:** qs_marketobs trails under every policy, by d −2.99 to −3.96.
  - Under historical_exp07 [−7.59, −0.27] and option_b_live [−6.23, −0.08] the interval excludes 0; under the circular sensitivity all five do (table below).
  - m12 is first at T−6h in every policy.
- **T−24h:** qs_marketobs has the highest wealth in every policy at both staleness bounds.
  - At 90 min it is 1.109 (78 bets) vs m12 1.052 (85 bets) under canonical, and 1.133 (81) vs 1.041 (81) under scenario Kelly.
  - At 360 min it is 1.462 (270) vs 1.163 (296) under canonical.
  - No committed noncircular interval excludes 0. The circular sensitivity excludes 0 in 3 of 5 policies at 360 min, which are the non-executable prices.
- **Per stake, the gap is turnover.** At T−25m, qs_marketobs returns as much per stake on 1X2 as m12 (18.0% vs 16.9% ROI) but stakes a third less (1X2 turnover 3.41 vs 5.14). It also does worse on Under 2.5 (12.0% on 141 bets vs 27.1% on 198). See the per-market table under Q3.

Sensitivity: qs_marketobs − m12 with the same bootstrap in numpy, noncircular (as committed) and circular. [`phase4_contrasts.csv`](results/phase4/phase4_contrasts.csv). The numpy noncircular bounds differ from the committed Julia ones by at most 0.71 ×10⁻³ ([check](results/phase4/phase4_check_vs_julia.csv)), which is Monte Carlo noise. The point estimates are identical.

|  Snapshot  |      Policy      | qs − m12 | noncircular, numpy |    circular    |
|------------|------------------|----------|--------------------|----------------|
| T-25m_s90  | canonical        | -4.05    | [-12.30, 1.74]     | [-10.92, 1.77] |
| T-25m_s90  | historical_exp07 | -3.18    | [-11.14, 2.04]     | [-9.90, 2.32]  |
| T-25m_s90  | option_b_live    | -2.55    | [-9.25, 1.82]      | [-7.97, 2.03]  |
| T-25m_s90  | plain_kelly      | -2.61    | [-11.12, 2.68]     | [-9.65, 3.00]  |
| T-25m_s90  | scenario_kelly   | -1.76    | [-11.21, 4.11]     | [-9.45, 4.52]  |
| T-6h_s90   | canonical        | -3.58    | [-7.73, 0.42]      | [-7.71, -0.17] |
| T-6h_s90   | historical_exp07 | -3.76    | [-7.78, -0.17]     | [-7.65, -0.68] |
| T-6h_s90   | option_b_live    | -2.99    | [-6.33, 0.02]      | [-6.22, -0.38] |
| T-6h_s90   | plain_kelly      | -3.96    | [-8.36, -0.19]     | [-8.18, -0.53] |
| T-6h_s90   | scenario_kelly   | -3.87    | [-7.83, 0.43]      | [-8.01, -0.30] |
| T-24h_s90  | canonical        | +1.64    | [-1.56, 3.96]      | [-1.28, 4.35]  |
| T-24h_s90  | historical_exp07 | +1.53    | [-1.73, 4.16]      | [-1.45, 4.35]  |
| T-24h_s90  | option_b_live    | +1.34    | [-1.46, 3.59]      | [-1.26, 3.73]  |
| T-24h_s90  | plain_kelly      | +2.19    | [-1.10, 4.99]      | [-0.84, 4.84]  |
| T-24h_s90  | scenario_kelly   | +2.65    | [-0.71, 4.38]      | [-0.26, 5.35]  |
| T-24h_s360 | canonical        | +3.37    | [-0.22, 4.77]      | [0.50, 5.87]   |
| T-24h_s360 | historical_exp07 | +2.80    | [-0.74, 4.43]      | [-0.15, 5.34]  |
| T-24h_s360 | option_b_live    | +2.40    | [-0.56, 3.73]      | [-0.02, 4.51]  |
| T-24h_s360 | plain_kelly      | +3.51    | [-0.57, 5.13]      | [0.22, 6.40]   |
| T-24h_s360 | scenario_kelly   | +3.11    | [-0.94, 4.72]      | [0.01, 5.83]   |

## Q2 — Is the edge larger at T−6h or T−24h than at T−25m?

**No evidence that it is.** The raw per-snapshot numbers above cannot answer this, because each early snapshot is a different, smaller fixture set. Fewer fixtures per day also means lower daily growth, so g is not comparable across snapshots.

The table below compares like with like: each model's canonical bets at the early snapshot, against its T−25m canonical bets on the same fixtures (the fixtures priced in that early book). ROI is per unit staked. Source: [`phase4_same_fixtures.csv`](results/phase4/phase4_same_fixtures.csv).

| Early snapshot |      Model       | Early fixtures / bets | Early ROI % [90%] | Early CLV pp | T−25m fixtures / bets | T−25m ROI % [90%] | T−25m CLV pp |
|----------------|------------------|-----------------------|-------------------|--------------|-----------------------|-------------------|--------------|
| T-6h_s90       | m12              | 203 / 351             | 26.3 [10.1, 48.0] | 0.54         | 261 / 539             | 24.5 [8.3, 46.0]  | 0.32         |
| T-6h_s90       | qs_marketobs     | 193 / 323             | 13.6 [2.4, 33.4]  | 0.70         | 248 / 480             | 22.8 [2.2, 42.1]  | 0.39         |
| T-6h_s90       | fusion_qs_nobias | 194 / 318             | 10.8 [-3.2, 33.8] | 0.91         | 246 / 477             | 21.9 [1.8, 42.5]  | 0.41         |
| T-6h_s90       | grw_joint        | 188 / 317             | 16.4 [2.0, 34.4]  | 0.49         | 251 / 488             | 25.4 [11.7, 39.3] | 0.34         |
| T-6h_s90       | market_c0        | 194 / 324             | 12.9 [1.8, 32.3]  | 0.86         | 252 / 475             | 16.6 [-2.5, 43.1] | 0.44         |
| T-24h_s90      | m12              | 47 / 85               | 11.5 [-1.3, 37.7] | 0.64         | 49 / 98               | 20.0 [13.0, 52.3] | 0.33         |
| T-24h_s90      | qs_marketobs     | 46 / 78               | 27.5 [15.1, 49.7] | 0.86         | 46 / 84               | 43.7 [34.3, 69.2] | 0.40         |
| T-24h_s90      | fusion_qs_nobias | 48 / 83               | 24.2 [11.2, 43.6] | 0.84         | 47 / 85               | 39.5 [22.4, 59.2] | 0.45         |
| T-24h_s90      | grw_joint        | 45 / 80               | 14.1 [-7.2, 42.5] | 0.67         | 47 / 92               | 32.3 [11.2, 62.3] | 0.33         |
| T-24h_s90      | market_c0        | 42 / 70               | 25.5 [13.2, 58.6] | 0.87         | 46 / 88               | 40.6 [33.7, 71.0] | 0.46         |
| T-24h_s360     | m12              | 166 / 296             | 10.3 [3.2, 39.3]  | 0.49         | 169 / 340             | 17.7 [3.4, 39.8]  | 0.34         |
| T-24h_s360     | qs_marketobs     | 162 / 270             | 27.6 [16.2, 47.4] | 0.94         | 163 / 308             | 36.3 [15.8, 59.7] | 0.46         |
| T-24h_s360     | fusion_qs_nobias | 157 / 268             | 29.4 [16.1, 54.5] | 1.05         | 163 / 305             | 37.1 [14.6, 64.0] | 0.46         |
| T-24h_s360     | grw_joint        | 151 / 265             | 20.3 [13.1, 53.8] | 0.74         | 163 / 319             | 28.4 [17.8, 57.0] | 0.43         |
| T-24h_s360     | market_c0        | 153 / 259             | 21.8 [9.9, 45.3]  | 1.01         | 160 / 294             | 25.2 [8.2, 54.8]  | 0.50         |

- **Every model except m12 at T−6h returns more per stake at T−25m than at the earlier snapshot on the same fixtures.** For qs_marketobs:
  - T−6h fixtures: 13.6% on 323 bets early vs 22.8% on 480 at T−25m.
  - T−24h/90 fixtures: 27.5% on 78 vs 43.7% on 84.
  - T−24h/360 fixtures: 27.6% on 270 vs 36.3% on 308.
- **The T−24h fixtures favour the market-informed models, not m12.** At T−25m, on the T−24h/360 fixtures against all T−25m fixtures: qs_marketobs 36.3% vs 17.2%, fusion 37.1% vs 16.0%, market_c0 25.2% vs 9.6%, but m12 17.7% vs 18.5%. So qs_marketobs's lead over m12 there is already present at T−25m: 36.3% vs 17.7%, and 43.7% vs 20.0% on the 90-min subset.
- **CLV is higher early (0.49–1.05 pp, against 0.32–0.50 pp at T−25m on the same fixtures), but that is mechanical.** An early price has more time to move before the close; the realised returns do not follow.
- **Coverage caveat.** T−24h/90 covers 47 complete-1X2 fixtures (6.6% of 710) and about 80 bets per model. T−24h/360 covers 170 (23.9%) and about 270 bets, with stale prices. Each interval rests on 32 or 68 slates.

## Q3 — What do the trust tiers add (canonical vs plain_kelly)?

**Nothing consistent.** The two policies differ in four things at once: the trust tiers, the markets (canonical drops BTTS and stakes only Under 2.5 among totals), BakerMcHale vs fractional 0.25 shrink, and the minimum stake. So this is a bundle comparison.

Paired daily differences, canonical − plain_kelly (and scenario − canonical for Q4), ×10⁻³/day, numpy bootstrap with noncircular and circular blocks. Source: [`phase4_contrasts.csv`](results/phase4/phase4_contrasts.csv).

|  Snapshot  |      Model       | canonical − plain_kelly [90%] |    circular    | scenario − canonical [90%] |   circular    |
|------------|------------------|-------------------------------|----------------|----------------------------|---------------|
| T-25m_s90  | m12              | +1.36 [-1.06, 2.18]           | [-0.49, 3.20]  | -0.79 [-1.47, 1.06]        | [-2.26, 0.85] |
| T-25m_s90  | qs_marketobs     | -0.08 [-2.85, 1.35]           | [-2.24, 2.01]  | +1.50 [0.14, 2.66]         | [0.32, 2.66]  |
| T-25m_s90  | fusion_qs_nobias | -0.47 [-3.08, 0.58]           | [-2.50, 1.63]  | +1.17 [0.32, 1.77]         | [0.45, 1.94]  |
| T-25m_s90  | grw_joint        | +0.21 [-3.46, 2.48]           | [-2.72, 2.90]  | -0.16 [-1.43, 0.86]        | [-1.28, 1.00] |
| T-25m_s90  | market_c0        | -1.42 [-3.63, 0.75]           | [-3.58, 0.51]  | +0.89 [0.04, 1.25]         | [0.14, 1.78]  |
| T-6h_s90   | m12              | +0.42 [-0.80, 0.78]           | [-0.56, 1.41]  | +0.36 [-0.26, 0.82]        | [-0.18, 0.92] |
| T-6h_s90   | qs_marketobs     | +0.71 [-0.17, 1.39]           | [-0.12, 1.59]  | +0.07 [-0.33, 0.91]        | [-0.51, 0.74] |
| T-6h_s90   | fusion_qs_nobias | +0.50 [-0.31, 1.05]           | [-0.32, 1.36]  | +0.15 [-0.27, 0.77]        | [-0.31, 0.68] |
| T-6h_s90   | grw_joint        | -0.03 [-1.21, 1.02]           | [-1.20, 1.33]  | -0.41 [-0.86, 0.12]        | [-0.91, 0.04] |
| T-6h_s90   | market_c0        | +0.65 [0.04, 1.42]            | [-0.05, 1.42]  | +0.12 [-0.44, 0.76]        | [-0.46, 0.83] |
| T-24h_s90  | m12              | -0.12 [-0.50, 1.17]           | [-1.23, 0.90]  | -0.32 [-0.40, 0.37]        | [-0.81, 0.19] |
| T-24h_s90  | qs_marketobs     | -0.60 [-0.91, -0.04]          | [-1.10, -0.07] | +0.69 [0.02, 1.18]         | [0.14, 1.27]  |
| T-24h_s90  | fusion_qs_nobias | -0.20 [-0.38, 0.36]           | [-0.76, 0.35]  | +0.20 [-0.28, 0.54]        | [-0.21, 0.63] |
| T-24h_s90  | grw_joint        | -1.16 [-1.20, 0.21]           | [-2.55, 0.13]  | +0.35 [-0.48, 1.08]        | [-0.35, 1.02] |
| T-24h_s90  | market_c0        | -0.88 [-1.02, 0.22]           | [-1.81, 0.05]  | +0.05 [-0.31, 0.46]        | [-0.32, 0.45] |
| T-24h_s360 | m12              | +0.29 [-0.14, 1.17]           | [-0.38, 0.95]  | +0.47 [0.05, 1.18]         | [-0.08, 1.03] |
| T-24h_s360 | qs_marketobs     | +0.20 [-0.40, 1.14]           | [-0.48, 1.04]  | +0.21 [-0.64, 1.49]        | [-0.64, 1.28] |
| T-24h_s360 | fusion_qs_nobias | +0.45 [0.11, 1.36]            | [-0.14, 1.11]  | -0.18 [-0.58, 0.54]        | [-0.69, 0.33] |
| T-24h_s360 | grw_joint        | -0.43 [-1.09, 0.71]           | [-1.27, 0.45]  | +0.19 [-0.26, 0.63]        | [-0.24, 0.66] |
| T-24h_s360 | market_c0        | +0.04 [-0.34, 0.86]           | [-0.57, 0.66]  | +0.26 [-0.17, 0.92]        | [-0.32, 0.86] |

- **At T−25m, the tiers help m12 and hurt the market-informed models:** +1.36 (m12), −0.08 (qs_marketobs), −1.42 (market_c0). Every interval spans 0.
- **The main visible mechanism is BTTS.** plain_kelly loses on it for m12 (−12.1% on 142 bets) and qs_marketobs (−1.8% on 151). Canonical gives BTTS zero trust.
- **Early snapshots are mixed.** canonical is ahead for most models at T−6h (qs_marketobs +0.71 [−0.17, 1.39]) and behind for most at T−24h/90 (qs_marketobs −0.60 [−0.91, −0.04]).

Per-market results for three models: bets / ROI % / mean CLV pp. Source: [`phase3_markets.csv`](results/phase3/phase3_markets.csv), which also has all five models and turnover.

|  Snapshot  |    Policy     | Market |     m12 · qs_marketobs · market_c0 (bets / ROI / CLV pp)      |
|------------|---------------|--------|---------------------------------------------------------------|
| T-25m_s90  | canonical     | 1X2    | 934 / 16.9% / 0.20 · 843 / 18.0% / 0.36 · 830 / 10.5% / 0.42  |
| T-25m_s90  | canonical     | OU2.5  | 198 / 27.1% / 0.13 · 141 / 12.0% / 0.27 · 144 / 3.4% / 0.17   |
| T-25m_s90  | option_b_live | 1X2    | 929 / 13.4% / 0.19 · 868 / 14.5% / 0.35 · 854 / 10.4% / 0.41  |
| T-25m_s90  | option_b_live | OU1.5  | 45 / 10.0% / -0.39 · 47 / 13.3% / 0.23 · 51 / 11.9% / 0.21    |
| T-25m_s90  | option_b_live | OU2.5  | 198 / 23.1% / 0.10 · 144 / 12.3% / 0.22 · 145 / 4.7% / 0.21   |
| T-25m_s90  | plain_kelly   | 1X2    | 935 / 12.9% / 0.19 · 886 / 13.9% / 0.34 · 868 / 9.6% / 0.41   |
| T-25m_s90  | plain_kelly   | BTTS   | 142 / -12.1% / 0.52 · 151 / -1.8% / 0.55 · 151 / 8.7% / 0.73  |
| T-25m_s90  | plain_kelly   | OU2.5  | 291 / 20.0% / -0.02 · 271 / 10.7% / -0.04 · 283 / 5.5% / 0.00 |
| T-6h_s90   | canonical     | 1X2    | 297 / 27.4% / 0.49 · 281 / 14.1% / 0.65 · 276 / 13.4% / 0.81  |
| T-6h_s90   | canonical     | OU2.5  | 54 / 19.0% / 0.98 · 42 / 10.9% / 1.27 · 48 / 10.4% / 1.34     |
| T-6h_s90   | option_b_live | 1X2    | 305 / 23.9% / 0.45 · 294 / 12.4% / 0.60 · 291 / 13.8% / 0.69  |
| T-6h_s90   | option_b_live | OU1.5  | 5 / 8.4% / -0.50 · 9 / 24.7% / 0.32 · 10 / 18.0% / 0.88       |
| T-6h_s90   | option_b_live | OU2.5  | 63 / 24.4% / 0.87 · 47 / 7.3% / 1.30 · 52 / 10.0% / 1.31      |
| T-6h_s90   | plain_kelly   | 1X2    | 305 / 21.7% / 0.46 · 296 / 11.7% / 0.56 · 289 / 12.6% / 0.68  |
| T-6h_s90   | plain_kelly   | BTTS   | 30 / 9.2% / 1.75 · 31 / -18.9% / 2.19 · 30 / -12.6% / 2.18    |
| T-6h_s90   | plain_kelly   | OU2.5  | 101 / 12.4% / 0.65 · 89 / -1.5% / 0.73 · 95 / -8.0% / 1.13    |
| T-24h_s90  | canonical     | 1X2    | 83 / 12.7% / 0.66 · 75 / 28.8% / 0.91 · 66 / 28.7% / 0.86     |
| T-24h_s90  | canonical     | OU2.5  | 2 / -44.1% / -0.96 · 3 / 3.7% / -1.08 · 4 / -29.6% / 1.28     |
| T-24h_s90  | option_b_live | 1X2    | 84 / 13.8% / 0.69 · 80 / 27.5% / 0.81 · 75 / 30.2% / 0.77     |
| T-24h_s90  | option_b_live | OU2.5  | 3 / -35.2% / -1.08 · 3 / 14.8% / -1.08 · 4 / -20.5% / 1.28    |
| T-24h_s90  | plain_kelly   | 1X2    | 85 / 9.7% / 0.67 · 80 / 26.2% / 0.81 · 76 / 28.5% / 0.81      |
| T-24h_s90  | plain_kelly   | BTTS   | 1 / 91.1% / 10.14 · 1 / 91.1% / 10.14 · 1 / 91.1% / 10.14     |
| T-24h_s90  | plain_kelly   | OU2.5  | 4 / -9.6% / -2.80 · 3 / 20.1% / -1.08 · 4 / -23.3% / 1.28     |
| T-24h_s360 | canonical     | 1X2    | 288 / 7.6% / 0.48 · 261 / 25.8% / 0.97 · 248 / 21.2% / 1.01   |
| T-24h_s360 | canonical     | OU2.5  | 8 / 91.3% / 0.85 · 9 / 69.4% / -0.03 · 11 / 37.0% / 0.85      |
| T-24h_s360 | option_b_live | 1X2    | 290 / 7.0% / 0.50 · 274 / 21.8% / 0.89 · 268 / 18.2% / 0.87   |
| T-24h_s360 | option_b_live | OU1.5  | 2 / -44.3% / 3.04 · 2 / -95.3% / 3.04 · 2 / -97.5% / 3.04     |
| T-24h_s360 | option_b_live | OU2.5  | 8 / 88.4% / 0.85 · 10 / 64.4% / -0.08 · 11 / 41.0% / 0.85     |
| T-24h_s360 | plain_kelly   | 1X2    | 293 / 5.5% / 0.48 · 274 / 22.6% / 0.89 · 271 / 18.8% / 0.87   |
| T-24h_s360 | plain_kelly   | BTTS   | 4 / 12.8% / 2.86 · 3 / 3.0% / 5.19 · 4 / -0.1% / 3.03         |
| T-24h_s360 | plain_kelly   | OU2.5  | 13 / 65.2% / 0.87 · 12 / 28.7% / 0.72 · 13 / 9.2% / 1.41      |

## Q4 — Does scenario Kelly help against per-match allocation?

**Somewhat, and only where slates are large.** The scenario allocator solves one daily expected-log problem over shared posterior draws, instead of one problem per match. Then the canonical trust, shrink, λ and cap steps run unchanged.

| Model at T−25m | scenario − canonical | Wealth, canonical → scenario |
|---|---|---|
| qs_marketobs | +1.50 [0.14, 2.66] | 1.857 → 2.154 |
| fusion_qs_nobias | +1.17 [0.32, 1.77] | 1.669 → 1.873 |
| market_c0 | +0.89 [0.04, 1.25] | 1.371 → 1.497 |
| grw_joint | −0.16 [−1.43, 0.86] | 2.259 → 2.223 |
| m12 | −0.79 [−1.47, 1.06] | 2.773 → 2.565 |

- **It narrows the gap to m12 without closing it.** qs_marketobs − m12 under scenario Kelly is −1.76 [−11.07, 3.48].
- **It does not change the top of the ranking:** m12 2.565, grw_joint 2.223, qs_marketobs 2.154.
- **At early snapshots, every scenario − canonical interval spans 0 except two:** qs_marketobs at T−24h/90 (+0.69 [0.02, 1.18], 81 bets) and m12 at T−24h/360 (+0.47 [0.05, 1.18], 299 bets). Both are in-sample and are not corrected for the 40 comparisons.

Joint solve diagnostics ([`phase3_scenario_days.csv`](results/phase3/phase3_scenario_days.csv)):

|  Snapshot  | Days | Mean legs | Max legs | Days at 0.99 budget | KKT > 1e-3 | All converged |
|------------|-----:|----------:|---------:|---------------------|-----------:|---------------|
| T-25m_s90  | 99.0 | 35.9      | 69       | 52%                 | 246        | true          |
| T-6h_s90   | 88.0 | 11.8      | 33       | 2%                  | 9          | true          |
| T-24h_s90  | 32.0 | 4.8       | 12       | 0%                  | 0          | true          |
| T-24h_s360 | 68.0 | 8.3       | 20       | 0%                  | 0          | true          |

- **The 0.99 budget binds only at T−25m.** On about half the days the joint raw stakes sum to the 0.99 budget, because slates carry up to 69 legs. Per-match Kelly gives each match its own 0.99 budget, so its raw stakes add up to 0.76–1.12 per day on average. The joint solve is the first allocator that respects one bankroll across the whole day.
- **"KKT > 1e-3" counts days, not models: it is the total over the 5 models (255 solve-days).** Every one is a budget-binding day.
  - `kkt_residual` treats the budget as binding only within 1e-6, but the log barrier stops 1e-5 to 1e-3 short of it.
  - So the residual is reporting the budget multiplier, not a failed solve. All solves converged.

## Q5 — Is CLV positive?

**Yes, in every cell, and small.** Canonical policy below; all policies are in [`phase4_clv_intervals.csv`](results/phase4/phase4_clv_intervals.csv) and [`phase3_cells.csv`](results/phase3/phase3_cells.csv). Intervals resample week blocks as above. "Beat %" is the share of bets placed at a price longer than the fair close.

|  Snapshot  |      Model       | Bets with close |   CLV pp [90%]    |    CLV % [90%]    | Stake-weighted CLV % |      Beat %       |
|------------|------------------|----------------:|-------------------|-------------------|----------------------|-------------------|
| T-25m_s90  | m12              | 1013            | 0.19 [0.16, 0.24] | 0.67 [0.50, 0.95] | 1.25 [0.91, 2.01]    | 55.7 [53.8, 57.7] |
| T-25m_s90  | qs_marketobs     | 892             | 0.35 [0.27, 0.43] | 1.21 [0.92, 1.59] | 2.09 [1.65, 3.04]    | 58.3 [55.6, 60.7] |
| T-25m_s90  | fusion_qs_nobias | 890             | 0.36 [0.31, 0.44] | 1.19 [0.98, 1.59] | 2.26 [1.81, 3.24]    | 58.0 [55.6, 60.3] |
| T-25m_s90  | grw_joint        | 926             | 0.21 [0.16, 0.30] | 0.76 [0.55, 1.14] | 1.23 [0.80, 2.16]    | 55.3 [53.3, 57.7] |
| T-25m_s90  | market_c0        | 880             | 0.39 [0.32, 0.44] | 1.32 [1.05, 1.60] | 2.28 [1.77, 2.98]    | 59.5 [56.8, 61.2] |
| T-6h_s90   | m12              | 317             | 0.54 [0.45, 0.80] | 2.02 [1.66, 2.95] | 3.83 [2.68, 5.29]    | 58.0 [56.9, 62.2] |
| T-6h_s90   | qs_marketobs     | 296             | 0.70 [0.56, 1.02] | 2.52 [2.11, 3.46] | 3.49 [2.84, 5.52]    | 59.8 [56.8, 65.8] |
| T-6h_s90   | fusion_qs_nobias | 292             | 0.91 [0.78, 1.21] | 3.18 [2.80, 4.07] | 4.36 [3.75, 6.29]    | 63.4 [60.6, 69.8] |
| T-6h_s90   | grw_joint        | 286             | 0.49 [0.36, 0.67] | 2.05 [1.62, 2.69] | 3.49 [2.12, 4.79]    | 55.9 [53.0, 60.4] |
| T-6h_s90   | market_c0        | 293             | 0.86 [0.69, 1.14] | 2.96 [2.50, 3.77] | 4.01 [3.15, 5.67]    | 62.5 [59.4, 69.1] |
| T-24h_s90  | m12              | 84              | 0.64 [0.28, 0.78] | 3.28 [2.06, 3.92] | 5.48 [3.09, 8.24]    | 61.9 [56.2, 64.0] |
| T-24h_s90  | qs_marketobs     | 77              | 0.86 [0.45, 1.23] | 4.14 [2.86, 5.64] | 4.36 [0.86, 9.90]    | 63.6 [57.8, 64.6] |
| T-24h_s90  | fusion_qs_nobias | 82              | 0.84 [0.26, 1.40] | 4.01 [2.29, 6.04] | 5.41 [1.33, 11.89]   | 62.2 [55.4, 63.2] |
| T-24h_s90  | grw_joint        | 79              | 0.67 [0.28, 0.87] | 3.48 [2.22, 4.28] | 4.95 [1.31, 7.45]    | 59.5 [54.4, 64.1] |
| T-24h_s90  | market_c0        | 69              | 0.87 [0.45, 1.36] | 3.96 [2.52, 6.22] | 5.24 [1.22, 11.93]   | 62.3 [56.9, 63.6] |
| T-24h_s360 | m12              | 289             | 0.49 [0.13, 0.92] | 2.33 [1.07, 3.94] | 4.10 [1.64, 7.68]    | 59.9 [56.8, 64.8] |
| T-24h_s360 | qs_marketobs     | 263             | 0.94 [0.57, 1.30] | 3.79 [2.47, 5.34] | 5.91 [2.78, 8.41]    | 62.4 [57.9, 66.3] |
| T-24h_s360 | fusion_qs_nobias | 262             | 1.05 [0.60, 1.55] | 4.13 [2.57, 6.09] | 6.93 [3.32, 10.28]   | 63.7 [59.1, 68.7] |
| T-24h_s360 | grw_joint        | 259             | 0.74 [0.42, 1.06] | 3.22 [2.06, 4.63] | 4.96 [2.45, 8.14]    | 60.6 [58.3, 65.5] |
| T-24h_s360 | market_c0        | 253             | 1.01 [0.63, 1.33] | 4.03 [2.54, 5.51] | 5.85 [2.89, 8.91]    | 63.6 [59.2, 66.8] |

- **Positive everywhere, small at T−25m.**
  - Mean clv_pp is positive in 100/100 cells, and the interval excludes 0 in 99. The exception is qs_marketobs × scenario Kelly × T−24h/90: 0.43 [−0.11, 0.66], 80 bets.
  - At T−25m it is small: +0.17 to +0.40 pp, with 54–60% of bets beating the close.
- **CLV does not track profit at T−25m.**
  - The market-informed models have the highest CLV: market_c0 0.39 pp, fusion_qs_nobias 0.36, qs_marketobs 0.35. They have the lowest wealth.
  - m12 has the lowest CLV (0.19 pp) and the highest wealth.
  - A model that partly prices from the market picks bets the market later moves towards. That does not by itself give the larger edges that compound.
- **Coverage.** CLV covers the bets with a valid close: 82–91% at T−25m and 97–99% at T−24h. At T−25m, coverage by market is 1X2 96%, OU2.5 61%, BTTS 43% and OU1.5 34% ([`phase3_markets.csv`](results/phase3/phase3_markets.csv)), so plain_kelly has the lowest. Totals and BTTS CLV rests on a minority of their bets.
- **Commission is not deducted.** The T−25m CLV of 0.7–1.3% of stake is about the size of the commission, which costs 2% of net winnings: roughly 1–1.5% of stake at typical odds.

## In-sample risks and interval widths

- **In-sample.** Two seasons, 24/25 and 25/26: 99 slates at T−25m, 86–88 at T−6h, 31–32 at T−24h/90 and 67–68 at T−24h/360.
  - Experiment 07 chose the trust tiers, λ and the canonical stack on these same seasons with m12 backtests, and historical_exp07 is that choice.
  - Every policy row is therefore partly in-sample, and the bias favours m12.
  - No policy was tuned here; all 100 cells are reported.
- **Wide intervals.**

  | Snapshot | Median 90% width of g | Median 90% width of d vs m12 | Median \|d\| |
  |---|---|---|---|
  | T−25m | 7.9 | 13.2 | 3.2 |
  | T−6h | 5.2 | 7.4 | 3.6 |
  | T−24h/90 | 3.7 | 5.9 | 1.2 |
  | T−24h/360 | 6.3 | 4.9 | 2.2 |

  Units are ×10⁻³/day. Of the 80 model-vs-m12 intervals, none favours the challenger and 12 favour m12 (3 at T−25m, 9 at T−6h). Of the 40 policy contrasts, 8 exclude 0, about twice the 4 expected by chance at 90%.
- **Uneven intervals.** The prescribed noncircular blocks under-sample the first and last weeks of each season. Intervals can therefore sit lopsided around the point estimate. An example is grw_joint canonical − plain_kelly at T−24h/90: −1.16 [−1.20, 0.21]. The circular columns are better centred and change no answer. They sharpen two of them: at T−6h all five qs_marketobs − m12 intervals exclude 0, and at T−24h/360 three of five do, on non-executable prices.
- **Prices.** Last-traded archive prices, not executable back/lay touch or depth. Even the 90-min snapshots can be up to 90 minutes stale (T−25m median 4 min, p95 62 min). Fills are not modelled.
- **Multiplicity.** 100 cells, 80 paired m12 comparisons and 40 policy contrasts, with no correction. Read single intervals that exclude 0 as leads, not findings.

## Traceability

| Item | Location |
|---|---|
| Runner | `l02_phase3_grid.jl` + `r02_phase3_grid.jl` (`4f1d3d43`) on mcmc-beast; log [`logs/phase3_grid.log`](logs/phase3_grid.log) |
| Phase 3 results | [`results/phase3/`](results/phase3/): cells, markets, daily, bootstrap, paired-vs-m12, scenario days, bets ledger with CLV (45,191 bets), gate, DB audit |
| Phase 4 tables | [`x04_phase4_tables.py`](x04_phase4_tables.py), local, reading the committed Phase 3 CSVs only (`uv run …`); outputs in [`results/phase4/`](results/phase4/) |
| Tables in this file | duckdb queries over those CSVs; every number above is a row or an aggregate of rows there |
| Earlier phases | [PHASE1_ALLOCATOR_AUDIT.md](PHASE1_ALLOCATOR_AUDIT.md), [PHASE2_SCENARIO_KELLY.md](PHASE2_SCENARIO_KELLY.md), [HISTORICAL_REPRODUCTION_RECIPE.md](HISTORICAL_REPRODUCTION_RECIPE.md), [READ_PATH_AUDIT.md](READ_PATH_AUDIT.md) |
| Decisions | [DECISIONS_LOG.md](DECISIONS_LOG.md) |

## Follow-ups

1. **`kkt_residual` budget tolerance.** It treats the budget as binding only within 1e-6, while the barrier stops up to 1e-3 short. The tolerance should scale with `barrier_mu`, or the check should use the barrier's own multiplier μ/(B − Σa). It is a diagnostic only; no stake changes.
2. **Bootstrap centring.** Switch the experiment-wide scheme to circular (or stationary) blocks, or report both, so that intervals sit around their point estimates.
3. **Executable prices.** Rerun the T−25m and T−6h cells with back/lay touch and depth from the order-book archive (26/27 onwards), instead of last-traded prices.
4. **CLV coverage.** 9–18% of T−25m bets have no valid close, and most of them are totals and BTTS (61% and 43% coverage, against 1X2 96%), because of the complete-market and overround filters. Report CLV on a fixed complete-close fixture set, or widen the close window; never impute.
5. **Out-of-sample season.** Freeze one policy per model and score 26/27 as it arrives; that is the first test without experiment 07's selection on these seasons.
6. **Scenario Kelly for the market-informed models.** It is the only intervention that excluded 0 for three models at T−25m. Confirm it out of sample before any production change. src is untouched and the policy steps after the joint solve are unchanged.
