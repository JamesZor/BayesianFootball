# r08 trust-pruning sweep — Task 016 (1-parameter smile spine)

Generated 2026-09-17 22:18 at `c992ada` on mcmc-beast. Arms: `m05_joint_grw_smile_spine_w040` (test), `m05_joint_grw_smile_supremacy_w040` (five-strike reference) and `m05_joint_grw_baseline` (no-smile control). Additions at trust 0.7143 on Option B's book plus O/U 4.5; risk, cap, filter and grouping are Option B's. Every smile container is staked through the anti-diagonal reweighted grid (T011), so these rows are not comparable with Task 015's r08.

### r04's prediction, recorded before this sweep ran

At Under 1.5 the spine prices 0.2715 against a market of 0.2305 and a realised rate of 0.2279 — a +4.1 pp edge where there is none — so `+U1.5` should lose money, and lose more than the five-strike arm (+1.7 pp) or the baseline (+1.2 pp). At Under 4.5 the spine prices 0.8078 below a market of 0.8578 while the realised rate is 0.9038, so it should decline the bet rather than gain from it.

## Gates

| gate | environment | model | pass | detail |
|---|---|---|---:|---|
| S0 extended book inert at trust 0 | close | m05_joint_grw_baseline | false | ext return 368.8441 vs canon 385.7776 (Δ -16.9335 pp), bets 1246 vs 1247 |
| S0 extended book inert at trust 0 | close | m05_joint_grw_smile_supremacy_w040 | false | ext return 536.6523 vs canon 553.4909 (Δ -16.8386 pp), bets 1253 vs 1250 |
| S2 smile routing + staking, all strikes | close | m05_joint_grw_smile_supremacy_w040 | true | 880 totals bets, max |p − smile| 2.3e-15, stake-side gap 3.55e-15 |
| S0 extended book inert at trust 0 | close | m05_joint_grw_smile_spine_w040 | false | ext return 449.8734 vs canon 469.3535 (Δ -19.4801 pp), bets 1247 vs 1232 |
| S2 smile routing + staking, all strikes | close | m05_joint_grw_smile_spine_w040 | true | 980 totals bets, max |p − smile| 3.0e-15, stake-side gap 3.00e-15 |
| S0 extended book inert at trust 0 | t25 | m05_joint_grw_baseline | false | ext return 509.5218 vs canon 531.7811 (Δ -22.2593 pp), bets 1123 vs 1124 |
| S0 extended book inert at trust 0 | t25 | m05_joint_grw_smile_supremacy_w040 | false | ext return 382.3781 vs canon 381.2263 (Δ +1.1518 pp), bets 1117 vs 1112 |
| S2 smile routing + staking, all strikes | t25 | m05_joint_grw_smile_supremacy_w040 | true | 624 totals bets, max |p − smile| 2.4e-15, stake-side gap 3.55e-15 |
| S0 extended book inert at trust 0 | t25 | m05_joint_grw_smile_spine_w040 | false | ext return 467.9181 vs canon 457.6981 (Δ +10.2200 pp), bets 1111 vs 1104 |
| S2 smile routing + staking, all strikes | t25 | m05_joint_grw_smile_spine_w040 | true | 690 totals bets, max |p − smile| 2.4e-15, stake-side gap 3.00e-15 |
| S1 close/P0 vs r06 | close | m05_joint_grw_baseline | true | return 385.7776 vs 385.7776, bets 1247 vs 1247 |
| S1 close/P0 vs r06 | close | m05_joint_grw_smile_supremacy_w040 | true | return 553.4909 vs 553.4909, bets 1250 vs 1250 |
| S1 close/P0 vs r06 | close | m05_joint_grw_smile_spine_w040 | true | return 469.3535 vs 469.3535, bets 1232 vs 1232 |

## Quoted fixtures per line (buildable panel)

| market_name | market_line | n_fixtures | environment |
|---|---:|---:|---|
| 1X2 | 0.0000 | 596 | close |
| BTTS | 0.0000 | 178 | close |
| CorrectScore | 0.0000 | 62 | close |
| OverUnder | 0.5000 | 387 | close |
| OverUnder | 1.5000 | 217 | close |
| OverUnder | 2.5000 | 380 | close |
| OverUnder | 3.5000 | 266 | close |
| OverUnder | 4.5000 | 106 | close |
| OverUnder | 5.5000 | 111 | close |
| 1X2 | 0.0000 | 545 | t25 |
| BTTS | 0.0000 | 190 | t25 |
| OverUnder | 0.5000 | 76 | t25 |
| OverUnder | 1.5000 | 166 | t25 |
| OverUnder | 2.5000 | 347 | t25 |
| OverUnder | 3.5000 | 179 | t25 |
| OverUnder | 4.5000 | 47 | t25 |
| OverUnder | 5.5000 | 39 | t25 |

## H5 — the two named lines

| environment | policy | model | added_n_bets | added_win_rate_pct | added_roi_pct | added_stake_share_pct | added_edge_pp | delta_return_pp | core_stake_vs_p0 | delta_core_roi_pp | roi_pct | sharpe_ann |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | +U1.5 | m05_joint_grw_baseline | 105 | 23.81 | 37.66 | 3.51 | 4.54 | +119.29 | 0.982 | +0.12 | 12.52 | 1.680 |
| close | +U1.5 | m05_joint_grw_smile_spine_w040 | 208 | 21.63 | -6.31 | 9.96 | 4.28 | -91.57 | 0.954 | -0.16 | 12.05 | 1.348 |
| close | +U1.5 | m05_joint_grw_smile_supremacy_w040 | 128 | 19.53 | -2.73 | 4.05 | 3.03 | -28.58 | 0.983 | -0.06 | 14.68 | 1.615 |
| close | +U4.5 | m05_joint_grw_baseline | 28 | 96.43 | 18.79 | 3.03 | 4.49 | +56.19 | 0.989 | +0.17 | 11.87 | 1.551 |
| close | +U4.5 | m05_joint_grw_smile_spine_w040 | 0 | n/a | n/a | 0.00 | n/a | +0.00 | 1.000 | +0.00 | 14.24 | 1.442 |
| close | +U4.5 | m05_joint_grw_smile_supremacy_w040 | 0 | n/a | n/a | 0.00 | n/a | +0.00 | 1.000 | +0.00 | 15.47 | 1.617 |
| t25 | +U1.5 | m05_joint_grw_baseline | 76 | 23.68 | 3.90 | 3.11 | 4.86 | +10.24 | 0.991 | +0.13 | 13.76 | 1.627 |
| t25 | +U1.5 | m05_joint_grw_smile_spine_w040 | 152 | 19.74 | -14.63 | 8.34 | 4.29 | -92.94 | 0.980 | +0.35 | 14.06 | 1.162 |
| t25 | +U1.5 | m05_joint_grw_smile_supremacy_w040 | 99 | 14.14 | -6.80 | 3.16 | 2.65 | -14.28 | 0.992 | +0.08 | 14.49 | 1.200 |
| t25 | +U4.5 | m05_joint_grw_baseline | 23 | 95.65 | 10.43 | 3.64 | 4.84 | +13.94 | 0.991 | -0.16 | 13.67 | 1.654 |
| t25 | +U4.5 | m05_joint_grw_smile_spine_w040 | 1 | 100.00 | 35.28 | 0.03 | 9.58 | +0.44 | 1.000 | -0.00 | 16.32 | 1.298 |
| t25 | +U4.5 | m05_joint_grw_smile_supremacy_w040 | 2 | 100.00 | 36.89 | 0.10 | 9.22 | +1.93 | 1.000 | -0.00 | 15.13 | 1.217 |

## Full sweep

| environment | model | policy | n_bets | total_return_pct | delta_return_pp | roi_pct | sharpe_ann | max_drawdown_pct | n_capped | added_n_bets | added_win_rate_pct | added_roi_pct | added_stake_share_pct | core_stake_vs_p0 | core_roi_pct | delta_core_roi_pp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | m05_joint_grw_baseline | P0 Option B | 1247 | 385.78 | +0.00 | 11.68 | 1.453 | -42.67 | 27 | 0 | n/a | n/a | 0.00 | 1.000 | 11.68 | +0.00 |
| close | m05_joint_grw_baseline | P0 ext book | 1246 | 368.84 | +0.00 | 11.49 | 1.417 | -44.19 | 27 | 0 | n/a | n/a | 0.00 | 1.000 | 11.49 | +0.00 |
| close | m05_joint_grw_baseline | +U0.5 | 1327 | 311.74 | -57.10 | 10.57 | 1.309 | -45.65 | 28 | 81 | 3.70 | -63.62 | 1.20 | 0.996 | 11.47 | -0.02 |
| close | m05_joint_grw_baseline | +U1.5 | 1351 | 488.13 | +119.29 | 12.52 | 1.680 | -39.72 | 32 | 105 | 23.81 | 37.66 | 3.51 | 0.982 | 11.61 | +0.12 |
| close | m05_joint_grw_baseline | +U3.5 | 1345 | 389.78 | +20.93 | 11.23 | 1.502 | -39.79 | 34 | 99 | 74.75 | 2.43 | 7.00 | 0.963 | 11.89 | +0.41 |
| close | m05_joint_grw_baseline | +U4.5 | 1274 | 425.03 | +56.19 | 11.87 | 1.551 | -41.16 | 32 | 28 | 96.43 | 18.79 | 3.03 | 0.989 | 11.65 | +0.17 |
| close | m05_joint_grw_baseline | +O2.5 | 1350 | 346.21 | -22.64 | 10.77 | 1.362 | -45.35 | 35 | 104 | 50.96 | 0.89 | 6.33 | 0.976 | 11.44 | -0.05 |
| close | m05_joint_grw_baseline | +O3.5 | 1366 | 345.81 | -23.03 | 10.66 | 1.343 | -44.82 | 39 | 120 | 36.67 | 2.94 | 7.49 | 0.978 | 11.29 | -0.20 |
| close | m05_joint_grw_baseline | +O4.5 | 1300 | 311.14 | -57.71 | 10.58 | 1.272 | -46.58 | 27 | 54 | 9.26 | -30.85 | 1.99 | 0.996 | 11.42 | -0.06 |
| close | m05_joint_grw_baseline | +BTTS_yes | 1298 | 339.93 | -28.91 | 10.93 | 1.357 | -43.77 | 34 | 52 | 55.77 | 2.03 | 2.84 | 0.987 | 11.19 | -0.29 |
| close | m05_joint_grw_baseline | +BTTS_no | 1315 | 347.57 | -21.27 | 11.05 | 1.369 | -44.28 | 30 | 69 | 42.03 | -7.06 | 2.87 | 0.987 | 11.58 | +0.10 |
| close | m05_joint_grw_baseline | +all_unders | 1559 | 487.78 | +118.93 | 11.70 | 1.770 | -34.25 | 42 | 313 | 41.21 | 8.46 | 13.33 | 0.929 | 12.20 | +0.71 |
| close | m05_joint_grw_baseline | +all_fringe | 1958 | 299.00 | -69.84 | 8.87 | 1.359 | -32.56 | 58 | 712 | 40.59 | 1.67 | 27.94 | 0.831 | 11.67 | +0.18 |
| close | m05_joint_grw_smile_supremacy_w040 | P0 Option B | 1250 | 553.49 | +0.00 | 15.72 | 1.640 | -39.08 | 16 | 0 | n/a | n/a | 0.00 | 1.000 | 15.72 | +0.00 |
| close | m05_joint_grw_smile_supremacy_w040 | P0 ext book | 1253 | 536.65 | +0.00 | 15.47 | 1.617 | -38.44 | 16 | 0 | n/a | n/a | 0.00 | 1.000 | 15.47 | +0.00 |
| close | m05_joint_grw_smile_supremacy_w040 | +U0.5 | 1402 | 251.46 | -285.19 | 10.47 | 1.111 | -32.93 | 24 | 149 | 3.36 | -59.26 | 6.31 | 1.009 | 15.16 | -0.31 |
| close | m05_joint_grw_smile_supremacy_w040 | +U1.5 | 1381 | 508.07 | -28.58 | 14.68 | 1.615 | -35.27 | 20 | 128 | 19.53 | -2.73 | 4.05 | 0.983 | 15.41 | -0.06 |
| close | m05_joint_grw_smile_supremacy_w040 | +U3.5 | 1303 | 579.20 | +42.55 | 15.63 | 1.700 | -36.38 | 18 | 50 | 68.00 | 13.62 | 2.33 | 0.990 | 15.68 | +0.21 |
| close | m05_joint_grw_smile_supremacy_w040 | +U4.5 | 1253 | 536.65 | +0.00 | 15.47 | 1.617 | -38.44 | 16 | 0 | n/a | n/a | 0.00 | 1.000 | 15.47 | +0.00 |
| close | m05_joint_grw_smile_supremacy_w040 | +O2.5 | 1349 | 520.90 | -15.75 | 14.73 | 1.586 | -38.61 | 20 | 96 | 48.96 | 6.16 | 4.79 | 0.991 | 15.16 | -0.32 |
| close | m05_joint_grw_smile_supremacy_w040 | +O3.5 | 1414 | 488.23 | -48.42 | 14.07 | 1.541 | -38.92 | 19 | 161 | 30.43 | -4.38 | 6.42 | 0.994 | 15.33 | -0.14 |
| close | m05_joint_grw_smile_supremacy_w040 | +O4.5 | 1346 | 529.56 | -7.09 | 14.91 | 1.567 | -40.06 | 23 | 93 | 9.68 | 6.99 | 4.50 | 0.995 | 15.28 | -0.19 |
| close | m05_joint_grw_smile_supremacy_w040 | +BTTS_yes | 1319 | 525.27 | -11.38 | 15.05 | 1.590 | -38.00 | 20 | 66 | 50.00 | -1.71 | 3.07 | 0.991 | 15.59 | +0.11 |
| close | m05_joint_grw_smile_supremacy_w040 | +BTTS_no | 1317 | 537.04 | +0.39 | 15.12 | 1.623 | -38.06 | 18 | 64 | 39.06 | -0.16 | 2.65 | 0.995 | 15.54 | +0.07 |
| close | m05_joint_grw_smile_supremacy_w040 | +all_unders | 1580 | 235.89 | -300.76 | 9.81 | 1.095 | -31.35 | 29 | 327 | 19.57 | -29.61 | 11.68 | 0.976 | 15.02 | -0.45 |
| close | m05_joint_grw_smile_supremacy_w040 | +all_fringe | 2060 | 183.88 | -352.78 | 7.84 | 0.918 | -32.46 | 49 | 807 | 28.13 | -9.37 | 26.78 | 0.915 | 14.13 | -1.34 |
| close | m05_joint_grw_smile_spine_w040 | P0 Option B | 1232 | 469.35 | +0.00 | 14.54 | 1.462 | -41.63 | 20 | 0 | n/a | n/a | 0.00 | 1.000 | 14.54 | +0.00 |
| close | m05_joint_grw_smile_spine_w040 | P0 ext book | 1247 | 449.87 | +0.00 | 14.24 | 1.442 | -40.65 | 22 | 0 | n/a | n/a | 0.00 | 1.000 | 14.24 | +0.00 |
| close | m05_joint_grw_smile_spine_w040 | +U0.5 | 1392 | 287.72 | -162.15 | 11.33 | 1.151 | -36.38 | 24 | 145 | 3.45 | -61.96 | 3.80 | 1.005 | 14.23 | -0.01 |
| close | m05_joint_grw_smile_spine_w040 | +U1.5 | 1455 | 358.30 | -91.57 | 12.05 | 1.348 | -35.14 | 28 | 208 | 21.63 | -6.31 | 9.96 | 0.954 | 14.08 | -0.16 |
| close | m05_joint_grw_smile_spine_w040 | +U3.5 | 1276 | 467.92 | +18.04 | 14.37 | 1.476 | -40.11 | 22 | 29 | 75.86 | 22.27 | 0.99 | 0.994 | 14.30 | +0.05 |
| close | m05_joint_grw_smile_spine_w040 | +U4.5 | 1247 | 449.87 | +0.00 | 14.24 | 1.442 | -40.65 | 22 | 0 | n/a | n/a | 0.00 | 1.000 | 14.24 | +0.00 |
| close | m05_joint_grw_smile_spine_w040 | +O2.5 | 1296 | 473.25 | +23.37 | 14.27 | 1.472 | -40.65 | 22 | 49 | 51.02 | 16.47 | 2.33 | 0.995 | 14.22 | -0.03 |
| close | m05_joint_grw_smile_spine_w040 | +O3.5 | 1453 | 402.14 | -47.73 | 12.41 | 1.356 | -40.04 | 34 | 206 | 30.58 | -1.80 | 9.76 | 0.993 | 13.94 | -0.30 |
| close | m05_joint_grw_smile_spine_w040 | +O4.5 | 1342 | 416.24 | -33.63 | 13.27 | 1.353 | -42.56 | 27 | 95 | 9.47 | 2.37 | 5.54 | 0.993 | 13.91 | -0.33 |
| close | m05_joint_grw_smile_spine_w040 | +BTTS_yes | 1299 | 458.93 | +9.05 | 14.14 | 1.461 | -39.71 | 24 | 52 | 57.69 | 9.58 | 2.52 | 0.988 | 14.26 | +0.01 |
| close | m05_joint_grw_smile_spine_w040 | +BTTS_no | 1312 | 445.86 | -4.01 | 13.85 | 1.433 | -40.17 | 24 | 65 | 40.00 | -0.48 | 3.22 | 0.992 | 14.33 | +0.08 |
| close | m05_joint_grw_smile_spine_w040 | +all_unders | 1629 | 222.68 | -227.19 | 9.48 | 1.034 | -37.81 | 31 | 382 | 18.85 | -18.90 | 13.74 | 0.946 | 14.00 | -0.24 |
| close | m05_joint_grw_smile_spine_w040 | +all_fringe | 2096 | 175.24 | -274.64 | 7.44 | 0.885 | -40.97 | 49 | 849 | 26.50 | -5.46 | 29.14 | 0.892 | 12.74 | -1.50 |
| t25 | m05_joint_grw_baseline | P0 Option B | 1124 | 531.78 | +0.00 | 14.15 | 1.658 | -41.86 | 21 | 0 | n/a | n/a | 0.00 | 1.000 | 14.15 | +0.00 |
| t25 | m05_joint_grw_baseline | P0 ext book | 1123 | 509.52 | +0.00 | 13.95 | 1.618 | -41.82 | 21 | 0 | n/a | n/a | 0.00 | 1.000 | 13.95 | +0.00 |
| t25 | m05_joint_grw_baseline | +U0.5 | 1160 | 450.88 | -58.65 | 13.20 | 1.533 | -41.89 | 21 | 37 | 2.70 | -89.53 | 0.71 | 0.999 | 13.93 | -0.02 |
| t25 | m05_joint_grw_baseline | +U1.5 | 1199 | 519.76 | +10.24 | 13.76 | 1.627 | -39.31 | 24 | 76 | 23.68 | 3.90 | 3.11 | 0.991 | 14.08 | +0.13 |
| t25 | m05_joint_grw_baseline | +U3.5 | 1190 | 426.59 | -82.94 | 12.41 | 1.525 | -40.17 | 25 | 67 | 56.72 | -15.00 | 6.16 | 0.974 | 14.21 | +0.26 |
| t25 | m05_joint_grw_baseline | +U4.5 | 1146 | 523.46 | +13.94 | 13.67 | 1.654 | -41.62 | 22 | 23 | 95.65 | 10.43 | 3.64 | 0.991 | 13.79 | -0.16 |
| t25 | m05_joint_grw_baseline | +O2.5 | 1239 | 562.16 | +52.64 | 13.63 | 1.685 | -41.30 | 29 | 116 | 58.62 | 10.77 | 8.55 | 0.974 | 13.89 | -0.06 |
| t25 | m05_joint_grw_baseline | +O3.5 | 1198 | 420.77 | -88.75 | 12.41 | 1.487 | -38.82 | 26 | 75 | 26.67 | -23.70 | 4.70 | 0.990 | 14.19 | +0.23 |
| t25 | m05_joint_grw_baseline | +O4.5 | 1144 | 509.18 | -0.34 | 13.85 | 1.618 | -41.82 | 22 | 21 | 4.76 | -9.29 | 0.68 | 1.000 | 14.01 | +0.06 |
| t25 | m05_joint_grw_baseline | +BTTS_yes | 1196 | 455.75 | -53.78 | 12.92 | 1.524 | -40.33 | 25 | 73 | 60.27 | -12.37 | 4.39 | 0.990 | 14.09 | +0.13 |
| t25 | m05_joint_grw_baseline | +BTTS_no | 1205 | 590.61 | +81.09 | 14.41 | 1.725 | -40.85 | 23 | 82 | 50.00 | 13.02 | 4.05 | 0.984 | 14.47 | +0.52 |
| t25 | m05_joint_grw_baseline | +all_unders | 1326 | 380.12 | -129.40 | 11.28 | 1.445 | -38.16 | 27 | 203 | 38.92 | -8.90 | 12.42 | 0.953 | 14.14 | +0.19 |
| t25 | m05_joint_grw_baseline | +all_fringe | 1693 | 350.31 | -159.21 | 9.86 | 1.418 | -34.71 | 51 | 570 | 44.39 | -3.32 | 27.26 | 0.867 | 14.80 | +0.85 |
| t25 | m05_joint_grw_smile_supremacy_w040 | P0 Option B | 1112 | 381.23 | +0.00 | 15.14 | 1.212 | -42.15 | 12 | 0 | n/a | n/a | 0.00 | 1.000 | 15.14 | +0.00 |
| t25 | m05_joint_grw_smile_supremacy_w040 | P0 ext book | 1117 | 382.38 | +0.00 | 15.11 | 1.214 | -42.20 | 12 | 0 | n/a | n/a | 0.00 | 1.000 | 15.11 | +0.00 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U0.5 | 1192 | 290.90 | -91.48 | 13.05 | 1.059 | -43.13 | 16 | 75 | 6.67 | -42.15 | 3.47 | 0.999 | 15.03 | -0.08 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U1.5 | 1216 | 368.10 | -14.28 | 14.49 | 1.200 | -40.51 | 14 | 99 | 14.14 | -6.80 | 3.16 | 0.992 | 15.19 | +0.08 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U3.5 | 1159 | 356.58 | -25.80 | 14.40 | 1.190 | -41.24 | 15 | 42 | 57.14 | -11.37 | 2.42 | 0.989 | 15.04 | -0.07 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U4.5 | 1119 | 384.31 | +1.93 | 15.13 | 1.217 | -42.20 | 12 | 2 | 100.00 | 36.89 | 0.10 | 1.000 | 15.10 | -0.00 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +O2.5 | 1232 | 390.70 | +8.32 | 14.48 | 1.234 | -42.51 | 14 | 115 | 52.17 | 9.25 | 6.45 | 0.982 | 14.84 | -0.26 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +O3.5 | 1211 | 388.17 | +5.79 | 14.57 | 1.231 | -41.22 | 14 | 94 | 27.66 | 0.40 | 3.78 | 1.000 | 15.13 | +0.02 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +O4.5 | 1160 | 330.64 | -51.74 | 14.07 | 1.120 | -42.28 | 14 | 43 | 2.33 | -41.28 | 1.76 | 0.999 | 15.06 | -0.05 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +BTTS_yes | 1171 | 305.88 | -76.50 | 13.41 | 1.082 | -42.06 | 13 | 54 | 50.00 | -45.28 | 3.65 | 0.993 | 15.63 | +0.52 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +BTTS_no | 1198 | 439.13 | +56.75 | 15.41 | 1.307 | -43.03 | 14 | 81 | 45.68 | 18.09 | 4.16 | 0.988 | 15.30 | +0.19 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +all_unders | 1335 | 254.33 | -128.04 | 11.82 | 1.003 | -40.66 | 20 | 218 | 20.64 | -21.68 | 8.63 | 0.977 | 14.99 | -0.12 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +all_fringe | 1722 | 210.31 | -172.07 | 9.52 | 0.899 | -41.31 | 34 | 605 | 32.40 | -9.35 | 23.45 | 0.936 | 15.30 | +0.20 |
| t25 | m05_joint_grw_smile_spine_w040 | P0 Option B | 1104 | 457.70 | +0.00 | 16.28 | 1.283 | -43.82 | 10 | 0 | n/a | n/a | 0.00 | 1.000 | 16.28 | +0.00 |
| t25 | m05_joint_grw_smile_spine_w040 | P0 ext book | 1111 | 467.92 | +0.00 | 16.32 | 1.297 | -43.88 | 11 | 0 | n/a | n/a | 0.00 | 1.000 | 16.32 | +0.00 |
| t25 | m05_joint_grw_smile_spine_w040 | +U0.5 | 1184 | 395.26 | -72.66 | 14.90 | 1.210 | -44.51 | 12 | 73 | 5.48 | -49.27 | 2.08 | 1.001 | 16.26 | -0.06 |
| t25 | m05_joint_grw_smile_spine_w040 | +U1.5 | 1263 | 374.97 | -92.94 | 14.06 | 1.162 | -41.13 | 16 | 152 | 19.74 | -14.63 | 8.34 | 0.980 | 16.68 | +0.35 |
| t25 | m05_joint_grw_smile_spine_w040 | +U3.5 | 1136 | 452.76 | -15.16 | 15.92 | 1.291 | -43.12 | 13 | 25 | 56.00 | -4.37 | 1.35 | 0.994 | 16.20 | -0.13 |
| t25 | m05_joint_grw_smile_spine_w040 | +U4.5 | 1112 | 468.36 | +0.44 | 16.32 | 1.298 | -43.88 | 11 | 1 | 100.00 | 35.28 | 0.03 | 1.000 | 16.32 | -0.00 |
| t25 | m05_joint_grw_smile_spine_w040 | +O2.5 | 1175 | 486.35 | +18.43 | 16.16 | 1.323 | -43.88 | 14 | 64 | 54.69 | 15.56 | 3.22 | 0.991 | 16.18 | -0.14 |
| t25 | m05_joint_grw_smile_spine_w040 | +O3.5 | 1227 | 481.38 | +13.46 | 15.18 | 1.325 | -41.92 | 20 | 116 | 27.59 | -7.88 | 6.67 | 1.010 | 16.82 | +0.50 |
| t25 | m05_joint_grw_smile_spine_w040 | +O4.5 | 1157 | 373.27 | -94.65 | 14.67 | 1.146 | -43.99 | 13 | 46 | 4.35 | -49.41 | 2.50 | 1.003 | 16.31 | -0.01 |
| t25 | m05_joint_grw_smile_spine_w040 | +BTTS_yes | 1157 | 402.33 | -65.59 | 15.00 | 1.208 | -43.48 | 12 | 46 | 50.00 | -38.42 | 3.30 | 0.993 | 16.82 | +0.50 |
| t25 | m05_joint_grw_smile_spine_w040 | +BTTS_no | 1208 | 535.12 | +67.21 | 16.38 | 1.392 | -44.53 | 13 | 97 | 40.21 | 11.47 | 5.20 | 0.988 | 16.65 | +0.33 |
| t25 | m05_joint_grw_smile_spine_w040 | +all_unders | 1362 | 294.16 | -173.75 | 12.37 | 1.051 | -41.20 | 20 | 251 | 19.52 | -19.33 | 11.21 | 0.968 | 16.37 | +0.05 |
| t25 | m05_joint_grw_smile_spine_w040 | +all_fringe | 1731 | 231.02 | -236.90 | 9.62 | 0.936 | -41.52 | 43 | 620 | 29.03 | -11.59 | 25.75 | 0.930 | 16.98 | +0.66 |

## Arm vs arm under each policy (paired slate log growth)

| environment | policy | pair | a_return_pct | b_return_pct | a_roi_pct | b_roi_pct | n_slates | delta_log_growth_per_slate | lo | hi | p_better |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | P0 Option B | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 469.35 | 385.78 | 14.54 | 11.68 | 100 | +0.00159 | -0.00948 | +0.01323 | 0.611 |
| close | P0 Option B | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 469.35 | 553.49 | 14.54 | 15.72 | 100 | -0.00138 | -0.00590 | +0.00311 | 0.279 |
| close | P0 Option B | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 553.49 | 385.78 | 15.72 | 11.68 | 100 | +0.00297 | -0.00767 | +0.01429 | 0.707 |
| close | +U0.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 287.72 | 311.74 | 11.33 | 10.57 | 100 | -0.00060 | -0.01179 | +0.01118 | 0.464 |
| close | +U0.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 287.72 | 251.46 | 11.33 | 10.47 | 100 | +0.00098 | -0.00333 | +0.00539 | 0.665 |
| close | +U0.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 251.46 | 311.74 | 10.47 | 10.57 | 100 | -0.00158 | -0.01255 | +0.00987 | 0.398 |
| close | +U1.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 358.30 | 488.13 | 12.05 | 12.52 | 100 | -0.00249 | -0.01386 | +0.00945 | 0.342 |
| close | +U1.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 358.30 | 508.07 | 12.05 | 14.68 | 100 | -0.00283 | -0.00773 | +0.00200 | 0.129 |
| close | +U1.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 508.07 | 488.13 | 14.68 | 12.52 | 100 | +0.00033 | -0.01043 | +0.01159 | 0.532 |
| close | +U3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 467.92 | 389.78 | 14.37 | 11.23 | 100 | +0.00148 | -0.00972 | +0.01325 | 0.601 |
| close | +U3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 467.92 | 579.20 | 14.37 | 15.63 | 100 | -0.00179 | -0.00610 | +0.00237 | 0.205 |
| close | +U3.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 579.20 | 389.78 | 15.63 | 11.23 | 100 | +0.00327 | -0.00762 | +0.01458 | 0.726 |
| close | +U4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 449.87 | 425.03 | 14.24 | 11.87 | 100 | +0.00046 | -0.01045 | +0.01174 | 0.535 |
| close | +U4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 449.87 | 536.65 | 14.24 | 15.47 | 100 | -0.00147 | -0.00584 | +0.00277 | 0.254 |
| close | +U4.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 536.65 | 425.03 | 15.47 | 11.87 | 100 | +0.00193 | -0.00847 | +0.01294 | 0.643 |
| close | +O2.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 473.25 | 346.21 | 14.27 | 10.77 | 100 | +0.00251 | -0.00928 | +0.01490 | 0.657 |
| close | +O2.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 473.25 | 520.90 | 14.27 | 14.73 | 100 | -0.00080 | -0.00524 | +0.00345 | 0.361 |
| close | +O2.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 520.90 | 346.21 | 14.73 | 10.77 | 100 | +0.00330 | -0.00836 | +0.01557 | 0.712 |
| close | +O3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 402.14 | 345.81 | 12.41 | 10.66 | 100 | +0.00119 | -0.01056 | +0.01381 | 0.574 |
| close | +O3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 402.14 | 488.23 | 12.41 | 14.07 | 100 | -0.00158 | -0.00607 | +0.00283 | 0.242 |
| close | +O3.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 488.23 | 345.81 | 14.07 | 10.66 | 100 | +0.00277 | -0.00868 | +0.01492 | 0.680 |
| close | +O4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 416.24 | 311.14 | 13.27 | 10.58 | 100 | +0.00228 | -0.00918 | +0.01466 | 0.649 |
| close | +O4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 416.24 | 529.56 | 13.27 | 14.91 | 100 | -0.00198 | -0.00634 | +0.00225 | 0.187 |
| close | +O4.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 529.56 | 311.14 | 14.91 | 10.58 | 100 | +0.00426 | -0.00707 | +0.01640 | 0.771 |
| close | +BTTS_yes | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 458.93 | 339.93 | 14.14 | 10.93 | 100 | +0.00239 | -0.00853 | +0.01385 | 0.660 |
| close | +BTTS_yes | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 458.93 | 525.27 | 14.14 | 15.05 | 100 | -0.00112 | -0.00554 | +0.00309 | 0.307 |
| close | +BTTS_yes | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 525.27 | 339.93 | 15.05 | 10.93 | 100 | +0.00352 | -0.00722 | +0.01469 | 0.739 |
| close | +BTTS_no | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 445.86 | 347.57 | 13.85 | 11.05 | 100 | +0.00199 | -0.00904 | +0.01359 | 0.636 |
| close | +BTTS_no | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 445.86 | 537.04 | 13.85 | 15.12 | 100 | -0.00154 | -0.00604 | +0.00275 | 0.243 |
| close | +BTTS_no | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 537.04 | 347.57 | 15.12 | 11.05 | 100 | +0.00353 | -0.00716 | +0.01473 | 0.739 |
| close | +all_unders | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 222.68 | 487.78 | 9.48 | 11.70 | 100 | -0.00600 | -0.01773 | +0.00613 | 0.169 |
| close | +all_unders | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 222.68 | 235.89 | 9.48 | 9.81 | 100 | -0.00040 | -0.00504 | +0.00425 | 0.427 |
| close | +all_unders | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 235.89 | 487.78 | 9.81 | 11.70 | 100 | -0.00560 | -0.01713 | +0.00615 | 0.176 |
| close | +all_fringe | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 175.24 | 299.00 | 7.44 | 8.87 | 100 | -0.00371 | -0.01579 | +0.00883 | 0.280 |
| close | +all_fringe | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 175.24 | 183.88 | 7.44 | 7.84 | 100 | -0.00031 | -0.00464 | +0.00402 | 0.440 |
| close | +all_fringe | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 183.88 | 299.00 | 7.84 | 8.87 | 100 | -0.00340 | -0.01595 | +0.00959 | 0.306 |
| t25 | P0 Option B | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 457.70 | 531.78 | 16.28 | 14.15 | 99 | -0.00126 | -0.01273 | +0.01060 | 0.417 |
| t25 | P0 Option B | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 457.70 | 381.23 | 16.28 | 15.14 | 99 | +0.00149 | -0.00284 | +0.00553 | 0.752 |
| t25 | P0 Option B | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 381.23 | 531.78 | 15.14 | 14.15 | 99 | -0.00275 | -0.01435 | +0.00947 | 0.330 |
| t25 | +U0.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 395.26 | 450.88 | 14.90 | 13.20 | 99 | -0.00108 | -0.01232 | +0.01060 | 0.428 |
| t25 | +U0.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 395.26 | 290.90 | 14.90 | 13.05 | 99 | +0.00239 | -0.00244 | +0.00683 | 0.840 |
| t25 | +U0.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 290.90 | 450.88 | 13.05 | 13.20 | 99 | -0.00347 | -0.01535 | +0.00884 | 0.295 |
| t25 | +U1.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 374.97 | 519.76 | 14.06 | 13.76 | 99 | -0.00269 | -0.01379 | +0.00855 | 0.320 |
| t25 | +U1.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 374.97 | 368.10 | 14.06 | 14.49 | 99 | +0.00015 | -0.00495 | +0.00516 | 0.514 |
| t25 | +U1.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 368.10 | 519.76 | 14.49 | 13.76 | 99 | -0.00284 | -0.01440 | +0.00916 | 0.322 |
| t25 | +U3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 452.76 | 426.59 | 15.92 | 12.41 | 99 | +0.00049 | -0.01115 | +0.01282 | 0.532 |
| t25 | +U3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 452.76 | 356.58 | 15.92 | 14.40 | 99 | +0.00193 | -0.00234 | +0.00599 | 0.814 |
| t25 | +U3.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 356.58 | 426.59 | 14.40 | 12.41 | 99 | -0.00144 | -0.01333 | +0.01099 | 0.410 |
| t25 | +U4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 468.36 | 523.46 | 16.32 | 13.67 | 99 | -0.00093 | -0.01240 | +0.01102 | 0.438 |
| t25 | +U4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 468.36 | 384.31 | 16.32 | 15.13 | 99 | +0.00162 | -0.00274 | +0.00569 | 0.771 |
| t25 | +U4.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 384.31 | 523.46 | 15.13 | 13.67 | 99 | -0.00255 | -0.01425 | +0.00962 | 0.340 |
| t25 | +O2.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 486.35 | 562.16 | 16.16 | 13.63 | 99 | -0.00123 | -0.01304 | +0.01092 | 0.421 |
| t25 | +O2.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 486.35 | 390.70 | 16.16 | 14.48 | 99 | +0.00180 | -0.00258 | +0.00599 | 0.787 |
| t25 | +O2.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 390.70 | 562.16 | 14.48 | 13.63 | 99 | -0.00303 | -0.01465 | +0.00913 | 0.312 |
| t25 | +O3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 481.38 | 420.77 | 15.18 | 12.41 | 99 | +0.00111 | -0.01046 | +0.01309 | 0.571 |
| t25 | +O3.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 481.38 | 388.17 | 15.18 | 14.57 | 99 | +0.00177 | -0.00249 | +0.00583 | 0.793 |
| t25 | +O3.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 388.17 | 420.77 | 14.57 | 12.41 | 99 | -0.00065 | -0.01236 | +0.01175 | 0.456 |
| t25 | +O4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 373.27 | 509.18 | 14.67 | 13.85 | 99 | -0.00255 | -0.01400 | +0.00937 | 0.335 |
| t25 | +O4.5 | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 373.27 | 330.64 | 14.67 | 14.07 | 99 | +0.00095 | -0.00352 | +0.00514 | 0.667 |
| t25 | +O4.5 | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 330.64 | 509.18 | 14.07 | 13.85 | 99 | -0.00350 | -0.01508 | +0.00860 | 0.287 |
| t25 | +BTTS_yes | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 402.33 | 455.75 | 15.00 | 12.92 | 99 | -0.00102 | -0.01240 | +0.01056 | 0.429 |
| t25 | +BTTS_yes | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 402.33 | 305.88 | 15.00 | 13.41 | 99 | +0.00215 | -0.00204 | +0.00614 | 0.843 |
| t25 | +BTTS_yes | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 305.88 | 455.75 | 13.41 | 12.92 | 99 | -0.00317 | -0.01475 | +0.00886 | 0.301 |
| t25 | +BTTS_no | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 535.12 | 590.61 | 16.38 | 14.41 | 99 | -0.00085 | -0.01197 | +0.01058 | 0.445 |
| t25 | +BTTS_no | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 535.12 | 439.13 | 16.38 | 15.41 | 99 | +0.00166 | -0.00269 | +0.00577 | 0.774 |
| t25 | +BTTS_no | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 439.13 | 590.61 | 15.41 | 14.41 | 99 | -0.00250 | -0.01404 | +0.00948 | 0.345 |
| t25 | +all_unders | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 294.16 | 380.12 | 12.37 | 11.28 | 99 | -0.00199 | -0.01345 | +0.00991 | 0.369 |
| t25 | +all_unders | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 294.16 | 254.33 | 12.37 | 11.82 | 99 | +0.00108 | -0.00411 | +0.00601 | 0.658 |
| t25 | +all_unders | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 254.33 | 380.12 | 11.82 | 11.28 | 99 | -0.00307 | -0.01527 | +0.00981 | 0.318 |
| t25 | +all_fringe | m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 231.02 | 350.31 | 9.62 | 9.86 | 99 | -0.00311 | -0.01486 | +0.00877 | 0.304 |
| t25 | +all_fringe | m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 231.02 | 210.31 | 9.62 | 9.52 | 99 | +0.00065 | -0.00415 | +0.00532 | 0.597 |
| t25 | +all_fringe | m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 210.31 | 350.31 | 9.52 | 9.86 | 99 | -0.00376 | -0.01562 | +0.00878 | 0.276 |

## Added families, by policy

| environment | policy | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp | odds_mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| close | +BTTS_no | BTTS_btts_no | m05_joint_grw_baseline | 69 | 42.03 | -7.06 | 2.87 | 3.52 | 2.28 |
| close | +BTTS_no | BTTS_btts_no | m05_joint_grw_smile_spine_w040 | 65 | 40.00 | -0.48 | 3.22 | 3.52 | 2.32 |
| close | +BTTS_no | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 64 | 39.06 | -0.16 | 2.65 | 2.49 | 2.34 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 69 | 42.03 | -6.82 | 2.17 | 3.52 | 2.28 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_spine_w040 | 65 | 40.00 | 0.96 | 2.42 | 3.52 | 2.32 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 64 | 39.06 | -0.53 | 2.04 | 2.49 | 2.34 |
| close | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_baseline | 52 | 55.77 | 2.03 | 2.84 | 1.85 | 1.80 |
| close | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_smile_spine_w040 | 52 | 57.69 | 9.58 | 2.52 | -0.12 | 1.87 |
| close | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 66 | 50.00 | -1.71 | 3.07 | 1.56 | 1.87 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 52 | 55.77 | 2.05 | 2.03 | 1.85 | 1.80 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_spine_w040 | 52 | 57.69 | 11.75 | 1.67 | -0.12 | 1.87 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 66 | 50.00 | -2.07 | 2.27 | 1.56 | 1.87 |
| close | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 3.70 | -63.62 | 1.20 | 1.78 | 16.13 |
| close | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 145 | 3.45 | -61.96 | 3.80 | 2.20 | 15.72 |
| close | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 149 | 3.36 | -59.26 | 6.31 | 3.16 | 15.61 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 3.70 | -65.50 | 0.88 | 1.78 | 16.13 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 145 | 3.45 | -66.17 | 2.87 | 2.20 | 15.72 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 149 | 3.36 | -62.68 | 5.07 | 3.16 | 15.61 |
| close | +all_unders | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 3.70 | -64.46 | 1.05 | 1.78 | 16.13 |
| close | +all_unders | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 145 | 3.45 | -63.46 | 3.42 | 2.20 | 15.72 |
| close | +all_unders | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 149 | 3.36 | -60.60 | 5.98 | 3.16 | 15.61 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 23.81 | 37.66 | 3.51 | 4.54 | 4.75 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 208 | 21.63 | -6.31 | 9.96 | 4.28 | 4.55 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 128 | 19.53 | -2.73 | 4.05 | 3.03 | 4.89 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 23.81 | 32.75 | 2.61 | 4.54 | 4.75 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 208 | 21.63 | -5.72 | 7.74 | 4.28 | 4.55 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 128 | 19.53 | -2.20 | 3.06 | 3.03 | 4.89 |
| close | +all_unders | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 23.81 | 36.75 | 3.13 | 4.54 | 4.75 |
| close | +all_unders | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 208 | 21.63 | -6.56 | 9.47 | 4.28 | 4.55 |
| close | +all_unders | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 128 | 19.53 | -3.51 | 3.65 | 3.03 | 4.89 |
| close | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_baseline | 104 | 50.96 | 0.89 | 6.33 | 5.89 | 1.93 |
| close | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_smile_spine_w040 | 49 | 51.02 | 16.47 | 2.33 | 2.32 | 2.11 |
| close | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 96 | 48.96 | 6.16 | 4.79 | 2.89 | 2.08 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 104 | 50.96 | 1.50 | 5.03 | 5.89 | 1.93 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_spine_w040 | 49 | 51.02 | 12.71 | 1.82 | 2.32 | 2.11 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 96 | 48.96 | 6.33 | 3.87 | 2.89 | 2.08 |
| close | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_baseline | 120 | 36.67 | 2.94 | 7.49 | 5.52 | 3.36 |
| close | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_smile_spine_w040 | 206 | 30.58 | -1.80 | 9.76 | 2.14 | 3.37 |
| close | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 161 | 30.43 | -4.38 | 6.42 | 2.49 | 3.48 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 120 | 36.67 | 2.28 | 6.00 | 5.52 | 3.36 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_spine_w040 | 206 | 30.58 | -4.38 | 7.75 | 2.14 | 3.37 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 161 | 30.43 | -5.65 | 5.27 | 2.49 | 3.48 |
| close | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 74.75 | 2.43 | 7.00 | 5.85 | 1.50 |
| close | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 29 | 75.86 | 22.27 | 0.99 | 3.10 | 1.62 |
| close | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 50 | 68.00 | 13.62 | 2.33 | 3.24 | 1.59 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 74.75 | 2.75 | 5.50 | 5.85 | 1.50 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 29 | 75.86 | 25.51 | 0.68 | 3.10 | 1.62 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 50 | 68.00 | 16.28 | 1.68 | 3.24 | 1.59 |
| close | +all_unders | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 74.75 | 2.34 | 6.45 | 5.85 | 1.50 |
| close | +all_unders | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 29 | 75.86 | 23.22 | 0.85 | 3.10 | 1.62 |
| close | +all_unders | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 50 | 68.00 | 14.29 | 2.05 | 3.24 | 1.59 |
| close | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_baseline | 54 | 9.26 | -30.85 | 1.99 | 3.66 | 7.01 |
| close | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_smile_spine_w040 | 95 | 9.47 | 2.37 | 5.54 | 3.70 | 6.81 |
| close | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 93 | 9.68 | 6.99 | 4.50 | 2.87 | 6.88 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 54 | 9.26 | -32.89 | 1.47 | 3.66 | 7.01 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_spine_w040 | 95 | 9.47 | 11.15 | 4.18 | 3.70 | 6.81 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 93 | 9.68 | 16.26 | 3.53 | 2.87 | 6.88 |
| close | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 96.43 | 18.79 | 3.03 | 4.49 | 1.23 |
| close | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 96.43 | 18.53 | 2.25 | 4.49 | 1.23 |
| close | +all_unders | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 96.43 | 18.55 | 2.70 | 4.49 | 1.23 |
| t25 | +BTTS_no | BTTS_btts_no | m05_joint_grw_baseline | 82 | 50.00 | 13.02 | 4.05 | 3.99 | 2.28 |
| t25 | +BTTS_no | BTTS_btts_no | m05_joint_grw_smile_spine_w040 | 97 | 40.21 | 11.47 | 5.20 | 3.55 | 2.37 |
| t25 | +BTTS_no | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 81 | 45.68 | 18.09 | 4.16 | 2.67 | 2.38 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 82 | 50.00 | 14.01 | 3.00 | 3.99 | 2.28 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_spine_w040 | 97 | 40.21 | 9.17 | 4.18 | 3.55 | 2.37 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 81 | 45.68 | 14.83 | 3.44 | 2.67 | 2.38 |
| t25 | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_baseline | 73 | 60.27 | -12.37 | 4.39 | 3.07 | 1.79 |
| t25 | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_smile_spine_w040 | 46 | 50.00 | -38.42 | 3.30 | 1.04 | 1.90 |
| t25 | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 54 | 50.00 | -45.28 | 3.65 | 2.09 | 1.89 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 73 | 60.27 | -12.94 | 3.23 | 3.07 | 1.79 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_spine_w040 | 46 | 50.00 | -38.21 | 2.49 | 1.04 | 1.90 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 54 | 50.00 | -44.61 | 2.92 | 2.09 | 1.89 |
| t25 | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 2.70 | -89.53 | 0.71 | 2.27 | 16.94 |
| t25 | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 73 | 5.48 | -49.27 | 2.08 | 2.45 | 15.98 |
| t25 | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 75 | 6.67 | -42.15 | 3.47 | 3.41 | 15.91 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 2.70 | -90.05 | 0.49 | 2.27 | 16.94 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 73 | 5.48 | -46.07 | 1.60 | 2.45 | 15.98 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 75 | 6.67 | -38.21 | 2.80 | 3.41 | 15.91 |
| t25 | +all_unders | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 2.70 | -89.53 | 0.61 | 2.27 | 16.94 |
| t25 | +all_unders | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 73 | 5.48 | -47.51 | 1.88 | 2.45 | 15.98 |
| t25 | +all_unders | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 75 | 6.67 | -41.31 | 3.29 | 3.41 | 15.91 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 23.68 | 3.90 | 3.11 | 4.86 | 4.45 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 152 | 19.74 | -14.63 | 8.34 | 4.29 | 4.40 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 99 | 14.14 | -6.80 | 3.16 | 2.65 | 4.65 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 23.68 | 2.41 | 2.35 | 4.86 | 4.45 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 152 | 19.74 | -14.14 | 6.77 | 4.29 | 4.40 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 99 | 14.14 | -8.25 | 2.50 | 2.65 | 4.65 |
| t25 | +all_unders | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 23.68 | 1.33 | 2.82 | 4.86 | 4.45 |
| t25 | +all_unders | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 152 | 19.74 | -14.99 | 8.10 | 4.29 | 4.40 |
| t25 | +all_unders | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 99 | 14.14 | -8.21 | 3.00 | 2.65 | 4.65 |
| t25 | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_baseline | 116 | 58.62 | 10.77 | 8.55 | 5.74 | 1.93 |
| t25 | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_smile_spine_w040 | 64 | 54.69 | 15.56 | 3.22 | 2.14 | 2.11 |
| t25 | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 115 | 52.17 | 9.25 | 6.45 | 2.47 | 2.07 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 116 | 58.62 | 10.25 | 6.72 | 5.74 | 1.93 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_spine_w040 | 64 | 54.69 | 14.33 | 2.56 | 2.14 | 2.11 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 115 | 52.17 | 9.03 | 5.40 | 2.47 | 2.07 |
| t25 | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_baseline | 75 | 26.67 | -23.70 | 4.70 | 4.99 | 3.14 |
| t25 | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_smile_spine_w040 | 116 | 27.59 | -7.88 | 6.67 | 1.76 | 3.23 |
| t25 | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 94 | 27.66 | 0.40 | 3.78 | 2.05 | 3.30 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 75 | 26.67 | -20.39 | 3.50 | 4.99 | 3.14 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_spine_w040 | 116 | 27.59 | -4.36 | 5.28 | 1.76 | 3.23 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 94 | 27.66 | 4.22 | 3.10 | 2.05 | 3.30 |
| t25 | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 56.72 | -15.00 | 6.16 | 6.01 | 1.51 |
| t25 | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 25 | 56.00 | -4.37 | 1.35 | 3.59 | 1.62 |
| t25 | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 57.14 | -11.37 | 2.42 | 3.56 | 1.58 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 56.72 | -16.23 | 4.66 | 6.01 | 1.51 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 25 | 56.00 | -3.99 | 0.96 | 3.59 | 1.62 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 57.14 | -11.32 | 1.83 | 3.56 | 1.58 |
| t25 | +all_unders | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 56.72 | -16.13 | 5.69 | 6.01 | 1.51 |
| t25 | +all_unders | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 25 | 56.00 | -5.77 | 1.21 | 3.59 | 1.62 |
| t25 | +all_unders | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 57.14 | -13.11 | 2.26 | 3.56 | 1.58 |
| t25 | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_baseline | 21 | 4.76 | -9.29 | 0.68 | 3.24 | 6.60 |
| t25 | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_smile_spine_w040 | 46 | 4.35 | -49.41 | 2.50 | 2.65 | 6.55 |
| t25 | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 43 | 2.33 | -41.28 | 1.76 | 2.09 | 6.67 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 21 | 4.76 | 1.13 | 0.50 | 3.24 | 6.60 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_spine_w040 | 46 | 4.35 | -43.66 | 1.90 | 2.65 | 6.55 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 43 | 2.33 | -40.44 | 1.38 | 2.09 | 6.67 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 95.65 | 10.43 | 3.64 | 4.84 | 1.22 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_smile_spine_w040 | 1 | 100.00 | 35.28 | 0.03 | 9.58 | 1.36 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 2 | 100.00 | 36.89 | 0.10 | 9.22 | 1.48 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 95.65 | 8.81 | 2.81 | 4.84 | 1.22 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_spine_w040 | 1 | 100.00 | 35.28 | 0.02 | 9.58 | 1.36 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 2 | 100.00 | 36.82 | 0.07 | 9.22 | 1.48 |
| t25 | +all_unders | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 95.65 | 9.82 | 3.29 | 4.84 | 1.22 |
| t25 | +all_unders | O/U 4.5_under_45 | m05_joint_grw_smile_spine_w040 | 1 | 100.00 | 35.28 | 0.03 | 9.58 | 1.36 |
| t25 | +all_unders | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 2 | 100.00 | 37.10 | 0.08 | 9.22 | 1.48 |

## Staked-bet calibration by family (the Jensen check)

Positive `model_minus_realised` = the staked bets were priced above their realised rate. Staked bets are a positive-edge selection, not the unconditional forecast — r04's strike ladder is the unconditional statement.

| environment | policy | family | model | n_bets | mean_p_model | mean_p_market | realised_win_rate | model_minus_realised | roi_pct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| close | +U1.5 | 1X2_away | m05_joint_grw_baseline | 399 | 0.3260 | 0.2703 | 0.2632 | +0.0628 | 7.85 |
| close | +U1.5 | 1X2_away | m05_joint_grw_smile_spine_w040 | 362 | 0.3087 | 0.2572 | 0.2514 | +0.0573 | 20.85 |
| close | +U1.5 | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 321 | 0.3002 | 0.2497 | 0.2555 | +0.0447 | 23.49 |
| close | +U1.5 | 1X2_draw | m05_joint_grw_baseline | 278 | 0.2490 | 0.2444 | 0.2446 | +0.0044 | 2.02 |
| close | +U1.5 | 1X2_draw | m05_joint_grw_smile_spine_w040 | 340 | 0.2567 | 0.2481 | 0.2559 | +0.0009 | 5.37 |
| close | +U1.5 | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 475 | 0.2739 | 0.2540 | 0.2632 | +0.0107 | 5.36 |
| close | +U1.5 | 1X2_home | m05_joint_grw_baseline | 307 | 0.3984 | 0.3488 | 0.3941 | +0.0043 | 20.30 |
| close | +U1.5 | 1X2_home | m05_joint_grw_smile_spine_w040 | 297 | 0.3863 | 0.3379 | 0.3872 | -0.0009 | 16.98 |
| close | +U1.5 | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 254 | 0.3729 | 0.3248 | 0.3819 | -0.0090 | 17.53 |
| close | +U1.5 | O/U 1.5_over_15 | m05_joint_grw_baseline | 63 | 0.7936 | 0.7586 | 0.7619 | +0.0317 | 0.98 |
| close | +U1.5 | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 2 | 0.7060 | 0.6960 | 1.0000 | -0.2940 | 41.27 |
| close | +U1.5 | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 41 | 0.7430 | 0.7392 | 0.6829 | +0.0600 | -7.22 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 0.2667 | 0.2213 | 0.2381 | +0.0286 | 37.66 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 208 | 0.2708 | 0.2279 | 0.2163 | +0.0544 | -6.31 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 128 | 0.2419 | 0.2116 | 0.1953 | +0.0466 | -2.73 |
| close | +U1.5 | O/U 2.5_under_25 | m05_joint_grw_baseline | 199 | 0.5180 | 0.4654 | 0.4874 | +0.0305 | 10.39 |
| close | +U1.5 | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 246 | 0.4888 | 0.4520 | 0.4878 | +0.0010 | 5.67 |
| close | +U1.5 | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 162 | 0.4740 | 0.4381 | 0.4506 | +0.0234 | 14.88 |
| close | +U4.5 | 1X2_away | m05_joint_grw_baseline | 399 | 0.3260 | 0.2703 | 0.2632 | +0.0628 | 7.74 |
| close | +U4.5 | 1X2_away | m05_joint_grw_smile_spine_w040 | 362 | 0.3087 | 0.2572 | 0.2514 | +0.0573 | 20.84 |
| close | +U4.5 | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 321 | 0.3002 | 0.2497 | 0.2555 | +0.0447 | 23.31 |
| close | +U4.5 | 1X2_draw | m05_joint_grw_baseline | 278 | 0.2490 | 0.2444 | 0.2446 | +0.0044 | 1.67 |
| close | +U4.5 | 1X2_draw | m05_joint_grw_smile_spine_w040 | 340 | 0.2567 | 0.2481 | 0.2559 | +0.0009 | 6.08 |
| close | +U4.5 | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 475 | 0.2739 | 0.2540 | 0.2632 | +0.0107 | 5.65 |
| close | +U4.5 | 1X2_home | m05_joint_grw_baseline | 307 | 0.3984 | 0.3488 | 0.3941 | +0.0043 | 20.11 |
| close | +U4.5 | 1X2_home | m05_joint_grw_smile_spine_w040 | 297 | 0.3863 | 0.3379 | 0.3872 | -0.0009 | 16.96 |
| close | +U4.5 | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 254 | 0.3729 | 0.3248 | 0.3819 | -0.0090 | 17.53 |
| close | +U4.5 | O/U 1.5_over_15 | m05_joint_grw_baseline | 63 | 0.7936 | 0.7586 | 0.7619 | +0.0317 | 1.69 |
| close | +U4.5 | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 2 | 0.7060 | 0.6960 | 1.0000 | -0.2940 | 41.28 |
| close | +U4.5 | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 41 | 0.7430 | 0.7392 | 0.6829 | +0.0600 | -7.12 |
| close | +U4.5 | O/U 2.5_under_25 | m05_joint_grw_baseline | 199 | 0.5180 | 0.4654 | 0.4874 | +0.0305 | 11.20 |
| close | +U4.5 | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 246 | 0.4888 | 0.4520 | 0.4878 | +0.0010 | 6.10 |
| close | +U4.5 | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 162 | 0.4740 | 0.4381 | 0.4506 | +0.0234 | 15.24 |
| close | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 0.8607 | 0.8158 | 0.9643 | -0.1036 | 18.79 |
| close | +all_fringe | 1X2_away | m05_joint_grw_baseline | 399 | 0.3260 | 0.2703 | 0.2632 | +0.0628 | 8.30 |
| close | +all_fringe | 1X2_away | m05_joint_grw_smile_spine_w040 | 362 | 0.3087 | 0.2572 | 0.2514 | +0.0573 | 19.61 |
| close | +all_fringe | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 321 | 0.3002 | 0.2497 | 0.2555 | +0.0447 | 22.49 |
| close | +all_fringe | 1X2_draw | m05_joint_grw_baseline | 278 | 0.2490 | 0.2444 | 0.2446 | +0.0044 | -1.27 |
| close | +all_fringe | 1X2_draw | m05_joint_grw_smile_spine_w040 | 340 | 0.2567 | 0.2481 | 0.2559 | +0.0009 | 1.40 |
| close | +all_fringe | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 475 | 0.2739 | 0.2540 | 0.2632 | +0.0107 | 2.78 |
| close | +all_fringe | 1X2_home | m05_joint_grw_baseline | 307 | 0.3984 | 0.3488 | 0.3941 | +0.0043 | 19.55 |
| close | +all_fringe | 1X2_home | m05_joint_grw_smile_spine_w040 | 297 | 0.3863 | 0.3379 | 0.3872 | -0.0009 | 15.91 |
| close | +all_fringe | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 254 | 0.3729 | 0.3248 | 0.3819 | -0.0090 | 16.87 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 69 | 0.4761 | 0.4409 | 0.4203 | +0.0558 | -6.82 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_spine_w040 | 65 | 0.4680 | 0.4328 | 0.4000 | +0.0680 | 0.96 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 64 | 0.4539 | 0.4290 | 0.3906 | +0.0633 | -0.53 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 52 | 0.5771 | 0.5585 | 0.5577 | +0.0194 | 2.05 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_spine_w040 | 52 | 0.5384 | 0.5396 | 0.5769 | -0.0385 | 11.75 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 66 | 0.5535 | 0.5379 | 0.5000 | +0.0535 | -2.07 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 0.0844 | 0.0665 | 0.0370 | +0.0473 | -65.50 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 145 | 0.0895 | 0.0674 | 0.0345 | +0.0550 | -66.17 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 149 | 0.0996 | 0.0680 | 0.0336 | +0.0660 | -62.68 |
| close | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_baseline | 63 | 0.7936 | 0.7586 | 0.7619 | +0.0317 | 2.14 |
| close | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 2 | 0.7060 | 0.6960 | 1.0000 | -0.2940 | 41.25 |
| close | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 41 | 0.7430 | 0.7392 | 0.6829 | +0.0600 | -7.34 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 0.2667 | 0.2213 | 0.2381 | +0.0286 | 32.75 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 208 | 0.2708 | 0.2279 | 0.2163 | +0.0544 | -5.72 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 128 | 0.2419 | 0.2116 | 0.1953 | +0.0466 | -2.20 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 104 | 0.5800 | 0.5211 | 0.5096 | +0.0704 | 1.50 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_spine_w040 | 49 | 0.4992 | 0.4760 | 0.5102 | -0.0110 | 12.71 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 96 | 0.5104 | 0.4814 | 0.4896 | +0.0208 | 6.33 |
| close | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_baseline | 199 | 0.5180 | 0.4654 | 0.4874 | +0.0305 | 12.18 |
| close | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 246 | 0.4888 | 0.4520 | 0.4878 | +0.0010 | 5.04 |
| close | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 162 | 0.4740 | 0.4381 | 0.4506 | +0.0234 | 13.70 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 120 | 0.3586 | 0.3035 | 0.3667 | -0.0080 | 2.28 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_spine_w040 | 206 | 0.3242 | 0.3027 | 0.3058 | +0.0183 | -4.38 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 161 | 0.3177 | 0.2927 | 0.3043 | +0.0133 | -5.65 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 0.7286 | 0.6701 | 0.7475 | -0.0189 | 2.75 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 29 | 0.6518 | 0.6209 | 0.7586 | -0.1068 | 25.51 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 50 | 0.6634 | 0.6310 | 0.6800 | -0.0166 | 16.28 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 54 | 0.1879 | 0.1513 | 0.0926 | +0.0953 | -32.89 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_spine_w040 | 95 | 0.1980 | 0.1610 | 0.0947 | +0.1033 | 11.15 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 93 | 0.1867 | 0.1580 | 0.0968 | +0.0899 | 16.26 |
| close | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 0.8607 | 0.8158 | 0.9643 | -0.1036 | 18.53 |
| close | P0 Option B | 1X2_away | m05_joint_grw_baseline | 403 | 0.3263 | 0.2713 | 0.2655 | +0.0608 | 7.77 |
| close | P0 Option B | 1X2_away | m05_joint_grw_smile_spine_w040 | 366 | 0.3094 | 0.2587 | 0.2514 | +0.0581 | 20.88 |
| close | P0 Option B | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 326 | 0.3009 | 0.2515 | 0.2607 | +0.0402 | 23.57 |
| close | P0 Option B | 1X2_draw | m05_joint_grw_baseline | 273 | 0.2494 | 0.2443 | 0.2418 | +0.0076 | 3.38 |
| close | P0 Option B | 1X2_draw | m05_joint_grw_smile_spine_w040 | 327 | 0.2570 | 0.2477 | 0.2569 | +0.0001 | 7.00 |
| close | P0 Option B | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 470 | 0.2740 | 0.2538 | 0.2638 | +0.0101 | 5.78 |
| close | P0 Option B | 1X2_home | m05_joint_grw_baseline | 310 | 0.3982 | 0.3492 | 0.3903 | +0.0079 | 19.99 |
| close | P0 Option B | 1X2_home | m05_joint_grw_smile_spine_w040 | 302 | 0.3868 | 0.3391 | 0.3874 | -0.0007 | 16.37 |
| close | P0 Option B | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 255 | 0.3727 | 0.3247 | 0.3804 | -0.0077 | 17.07 |
| close | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_baseline | 64 | 0.7939 | 0.7592 | 0.7500 | +0.0439 | 1.25 |
| close | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 2 | 0.7060 | 0.6960 | 1.0000 | -0.2940 | 41.28 |
| close | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 42 | 0.7431 | 0.7391 | 0.6905 | +0.0526 | -5.97 |
| close | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_baseline | 197 | 0.5184 | 0.4652 | 0.4924 | +0.0260 | 10.75 |
| close | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 235 | 0.4891 | 0.4511 | 0.4894 | -0.0003 | 7.03 |
| close | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 157 | 0.4739 | 0.4372 | 0.4586 | +0.0153 | 17.39 |
| t25 | +U1.5 | 1X2_away | m05_joint_grw_baseline | 361 | 0.3276 | 0.2711 | 0.2604 | +0.0672 | 8.25 |
| t25 | +U1.5 | 1X2_away | m05_joint_grw_smile_spine_w040 | 316 | 0.3103 | 0.2553 | 0.2563 | +0.0540 | 21.59 |
| t25 | +U1.5 | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 288 | 0.3015 | 0.2485 | 0.2431 | +0.0584 | 22.01 |
| t25 | +U1.5 | 1X2_draw | m05_joint_grw_baseline | 253 | 0.2503 | 0.2468 | 0.2372 | +0.0132 | -0.28 |
| t25 | +U1.5 | 1X2_draw | m05_joint_grw_smile_spine_w040 | 316 | 0.2572 | 0.2481 | 0.2563 | +0.0009 | 7.64 |
| t25 | +U1.5 | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 450 | 0.2738 | 0.2544 | 0.2578 | +0.0160 | 2.49 |
| t25 | +U1.5 | 1X2_home | m05_joint_grw_baseline | 281 | 0.3968 | 0.3479 | 0.3665 | +0.0303 | 21.23 |
| t25 | +U1.5 | 1X2_home | m05_joint_grw_smile_spine_w040 | 266 | 0.3854 | 0.3340 | 0.3684 | +0.0170 | 18.19 |
| t25 | +U1.5 | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 225 | 0.3711 | 0.3197 | 0.3378 | +0.0333 | 19.70 |
| t25 | +U1.5 | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 0.7913 | 0.7527 | 0.8448 | -0.0535 | 18.34 |
| t25 | +U1.5 | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 3 | 0.6809 | 0.6698 | 0.3333 | +0.3476 | -1.81 |
| t25 | +U1.5 | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 27 | 0.7337 | 0.7277 | 0.8519 | -0.1182 | 7.85 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 0.2814 | 0.2328 | 0.2368 | +0.0445 | 3.90 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 152 | 0.2756 | 0.2327 | 0.1974 | +0.0783 | -14.63 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 99 | 0.2461 | 0.2196 | 0.1414 | +0.1047 | -6.80 |
| t25 | +U1.5 | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 0.5256 | 0.4709 | 0.5294 | -0.0038 | 17.57 |
| t25 | +U1.5 | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 210 | 0.4888 | 0.4528 | 0.5000 | -0.0112 | 13.23 |
| t25 | +U1.5 | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 127 | 0.4743 | 0.4393 | 0.5039 | -0.0296 | 15.21 |
| t25 | +U4.5 | 1X2_away | m05_joint_grw_baseline | 361 | 0.3276 | 0.2711 | 0.2604 | +0.0672 | 7.83 |
| t25 | +U4.5 | 1X2_away | m05_joint_grw_smile_spine_w040 | 316 | 0.3103 | 0.2553 | 0.2563 | +0.0540 | 20.66 |
| t25 | +U4.5 | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 288 | 0.3015 | 0.2485 | 0.2431 | +0.0584 | 21.77 |
| t25 | +U4.5 | 1X2_draw | m05_joint_grw_baseline | 253 | 0.2503 | 0.2468 | 0.2372 | +0.0132 | 0.40 |
| t25 | +U4.5 | 1X2_draw | m05_joint_grw_smile_spine_w040 | 316 | 0.2572 | 0.2481 | 0.2563 | +0.0009 | 7.77 |
| t25 | +U4.5 | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 450 | 0.2738 | 0.2544 | 0.2578 | +0.0160 | 2.56 |
| t25 | +U4.5 | 1X2_home | m05_joint_grw_baseline | 281 | 0.3968 | 0.3479 | 0.3665 | +0.0303 | 21.06 |
| t25 | +U4.5 | 1X2_home | m05_joint_grw_smile_spine_w040 | 266 | 0.3854 | 0.3340 | 0.3684 | +0.0170 | 18.10 |
| t25 | +U4.5 | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 225 | 0.3711 | 0.3197 | 0.3378 | +0.0333 | 19.55 |
| t25 | +U4.5 | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 0.7913 | 0.7527 | 0.8448 | -0.0535 | 18.67 |
| t25 | +U4.5 | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 3 | 0.6809 | 0.6698 | 0.3333 | +0.3476 | -2.27 |
| t25 | +U4.5 | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 27 | 0.7337 | 0.7277 | 0.8519 | -0.1182 | 7.95 |
| t25 | +U4.5 | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 0.5256 | 0.4709 | 0.5294 | -0.0038 | 16.74 |
| t25 | +U4.5 | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 210 | 0.4888 | 0.4528 | 0.5000 | -0.0112 | 12.95 |
| t25 | +U4.5 | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 127 | 0.4743 | 0.4393 | 0.5039 | -0.0296 | 15.46 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 0.8726 | 0.8242 | 0.9565 | -0.0840 | 10.43 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_smile_spine_w040 | 1 | 0.8296 | 0.7339 | 1.0000 | -0.1704 | 35.28 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 2 | 0.7715 | 0.6793 | 1.0000 | -0.2285 | 36.89 |
| t25 | +all_fringe | 1X2_away | m05_joint_grw_baseline | 361 | 0.3276 | 0.2711 | 0.2604 | +0.0672 | 11.11 |
| t25 | +all_fringe | 1X2_away | m05_joint_grw_smile_spine_w040 | 316 | 0.3103 | 0.2553 | 0.2563 | +0.0540 | 23.29 |
| t25 | +all_fringe | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 288 | 0.3015 | 0.2485 | 0.2431 | +0.0584 | 22.65 |
| t25 | +all_fringe | 1X2_draw | m05_joint_grw_baseline | 253 | 0.2503 | 0.2468 | 0.2372 | +0.0132 | -3.90 |
| t25 | +all_fringe | 1X2_draw | m05_joint_grw_smile_spine_w040 | 316 | 0.2572 | 0.2481 | 0.2563 | +0.0009 | 6.17 |
| t25 | +all_fringe | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 450 | 0.2738 | 0.2544 | 0.2578 | +0.0160 | 0.79 |
| t25 | +all_fringe | 1X2_home | m05_joint_grw_baseline | 281 | 0.3968 | 0.3479 | 0.3665 | +0.0303 | 20.85 |
| t25 | +all_fringe | 1X2_home | m05_joint_grw_smile_spine_w040 | 266 | 0.3854 | 0.3340 | 0.3684 | +0.0170 | 17.25 |
| t25 | +all_fringe | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 225 | 0.3711 | 0.3197 | 0.3378 | +0.0333 | 20.21 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 82 | 0.4802 | 0.4403 | 0.5000 | -0.0198 | 14.01 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_spine_w040 | 97 | 0.4589 | 0.4233 | 0.4021 | +0.0568 | 9.17 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 81 | 0.4490 | 0.4222 | 0.4568 | -0.0078 | 14.83 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 73 | 0.5945 | 0.5638 | 0.6027 | -0.0082 | -12.94 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_spine_w040 | 46 | 0.5421 | 0.5317 | 0.5000 | +0.0421 | -38.21 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 54 | 0.5533 | 0.5324 | 0.5000 | +0.0533 | -44.61 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 0.0890 | 0.0663 | 0.0270 | +0.0620 | -90.05 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_spine_w040 | 73 | 0.0918 | 0.0674 | 0.0548 | +0.0370 | -46.07 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 75 | 0.1018 | 0.0677 | 0.0667 | +0.0351 | -38.21 |
| t25 | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 0.7913 | 0.7527 | 0.8448 | -0.0535 | 18.51 |
| t25 | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 3 | 0.6809 | 0.6698 | 0.3333 | +0.3476 | -0.06 |
| t25 | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 27 | 0.7337 | 0.7277 | 0.8519 | -0.1182 | 4.61 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 0.2814 | 0.2328 | 0.2368 | +0.0445 | 2.41 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_spine_w040 | 152 | 0.2756 | 0.2327 | 0.1974 | +0.0783 | -14.14 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 99 | 0.2461 | 0.2196 | 0.1414 | +0.1047 | -8.25 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 116 | 0.5787 | 0.5213 | 0.5862 | -0.0075 | 10.25 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_spine_w040 | 64 | 0.4955 | 0.4742 | 0.5469 | -0.0514 | 14.33 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 115 | 0.5085 | 0.4837 | 0.5217 | -0.0133 | 9.03 |
| t25 | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 0.5256 | 0.4709 | 0.5294 | -0.0038 | 18.46 |
| t25 | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 210 | 0.4888 | 0.4528 | 0.5000 | -0.0112 | 14.70 |
| t25 | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 127 | 0.4743 | 0.4393 | 0.5039 | -0.0296 | 17.83 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 75 | 0.3746 | 0.3247 | 0.2667 | +0.1080 | -20.39 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_spine_w040 | 116 | 0.3336 | 0.3160 | 0.2759 | +0.0577 | -4.36 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 94 | 0.3299 | 0.3094 | 0.2766 | +0.0533 | 4.22 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 0.7260 | 0.6660 | 0.5672 | +0.1589 | -16.23 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_spine_w040 | 25 | 0.6603 | 0.6244 | 0.5600 | +0.1003 | -3.99 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 0.6716 | 0.6360 | 0.5714 | +0.1002 | -11.32 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 21 | 0.1904 | 0.1580 | 0.0476 | +0.1428 | 1.13 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_spine_w040 | 46 | 0.1915 | 0.1649 | 0.0435 | +0.1480 | -43.66 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 43 | 0.1788 | 0.1579 | 0.0233 | +0.1555 | -40.44 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 0.8726 | 0.8242 | 0.9565 | -0.0840 | 8.81 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_spine_w040 | 1 | 0.8296 | 0.7339 | 1.0000 | -0.1704 | 35.28 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 2 | 0.7715 | 0.6793 | 1.0000 | -0.2285 | 36.82 |
| t25 | P0 Option B | 1X2_away | m05_joint_grw_baseline | 363 | 0.3285 | 0.2723 | 0.2645 | +0.0641 | 8.44 |
| t25 | P0 Option B | 1X2_away | m05_joint_grw_smile_spine_w040 | 316 | 0.3103 | 0.2553 | 0.2563 | +0.0540 | 20.40 |
| t25 | P0 Option B | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 288 | 0.3015 | 0.2485 | 0.2431 | +0.0584 | 21.74 |
| t25 | P0 Option B | 1X2_draw | m05_joint_grw_baseline | 252 | 0.2504 | 0.2467 | 0.2381 | +0.0123 | 0.86 |
| t25 | P0 Option B | 1X2_draw | m05_joint_grw_smile_spine_w040 | 311 | 0.2573 | 0.2479 | 0.2572 | +0.0001 | 7.27 |
| t25 | P0 Option B | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 448 | 0.2739 | 0.2544 | 0.2567 | +0.0172 | 2.51 |
| t25 | P0 Option B | 1X2_home | m05_joint_grw_baseline | 281 | 0.3968 | 0.3479 | 0.3665 | +0.0303 | 20.76 |
| t25 | P0 Option B | 1X2_home | m05_joint_grw_smile_spine_w040 | 267 | 0.3849 | 0.3337 | 0.3670 | +0.0179 | 18.53 |
| t25 | P0 Option B | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 226 | 0.3706 | 0.3195 | 0.3363 | +0.0343 | 19.75 |
| t25 | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 0.7913 | 0.7527 | 0.8448 | -0.0535 | 18.60 |
| t25 | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 3 | 0.6809 | 0.6698 | 0.3333 | +0.3476 | -17.09 |
| t25 | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 27 | 0.7337 | 0.7277 | 0.8519 | -0.1182 | 7.91 |
| t25 | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 0.5256 | 0.4709 | 0.5294 | -0.0038 | 17.81 |
| t25 | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 207 | 0.4881 | 0.4520 | 0.4976 | -0.0095 | 12.57 |
| t25 | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 123 | 0.4732 | 0.4370 | 0.4959 | -0.0228 | 15.05 |
