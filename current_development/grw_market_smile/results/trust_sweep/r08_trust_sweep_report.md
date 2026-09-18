# r08 trust-pruning sweep — Task 015

Generated 2026-09-13 22:17. Arms: `m05_joint_grw_smile_supremacy_w040` (test) and `m05_joint_grw_baseline` (control). Additions at trust 0.7143 on Option B's book plus O/U 4.5; risk, cap and grouping are Option B's.

## Gates

| gate | environment | model | pass | detail |
|---|---|---|---:|---|
| S0 extended book inert at trust 0 | close | m05_joint_grw_baseline | false | ext return 368.8441 vs canon 385.7776 (Δ -16.9335 pp), bets 1246 vs 1247 |
| S0 extended book inert at trust 0 | close | m05_joint_grw_smile_supremacy_w040 | false | ext return 524.7466 vs canon 545.0098 (Δ -20.2632 pp), bets 1238 vs 1237 |
| S2 smile routing, all strikes | close | m05_joint_grw_smile_supremacy_w040 | true | 732 totals bets, max |p − smile| 2.7e-15 |
| S0 extended book inert at trust 0 | t25 | m05_joint_grw_baseline | false | ext return 509.5218 vs canon 531.7811 (Δ -22.2593 pp), bets 1123 vs 1124 |
| S0 extended book inert at trust 0 | t25 | m05_joint_grw_smile_supremacy_w040 | false | ext return 472.8663 vs canon 479.4433 (Δ -6.5771 pp), bets 1085 vs 1087 |
| S2 smile routing, all strikes | t25 | m05_joint_grw_smile_supremacy_w040 | true | 539 totals bets, max |p − smile| 2.4e-15 |
| S1 close/P0 vs r06 | close | m05_joint_grw_baseline | true | return 385.7776 vs 385.7776, bets 1247 vs 1247 |
| S1 close/P0 vs r06 | close | m05_joint_grw_smile_supremacy_w040 | true | return 545.0098 vs 545.0098, bets 1237 vs 1237 |

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

## Sweep

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
| close | m05_joint_grw_smile_supremacy_w040 | P0 Option B | 1237 | 545.01 | +0.00 | 15.82 | 1.516 | -44.03 | 13 | 0 | n/a | n/a | 0.00 | 1.000 | 15.82 | +0.00 |
| close | m05_joint_grw_smile_supremacy_w040 | P0 ext book | 1238 | 524.75 | +0.00 | 15.61 | 1.486 | -44.37 | 13 | 0 | n/a | n/a | 0.00 | 1.000 | 15.61 | +0.00 |
| close | m05_joint_grw_smile_supremacy_w040 | +U0.5 | 1291 | 507.36 | -17.38 | 15.33 | 1.471 | -43.81 | 15 | 53 | 3.77 | -41.80 | 0.42 | 0.999 | 15.57 | -0.04 |
| close | m05_joint_grw_smile_supremacy_w040 | +U1.5 | 1331 | 547.22 | +22.47 | 15.61 | 1.535 | -42.50 | 16 | 93 | 19.35 | 15.85 | 1.85 | 0.992 | 15.61 | -0.00 |
| close | m05_joint_grw_smile_supremacy_w040 | +U3.5 | 1326 | 556.43 | +31.68 | 15.36 | 1.556 | -41.62 | 16 | 88 | 64.77 | 2.38 | 4.75 | 0.982 | 16.01 | +0.40 |
| close | m05_joint_grw_smile_supremacy_w040 | +U4.5 | 1269 | 546.57 | +21.83 | 15.57 | 1.518 | -43.61 | 16 | 31 | 93.55 | 21.77 | 1.99 | 0.997 | 15.44 | -0.17 |
| close | m05_joint_grw_smile_supremacy_w040 | +O2.5 | 1311 | 555.23 | +30.48 | 15.52 | 1.525 | -44.42 | 17 | 73 | 50.68 | 19.61 | 3.24 | 0.994 | 15.38 | -0.23 |
| close | m05_joint_grw_smile_supremacy_w040 | +O3.5 | 1333 | 506.64 | -18.10 | 14.97 | 1.467 | -44.56 | 15 | 95 | 27.37 | -5.02 | 2.79 | 0.999 | 15.55 | -0.07 |
| close | m05_joint_grw_smile_supremacy_w040 | +O4.5 | 1285 | 595.07 | +70.32 | 16.22 | 1.578 | -44.60 | 15 | 47 | 12.77 | 93.73 | 0.79 | 0.999 | 15.60 | -0.01 |
| close | m05_joint_grw_smile_supremacy_w040 | +BTTS_yes | 1300 | 537.29 | +12.54 | 15.39 | 1.489 | -43.00 | 16 | 62 | 51.61 | 2.09 | 3.26 | 0.994 | 15.84 | +0.23 |
| close | m05_joint_grw_smile_supremacy_w040 | +BTTS_no | 1291 | 514.31 | -10.44 | 15.29 | 1.474 | -43.37 | 13 | 53 | 39.62 | -7.70 | 1.54 | 0.998 | 15.65 | +0.04 |
| close | m05_joint_grw_smile_supremacy_w040 | +all_unders | 1503 | 561.95 | +37.20 | 14.96 | 1.593 | -38.73 | 22 | 265 | 40.00 | 6.81 | 8.38 | 0.967 | 15.71 | +0.10 |
| close | m05_joint_grw_smile_supremacy_w040 | +all_fringe | 1833 | 614.24 | +89.49 | 14.30 | 1.675 | -37.17 | 34 | 595 | 38.32 | 8.50 | 17.39 | 0.939 | 15.52 | -0.09 |
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
| t25 | m05_joint_grw_smile_supremacy_w040 | P0 Option B | 1087 | 479.44 | +0.00 | 16.80 | 1.322 | -43.62 | 6 | 0 | n/a | n/a | 0.00 | 1.000 | 16.80 | +0.00 |
| t25 | m05_joint_grw_smile_supremacy_w040 | P0 ext book | 1085 | 472.87 | +0.00 | 16.75 | 1.313 | -43.56 | 5 | 0 | n/a | n/a | 0.00 | 1.000 | 16.75 | +0.00 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U0.5 | 1110 | 450.29 | -22.58 | 16.41 | 1.284 | -43.56 | 6 | 25 | 0.00 | -100.00 | 0.26 | 1.000 | 16.71 | -0.04 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U1.5 | 1146 | 481.90 | +9.03 | 16.73 | 1.319 | -42.60 | 5 | 61 | 22.95 | 4.49 | 1.39 | 0.997 | 16.90 | +0.15 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U3.5 | 1168 | 438.32 | -34.55 | 15.45 | 1.290 | -41.63 | 13 | 83 | 63.86 | -11.94 | 5.71 | 0.985 | 17.11 | +0.36 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +U4.5 | 1113 | 514.34 | +41.47 | 16.80 | 1.364 | -43.12 | 6 | 28 | 96.43 | 16.41 | 2.95 | 0.999 | 16.81 | +0.06 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +O2.5 | 1172 | 491.23 | +18.36 | 16.42 | 1.334 | -43.63 | 11 | 87 | 51.72 | 10.23 | 4.18 | 0.992 | 16.70 | -0.06 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +O3.5 | 1127 | 478.71 | +5.84 | 16.63 | 1.318 | -43.46 | 9 | 42 | 28.57 | 8.07 | 1.12 | 1.001 | 16.73 | -0.03 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +O4.5 | 1096 | 492.16 | +19.29 | 16.98 | 1.336 | -43.56 | 5 | 11 | 9.09 | 131.39 | 0.21 | 1.000 | 16.74 | -0.01 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +BTTS_yes | 1155 | 417.27 | -55.60 | 15.30 | 1.239 | -42.91 | 9 | 70 | 58.57 | -27.53 | 4.50 | 0.997 | 17.32 | +0.56 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +BTTS_no | 1144 | 517.08 | +44.22 | 16.91 | 1.368 | -45.16 | 5 | 59 | 45.76 | 14.33 | 2.86 | 0.995 | 16.98 | +0.23 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +all_unders | 1282 | 440.37 | -32.49 | 15.01 | 1.291 | -40.42 | 16 | 197 | 47.72 | -4.49 | 9.60 | 0.976 | 17.08 | +0.33 |
| t25 | m05_joint_grw_smile_supremacy_w040 | +all_fringe | 1551 | 458.44 | -14.42 | 13.87 | 1.327 | -41.74 | 23 | 466 | 47.21 | -2.42 | 19.35 | 0.953 | 17.77 | +1.02 |

## Smile vs baseline under each policy (paired slate log growth)

| environment | policy | smile_return_pct | baseline_return_pct | smile_roi_pct | baseline_roi_pct | n_slates | delta_log_growth_per_slate | lo | hi | p_better | delta_total_log_growth |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | P0 Option B | 545.01 | 385.78 | 15.82 | 11.68 | 100 | +0.00284 | -0.00835 | +0.01443 | 0.687 | +0.284 |
| close | +U0.5 | 507.36 | 311.74 | 15.33 | 10.57 | 100 | +0.00389 | -0.00722 | +0.01551 | 0.749 | +0.389 |
| close | +U1.5 | 547.22 | 488.13 | 15.61 | 12.52 | 100 | +0.00096 | -0.01064 | +0.01289 | 0.568 | +0.096 |
| close | +U3.5 | 556.43 | 389.78 | 15.36 | 11.23 | 100 | +0.00293 | -0.00850 | +0.01496 | 0.690 | +0.293 |
| close | +U4.5 | 546.57 | 425.03 | 15.57 | 11.87 | 100 | +0.00208 | -0.00891 | +0.01352 | 0.642 | +0.208 |
| close | +O2.5 | 555.23 | 346.21 | 15.52 | 10.77 | 100 | +0.00384 | -0.00795 | +0.01630 | 0.732 | +0.384 |
| close | +O3.5 | 506.64 | 345.81 | 14.97 | 10.66 | 100 | +0.00308 | -0.00859 | +0.01581 | 0.689 | +0.308 |
| close | +O4.5 | 595.07 | 311.14 | 16.22 | 10.58 | 100 | +0.00525 | -0.00582 | +0.01702 | 0.820 | +0.525 |
| close | +BTTS_yes | 537.29 | 339.93 | 15.39 | 10.93 | 100 | +0.00371 | -0.00741 | +0.01543 | 0.737 | +0.371 |
| close | +BTTS_no | 514.31 | 347.57 | 15.29 | 11.05 | 100 | +0.00317 | -0.00794 | +0.01479 | 0.706 | +0.317 |
| close | +all_unders | 561.95 | 487.78 | 14.96 | 11.70 | 100 | +0.00119 | -0.01073 | +0.01340 | 0.582 | +0.119 |
| close | +all_fringe | 614.24 | 299.00 | 14.30 | 8.87 | 100 | +0.00582 | -0.00722 | +0.01922 | 0.811 | +0.582 |
| t25 | P0 Option B | 479.44 | 531.78 | 16.80 | 14.15 | 99 | -0.00087 | -0.01207 | +0.01074 | 0.441 | -0.086 |
| t25 | +U0.5 | 450.29 | 450.88 | 16.41 | 13.20 | 99 | -0.00001 | -0.01128 | +0.01163 | 0.497 | -0.001 |
| t25 | +U1.5 | 481.90 | 519.76 | 16.73 | 13.76 | 99 | -0.00064 | -0.01211 | +0.01118 | 0.458 | -0.063 |
| t25 | +U3.5 | 438.32 | 426.59 | 15.45 | 12.41 | 99 | +0.00022 | -0.01107 | +0.01175 | 0.517 | +0.022 |
| t25 | +U4.5 | 514.34 | 523.46 | 16.80 | 13.67 | 99 | -0.00015 | -0.01153 | +0.01156 | 0.492 | -0.015 |
| t25 | +O2.5 | 491.23 | 562.16 | 16.42 | 13.63 | 99 | -0.00114 | -0.01250 | +0.01069 | 0.423 | -0.113 |
| t25 | +O3.5 | 478.71 | 420.77 | 16.63 | 12.41 | 99 | +0.00107 | -0.01079 | +0.01343 | 0.563 | +0.105 |
| t25 | +O4.5 | 492.16 | 509.18 | 16.98 | 13.85 | 99 | -0.00029 | -0.01153 | +0.01136 | 0.484 | -0.028 |
| t25 | +BTTS_yes | 417.27 | 455.75 | 15.30 | 12.92 | 99 | -0.00072 | -0.01194 | +0.01088 | 0.451 | -0.072 |
| t25 | +BTTS_no | 517.08 | 590.61 | 16.91 | 14.41 | 99 | -0.00114 | -0.01245 | +0.01047 | 0.425 | -0.113 |
| t25 | +all_unders | 440.37 | 380.12 | 15.01 | 11.28 | 99 | +0.00119 | -0.01031 | +0.01302 | 0.581 | +0.118 |
| t25 | +all_fringe | 458.44 | 350.31 | 13.87 | 9.86 | 99 | +0.00217 | -0.00973 | +0.01458 | 0.633 | +0.215 |

## Added families, by policy

| environment | policy | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp | odds_mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| close | +BTTS_no | BTTS_btts_no | m05_joint_grw_baseline | 69 | 42.03 | -7.06 | 2.87 | 3.52 | 2.28 |
| close | +BTTS_no | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 53 | 39.62 | -7.70 | 1.54 | 2.54 | 2.36 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 69 | 42.03 | -6.82 | 2.17 | 3.52 | 2.28 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 53 | 39.62 | -7.50 | 1.31 | 2.54 | 2.36 |
| close | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_baseline | 52 | 55.77 | 2.03 | 2.84 | 1.85 | 1.80 |
| close | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 62 | 51.61 | 2.09 | 3.26 | 1.78 | 1.87 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 52 | 55.77 | 2.05 | 2.03 | 1.85 | 1.80 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 62 | 51.61 | -0.05 | 2.70 | 1.78 | 1.87 |
| close | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 3.70 | -63.62 | 1.20 | 1.78 | 16.13 |
| close | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 53 | 3.77 | -41.80 | 0.42 | 4.17 | 18.06 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 3.70 | -65.50 | 0.88 | 1.78 | 16.13 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 53 | 3.77 | -41.77 | 0.33 | 4.17 | 18.06 |
| close | +all_unders | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 3.70 | -64.46 | 1.05 | 1.78 | 16.13 |
| close | +all_unders | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 53 | 3.77 | -41.42 | 0.37 | 4.17 | 18.06 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 23.81 | 37.66 | 3.51 | 4.54 | 4.75 |
| close | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 93 | 19.35 | 15.85 | 1.85 | 3.73 | 5.07 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 23.81 | 32.75 | 2.61 | 4.54 | 4.75 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 93 | 19.35 | 13.96 | 1.51 | 3.73 | 5.07 |
| close | +all_unders | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 23.81 | 36.75 | 3.13 | 4.54 | 4.75 |
| close | +all_unders | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 93 | 19.35 | 14.32 | 1.69 | 3.73 | 5.07 |
| close | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_baseline | 104 | 50.96 | 0.89 | 6.33 | 5.89 | 1.93 |
| close | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 73 | 50.68 | 19.61 | 3.24 | 3.79 | 2.11 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 104 | 50.96 | 1.50 | 5.03 | 5.89 | 1.93 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 73 | 50.68 | 18.78 | 2.82 | 3.79 | 2.11 |
| close | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_baseline | 120 | 36.67 | 2.94 | 7.49 | 5.52 | 3.36 |
| close | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 95 | 27.37 | -5.02 | 2.79 | 3.89 | 3.71 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 120 | 36.67 | 2.28 | 6.00 | 5.52 | 3.36 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 95 | 27.37 | -5.67 | 2.43 | 3.89 | 3.71 |
| close | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 74.75 | 2.43 | 7.00 | 5.85 | 1.50 |
| close | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 88 | 64.77 | 2.38 | 4.75 | 2.48 | 1.56 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 74.75 | 2.75 | 5.50 | 5.85 | 1.50 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 88 | 64.77 | 2.39 | 4.00 | 2.48 | 1.56 |
| close | +all_unders | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 74.75 | 2.34 | 6.45 | 5.85 | 1.50 |
| close | +all_unders | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 88 | 64.77 | 1.99 | 4.50 | 2.48 | 1.56 |
| close | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_baseline | 54 | 9.26 | -30.85 | 1.99 | 3.66 | 7.01 |
| close | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 47 | 12.77 | 93.73 | 0.79 | 4.60 | 7.92 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 54 | 9.26 | -32.89 | 1.47 | 3.66 | 7.01 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 47 | 12.77 | 101.05 | 0.66 | 4.60 | 7.92 |
| close | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 96.43 | 18.79 | 3.03 | 4.49 | 1.23 |
| close | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 31 | 93.55 | 21.77 | 1.99 | -0.24 | 1.25 |
| close | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 96.43 | 18.53 | 2.25 | 4.49 | 1.23 |
| close | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 31 | 93.55 | 21.49 | 1.62 | -0.24 | 1.25 |
| close | +all_unders | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 96.43 | 18.55 | 2.70 | 4.49 | 1.23 |
| close | +all_unders | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 31 | 93.55 | 21.67 | 1.81 | -0.24 | 1.25 |
| t25 | +BTTS_no | BTTS_btts_no | m05_joint_grw_baseline | 82 | 50.00 | 13.02 | 4.05 | 3.99 | 2.28 |
| t25 | +BTTS_no | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 59 | 45.76 | 14.33 | 2.86 | 3.18 | 2.40 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 82 | 50.00 | 14.01 | 3.00 | 3.99 | 2.28 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 59 | 45.76 | 12.48 | 2.41 | 3.18 | 2.40 |
| t25 | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_baseline | 73 | 60.27 | -12.37 | 4.39 | 3.07 | 1.79 |
| t25 | +BTTS_yes | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 70 | 58.57 | -27.53 | 4.50 | 2.08 | 1.88 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 73 | 60.27 | -12.94 | 3.23 | 3.07 | 1.79 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 70 | 58.57 | -26.50 | 3.78 | 2.08 | 1.88 |
| t25 | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 2.70 | -89.53 | 0.71 | 2.27 | 16.94 |
| t25 | +U0.5 | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 25 | 0.00 | -100.00 | 0.26 | 4.49 | 19.86 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 2.70 | -90.05 | 0.49 | 2.27 | 16.94 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 25 | 0.00 | -100.00 | 0.21 | 4.49 | 19.86 |
| t25 | +all_unders | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 2.70 | -89.53 | 0.61 | 2.27 | 16.94 |
| t25 | +all_unders | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 25 | 0.00 | -100.00 | 0.24 | 4.49 | 19.86 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 23.68 | 3.90 | 3.11 | 4.86 | 4.45 |
| t25 | +U1.5 | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 61 | 22.95 | 4.49 | 1.39 | 3.71 | 4.86 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 23.68 | 2.41 | 2.35 | 4.86 | 4.45 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 61 | 22.95 | 3.83 | 1.14 | 3.71 | 4.86 |
| t25 | +all_unders | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 23.68 | 1.33 | 2.82 | 4.86 | 4.45 |
| t25 | +all_unders | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 61 | 22.95 | 3.89 | 1.28 | 3.71 | 4.86 |
| t25 | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_baseline | 116 | 58.62 | 10.77 | 8.55 | 5.74 | 1.93 |
| t25 | +O2.5 | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 87 | 51.72 | 10.23 | 4.18 | 3.13 | 2.10 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 116 | 58.62 | 10.25 | 6.72 | 5.74 | 1.93 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 87 | 51.72 | 8.02 | 3.59 | 3.13 | 2.10 |
| t25 | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_baseline | 75 | 26.67 | -23.70 | 4.70 | 4.99 | 3.14 |
| t25 | +O3.5 | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 28.57 | 8.07 | 1.12 | 3.32 | 3.50 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 75 | 26.67 | -20.39 | 3.50 | 4.99 | 3.14 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 28.57 | 10.79 | 0.93 | 3.32 | 3.50 |
| t25 | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 56.72 | -15.00 | 6.16 | 6.01 | 1.51 |
| t25 | +U3.5 | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 83 | 63.86 | -11.94 | 5.71 | 2.24 | 1.54 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 56.72 | -16.23 | 4.66 | 6.01 | 1.51 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 83 | 63.86 | -12.71 | 4.71 | 2.24 | 1.54 |
| t25 | +all_unders | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 56.72 | -16.13 | 5.69 | 6.01 | 1.51 |
| t25 | +all_unders | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 83 | 63.86 | -12.38 | 5.40 | 2.24 | 1.54 |
| t25 | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_baseline | 21 | 4.76 | -9.29 | 0.68 | 3.24 | 6.60 |
| t25 | +O4.5 | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 11 | 9.09 | 131.39 | 0.21 | 4.69 | 8.34 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 21 | 4.76 | 1.13 | 0.50 | 3.24 | 6.60 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 11 | 9.09 | 147.10 | 0.17 | 4.69 | 8.34 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 95.65 | 10.43 | 3.64 | 4.84 | 1.22 |
| t25 | +U4.5 | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 28 | 96.43 | 16.41 | 2.95 | 0.00 | 1.24 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 95.65 | 8.81 | 2.81 | 4.84 | 1.22 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 28 | 96.43 | 15.40 | 2.41 | 0.00 | 1.24 |
| t25 | +all_unders | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 95.65 | 9.82 | 3.29 | 4.84 | 1.22 |
| t25 | +all_unders | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 28 | 96.43 | 15.80 | 2.68 | 0.00 | 1.24 |

## Staked-bet calibration by family (the Jensen check)

Positive `model_minus_realised` = the staked bets were priced above their realised rate. Staked bets are a positive-edge selection, not the unconditional forecast.

| environment | policy | family | model | n_bets | mean_p_model | mean_p_market | realised_win_rate | model_minus_realised | roi_pct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| close | +all_fringe | 1X2_away | m05_joint_grw_baseline | 399 | 0.3260 | 0.2703 | 0.2632 | +0.0628 | 8.30 |
| close | +all_fringe | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 389 | 0.3166 | 0.2643 | 0.2596 | +0.0570 | 19.43 |
| close | +all_fringe | 1X2_draw | m05_joint_grw_baseline | 278 | 0.2490 | 0.2444 | 0.2446 | +0.0044 | -1.27 |
| close | +all_fringe | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 286 | 0.2465 | 0.2445 | 0.2413 | +0.0052 | 5.21 |
| close | +all_fringe | 1X2_home | m05_joint_grw_baseline | 307 | 0.3984 | 0.3488 | 0.3941 | +0.0043 | 19.55 |
| close | +all_fringe | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 311 | 0.3912 | 0.3403 | 0.3859 | +0.0054 | 18.43 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 69 | 0.4761 | 0.4409 | 0.4203 | +0.0558 | -6.82 |
| close | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 53 | 0.4499 | 0.4244 | 0.3962 | +0.0536 | -7.50 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 52 | 0.5771 | 0.5585 | 0.5577 | +0.0194 | 2.05 |
| close | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 62 | 0.5561 | 0.5382 | 0.5161 | +0.0399 | -0.05 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 81 | 0.0844 | 0.0665 | 0.0370 | +0.0473 | -65.50 |
| close | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 53 | 0.1009 | 0.0592 | 0.0377 | +0.0631 | -41.77 |
| close | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_baseline | 63 | 0.7936 | 0.7586 | 0.7619 | +0.0317 | 2.14 |
| close | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 56 | 0.7423 | 0.7332 | 0.6964 | +0.0459 | 1.95 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 105 | 0.2667 | 0.2213 | 0.2381 | +0.0286 | 32.75 |
| close | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 93 | 0.2422 | 0.2049 | 0.1935 | +0.0486 | 13.96 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 104 | 0.5800 | 0.5211 | 0.5096 | +0.0704 | 1.50 |
| close | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 73 | 0.5143 | 0.4764 | 0.5068 | +0.0075 | 18.78 |
| close | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_baseline | 199 | 0.5180 | 0.4654 | 0.4874 | +0.0305 | 12.18 |
| close | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 196 | 0.4776 | 0.4453 | 0.4592 | +0.0184 | 8.39 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 120 | 0.3586 | 0.3035 | 0.3667 | -0.0080 | 2.28 |
| close | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 95 | 0.3130 | 0.2742 | 0.2737 | +0.0394 | -5.67 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 99 | 0.7286 | 0.6701 | 0.7475 | -0.0189 | 2.75 |
| close | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 88 | 0.6673 | 0.6425 | 0.6477 | +0.0196 | 2.39 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 54 | 0.1879 | 0.1513 | 0.0926 | +0.0953 | -32.89 |
| close | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 47 | 0.1797 | 0.1337 | 0.1277 | +0.0520 | 101.05 |
| close | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 28 | 0.8607 | 0.8158 | 0.9643 | -0.1036 | 18.53 |
| close | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 31 | 0.8000 | 0.8025 | 0.9355 | -0.1354 | 21.49 |
| close | P0 Option B | 1X2_away | m05_joint_grw_baseline | 403 | 0.3263 | 0.2713 | 0.2655 | +0.0608 | 7.77 |
| close | P0 Option B | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 389 | 0.3166 | 0.2643 | 0.2596 | +0.0570 | 19.36 |
| close | P0 Option B | 1X2_draw | m05_joint_grw_baseline | 273 | 0.2494 | 0.2443 | 0.2418 | +0.0076 | 3.38 |
| close | P0 Option B | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 284 | 0.2465 | 0.2442 | 0.2430 | +0.0035 | 8.38 |
| close | P0 Option B | 1X2_home | m05_joint_grw_baseline | 310 | 0.3982 | 0.3492 | 0.3903 | +0.0079 | 19.99 |
| close | P0 Option B | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 311 | 0.3912 | 0.3403 | 0.3859 | +0.0054 | 17.89 |
| close | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_baseline | 64 | 0.7939 | 0.7592 | 0.7500 | +0.0439 | 1.25 |
| close | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 56 | 0.7423 | 0.7332 | 0.6964 | +0.0459 | 0.78 |
| close | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_baseline | 197 | 0.5184 | 0.4652 | 0.4924 | +0.0260 | 10.75 |
| close | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 197 | 0.4771 | 0.4449 | 0.4619 | +0.0152 | 10.32 |
| t25 | +all_fringe | 1X2_away | m05_joint_grw_baseline | 361 | 0.3276 | 0.2711 | 0.2604 | +0.0672 | 11.11 |
| t25 | +all_fringe | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 341 | 0.3185 | 0.2634 | 0.2551 | +0.0633 | 20.49 |
| t25 | +all_fringe | 1X2_draw | m05_joint_grw_baseline | 253 | 0.2503 | 0.2468 | 0.2372 | +0.0132 | -3.90 |
| t25 | +all_fringe | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 259 | 0.2469 | 0.2443 | 0.2548 | -0.0079 | 9.36 |
| t25 | +all_fringe | 1X2_home | m05_joint_grw_baseline | 281 | 0.3968 | 0.3479 | 0.3665 | +0.0303 | 20.85 |
| t25 | +all_fringe | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 283 | 0.3911 | 0.3387 | 0.3604 | +0.0306 | 18.58 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_baseline | 82 | 0.4802 | 0.4403 | 0.5000 | -0.0198 | 14.01 |
| t25 | +all_fringe | BTTS_btts_no | m05_joint_grw_smile_supremacy_w040 | 59 | 0.4491 | 0.4173 | 0.4576 | -0.0086 | 12.48 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_baseline | 73 | 0.5945 | 0.5638 | 0.6027 | -0.0082 | -12.94 |
| t25 | +all_fringe | BTTS_btts_yes | m05_joint_grw_smile_supremacy_w040 | 70 | 0.5576 | 0.5368 | 0.5857 | -0.0281 | -26.50 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_baseline | 37 | 0.0890 | 0.0663 | 0.0270 | +0.0620 | -90.05 |
| t25 | +all_fringe | O/U 0.5_under_05 | m05_joint_grw_smile_supremacy_w040 | 25 | 0.0995 | 0.0546 | 0.0000 | +0.0995 | -100.00 |
| t25 | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 0.7913 | 0.7527 | 0.8448 | -0.0535 | 18.51 |
| t25 | +all_fringe | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 48 | 0.7355 | 0.7277 | 0.7917 | -0.0562 | 9.65 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_baseline | 76 | 0.2814 | 0.2328 | 0.2368 | +0.0445 | 2.41 |
| t25 | +all_fringe | O/U 1.5_under_15 | m05_joint_grw_smile_supremacy_w040 | 61 | 0.2474 | 0.2103 | 0.2295 | +0.0178 | 3.83 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_baseline | 116 | 0.5787 | 0.5213 | 0.5862 | -0.0075 | 10.25 |
| t25 | +all_fringe | O/U 2.5_over_25 | m05_joint_grw_smile_supremacy_w040 | 87 | 0.5084 | 0.4772 | 0.5172 | -0.0088 | 8.02 |
| t25 | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 0.5256 | 0.4709 | 0.5294 | -0.0038 | 18.46 |
| t25 | +all_fringe | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 154 | 0.4740 | 0.4427 | 0.5000 | -0.0260 | 16.98 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_baseline | 75 | 0.3746 | 0.3247 | 0.2667 | +0.1080 | -20.39 |
| t25 | +all_fringe | O/U 3.5_over_35 | m05_joint_grw_smile_supremacy_w040 | 42 | 0.3263 | 0.2931 | 0.2857 | +0.0406 | 10.79 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_baseline | 67 | 0.7260 | 0.6660 | 0.5672 | +0.1589 | -16.23 |
| t25 | +all_fringe | O/U 3.5_under_35 | m05_joint_grw_smile_supremacy_w040 | 83 | 0.6756 | 0.6532 | 0.6386 | +0.0370 | -12.71 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_baseline | 21 | 0.1904 | 0.1580 | 0.0476 | +0.1428 | 1.13 |
| t25 | +all_fringe | O/U 4.5_over_45 | m05_joint_grw_smile_supremacy_w040 | 11 | 0.1710 | 0.1241 | 0.0909 | +0.0801 | 147.10 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_baseline | 23 | 0.8726 | 0.8242 | 0.9565 | -0.0840 | 8.81 |
| t25 | +all_fringe | O/U 4.5_under_45 | m05_joint_grw_smile_supremacy_w040 | 28 | 0.8112 | 0.8112 | 0.9643 | -0.1531 | 15.40 |
| t25 | P0 Option B | 1X2_away | m05_joint_grw_baseline | 363 | 0.3285 | 0.2723 | 0.2645 | +0.0641 | 8.44 |
| t25 | P0 Option B | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 341 | 0.3185 | 0.2634 | 0.2551 | +0.0633 | 18.40 |
| t25 | P0 Option B | 1X2_draw | m05_joint_grw_baseline | 252 | 0.2504 | 0.2467 | 0.2381 | +0.0123 | 0.86 |
| t25 | P0 Option B | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 259 | 0.2469 | 0.2443 | 0.2548 | -0.0079 | 9.79 |
| t25 | P0 Option B | 1X2_home | m05_joint_grw_baseline | 281 | 0.3968 | 0.3479 | 0.3665 | +0.0303 | 20.76 |
| t25 | P0 Option B | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 283 | 0.3911 | 0.3387 | 0.3604 | +0.0306 | 17.88 |
| t25 | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 0.7913 | 0.7527 | 0.8448 | -0.0535 | 18.60 |
| t25 | P0 Option B | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 49 | 0.7365 | 0.7290 | 0.7959 | -0.0594 | 9.70 |
| t25 | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 0.5256 | 0.4709 | 0.5294 | -0.0038 | 17.81 |
| t25 | P0 Option B | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 155 | 0.4743 | 0.4431 | 0.5032 | -0.0290 | 16.94 |
