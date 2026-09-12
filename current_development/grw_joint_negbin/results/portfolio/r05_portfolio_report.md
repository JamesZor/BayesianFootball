# r05 closing-line portfolio — Task 014 (JointGammaNegBinObservation)

Generated 2026-09-12 07:10. Contract: `MatchDay.option_b_system()`. Book: de-vigged Betfair TWA(−20, 0] close. Panel: 632 fixtures buildable by every arm.

Closing-line simulation, not a tradeable-price test. Growth intervals of neighbouring arms overlap at this sample size; read the attribution, not the ranking.

## Headline

| model | likelihood | n_bets | total_return_pct | roi_pct | sharpe_ann | calmar | max_drawdown_pct | win_rate_pct | capture_ratio | mean_edge_pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_poisson | Poisson | 1296 | 491.55 | 12.32 | 1.412 | 12.652 | -38.85 | 34.80 | 1.036 | 4.78 |
| m00_baseline_grw_negbin | NegBin | 1300 | 461.37 | 12.05 | 1.353 | 11.896 | -38.78 | 34.15 | 1.017 | 4.75 |
| m05_poisson | Joint Gamma-Poisson | 1247 | 385.78 | 11.68 | 1.453 | 9.042 | -42.67 | 35.20 | 1.080 | 4.13 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 1235 | 384.62 | 11.64 | 1.434 | 9.053 | -42.49 | 34.90 | 1.041 | 4.22 |
| m10_lineup_grw_negbin | NegBin | 1301 | 370.28 | 11.02 | 1.243 | 8.203 | -45.14 | 34.67 | 0.924 | 4.87 |
| m12_poisson | Joint Gamma-Poisson | 1253 | 351.90 | 11.36 | 1.309 | 6.689 | -52.61 | 34.40 | 1.043 | 4.29 |
| m10_poisson | Poisson | 1299 | 348.82 | 10.75 | 1.214 | 7.266 | -48.01 | 34.95 | 0.957 | 4.88 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1244 | 297.52 | 10.50 | 1.196 | 5.923 | -50.23 | 34.16 | 1.007 | 4.37 |

## Shared-bet sizing and overlap (Task 012)

On the shared set the fixture, selection, price and outcome are identical, so `sizing_delta_pnl` is attributable to stake size alone.

| pair | n_shared | n_only_a | n_only_b | overlap_pct | stake_mean_a | stake_mean_b | sizing_delta_pnl | shared_roi_a_pct | shared_roi_b_pct | capture_ratio_a | capture_ratio_b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin vs m00_poisson | 1258 | 42 | 38 | 94.02 | 0.01461 | 0.01455 | -0.0376 | 12.09 | 12.34 | 1.017 | 1.036 |
| m05_wealth_grw_negbin vs m05_poisson | 1188 | 47 | 59 | 91.81 | 0.01404 | 0.01392 | +0.0136 | 11.71 | 11.73 | 1.041 | 1.080 |
| m10_lineup_grw_negbin vs m10_poisson | 1255 | 46 | 44 | 93.31 | 0.01452 | 0.01449 | +0.0470 | 10.96 | 10.73 | 0.924 | 0.957 |
| m12_joint_hybrid_synergy_negbin vs m12_poisson | 1209 | 35 | 44 | 93.87 | 0.01399 | 0.01386 | -0.1270 | 10.54 | 11.40 | 1.007 | 1.043 |
| m12_joint_hybrid_synergy_negbin vs m05_wealth_grw_negbin | 1132 | 112 | 103 | 84.04 | 0.01471 | 0.01444 | -0.0851 | 11.06 | 11.79 | 1.007 | 1.041 |
| m10_lineup_grw_negbin vs m00_baseline_grw_negbin | 1194 | 107 | 106 | 84.86 | 0.01505 | 0.01511 | -0.4337 | 10.61 | 12.97 | 0.924 | 1.017 |

## Shared and exclusive bet sets

| pair | bet_set | owner | n_bets | win_rate_pct | cap_weighted_win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin vs m00_poisson | shared | m00_baseline_grw_negbin | 1258 | 33.86 | 35.54 | 12.09 | 4.98 | 1.037 |
| m00_baseline_grw_negbin vs m00_poisson | shared | m00_poisson | 1258 | 33.86 | 36.06 | 12.34 | 4.90 | 1.058 |
| m00_baseline_grw_negbin vs m00_poisson | exclusive | m00_baseline_grw_negbin | 42 | 42.86 | 39.93 | 1.09 | -1.99 | n/a |
| m00_baseline_grw_negbin vs m00_poisson | exclusive | m00_poisson | 38 | 65.79 | 71.88 | 5.94 | 0.72 | n/a |
| m05_wealth_grw_negbin vs m05_poisson | shared | m05_wealth_grw_negbin | 1188 | 34.85 | 37.01 | 11.71 | 4.41 | 1.044 |
| m05_wealth_grw_negbin vs m05_poisson | shared | m05_poisson | 1188 | 34.85 | 37.68 | 11.73 | 4.34 | 1.068 |
| m05_wealth_grw_negbin vs m05_poisson | exclusive | m05_wealth_grw_negbin | 47 | 36.17 | 34.47 | -9.88 | -0.74 | n/a |
| m05_wealth_grw_negbin vs m05_poisson | exclusive | m05_poisson | 59 | 42.37 | 69.34 | 2.55 | -0.12 | n/a |
| m10_lineup_grw_negbin vs m10_poisson | shared | m10_lineup_grw_negbin | 1255 | 34.02 | 33.93 | 10.96 | 5.10 | 0.965 |
| m10_lineup_grw_negbin vs m10_poisson | shared | m10_poisson | 1255 | 34.02 | 34.20 | 10.73 | 5.04 | 0.974 |
| m10_lineup_grw_negbin vs m10_poisson | exclusive | m10_lineup_grw_negbin | 46 | 52.17 | 53.98 | 26.98 | -1.32 | n/a |
| m10_lineup_grw_negbin vs m10_poisson | exclusive | m10_poisson | 44 | 61.36 | 77.05 | 15.59 | 0.54 | n/a |
| m12_joint_hybrid_synergy_negbin vs m12_poisson | shared | m12_joint_hybrid_synergy_negbin | 1209 | 34.00 | 35.38 | 10.54 | 4.53 | 1.023 |
| m12_joint_hybrid_synergy_negbin vs m12_poisson | shared | m12_poisson | 1209 | 34.00 | 36.19 | 11.40 | 4.45 | 1.048 |
| m12_joint_hybrid_synergy_negbin vs m12_poisson | exclusive | m12_joint_hybrid_synergy_negbin | 35 | 40.00 | 42.37 | -0.75 | -1.07 | n/a |
| m12_joint_hybrid_synergy_negbin vs m12_poisson | exclusive | m12_poisson | 44 | 45.45 | 66.84 | 3.18 | -0.31 | n/a |
| m12_joint_hybrid_synergy_negbin vs m05_wealth_grw_negbin | shared | m12_joint_hybrid_synergy_negbin | 1132 | 34.10 | 35.52 | 11.06 | 4.79 | 1.012 |
| m12_joint_hybrid_synergy_negbin vs m05_wealth_grw_negbin | shared | m05_wealth_grw_negbin | 1132 | 34.10 | 36.80 | 11.79 | 4.57 | 1.069 |
| m12_joint_hybrid_synergy_negbin vs m05_wealth_grw_negbin | exclusive | m12_joint_hybrid_synergy_negbin | 112 | 34.82 | 29.64 | -18.14 | 0.15 | 0.515 |
| m12_joint_hybrid_synergy_negbin vs m05_wealth_grw_negbin | exclusive | m05_wealth_grw_negbin | 103 | 43.69 | 45.43 | 5.41 | 0.34 | 2.309 |
| m10_lineup_grw_negbin vs m00_baseline_grw_negbin | shared | m10_lineup_grw_negbin | 1194 | 34.25 | 33.83 | 10.61 | 5.32 | 0.935 |
| m10_lineup_grw_negbin vs m00_baseline_grw_negbin | shared | m00_baseline_grw_negbin | 1194 | 34.25 | 35.61 | 12.97 | 5.16 | 1.000 |
| m10_lineup_grw_negbin vs m00_baseline_grw_negbin | exclusive | m10_lineup_grw_negbin | 107 | 39.25 | 43.79 | 34.29 | -0.14 | n/a |
| m10_lineup_grw_negbin vs m00_baseline_grw_negbin | exclusive | m00_baseline_grw_negbin | 106 | 33.02 | 32.66 | -29.45 | 0.15 | n/a |

## Return by selection family

Option B stakes 1X2 home/draw/away, Under 2.5 and Over 1.5. The two totals rows are where the likelihoods are expected to differ; the 1X2 rows are the control.

| model | likelihood | selection_family | n_bets | win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | NegBin | 1X2_away | 400 | 26.00 | 17.85 | 6.39 | 0.960 |
| m00_poisson | Poisson | 1X2_away | 388 | 24.48 | 18.44 | 6.39 | 1.079 |
| m05_poisson | Joint Gamma-Poisson | 1X2_away | 403 | 26.55 | 7.77 | 5.50 | 0.932 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 1X2_away | 420 | 27.38 | 8.29 | 5.43 | 0.875 |
| m10_lineup_grw_negbin | NegBin | 1X2_away | 390 | 26.41 | 14.04 | 6.86 | 0.887 |
| m10_poisson | Poisson | 1X2_away | 378 | 25.40 | 14.78 | 6.94 | 0.967 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1X2_away | 421 | 27.32 | 9.30 | 5.77 | 0.833 |
| m12_poisson | Joint Gamma-Poisson | 1X2_away | 413 | 26.88 | 10.03 | 5.70 | 0.878 |
| m00_baseline_grw_negbin | NegBin | 1X2_draw | 300 | 24.00 | 7.92 | 0.50 | 2.461 |
| m00_poisson | Poisson | 1X2_draw | 306 | 24.51 | 7.89 | 0.59 | 2.041 |
| m05_poisson | Joint Gamma-Poisson | 1X2_draw | 273 | 24.18 | 3.38 | 0.51 | 1.317 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 1X2_draw | 244 | 24.59 | 4.72 | 0.61 | 1.108 |
| m10_lineup_grw_negbin | NegBin | 1X2_draw | 315 | 25.71 | 5.30 | 0.46 | 1.612 |
| m10_poisson | Poisson | 1X2_draw | 326 | 25.46 | 3.92 | 0.55 | 1.518 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1X2_draw | 260 | 23.85 | 1.48 | 0.70 | 1.198 |
| m12_poisson | Joint Gamma-Poisson | 1X2_draw | 282 | 24.47 | 0.55 | 0.65 | 1.211 |
| m00_baseline_grw_negbin | NegBin | 1X2_home | 336 | 38.99 | 11.14 | 6.27 | 0.778 |
| m00_poisson | Poisson | 1X2_home | 328 | 39.02 | 11.79 | 6.47 | 0.810 |
| m05_poisson | Joint Gamma-Poisson | 1X2_home | 310 | 39.03 | 19.99 | 4.90 | 0.953 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 1X2_home | 315 | 39.05 | 19.57 | 4.84 | 0.913 |
| m10_lineup_grw_negbin | NegBin | 1X2_home | 330 | 39.70 | 12.34 | 6.79 | 0.745 |
| m10_poisson | Poisson | 1X2_home | 320 | 39.38 | 11.56 | 7.00 | 0.770 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1X2_home | 308 | 37.34 | 16.86 | 5.17 | 0.935 |
| m12_poisson | Joint Gamma-Poisson | 1X2_home | 301 | 37.21 | 18.12 | 5.32 | 0.964 |
| m00_baseline_grw_negbin | NegBin | O/U 1.5_over_15 | 33 | 69.70 | -0.05 | 3.38 | 1.439 |
| m00_poisson | Poisson | O/U 1.5_over_15 | 50 | 78.00 | 3.65 | 3.49 | 1.088 |
| m05_poisson | Joint Gamma-Poisson | O/U 1.5_over_15 | 64 | 75.00 | 1.25 | 3.47 | 1.279 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | O/U 1.5_over_15 | 46 | 71.74 | 0.63 | 3.30 | 1.535 |
| m10_lineup_grw_negbin | NegBin | O/U 1.5_over_15 | 27 | 66.67 | -2.29 | 2.93 | 1.980 |
| m10_poisson | Poisson | O/U 1.5_over_15 | 47 | 78.72 | 1.64 | 2.97 | 1.166 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | O/U 1.5_over_15 | 45 | 71.11 | 0.68 | 3.15 | 1.430 |
| m12_poisson | Joint Gamma-Poisson | O/U 1.5_over_15 | 60 | 73.33 | 1.29 | 3.32 | 1.220 |
| m00_baseline_grw_negbin | NegBin | O/U 2.5_under_25 | 231 | 49.35 | 8.07 | 5.42 | 1.066 |
| m00_poisson | Poisson | O/U 2.5_under_25 | 224 | 50.89 | 7.54 | 5.51 | 0.997 |
| m05_poisson | Joint Gamma-Poisson | O/U 2.5_under_25 | 197 | 49.24 | 10.75 | 5.32 | 1.142 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | O/U 2.5_under_25 | 210 | 47.62 | 8.91 | 5.26 | 1.203 |
| m10_lineup_grw_negbin | NegBin | O/U 2.5_under_25 | 239 | 49.37 | 7.61 | 5.00 | 0.988 |
| m10_poisson | Poisson | O/U 2.5_under_25 | 228 | 49.12 | 7.77 | 5.09 | 1.044 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | O/U 2.5_under_25 | 210 | 48.10 | 7.59 | 5.19 | 1.127 |
| m12_poisson | Joint Gamma-Poisson | O/U 2.5_under_25 | 197 | 48.22 | 10.12 | 5.24 | 1.167 |
