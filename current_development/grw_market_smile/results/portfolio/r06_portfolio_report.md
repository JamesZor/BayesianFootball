# r06 closing-line portfolio — Task 015 (market-anchored MultiScaleGRW)

Generated 2026-09-13 21:18. Contract: `MatchDay.option_b_system()`. Book: de-vigged Betfair TWA(−20, 0] close. Panel: 710 walk-forward → 635 quoted → 632 buildable by every arm. Bootstrap B = 4000.

P1 reproduction — baseline +385.78% / ROI 11.68% / 1247 bets on 632 fixtures; published +385.8% / 11.68% / 1247 on 632: **reproduced**.

## Headline

| model | n_bets | n_bets_1x2 | n_bets_totals | total_return_pct | roi_pct | sharpe_ann | calmar | max_drawdown_pct | win_rate_pct | growth_lo | growth_hi | p_roi_positive | capture_ratio | mean_edge_pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 1247 | 986 | 261 | 385.78 | 11.68 | 1.453 | 9.042 | -42.67 | 35.20 | -0.0004 | +0.0323 | 0.984 | 1.080 | 4.13 |
| m05_joint_grw_supremacy_w040 | 1244 | 983 | 261 | 404.57 | 12.75 | 1.551 | 9.488 | -42.64 | 36.50 | +0.0007 | +0.0321 | 0.992 | 1.070 | 3.76 |
| m05_joint_grw_smile_supremacy_w020 | 1240 | 988 | 252 | 588.42 | 15.71 | 1.495 | 13.367 | -44.02 | 33.95 | -0.0003 | +0.0390 | 0.996 | 0.981 | 3.76 |
| m05_joint_grw_smile_supremacy_w040 | 1237 | 984 | 253 | 545.01 | 15.82 | 1.516 | 12.377 | -44.03 | 33.95 | +0.0004 | +0.0376 | 0.995 | 1.000 | 3.53 |
| m05_joint_grw_smile_supremacy_w070 | 1231 | 978 | 253 | 481.38 | 15.48 | 1.530 | 11.362 | -42.37 | 35.01 | +0.0006 | +0.0354 | 0.993 | 0.973 | 3.31 |

## Market breakdown

| level | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp | odds_mean | capture_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| family | 1X2_away | m05_joint_grw_baseline | 403 | 26.55 | 7.77 | 33.30 | 5.50 | 4.36 | 0.932 |
| family | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 381 | 25.98 | 18.75 | 34.12 | 5.72 | 4.50 | 1.053 |
| family | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 389 | 25.96 | 19.36 | 34.14 | 5.23 | 4.45 | 1.075 |
| family | 1X2_away | m05_joint_grw_smile_supremacy_w070 | 391 | 26.85 | 17.11 | 34.43 | 4.92 | 4.41 | 1.018 |
| family | 1X2_away | m05_joint_grw_supremacy_w040 | 394 | 26.65 | 16.64 | 27.42 | 4.20 | 4.39 | 1.015 |
| family | 1X2_draw | m05_joint_grw_baseline | 273 | 24.18 | 3.38 | 9.48 | 0.51 | 4.21 | 1.317 |
| family | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 302 | 23.51 | 9.57 | 11.81 | 0.26 | 4.19 | 0.944 |
| family | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 284 | 24.30 | 8.38 | 10.98 | 0.22 | 4.20 | 0.367 |
| family | 1X2_draw | m05_joint_grw_smile_supremacy_w070 | 274 | 25.55 | 7.76 | 9.99 | 0.09 | 4.21 | -0.532 |
| family | 1X2_draw | m05_joint_grw_supremacy_w040 | 250 | 28.00 | 11.06 | 8.64 | 0.70 | 4.22 | 0.705 |
| family | 1X2_home | m05_joint_grw_baseline | 310 | 39.03 | 19.99 | 33.14 | 4.90 | 3.19 | 0.953 |
| family | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 305 | 39.67 | 17.70 | 38.87 | 5.65 | 3.27 | 0.832 |
| family | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 311 | 38.59 | 17.89 | 38.43 | 5.09 | 3.24 | 0.896 |
| family | 1X2_home | m05_joint_grw_smile_supremacy_w070 | 313 | 39.30 | 18.51 | 37.77 | 4.63 | 3.23 | 0.906 |
| family | 1X2_home | m05_joint_grw_supremacy_w040 | 339 | 39.23 | 13.93 | 35.76 | 4.40 | 3.14 | 0.846 |
| family | O/U 1.5_over_15 | m05_joint_grw_baseline | 64 | 75.00 | 1.25 | 4.67 | 3.47 | 1.32 | 1.279 |
| family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 57 | 70.18 | -0.68 | 2.85 | 1.00 | 1.37 | 1.092 |
| family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 56 | 69.64 | 0.78 | 2.97 | 0.92 | 1.36 | 1.085 |
| family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w070 | 53 | 71.70 | 2.16 | 3.02 | 0.82 | 1.36 | 1.092 |
| family | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 58 | 75.86 | -0.36 | 4.90 | 3.68 | 1.31 | 1.299 |
| family | O/U 2.5_under_25 | m05_joint_grw_baseline | 197 | 49.24 | 10.75 | 19.41 | 5.32 | 2.17 | 1.142 |
| family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 195 | 46.15 | 10.67 | 12.36 | 3.18 | 2.27 | 1.135 |
| family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 197 | 46.19 | 10.32 | 13.49 | 3.22 | 2.27 | 1.130 |
| family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w070 | 200 | 47.50 | 11.87 | 14.79 | 3.17 | 2.26 | 1.100 |
| family | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 203 | 50.25 | 9.74 | 23.28 | 5.61 | 2.17 | 1.092 |
| market | 1X2 | m05_joint_grw_baseline | 986 | 29.82 | 12.55 | 75.92 | 3.93 | 3.95 | 1.025 |
| market | 1X2 | m05_joint_grw_smile_supremacy_w020 | 988 | 29.45 | 16.99 | 84.79 | 4.03 | 4.03 | 1.052 |
| market | 1X2 | m05_joint_grw_smile_supremacy_w040 | 984 | 29.47 | 17.24 | 83.54 | 3.74 | 4.00 | 1.065 |
| market | 1X2 | m05_joint_grw_smile_supremacy_w070 | 978 | 30.47 | 16.62 | 82.19 | 3.47 | 3.98 | 1.025 |
| market | 1X2 | m05_joint_grw_supremacy_w040 | 983 | 31.33 | 14.62 | 71.82 | 3.38 | 3.92 | 0.958 |
| market | totals | m05_joint_grw_baseline | 261 | 55.56 | 8.91 | 24.08 | 4.86 | 1.96 | 1.071 |
| market | totals | m05_joint_grw_smile_supremacy_w020 | 252 | 51.59 | 8.54 | 15.21 | 2.69 | 2.07 | 0.985 |
| market | totals | m05_joint_grw_smile_supremacy_w040 | 253 | 51.38 | 8.60 | 16.46 | 2.71 | 2.07 | 0.980 |
| market | totals | m05_joint_grw_smile_supremacy_w070 | 253 | 52.57 | 10.23 | 17.81 | 2.68 | 2.07 | 0.953 |
| market | totals | m05_joint_grw_supremacy_w040 | 261 | 55.94 | 7.99 | 28.18 | 5.18 | 1.98 | 1.038 |

## P2 smile routing on the staked ledger

| model | n_totals_bets | max_abs_vs_smile | min_abs_vs_grid |
|---|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w020 | 252 | 1.4e-15 | 0.00006 |
| m05_joint_grw_smile_supremacy_w040 | 253 | 1.8e-15 | 0.00013 |
| m05_joint_grw_smile_supremacy_w070 | 253 | 1.8e-15 | 0.00002 |

## Shared vs exclusive bets against the baseline

| pair | bet_set | owner | n_bets | win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---:|---:|---:|---:|---:|
| m05_joint_grw_supremacy_w040 vs baseline | shared | m05_joint_grw_supremacy_w040 | 1074 | 35.57 | 13.10 | 4.22 | 1.093 |
| m05_joint_grw_supremacy_w040 vs baseline | shared | baseline | 1074 | 35.57 | 13.54 | 4.71 | 1.064 |
| m05_joint_grw_supremacy_w040 vs baseline | exclusive | m05_joint_grw_supremacy_w040 | 170 | 42.35 | 7.22 | 0.82 | 1.507 |
| m05_joint_grw_supremacy_w040 vs baseline | exclusive | baseline | 173 | 32.95 | -25.77 | 0.50 | 1.134 |
| m05_joint_grw_smile_supremacy_w020 vs baseline | shared | m05_joint_grw_smile_supremacy_w020 | 978 | 32.00 | 15.94 | 4.44 | 1.017 |
| m05_joint_grw_smile_supremacy_w020 vs baseline | shared | baseline | 978 | 32.00 | 13.11 | 4.56 | 1.091 |
| m05_joint_grw_smile_supremacy_w020 vs baseline | exclusive | m05_joint_grw_smile_supremacy_w020 | 262 | 41.22 | 13.80 | 1.20 | 1.395 |
| m05_joint_grw_smile_supremacy_w020 vs baseline | exclusive | baseline | 269 | 46.84 | 4.44 | 2.54 | 1.495 |
| m05_joint_grw_smile_supremacy_w040 vs baseline | shared | m05_joint_grw_smile_supremacy_w040 | 960 | 31.77 | 16.35 | 4.19 | 1.051 |
| m05_joint_grw_smile_supremacy_w040 vs baseline | shared | baseline | 960 | 31.77 | 11.65 | 4.63 | 1.072 |
| m05_joint_grw_smile_supremacy_w040 vs baseline | exclusive | m05_joint_grw_smile_supremacy_w040 | 277 | 41.52 | 11.88 | 1.24 | 1.241 |
| m05_joint_grw_smile_supremacy_w040 vs baseline | exclusive | baseline | 287 | 46.69 | 11.82 | 2.44 | 1.735 |
| m05_joint_grw_smile_supremacy_w070 vs baseline | shared | m05_joint_grw_smile_supremacy_w070 | 938 | 32.52 | 16.24 | 3.94 | 1.032 |
| m05_joint_grw_smile_supremacy_w070 vs baseline | shared | baseline | 938 | 32.52 | 13.55 | 4.67 | 1.059 |
| m05_joint_grw_smile_supremacy_w070 vs baseline | exclusive | m05_joint_grw_smile_supremacy_w070 | 293 | 43.00 | 10.74 | 1.29 | 1.123 |
| m05_joint_grw_smile_supremacy_w070 vs baseline | exclusive | baseline | 309 | 43.37 | 3.34 | 2.46 | 1.617 |

## Persisted portfolios

| model | model_run_id | portfolio_run_id |
|---|---|---|
| m05_joint_grw_supremacy_w040 | 0ee58d18-b7e9-4168-8d78-93887b1a8c26 | 1e6b80b0-0b76-4f09-ac15-95a30959446c |
| m05_joint_grw_smile_supremacy_w020 | fcd5e974-9a46-4a10-9828-6b987a5484d6 | b8ea28ef-3373-49d8-baa9-16d2c596e475 |
| m05_joint_grw_smile_supremacy_w040 | 30620d3e-e4bd-4c05-b1a1-85cefa36b728 | 38d3038e-2b34-435a-a5b3-7d098c1f8031 |
| m05_joint_grw_smile_supremacy_w070 | 32d588f1-d666-4112-a7e1-5c9545fbbe3d | 5cff0ccc-0df1-4f10-8cf9-f952123adba3 |
