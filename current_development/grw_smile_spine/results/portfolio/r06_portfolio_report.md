# r06 closing-line portfolio — Task 016 (1-parameter smile spine)

Generated 2026-09-17 22:11 at `c992ada` on mcmc-beast. Contract: `MatchDay.option_b_system()`. Book: de-vigged Betfair TWA(−20, 0] close. Panel: 710 walk-forward → 635 quoted → 632 buildable under both staking routes. Bootstrap B = 4000.

P1 reproduction — pinned baseline +385.78% / ROI 11.68% / 1247 bets on 632 fixtures; published +385.8% / 11.68% / 1247 on 632: **reproduced**.

`route = reweighted` solves stakes on the anti-diagonal reweighted grid (ticket T011 fixed); `route = grid` is Task 015's path, kept so its rows stay reproducible. Count arms are identical under both and are staked once.

## Headline

| route | model | n_bets | n_bets_1x2 | n_bets_totals | total_return_pct | roi_pct | sharpe_ann | calmar | max_drawdown_pct | win_rate_pct | growth_lo | growth_hi | p_roi_positive | capture_ratio | mean_edge_pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | m05_joint_grw_smile_spine_w020 | 1283 | 1001 | 282 | 558.01 | 14.77 | 1.491 | 12.663 | -44.07 | 34.06 | -0.0002 | +0.0379 | 0.993 | 0.975 | 3.84 |
| grid | m05_joint_grw_smile_spine_w040 | 1269 | 987 | 282 | 538.80 | 15.02 | 1.519 | 12.242 | -44.01 | 34.36 | +0.0003 | +0.0371 | 0.996 | 0.981 | 3.64 |
| grid | m05_joint_grw_smile_supremacy_w020 | 1240 | 988 | 252 | 588.42 | 15.71 | 1.495 | 13.367 | -44.02 | 33.95 | -0.0003 | +0.0390 | 0.996 | 0.981 | 3.76 |
| grid | m05_joint_grw_smile_supremacy_w040 | 1237 | 984 | 253 | 545.01 | 15.82 | 1.516 | 12.377 | -44.03 | 33.95 | +0.0004 | +0.0376 | 0.995 | 1.000 | 3.53 |
| reweighted | m05_joint_grw_baseline | 1247 | 986 | 261 | 385.78 | 11.68 | 1.453 | 9.042 | -42.67 | 35.20 | -0.0004 | +0.0323 | 0.984 | 1.080 | 4.13 |
| reweighted | m05_joint_grw_smile_spine_w020 | 1225 | 986 | 239 | 485.04 | 14.30 | 1.427 | 11.550 | -41.99 | 33.39 | -0.0010 | +0.0365 | 0.991 | 1.006 | 3.93 |
| reweighted | m05_joint_grw_smile_spine_w040 | 1232 | 995 | 237 | 469.35 | 14.54 | 1.462 | 11.274 | -41.63 | 33.28 | -0.0002 | +0.0355 | 0.995 | 1.033 | 3.65 |
| reweighted | m05_joint_grw_smile_supremacy_w020 | 1253 | 1053 | 200 | 606.46 | 15.69 | 1.611 | 14.916 | -40.66 | 32.08 | +0.0014 | +0.0376 | 0.995 | 0.998 | 3.75 |
| reweighted | m05_joint_grw_smile_supremacy_w040 | 1250 | 1051 | 199 | 553.49 | 15.72 | 1.640 | 14.162 | -39.08 | 32.56 | +0.0021 | +0.0360 | 0.995 | 0.983 | 3.50 |
| reweighted | m05_joint_grw_supremacy_w040 | 1244 | 983 | 261 | 404.57 | 12.75 | 1.551 | 9.488 | -42.64 | 36.50 | +0.0007 | +0.0321 | 0.992 | 1.070 | 3.76 |

## Ticket T011 — what correct staking changed

Task 015 r07 measured a smile container and its φ-stripped twin staking identical ledgers. These rows are the same comparison for the fix: the plain-grid route against the reweighted one, same posterior, same panel, same contract.

| model | n_bets_grid | n_bets_reweighted | n_shared | n_only_grid | n_only_reweighted | max_shared_stake_gap | roi_grid_pct | roi_reweighted_pct | delta_roi_pp | return_grid_pct | return_reweighted_pct | delta_return_pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w020 | 1240 | 1253 | 1047 | 193 | 206 | 1.75e-02 | 15.71 | 15.69 | -0.02 | 588.42 | 606.46 | +18.04 |
| m05_joint_grw_smile_supremacy_w040 | 1237 | 1250 | 1035 | 202 | 215 | 1.50e-02 | 15.82 | 15.72 | -0.09 | 545.01 | 553.49 | +8.48 |
| m05_joint_grw_smile_spine_w020 | 1283 | 1225 | 1141 | 142 | 84 | 1.86e-02 | 14.77 | 14.30 | -0.46 | 558.01 | 485.04 | -72.97 |
| m05_joint_grw_smile_spine_w040 | 1269 | 1232 | 1135 | 134 | 97 | 1.87e-02 | 15.02 | 14.54 | -0.48 | 538.80 | 469.35 | -69.45 |

## P2 reported price and P2b stake-side coherence

P2 is the ledger's `p_model`; P2b is the distribution the Kelly solve read. A non-zero P2b gap on the `grid` route is ticket T011 measured, not a failure of this runner.

| model | route | n_totals_bets | max_abs_vs_smile | min_abs_vs_grid |
|---|---|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w020 | reweighted | 200 | 1.9e-15 | 0.00006 |
| m05_joint_grw_smile_supremacy_w020 | grid | 252 | 1.4e-15 | 0.00006 |
| m05_joint_grw_smile_supremacy_w040 | reweighted | 199 | 1.8e-15 | 0.00013 |
| m05_joint_grw_smile_supremacy_w040 | grid | 253 | 1.8e-15 | 0.00013 |
| m05_joint_grw_smile_spine_w020 | reweighted | 239 | 1.6e-15 | 0.00000 |
| m05_joint_grw_smile_spine_w020 | grid | 282 | 1.6e-15 | 0.00000 |
| m05_joint_grw_smile_spine_w040 | reweighted | 237 | 1.6e-15 | 0.00000 |
| m05_joint_grw_smile_spine_w040 | grid | 282 | 1.6e-15 | 0.00000 |

| model | route | n_books | max_abs_gap | worst_fixture | worst_K | per_strike_gap | pass |
|---|---|---:|---:|---:|---:|---|---:|
| m05_joint_grw_smile_supremacy_w020 | reweighted | 632 | 3.22e-15 | 12476643 | 4 | 1.1e-16/1.2e-15/2.2e-15/3.1e-15/3.2e-15 | true |
| m05_joint_grw_smile_supremacy_w020 | grid | 632 | 4.78e-02 | 14035729 | 4 | 4.1e-02/1.9e-02/7.3e-03/2.3e-02/4.8e-02 | false |
| m05_joint_grw_smile_supremacy_w040 | reweighted | 632 | 3.55e-15 | 12476775 | 4 | 1.2e-16/9.4e-16/1.8e-15/2.9e-15/3.6e-15 | true |
| m05_joint_grw_smile_supremacy_w040 | grid | 632 | 4.71e-02 | 14035729 | 4 | 4.1e-02/1.9e-02/7.3e-03/2.2e-02/4.7e-02 | false |
| m05_joint_grw_smile_spine_w020 | reweighted | 632 | 3.44e-15 | 14035697 | 3 | 1.1e-16/1.4e-15/1.8e-15/3.4e-15/3.3e-15 | true |
| m05_joint_grw_smile_spine_w020 | grid | 632 | 7.06e-02 | 14035729 | 4 | 2.5e-02/2.8e-02/2.6e-06/3.9e-02/7.1e-02 | false |
| m05_joint_grw_smile_spine_w040 | reweighted | 632 | 3.00e-15 | 12476611 | 3 | 9.7e-17/1.4e-15/1.8e-15/3.0e-15/3.0e-15 | true |
| m05_joint_grw_smile_spine_w040 | grid | 632 | 7.03e-02 | 14035729 | 4 | 2.5e-02/2.8e-02/2.7e-06/3.9e-02/7.0e-02 | false |

## Market breakdown (both routes)

| route | level | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp | odds_mean | capture_ratio |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| grid | family | 1X2_away | m05_joint_grw_smile_spine_w020 | 379 | 25.59 | 18.22 | 31.84 | 5.64 | 4.52 | 1.076 |
| grid | family | 1X2_away | m05_joint_grw_smile_spine_w040 | 379 | 25.07 | 19.25 | 31.55 | 5.24 | 4.50 | 1.127 |
| grid | family | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 381 | 25.98 | 18.75 | 34.12 | 5.72 | 4.50 | 1.053 |
| grid | family | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 389 | 25.96 | 19.36 | 34.14 | 5.23 | 4.45 | 1.075 |
| reweighted | family | 1X2_away | m05_joint_grw_baseline | 403 | 26.55 | 7.77 | 33.30 | 5.50 | 4.36 | 0.932 |
| reweighted | family | 1X2_away | m05_joint_grw_smile_spine_w020 | 361 | 25.21 | 19.44 | 30.90 | 5.55 | 4.58 | 1.105 |
| reweighted | family | 1X2_away | m05_joint_grw_smile_spine_w040 | 366 | 25.14 | 20.88 | 30.51 | 5.07 | 4.54 | 1.125 |
| reweighted | family | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 333 | 25.83 | 23.05 | 30.33 | 5.28 | 4.70 | 1.076 |
| reweighted | family | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 326 | 26.07 | 23.57 | 30.16 | 4.94 | 4.70 | 1.075 |
| reweighted | family | 1X2_away | m05_joint_grw_supremacy_w040 | 394 | 26.65 | 16.64 | 27.42 | 4.20 | 4.39 | 1.015 |
| grid | family | 1X2_draw | m05_joint_grw_smile_spine_w020 | 320 | 24.38 | 9.24 | 12.04 | 0.51 | 4.17 | 0.982 |
| grid | family | 1X2_draw | m05_joint_grw_smile_spine_w040 | 302 | 25.17 | 10.24 | 11.26 | 0.46 | 4.18 | 0.751 |
| grid | family | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 302 | 23.51 | 9.57 | 11.81 | 0.26 | 4.19 | 0.944 |
| grid | family | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 284 | 24.30 | 8.38 | 10.98 | 0.22 | 4.20 | 0.367 |
| reweighted | family | 1X2_draw | m05_joint_grw_baseline | 273 | 24.18 | 3.38 | 9.48 | 0.51 | 4.21 | 1.317 |
| reweighted | family | 1X2_draw | m05_joint_grw_smile_spine_w020 | 334 | 26.35 | 6.72 | 13.30 | 0.99 | 4.13 | 0.969 |
| reweighted | family | 1X2_draw | m05_joint_grw_smile_spine_w040 | 327 | 25.69 | 7.00 | 12.68 | 0.93 | 4.13 | 0.880 |
| reweighted | family | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 473 | 25.79 | 6.81 | 24.57 | 2.07 | 4.01 | 1.027 |
| reweighted | family | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 470 | 26.38 | 5.78 | 24.45 | 2.01 | 4.01 | 0.954 |
| reweighted | family | 1X2_draw | m05_joint_grw_supremacy_w040 | 250 | 28.00 | 11.06 | 8.64 | 0.70 | 4.22 | 0.705 |
| grid | family | 1X2_home | m05_joint_grw_smile_spine_w020 | 302 | 39.40 | 17.26 | 36.54 | 5.60 | 3.28 | 0.819 |
| grid | family | 1X2_home | m05_joint_grw_smile_spine_w040 | 306 | 39.54 | 17.38 | 35.96 | 5.11 | 3.26 | 0.840 |
| grid | family | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 305 | 39.67 | 17.70 | 38.87 | 5.65 | 3.27 | 0.832 |
| grid | family | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 311 | 38.59 | 17.89 | 38.43 | 5.09 | 3.24 | 0.896 |
| reweighted | family | 1X2_home | m05_joint_grw_baseline | 310 | 39.03 | 19.99 | 33.14 | 4.90 | 3.19 | 0.953 |
| reweighted | family | 1X2_home | m05_joint_grw_smile_spine_w020 | 291 | 38.83 | 16.74 | 35.48 | 5.46 | 3.32 | 0.830 |
| reweighted | family | 1X2_home | m05_joint_grw_smile_spine_w040 | 302 | 38.74 | 16.37 | 34.84 | 4.77 | 3.26 | 0.852 |
| reweighted | family | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 247 | 37.25 | 16.89 | 33.71 | 5.55 | 3.47 | 0.852 |
| reweighted | family | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 255 | 38.04 | 17.07 | 32.96 | 4.79 | 3.41 | 0.857 |
| reweighted | family | 1X2_home | m05_joint_grw_supremacy_w040 | 339 | 39.23 | 13.93 | 35.76 | 4.40 | 3.14 | 0.846 |
| grid | family | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w020 | 36 | 69.44 | -2.14 | 1.63 | -0.79 | 1.38 | n/a |
| grid | family | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 37 | 70.27 | -0.46 | 1.67 | -0.92 | 1.38 | n/a |
| grid | family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 57 | 70.18 | -0.68 | 2.85 | 1.00 | 1.37 | 1.092 |
| grid | family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 56 | 69.64 | 0.78 | 2.97 | 0.92 | 1.36 | 1.085 |
| reweighted | family | O/U 1.5_over_15 | m05_joint_grw_baseline | 64 | 75.00 | 1.25 | 4.67 | 3.47 | 1.32 | 1.279 |
| reweighted | family | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w020 | 2 | 100.00 | 41.53 | 0.13 | 1.56 | 1.43 | n/a |
| reweighted | family | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 2 | 100.00 | 41.28 | 0.10 | 1.01 | 1.43 | n/a |
| reweighted | family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 42 | 69.05 | -6.90 | 2.35 | 0.71 | 1.36 | 1.303 |
| reweighted | family | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 42 | 69.05 | -5.97 | 2.51 | 0.40 | 1.35 | 0.961 |
| reweighted | family | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 58 | 75.86 | -0.36 | 4.90 | 3.68 | 1.31 | 1.299 |
| grid | family | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w020 | 246 | 47.97 | 8.80 | 17.95 | 3.90 | 2.24 | 1.007 |
| grid | family | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 245 | 48.16 | 7.94 | 19.55 | 3.93 | 2.23 | 0.995 |
| grid | family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 195 | 46.15 | 10.67 | 12.36 | 3.18 | 2.27 | 1.135 |
| grid | family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 197 | 46.19 | 10.32 | 13.49 | 3.22 | 2.27 | 1.130 |
| reweighted | family | O/U 2.5_under_25 | m05_joint_grw_baseline | 197 | 49.24 | 10.75 | 19.41 | 5.32 | 2.17 | 1.142 |
| reweighted | family | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w020 | 237 | 48.52 | 6.98 | 20.20 | 3.74 | 2.24 | 1.011 |
| reweighted | family | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 235 | 48.94 | 7.03 | 21.86 | 3.80 | 2.24 | 1.024 |
| reweighted | family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 158 | 46.20 | 16.54 | 9.04 | 3.61 | 2.31 | 1.197 |
| reweighted | family | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 157 | 45.86 | 17.39 | 9.92 | 3.67 | 2.31 | 1.197 |
| reweighted | family | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 203 | 50.25 | 9.74 | 23.28 | 5.61 | 2.17 | 1.092 |
| grid | market | 1X2 | m05_joint_grw_smile_spine_w020 | 1001 | 29.37 | 16.44 | 80.42 | 3.99 | 4.03 | 1.039 |
| grid | market | 1X2 | m05_joint_grw_smile_spine_w040 | 987 | 29.58 | 17.11 | 78.77 | 3.74 | 4.02 | 1.046 |
| grid | market | 1X2 | m05_joint_grw_smile_supremacy_w020 | 988 | 29.45 | 16.99 | 84.79 | 4.03 | 4.03 | 1.052 |
| grid | market | 1X2 | m05_joint_grw_smile_supremacy_w040 | 984 | 29.47 | 17.24 | 83.54 | 3.74 | 4.00 | 1.065 |
| reweighted | market | 1X2 | m05_joint_grw_baseline | 986 | 29.82 | 12.55 | 75.92 | 3.93 | 3.95 | 1.025 |
| reweighted | market | 1X2 | m05_joint_grw_smile_spine_w020 | 986 | 29.61 | 16.11 | 79.68 | 3.98 | 4.05 | 1.019 |
| reweighted | market | 1X2 | m05_joint_grw_smile_spine_w040 | 995 | 29.45 | 16.61 | 78.03 | 3.62 | 4.02 | 1.032 |
| reweighted | market | 1X2 | m05_joint_grw_smile_supremacy_w020 | 1053 | 28.49 | 16.20 | 88.62 | 3.90 | 4.10 | 1.032 |
| reweighted | market | 1X2 | m05_joint_grw_smile_supremacy_w040 | 1051 | 29.12 | 16.16 | 87.57 | 3.60 | 4.08 | 1.006 |
| reweighted | market | 1X2 | m05_joint_grw_supremacy_w040 | 983 | 31.33 | 14.62 | 71.82 | 3.38 | 3.92 | 0.958 |
| grid | market | totals | m05_joint_grw_smile_spine_w020 | 282 | 50.71 | 7.89 | 19.58 | 3.30 | 2.13 | 0.888 |
| grid | market | totals | m05_joint_grw_smile_spine_w040 | 282 | 51.06 | 7.27 | 21.23 | 3.29 | 2.12 | 0.862 |
| grid | market | totals | m05_joint_grw_smile_supremacy_w020 | 252 | 51.59 | 8.54 | 15.21 | 2.69 | 2.07 | 0.985 |
| grid | market | totals | m05_joint_grw_smile_supremacy_w040 | 253 | 51.38 | 8.60 | 16.46 | 2.71 | 2.07 | 0.980 |
| reweighted | market | totals | m05_joint_grw_baseline | 261 | 55.56 | 8.91 | 24.08 | 4.86 | 1.96 | 1.071 |
| reweighted | market | totals | m05_joint_grw_smile_spine_w020 | 239 | 48.95 | 7.20 | 20.32 | 3.72 | 2.23 | 1.001 |
| reweighted | market | totals | m05_joint_grw_smile_spine_w040 | 237 | 49.37 | 7.19 | 21.97 | 3.78 | 2.23 | 1.011 |
| reweighted | market | totals | m05_joint_grw_smile_supremacy_w020 | 200 | 51.00 | 11.71 | 11.38 | 3.00 | 2.11 | 1.035 |
| reweighted | market | totals | m05_joint_grw_smile_supremacy_w040 | 199 | 50.75 | 12.67 | 12.43 | 2.98 | 2.11 | 1.004 |
| reweighted | market | totals | m05_joint_grw_supremacy_w040 | 261 | 55.94 | 7.99 | 28.18 | 5.18 | 1.98 | 1.038 |

## Shared vs exclusive bets (reweighted route)

| pair | bet_set | owner | n_bets | win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | shared | m05_joint_grw_smile_spine_w020 | 946 | 31.08 | 14.14 | 4.71 | 1.038 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | shared | m05_joint_grw_baseline | 946 | 31.08 | 13.08 | 4.69 | 1.091 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | exclusive | m05_joint_grw_smile_spine_w020 | 279 | 41.22 | 15.44 | 1.30 | 1.635 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | exclusive | m05_joint_grw_baseline | 301 | 48.17 | 5.09 | 2.36 | 1.751 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | shared | m05_joint_grw_smile_spine_w040 | 936 | 31.30 | 14.49 | 4.38 | 1.052 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | shared | m05_joint_grw_baseline | 936 | 31.30 | 13.52 | 4.67 | 1.083 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | exclusive | m05_joint_grw_smile_spine_w040 | 296 | 39.53 | 14.86 | 1.33 | 1.573 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | exclusive | m05_joint_grw_baseline | 311 | 46.95 | 3.59 | 2.49 | 1.634 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | shared | m05_joint_grw_smile_spine_w020 | 1072 | 31.72 | 14.27 | 4.40 | 1.066 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | shared | m05_joint_grw_smile_supremacy_w020 | 1072 | 31.72 | 16.83 | 4.26 | 1.015 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | exclusive | m05_joint_grw_smile_spine_w020 | 153 | 45.10 | 15.14 | 0.64 | 1.370 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | exclusive | m05_joint_grw_smile_supremacy_w020 | 181 | 34.25 | -8.47 | 0.74 | 0.890 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | shared | m05_joint_grw_smile_spine_w040 | 1065 | 31.83 | 14.73 | 4.13 | 1.082 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | shared | m05_joint_grw_smile_supremacy_w040 | 1065 | 31.83 | 16.62 | 3.98 | 1.021 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | exclusive | m05_joint_grw_smile_spine_w040 | 167 | 42.51 | 10.80 | 0.59 | 1.567 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | exclusive | m05_joint_grw_smile_supremacy_w040 | 185 | 36.76 | -2.85 | 0.73 | 0.707 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | shared | m05_joint_grw_smile_spine_w040 | 1011 | 32.25 | 14.96 | 4.17 | 1.032 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | shared | m05_joint_grw_supremacy_w040 | 1011 | 32.25 | 13.24 | 4.05 | 1.080 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | exclusive | m05_joint_grw_smile_spine_w040 | 221 | 38.01 | 10.68 | 1.25 | 1.640 |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | exclusive | m05_joint_grw_supremacy_w040 | 233 | 54.94 | 9.85 | 2.50 | 1.668 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | shared | m05_joint_grw_smile_spine_w020 | 1176 | 33.25 | 14.23 | 4.08 | 1.013 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | shared | m05_joint_grw_smile_spine_w040 | 1176 | 33.25 | 14.81 | 3.82 | 1.045 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | exclusive | m05_joint_grw_smile_spine_w020 | 49 | 36.73 | 27.06 | 0.33 | 0.752 |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | exclusive | m05_joint_grw_smile_spine_w040 | 56 | 33.93 | -16.16 | -0.04 | -2.496 |

## Persisted portfolios (this task's runs only)

| model | model_run_id | portfolio_run_id |
|---|---|---|
| m05_joint_grw_smile_spine_w020 | eaf53852-a078-4190-b744-089966a306f6 | 3bd3a461-c505-4ceb-822c-ce1623e0aaf7 |
| m05_joint_grw_smile_spine_w040 | 582035c0-e145-44f7-9f40-89e25388e79a | 785b5d7f-f596-491b-8442-9d71b5a350e6 |
