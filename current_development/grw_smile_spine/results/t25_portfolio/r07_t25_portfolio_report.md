# r07 raw vs calibrated portfolio at T−25 — Task 016

Generated 2026-09-17 22:15 at `c992ada` on mcmc-beast. Contract `MatchDay.option_b_system()` for every row. Calibrator `scot_lower_t25_inv` (inv_w0.25_s0.35). Every smile container is staked through the anti-diagonal reweighted grid (T011), so these rows are NOT comparable with Task 015's r07. Bankroll figures are comparable within an environment only.

| environment | n_walk_forward | n_quoted | n_buildable | n_dropped |
|---|---:|---:|---:|---:|
| close | 710 | 635 | 632 | 3 |
| t25 | 710 | 611 | 611 | 0 |

## Gates

| gate | model | return_pct | reference_pct | delta_pp | n_bets | reference_bets |
|---|---|---:|---:|---:|---:|---:|
| T1 close/raw vs r06 | m05_joint_grw_baseline | 385.78 | 385.78 | 0.00e+00 | 1247 | 1247 |
| T1 close/raw vs r06 | m05_joint_grw_supremacy_w040 | 404.57 | 404.57 | 0.00e+00 | 1244 | 1244 |
| T1 close/raw vs r06 | m05_joint_grw_smile_supremacy_w020 | 606.46 | 606.46 | 0.00e+00 | 1253 | 1253 |
| T1 close/raw vs r06 | m05_joint_grw_smile_supremacy_w040 | 553.49 | 553.49 | 0.00e+00 | 1250 | 1250 |
| T1 close/raw vs r06 | m05_joint_grw_smile_spine_w020 | 485.04 | 485.04 | 0.00e+00 | 1225 | 1225 |
| T1 close/raw vs r06 | m05_joint_grw_smile_spine_w040 | 469.35 | 469.35 | 0.00e+00 | 1232 | 1232 |
| T2 t25/raw vs Task 014 | m05_joint_grw_baseline | 531.78 | 531.78 | 3.41e-05 | 1124 | 1124 |
| T2 t25/t25_inv vs Task 014 | m05_joint_grw_baseline | 245.85 | 245.85 | -1.96e-08 | 969 | 969 |

T2 comparable: true (T−25 panel 611, Task 014 611).

T3 — reported price (`max_abs_vs_smile`) and stake side (`stake_max_abs_gap`):

| environment | variant | model | n_totals_bets | max_abs_vs_smile | min_abs_vs_grid | stake_max_abs_gap | stake_per_strike_gap |
|---|---|---|---:|---:|---:|---:|---|
| close | raw | m05_joint_grw_smile_supremacy_w020 | 200 | 1.9e-15 | 0.00006 | 3.22e-15 | 1.1e-16/1.2e-15/2.2e-15/3.1e-15/3.2e-15 |
| close | raw | m05_joint_grw_smile_supremacy_w040 | 199 | 1.8e-15 | 0.00013 | 3.55e-15 | 1.2e-16/9.4e-16/1.8e-15/2.9e-15/3.6e-15 |
| close | raw | m05_joint_grw_smile_spine_w020 | 239 | 1.6e-15 | 0.00000 | 3.44e-15 | 1.1e-16/1.4e-15/1.8e-15/3.4e-15/3.3e-15 |
| close | raw | m05_joint_grw_smile_spine_w040 | 237 | 1.6e-15 | 0.00000 | 3.00e-15 | 9.7e-17/1.4e-15/1.8e-15/3.0e-15/3.0e-15 |
| t25 | raw | m05_joint_grw_smile_supremacy_w020 | 145 | 1.8e-15 | 0.00008 | 3.22e-15 | 1.1e-16/1.2e-15/2.2e-15/3.1e-15/3.2e-15 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w020 | 113 | 1.8e-15 | 0.00013 | 3.44e-15 | 1.4e-16/1.3e-15/1.9e-15/3.4e-15/3.1e-15 |
| t25 | raw | m05_joint_grw_smile_supremacy_w040 | 150 | 1.6e-15 | 0.00013 | 3.55e-15 | 1.2e-16/9.4e-16/1.8e-15/2.9e-15/3.6e-15 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w040 | 117 | 1.6e-15 | 0.00007 | 3.44e-15 | 1.4e-16/1.4e-15/2.4e-15/3.1e-15/3.4e-15 |
| t25 | raw | m05_joint_grw_smile_spine_w020 | 207 | 1.7e-15 | 0.00000 | 3.44e-15 | 1.1e-16/1.4e-15/1.8e-15/3.4e-15/3.3e-15 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_spine_w020 | 174 | 1.7e-15 | 0.00000 | 3.44e-15 | 1.1e-16/1.4e-15/2.0e-15/2.9e-15/3.4e-15 |
| t25 | raw | m05_joint_grw_smile_spine_w040 | 210 | 1.7e-15 | 0.00000 | 3.00e-15 | 9.7e-17/1.4e-15/1.8e-15/3.0e-15/3.0e-15 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_spine_w040 | 169 | 1.4e-15 | 0.00000 | 3.11e-15 | 1.4e-16/1.3e-15/2.0e-15/2.6e-15/3.1e-15 |

## Headline

| environment | calibration | variant | route | model | n_bets | n_bets_1x2 | n_bets_totals | total_return_pct | roi_pct | sharpe_ann | max_drawdown_pct | win_rate_pct | growth_lo | growth_hi | mean_edge_pp |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | raw | raw | native | m05_joint_grw_baseline | 1247 | 986 | 261 | 385.78 | 11.68 | 1.453 | -42.67 | 35.20 | -0.0004 | +0.0323 | 4.13 |
| close | raw | raw | smile | m05_joint_grw_smile_spine_w020 | 1225 | 986 | 239 | 485.04 | 14.30 | 1.427 | -41.99 | 33.39 | -0.0010 | +0.0365 | 3.93 |
| close | raw | raw | smile | m05_joint_grw_smile_spine_w040 | 1232 | 995 | 237 | 469.35 | 14.54 | 1.462 | -41.63 | 33.28 | -0.0002 | +0.0355 | 3.65 |
| close | raw | raw | smile | m05_joint_grw_smile_supremacy_w020 | 1253 | 1053 | 200 | 606.46 | 15.69 | 1.611 | -40.66 | 32.08 | +0.0014 | +0.0376 | 3.75 |
| close | raw | raw | smile | m05_joint_grw_smile_supremacy_w040 | 1250 | 1051 | 199 | 553.49 | 15.72 | 1.640 | -39.08 | 32.56 | +0.0021 | +0.0360 | 3.50 |
| close | raw | raw | native | m05_joint_grw_supremacy_w040 | 1244 | 983 | 261 | 404.57 | 12.75 | 1.551 | -42.64 | 36.50 | +0.0007 | +0.0321 | 3.76 |
| t25 | raw | raw | native | m05_joint_grw_baseline | 1124 | 896 | 228 | 531.78 | 14.15 | 1.658 | -41.86 | 35.41 | +0.0023 | +0.0354 | 4.15 |
| t25 | raw | raw | smile | m05_joint_grw_smile_spine_w020 | 1115 | 908 | 207 | 507.47 | 16.21 | 1.308 | -42.91 | 32.91 | -0.0024 | +0.0400 | 3.97 |
| t25 | raw | raw | smile | m05_joint_grw_smile_spine_w040 | 1104 | 894 | 210 | 457.70 | 16.28 | 1.283 | -43.82 | 32.88 | -0.0024 | +0.0387 | 3.76 |
| t25 | raw | raw | smile | m05_joint_grw_smile_supremacy_w020 | 1119 | 974 | 145 | 450.05 | 15.44 | 1.271 | -42.03 | 31.10 | -0.0029 | +0.0379 | 3.85 |
| t25 | raw | raw | smile | m05_joint_grw_smile_supremacy_w040 | 1112 | 962 | 150 | 381.23 | 15.14 | 1.212 | -42.15 | 31.03 | -0.0031 | +0.0358 | 3.61 |
| t25 | raw | raw | native | m05_joint_grw_supremacy_w040 | 1089 | 864 | 225 | 605.63 | 16.17 | 1.711 | -38.27 | 36.82 | +0.0031 | +0.0372 | 3.94 |
| t25 | t25_inv | t25_inv | native | m05_joint_grw_baseline | 969 | 776 | 193 | 245.85 | 17.39 | 1.976 | -21.99 | 36.33 | +0.0031 | +0.0220 | 2.36 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_spine_w020 | 980 | 787 | 193 | 232.10 | 20.45 | 1.768 | -18.64 | 34.39 | +0.0025 | +0.0228 | 2.09 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_spine_w020 | 975 | 801 | 174 | 187.84 | 20.13 | 1.618 | -22.24 | 32.41 | +0.0013 | +0.0209 | 1.93 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_spine_w040 | 966 | 774 | 192 | 220.99 | 21.04 | 1.818 | -16.83 | 35.40 | +0.0025 | +0.0218 | 1.97 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_spine_w040 | 966 | 797 | 169 | 171.12 | 20.55 | 1.633 | -19.94 | 32.51 | +0.0011 | +0.0199 | 1.80 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_supremacy_w020 | 955 | 790 | 165 | 242.41 | 21.78 | 1.818 | -18.35 | 34.35 | +0.0024 | +0.0229 | 2.03 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_supremacy_w020 | 953 | 840 | 113 | 170.84 | 18.67 | 1.500 | -20.33 | 30.54 | +0.0004 | +0.0204 | 2.02 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_supremacy_w040 | 947 | 782 | 165 | 223.66 | 22.10 | 1.825 | -16.67 | 35.27 | +0.0024 | +0.0220 | 1.90 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_supremacy_w040 | 946 | 829 | 117 | 155.78 | 19.01 | 1.501 | -17.11 | 31.18 | +0.0006 | +0.0193 | 1.86 |
| t25 | t25_inv | t25_inv | native | m05_joint_grw_supremacy_w040 | 963 | 772 | 191 | 219.53 | 17.98 | 1.856 | -19.64 | 37.80 | +0.0026 | +0.0214 | 2.17 |

## What the calibrator did

| environment | variant | model | n_shifted | w_median | w_p10 | w_p90 | var_retention_median | market_share_median |
|---|---|---|---:|---:|---:|---:|---:|---:|
| t25 | t25_inv | m05_joint_grw_baseline | 580 | 0.284 | 0.251 | 0.454 | 0.081 | 0.716 |
| t25 | t25_inv | m05_joint_grw_supremacy_w040 | 580 | 0.276 | 0.251 | 0.431 | 0.076 | 0.724 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w020 | 580 | 0.281 | 0.251 | 0.428 | 0.079 | 0.719 |
| t25 | t25_inv_grid | m05_joint_grw_smile_supremacy_w020 | 580 | 0.281 | 0.251 | 0.428 | 0.079 | 0.719 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w040 | 580 | 0.277 | 0.251 | 0.400 | 0.077 | 0.723 |
| t25 | t25_inv_grid | m05_joint_grw_smile_supremacy_w040 | 580 | 0.277 | 0.251 | 0.400 | 0.077 | 0.723 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_spine_w020 | 580 | 0.281 | 0.251 | 0.429 | 0.079 | 0.719 |
| t25 | t25_inv_grid | m05_joint_grw_smile_spine_w020 | 580 | 0.281 | 0.251 | 0.429 | 0.079 | 0.719 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_spine_w040 | 580 | 0.277 | 0.251 | 0.402 | 0.077 | 0.723 |
| t25 | t25_inv_grid | m05_joint_grw_smile_spine_w040 | 580 | 0.277 | 0.251 | 0.402 | 0.077 | 0.723 |

## Paired contrasts (slate-level log-growth bootstrap, B = 10000)

Δ is a − b per slate; a slate one arm did not stake counts 0 for it. `p_better` is the share of resamples with Δ > 0.

| question | environment | pair | return_a_pct | return_b_pct | roi_a_pct | roi_b_pct | delta_roi_pp | n_slates | delta_log_growth_per_slate | lo | hi | p_better |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 raw lead at T−25 | t25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | 507.47 | 531.78 | 16.21 | 14.15 | +2.06 | 99 | -0.00040 | -0.01160 | +0.01121 | 0.477 |
| Q2 beyond L2, smile kept | t25 | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | 187.84 | 245.85 | 20.13 | 17.39 | +2.74 | 99 | -0.00185 | -0.00684 | +0.00344 | 0.237 |
| Q3 beyond L2, smile dropped | t25 | m05_joint_grw_smile_spine_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | 232.10 | 245.85 | 20.45 | 17.39 | +3.06 | 99 | -0.00041 | -0.00531 | +0.00477 | 0.436 |
| Q4 pillar instead of L2 | t25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[t25_inv] | 507.47 | 245.85 | 16.21 | 17.39 | -1.18 | 99 | +0.00569 | -0.00832 | +0.01963 | 0.794 |
| H4 spine − five-strike, raw | t25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_smile_supremacy_w020[raw] | 507.47 | 450.05 | 16.21 | 15.44 | +0.77 | 99 | +0.00100 | -0.00334 | +0.00513 | 0.680 |
| H4 spine − five-strike, pooltot | t25 | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] | 187.84 | 170.84 | 20.13 | 18.67 | +1.46 | 99 | +0.00061 | -0.00294 | +0.00399 | 0.633 |
| within arm: pooltot − grid | t25 | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_spine_w020[t25_inv_grid] | 187.84 | 232.10 | 20.13 | 20.45 | -0.32 | 99 | -0.00144 | -0.00352 | +0.00062 | 0.084 |
| close reference (r06) | close | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | 485.04 | 385.78 | 14.30 | 11.68 | +2.63 | 100 | +0.00186 | -0.00905 | +0.01335 | 0.629 |
| Q1 raw lead at T−25 | t25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | 457.70 | 531.78 | 16.28 | 14.15 | +2.12 | 99 | -0.00126 | -0.01273 | +0.01060 | 0.417 |
| Q2 beyond L2, smile kept | t25 | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | 171.12 | 245.85 | 20.55 | 17.39 | +3.17 | 99 | -0.00246 | -0.00760 | +0.00295 | 0.179 |
| Q3 beyond L2, smile dropped | t25 | m05_joint_grw_smile_spine_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | 220.99 | 245.85 | 21.04 | 17.39 | +3.65 | 99 | -0.00075 | -0.00579 | +0.00451 | 0.386 |
| Q4 pillar instead of L2 | t25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[t25_inv] | 457.70 | 245.85 | 16.28 | 17.39 | -1.11 | 99 | +0.00483 | -0.00891 | +0.01836 | 0.760 |
| H4 spine − five-strike, raw | t25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_smile_supremacy_w040[raw] | 457.70 | 381.23 | 16.28 | 15.14 | +1.13 | 99 | +0.00149 | -0.00284 | +0.00553 | 0.752 |
| H4 spine − five-strike, pooltot | t25 | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] | 171.12 | 155.78 | 20.55 | 19.01 | +1.55 | 99 | +0.00059 | -0.00276 | +0.00380 | 0.637 |
| within arm: pooltot − grid | t25 | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_spine_w040[t25_inv_grid] | 171.12 | 220.99 | 20.55 | 21.04 | -0.49 | 99 | -0.00171 | -0.00373 | +0.00030 | 0.048 |
| close reference (r06) | close | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | 469.35 | 385.78 | 14.54 | 11.68 | +2.87 | 100 | +0.00159 | -0.00948 | +0.01323 | 0.611 |
| Q1 raw lead at T−25 | t25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | 605.63 | 531.78 | 16.17 | 14.15 | +2.02 | 99 | +0.00112 | -0.00720 | +0.00941 | 0.600 |
| Q2 beyond L2 | t25 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | 219.53 | 245.85 | 17.98 | 17.39 | +0.59 | 99 | -0.00080 | -0.00463 | +0.00308 | 0.346 |

## Market breakdown

| environment | variant | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| close | raw | 1X2 | m05_joint_grw_baseline | 986 | 29.82 | 12.55 | 75.92 | 3.93 |
| close | raw | 1X2 | m05_joint_grw_smile_spine_w020 | 986 | 29.61 | 16.11 | 79.68 | 3.98 |
| close | raw | 1X2 | m05_joint_grw_smile_spine_w040 | 995 | 29.45 | 16.61 | 78.03 | 3.62 |
| close | raw | 1X2 | m05_joint_grw_smile_supremacy_w020 | 1053 | 28.49 | 16.20 | 88.62 | 3.90 |
| close | raw | 1X2 | m05_joint_grw_smile_supremacy_w040 | 1051 | 29.12 | 16.16 | 87.57 | 3.60 |
| close | raw | 1X2 | m05_joint_grw_supremacy_w040 | 983 | 31.33 | 14.62 | 71.82 | 3.38 |
| close | raw | totals | m05_joint_grw_baseline | 261 | 55.56 | 8.91 | 24.08 | 4.86 |
| close | raw | totals | m05_joint_grw_smile_spine_w020 | 239 | 48.95 | 7.20 | 20.32 | 3.72 |
| close | raw | totals | m05_joint_grw_smile_spine_w040 | 237 | 49.37 | 7.19 | 21.97 | 3.78 |
| close | raw | totals | m05_joint_grw_smile_supremacy_w020 | 200 | 51.00 | 11.71 | 11.38 | 3.00 |
| close | raw | totals | m05_joint_grw_smile_supremacy_w040 | 199 | 50.75 | 12.67 | 12.43 | 2.98 |
| close | raw | totals | m05_joint_grw_supremacy_w040 | 261 | 55.94 | 7.99 | 28.18 | 5.18 |
| t25 | raw | 1X2 | m05_joint_grw_baseline | 896 | 28.91 | 12.82 | 74.07 | 3.92 |
| t25 | raw | 1X2 | m05_joint_grw_smile_spine_w020 | 908 | 29.30 | 16.93 | 81.03 | 4.07 |
| t25 | raw | 1X2 | m05_joint_grw_smile_spine_w040 | 894 | 28.97 | 17.26 | 79.59 | 3.80 |
| t25 | raw | 1X2 | m05_joint_grw_smile_supremacy_w020 | 974 | 27.62 | 15.37 | 90.07 | 3.95 |
| t25 | raw | 1X2 | m05_joint_grw_smile_supremacy_w040 | 962 | 27.13 | 15.29 | 89.11 | 3.69 |
| t25 | raw | 1X2 | m05_joint_grw_supremacy_w040 | 864 | 30.09 | 14.96 | 70.97 | 3.60 |
| t25 | t25_inv | 1X2 | m05_joint_grw_baseline | 776 | 30.03 | 15.68 | 70.05 | 2.19 |
| t25 | t25_inv | 1X2 | m05_joint_grw_supremacy_w040 | 772 | 31.74 | 16.60 | 69.14 | 1.97 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_spine_w020 | 787 | 29.35 | 21.54 | 79.97 | 2.13 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_spine_w040 | 774 | 30.36 | 22.25 | 79.32 | 2.00 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_supremacy_w020 | 790 | 29.75 | 22.22 | 84.36 | 2.13 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_supremacy_w040 | 782 | 30.69 | 22.78 | 83.81 | 1.98 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_spine_w020 | 801 | 29.09 | 21.88 | 75.03 | 1.93 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_spine_w040 | 797 | 29.11 | 22.64 | 74.29 | 1.79 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_supremacy_w020 | 840 | 27.14 | 18.25 | 87.51 | 2.10 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_supremacy_w040 | 829 | 27.50 | 18.99 | 86.63 | 1.94 |
| t25 | raw | totals | m05_joint_grw_baseline | 228 | 60.96 | 17.97 | 25.93 | 5.06 |
| t25 | raw | totals | m05_joint_grw_smile_spine_w020 | 207 | 48.79 | 13.14 | 18.97 | 3.56 |
| t25 | raw | totals | m05_joint_grw_smile_spine_w040 | 210 | 49.52 | 12.45 | 20.41 | 3.57 |
| t25 | raw | totals | m05_joint_grw_smile_supremacy_w020 | 145 | 54.48 | 16.15 | 9.93 | 3.22 |
| t25 | raw | totals | m05_joint_grw_smile_supremacy_w040 | 150 | 56.00 | 13.93 | 10.89 | 3.08 |
| t25 | raw | totals | m05_joint_grw_supremacy_w040 | 225 | 62.67 | 19.14 | 29.03 | 5.27 |
| t25 | t25_inv | totals | m05_joint_grw_baseline | 193 | 61.66 | 21.39 | 29.95 | 3.07 |
| t25 | t25_inv | totals | m05_joint_grw_supremacy_w040 | 191 | 62.30 | 21.07 | 30.86 | 2.98 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_spine_w020 | 193 | 54.92 | 16.07 | 20.03 | 1.91 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_spine_w040 | 192 | 55.73 | 16.41 | 20.68 | 1.82 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_supremacy_w020 | 165 | 56.36 | 19.39 | 15.64 | 1.58 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_supremacy_w040 | 165 | 56.97 | 18.58 | 16.19 | 1.53 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_spine_w020 | 174 | 47.70 | 14.88 | 24.97 | 1.91 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_spine_w040 | 169 | 48.52 | 14.53 | 25.71 | 1.85 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_supremacy_w020 | 113 | 55.75 | 21.63 | 12.49 | 1.44 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_supremacy_w040 | 117 | 57.26 | 19.14 | 13.37 | 1.36 |

## By selection family

| environment | variant | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| close | raw | 1X2_away | m05_joint_grw_baseline | 403 | 26.55 | 7.77 | 33.30 | 5.50 |
| close | raw | 1X2_away | m05_joint_grw_smile_spine_w020 | 361 | 25.21 | 19.44 | 30.90 | 5.55 |
| close | raw | 1X2_away | m05_joint_grw_smile_spine_w040 | 366 | 25.14 | 20.88 | 30.51 | 5.07 |
| close | raw | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 333 | 25.83 | 23.05 | 30.33 | 5.28 |
| close | raw | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 326 | 26.07 | 23.57 | 30.16 | 4.94 |
| close | raw | 1X2_away | m05_joint_grw_supremacy_w040 | 394 | 26.65 | 16.64 | 27.42 | 4.20 |
| close | raw | 1X2_draw | m05_joint_grw_baseline | 273 | 24.18 | 3.38 | 9.48 | 0.51 |
| close | raw | 1X2_draw | m05_joint_grw_smile_spine_w020 | 334 | 26.35 | 6.72 | 13.30 | 0.99 |
| close | raw | 1X2_draw | m05_joint_grw_smile_spine_w040 | 327 | 25.69 | 7.00 | 12.68 | 0.93 |
| close | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 473 | 25.79 | 6.81 | 24.57 | 2.07 |
| close | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 470 | 26.38 | 5.78 | 24.45 | 2.01 |
| close | raw | 1X2_draw | m05_joint_grw_supremacy_w040 | 250 | 28.00 | 11.06 | 8.64 | 0.70 |
| close | raw | 1X2_home | m05_joint_grw_baseline | 310 | 39.03 | 19.99 | 33.14 | 4.90 |
| close | raw | 1X2_home | m05_joint_grw_smile_spine_w020 | 291 | 38.83 | 16.74 | 35.48 | 5.46 |
| close | raw | 1X2_home | m05_joint_grw_smile_spine_w040 | 302 | 38.74 | 16.37 | 34.84 | 4.77 |
| close | raw | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 247 | 37.25 | 16.89 | 33.71 | 5.55 |
| close | raw | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 255 | 38.04 | 17.07 | 32.96 | 4.79 |
| close | raw | 1X2_home | m05_joint_grw_supremacy_w040 | 339 | 39.23 | 13.93 | 35.76 | 4.40 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_baseline | 64 | 75.00 | 1.25 | 4.67 | 3.47 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w020 | 2 | 100.00 | 41.53 | 0.13 | 1.56 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 2 | 100.00 | 41.28 | 0.10 | 1.01 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 42 | 69.05 | -6.90 | 2.35 | 0.71 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 42 | 69.05 | -5.97 | 2.51 | 0.40 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 58 | 75.86 | -0.36 | 4.90 | 3.68 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_baseline | 197 | 49.24 | 10.75 | 19.41 | 5.32 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w020 | 237 | 48.52 | 6.98 | 20.20 | 3.74 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 235 | 48.94 | 7.03 | 21.86 | 3.80 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 158 | 46.20 | 16.54 | 9.04 | 3.61 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 157 | 45.86 | 17.39 | 9.92 | 3.67 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 203 | 50.25 | 9.74 | 23.28 | 5.61 |
| t25 | raw | 1X2_away | m05_joint_grw_baseline | 363 | 26.45 | 8.44 | 32.34 | 5.62 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_spine_w020 | 326 | 26.07 | 19.99 | 30.89 | 5.75 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_spine_w040 | 316 | 25.63 | 20.40 | 30.62 | 5.50 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 294 | 25.17 | 22.02 | 29.97 | 5.65 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 288 | 24.31 | 21.74 | 29.88 | 5.29 |
| t25 | raw | 1X2_away | m05_joint_grw_supremacy_w040 | 337 | 26.71 | 16.92 | 26.45 | 4.53 |
| t25 | t25_inv | 1X2_away | m05_joint_grw_baseline | 358 | 25.42 | 7.49 | 32.01 | 2.70 |
| t25 | t25_inv | 1X2_away | m05_joint_grw_supremacy_w040 | 351 | 26.78 | 13.96 | 27.86 | 2.19 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_spine_w020 | 361 | 26.59 | 19.56 | 34.02 | 2.53 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_spine_w040 | 357 | 27.17 | 19.93 | 34.11 | 2.38 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 365 | 26.85 | 19.41 | 36.44 | 2.56 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 364 | 26.92 | 19.64 | 36.73 | 2.39 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_spine_w020 | 335 | 26.87 | 22.03 | 30.23 | 2.35 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_spine_w040 | 333 | 27.33 | 23.24 | 30.18 | 2.18 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 258 | 23.26 | 26.99 | 25.44 | 2.19 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 256 | 23.83 | 29.29 | 24.95 | 1.96 |
| t25 | raw | 1X2_draw | m05_joint_grw_baseline | 252 | 23.81 | 0.86 | 9.54 | 0.37 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_spine_w020 | 322 | 26.71 | 6.88 | 14.71 | 1.01 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_spine_w040 | 311 | 25.72 | 7.27 | 14.08 | 0.94 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 459 | 25.71 | 2.88 | 26.51 | 1.96 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 448 | 25.67 | 2.51 | 26.50 | 1.94 |
| t25 | raw | 1X2_draw | m05_joint_grw_supremacy_w040 | 231 | 26.84 | 17.47 | 9.06 | 0.67 |
| t25 | t25_inv | 1X2_draw | m05_joint_grw_baseline | 126 | 23.81 | 24.13 | 6.56 | 0.86 |
| t25 | t25_inv | 1X2_draw | m05_joint_grw_supremacy_w040 | 111 | 27.93 | 27.04 | 5.48 | 0.86 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_spine_w020 | 136 | 22.06 | 21.71 | 7.78 | 0.74 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_spine_w040 | 125 | 24.80 | 23.89 | 7.00 | 0.70 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 131 | 22.14 | 24.42 | 7.49 | 0.55 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 121 | 26.45 | 24.68 | 6.63 | 0.50 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_spine_w020 | 199 | 22.11 | 16.36 | 11.60 | 1.02 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_spine_w040 | 195 | 23.08 | 15.63 | 10.92 | 0.95 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 392 | 25.51 | 1.85 | 36.60 | 2.08 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 389 | 26.22 | 0.66 | 37.27 | 2.02 |
| t25 | raw | 1X2_home | m05_joint_grw_baseline | 281 | 36.65 | 20.76 | 32.18 | 4.90 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_spine_w020 | 260 | 36.54 | 18.44 | 35.43 | 5.75 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_spine_w040 | 267 | 36.70 | 18.53 | 34.90 | 5.12 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 221 | 34.84 | 19.29 | 33.59 | 5.82 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 226 | 33.63 | 19.75 | 32.73 | 5.10 |
| t25 | raw | 1X2_home | m05_joint_grw_supremacy_w040 | 296 | 36.49 | 12.86 | 35.46 | 4.82 |
| t25 | t25_inv | 1X2_home | m05_joint_grw_baseline | 292 | 38.36 | 22.24 | 31.48 | 2.13 |
| t25 | t25_inv | 1X2_home | m05_joint_grw_supremacy_w040 | 310 | 38.71 | 17.06 | 35.80 | 2.11 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_spine_w020 | 290 | 36.21 | 23.28 | 38.17 | 2.30 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_spine_w040 | 292 | 36.64 | 24.02 | 38.21 | 2.10 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 294 | 36.73 | 24.35 | 40.43 | 2.29 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 297 | 37.04 | 25.32 | 40.44 | 2.09 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_spine_w020 | 267 | 37.08 | 23.67 | 33.19 | 2.09 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_spine_w040 | 269 | 35.69 | 24.40 | 33.18 | 1.90 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 190 | 35.79 | 33.07 | 25.47 | 1.99 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 184 | 35.33 | 36.45 | 24.41 | 1.71 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 84.48 | 18.60 | 5.09 | 3.86 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w020 | 3 | 33.33 | -10.50 | 0.10 | 1.26 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 3 | 33.33 | -17.09 | 0.08 | 1.12 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 26 | 84.62 | 2.39 | 1.58 | 0.84 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 27 | 85.19 | 7.91 | 1.70 | 0.60 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 53 | 86.79 | 21.59 | 4.92 | 3.73 |
| t25 | t25_inv | O/U 1.5_over_15 | m05_joint_grw_baseline | 41 | 90.24 | 22.80 | 6.24 | 2.21 |
| t25 | t25_inv | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 41 | 90.24 | 24.12 | 5.72 | 1.89 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w020 | 21 | 85.71 | 3.37 | 1.86 | 0.86 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 22 | 86.36 | 5.12 | 2.03 | 0.78 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 31 | 87.10 | 8.20 | 3.07 | 1.22 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 29 | 86.21 | 8.64 | 3.22 | 1.17 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w020 | 1 | 0.00 | -100.00 | 0.04 | 1.04 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_spine_w040 | 1 | 0.00 | -100.00 | 0.02 | 0.77 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 22 | 81.82 | 3.94 | 2.93 | -0.56 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 23 | 78.26 | 3.78 | 3.25 | -0.75 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 52.94 | 17.81 | 20.84 | 5.48 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w020 | 204 | 49.02 | 13.27 | 18.87 | 3.59 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 207 | 49.76 | 12.57 | 20.32 | 3.61 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 119 | 47.90 | 18.75 | 8.35 | 3.74 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 123 | 49.59 | 15.05 | 9.18 | 3.62 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 172 | 55.23 | 18.64 | 24.11 | 5.74 |
| t25 | t25_inv | O/U 2.5_under_25 | m05_joint_grw_baseline | 152 | 53.95 | 21.01 | 23.72 | 3.30 |
| t25 | t25_inv | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 150 | 54.67 | 20.37 | 25.14 | 3.28 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w020 | 172 | 51.16 | 17.37 | 18.17 | 2.04 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 170 | 51.76 | 17.63 | 18.65 | 1.95 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 134 | 49.25 | 22.12 | 12.57 | 1.67 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 136 | 50.74 | 21.05 | 12.98 | 1.60 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w020 | 173 | 47.98 | 15.07 | 24.93 | 1.91 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_spine_w040 | 168 | 48.81 | 14.63 | 25.69 | 1.85 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 91 | 49.45 | 27.05 | 9.56 | 1.93 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 94 | 52.13 | 24.07 | 10.12 | 1.88 |

## Shared vs exclusive bets per contrast

| environment | question | pair | bet_set | owner | n_bets | win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | shared | a | 839 | 30.99 | 18.57 | 4.79 | 1.020 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | shared | b | 839 | 30.99 | 15.04 | 4.74 | 1.076 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 276 | 38.77 | 0.42 | 1.49 | 0.961 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 285 | 48.42 | 10.46 | 2.40 | 2.305 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | a | 747 | 31.73 | 24.12 | 2.30 | 1.035 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | b | 747 | 31.73 | 17.19 | 2.61 | 1.059 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 228 | 34.65 | -3.32 | 0.70 | 1.174 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 222 | 51.80 | 18.24 | 1.51 | 1.826 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | a | 796 | 32.91 | 23.42 | 2.34 | 1.001 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | b | 796 | 32.91 | 17.27 | 2.55 | 1.058 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 184 | 40.76 | -7.42 | 0.99 | 0.978 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 173 | 52.02 | 18.16 | 1.48 | 1.881 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[t25_inv] | shared | a | 734 | 31.47 | 20.56 | 5.27 | 0.991 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[t25_inv] | shared | b | 734 | 31.47 | 18.34 | 2.73 | 1.094 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 381 | 35.70 | -2.84 | 1.48 | 1.063 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 235 | 51.49 | 13.67 | 1.20 | 2.099 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_smile_supremacy_w020[raw] | shared | a | 959 | 30.76 | 16.27 | 4.49 | 1.049 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_smile_supremacy_w020[raw] | shared | b | 959 | 30.76 | 16.18 | 4.43 | 1.000 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_smile_supremacy_w020[raw] | exclusive | a | 156 | 46.15 | 15.19 | 0.83 | 0.768 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_smile_supremacy_w020[raw] | exclusive | b | 160 | 33.12 | -4.66 | 0.42 | 2.717 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] | shared | a | 736 | 29.35 | 21.62 | 2.37 | 1.141 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] | shared | b | 736 | 29.35 | 19.76 | 2.29 | 1.033 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] | exclusive | a | 239 | 41.84 | 9.94 | 0.58 | 1.059 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] | exclusive | b | 217 | 34.56 | 10.16 | 1.08 | 0.654 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_spine_w020[t25_inv_grid] | shared | a | 852 | 32.98 | 21.62 | 2.21 | 0.996 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_spine_w020[t25_inv_grid] | shared | b | 852 | 32.98 | 21.67 | 2.33 | 1.014 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_spine_w020[t25_inv_grid] | exclusive | a | 123 | 28.46 | -7.25 | -0.02 | n/a |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w020[t25_inv_pooltot] − m05_joint_grw_smile_spine_w020[t25_inv_grid] | exclusive | b | 128 | 43.75 | -2.99 | 0.49 | 0.969 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | shared | a | 946 | 31.08 | 14.14 | 4.71 | 1.038 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | shared | b | 946 | 31.08 | 13.08 | 4.69 | 1.091 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 279 | 41.22 | 15.44 | 1.30 | 1.635 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 301 | 48.17 | 5.09 | 2.36 | 1.751 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 824 | 31.07 | 18.98 | 4.52 | 1.032 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 824 | 31.07 | 14.97 | 4.81 | 1.062 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 280 | 38.21 | 0.10 | 1.52 | 1.086 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 300 | 47.33 | 10.81 | 2.34 | 2.442 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | a | 732 | 31.28 | 24.98 | 2.14 | 1.081 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | b | 732 | 31.28 | 16.43 | 2.59 | 1.078 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 234 | 36.32 | -3.69 | 0.72 | 1.008 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 237 | 51.90 | 21.10 | 1.65 | 1.493 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | a | 776 | 33.76 | 24.09 | 2.21 | 0.989 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | b | 776 | 33.76 | 17.62 | 2.57 | 1.027 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 190 | 42.11 | -4.75 | 0.96 | 0.956 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_spine_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 193 | 46.63 | 16.08 | 1.53 | 1.817 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[t25_inv] | shared | a | 727 | 31.77 | 21.13 | 4.93 | 1.004 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[t25_inv] | shared | b | 727 | 31.77 | 18.20 | 2.74 | 1.071 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 377 | 35.01 | -3.56 | 1.50 | 1.104 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 242 | 50.00 | 14.27 | 1.24 | 2.144 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_smile_supremacy_w040[raw] | shared | a | 949 | 30.24 | 16.43 | 4.22 | 1.099 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_smile_supremacy_w040[raw] | shared | b | 949 | 30.24 | 15.56 | 4.13 | 1.041 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_smile_supremacy_w040[raw] | exclusive | a | 155 | 49.03 | 13.55 | 0.93 | 0.749 |
| t25 | H4 spine − five-strike, raw | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_smile_supremacy_w040[raw] | exclusive | b | 163 | 35.58 | 4.93 | 0.54 | 1.796 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] | shared | a | 729 | 30.18 | 22.79 | 2.19 | 1.150 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] | shared | b | 729 | 30.18 | 20.55 | 2.11 | 1.042 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] | exclusive | a | 237 | 39.66 | 5.11 | 0.58 | 0.809 |
| t25 | H4 spine − five-strike, pooltot | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] | exclusive | b | 217 | 34.56 | 7.69 | 1.03 | 0.735 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_spine_w040[t25_inv_grid] | shared | a | 841 | 33.41 | 23.06 | 2.05 | 1.007 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_spine_w040[t25_inv_grid] | shared | b | 841 | 33.41 | 21.58 | 2.18 | 1.013 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_spine_w040[t25_inv_grid] | exclusive | a | 125 | 26.40 | -21.71 | 0.10 | 0.033 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_spine_w040[t25_inv_pooltot] − m05_joint_grw_smile_spine_w040[t25_inv_grid] | exclusive | b | 125 | 48.80 | 11.71 | 0.55 | 0.985 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 936 | 31.30 | 14.49 | 4.38 | 1.052 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 936 | 31.30 | 13.52 | 4.67 | 1.083 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 296 | 39.53 | 14.86 | 1.33 | 1.573 |
| close | close reference (r06) | m05_joint_grw_smile_spine_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 311 | 46.95 | 3.59 | 2.49 | 1.634 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 955 | 36.65 | 17.12 | 4.34 | 1.085 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 955 | 36.65 | 16.20 | 4.77 | 1.048 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 134 | 38.06 | 0.32 | 1.09 | 1.237 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 169 | 28.40 | -19.51 | 0.62 | 1.812 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | shared | a | 859 | 37.60 | 18.97 | 2.34 | 1.047 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | shared | b | 859 | 37.60 | 17.99 | 2.56 | 1.000 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 104 | 39.42 | -5.07 | 0.79 | 0.779 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 110 | 26.36 | 2.87 | 0.81 | 2.155 |
