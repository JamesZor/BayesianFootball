# r07 raw vs calibrated portfolio at T−25 — Task 015

Generated 2026-09-13 22:15. Contract `MatchDay.option_b_system()` for every row. Calibrator `scot_lower_t25_inv` (inv_w0.25_s0.35). Bankroll figures are comparable within an environment only.

| environment | n_walk_forward | n_quoted | n_buildable | n_dropped |
|---|---:|---:|---:|---:|
| close | 710 | 635 | 632 | 3 |
| t25 | 710 | 611 | 611 | 0 |

## Gates

| gate | model | return_pct | reference_pct | delta_pp | n_bets | reference_bets |
|---|---|---:|---:|---:|---:|---:|
| T1 close/raw vs r06 | m05_joint_grw_baseline | 385.78 | 385.78 | 0.00e+00 | 1247 | 1247 |
| T1 close/raw vs r06 | m05_joint_grw_supremacy_w040 | 404.57 | 404.57 | 0.00e+00 | 1244 | 1244 |
| T1 close/raw vs r06 | m05_joint_grw_smile_supremacy_w020 | 588.42 | 588.42 | 0.00e+00 | 1240 | 1240 |
| T1 close/raw vs r06 | m05_joint_grw_smile_supremacy_w040 | 545.01 | 545.01 | 0.00e+00 | 1237 | 1237 |
| T1 close/raw vs r06 | m05_joint_grw_smile_supremacy_w070 | 481.38 | 481.38 | 0.00e+00 | 1231 | 1231 |
| T2 t25/raw vs Task 014 | m05_joint_grw_baseline | 531.78 | 531.78 | 3.41e-05 | 1124 | 1124 |
| T2 t25/t25_inv vs Task 014 | m05_joint_grw_baseline | 245.85 | 245.85 | -1.96e-08 | 969 | 969 |

T2 comparable: true (T−25 panel 611, Task 014 611).

| environment | variant | model | n_totals_bets | max_abs_vs_smile | min_abs_vs_grid |
|---|---|---|---:|---:|---:|
| close | raw | m05_joint_grw_smile_supremacy_w020 | 252 | 1.4e-15 | 0.00006 |
| close | raw | m05_joint_grw_smile_supremacy_w040 | 253 | 1.8e-15 | 0.00013 |
| close | raw | m05_joint_grw_smile_supremacy_w070 | 253 | 1.8e-15 | 0.00002 |
| t25 | raw | m05_joint_grw_smile_supremacy_w020 | 209 | 1.8e-15 | 0.00007 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w020 | 165 | 1.8e-15 | 0.00013 |
| t25 | raw | m05_joint_grw_smile_supremacy_w040 | 204 | 1.8e-15 | 0.00012 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w040 | 165 | 1.6e-15 | 0.00007 |
| t25 | raw | m05_joint_grw_smile_supremacy_w070 | 209 | 1.6e-15 | 0.00002 |
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w070 | 173 | 1.6e-15 | 0.00005 |

## Headline

| environment | calibration | variant | route | model | n_bets | n_bets_1x2 | n_bets_totals | total_return_pct | roi_pct | sharpe_ann | max_drawdown_pct | win_rate_pct | growth_lo | growth_hi | mean_edge_pp |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | raw | raw | native | m05_joint_grw_baseline | 1247 | 986 | 261 | 385.78 | 11.68 | 1.453 | -42.67 | 35.20 | -0.0004 | +0.0323 | 4.13 |
| close | raw | raw | smile | m05_joint_grw_smile_supremacy_w020 | 1240 | 988 | 252 | 588.42 | 15.71 | 1.495 | -44.02 | 33.95 | -0.0003 | +0.0390 | 3.76 |
| close | raw | raw | smile | m05_joint_grw_smile_supremacy_w040 | 1237 | 984 | 253 | 545.01 | 15.82 | 1.516 | -44.03 | 33.95 | +0.0004 | +0.0376 | 3.53 |
| close | raw | raw | smile | m05_joint_grw_smile_supremacy_w070 | 1231 | 978 | 253 | 481.38 | 15.48 | 1.530 | -42.37 | 35.01 | +0.0006 | +0.0354 | 3.31 |
| close | raw | raw | native | m05_joint_grw_supremacy_w040 | 1244 | 983 | 261 | 404.57 | 12.75 | 1.551 | -42.64 | 36.50 | +0.0007 | +0.0321 | 3.76 |
| t25 | raw | raw | native | m05_joint_grw_baseline | 1124 | 896 | 228 | 531.78 | 14.15 | 1.658 | -41.86 | 35.41 | +0.0023 | +0.0354 | 4.15 |
| t25 | raw | raw | smile | m05_joint_grw_smile_supremacy_w020 | 1111 | 902 | 209 | 542.40 | 16.91 | 1.342 | -42.94 | 34.11 | -0.0024 | +0.0409 | 3.81 |
| t25 | raw | raw | smile | m05_joint_grw_smile_supremacy_w040 | 1087 | 883 | 204 | 479.44 | 16.80 | 1.322 | -43.62 | 34.22 | -0.0024 | +0.0385 | 3.63 |
| t25 | raw | raw | smile | m05_joint_grw_smile_supremacy_w070 | 1094 | 885 | 209 | 381.55 | 15.78 | 1.276 | -41.79 | 34.10 | -0.0019 | +0.0341 | 3.38 |
| t25 | raw | raw | native | m05_joint_grw_supremacy_w040 | 1089 | 864 | 225 | 605.63 | 16.17 | 1.711 | -38.27 | 36.82 | +0.0031 | +0.0372 | 3.94 |
| t25 | t25_inv | t25_inv | native | m05_joint_grw_baseline | 969 | 776 | 193 | 245.85 | 17.39 | 1.976 | -21.99 | 36.33 | +0.0031 | +0.0220 | 2.36 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_supremacy_w020 | 955 | 790 | 165 | 242.41 | 21.78 | 1.818 | -18.35 | 34.35 | +0.0024 | +0.0229 | 2.03 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_supremacy_w020 | 955 | 790 | 165 | 242.41 | 21.78 | 1.818 | -18.35 | 34.35 | +0.0024 | +0.0229 | 2.00 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_supremacy_w040 | 947 | 782 | 165 | 223.66 | 22.10 | 1.825 | -16.67 | 35.27 | +0.0024 | +0.0220 | 1.90 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_supremacy_w040 | 947 | 782 | 165 | 223.66 | 22.10 | 1.825 | -16.67 | 35.27 | +0.0024 | +0.0220 | 1.87 |
| t25 | t25_inv | t25_inv_grid | calibrated grid, φ dropped | m05_joint_grw_smile_supremacy_w070 | 940 | 767 | 173 | 178.79 | 20.37 | 1.697 | -14.74 | 36.06 | +0.0018 | +0.0200 | 1.80 |
| t25 | t25_inv | t25_inv_pooltot | smile, calibrated λ_tot × fitted φ | m05_joint_grw_smile_supremacy_w070 | 940 | 767 | 173 | 178.79 | 20.37 | 1.697 | -14.74 | 36.06 | +0.0018 | +0.0200 | 1.76 |
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
| t25 | t25_inv_pooltot | m05_joint_grw_smile_supremacy_w070 | 580 | 0.274 | 0.251 | 0.393 | 0.075 | 0.726 |
| t25 | t25_inv_grid | m05_joint_grw_smile_supremacy_w070 | 580 | 0.274 | 0.251 | 0.393 | 0.075 | 0.726 |

## Paired contrasts (slate-level log-growth bootstrap, B = 10000)

Δ is a − b per slate; a slate one arm did not stake counts 0 for it. `p_better` is the share of resamples with Δ > 0.

| question | environment | pair | return_a_pct | return_b_pct | roi_a_pct | roi_b_pct | delta_roi_pp | n_slates | delta_log_growth_per_slate | lo | hi | p_better |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 raw lead at T−25 | t25 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | 542.40 | 531.78 | 16.91 | 14.15 | +2.75 | 99 | +0.00017 | -0.01124 | +0.01194 | 0.513 |
| Q2 beyond L2, smile kept | t25 | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | 242.41 | 245.85 | 21.78 | 17.39 | +4.39 | 99 | -0.00010 | -0.00519 | +0.00532 | 0.485 |
| Q3 beyond L2, smile dropped | t25 | m05_joint_grw_smile_supremacy_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | 242.41 | 245.85 | 21.78 | 17.39 | +4.39 | 99 | -0.00010 | -0.00519 | +0.00532 | 0.485 |
| Q4 pillar instead of L2 | t25 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[t25_inv] | 542.40 | 245.85 | 16.91 | 17.39 | -0.48 | 99 | +0.00625 | -0.00773 | +0.02013 | 0.814 |
| close reference (r06) | close | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | 588.42 | 385.78 | 15.71 | 11.68 | +4.03 | 100 | +0.00349 | -0.00754 | +0.01520 | 0.722 |
| Q1 raw lead at T−25 | t25 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | 479.44 | 531.78 | 16.80 | 14.15 | +2.65 | 99 | -0.00087 | -0.01207 | +0.01074 | 0.441 |
| Q2 beyond L2, smile kept | t25 | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | 223.66 | 245.85 | 22.10 | 17.39 | +4.71 | 99 | -0.00067 | -0.00589 | +0.00489 | 0.406 |
| Q3 beyond L2, smile dropped | t25 | m05_joint_grw_smile_supremacy_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | 223.66 | 245.85 | 22.10 | 17.39 | +4.71 | 99 | -0.00067 | -0.00589 | +0.00489 | 0.406 |
| Q4 pillar instead of L2 | t25 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[t25_inv] | 479.44 | 245.85 | 16.80 | 17.39 | -0.59 | 99 | +0.00521 | -0.00817 | +0.01857 | 0.783 |
| close reference (r06) | close | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | 545.01 | 385.78 | 15.82 | 11.68 | +4.14 | 100 | +0.00284 | -0.00835 | +0.01443 | 0.687 |
| Q1 raw lead at T−25 | t25 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | 381.55 | 531.78 | 15.78 | 14.15 | +1.63 | 99 | -0.00274 | -0.01392 | +0.00882 | 0.325 |
| Q2 beyond L2, smile kept | t25 | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | 178.79 | 245.85 | 20.37 | 17.39 | +2.98 | 99 | -0.00218 | -0.00780 | +0.00351 | 0.226 |
| Q3 beyond L2, smile dropped | t25 | m05_joint_grw_smile_supremacy_w070[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | 178.79 | 245.85 | 20.37 | 17.39 | +2.98 | 99 | -0.00218 | -0.00780 | +0.00351 | 0.226 |
| Q4 pillar instead of L2 | t25 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[t25_inv] | 381.55 | 245.85 | 15.78 | 17.39 | -1.61 | 99 | +0.00334 | -0.00899 | +0.01564 | 0.705 |
| close reference (r06) | close | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | 481.38 | 385.78 | 15.48 | 11.68 | +3.80 | 100 | +0.00180 | -0.00947 | +0.01339 | 0.622 |
| Q1 raw lead at T−25 | t25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | 605.63 | 531.78 | 16.17 | 14.15 | +2.02 | 99 | +0.00112 | -0.00720 | +0.00941 | 0.600 |
| Q2 beyond L2 | t25 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | 219.53 | 245.85 | 17.98 | 17.39 | +0.59 | 99 | -0.00080 | -0.00463 | +0.00308 | 0.346 |
| close reference (r06) | close | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | 404.57 | 385.78 | 12.75 | 11.68 | +1.08 | 100 | +0.00038 | -0.00843 | +0.00962 | 0.534 |
| within arm: pooltot − grid | t25 | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_grid] | 242.41 | 242.41 | 21.78 | 21.78 | +0.00 | 99 | +0.00000 | +0.00000 | +0.00000 | 0.000 |
| within arm: pooltot − grid | t25 | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_grid] | 223.66 | 223.66 | 22.10 | 22.10 | +0.00 | 99 | +0.00000 | +0.00000 | +0.00000 | 0.000 |
| within arm: pooltot − grid | t25 | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w070[t25_inv_grid] | 178.79 | 178.79 | 20.37 | 20.37 | +0.00 | 99 | +0.00000 | +0.00000 | +0.00000 | 0.000 |

## Market breakdown

| environment | variant | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| close | raw | 1X2 | m05_joint_grw_baseline | 986 | 29.82 | 12.55 | 75.92 | 3.93 |
| close | raw | 1X2 | m05_joint_grw_smile_supremacy_w020 | 988 | 29.45 | 16.99 | 84.79 | 4.03 |
| close | raw | 1X2 | m05_joint_grw_smile_supremacy_w040 | 984 | 29.47 | 17.24 | 83.54 | 3.74 |
| close | raw | 1X2 | m05_joint_grw_smile_supremacy_w070 | 978 | 30.47 | 16.62 | 82.19 | 3.47 |
| close | raw | 1X2 | m05_joint_grw_supremacy_w040 | 983 | 31.33 | 14.62 | 71.82 | 3.38 |
| close | raw | totals | m05_joint_grw_baseline | 261 | 55.56 | 8.91 | 24.08 | 4.86 |
| close | raw | totals | m05_joint_grw_smile_supremacy_w020 | 252 | 51.59 | 8.54 | 15.21 | 2.69 |
| close | raw | totals | m05_joint_grw_smile_supremacy_w040 | 253 | 51.38 | 8.60 | 16.46 | 2.71 |
| close | raw | totals | m05_joint_grw_smile_supremacy_w070 | 253 | 52.57 | 10.23 | 17.81 | 2.68 |
| close | raw | totals | m05_joint_grw_supremacy_w040 | 261 | 55.94 | 7.99 | 28.18 | 5.18 |
| t25 | raw | 1X2 | m05_joint_grw_baseline | 896 | 28.91 | 12.82 | 74.07 | 3.92 |
| t25 | raw | 1X2 | m05_joint_grw_smile_supremacy_w020 | 902 | 28.94 | 16.94 | 85.81 | 4.11 |
| t25 | raw | 1X2 | m05_joint_grw_smile_supremacy_w040 | 883 | 28.88 | 17.01 | 84.86 | 3.88 |
| t25 | raw | 1X2 | m05_joint_grw_smile_supremacy_w070 | 885 | 29.04 | 16.01 | 83.60 | 3.57 |
| t25 | raw | 1X2 | m05_joint_grw_supremacy_w040 | 864 | 30.09 | 14.96 | 70.97 | 3.60 |
| t25 | t25_inv | 1X2 | m05_joint_grw_baseline | 776 | 30.03 | 15.68 | 70.05 | 2.19 |
| t25 | t25_inv | 1X2 | m05_joint_grw_supremacy_w040 | 772 | 31.74 | 16.60 | 69.14 | 1.97 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_supremacy_w020 | 790 | 29.75 | 22.22 | 84.36 | 2.13 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_supremacy_w040 | 782 | 30.69 | 22.78 | 83.81 | 1.98 |
| t25 | t25_inv_grid | 1X2 | m05_joint_grw_smile_supremacy_w070 | 767 | 30.90 | 21.13 | 82.81 | 1.88 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_supremacy_w020 | 790 | 29.75 | 22.22 | 84.36 | 2.13 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_supremacy_w040 | 782 | 30.69 | 22.78 | 83.81 | 1.98 |
| t25 | t25_inv_pooltot | 1X2 | m05_joint_grw_smile_supremacy_w070 | 767 | 30.90 | 21.13 | 82.81 | 1.88 |
| t25 | raw | totals | m05_joint_grw_baseline | 228 | 60.96 | 17.97 | 25.93 | 5.06 |
| t25 | raw | totals | m05_joint_grw_smile_supremacy_w020 | 209 | 56.46 | 16.72 | 14.19 | 2.54 |
| t25 | raw | totals | m05_joint_grw_smile_supremacy_w040 | 204 | 57.35 | 15.63 | 15.14 | 2.55 |
| t25 | raw | totals | m05_joint_grw_smile_supremacy_w070 | 209 | 55.50 | 14.60 | 16.40 | 2.56 |
| t25 | raw | totals | m05_joint_grw_supremacy_w040 | 225 | 62.67 | 19.14 | 29.03 | 5.27 |
| t25 | t25_inv | totals | m05_joint_grw_baseline | 193 | 61.66 | 21.39 | 29.95 | 3.07 |
| t25 | t25_inv | totals | m05_joint_grw_supremacy_w040 | 191 | 62.30 | 21.07 | 30.86 | 2.98 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_supremacy_w020 | 165 | 56.36 | 19.39 | 15.64 | 1.58 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_supremacy_w040 | 165 | 56.97 | 18.58 | 16.19 | 1.53 |
| t25 | t25_inv_grid | totals | m05_joint_grw_smile_supremacy_w070 | 173 | 58.96 | 16.68 | 17.19 | 1.46 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_supremacy_w020 | 165 | 56.36 | 19.39 | 15.64 | 1.36 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_supremacy_w040 | 165 | 56.97 | 18.58 | 16.19 | 1.32 |
| t25 | t25_inv_pooltot | totals | m05_joint_grw_smile_supremacy_w070 | 173 | 58.96 | 16.68 | 17.19 | 1.24 |

## By selection family

| environment | variant | group | model | n_bets | win_rate_pct | roi_pct | stake_share_pct | edge_mean_pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| close | raw | 1X2_away | m05_joint_grw_baseline | 403 | 26.55 | 7.77 | 33.30 | 5.50 |
| close | raw | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 381 | 25.98 | 18.75 | 34.12 | 5.72 |
| close | raw | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 389 | 25.96 | 19.36 | 34.14 | 5.23 |
| close | raw | 1X2_away | m05_joint_grw_smile_supremacy_w070 | 391 | 26.85 | 17.11 | 34.43 | 4.92 |
| close | raw | 1X2_away | m05_joint_grw_supremacy_w040 | 394 | 26.65 | 16.64 | 27.42 | 4.20 |
| close | raw | 1X2_draw | m05_joint_grw_baseline | 273 | 24.18 | 3.38 | 9.48 | 0.51 |
| close | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 302 | 23.51 | 9.57 | 11.81 | 0.26 |
| close | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 284 | 24.30 | 8.38 | 10.98 | 0.22 |
| close | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w070 | 274 | 25.55 | 7.76 | 9.99 | 0.09 |
| close | raw | 1X2_draw | m05_joint_grw_supremacy_w040 | 250 | 28.00 | 11.06 | 8.64 | 0.70 |
| close | raw | 1X2_home | m05_joint_grw_baseline | 310 | 39.03 | 19.99 | 33.14 | 4.90 |
| close | raw | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 305 | 39.67 | 17.70 | 38.87 | 5.65 |
| close | raw | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 311 | 38.59 | 17.89 | 38.43 | 5.09 |
| close | raw | 1X2_home | m05_joint_grw_smile_supremacy_w070 | 313 | 39.30 | 18.51 | 37.77 | 4.63 |
| close | raw | 1X2_home | m05_joint_grw_supremacy_w040 | 339 | 39.23 | 13.93 | 35.76 | 4.40 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_baseline | 64 | 75.00 | 1.25 | 4.67 | 3.47 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 57 | 70.18 | -0.68 | 2.85 | 1.00 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 56 | 69.64 | 0.78 | 2.97 | 0.92 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w070 | 53 | 71.70 | 2.16 | 3.02 | 0.82 |
| close | raw | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 58 | 75.86 | -0.36 | 4.90 | 3.68 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_baseline | 197 | 49.24 | 10.75 | 19.41 | 5.32 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 195 | 46.15 | 10.67 | 12.36 | 3.18 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 197 | 46.19 | 10.32 | 13.49 | 3.22 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w070 | 200 | 47.50 | 11.87 | 14.79 | 3.17 |
| close | raw | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 203 | 50.25 | 9.74 | 23.28 | 5.61 |
| t25 | raw | 1X2_away | m05_joint_grw_baseline | 363 | 26.45 | 8.44 | 32.34 | 5.62 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 346 | 26.01 | 19.07 | 34.32 | 5.84 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 341 | 25.51 | 18.40 | 34.42 | 5.51 |
| t25 | raw | 1X2_away | m05_joint_grw_smile_supremacy_w070 | 349 | 25.50 | 15.33 | 34.67 | 5.04 |
| t25 | raw | 1X2_away | m05_joint_grw_supremacy_w040 | 337 | 26.71 | 16.92 | 26.45 | 4.53 |
| t25 | t25_inv | 1X2_away | m05_joint_grw_baseline | 358 | 25.42 | 7.49 | 32.01 | 2.70 |
| t25 | t25_inv | 1X2_away | m05_joint_grw_supremacy_w040 | 351 | 26.78 | 13.96 | 27.86 | 2.19 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 365 | 26.85 | 19.41 | 36.44 | 2.56 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 364 | 26.92 | 19.64 | 36.73 | 2.39 |
| t25 | t25_inv_grid | 1X2_away | m05_joint_grw_smile_supremacy_w070 | 365 | 27.40 | 18.39 | 36.74 | 2.24 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_supremacy_w020 | 365 | 26.85 | 19.41 | 36.44 | 2.56 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_supremacy_w040 | 364 | 26.92 | 19.64 | 36.73 | 2.39 |
| t25 | t25_inv_pooltot | 1X2_away | m05_joint_grw_smile_supremacy_w070 | 365 | 27.40 | 18.39 | 36.74 | 2.24 |
| t25 | raw | 1X2_draw | m05_joint_grw_baseline | 252 | 23.81 | 0.86 | 9.54 | 0.37 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 278 | 25.18 | 8.18 | 12.12 | 0.29 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 259 | 25.48 | 9.79 | 11.33 | 0.27 |
| t25 | raw | 1X2_draw | m05_joint_grw_smile_supremacy_w070 | 251 | 25.50 | 11.48 | 10.28 | 0.16 |
| t25 | raw | 1X2_draw | m05_joint_grw_supremacy_w040 | 231 | 26.84 | 17.47 | 9.06 | 0.67 |
| t25 | t25_inv | 1X2_draw | m05_joint_grw_baseline | 126 | 23.81 | 24.13 | 6.56 | 0.86 |
| t25 | t25_inv | 1X2_draw | m05_joint_grw_supremacy_w040 | 111 | 27.93 | 27.04 | 5.48 | 0.86 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 131 | 22.14 | 24.42 | 7.49 | 0.55 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 121 | 26.45 | 24.68 | 6.63 | 0.50 |
| t25 | t25_inv_grid | 1X2_draw | m05_joint_grw_smile_supremacy_w070 | 99 | 26.26 | 17.94 | 5.66 | 0.53 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_supremacy_w020 | 131 | 22.14 | 24.42 | 7.49 | 0.55 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_supremacy_w040 | 121 | 26.45 | 24.68 | 6.63 | 0.50 |
| t25 | t25_inv_pooltot | 1X2_draw | m05_joint_grw_smile_supremacy_w070 | 99 | 26.26 | 17.94 | 5.66 | 0.53 |
| t25 | raw | 1X2_home | m05_joint_grw_baseline | 281 | 36.65 | 20.76 | 32.18 | 4.90 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 278 | 36.33 | 17.77 | 39.37 | 5.77 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 283 | 36.04 | 17.88 | 39.11 | 5.23 |
| t25 | raw | 1X2_home | m05_joint_grw_smile_supremacy_w070 | 285 | 36.49 | 17.84 | 38.65 | 4.77 |
| t25 | raw | 1X2_home | m05_joint_grw_supremacy_w040 | 296 | 36.49 | 12.86 | 35.46 | 4.82 |
| t25 | t25_inv | 1X2_home | m05_joint_grw_baseline | 292 | 38.36 | 22.24 | 31.48 | 2.13 |
| t25 | t25_inv | 1X2_home | m05_joint_grw_supremacy_w040 | 310 | 38.71 | 17.06 | 35.80 | 2.11 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 294 | 36.73 | 24.35 | 40.43 | 2.29 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 297 | 37.04 | 25.32 | 40.44 | 2.09 |
| t25 | t25_inv_grid | 1X2_home | m05_joint_grw_smile_supremacy_w070 | 303 | 36.63 | 24.07 | 40.41 | 1.89 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_supremacy_w020 | 294 | 36.73 | 24.35 | 40.43 | 2.29 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_supremacy_w040 | 297 | 37.04 | 25.32 | 40.44 | 2.09 |
| t25 | t25_inv_pooltot | 1X2_home | m05_joint_grw_smile_supremacy_w070 | 303 | 36.63 | 24.07 | 40.41 | 1.89 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_baseline | 58 | 84.48 | 18.60 | 5.09 | 3.86 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 52 | 76.92 | 9.65 | 2.71 | 0.86 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 49 | 79.59 | 9.70 | 2.74 | 0.75 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w070 | 46 | 78.26 | 10.27 | 2.80 | 0.80 |
| t25 | raw | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 53 | 86.79 | 21.59 | 4.92 | 3.73 |
| t25 | t25_inv | O/U 1.5_over_15 | m05_joint_grw_baseline | 41 | 90.24 | 22.80 | 6.24 | 2.21 |
| t25 | t25_inv | O/U 1.5_over_15 | m05_joint_grw_supremacy_w040 | 41 | 90.24 | 24.12 | 5.72 | 1.89 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 31 | 87.10 | 8.20 | 3.07 | 1.22 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 29 | 86.21 | 8.64 | 3.22 | 1.17 |
| t25 | t25_inv_grid | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w070 | 30 | 86.67 | 8.16 | 3.49 | 1.08 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w020 | 31 | 87.10 | 8.20 | 3.07 | -0.05 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w040 | 29 | 86.21 | 8.64 | 3.22 | -0.11 |
| t25 | t25_inv_pooltot | O/U 1.5_over_15 | m05_joint_grw_smile_supremacy_w070 | 30 | 86.67 | 8.16 | 3.49 | -0.18 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_baseline | 170 | 52.94 | 17.81 | 20.84 | 5.48 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 157 | 49.68 | 18.39 | 11.47 | 3.10 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 155 | 50.32 | 16.94 | 12.40 | 3.12 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w070 | 163 | 49.08 | 15.49 | 13.60 | 3.05 |
| t25 | raw | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 172 | 55.23 | 18.64 | 24.11 | 5.74 |
| t25 | t25_inv | O/U 2.5_under_25 | m05_joint_grw_baseline | 152 | 53.95 | 21.01 | 23.72 | 3.30 |
| t25 | t25_inv | O/U 2.5_under_25 | m05_joint_grw_supremacy_w040 | 150 | 54.67 | 20.37 | 25.14 | 3.28 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 134 | 49.25 | 22.12 | 12.57 | 1.67 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 136 | 50.74 | 21.05 | 12.98 | 1.60 |
| t25 | t25_inv_grid | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w070 | 143 | 53.15 | 18.86 | 13.70 | 1.54 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w020 | 134 | 49.25 | 22.12 | 12.57 | 1.69 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w040 | 136 | 50.74 | 21.05 | 12.98 | 1.63 |
| t25 | t25_inv_pooltot | O/U 2.5_under_25 | m05_joint_grw_smile_supremacy_w070 | 143 | 53.15 | 18.86 | 13.70 | 1.54 |

## Shared vs exclusive bets per contrast

| environment | question | pair | bet_set | owner | n_bets | win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | shared | a | 862 | 32.25 | 19.48 | 4.58 | 0.972 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | shared | b | 862 | 32.25 | 15.07 | 4.76 | 1.059 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 249 | 40.56 | -3.74 | 1.16 | 0.793 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 262 | 45.80 | 9.74 | 2.15 | 2.676 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | a | 786 | 33.33 | 24.39 | 2.27 | 0.947 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | b | 786 | 33.33 | 18.97 | 2.53 | 1.062 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 169 | 39.05 | -6.65 | 0.73 | 0.792 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 183 | 49.18 | 8.82 | 1.65 | 1.463 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | a | 786 | 33.33 | 24.39 | 2.30 | 0.977 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | b | 786 | 33.33 | 18.97 | 2.53 | 1.062 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 169 | 39.05 | -6.65 | 0.81 | 1.030 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w020[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 183 | 49.18 | 8.82 | 1.65 | 1.463 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[t25_inv] | shared | a | 764 | 32.98 | 21.26 | 5.06 | 0.925 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[t25_inv] | shared | b | 764 | 32.98 | 19.64 | 2.65 | 1.071 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 347 | 36.60 | -6.31 | 1.07 | 1.060 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 205 | 48.78 | 7.03 | 1.28 | 1.833 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | shared | a | 978 | 32.00 | 15.94 | 4.44 | 1.017 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | shared | b | 978 | 32.00 | 13.11 | 4.56 | 1.091 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 262 | 41.22 | 13.80 | 1.20 | 1.395 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w020[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 269 | 46.84 | 4.44 | 2.54 | 1.495 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 837 | 32.14 | 19.57 | 4.34 | 0.986 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 837 | 32.14 | 14.95 | 4.83 | 1.052 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 250 | 41.20 | -3.22 | 1.25 | 0.944 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 287 | 44.95 | 10.61 | 2.16 | 2.588 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | a | 770 | 33.64 | 25.05 | 2.13 | 0.946 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | b | 770 | 33.64 | 18.93 | 2.53 | 1.016 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 177 | 42.37 | -6.78 | 0.74 | 0.724 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 199 | 46.73 | 9.95 | 1.72 | 1.691 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | a | 770 | 33.64 | 25.05 | 2.15 | 0.976 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | b | 770 | 33.64 | 18.93 | 2.53 | 1.016 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 177 | 42.37 | -6.78 | 0.82 | 0.928 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w040[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 199 | 46.73 | 9.95 | 1.72 | 1.691 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[t25_inv] | shared | a | 754 | 32.76 | 21.65 | 4.73 | 0.959 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[t25_inv] | shared | b | 754 | 32.76 | 19.42 | 2.66 | 1.075 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 333 | 37.54 | -7.54 | 1.15 | 1.025 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 215 | 48.84 | 8.46 | 1.31 | 1.744 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 960 | 31.77 | 16.35 | 4.19 | 1.051 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 960 | 31.77 | 11.65 | 4.63 | 1.072 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 277 | 41.52 | 11.88 | 1.24 | 1.241 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 287 | 46.69 | 11.82 | 2.44 | 1.735 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | shared | a | 831 | 32.01 | 18.88 | 4.01 | 1.001 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | shared | b | 831 | 32.01 | 14.79 | 4.84 | 1.060 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 263 | 40.68 | -3.04 | 1.37 | 1.051 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 293 | 45.05 | 11.40 | 2.20 | 2.388 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | a | 753 | 34.13 | 23.59 | 2.01 | 0.942 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | shared | b | 753 | 34.13 | 16.86 | 2.55 | 0.994 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 187 | 43.85 | -6.79 | 0.77 | 0.639 |
| t25 | Q2 beyond L2, smile kept | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 216 | 43.98 | 19.87 | 1.69 | 1.765 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w070[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | a | 753 | 34.13 | 23.59 | 2.03 | 0.972 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w070[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | shared | b | 753 | 34.13 | 16.86 | 2.55 | 0.994 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w070[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 187 | 43.85 | -6.79 | 0.87 | 0.848 |
| t25 | Q3 beyond L2, smile dropped | m05_joint_grw_smile_supremacy_w070[t25_inv_grid] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 216 | 43.98 | 19.87 | 1.69 | 1.765 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[t25_inv] | shared | a | 747 | 32.53 | 20.69 | 4.38 | 0.983 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[t25_inv] | shared | b | 747 | 32.53 | 18.86 | 2.67 | 1.084 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 347 | 37.46 | -6.30 | 1.23 | 1.089 |
| t25 | Q4 pillar instead of L2 | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 222 | 49.10 | 10.98 | 1.33 | 1.667 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | shared | a | 938 | 32.52 | 16.24 | 3.94 | 1.032 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | shared | b | 938 | 32.52 | 13.55 | 4.67 | 1.059 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 293 | 43.00 | 10.74 | 1.29 | 1.123 |
| close | close reference (r06) | m05_joint_grw_smile_supremacy_w070[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 309 | 43.37 | 3.34 | 2.46 | 1.617 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 955 | 36.65 | 17.12 | 4.34 | 1.085 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 955 | 36.65 | 16.20 | 4.77 | 1.048 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 134 | 38.06 | 0.32 | 1.09 | 1.237 |
| t25 | Q1 raw lead at T−25 | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 169 | 28.40 | -19.51 | 0.62 | 1.812 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | shared | a | 859 | 37.60 | 18.97 | 2.34 | 1.047 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | shared | b | 859 | 37.60 | 17.99 | 2.56 | 1.000 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | exclusive | a | 104 | 39.42 | -5.07 | 0.79 | 0.779 |
| t25 | Q2 beyond L2 | m05_joint_grw_supremacy_w040[t25_inv] − m05_joint_grw_baseline[t25_inv] | exclusive | b | 110 | 26.36 | 2.87 | 0.81 | 2.155 |
| close | close reference (r06) | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | a | 1074 | 35.57 | 13.10 | 4.22 | 1.093 |
| close | close reference (r06) | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | shared | b | 1074 | 35.57 | 13.54 | 4.71 | 1.064 |
| close | close reference (r06) | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | a | 170 | 42.35 | 7.22 | 0.82 | 1.507 |
| close | close reference (r06) | m05_joint_grw_supremacy_w040[raw] − m05_joint_grw_baseline[raw] | exclusive | b | 173 | 32.95 | -25.77 | 0.50 | 1.134 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_grid] | shared | a | 955 | 34.35 | 21.78 | 2.00 | 0.909 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_grid] | shared | b | 955 | 34.35 | 21.78 | 2.03 | 0.955 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_grid] | exclusive | a | 0 | n/a | n/a | n/a | n/a |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w020[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w020[t25_inv_grid] | exclusive | b | 0 | n/a | n/a | n/a | n/a |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_grid] | shared | a | 947 | 35.27 | 22.10 | 1.87 | 0.887 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_grid] | shared | b | 947 | 35.27 | 22.10 | 1.90 | 0.933 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_grid] | exclusive | a | 0 | n/a | n/a | n/a | n/a |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w040[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w040[t25_inv_grid] | exclusive | b | 0 | n/a | n/a | n/a | n/a |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w070[t25_inv_grid] | shared | a | 940 | 36.06 | 20.37 | 1.76 | 0.867 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w070[t25_inv_grid] | shared | b | 940 | 36.06 | 20.37 | 1.80 | 0.917 |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w070[t25_inv_grid] | exclusive | a | 0 | n/a | n/a | n/a | n/a |
| t25 | within arm: pooltot − grid | m05_joint_grw_smile_supremacy_w070[t25_inv_pooltot] − m05_joint_grw_smile_supremacy_w070[t25_inv_grid] | exclusive | b | 0 | n/a | n/a | n/a | n/a |
