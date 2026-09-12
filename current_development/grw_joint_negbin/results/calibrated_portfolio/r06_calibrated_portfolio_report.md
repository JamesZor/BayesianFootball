# r06 raw vs calibrated portfolio — Task 014

Generated 2026-09-12 10:49. Contract: `MatchDay.option_b_system()`, identical for every row. No MCMC: all eight posteriors loaded by UUID.

Two market environments, each with its own buildable panel. Bankroll figures are comparable WITHIN an environment and not across them.

| environment | as_of_minutes | book | n_walk_forward | n_quoted | n_buildable | n_dropped |
|---|---:|---|---:|---:|---:|---:|
| close | 0.0000 | de-vigged Betfair TWA(−20, 0] close | 710 | 635 | 632 | 3 |
| t25 | -25.0000 | tradeable T−25 point-in-time order book | 710 | 611 | 611 | 0 |

## Headline — every arm, every variant

| environment | variant | model | likelihood | n_bets | total_return_pct | roi_pct | sharpe_ann | max_drawdown_pct | win_rate_pct | capture_ratio | mean_edge_pp | p_roi_positive |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | close_std | m00_poisson | Poisson | 1292 | 418.92 | 11.62 | 1.388 | -38.82 | 34.91 | 1.029 | 4.36 | 0.981 |
| close | close_std | m00_baseline_grw_negbin | NegBin | 1295 | 402.48 | 11.45 | 1.339 | -38.68 | 34.21 | 1.015 | 4.37 | 0.979 |
| close | close_std | m10_lineup_grw_negbin | NegBin | 1291 | 354.66 | 10.86 | 1.276 | -44.91 | 34.62 | 0.928 | 4.50 | 0.967 |
| close | close_std | m05_poisson | Joint Gamma-Poisson | 1233 | 331.95 | 11.27 | 1.429 | -42.87 | 35.44 | 1.058 | 3.84 | 0.983 |
| close | close_std | m05_wealth_grw_negbin | Joint Gamma-NegBin | 1227 | 331.67 | 11.24 | 1.408 | -41.82 | 34.96 | 1.030 | 3.92 | 0.983 |
| close | close_std | m10_poisson | Poisson | 1287 | 322.68 | 10.44 | 1.232 | -47.63 | 34.81 | 0.958 | 4.48 | 0.964 |
| close | close_std | m12_poisson | Joint Gamma-Poisson | 1243 | 313.93 | 11.05 | 1.314 | -51.38 | 34.51 | 1.039 | 3.98 | 0.977 |
| close | close_std | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1243 | 270.65 | 10.27 | 1.206 | -48.60 | 34.35 | 0.990 | 4.02 | 0.964 |
| close | close_std_t007 | m10_lineup_grw_negbin | NegBin | 1297 | 332.71 | 10.56 | 1.278 | -45.10 | 34.70 | 0.919 | 4.22 | 0.971 |
| close | close_std_t007 | m00_poisson | Poisson | 1290 | 328.23 | 10.55 | 1.276 | -40.31 | 34.73 | 1.037 | 4.10 | 0.978 |
| close | close_std_t007 | m00_baseline_grw_negbin | NegBin | 1293 | 323.53 | 10.47 | 1.243 | -39.20 | 34.34 | 1.005 | 4.13 | 0.975 |
| close | close_std_t007 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 1228 | 307.61 | 10.96 | 1.371 | -42.52 | 35.02 | 1.036 | 3.79 | 0.981 |
| close | close_std_t007 | m05_poisson | Joint Gamma-Poisson | 1234 | 300.96 | 10.88 | 1.377 | -44.45 | 35.33 | 1.073 | 3.70 | 0.982 |
| close | close_std_t007 | m10_poisson | Poisson | 1290 | 291.20 | 9.97 | 1.209 | -48.16 | 34.96 | 0.961 | 4.20 | 0.965 |
| close | close_std_t007 | m12_poisson | Joint Gamma-Poisson | 1246 | 287.51 | 10.66 | 1.293 | -52.50 | 34.43 | 1.054 | 3.79 | 0.972 |
| close | close_std_t007 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1244 | 252.12 | 9.97 | 1.191 | -49.62 | 34.32 | 1.003 | 3.86 | 0.968 |
| close | raw | m00_poisson | Poisson | 1296 | 491.55 | 12.32 | 1.412 | -38.85 | 34.80 | 1.036 | 4.78 | 0.982 |
| close | raw | m00_baseline_grw_negbin | NegBin | 1300 | 461.37 | 12.05 | 1.353 | -38.78 | 34.15 | 1.017 | 4.75 | 0.980 |
| close | raw | m05_poisson | Joint Gamma-Poisson | 1247 | 385.78 | 11.68 | 1.453 | -42.67 | 35.20 | 1.080 | 4.13 | 0.984 |
| close | raw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 1235 | 384.62 | 11.64 | 1.434 | -42.49 | 34.90 | 1.041 | 4.22 | 0.982 |
| close | raw | m10_lineup_grw_negbin | NegBin | 1301 | 370.28 | 11.02 | 1.243 | -45.14 | 34.67 | 0.924 | 4.87 | 0.966 |
| close | raw | m12_poisson | Joint Gamma-Poisson | 1253 | 351.90 | 11.36 | 1.309 | -52.61 | 34.40 | 1.043 | 4.29 | 0.975 |
| close | raw | m10_poisson | Poisson | 1299 | 348.82 | 10.75 | 1.214 | -48.01 | 34.95 | 0.957 | 4.88 | 0.965 |
| close | raw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1244 | 297.52 | 10.50 | 1.196 | -50.23 | 34.16 | 1.007 | 4.37 | 0.965 |
| t25 | raw | m12_poisson | Joint Gamma-Poisson | 1118 | 754.05 | 16.12 | 1.863 | -39.50 | 34.62 | 1.122 | 4.41 | 0.993 |
| t25 | raw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1119 | 684.13 | 15.65 | 1.740 | -39.48 | 34.50 | 1.076 | 4.43 | 0.994 |
| t25 | raw | m05_poisson | Joint Gamma-Poisson | 1124 | 531.78 | 14.15 | 1.658 | -41.86 | 35.41 | 1.110 | 4.15 | 0.989 |
| t25 | raw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 1122 | 514.38 | 14.04 | 1.581 | -41.96 | 35.47 | 1.066 | 4.21 | 0.988 |
| t25 | raw | m10_lineup_grw_negbin | NegBin | 1153 | 426.16 | 12.54 | 1.260 | -46.00 | 34.95 | 0.944 | 4.96 | 0.976 |
| t25 | raw | m00_poisson | Poisson | 1155 | 424.15 | 12.53 | 1.271 | -47.09 | 34.81 | 1.022 | 4.75 | 0.979 |
| t25 | raw | m10_poisson | Poisson | 1143 | 420.16 | 12.36 | 1.291 | -47.40 | 35.43 | 0.947 | 4.96 | 0.977 |
| t25 | raw | m00_baseline_grw_negbin | NegBin | 1149 | 386.54 | 12.18 | 1.188 | -47.00 | 34.90 | 0.981 | 4.82 | 0.973 |
| t25 | t25_inv | m00_poisson | Poisson | 1027 | 359.25 | 17.79 | 1.837 | -27.16 | 36.22 | 1.013 | 2.82 | 0.996 |
| t25 | t25_inv | m00_baseline_grw_negbin | NegBin | 1063 | 342.76 | 16.64 | 1.740 | -28.16 | 35.65 | 0.999 | 2.87 | 0.995 |
| t25 | t25_inv | m12_poisson | Joint Gamma-Poisson | 984 | 302.08 | 18.81 | 2.080 | -24.31 | 36.69 | 1.013 | 2.48 | 0.998 |
| t25 | t25_inv | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1011 | 300.93 | 17.73 | 1.971 | -24.50 | 36.30 | 0.979 | 2.55 | 0.997 |
| t25 | t25_inv | m10_poisson | Poisson | 1046 | 299.79 | 16.41 | 1.714 | -29.93 | 35.76 | 0.949 | 2.82 | 0.990 |
| t25 | t25_inv | m10_lineup_grw_negbin | NegBin | 1071 | 296.71 | 15.62 | 1.640 | -28.87 | 35.57 | 0.928 | 2.88 | 0.987 |
| t25 | t25_inv | m05_wealth_grw_negbin | Joint Gamma-NegBin | 998 | 253.59 | 16.70 | 1.902 | -23.31 | 35.57 | 1.035 | 2.46 | 0.995 |
| t25 | t25_inv | m05_poisson | Joint Gamma-Poisson | 969 | 245.85 | 17.39 | 1.976 | -21.99 | 36.33 | 1.067 | 2.36 | 0.996 |

## What the calibrator did to the posterior

`w_median` is the weight kept on the MODEL's log-rate; `1 - w` is the market's share of the pooled location. `var_retention_median` is the retained posterior log-variance.

| environment | variant | model | law | n_shifted | w_median | w_p10 | w_p90 | var_retention_median | market_share_median |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| close | close_std | m00_baseline_grw_negbin | std_w0.85_s0.15 | 627 | 0.964 | 0.869 | 0.999 | 0.929 | 0.036 |
| close | close_std_t007 | m00_baseline_grw_negbin | std_w0.30_s0.40 | 627 | 0.973 | 0.825 | 0.999 | 0.947 | 0.027 |
| close | close_std | m05_wealth_grw_negbin | std_w0.85_s0.15 | 627 | 0.972 | 0.886 | 0.999 | 0.945 | 0.028 |
| close | close_std_t007 | m05_wealth_grw_negbin | std_w0.30_s0.40 | 627 | 0.980 | 0.873 | 0.999 | 0.961 | 0.020 |
| close | close_std | m10_lineup_grw_negbin | std_w0.85_s0.15 | 627 | 0.963 | 0.868 | 0.999 | 0.927 | 0.037 |
| close | close_std_t007 | m10_lineup_grw_negbin | std_w0.30_s0.40 | 627 | 0.972 | 0.818 | 0.999 | 0.946 | 0.028 |
| close | close_std | m12_joint_hybrid_synergy_negbin | std_w0.85_s0.15 | 627 | 0.970 | 0.884 | 0.999 | 0.940 | 0.030 |
| close | close_std_t007 | m12_joint_hybrid_synergy_negbin | std_w0.30_s0.40 | 627 | 0.978 | 0.868 | 0.999 | 0.957 | 0.022 |
| close | close_std | m00_poisson | std_w0.85_s0.15 | 627 | 0.965 | 0.869 | 0.999 | 0.931 | 0.035 |
| close | close_std_t007 | m00_poisson | std_w0.30_s0.40 | 627 | 0.974 | 0.823 | 0.999 | 0.949 | 0.026 |
| close | close_std | m05_poisson | std_w0.85_s0.15 | 627 | 0.973 | 0.887 | 0.999 | 0.946 | 0.027 |
| close | close_std_t007 | m05_poisson | std_w0.30_s0.40 | 627 | 0.980 | 0.874 | 0.999 | 0.961 | 0.020 |
| close | close_std | m10_poisson | std_w0.85_s0.15 | 627 | 0.962 | 0.867 | 0.999 | 0.926 | 0.038 |
| close | close_std_t007 | m10_poisson | std_w0.30_s0.40 | 627 | 0.972 | 0.814 | 0.999 | 0.945 | 0.028 |
| close | close_std | m12_poisson | std_w0.85_s0.15 | 627 | 0.970 | 0.883 | 0.999 | 0.942 | 0.030 |
| close | close_std_t007 | m12_poisson | std_w0.30_s0.40 | 627 | 0.979 | 0.865 | 0.999 | 0.958 | 0.021 |
| t25 | t25_inv | m00_baseline_grw_negbin | inv_w0.25_s0.35 | 580 | 0.292 | 0.251 | 0.497 | 0.085 | 0.708 |
| t25 | t25_inv | m05_wealth_grw_negbin | inv_w0.25_s0.35 | 580 | 0.283 | 0.251 | 0.451 | 0.080 | 0.717 |
| t25 | t25_inv | m10_lineup_grw_negbin | inv_w0.25_s0.35 | 580 | 0.295 | 0.252 | 0.520 | 0.087 | 0.705 |
| t25 | t25_inv | m12_joint_hybrid_synergy_negbin | inv_w0.25_s0.35 | 580 | 0.282 | 0.251 | 0.457 | 0.080 | 0.718 |
| t25 | t25_inv | m00_poisson | inv_w0.25_s0.35 | 580 | 0.292 | 0.251 | 0.501 | 0.085 | 0.708 |
| t25 | t25_inv | m05_poisson | inv_w0.25_s0.35 | 580 | 0.284 | 0.251 | 0.454 | 0.081 | 0.716 |
| t25 | t25_inv | m10_poisson | inv_w0.25_s0.35 | 580 | 0.295 | 0.252 | 0.515 | 0.087 | 0.705 |
| t25 | t25_inv | m12_poisson | inv_w0.25_s0.35 | 580 | 0.282 | 0.251 | 0.459 | 0.080 | 0.718 |

## Raw vs calibrated, arm by arm

`n_only_a` are bets only the CALIBRATED ledger struck, `n_only_b` only the raw one. On the shared set price and settlement are identical, so `sizing_delta_pnl` is stake size alone.

| environment | variant | model | likelihood | n_shared | n_only_a | n_only_b | overlap_pct | sizing_delta_pnl | shared_roi_a_pct | shared_roi_b_pct |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| close | close_std | m00_baseline_grw_negbin | NegBin | 1284 | 11 | 16 | 97.94 | -0.1708 | 11.45 | 12.09 |
| close | close_std_t007 | m00_baseline_grw_negbin | NegBin | 1282 | 11 | 18 | 97.79 | -0.3839 | 10.45 | 12.10 |
| close | close_std | m05_wealth_grw_negbin | Joint Gamma-NegBin | 1222 | 5 | 13 | 98.55 | -0.1596 | 11.23 | 11.65 |
| close | close_std_t007 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 1222 | 6 | 13 | 98.47 | -0.2276 | 10.96 | 11.67 |
| close | close_std | m10_lineup_grw_negbin | NegBin | 1287 | 4 | 14 | 98.62 | -0.0773 | 10.87 | 11.03 |
| close | close_std_t007 | m10_lineup_grw_negbin | NegBin | 1287 | 10 | 14 | 98.17 | -0.1496 | 10.58 | 11.01 |
| close | close_std | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1231 | 12 | 13 | 98.01 | -0.1196 | 10.25 | 10.51 |
| close | close_std_t007 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1232 | 12 | 12 | 98.09 | -0.1835 | 9.98 | 10.50 |
| close | close_std | m00_poisson | Poisson | 1289 | 3 | 7 | 99.23 | -0.1878 | 11.62 | 12.31 |
| close | close_std_t007 | m00_poisson | Poisson | 1286 | 4 | 10 | 98.92 | -0.4123 | 10.55 | 12.30 |
| close | close_std | m05_poisson | Joint Gamma-Poisson | 1229 | 4 | 18 | 98.24 | -0.1613 | 11.26 | 11.68 |
| close | close_std_t007 | m05_poisson | Joint Gamma-Poisson | 1229 | 5 | 18 | 98.16 | -0.2493 | 10.86 | 11.70 |
| close | close_std | m10_poisson | Poisson | 1283 | 4 | 16 | 98.47 | -0.1072 | 10.43 | 10.73 |
| close | close_std_t007 | m10_poisson | Poisson | 1284 | 6 | 15 | 98.39 | -0.2246 | 9.94 | 10.77 |
| close | close_std | m12_poisson | Joint Gamma-Poisson | 1235 | 8 | 18 | 97.94 | -0.1366 | 11.07 | 11.38 |
| close | close_std_t007 | m12_poisson | Joint Gamma-Poisson | 1237 | 9 | 16 | 98.02 | -0.2300 | 10.70 | 11.43 |
| t25 | t25_inv | m00_baseline_grw_negbin | NegBin | 977 | 86 | 172 | 79.11 | -0.4263 | 16.72 | 12.54 |
| t25 | t25_inv | m05_wealth_grw_negbin | Joint Gamma-NegBin | 914 | 84 | 208 | 75.79 | -0.8153 | 16.82 | 14.51 |
| t25 | t25_inv | m10_lineup_grw_negbin | NegBin | 984 | 87 | 169 | 79.35 | -0.6333 | 15.79 | 13.11 |
| t25 | t25_inv | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 917 | 94 | 202 | 75.60 | -1.0840 | 17.43 | 16.83 |
| t25 | t25_inv | m00_poisson | Poisson | 954 | 73 | 201 | 77.69 | -0.5931 | 17.34 | 13.45 |
| t25 | t25_inv | m05_poisson | Joint Gamma-Poisson | 887 | 82 | 237 | 73.55 | -0.9130 | 17.64 | 15.21 |
| t25 | t25_inv | m10_poisson | Poisson | 963 | 83 | 180 | 78.55 | -0.5986 | 16.57 | 12.92 |
| t25 | t25_inv | m12_poisson | Joint Gamma-Poisson | 905 | 79 | 213 | 75.61 | -1.1843 | 18.80 | 17.69 |

## NegBin vs Poisson, inside each calibration state

| environment | variant | pair | n_shared | n_only_a | n_only_b | overlap_pct | sizing_delta_pnl | shared_roi_a_pct | shared_roi_b_pct | capture_ratio_a | capture_ratio_b |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | raw | m00_baseline_grw_negbin vs m00_poisson | 1258 | 42 | 38 | 94.02 | -0.0376 | 12.09 | 12.34 | 1.017 | 1.036 |
| close | raw | m05_wealth_grw_negbin vs m05_poisson | 1188 | 47 | 59 | 91.81 | +0.0136 | 11.71 | 11.73 | 1.041 | 1.080 |
| close | raw | m10_lineup_grw_negbin vs m10_poisson | 1255 | 46 | 44 | 93.31 | +0.0470 | 10.96 | 10.73 | 0.924 | 0.957 |
| close | raw | m12_joint_hybrid_synergy_negbin vs m12_poisson | 1209 | 35 | 44 | 93.87 | -0.1270 | 10.54 | 11.40 | 1.007 | 1.043 |
| close | close_std | m00_baseline_grw_negbin vs m00_poisson | 1246 | 49 | 46 | 92.92 | -0.0109 | 11.51 | 11.65 | 1.015 | 1.029 |
| close | close_std | m05_wealth_grw_negbin vs m05_poisson | 1180 | 47 | 53 | 92.19 | +0.0023 | 11.31 | 11.42 | 1.030 | 1.058 |
| close | close_std | m10_lineup_grw_negbin vs m10_poisson | 1242 | 49 | 45 | 92.96 | +0.0744 | 10.78 | 10.42 | 0.928 | 0.958 |
| close | close_std | m12_joint_hybrid_synergy_negbin vs m12_poisson | 1202 | 41 | 41 | 93.61 | -0.1149 | 10.27 | 11.10 | 0.990 | 1.039 |
| close | close_std_t007 | m00_baseline_grw_negbin vs m00_poisson | 1244 | 49 | 46 | 92.91 | +0.0035 | 10.49 | 10.57 | 1.005 | 1.037 |
| close | close_std_t007 | m05_wealth_grw_negbin vs m05_poisson | 1180 | 48 | 54 | 92.04 | +0.0181 | 11.04 | 11.04 | 1.036 | 1.073 |
| close | close_std_t007 | m10_lineup_grw_negbin vs m10_poisson | 1244 | 53 | 46 | 92.63 | +0.1063 | 10.49 | 9.95 | 0.919 | 0.961 |
| close | close_std_t007 | m12_joint_hybrid_synergy_negbin vs m12_poisson | 1203 | 41 | 43 | 93.47 | -0.0971 | 9.96 | 10.71 | 1.003 | 1.054 |
| t25 | raw | m00_baseline_grw_negbin vs m00_poisson | 1116 | 33 | 39 | 93.94 | -0.0949 | 12.06 | 12.66 | 0.981 | 1.022 |
| t25 | raw | m05_wealth_grw_negbin vs m05_poisson | 1081 | 41 | 43 | 92.79 | -0.0188 | 14.04 | 14.26 | 1.066 | 1.110 |
| t25 | raw | m10_lineup_grw_negbin vs m10_poisson | 1113 | 40 | 30 | 94.08 | +0.0492 | 12.61 | 12.40 | 0.944 | 0.947 |
| t25 | raw | m12_joint_hybrid_synergy_negbin vs m12_poisson | 1084 | 35 | 34 | 94.02 | -0.0839 | 15.68 | 16.28 | 1.076 | 1.122 |
| t25 | t25_inv | m00_baseline_grw_negbin vs m00_poisson | 993 | 70 | 34 | 90.52 | -0.0228 | 16.83 | 17.95 | 0.999 | 1.013 |
| t25 | t25_inv | m05_wealth_grw_negbin vs m05_poisson | 929 | 69 | 40 | 89.50 | +0.0380 | 16.62 | 17.27 | 1.035 | 1.067 |
| t25 | t25_inv | m10_lineup_grw_negbin vs m10_poisson | 1009 | 62 | 37 | 91.06 | -0.0052 | 15.64 | 16.57 | 0.928 | 0.949 |
| t25 | t25_inv | m12_joint_hybrid_synergy_negbin vs m12_poisson | 943 | 68 | 41 | 89.64 | +0.0134 | 17.82 | 18.84 | 0.979 | 1.013 |

## The Over 1.5 drag

`n_only_b` is the count of Over 1.5 bets the POISSON control struck and the NegBin rung declined; `declined_by_a_pct` is that as a share of the control's whole Over 1.5 book, and `only_b_roi_pct` is what those declined bets returned the control. The question is whether calibration shrinks the first two.

| environment | variant | pair | n_a | n_b | n_shared | n_only_a | n_only_b | declined_by_a_pct | only_b_roi_pct | only_b_win_rate_pct | shared_p_model_a | shared_p_model_b |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close | raw | m00_baseline_grw_negbin vs m00_poisson | 33 | 50 | 33 | 0 | 17 | 34.00 | 20.81 | 94.12 | 0.7955 | 0.8038 |
| close | raw | m05_wealth_grw_negbin vs m05_poisson | 46 | 64 | 46 | 0 | 18 | 28.12 | 16.00 | 83.33 | 0.7898 | 0.7978 |
| close | raw | m10_lineup_grw_negbin vs m10_poisson | 27 | 47 | 27 | 0 | 20 | 42.55 | 19.95 | 95.00 | 0.7839 | 0.7933 |
| close | raw | m12_joint_hybrid_synergy_negbin vs m12_poisson | 45 | 60 | 45 | 0 | 15 | 25.00 | 17.33 | 80.00 | 0.7849 | 0.7930 |
| close | close_std | m00_baseline_grw_negbin vs m00_poisson | 31 | 49 | 31 | 0 | 18 | 36.73 | 15.66 | 88.89 | 0.7920 | 0.8004 |
| close | close_std | m05_wealth_grw_negbin vs m05_poisson | 43 | 63 | 43 | 0 | 20 | 31.75 | 5.91 | 80.00 | 0.7885 | 0.7966 |
| close | close_std | m10_lineup_grw_negbin vs m10_poisson | 25 | 46 | 25 | 0 | 21 | 45.65 | 21.17 | 95.24 | 0.7789 | 0.7881 |
| close | close_std | m12_joint_hybrid_synergy_negbin vs m12_poisson | 44 | 59 | 44 | 0 | 15 | 25.42 | 16.98 | 80.00 | 0.7811 | 0.7891 |
| close | close_std_t007 | m00_baseline_grw_negbin vs m00_poisson | 30 | 48 | 30 | 0 | 18 | 37.50 | 15.90 | 83.33 | 0.7899 | 0.7983 |
| close | close_std_t007 | m05_wealth_grw_negbin vs m05_poisson | 44 | 62 | 44 | 0 | 18 | 29.03 | 3.71 | 77.78 | 0.7854 | 0.7934 |
| close | close_std_t007 | m10_lineup_grw_negbin vs m10_poisson | 26 | 47 | 26 | 0 | 21 | 44.68 | 21.38 | 95.24 | 0.7729 | 0.7819 |
| close | close_std_t007 | m12_joint_hybrid_synergy_negbin vs m12_poisson | 43 | 58 | 43 | 0 | 15 | 25.86 | 16.11 | 80.00 | 0.7799 | 0.7878 |
| t25 | raw | m00_baseline_grw_negbin vs m00_poisson | 28 | 39 | 28 | 0 | 11 | 28.21 | -20.03 | 63.64 | 0.7868 | 0.7947 |
| t25 | raw | m05_wealth_grw_negbin vs m05_poisson | 45 | 58 | 45 | 0 | 13 | 22.41 | 5.79 | 76.92 | 0.7879 | 0.7956 |
| t25 | raw | m10_lineup_grw_negbin vs m10_poisson | 30 | 42 | 30 | 0 | 12 | 28.57 | 8.23 | 83.33 | 0.7754 | 0.7841 |
| t25 | raw | m12_joint_hybrid_synergy_negbin vs m12_poisson | 43 | 55 | 43 | 0 | 12 | 21.82 | -12.76 | 83.33 | 0.7817 | 0.7892 |
| t25 | t25_inv | m00_baseline_grw_negbin vs m00_poisson | 16 | 30 | 16 | 0 | 14 | 46.67 | 8.33 | 85.71 | 0.7647 | 0.7727 |
| t25 | t25_inv | m05_wealth_grw_negbin vs m05_poisson | 26 | 41 | 26 | 0 | 15 | 36.59 | 18.28 | 93.33 | 0.7671 | 0.7745 |
| t25 | t25_inv | m10_lineup_grw_negbin vs m10_poisson | 16 | 34 | 16 | 0 | 18 | 52.94 | 4.29 | 77.78 | 0.7613 | 0.7701 |
| t25 | t25_inv | m12_joint_hybrid_synergy_negbin vs m12_poisson | 26 | 42 | 26 | 0 | 16 | 38.10 | 6.75 | 87.50 | 0.7604 | 0.7680 |

## Return by selection family

| environment | variant | selection_family | model | likelihood | n_bets | win_rate_pct | roi_pct | edge_mean_pp | capture_ratio |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| close | close_std | 1X2_away | m00_baseline_grw_negbin | NegBin | 405 | 26.42 | 15.47 | 5.78 | 0.941 |
| close | close_std | 1X2_away | m00_poisson | Poisson | 389 | 24.68 | 15.85 | 5.82 | 1.068 |
| close | close_std | 1X2_away | m05_poisson | Joint Gamma-Poisson | 406 | 27.09 | 6.94 | 5.03 | 0.894 |
| close | close_std | 1X2_away | m05_wealth_grw_negbin | Joint Gamma-NegBin | 423 | 27.42 | 7.30 | 5.00 | 0.872 |
| close | close_std | 1X2_away | m10_lineup_grw_negbin | NegBin | 390 | 26.41 | 12.40 | 6.28 | 0.891 |
| close | close_std | 1X2_away | m10_poisson | Poisson | 380 | 25.79 | 12.88 | 6.27 | 0.944 |
| close | close_std | 1X2_away | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 426 | 27.46 | 8.46 | 5.27 | 0.830 |
| close | close_std | 1X2_away | m12_poisson | Joint Gamma-Poisson | 416 | 26.92 | 9.08 | 5.21 | 0.880 |
| close | close_std | 1X2_draw | m00_baseline_grw_negbin | NegBin | 290 | 23.45 | 9.20 | 0.50 | 2.345 |
| close | close_std | 1X2_draw | m00_poisson | Poisson | 301 | 24.25 | 8.86 | 0.54 | 2.009 |
| close | close_std | 1X2_draw | m05_poisson | Joint Gamma-Poisson | 257 | 24.12 | 3.65 | 0.50 | 1.162 |
| close | close_std | 1X2_draw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 236 | 25.00 | 4.84 | 0.54 | 0.933 |
| close | close_std | 1X2_draw | m10_lineup_grw_negbin | NegBin | 304 | 25.33 | 6.79 | 0.49 | 1.448 |
| close | close_std | 1X2_draw | m10_poisson | Poisson | 316 | 24.37 | 5.44 | 0.54 | 1.584 |
| close | close_std | 1X2_draw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 249 | 23.69 | 1.83 | 0.65 | 1.164 |
| close | close_std | 1X2_draw | m12_poisson | Joint Gamma-Poisson | 266 | 24.44 | 0.95 | 0.66 | 1.176 |
| close | close_std | 1X2_home | m00_baseline_grw_negbin | NegBin | 337 | 39.17 | 10.96 | 5.72 | 0.774 |
| close | close_std | 1X2_home | m00_poisson | Poisson | 330 | 39.39 | 11.60 | 5.87 | 0.792 |
| close | close_std | 1X2_home | m05_poisson | Joint Gamma-Poisson | 311 | 38.91 | 19.89 | 4.54 | 0.967 |
| close | close_std | 1X2_home | m05_wealth_grw_negbin | Joint Gamma-NegBin | 316 | 38.92 | 19.64 | 4.49 | 0.924 |
| close | close_std | 1X2_home | m10_lineup_grw_negbin | NegBin | 333 | 39.64 | 12.79 | 6.15 | 0.758 |
| close | close_std | 1X2_home | m10_poisson | Poisson | 322 | 39.75 | 11.79 | 6.34 | 0.763 |
| close | close_std | 1X2_home | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 314 | 37.90 | 16.94 | 4.67 | 0.900 |
| close | close_std | 1X2_home | m12_poisson | Joint Gamma-Poisson | 306 | 37.25 | 18.07 | 4.83 | 0.968 |
| close | close_std | O/U 1.5_over_15 | m00_baseline_grw_negbin | NegBin | 31 | 70.97 | -1.08 | 2.98 | 1.569 |
| close | close_std | O/U 1.5_over_15 | m00_poisson | Poisson | 49 | 77.55 | 2.99 | 3.20 | 1.139 |
| close | close_std | O/U 1.5_over_15 | m05_poisson | Joint Gamma-Poisson | 63 | 74.60 | 0.58 | 3.18 | 1.270 |
| close | close_std | O/U 1.5_over_15 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 43 | 72.09 | -0.33 | 3.05 | 1.541 |
| close | close_std | O/U 1.5_over_15 | m10_lineup_grw_negbin | NegBin | 25 | 64.00 | -4.92 | 2.67 | 1.998 |
| close | close_std | O/U 1.5_over_15 | m10_poisson | Poisson | 46 | 78.26 | -0.38 | 2.76 | 1.148 |
| close | close_std | O/U 1.5_over_15 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 44 | 70.45 | -0.45 | 2.83 | 1.462 |
| close | close_std | O/U 1.5_over_15 | m12_poisson | Joint Gamma-Poisson | 59 | 72.88 | 0.41 | 3.06 | 1.233 |
| close | close_std | O/U 2.5_under_25 | m00_baseline_grw_negbin | NegBin | 232 | 49.14 | 8.44 | 4.97 | 1.078 |
| close | close_std | O/U 2.5_under_25 | m00_poisson | Poisson | 223 | 51.12 | 7.82 | 5.03 | 0.995 |
| close | close_std | O/U 2.5_under_25 | m05_poisson | Joint Gamma-Poisson | 196 | 49.49 | 10.00 | 4.88 | 1.125 |
| close | close_std | O/U 2.5_under_25 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 209 | 47.85 | 8.30 | 4.85 | 1.187 |
| close | close_std | O/U 2.5_under_25 | m10_lineup_grw_negbin | NegBin | 239 | 49.79 | 7.85 | 4.62 | 0.972 |
| close | close_std | O/U 2.5_under_25 | m10_poisson | Poisson | 223 | 48.88 | 8.00 | 4.70 | 1.035 |
| close | close_std | O/U 2.5_under_25 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 210 | 48.10 | 7.56 | 4.78 | 1.118 |
| close | close_std | O/U 2.5_under_25 | m12_poisson | Joint Gamma-Poisson | 196 | 48.47 | 10.04 | 4.81 | 1.162 |
| close | close_std_t007 | 1X2_away | m00_baseline_grw_negbin | NegBin | 404 | 26.49 | 12.63 | 5.55 | 0.940 |
| close | close_std_t007 | 1X2_away | m00_poisson | Poisson | 389 | 24.68 | 12.78 | 5.54 | 1.072 |
| close | close_std_t007 | 1X2_away | m05_poisson | Joint Gamma-Poisson | 407 | 27.03 | 6.87 | 4.89 | 0.911 |
| close | close_std_t007 | 1X2_away | m05_wealth_grw_negbin | Joint Gamma-NegBin | 423 | 27.42 | 7.44 | 4.88 | 0.883 |
| close | close_std_t007 | 1X2_away | m10_lineup_grw_negbin | NegBin | 390 | 26.41 | 10.96 | 5.97 | 0.908 |
| close | close_std_t007 | 1X2_away | m10_poisson | Poisson | 379 | 25.59 | 11.21 | 5.97 | 0.975 |
| close | close_std_t007 | 1X2_away | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 426 | 27.70 | 8.75 | 5.13 | 0.840 |
| close | close_std_t007 | 1X2_away | m12_poisson | Joint Gamma-Poisson | 417 | 26.86 | 9.22 | 5.03 | 0.905 |
| close | close_std_t007 | 1X2_draw | m00_baseline_grw_negbin | NegBin | 289 | 23.53 | 8.08 | 0.39 | 2.163 |
| close | close_std_t007 | 1X2_draw | m00_poisson | Poisson | 300 | 24.00 | 7.46 | 0.41 | 2.042 |
| close | close_std_t007 | 1X2_draw | m05_poisson | Joint Gamma-Poisson | 260 | 24.23 | 2.42 | 0.39 | 1.233 |
| close | close_std_t007 | 1X2_draw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 236 | 25.00 | 3.59 | 0.44 | 0.948 |
| close | close_std_t007 | 1X2_draw | m10_lineup_grw_negbin | NegBin | 305 | 25.90 | 5.14 | 0.37 | 1.100 |
| close | close_std_t007 | 1X2_draw | m10_poisson | Poisson | 317 | 24.92 | 3.32 | 0.40 | 1.224 |
| close | close_std_t007 | 1X2_draw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 251 | 23.51 | 0.07 | 0.50 | 1.274 |
| close | close_std_t007 | 1X2_draw | m12_poisson | Joint Gamma-Poisson | 269 | 24.54 | -0.86 | 0.50 | 1.157 |
| close | close_std_t007 | 1X2_home | m00_baseline_grw_negbin | NegBin | 337 | 39.17 | 10.94 | 5.48 | 0.784 |
| close | close_std_t007 | 1X2_home | m00_poisson | Poisson | 329 | 39.21 | 11.50 | 5.63 | 0.810 |
| close | close_std_t007 | 1X2_home | m05_poisson | Joint Gamma-Poisson | 311 | 38.91 | 19.61 | 4.46 | 0.969 |
| close | close_std_t007 | 1X2_home | m05_wealth_grw_negbin | Joint Gamma-NegBin | 316 | 38.92 | 19.40 | 4.41 | 0.927 |
| close | close_std_t007 | 1X2_home | m10_lineup_grw_negbin | NegBin | 333 | 39.64 | 13.90 | 5.86 | 0.779 |
| close | close_std_t007 | 1X2_home | m10_poisson | Poisson | 322 | 39.75 | 12.77 | 6.04 | 0.783 |
| close | close_std_t007 | 1X2_home | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 313 | 37.70 | 16.74 | 4.58 | 0.923 |
| close | close_std_t007 | 1X2_home | m12_poisson | Joint Gamma-Poisson | 305 | 37.05 | 17.86 | 4.72 | 0.988 |
| close | close_std_t007 | O/U 1.5_over_15 | m00_baseline_grw_negbin | NegBin | 30 | 73.33 | -1.75 | 2.86 | 1.790 |
| close | close_std_t007 | O/U 1.5_over_15 | m00_poisson | Poisson | 48 | 77.08 | 2.86 | 2.99 | 1.462 |
| close | close_std_t007 | O/U 1.5_over_15 | m05_poisson | Joint Gamma-Poisson | 62 | 74.19 | 0.10 | 3.13 | 1.289 |
| close | close_std_t007 | O/U 1.5_over_15 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 44 | 72.73 | -0.58 | 2.93 | 1.542 |
| close | close_std_t007 | O/U 1.5_over_15 | m10_lineup_grw_negbin | NegBin | 26 | 65.38 | -6.15 | 2.31 | 1.642 |
| close | close_std_t007 | O/U 1.5_over_15 | m10_poisson | Poisson | 47 | 78.72 | -0.85 | 2.57 | 1.056 |
| close | close_std_t007 | O/U 1.5_over_15 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 43 | 72.09 | -1.06 | 2.74 | 1.471 |
| close | close_std_t007 | O/U 1.5_over_15 | m12_poisson | Joint Gamma-Poisson | 58 | 74.14 | 0.01 | 2.97 | 1.245 |
| close | close_std_t007 | O/U 2.5_under_25 | m00_baseline_grw_negbin | NegBin | 233 | 49.36 | 8.52 | 4.53 | 1.065 |
| close | close_std_t007 | O/U 2.5_under_25 | m00_poisson | Poisson | 224 | 50.89 | 7.94 | 4.55 | 1.000 |
| close | close_std_t007 | O/U 2.5_under_25 | m05_poisson | Joint Gamma-Poisson | 194 | 49.48 | 9.13 | 4.63 | 1.151 |
| close | close_std_t007 | O/U 2.5_under_25 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 209 | 47.85 | 7.48 | 4.58 | 1.202 |
| close | close_std_t007 | O/U 2.5_under_25 | m10_lineup_grw_negbin | NegBin | 243 | 48.97 | 7.26 | 4.19 | 0.952 |
| close | close_std_t007 | O/U 2.5_under_25 | m10_poisson | Poisson | 225 | 48.89 | 7.44 | 4.27 | 1.048 |
| close | close_std_t007 | O/U 2.5_under_25 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 211 | 47.87 | 6.51 | 4.48 | 1.118 |
| close | close_std_t007 | O/U 2.5_under_25 | m12_poisson | Joint Gamma-Poisson | 197 | 48.22 | 8.86 | 4.46 | 1.175 |
| close | raw | 1X2_away | m00_baseline_grw_negbin | NegBin | 400 | 26.00 | 17.85 | 6.39 | 0.960 |
| close | raw | 1X2_away | m00_poisson | Poisson | 388 | 24.48 | 18.44 | 6.39 | 1.079 |
| close | raw | 1X2_away | m05_poisson | Joint Gamma-Poisson | 403 | 26.55 | 7.77 | 5.50 | 0.932 |
| close | raw | 1X2_away | m05_wealth_grw_negbin | Joint Gamma-NegBin | 420 | 27.38 | 8.29 | 5.43 | 0.875 |
| close | raw | 1X2_away | m10_lineup_grw_negbin | NegBin | 390 | 26.41 | 14.04 | 6.86 | 0.887 |
| close | raw | 1X2_away | m10_poisson | Poisson | 378 | 25.40 | 14.78 | 6.94 | 0.967 |
| close | raw | 1X2_away | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 421 | 27.32 | 9.30 | 5.77 | 0.833 |
| close | raw | 1X2_away | m12_poisson | Joint Gamma-Poisson | 413 | 26.88 | 10.03 | 5.70 | 0.878 |
| close | raw | 1X2_draw | m00_baseline_grw_negbin | NegBin | 300 | 24.00 | 7.92 | 0.50 | 2.461 |
| close | raw | 1X2_draw | m00_poisson | Poisson | 306 | 24.51 | 7.89 | 0.59 | 2.041 |
| close | raw | 1X2_draw | m05_poisson | Joint Gamma-Poisson | 273 | 24.18 | 3.38 | 0.51 | 1.317 |
| close | raw | 1X2_draw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 244 | 24.59 | 4.72 | 0.61 | 1.108 |
| close | raw | 1X2_draw | m10_lineup_grw_negbin | NegBin | 315 | 25.71 | 5.30 | 0.46 | 1.612 |
| close | raw | 1X2_draw | m10_poisson | Poisson | 326 | 25.46 | 3.92 | 0.55 | 1.518 |
| close | raw | 1X2_draw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 260 | 23.85 | 1.48 | 0.70 | 1.198 |
| close | raw | 1X2_draw | m12_poisson | Joint Gamma-Poisson | 282 | 24.47 | 0.55 | 0.65 | 1.211 |
| close | raw | 1X2_home | m00_baseline_grw_negbin | NegBin | 336 | 38.99 | 11.14 | 6.27 | 0.778 |
| close | raw | 1X2_home | m00_poisson | Poisson | 328 | 39.02 | 11.79 | 6.47 | 0.810 |
| close | raw | 1X2_home | m05_poisson | Joint Gamma-Poisson | 310 | 39.03 | 19.99 | 4.90 | 0.953 |
| close | raw | 1X2_home | m05_wealth_grw_negbin | Joint Gamma-NegBin | 315 | 39.05 | 19.57 | 4.84 | 0.913 |
| close | raw | 1X2_home | m10_lineup_grw_negbin | NegBin | 330 | 39.70 | 12.34 | 6.79 | 0.745 |
| close | raw | 1X2_home | m10_poisson | Poisson | 320 | 39.38 | 11.56 | 7.00 | 0.770 |
| close | raw | 1X2_home | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 308 | 37.34 | 16.86 | 5.17 | 0.935 |
| close | raw | 1X2_home | m12_poisson | Joint Gamma-Poisson | 301 | 37.21 | 18.12 | 5.32 | 0.964 |
| close | raw | O/U 1.5_over_15 | m00_baseline_grw_negbin | NegBin | 33 | 69.70 | -0.05 | 3.38 | 1.439 |
| close | raw | O/U 1.5_over_15 | m00_poisson | Poisson | 50 | 78.00 | 3.65 | 3.49 | 1.088 |
| close | raw | O/U 1.5_over_15 | m05_poisson | Joint Gamma-Poisson | 64 | 75.00 | 1.25 | 3.47 | 1.279 |
| close | raw | O/U 1.5_over_15 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 46 | 71.74 | 0.63 | 3.30 | 1.535 |
| close | raw | O/U 1.5_over_15 | m10_lineup_grw_negbin | NegBin | 27 | 66.67 | -2.29 | 2.93 | 1.980 |
| close | raw | O/U 1.5_over_15 | m10_poisson | Poisson | 47 | 78.72 | 1.64 | 2.97 | 1.166 |
| close | raw | O/U 1.5_over_15 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 45 | 71.11 | 0.68 | 3.15 | 1.430 |
| close | raw | O/U 1.5_over_15 | m12_poisson | Joint Gamma-Poisson | 60 | 73.33 | 1.29 | 3.32 | 1.220 |
| close | raw | O/U 2.5_under_25 | m00_baseline_grw_negbin | NegBin | 231 | 49.35 | 8.07 | 5.42 | 1.066 |
| close | raw | O/U 2.5_under_25 | m00_poisson | Poisson | 224 | 50.89 | 7.54 | 5.51 | 0.997 |
| close | raw | O/U 2.5_under_25 | m05_poisson | Joint Gamma-Poisson | 197 | 49.24 | 10.75 | 5.32 | 1.142 |
| close | raw | O/U 2.5_under_25 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 210 | 47.62 | 8.91 | 5.26 | 1.203 |
| close | raw | O/U 2.5_under_25 | m10_lineup_grw_negbin | NegBin | 239 | 49.37 | 7.61 | 5.00 | 0.988 |
| close | raw | O/U 2.5_under_25 | m10_poisson | Poisson | 228 | 49.12 | 7.77 | 5.09 | 1.044 |
| close | raw | O/U 2.5_under_25 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 210 | 48.10 | 7.59 | 5.19 | 1.127 |
| close | raw | O/U 2.5_under_25 | m12_poisson | Joint Gamma-Poisson | 197 | 48.22 | 10.12 | 5.24 | 1.167 |
| t25 | raw | 1X2_away | m00_baseline_grw_negbin | NegBin | 348 | 27.01 | 13.07 | 6.55 | 0.850 |
| t25 | raw | 1X2_away | m00_poisson | Poisson | 341 | 26.69 | 13.84 | 6.48 | 0.882 |
| t25 | raw | 1X2_away | m05_poisson | Joint Gamma-Poisson | 363 | 26.45 | 8.44 | 5.62 | 0.858 |
| t25 | raw | 1X2_away | m05_wealth_grw_negbin | Joint Gamma-NegBin | 378 | 27.25 | 9.18 | 5.55 | 0.802 |
| t25 | raw | 1X2_away | m10_lineup_grw_negbin | NegBin | 350 | 26.57 | 12.26 | 6.91 | 0.846 |
| t25 | raw | 1X2_away | m10_poisson | Poisson | 336 | 25.89 | 12.34 | 7.00 | 0.890 |
| t25 | raw | 1X2_away | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 376 | 26.60 | 14.35 | 5.98 | 0.858 |
| t25 | raw | 1X2_away | m12_poisson | Joint Gamma-Poisson | 365 | 26.85 | 14.70 | 5.98 | 0.860 |
| t25 | raw | 1X2_draw | m00_baseline_grw_negbin | NegBin | 279 | 26.16 | 8.73 | 0.44 | 1.117 |
| t25 | raw | 1X2_draw | m00_poisson | Poisson | 289 | 26.64 | 8.40 | 0.47 | 1.051 |
| t25 | raw | 1X2_draw | m05_poisson | Joint Gamma-Poisson | 252 | 23.81 | 0.86 | 0.37 | 1.889 |
| t25 | raw | 1X2_draw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 238 | 24.37 | 0.04 | 0.39 | 1.627 |
| t25 | raw | 1X2_draw | m10_lineup_grw_negbin | NegBin | 286 | 26.92 | 5.91 | 0.50 | 1.076 |
| t25 | raw | 1X2_draw | m10_poisson | Poisson | 291 | 26.80 | 5.01 | 0.60 | 1.100 |
| t25 | raw | 1X2_draw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 245 | 23.67 | 1.42 | 0.64 | 1.260 |
| t25 | raw | 1X2_draw | m12_poisson | Joint Gamma-Poisson | 260 | 23.46 | 1.21 | 0.67 | 1.404 |
| t25 | raw | 1X2_home | m00_baseline_grw_negbin | NegBin | 300 | 36.00 | 11.93 | 6.63 | 0.932 |
| t25 | raw | 1X2_home | m00_poisson | Poisson | 300 | 36.00 | 12.42 | 6.58 | 0.960 |
| t25 | raw | 1X2_home | m05_poisson | Joint Gamma-Poisson | 281 | 36.65 | 20.76 | 4.90 | 1.061 |
| t25 | raw | 1X2_home | m05_wealth_grw_negbin | Joint Gamma-NegBin | 288 | 36.46 | 20.09 | 4.78 | 1.051 |
| t25 | raw | 1X2_home | m10_lineup_grw_negbin | NegBin | 292 | 35.62 | 12.89 | 7.13 | 0.937 |
| t25 | raw | 1X2_home | m10_poisson | Poisson | 287 | 36.59 | 12.68 | 7.23 | 0.901 |
| t25 | raw | 1X2_home | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 282 | 34.40 | 18.30 | 5.08 | 1.104 |
| t25 | raw | 1X2_home | m12_poisson | Joint Gamma-Poisson | 275 | 33.45 | 19.41 | 5.24 | 1.188 |
| t25 | raw | O/U 1.5_over_15 | m00_baseline_grw_negbin | NegBin | 28 | 89.29 | 21.83 | 3.43 | 1.041 |
| t25 | raw | O/U 1.5_over_15 | m00_poisson | Poisson | 39 | 82.05 | 16.78 | 3.70 | 1.477 |
| t25 | raw | O/U 1.5_over_15 | m05_poisson | Joint Gamma-Poisson | 58 | 84.48 | 18.60 | 3.86 | 1.655 |
| t25 | raw | O/U 1.5_over_15 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 45 | 86.67 | 20.80 | 3.81 | 1.442 |
| t25 | raw | O/U 1.5_over_15 | m10_lineup_grw_negbin | NegBin | 30 | 80.00 | 19.06 | 2.70 | 1.358 |
| t25 | raw | O/U 1.5_over_15 | m10_poisson | Poisson | 42 | 80.95 | 12.19 | 3.12 | 1.319 |
| t25 | raw | O/U 1.5_over_15 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 43 | 88.37 | 24.11 | 3.26 | 1.198 |
| t25 | raw | O/U 1.5_over_15 | m12_poisson | Joint Gamma-Poisson | 55 | 87.27 | 19.65 | 3.77 | 1.344 |
| t25 | raw | O/U 2.5_under_25 | m00_baseline_grw_negbin | NegBin | 194 | 52.06 | 12.39 | 5.45 | 1.041 |
| t25 | raw | O/U 2.5_under_25 | m00_poisson | Poisson | 186 | 50.54 | 12.70 | 5.48 | 1.119 |
| t25 | raw | O/U 2.5_under_25 | m05_poisson | Joint Gamma-Poisson | 170 | 52.94 | 17.81 | 5.48 | 1.149 |
| t25 | raw | O/U 2.5_under_25 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 173 | 53.76 | 17.16 | 5.67 | 1.117 |
| t25 | raw | O/U 2.5_under_25 | m10_lineup_grw_negbin | NegBin | 195 | 53.85 | 16.16 | 5.11 | 1.009 |
| t25 | raw | O/U 2.5_under_25 | m10_poisson | Poisson | 187 | 54.01 | 17.02 | 5.02 | 1.016 |
| t25 | raw | O/U 2.5_under_25 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 173 | 53.76 | 19.16 | 5.63 | 1.178 |
| t25 | raw | O/U 2.5_under_25 | m12_poisson | Joint Gamma-Poisson | 163 | 53.99 | 20.19 | 5.65 | 1.173 |
| t25 | t25_inv | 1X2_away | m00_baseline_grw_negbin | NegBin | 369 | 28.46 | 13.05 | 3.21 | 0.797 |
| t25 | t25_inv | 1X2_away | m00_poisson | Poisson | 348 | 27.87 | 13.60 | 3.13 | 0.827 |
| t25 | t25_inv | 1X2_away | m05_poisson | Joint Gamma-Poisson | 358 | 25.42 | 7.49 | 2.70 | 0.927 |
| t25 | t25_inv | 1X2_away | m05_wealth_grw_negbin | Joint Gamma-NegBin | 386 | 26.94 | 7.89 | 2.76 | 0.850 |
| t25 | t25_inv | 1X2_away | m10_lineup_grw_negbin | NegBin | 372 | 28.49 | 8.66 | 3.33 | 0.722 |
| t25 | t25_inv | 1X2_away | m10_poisson | Poisson | 349 | 26.93 | 8.96 | 3.28 | 0.792 |
| t25 | t25_inv | 1X2_away | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 396 | 28.03 | 8.44 | 2.86 | 0.786 |
| t25 | t25_inv | 1X2_away | m12_poisson | Joint Gamma-Poisson | 373 | 27.88 | 9.25 | 2.78 | 0.819 |
| t25 | t25_inv | 1X2_draw | m00_baseline_grw_negbin | NegBin | 170 | 27.06 | 33.43 | 0.97 | 1.001 |
| t25 | t25_inv | 1X2_draw | m00_poisson | Poisson | 174 | 25.29 | 33.66 | 1.04 | 1.157 |
| t25 | t25_inv | 1X2_draw | m05_poisson | Joint Gamma-Poisson | 126 | 23.81 | 24.13 | 0.86 | 1.155 |
| t25 | t25_inv | 1X2_draw | m05_wealth_grw_negbin | Joint Gamma-NegBin | 118 | 22.88 | 21.06 | 0.97 | 1.181 |
| t25 | t25_inv | 1X2_draw | m10_lineup_grw_negbin | NegBin | 177 | 25.42 | 29.37 | 1.01 | 1.201 |
| t25 | t25_inv | 1X2_draw | m10_poisson | Poisson | 180 | 25.56 | 29.43 | 1.14 | 1.145 |
| t25 | t25_inv | 1X2_draw | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 124 | 23.39 | 27.97 | 1.17 | 1.100 |
| t25 | t25_inv | 1X2_draw | m12_poisson | Joint Gamma-Poisson | 134 | 24.63 | 28.82 | 1.11 | 1.027 |
| t25 | t25_inv | 1X2_home | m00_baseline_grw_negbin | NegBin | 314 | 35.99 | 13.43 | 3.13 | 0.922 |
| t25 | t25_inv | 1X2_home | m00_poisson | Poisson | 303 | 36.96 | 14.79 | 3.13 | 0.905 |
| t25 | t25_inv | 1X2_home | m05_poisson | Joint Gamma-Poisson | 292 | 38.36 | 22.24 | 2.13 | 0.953 |
| t25 | t25_inv | 1X2_home | m05_wealth_grw_negbin | Joint Gamma-NegBin | 300 | 37.33 | 21.09 | 2.19 | 0.919 |
| t25 | t25_inv | 1X2_home | m10_lineup_grw_negbin | NegBin | 313 | 36.10 | 12.71 | 3.24 | 0.865 |
| t25 | t25_inv | 1X2_home | m10_poisson | Poisson | 304 | 36.18 | 13.12 | 3.22 | 0.869 |
| t25 | t25_inv | 1X2_home | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 297 | 37.04 | 20.32 | 2.32 | 0.928 |
| t25 | t25_inv | 1X2_home | m12_poisson | Joint Gamma-Poisson | 285 | 36.49 | 21.56 | 2.37 | 1.000 |
| t25 | t25_inv | O/U 1.5_over_15 | m00_baseline_grw_negbin | NegBin | 16 | 93.75 | 27.41 | 1.56 | 0.785 |
| t25 | t25_inv | O/U 1.5_over_15 | m00_poisson | Poisson | 30 | 90.00 | 20.88 | 1.84 | 1.114 |
| t25 | t25_inv | O/U 1.5_over_15 | m05_poisson | Joint Gamma-Poisson | 41 | 90.24 | 22.80 | 2.21 | 1.468 |
| t25 | t25_inv | O/U 1.5_over_15 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 26 | 88.46 | 24.60 | 2.15 | 2.147 |
| t25 | t25_inv | O/U 1.5_over_15 | m10_lineup_grw_negbin | NegBin | 16 | 93.75 | 31.34 | 1.22 | 0.984 |
| t25 | t25_inv | O/U 1.5_over_15 | m10_poisson | Poisson | 34 | 85.29 | 21.25 | 1.50 | 1.584 |
| t25 | t25_inv | O/U 1.5_over_15 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 26 | 92.31 | 28.53 | 1.70 | 4.261 |
| t25 | t25_inv | O/U 1.5_over_15 | m12_poisson | Joint Gamma-Poisson | 42 | 90.48 | 25.01 | 1.87 | 1.830 |
| t25 | t25_inv | O/U 2.5_under_25 | m00_baseline_grw_negbin | NegBin | 194 | 51.55 | 19.09 | 3.56 | 1.274 |
| t25 | t25_inv | O/U 2.5_under_25 | m00_poisson | Poisson | 172 | 53.49 | 20.70 | 3.60 | 1.231 |
| t25 | t25_inv | O/U 2.5_under_25 | m05_poisson | Joint Gamma-Poisson | 152 | 53.95 | 21.01 | 3.30 | 1.166 |
| t25 | t25_inv | O/U 2.5_under_25 | m05_wealth_grw_negbin | Joint Gamma-NegBin | 168 | 52.98 | 20.63 | 3.38 | 1.190 |
| t25 | t25_inv | O/U 2.5_under_25 | m10_lineup_grw_negbin | NegBin | 193 | 52.85 | 23.23 | 3.30 | 1.208 |
| t25 | t25_inv | O/U 2.5_under_25 | m10_poisson | Poisson | 179 | 53.07 | 25.48 | 3.20 | 1.243 |
| t25 | t25_inv | O/U 2.5_under_25 | m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 168 | 55.36 | 23.02 | 3.40 | 1.151 |
| t25 | t25_inv | O/U 2.5_under_25 | m12_poisson | Joint Gamma-Poisson | 150 | 54.67 | 23.77 | 3.37 | 1.177 |

## r05 reproduction gate

| model | r06_return_pct | r05_return_pct | delta_return_pct | r06_n_bets | r05_n_bets | delta_n_bets |
|---|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | 461.37 | 461.37 | +0.00000000 | 1300 | 1300 | 0 |
| m05_wealth_grw_negbin | 384.62 | 384.62 | +0.00000000 | 1235 | 1235 | 0 |
| m10_lineup_grw_negbin | 370.28 | 370.28 | +0.00000000 | 1301 | 1301 | 0 |
| m12_joint_hybrid_synergy_negbin | 297.52 | 297.52 | +0.00000000 | 1244 | 1244 | 0 |
| m00_poisson | 491.55 | 491.55 | +0.00000000 | 1296 | 1296 | 0 |
| m05_poisson | 385.78 | 385.78 | +0.00000000 | 1247 | 1247 | 0 |
| m10_poisson | 348.82 | 348.82 | +0.00000000 | 1299 | 1299 | 0 |
| m12_poisson | 351.90 | 351.90 | +0.00000000 | 1253 | 1253 | 0 |
