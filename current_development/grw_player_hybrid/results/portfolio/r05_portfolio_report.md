# r05 closing-line portfolio and attribution — Task 013

Generated 2026-09-11 22:54. Contract: `MatchDay.option_b_system()` (canonical markets, DeArb, FractionalKelly 0.30, 2% commission; TieredTrust Home/U2.5 = 1, Draw/Away/O1.5 = 1/1.4; SlateDrawdown 8; FixedCap 0.25; DailySlate). Prices: de-vigged Betfair TWA(−20, 0] close. Panel: 632 fixtures buildable by every arm (of 635 quoted). Bootstrap B = 4000.

## Headline

| model | dynamics | n_bets | total_return_pct | roi_pct | growth_lo | growth_hi | sharpe_ann | calmar | max_drawdown_pct | win_rate_pct | cap_weighted_win_rate_pct | capture_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m12_hybrid_td_raw | TimeDecay(180) | 1302 | 606.54 | 14.15 | -0.00009 | +0.03933 | 1.487 | 14.32 | -42.36 | 33.95 | 34.05 | 0.949 |
| m05_joint_td_raw | TimeDecay(180) | 1280 | 495.50 | 13.57 | -0.00146 | +0.03696 | 1.388 | 12.31 | -40.27 | 33.83 | 33.68 | 0.945 |
| m00_baseline_grw | MultiScaleGRW | 1296 | 491.55 | 12.32 | -0.00076 | +0.03631 | 1.412 | 12.65 | -38.85 | 34.80 | 36.22 | 1.036 |
| m05_joint_grw_raw | MultiScaleGRW | 1235 | 404.34 | 11.87 | +0.00024 | +0.03278 | 1.498 | 9.45 | -42.80 | 35.14 | 37.96 | 1.094 |
| m05_wealth_grw | MultiScaleGRW | 1247 | 385.78 | 11.68 | -0.00037 | +0.03233 | 1.453 | 9.04 | -42.67 | 35.20 | 37.87 | 1.080 |
| m12_joint_hybrid_synergy_grw | MultiScaleGRW | 1253 | 351.90 | 11.36 | -0.00192 | +0.03241 | 1.309 | 6.69 | -52.61 | 34.40 | 36.31 | 1.043 |
| m10_lineup_grw | MultiScaleGRW | 1299 | 348.82 | 10.75 | -0.00318 | +0.03352 | 1.214 | 7.27 | -48.01 | 34.95 | 34.37 | 0.957 |

## Shared-bet sizing and disjoint sets

`sizing_delta_pnl = Σ (s_a − s_b) · settle` over shared bets, in bankroll fractions; `roi_when_a_larger` is the return on the extra stake where model A sized up.

| pair | n_shared | n_only_a | n_only_b | overlap_pct | stake_mean_a | stake_mean_b | n_a_larger | roi_when_a_larger_pct | roi_when_b_larger_pct | sizing_delta_pnl | shared_roi_a_pct | shared_roi_b_pct | capture_ratio_a | capture_ratio_b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m12_joint_hybrid_synergy_grw vs m12_hybrid_td_raw | 1062 | 191 | 240 | 71.13 | 0.01456 | 0.01534 | 488 | 0.36 | 1.11 | -0.0318 | 13.41 | 12.92 | 1.043 | 0.949 |
| m05_wealth_grw vs m05_joint_td_raw | 1061 | 186 | 219 | 72.37 | 0.01438 | 0.01488 | 470 | 8.60 | 9.34 | -0.0728 | 12.44 | 12.48 | 1.080 | 0.945 |
| m12_joint_hybrid_synergy_grw vs m05_wealth_grw | 1136 | 117 | 111 | 83.28 | 0.01449 | 0.01428 | 611 | 5.77 | 5.99 | +0.0087 | 11.98 | 12.10 | 1.043 | 1.080 |
| m10_lineup_grw vs m00_baseline_grw | 1179 | 120 | 117 | 83.26 | 0.01515 | 0.01519 | 596 | 1.42 | 20.17 | -0.4631 | 10.75 | 13.31 | 0.957 | 1.036 |
| m12_joint_hybrid_synergy_grw vs m05_joint_grw_raw | 1130 | 123 | 105 | 83.21 | 0.01454 | 0.01436 | 606 | 4.73 | 6.90 | -0.0396 | 11.91 | 12.30 | 1.043 | 1.094 |

## Partition detail

| pair | bet_set | owner | n_bets | win_rate_pct | cap_weighted_win_rate_pct | roi_pct | stake_mean | odds_mean | edge_win_pp | edge_loss_pp | capture_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m12_joint_hybrid_synergy_grw vs m12_hybrid_td_raw | shared | m12_joint_hybrid_synergy_grw | 1062 | 32.77 | 35.53 | 13.41 | 0.01456 | 3.77 | 5.05 | 4.65 | 1.087 |
| m12_joint_hybrid_synergy_grw vs m12_hybrid_td_raw | shared | m12_hybrid_td_raw | 1062 | 32.77 | 32.94 | 12.92 | 0.01534 | 3.77 | 5.24 | 5.51 | 0.952 |
| m12_joint_hybrid_synergy_grw vs m12_hybrid_td_raw | exclusive | m12_joint_hybrid_synergy_grw | 191 | 43.46 | 45.15 | -11.88 | 0.00712 | 2.44 | 1.70 | 1.44 | 1.186 |
| m12_joint_hybrid_synergy_grw vs m12_hybrid_td_raw | exclusive | m12_hybrid_td_raw | 240 | 39.17 | 48.01 | 29.67 | 0.00538 | 3.11 | 1.42 | 0.55 | 2.574 |
| m05_wealth_grw vs m05_joint_td_raw | shared | m05_wealth_grw | 1061 | 32.89 | 36.34 | 12.44 | 0.01438 | 3.74 | 4.97 | 4.41 | 1.127 |
| m05_wealth_grw vs m05_joint_td_raw | shared | m05_joint_td_raw | 1061 | 32.89 | 32.73 | 12.48 | 0.01488 | 3.74 | 4.97 | 5.21 | 0.954 |
| m05_wealth_grw vs m05_joint_td_raw | exclusive | m05_wealth_grw | 186 | 48.39 | 54.77 | 3.20 | 0.00740 | 2.36 | 1.85 | 1.05 | 1.767 |
| m05_wealth_grw vs m05_joint_td_raw | exclusive | m05_joint_td_raw | 219 | 38.36 | 47.21 | 29.04 | 0.00508 | 3.20 | 1.33 | 0.83 | 1.614 |
| m12_joint_hybrid_synergy_grw vs m05_wealth_grw | shared | m12_joint_hybrid_synergy_grw | 1136 | 34.68 | 36.42 | 11.98 | 0.01449 | 3.62 | 4.79 | 4.65 | 1.029 |
| m12_joint_hybrid_synergy_grw vs m05_wealth_grw | shared | m05_wealth_grw | 1136 | 34.68 | 37.75 | 12.10 | 0.01428 | 3.62 | 4.76 | 4.35 | 1.093 |
| m12_joint_hybrid_synergy_grw vs m05_wealth_grw | exclusive | m12_joint_hybrid_synergy_grw | 117 | 31.62 | 31.14 | -17.02 | 0.00306 | 3.08 | 0.36 | 0.24 | 1.524 |
| m12_joint_hybrid_synergy_grw vs m05_wealth_grw | exclusive | m05_wealth_grw | 111 | 40.54 | 42.49 | -5.43 | 0.00365 | 2.71 | 0.58 | 0.16 | 3.598 |
| m10_lineup_grw vs m00_baseline_grw | shared | m10_lineup_grw | 1179 | 34.69 | 34.31 | 10.75 | 0.01515 | 3.64 | 5.19 | 5.45 | 0.952 |
| m10_lineup_grw vs m00_baseline_grw | shared | m00_baseline_grw | 1179 | 34.69 | 36.26 | 13.31 | 0.01519 | 3.64 | 5.29 | 5.13 | 1.030 |
| m10_lineup_grw vs m00_baseline_grw | exclusive | m10_lineup_grw | 120 | 37.50 | 37.19 | 10.58 | 0.00336 | 3.09 | 0.69 | -0.08 | n/a |
| m10_lineup_grw vs m00_baseline_grw | exclusive | m00_baseline_grw | 117 | 35.90 | 34.62 | -24.88 | 0.00408 | 2.72 | 1.02 | 0.48 | 2.118 |
| m12_joint_hybrid_synergy_grw vs m05_joint_grw_raw | shared | m12_joint_hybrid_synergy_grw | 1130 | 34.51 | 36.40 | 11.91 | 0.01454 | 3.62 | 4.84 | 4.64 | 1.044 |
| m12_joint_hybrid_synergy_grw vs m05_joint_grw_raw | shared | m05_joint_grw_raw | 1130 | 34.51 | 37.84 | 12.30 | 0.01436 | 3.62 | 4.84 | 4.34 | 1.114 |
| m12_joint_hybrid_synergy_grw vs m05_joint_grw_raw | exclusive | m12_joint_hybrid_synergy_grw | 123 | 33.33 | 32.44 | -11.44 | 0.00317 | 3.10 | 0.24 | 0.47 | 0.512 |
| m12_joint_hybrid_synergy_grw vs m05_joint_grw_raw | exclusive | m05_joint_grw_raw | 105 | 41.90 | 42.64 | -5.19 | 0.00392 | 2.73 | 0.71 | 0.35 | 2.020 |

## Persisted portfolios

| model | model_run_id | portfolio_run_id |
|---|---|---|
| m00_baseline_grw | 158d2a80-7ea3-4d6c-b3ab-be62bcf1bc11 | ec5bf23e-c500-4253-a7fb-9e952d732eba |
| m05_wealth_grw | b0961bc4-c40c-4dbe-9c05-57df7ae0839e | 3982ffb9-1d1c-482c-92d0-995b6bca6d8e |
| m10_lineup_grw | b13c8fb9-ce34-4210-aa3f-9d2ed493c286 | f988117b-78cc-46fa-bc30-913659bafdf8 |
| m12_joint_hybrid_synergy_grw | 3a9a4c7e-378b-45d0-a2d2-c8b69b46786b | c2aedffe-6e77-4f29-a79b-b8c688ae33bc |
