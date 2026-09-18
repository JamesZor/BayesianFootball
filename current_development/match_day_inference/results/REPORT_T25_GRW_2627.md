# T−25 Order Book Backtest — MultiScaleGRW vs TimeDecay & Hybrid

Scottish League One [56] and League Two [57], **2026/27 Season Opening Slates**.
Generated 2026-09-18 21:24 from `fix/t013-grw-builder-dispatch` @ 6be772d4e9da.

Evaluated on `betfair_live.order_book_1m` 25 minutes before kick-off with **£500 opening bankroll**, compounding slate by slate.

## 0. Model Arms Evaluated

| Arm | Architecture | State Dynamics | Calibration |
|---|---|---|---|
| `m05_joint_grw_raw` | Two-arm Joint (proxy xG + goals) | **MultiScaleGRW** | Raw posterior |
| `m05_joint_grw_cal_optB` | Two-arm Joint (proxy xG + goals) | **MultiScaleGRW** | Option B (`scot_lower_t25_inv`) |
| `m05_wealth_grw_raw` | Production Wealth | **MultiScaleGRW** | Raw posterior |
| `m05_wealth_grw_cal_optB` | Production Wealth | **MultiScaleGRW** | Option B (`scot_lower_t25_inv`) |
| `m05_joint_td_raw` | Two-arm Joint | Time Decay | Raw posterior |
| `m12_hybrid_td_raw` | Joint + Player RAPM Lineup | Time Decay | Raw posterior (Current Production) |
| `m12_hybrid_td_cal_optB` | Joint + Player RAPM Lineup | Time Decay | Option B (`scot_lower_t25_inv`) |

## 1. Slate Inventory

| Slate | Finished fixtures | T−25 book | Closing quotes | Status |
|---|---:|---|---:|---|
| 2026-08-01 | 10 | present | 170 | priced |
| 2026-08-08 | 10 | present | 170 | priced |
| 2026-08-15 | 9 | present | 145 | priced |
| 2026-08-22 | 10 | **absent** | 0 | **REFUSED** |
| 2026-09-05 | 10 | present | 146 | priced |

### Refused slates

* **2026-08-22** — betfair_live.order_book_1m holds no tick at or before T-25 (2026-08-22T13:35:00) for any fixture on this card; first archived tick is 2026-08-22T14:06:00+00:00. There is no tradeable book to price against, so this slate is REFUSED rather than priced off a post-kick-off ladder.

## 2. Executive Summary

| Arm | Fill model | Initial | Final | Net P&L | Total staked | ROI% | Win rate% | Sharpe (slate) | Sharpe (ann.) | Max DD% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m05_joint_grw_raw` | `touch_only` | £500.00 | £550.15 | £50.15 | £177.95 | 28.18% | 41.18% | 0.779 | 5.035 | -1.77 |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | £500.00 | £552.28 | £52.28 | £188.55 | 27.72% | 41.18% | 0.656 | 4.241 | -2.48 |
| `m05_joint_grw_cal_optB` | `touch_only` | £500.00 | £512.34 | £12.34 | £60.62 | 20.36% | 41.67% | 0.303 | 1.958 | -1.55 |
| `m05_joint_grw_cal_optB` | `ladder_sweep_v1` | £500.00 | £512.34 | £12.34 | £60.62 | 20.36% | 41.67% | 0.303 | 1.958 | -1.55 |
| `m05_wealth_grw_raw` | `touch_only` | £500.00 | £539.66 | £39.66 | £171.03 | 23.19% | 39.13% | 0.579 | 3.743 | -2.92 |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | £500.00 | £547.57 | £47.57 | £189.08 | 25.16% | 39.13% | 0.559 | 3.613 | -3.55 |
| `m05_wealth_grw_cal_optB` | `touch_only` | £500.00 | £507.46 | £7.46 | £74.88 | 9.96% | 44.83% | 0.166 | 1.074 | -2.38 |
| `m05_wealth_grw_cal_optB` | `ladder_sweep_v1` | £500.00 | £508.92 | £8.92 | £78.81 | 11.31% | 44.83% | 0.192 | 1.24 | -2.4 |
| `m05_joint_td_raw` | `touch_only` | £500.00 | £546.06 | £46.06 | £170.01 | 27.09% | 41.67% | 1.197 | 7.734 | -0.01 |
| `m05_joint_td_raw` | `ladder_sweep_v1` | £500.00 | £546.26 | £46.26 | £178.13 | 25.97% | 40.82% | 0.962 | 6.216 | -0.64 |
| `m12_hybrid_td_raw` | `touch_only` | £500.00 | £533.40 | £33.40 | £179.72 | 18.58% | 37.74% | 0.624 | 4.029 | -0.94 |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | £500.00 | £531.78 | £31.78 | £184.82 | 17.19% | 37.74% | 0.556 | 3.593 | -0.97 |
| `m12_hybrid_td_cal_optB` | `touch_only` | £500.00 | £496.61 | −£3.39 | £58.09 | -5.84% | 34.78% | -0.089 | -0.572 | -2.95 |
| `m12_hybrid_td_cal_optB` | `ladder_sweep_v1` | £500.00 | £496.61 | −£3.39 | £58.09 | -5.84% | 34.78% | -0.089 | -0.572 | -2.95 |

Initial bankroll is £500.00. Returns compound per slate. ROI is measured on filled risk.

## 3. Bankroll Trajectory (£500 Initial)

| Track | Opening | 2026-08-01 | 2026-08-08 | 2026-08-15 | 2026-09-05 |
|---|---:|---:|---:|---:|---:|
| `m05_joint_grw_raw__touch_only` | £500.00 | £524.15 | £514.86 | £539.41 | £550.15 |
| `m05_joint_grw_raw__ladder_sweep_v1` | £500.00 | £523.25 | £510.29 | £542.59 | £552.28 |
| `m05_joint_grw_cal_optB__touch_only` | £500.00 | £513.83 | £505.87 | £515.30 | £512.34 |
| `m05_joint_grw_cal_optB__ladder_sweep_v1` | £500.00 | £513.83 | £505.87 | £515.30 | £512.34 |
| `m05_wealth_grw_raw__touch_only` | £500.00 | £516.98 | £501.88 | £517.08 | £539.66 |
| `m05_wealth_grw_raw__ladder_sweep_v1` | £500.00 | £516.98 | £498.62 | £527.89 | £547.57 |
| `m05_wealth_grw_cal_optB__touch_only` | £500.00 | £514.73 | £502.48 | £507.96 | £507.46 |
| `m05_wealth_grw_cal_optB__ladder_sweep_v1` | £500.00 | £514.73 | £502.39 | £510.07 | £508.92 |
| `m05_joint_td_raw__touch_only` | £500.00 | £521.88 | £521.83 | £537.46 | £546.06 |
| `m05_joint_td_raw__ladder_sweep_v1` | £500.00 | £521.54 | £518.19 | £538.94 | £546.26 |
| `m12_hybrid_td_raw__touch_only` | £500.00 | £524.33 | £524.28 | £538.48 | £533.40 |
| `m12_hybrid_td_raw__ladder_sweep_v1` | £500.00 | £523.62 | £520.38 | £536.98 | £531.78 |
| `m12_hybrid_td_cal_optB__touch_only` | £500.00 | £511.72 | £499.75 | £498.71 | £496.61 |
| `m12_hybrid_td_cal_optB__ladder_sweep_v1` | £500.00 | £511.72 | £499.75 | £498.71 | £496.61 |

## 4. Liquidity and Capacity Audit

| Arm | Fill model | Requested stake | Filled stake | Fill rate | Legs | Legs fully unfilled | Mean levels used |
|---|---|---:|---:|---:|---:|---:|---:|
| `m05_joint_grw_raw` | `touch_only` | £199.54 | £186.02 | 93.22% | 51 | 0 | 1.0 |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | £199.32 | £196.68 | 98.68% | 51 | 0 | 1.12 |
| `m05_joint_grw_cal_optB` | `touch_only` | £65.52 | £65.52 | 100.00% | 24 | 0 | 1.0 |
| `m05_joint_grw_cal_optB` | `ladder_sweep_v1` | £65.52 | £65.52 | 100.00% | 24 | 0 | 1.0 |
| `m05_wealth_grw_raw` | `touch_only` | £198.36 | £178.40 | 89.94% | 46 | 0 | 1.0 |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | £199.24 | £196.52 | 98.63% | 46 | 0 | 1.11 |
| `m05_wealth_grw_cal_optB` | `touch_only` | £83.17 | £79.33 | 95.38% | 29 | 0 | 1.0 |
| `m05_wealth_grw_cal_optB` | `ladder_sweep_v1` | £83.27 | £83.27 | 100.00% | 29 | 0 | 1.17 |
| `m05_joint_td_raw` | `touch_only` | £188.61 | £179.16 | 94.99% | 48 | 0 | 1.0 |
| `m05_joint_td_raw` | `ladder_sweep_v1` | £189.37 | £187.28 | 98.90% | 49 | 0 | 1.08 |
| `m12_hybrid_td_raw` | `touch_only` | £197.60 | £188.74 | 95.52% | 53 | 0 | 1.0 |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | £196.93 | £193.81 | 98.41% | 53 | 0 | 1.08 |
| `m12_hybrid_td_cal_optB` | `touch_only` | £63.64 | £63.64 | 100.00% | 23 | 0 | 1.0 |
| `m12_hybrid_td_cal_optB` | `ladder_sweep_v1` | £63.64 | £63.64 | 100.00% | 23 | 0 | 1.0 |

### Slippage: LadderSweep vs TouchOnly

| Arm | Touch filled | Sweep filled | Extra stake matched | Touch P&L | Sweep P&L | Sweep − Touch | Mean sweep slippage vs touch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m05_joint_grw_cal_optB` | £65.52 | £65.52 | £0.00 | £12.34 | £12.34 | £0.00 | -0.00% |
| `m05_joint_grw_raw` | £186.02 | £196.68 | £10.66 | £50.15 | £52.28 | £2.13 | 0.04% |
| `m05_joint_td_raw` | £179.16 | £187.28 | £8.13 | £46.06 | £46.26 | £0.20 | 0.03% |
| `m05_wealth_grw_cal_optB` | £79.33 | £83.27 | £3.94 | £7.46 | £8.92 | £1.46 | 0.03% |
| `m05_wealth_grw_raw` | £178.40 | £196.52 | £18.12 | £39.66 | £47.57 | £7.91 | 0.06% |
| `m12_hybrid_td_cal_optB` | £63.64 | £63.64 | £0.00 | −£3.39 | −£3.39 | £0.00 | 0.00% |
| `m12_hybrid_td_raw` | £188.74 | £193.81 | £5.07 | £33.40 | £31.78 | −£1.62 | 0.02% |

## 5. Closing Line Value (CLV)

| Arm | Fill model | Filled legs | Legs with a close | Beat close % | Mean CLV (pp) | Median CLV (pp) |
|---|---|---:|---:|---:|---:|---:|
| `m05_joint_grw_raw` | `touch_only` | 51 | 51 | 37.25% | -0.178 | -0.509 |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | 51 | 51 | 37.25% | -0.193 | -0.509 |
| `m05_joint_grw_cal_optB` | `touch_only` | 24 | 24 | 45.83% | 0.26 | -0.117 |
| `m05_joint_grw_cal_optB` | `ladder_sweep_v1` | 24 | 24 | 45.83% | 0.26 | -0.117 |
| `m05_wealth_grw_raw` | `touch_only` | 46 | 46 | 39.13% | -0.096 | -0.523 |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | 46 | 46 | 39.13% | -0.117 | -0.523 |
| `m05_wealth_grw_cal_optB` | `touch_only` | 29 | 29 | 51.72% | 0.446 | 0.249 |
| `m05_wealth_grw_cal_optB` | `ladder_sweep_v1` | 29 | 29 | 51.72% | 0.433 | 0.249 |
| `m05_joint_td_raw` | `touch_only` | 48 | 48 | 35.42% | -0.38 | -0.592 |
| `m05_joint_td_raw` | `ladder_sweep_v1` | 49 | 49 | 34.69% | -0.413 | -0.648 |
| `m12_hybrid_td_raw` | `touch_only` | 53 | 52 | 36.54% | -0.202 | -0.522 |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | 53 | 52 | 36.54% | -0.208 | -0.522 |
| `m12_hybrid_td_cal_optB` | `touch_only` | 23 | 22 | 45.45% | 0.125 | -0.117 |
| `m12_hybrid_td_cal_optB` | `ladder_sweep_v1` | 23 | 22 | 45.45% | 0.125 | -0.117 |

## 6. Where the Money Came From (`TouchOnly`)

| Arm | Market | Selection | Legs | Filled risk | Net P&L | ROI% | Win rate% |
|---|---|---|---:|---:|---:|---:|---:|
| `m05_joint_grw_cal_optB` | 1X2 | away | 5 | £11.14 | −£11.14 | -100.00% | 0.00% |
| `m05_joint_grw_cal_optB` | 1X2 | draw | 2 | £4.01 | £2.09 | 52.13% | 50.00% |
| `m05_joint_grw_cal_optB` | 1X2 | home | 10 | £32.06 | £25.52 | 79.61% | 60.00% |
| `m05_joint_grw_cal_optB` | OverUnder 2.5 | under_25 | 7 | £13.41 | −£4.13 | -30.78% | 42.86% |
| `m05_joint_grw_raw` | 1X2 | away | 11 | £34.07 | −£8.36 | -24.53% | 27.27% |
| `m05_joint_grw_raw` | 1X2 | draw | 7 | £14.05 | £1.48 | 10.53% | 28.57% |
| `m05_joint_grw_raw` | 1X2 | home | 17 | £90.70 | £59.99 | 66.14% | 47.06% |
| `m05_joint_grw_raw` | OverUnder 1.5 | over_15 | 2 | £3.84 | −£2.43 | -63.27% | 50.00% |
| `m05_joint_grw_raw` | OverUnder 2.5 | under_25 | 14 | £35.29 | −£0.53 | -1.51% | 50.00% |
| `m05_joint_td_raw` | 1X2 | away | 10 | £32.86 | −£12.42 | -37.80% | 20.00% |
| `m05_joint_td_raw` | 1X2 | draw | 5 | £12.09 | £5.48 | 45.36% | 40.00% |
| `m05_joint_td_raw` | 1X2 | home | 17 | £89.34 | £54.68 | 61.21% | 47.06% |
| `m05_joint_td_raw` | OverUnder 1.5 | over_15 | 1 | £2.32 | −£2.32 | -100.00% | 0.00% |
| `m05_joint_td_raw` | OverUnder 2.5 | under_25 | 15 | £33.40 | £0.63 | 1.90% | 53.33% |
| `m05_wealth_grw_cal_optB` | 1X2 | away | 6 | £15.18 | −£11.77 | -77.51% | 16.67% |
| `m05_wealth_grw_cal_optB` | 1X2 | draw | 3 | £5.80 | £0.83 | 14.25% | 33.33% |
| `m05_wealth_grw_cal_optB` | 1X2 | home | 13 | £40.69 | £20.93 | 51.43% | 53.85% |
| `m05_wealth_grw_cal_optB` | OverUnder 2.5 | under_25 | 7 | £13.21 | −£2.53 | -19.12% | 57.14% |
| `m05_wealth_grw_raw` | 1X2 | away | 11 | £34.34 | −£19.49 | -56.75% | 18.18% |
| `m05_wealth_grw_raw` | 1X2 | draw | 6 | £13.97 | £2.96 | 21.20% | 33.33% |
| `m05_wealth_grw_raw` | 1X2 | home | 15 | £93.49 | £59.49 | 63.63% | 53.33% |
| `m05_wealth_grw_raw` | OverUnder 1.5 | over_15 | 1 | £1.36 | £0.36 | 26.46% | 100.00% |
| `m05_wealth_grw_raw` | OverUnder 2.5 | under_25 | 13 | £27.87 | −£3.66 | -13.15% | 38.46% |
| `m12_hybrid_td_cal_optB` | 1X2 | away | 5 | £13.49 | −£13.49 | -100.00% | 0.00% |
| `m12_hybrid_td_cal_optB` | 1X2 | draw | 3 | £5.67 | £2.84 | 49.98% | 33.33% |
| `m12_hybrid_td_cal_optB` | 1X2 | home | 10 | £28.24 | £14.88 | 52.70% | 60.00% |
| `m12_hybrid_td_cal_optB` | OverUnder 2.5 | under_25 | 5 | £10.68 | −£7.62 | -71.36% | 20.00% |
| `m12_hybrid_td_raw` | 1X2 | away | 12 | £36.98 | −£12.73 | -34.42% | 25.00% |
| `m12_hybrid_td_raw` | 1X2 | draw | 7 | £15.40 | £5.77 | 37.50% | 28.57% |
| `m12_hybrid_td_raw` | 1X2 | home | 19 | £95.43 | £49.19 | 51.54% | 52.63% |
| `m12_hybrid_td_raw` | OverUnder 1.5 | over_15 | 1 | £1.03 | −£1.03 | -100.00% | 0.00% |
| `m12_hybrid_td_raw` | OverUnder 2.5 | under_25 | 14 | £30.88 | −£7.81 | -25.27% | 35.71% |

## 7. Artifacts

| File | Rows |
|---|---:|
| `r13_grw_trades_touch_only.csv` | 274 |
| `r13_grw_trades_ladder_sweep.csv` | 275 |
| `r13_grw_slate_trajectory.csv` | 56 |
