# r06 T−25 order-book backtest — Task 013

Scottish League One [56] + League Two [57], 2026/27 opening slates. Generated 2026-09-11 23:48 at `037d651c`. £500 opening bankroll, compounding per slate; `betfair_live.order_book_1m` at T−25; Option B staking.

## Slates

| Slate | Finished fixtures | T−25 book | Status |
|---|---:|---|---|
| 2026-08-01 | 10 | present | priced |
| 2026-08-08 | 10 | present | priced |
| 2026-08-15 | 9 | present | priced |
| 2026-08-22 | 10 | **absent** | **REFUSED** — betfair_live.order_book_1m holds no tick at or before T-25 (2026-08-22T13:35:00) for any fixture on this card; first archived tick is 2026-08-22T14:06:00+00:00. There is no tradeable book to price against, so this slate is REFUSED rather than priced off a post-kick-off ladder. |
| 2026-09-05 | 10 | present | priced |

## Executive summary

| Arm | Fill model | Initial | Final | Net P&L | Total staked | ROI% | Win rate% | Sharpe (slate) | Sharpe (ann.) | Max DD% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m12_hybrid_grw_raw` | `touch_only` | £500.00 | £557.71 | £57.71 | £344.72 | 16.74% | 37.10% | 2.582 | 16.684 | 0.0 |
| `m12_hybrid_grw_raw` | `ladder_sweep_v1` | £500.00 | £595.53 | £95.53 | £434.50 | 21.99% | 37.10% | 1.425 | 9.204 | 0.0 |
| `m12_hybrid_grw_cal_optB` | `touch_only` | £500.00 | £506.77 | £6.77 | £174.37 | 3.88% | 34.15% | 0.098 | 0.633 | -3.63 |
| `m12_hybrid_grw_cal_optB` | `ladder_sweep_v1` | £500.00 | £491.12 | −£8.88 | £210.71 | -4.21% | 34.15% | -0.096 | -0.623 | -7.21 |
| `m05_wealth_grw_raw` | `touch_only` | £500.00 | £588.43 | £88.43 | £351.52 | 25.16% | 42.19% | 1.23 | 7.948 | -0.21 |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | £500.00 | £600.33 | £100.33 | £430.27 | 23.32% | 42.19% | 0.766 | 4.952 | -1.9 |
| `m10_lineup_grw_raw` | `touch_only` | £500.00 | £518.58 | £18.58 | £341.39 | 5.44% | 40.98% | 0.181 | 1.168 | -5.88 |
| `m10_lineup_grw_raw` | `ladder_sweep_v1` | £500.00 | £529.02 | £29.02 | £415.65 | 6.98% | 40.32% | 0.181 | 1.173 | -9.1 |
| `m00_baseline_grw_raw` | `touch_only` | £500.00 | £536.85 | £36.85 | £355.65 | 10.36% | 42.86% | 0.26 | 1.682 | -7.41 |
| `m00_baseline_grw_raw` | `ladder_sweep_v1` | £500.00 | £538.44 | £38.44 | £425.27 | 9.04% | 42.86% | 0.19 | 1.225 | -10.82 |
| `m05_joint_grw_raw` | `touch_only` | £500.00 | £590.05 | £90.05 | £351.88 | 25.59% | 42.19% | 1.247 | 8.057 | -0.13 |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | £500.00 | £600.22 | £100.22 | £431.50 | 23.23% | 41.54% | 0.757 | 4.89 | -1.88 |
| `m12_hybrid_td_raw` | `touch_only` | £500.00 | £551.97 | £51.97 | £379.60 | 13.69% | 36.92% | 0.65 | 4.199 | -2.74 |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | £500.00 | £575.94 | £75.94 | £448.15 | 16.94% | 36.36% | 0.662 | 4.274 | -3.12 |
| `m05_joint_td_raw` | `touch_only` | £500.00 | £569.45 | £69.45 | £358.09 | 19.39% | 34.92% | 1.545 | 9.983 | 0.0 |
| `m05_joint_td_raw` | `ladder_sweep_v1` | £500.00 | £587.76 | £87.76 | £433.50 | 20.24% | 34.92% | 1.016 | 6.566 | 0.0 |

| Arm | Fill model | Capture ratio | Reproduces published |
|---|---|---:|---|
| `m12_hybrid_grw_raw` | `touch_only` | 1.217 | — |
| `m12_hybrid_grw_raw` | `ladder_sweep_v1` | 1.217 | — |
| `m12_hybrid_grw_cal_optB` | `touch_only` | 0.901 | — |
| `m12_hybrid_grw_cal_optB` | `ladder_sweep_v1` | 0.901 | — |
| `m05_wealth_grw_raw` | `touch_only` | 1.191 | — |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | 1.191 | — |
| `m10_lineup_grw_raw` | `touch_only` | 1.161 | — |
| `m10_lineup_grw_raw` | `ladder_sweep_v1` | 1.193 | — |
| `m00_baseline_grw_raw` | `touch_only` | 1.236 | — |
| `m00_baseline_grw_raw` | `ladder_sweep_v1` | 1.236 | — |
| `m05_joint_grw_raw` | `touch_only` | 1.191 | true |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | 1.225 | true |
| `m12_hybrid_td_raw` | `touch_only` | 1.048 | true |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | 1.065 | true |
| `m05_joint_td_raw` | `touch_only` | 1.295 | true |
| `m05_joint_td_raw` | `ladder_sweep_v1` | 1.295 | true |

## Bankroll trajectory

| Track | Opening | 2026-08-01 | 2026-08-08 | 2026-08-15 | 2026-09-05 |
|---|---:|---:|---:|---:|---:|
| `m12_hybrid_grw_raw__touch_only` | £500.00 | £520.15 | £527.86 | £544.61 | £557.71 |
| `m12_hybrid_grw_raw__ladder_sweep_v1` | £500.00 | £530.90 | £538.95 | £583.10 | £595.53 |
| `m12_hybrid_grw_cal_optB__touch_only` | £500.00 | £525.87 | £511.36 | £514.31 | £506.77 |
| `m12_hybrid_grw_cal_optB__ladder_sweep_v1` | £500.00 | £529.29 | £502.76 | £503.98 | £491.12 |
| `m05_wealth_grw_raw__touch_only` | £500.00 | £523.04 | £521.94 | £564.59 | £588.43 |
| `m05_wealth_grw_raw__ladder_sweep_v1` | £500.00 | £531.21 | £521.13 | £587.98 | £600.33 |
| `m10_lineup_grw_raw__touch_only` | £500.00 | £496.68 | £470.62 | £487.97 | £518.58 |
| `m10_lineup_grw_raw__ladder_sweep_v1` | £500.00 | £508.98 | £462.67 | £502.80 | £529.02 |
| `m00_baseline_grw_raw__touch_only` | £500.00 | £490.77 | £462.95 | £490.48 | £536.85 |
| `m00_baseline_grw_raw__ladder_sweep_v1` | £500.00 | £504.08 | £449.55 | £500.24 | £538.44 |
| `m05_joint_grw_raw__touch_only` | £500.00 | £523.35 | £522.65 | £565.96 | £590.05 |
| `m05_joint_grw_raw__ladder_sweep_v1` | £500.00 | £531.09 | £521.10 | £588.75 | £600.22 |
| `m12_hybrid_td_raw__touch_only` | £500.00 | £518.16 | £532.97 | £567.52 | £551.97 |
| `m12_hybrid_td_raw__ladder_sweep_v1` | £500.00 | £530.97 | £542.11 | £594.50 | £575.94 |
| `m05_joint_td_raw__touch_only` | £500.00 | £508.55 | £530.41 | £561.87 | £569.45 |
| `m05_joint_td_raw__ladder_sweep_v1` | £500.00 | £519.65 | £532.21 | £585.84 | £587.76 |

## Liquidity

| Arm | Fill model | Requested stake | Filled stake | Fill rate | Legs | Legs fully unfilled | Mean levels used |
|---|---|---:|---:|---:|---:|---:|---:|
| `m12_hybrid_grw_raw` | `touch_only` | £511.92 | £356.69 | 69.68% | 62 | 0 | 1.0 |
| `m12_hybrid_grw_raw` | `ladder_sweep_v1` | £527.13 | £450.41 | 85.44% | 62 | 0 | 1.39 |
| `m12_hybrid_grw_cal_optB` | `touch_only` | £259.71 | £183.68 | 70.73% | 41 | 0 | 1.0 |
| `m12_hybrid_grw_cal_optB` | `ladder_sweep_v1` | £256.99 | £226.65 | 88.19% | 41 | 0 | 1.27 |
| `m05_wealth_grw_raw` | `touch_only` | £503.88 | £364.54 | 72.35% | 64 | 0 | 1.0 |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | £511.43 | £447.98 | 87.59% | 64 | 0 | 1.34 |
| `m10_lineup_grw_raw` | `touch_only` | £476.98 | £351.52 | 73.70% | 61 | 0 | 1.0 |
| `m10_lineup_grw_raw` | `ladder_sweep_v1` | £482.57 | £427.33 | 88.55% | 62 | 0 | 1.29 |
| `m00_baseline_grw_raw` | `touch_only` | £477.91 | £362.32 | 75.81% | 63 | 0 | 1.0 |
| `m00_baseline_grw_raw` | `ladder_sweep_v1` | £480.03 | £432.88 | 90.18% | 63 | 0 | 1.22 |
| `m05_joint_grw_raw` | `touch_only` | £505.17 | £364.75 | 72.20% | 64 | 0 | 1.0 |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | £513.29 | £449.08 | 87.49% | 65 | 0 | 1.35 |
| `m12_hybrid_td_raw` | `touch_only` | £526.16 | £396.90 | 75.43% | 65 | 0 | 1.0 |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | £539.65 | £472.45 | 87.55% | 66 | 0 | 1.26 |
| `m05_joint_td_raw` | `touch_only` | £505.85 | £373.97 | 73.93% | 63 | 0 | 1.0 |
| `m05_joint_td_raw` | `ladder_sweep_v1` | £515.21 | £456.63 | 88.63% | 63 | 0 | 1.33 |

### LadderSweep vs TouchOnly

| Arm | Touch filled | Sweep filled | Extra stake matched | Touch P&L | Sweep P&L | Sweep − Touch | Mean sweep slippage vs touch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw_raw` | £362.32 | £432.88 | £70.56 | £36.85 | £38.44 | £1.59 | 0.13% |
| `m05_joint_grw_raw` | £364.75 | £449.08 | £84.33 | £90.05 | £100.22 | £10.17 | 0.16% |
| `m05_joint_td_raw` | £373.97 | £456.63 | £82.67 | £69.45 | £87.76 | £18.31 | 0.15% |
| `m05_wealth_grw_raw` | £364.54 | £447.98 | £83.44 | £88.43 | £100.33 | £11.90 | 0.16% |
| `m10_lineup_grw_raw` | £351.52 | £427.33 | £75.81 | £18.58 | £29.02 | £10.45 | 0.16% |
| `m12_hybrid_grw_cal_optB` | £183.68 | £226.65 | £42.97 | £6.77 | −£8.88 | −£15.65 | 0.17% |
| `m12_hybrid_grw_raw` | £356.69 | £450.41 | £93.72 | £57.71 | £95.53 | £37.82 | 0.20% |
| `m12_hybrid_td_raw` | £396.90 | £472.45 | £75.55 | £51.97 | £75.94 | £23.97 | 0.14% |

## Closing-line value

| Arm | Fill model | Filled legs | Legs with a close | Beat close % | Mean CLV (pp) | Median CLV (pp) |
|---|---|---:|---:|---:|---:|---:|
| `m12_hybrid_grw_raw` | `touch_only` | 62 | 62 | 33.87% | -0.302 | -0.592 |
| `m12_hybrid_grw_raw` | `ladder_sweep_v1` | 62 | 62 | 33.87% | -0.373 | -0.616 |
| `m12_hybrid_grw_cal_optB` | `touch_only` | 41 | 41 | 36.59% | -0.206 | -0.648 |
| `m12_hybrid_grw_cal_optB` | `ladder_sweep_v1` | 41 | 41 | 36.59% | -0.272 | -0.648 |
| `m05_wealth_grw_raw` | `touch_only` | 64 | 64 | 31.25% | -0.495 | -0.55 |
| `m05_wealth_grw_raw` | `ladder_sweep_v1` | 64 | 64 | 31.25% | -0.557 | -0.615 |
| `m10_lineup_grw_raw` | `touch_only` | 61 | 61 | 34.43% | -0.323 | -0.51 |
| `m10_lineup_grw_raw` | `ladder_sweep_v1` | 62 | 62 | 33.87% | -0.396 | -0.55 |
| `m00_baseline_grw_raw` | `touch_only` | 63 | 63 | 33.33% | -0.442 | -0.563 |
| `m00_baseline_grw_raw` | `ladder_sweep_v1` | 63 | 63 | 33.33% | -0.492 | -0.65 |
| `m05_joint_grw_raw` | `touch_only` | 64 | 64 | 31.25% | -0.495 | -0.55 |
| `m05_joint_grw_raw` | `ladder_sweep_v1` | 65 | 65 | 30.77% | -0.572 | -0.616 |
| `m12_hybrid_td_raw` | `touch_only` | 65 | 64 | 34.38% | -0.322 | -0.55 |
| `m12_hybrid_td_raw` | `ladder_sweep_v1` | 66 | 65 | 33.85% | -0.378 | -0.611 |
| `m05_joint_td_raw` | `touch_only` | 63 | 63 | 30.16% | -0.52 | -0.682 |
| `m05_joint_td_raw` | `ladder_sweep_v1` | 63 | 63 | 30.16% | -0.574 | -0.682 |

## Where the money came from (TouchOnly)

| Arm | Market | Selection | Legs | Filled risk | Net P&L | ROI% | Win rate% |
|---|---|---|---:|---:|---:|---:|---:|
| `m00_baseline_grw_raw` | 1X2 | away | 16 | £68.26 | −£13.93 | -20.41% | 37.50% |
| `m00_baseline_grw_raw` | 1X2 | draw | 11 | £42.53 | −£12.86 | -30.24% | 18.18% |
| `m00_baseline_grw_raw` | 1X2 | home | 17 | £172.28 | £58.38 | 33.88% | 47.06% |
| `m00_baseline_grw_raw` | OverUnder 1.5 | over_15 | 3 | £12.00 | £3.58 | 29.81% | 100.00% |
| `m00_baseline_grw_raw` | OverUnder 2.5 | under_25 | 16 | £60.58 | £1.70 | 2.80% | 50.00% |
| `m05_joint_grw_raw` | 1X2 | away | 16 | £63.84 | £2.24 | 3.50% | 31.25% |
| `m05_joint_grw_raw` | 1X2 | draw | 11 | £43.02 | −£8.72 | -20.28% | 27.27% |
| `m05_joint_grw_raw` | 1X2 | home | 18 | £171.87 | £96.38 | 56.08% | 50.00% |
| `m05_joint_grw_raw` | OverUnder 1.5 | over_15 | 3 | £13.38 | −£7.67 | -57.32% | 66.67% |
| `m05_joint_grw_raw` | OverUnder 2.5 | under_25 | 16 | £59.77 | £7.83 | 13.09% | 50.00% |
| `m05_joint_td_raw` | 1X2 | away | 13 | £60.54 | −£11.85 | -19.57% | 15.38% |
| `m05_joint_td_raw` | 1X2 | draw | 12 | £45.00 | −£2.72 | -6.04% | 25.00% |
| `m05_joint_td_raw` | 1X2 | home | 18 | £174.48 | £88.20 | 50.55% | 44.44% |
| `m05_joint_td_raw` | OverUnder 1.5 | over_15 | 2 | £8.55 | −£6.93 | -81.00% | 50.00% |
| `m05_joint_td_raw` | OverUnder 2.5 | under_25 | 18 | £69.52 | £2.74 | 3.95% | 44.44% |
| `m05_wealth_grw_raw` | 1X2 | away | 16 | £63.49 | £1.89 | 2.98% | 31.25% |
| `m05_wealth_grw_raw` | 1X2 | draw | 11 | £42.83 | −£8.77 | -20.48% | 27.27% |
| `m05_wealth_grw_raw` | 1X2 | home | 18 | £171.54 | £95.12 | 55.45% | 50.00% |
| `m05_wealth_grw_raw` | OverUnder 1.5 | over_15 | 3 | £13.17 | −£7.66 | -58.18% | 66.67% |
| `m05_wealth_grw_raw` | OverUnder 2.5 | under_25 | 16 | £60.49 | £7.85 | 12.98% | 50.00% |
| `m10_lineup_grw_raw` | 1X2 | away | 13 | £57.61 | −£14.27 | -24.77% | 30.77% |
| `m10_lineup_grw_raw` | 1X2 | draw | 11 | £41.06 | −£16.63 | -40.50% | 18.18% |
| `m10_lineup_grw_raw` | 1X2 | home | 18 | £172.08 | £46.87 | 27.24% | 50.00% |
| `m10_lineup_grw_raw` | OverUnder 1.5 | over_15 | 2 | £6.42 | £2.11 | 32.94% | 100.00% |
| `m10_lineup_grw_raw` | OverUnder 2.5 | under_25 | 17 | £64.22 | £0.50 | 0.77% | 47.06% |
| `m12_hybrid_grw_cal_optB` | 1X2 | away | 12 | £39.18 | −£15.92 | -40.63% | 16.67% |
| `m12_hybrid_grw_cal_optB` | 1X2 | draw | 5 | £21.23 | −£7.94 | -37.41% | 20.00% |
| `m12_hybrid_grw_cal_optB` | 1X2 | home | 14 | £82.77 | £33.48 | 40.45% | 50.00% |
| `m12_hybrid_grw_cal_optB` | OverUnder 2.5 | under_25 | 10 | £31.18 | −£2.84 | -9.11% | 40.00% |
| `m12_hybrid_grw_raw` | 1X2 | away | 12 | £62.63 | −£11.42 | -18.23% | 16.67% |
| `m12_hybrid_grw_raw` | 1X2 | draw | 12 | £46.85 | −£7.66 | -16.34% | 25.00% |
| `m12_hybrid_grw_raw` | 1X2 | home | 19 | £166.11 | £80.25 | 48.31% | 52.63% |
| `m12_hybrid_grw_raw` | OverUnder 1.5 | over_15 | 2 | £5.52 | −£1.62 | -29.34% | 50.00% |
| `m12_hybrid_grw_raw` | OverUnder 2.5 | under_25 | 17 | £63.59 | −£1.85 | -2.91% | 41.18% |
| `m12_hybrid_td_raw` | 1X2 | away | 13 | £68.50 | −£16.45 | -24.01% | 23.08% |
| `m12_hybrid_td_raw` | 1X2 | draw | 12 | £48.15 | £2.16 | 4.49% | 25.00% |
| `m12_hybrid_td_raw` | 1X2 | home | 20 | £185.66 | £78.40 | 42.23% | 50.00% |
| `m12_hybrid_td_raw` | OverUnder 1.5 | over_15 | 1 | £3.10 | −£3.10 | -100.00% | 0.00% |
| `m12_hybrid_td_raw` | OverUnder 2.5 | under_25 | 19 | £74.19 | −£9.06 | -12.21% | 42.11% |
