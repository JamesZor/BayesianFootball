# Task 007 follow-up — portfolio backtest and Layer 2 calibration

Generated 2026-09-11T13:15:22.991

## 1. What was run

Seven persisted posteriors, repriced with no new MCMC, on one frozen panel of 622 Betfair-quoted fixtures spanning 630 days.

| Contract element | Value |
|---|---|
| Book | de-vigged Betfair TWA[-20,0] |
| Markets | 1X2 + O/U 2.5 |
| Shrinkage | Baker-McHale |
| Commission | 2% per bet |
| Policy | FlatTrust(0.30), SlateDrawdown(23.0), FixedCap(0.20), DailySlate |
| Bootstrap | 4000 resamples, match-clustered for ROI, slate-blocked for g |
| Panel | 622 fixtures, identical for every arm |

**Every arm stakes the same book.** The calibrated variants differ from the raw ones only in the posterior they carry; the prices, markets, commission, shrinkage, policy and fixture panel are byte-identical. A P&L difference is therefore attributable to the posterior and not to the book.

## 2. Raw posteriors — headline

| Model | Family | Bets | Return % | CAGR % | g / slate | g 95% CI | Sharpe ann | Calmar | MDD % | Win % | Edge pp |
|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| `m05_production_wealth_grw` | GRW | 1298 | +174.88 | +79.7 | +0.01021 | [+0.0015, +0.0193] | 1.715 | 8.86 | -19.7 | 34.9 | +5.02 |
| `m00_baseline_grw` | GRW | 1324 | +153.84 | +71.6 | +0.00941 | [+0.0008, +0.0178] | 1.636 | 8.41 | -18.3 | 35.4 | +5.03 |
| `m12_joint_hybrid_synergy` | Production | 1328 | +129.32 | +61.8 | +0.00838 | [-0.0012, +0.0176] | 1.355 | 6.59 | -19.6 | 34.1 | +4.67 |
| `m05_joint_production_wealth_grw` | GRW | 1263 | +125.06 | +60.0 | +0.00819 | [+0.0010, +0.0157] | 1.647 | 6.25 | -20.0 | 34.2 | +4.50 |
| `m05_production_wealth` | TimeDecay | 1298 | +124.25 | +59.7 | +0.00816 | [-0.0011, +0.0176] | 1.305 | 5.91 | -21.0 | 33.0 | +4.87 |
| `m05_joint_production_wealth` | TimeDecay | 1301 | +117.34 | +56.8 | +0.00784 | [-0.0005, +0.0165] | 1.395 | 6.30 | -18.6 | 33.8 | +4.55 |
| `m00_baseline` | TimeDecay | 1301 | +115.48 | +56.1 | +0.00775 | [-0.0018, +0.0172] | 1.215 | 5.47 | -21.1 | 32.4 | +4.98 |

### 2.1 GRW versus its matched control

| GRW candidate | TimeDecay control | Δ Return pp | Δ CAGR pp | Δ g / slate | Δ Sharpe | Δ Calmar | Δ MDD pp |
|---|---|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | `m00_baseline` | +38.36 | +15.6 | +0.00166 | +0.420 | +2.94 | +2.8 |
| `m05_production_wealth_grw` | `m05_production_wealth` | +50.63 | +20.0 | +0.00206 | +0.410 | +2.95 | +1.3 |
| `m05_joint_production_wealth_grw` | `m05_joint_production_wealth` | +7.72 | +3.2 | +0.00035 | +0.252 | -0.05 | -1.4 |

## 3. Compounding growth and the hurdle model

`g` is mean per-slate log growth — the quantity Kelly maximises — with a 95% block bootstrap over whole slates. Hurdle `G` is the parametric growth rate from the Bernoulli-Gamma fit to per-bet ROI, in basis points.

| Model | Family | Bets | Flat ROI % | Hurdle G (bps) | Empirical G (bps) | Hurdle Sharpe |
|---|---|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | GRW | 1324 | +13.96 | +3.62 | +6.50 | 0.049 |
| `m12_joint_hybrid_synergy` | Production | 1328 | +13.22 | +3.61 | +5.27 | 0.048 |
| `m05_joint_production_wealth` | TimeDecay | 1301 | +13.36 | +2.78 | +4.85 | 0.035 |
| `m05_production_wealth_grw` | GRW | 1298 | +15.56 | +2.69 | +6.99 | 0.035 |
| `m05_production_wealth` | TimeDecay | 1298 | +13.73 | +2.50 | +4.14 | 0.035 |
| `m00_baseline` | TimeDecay | 1301 | +12.76 | +1.87 | +3.87 | 0.031 |
| `m05_joint_production_wealth_grw` | GRW | 1263 | +14.00 | +0.90 | +5.87 | 0.011 |

## 4. Raw versus calibrated

| Model | Variant | Bets | Return % | g / slate | Sharpe ann | MDD % | Edge pp |
|---|---|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw` | raw | 1324 | +153.84 | +0.00941 | 1.636 | -18.3 | +5.03 |
| `m00_baseline_grw` | t25_inv | 1243 | +110.70 | +0.00753 | 1.937 | -9.2 | +3.40 |
| `m00_baseline_grw` | close_std | 1312 | +104.63 | +0.00723 | 1.555 | -15.6 | +4.34 |
| `m05_production_wealth_grw` | raw | 1298 | +174.88 | +0.01021 | 1.715 | -19.7 | +5.02 |
| `m05_production_wealth_grw` | t25_inv | 1223 | +116.72 | +0.00781 | 2.014 | -9.5 | +3.38 |
| `m05_production_wealth_grw` | close_std | 1295 | +122.54 | +0.00808 | 1.667 | -16.3 | +4.31 |
| `m05_joint_production_wealth_grw` | raw | 1263 | +125.06 | +0.00819 | 1.647 | -20.0 | +4.50 |
| `m05_joint_production_wealth_grw` | t25_inv | 1195 | +61.24 | +0.00483 | 1.523 | -12.7 | +2.90 |
| `m05_joint_production_wealth_grw` | close_std | 1254 | +100.44 | +0.00702 | 1.644 | -16.8 | +4.02 |
| `m00_baseline` | raw | 1301 | +115.48 | +0.00775 | 1.215 | -21.1 | +4.98 |
| `m00_baseline` | t25_inv | 1227 | +80.74 | +0.00598 | 1.464 | -12.8 | +3.32 |
| `m00_baseline` | close_std | 1287 | +111.69 | +0.00758 | 1.410 | -17.3 | +4.38 |
| `m05_production_wealth` | raw | 1298 | +124.25 | +0.00816 | 1.305 | -21.0 | +4.87 |
| `m05_production_wealth` | t25_inv | 1211 | +76.61 | +0.00575 | 1.443 | -12.8 | +3.27 |
| `m05_production_wealth` | close_std | 1288 | +117.09 | +0.00783 | 1.476 | -16.6 | +4.28 |
| `m05_joint_production_wealth` | raw | 1301 | +117.34 | +0.00784 | 1.395 | -18.6 | +4.55 |
| `m05_joint_production_wealth` | t25_inv | 1219 | +70.74 | +0.00540 | 1.485 | -11.9 | +2.98 |
| `m05_joint_production_wealth` | close_std | 1296 | +114.30 | +0.00770 | 1.545 | -14.9 | +4.04 |
| `m12_joint_hybrid_synergy` | raw | 1328 | +129.32 | +0.00838 | 1.355 | -19.6 | +4.67 |
| `m12_joint_hybrid_synergy` | t25_inv | 1213 | +76.38 | +0.00573 | 1.539 | -13.6 | +3.09 |
| `m12_joint_hybrid_synergy` | close_std | 1316 | +124.39 | +0.00816 | 1.503 | -18.1 | +4.17 |

## 5. Calibrated posteriors — headline


### t25_inv

| Model | Family | Bets | Return % | CAGR % | g / slate | g 95% CI | Sharpe ann | Calmar | MDD % | Win % | Edge pp |
|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| `m05_production_wealth_grw` | GRW | 1223 | +116.72 | +56.6 | +0.00781 | [+0.0019, +0.0137] | 2.014 | 12.32 | -9.5 | 36.0 | +3.38 |
| `m00_baseline_grw` | GRW | 1243 | +110.70 | +54.0 | +0.00753 | [+0.0017, +0.0134] | 1.937 | 11.97 | -9.2 | 35.5 | +3.40 |
| `m00_baseline` | TimeDecay | 1227 | +80.74 | +40.9 | +0.00598 | [+0.0000, +0.0122] | 1.464 | 6.29 | -12.8 | 33.7 | +3.32 |
| `m05_production_wealth` | TimeDecay | 1211 | +76.61 | +39.1 | +0.00575 | [+0.0000, +0.0119] | 1.443 | 5.97 | -12.8 | 33.9 | +3.27 |
| `m12_joint_hybrid_synergy` | Production | 1213 | +76.38 | +39.0 | +0.00573 | [+0.0001, +0.0112] | 1.539 | 5.64 | -13.6 | 34.3 | +3.09 |
| `m05_joint_production_wealth` | TimeDecay | 1219 | +70.74 | +36.4 | +0.00540 | [-0.0001, +0.0108] | 1.485 | 5.95 | -11.9 | 34.7 | +2.98 |
| `m05_joint_production_wealth_grw` | GRW | 1195 | +61.24 | +31.9 | +0.00483 | [+0.0000, +0.0095] | 1.523 | 4.84 | -12.7 | 35.7 | +2.90 |

### close_std

| Model | Family | Bets | Return % | CAGR % | g / slate | g 95% CI | Sharpe ann | Calmar | MDD % | Win % | Edge pp |
|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| `m12_joint_hybrid_synergy` | Production | 1316 | +124.39 | +59.8 | +0.00816 | [-0.0000, +0.0163] | 1.503 | 6.88 | -18.1 | 33.9 | +4.17 |
| `m05_production_wealth_grw` | GRW | 1295 | +122.54 | +59.0 | +0.00808 | [+0.0010, +0.0153] | 1.667 | 7.51 | -16.3 | 34.7 | +4.31 |
| `m05_production_wealth` | TimeDecay | 1288 | +117.09 | +56.7 | +0.00783 | [+0.0000, +0.0158] | 1.476 | 7.04 | -16.6 | 32.9 | +4.28 |
| `m05_joint_production_wealth` | TimeDecay | 1296 | +114.30 | +55.6 | +0.00770 | [+0.0003, +0.0153] | 1.545 | 7.68 | -14.9 | 33.8 | +4.04 |
| `m00_baseline` | TimeDecay | 1287 | +111.69 | +54.5 | +0.00758 | [-0.0004, +0.0155] | 1.410 | 6.45 | -17.3 | 32.2 | +4.38 |
| `m00_baseline_grw` | GRW | 1312 | +104.63 | +51.5 | +0.00723 | [+0.0003, +0.0143] | 1.555 | 6.70 | -15.6 | 35.2 | +4.34 |
| `m05_joint_production_wealth_grw` | GRW | 1254 | +100.44 | +49.7 | +0.00702 | [+0.0008, +0.0134] | 1.644 | 5.99 | -16.8 | 34.4 | +4.02 |

### 5.1 How much the market supplied

| Model | Instant | Fixtures shifted | median w | w p10 | w p90 | median var retained | median market share |
|---|---|---:|---:|---:|---:|---:|---:|
| `m00_baseline` | close_std | 618 | 0.971 | 0.824 | 0.999 | 0.943 | 0.029 |
| `m00_baseline` | t25_inv | 567 | 0.295 | 0.252 | 0.511 | 0.087 | 0.705 |
| `m00_baseline_grw` | close_std | 618 | 0.974 | 0.824 | 0.999 | 0.949 | 0.026 |
| `m00_baseline_grw` | t25_inv | 567 | 0.292 | 0.251 | 0.498 | 0.085 | 0.708 |
| `m05_joint_production_wealth` | close_std | 618 | 0.977 | 0.874 | 0.999 | 0.955 | 0.023 |
| `m05_joint_production_wealth` | t25_inv | 567 | 0.286 | 0.251 | 0.458 | 0.082 | 0.714 |
| `m05_joint_production_wealth_grw` | close_std | 618 | 0.981 | 0.872 | 0.999 | 0.961 | 0.019 |
| `m05_joint_production_wealth_grw` | t25_inv | 567 | 0.284 | 0.251 | 0.451 | 0.080 | 0.716 |
| `m05_production_wealth` | close_std | 618 | 0.971 | 0.829 | 0.999 | 0.944 | 0.029 |
| `m05_production_wealth` | t25_inv | 567 | 0.293 | 0.252 | 0.491 | 0.086 | 0.707 |
| `m05_production_wealth_grw` | close_std | 618 | 0.975 | 0.828 | 0.999 | 0.952 | 0.025 |
| `m05_production_wealth_grw` | t25_inv | 567 | 0.290 | 0.251 | 0.492 | 0.084 | 0.710 |
| `m12_joint_hybrid_synergy` | close_std | 618 | 0.974 | 0.856 | 0.999 | 0.949 | 0.026 |
| `m12_joint_hybrid_synergy` | t25_inv | 567 | 0.290 | 0.251 | 0.478 | 0.084 | 0.710 |

### 5.2 Derivative coherence

Worst `max_family_spread` across 14 calibrated containers: **6.661338147750939e-16**. 1X2, O/U and BTTS are three partitions of one 12x12 score tensor, so this is zero to rounding by construction; measuring it verifies the construction rather than assuming it.


## 6. Reading this honestly

* **The interval is wider than the ranking.** With 622 fixtures the 95% band on per-slate growth overlaps zero for most arms. Order the table by return and the top row is not reliably the best model; it is the best sample path.
* **Proper scores already warned about this.** Task 007 measured Δ LogLoss of −0.00170 for the plain Poisson GRW but only −0.00031 for the joint Gamma-Poisson arm. The GRW state and the proxy-xG arm are substitutive: they compete to explain the same temporal signal, so a portfolio gain on `m00` does not license one on the production-shaped joint model.
* **`t25_inv` pooled to a median weight of 0.290**, leaving the market 0.710 of the location and 0.084 of the posterior log-variance. Across the seven arms it moved median staked edge by -1.604 pp, median return by -48.450 pp, median drawdown by +7.087 pp and median annual Sharpe by +0.128. Shrinking toward the book removes edge the allocator would have sized on, so a lower return here is the construction working rather than the model failing — the question a calibrator answers is whether the edge it removed was real.
* **`close_std` pooled to a median weight of 0.974**, leaving the market 0.026 of the location and 0.949 of the posterior log-variance. Across the seven arms it moved median staked edge by -0.592 pp, median return by -10.759 pp, median drawdown by +3.111 pp and median annual Sharpe by +0.150. Shrinking toward the book removes edge the allocator would have sized on, so a lower return here is the construction working rather than the model failing — the question a calibrator answers is whether the edge it removed was real.
* **Cost is not in these tables.** `m05_production_wealth_grw` sampled in 60.7 minutes against roughly 2.0 for its control — about 30x — and the joint GRW took 90.9. Portfolio return per unit of compute is materially worse than these rows alone suggest.
* **This is a historical simulation**, carrying the portfolio's stated fill assumptions on an exchange archive. It is not a prospective return claim.

