# CLV napkin: do any of our models beat the closing line, and where?

2026-09-28 · branch `proto/clv-napkin` · prototype, not a verdict.
Code: `l01_clv_napkin.jl` (loader) and `r01_clv_napkin.jl` (runner). Every number here is in
`out/*.csv`, and the run log is `out/r01_run.log`.

## 1. The answer in one paragraph

**At T−25, every model beats the close, and none beats it by enough to pay Betfair commission.**
- **CLV is positive for all 40 runs.** At e = 2 pp, mean CLV is +0.07 to +0.27 pp
  (clv_pct +0.4% to +0.8%). The market-only null is −0.06 to −0.09 pp.
- **A chunk of that is not football.** The T−25 entry is a last *traded* price, which bounces
  and goes stale, and the close reverts it.
  - A market-only placebo, whose "model" is the market's own de-vigged TWA over [T−180, T−25],
    predicts the entry→close move about 3–5× better than any model does (β 0.10–0.13 vs
    0.02–0.04).
  - Even the pure team-strength Poisson floor collects +0.11 pp.
- **The models still know something the market's own prices don't.** Controlling for the
  placebo:
  - every joint/pxG model's edge still predicts the move (β ≈ 0.016–0.037, bootstrap CI above 0,
    z ≈ 2–4);
  - pure-goals Poisson models are about half that and touch 0.
- **That information is small:** the market closes only about 3% of the disagreement the model
  claims.
- **Where it lives:** 1X2 (and BTTS, on tiny n) in League Two more than League One; nothing on
  O/U 2.5; nothing in the Premiership monitor.
- **Money:** the close-implied expected ROI after 2% commission is **negative for every model**
  (−0.5% to −0.9%). The 95% CI is wholly below 0 for all 40 runs.
  - Realised flat-stake ROI is +2% to +10%, but every CI spans 0.
  - That gap is luck, not evidence.
- **The one strong signal is timing.** Entering at the earliest complete Betfair book (median
  about 21 h before kick-off) instead of T−25:
  - raises CLV about 7× (m12: +1.02 vs +0.15 pp on the same selections);
  - raises β from 0.03 to 0.24;
  - raises the close-implied EV to about +9.5% [+2, +21].
  - Caveats: stale, thin early LTPs, and m12/m13 use the XI before it is announced. Section 7
    covers this.
- **Compression doesn't explain CLV.** On W0/W1 the observation model (joint pxG vs goals-only)
  does. On W2, the ρ = +0.64 comes from the pooled-TD arms, not from the slope.

## 2. Method (napkin)

- **Model probabilities:** posterior means from each persisted fit's held-out walk-forward
  latents, priced by `Predictions.model_inference`. No sampling, calibration or Kelly.
- **Entry:** the last traded price at or before T−25 (`Calibration.point_in_time_prices`).
  - The freshest tick must be ≤ 90 min old, the market complete, and the overround in
    [0.90, 1.10].
  - `p_entry_fair` (de-vigged entry) drives `edge_pp`; `p_entry = 1/odds_entry` is not de-vigged.
- **Close:** the harness's TWA over (−20, 0], de-vigged multiplicatively within each market.
  Incomplete markets are dropped, not normalised.
- **Bets and P&L:** 1-unit back bets with `edge_pp ≥ e`. Commission is charged on net winnings
  per Betfair market.
- **EV@close:** `100·mean[p_close·(odds−1)·(1−c) − (1−p_close)]`, the expected ROI if the close
  is true. It is the low-variance twin of realised ROI.
- **CIs:** 95% slate-clustered bootstrap (slate = match day, B = 2,000, seed 20260928).
  `z = point / bootstrap SD`.
- **`β move`:** the OLS slope of `p_close − p_entry_fair` on `p_model − p_entry_fair`, over all
  priced selections. It is the share of the model's disagreement that the market closes by
  kick-off.
- **Panels:** each run is scored on its own panel. W2 is 56/57, 23/24–25/26, 1,070 fixtures.
  W0/W1 is 56/57, 24/25–25/26, 710.
  - "W2@710" re-reads W2's frames on 24/25–25/26 for a like-for-like line with W0/W1. The
    Poisson floor exists only on the 710 panel.

**Coverage** (fixtures with a complete, fresh T−25 book *and* a complete close):

| market | W2 panel (1,070) | 710 panel |
|---|---|---|
| 1X2 | 784 | 520 |
| O/U 2.5 | 348 | 217 |
| O/U 3.5 / 1.5 / 0.5 | 153 / 104 / 39 | 90 / 59 / 25 |
| O/U 4.5 / 5.5 | 20 / 8 | 11 / 4 |
| BTTS | 110 | 78 |
| Correct score | **0** | **0** |
| DC / DNB / AH | not in the archive for tournaments 54–57 | same |

## 3. Table A: by model, e = 2 pp, all priced markets, entry T−25

"EV@close 2%" is the close-implied expected ROI. `z_clv` is the CLV point estimate over its
bootstrap SD.

**W0 / W1 panel (56/57, 24/25–25/26, 710 fixtures)**

| model | slope | n bets | mean clv_pp [95% CI] | beat % | ROI 2% [CI] | ROI 5% [CI] | EV@close 2% [CI] | z_clv | β move |
|---|---|---|---|---|---|---|---|---|---|
| MARKET NULL (back all) | — | 2528 | −0.055 [−0.085, −0.025] | 48.3 [47.0, 49.7] | +0.3 [−2.2, +2.7] | −0.2 [−2.6, +2.2] | −1.25 [−1.39, −1.10] | −3.5 | — |
| PLACEBO market TWA[−180,−25] | — | 164 | +0.249 [−0.024, +0.551] | 53.7 [47.1, 60.1] | +0.4 [−19.8, +22.0] | −1.6 [−21.4, +19.6] | −0.20 [−1.20, +0.82] | 1.7 | 0.102 |
| td_spfl_cups_joint | 1.50 | 889 | +0.247 [+0.115, +0.378] | 55.5 [51.8, 59.0] | +5.7 [−4.1, +16.2] | +3.7 [−5.9, +14.0] | −0.50 [−1.01, −0.02] | 3.7 | 0.041 |
| m05_joint_td | 1.87 | 813 | +0.204 [+0.087, +0.332] | 55.2 [51.9, 58.8] | +5.6 [−4.1, +15.5] | +3.5 [−6.0, +13.3] | −0.62 [−1.10, −0.11] | 3.3 | 0.034 |
| m13_joint_td | 1.49 | 834 | +0.191 [+0.072, +0.311] | 55.3 [51.9, 58.6] | +9.6 [−0.4, +19.4] | +7.5 [−2.3, +17.1] | −0.63 [−1.11, −0.15] | 3.2 | 0.033 |
| s11_m02_td_joint | 1.93 | 824 | +0.191 [+0.069, +0.313] | 54.7 [51.2, 58.1] | +4.3 [−5.5, +14.1] | +2.3 [−7.4, +11.8] | −0.63 [−1.13, −0.13] | 3.1 | 0.031 |
| td_lower_joint (= s12_m02) | 1.93 | 825 | +0.189 [+0.069, +0.311] | 54.7 [51.1, 58.0] | +4.4 [−5.5, +14.2] | +2.4 [−7.3, +11.9] | −0.64 [−1.13, −0.14] | 3.0 | 0.031 |
| td_spfl_joint | 1.40 | 942 | +0.186 [+0.068, +0.309] | 54.0 [50.5, 57.3] | +5.4 [−4.2, +15.0] | +3.4 [−6.0, +12.9] | −0.65 [−1.13, −0.16] | 3.1 | 0.029 |
| grw_spfl_cups_joint (= g3) | 1.10 | 915 | +0.186 [+0.063, +0.313] | 53.8 [50.4, 57.3] | +4.1 [−4.7, +12.9] | +2.2 [−6.4, +10.9] | −0.62 [−1.07, −0.13] | 2.9 | 0.036 |
| grw_spfl_joint | 1.07 | 923 | +0.182 [+0.065, +0.309] | 53.7 [50.4, 57.3] | +3.4 [−5.3, +11.7] | +1.5 [−7.0, +9.7] | −0.62 [−1.11, −0.14] | 2.9 | 0.028 |
| **m12_td (live)** | 1.57 | 823 | +0.181 [+0.062, +0.307] | 55.0 [51.8, 58.3] | +6.7 [−3.5, +17.0] | +4.6 [−5.4, +14.8] | −0.68 [−1.16, −0.19] | 2.9 | 0.034 |
| s12_m03_funnel_shared | 2.23 | 871 | +0.181 [+0.078, +0.290] | 54.3 [51.2, 57.2] | +4.5 [−5.6, +14.5] | +2.5 [−7.4, +12.2] | −0.71 [−1.15, −0.24] | 3.3 | 0.025 |
| s12_m04_funnel_hier | 2.34 | 850 | +0.173 [+0.065, +0.284] | 54.9 [51.6, 57.9] | +3.3 [−7.4, +14.2] | +1.3 [−9.2, +12.0] | −0.71 [−1.19, −0.26] | 3.2 | 0.026 |
| m05_joint_grw | 1.24 | 845 | +0.169 [+0.041, +0.312] | 53.5 [49.4, 57.6] | +5.2 [−3.7, +15.0] | +3.1 [−5.5, +12.8] | −0.70 [−1.22, −0.19] | 2.5 | 0.037 |
| grw_lower_joint | 1.24 | 841 | +0.167 [+0.037, +0.304] | 53.2 [49.3, 57.0] | +4.8 [−3.9, +14.0] | +2.8 [−5.7, +11.8] | −0.74 [−1.23, −0.24] | 2.5 | 0.034 |
| td_spfl_poisson | 1.91 | 840 | +0.162 [+0.033, +0.291] | 53.0 [49.0, 56.8] | +5.6 [−4.5, +16.7] | +3.5 [−6.3, +14.4] | −0.74 [−1.24, −0.25] | 2.5 | 0.019 |
| s11_m03_negbin_pxg | 1.48 | 848 | +0.161 [+0.035, +0.293] | 53.7 [50.4, 57.1] | +1.1 [−10.3, +12.2] | −1.0 [−12.0, +9.9] | −0.74 [−1.23, −0.24] | 2.4 | 0.037 |
| m12_grw | 1.33 | 867 | +0.149 [+0.024, +0.278] | 53.5 [49.8, 57.2] | +8.1 [−1.0, +17.7] | +6.1 [−2.9, +15.4] | −0.77 [−1.25, −0.30] | 2.3 | 0.033 |
| td_spfl_cups_poisson | 2.03 | 826 | +0.136 [+0.002, +0.269] | 52.5 [48.5, 56.4] | +6.7 [−4.0, +18.2] | +4.6 [−5.9, +15.9] | −0.82 [−1.33, −0.30] | 2.0 | 0.021 |
| s10_m02_grw_poisson | 1.19 | 883 | +0.116 [−0.018, +0.250] | 52.3 [49.0, 55.9] | +4.9 [−4.7, +14.7] | +2.9 [−6.5, +12.5] | −0.77 [−1.28, −0.26] | 1.7 | 0.018 |
| **FLOOR s12_m01_td_poisson** | 2.78 | 837 | +0.113 [+0.003, +0.226] | 53.4 [50.1, 56.7] | +4.7 [−6.2, +15.4] | +2.6 [−8.0, +13.1] | −0.87 [−1.36, −0.38] | 2.0 | 0.019 |
| s10_m01_td_poisson | 2.79 | 834 | +0.103 [−0.007, +0.215] | 53.1 [49.7, 56.4] | +4.6 [−5.7, +15.2] | +2.5 [−7.5, +12.9] | −0.88 [−1.38, −0.39] | 1.8 | 0.019 |
| s11_m01_td_poisson | 2.77 | 840 | +0.097 [−0.015, +0.210] | 52.9 [49.5, 56.1] | +5.2 [−5.7, +16.1] | +3.1 [−7.6, +13.8] | −0.92 [−1.40, −0.42] | 1.7 | 0.019 |
| grw_lower_poisson (= m00) | 1.19 | 884 | +0.094 [−0.042, +0.231] | 51.6 [48.1, 55.2] | +5.5 [−4.1, +15.3] | +3.5 [−5.9, +13.1] | −0.85 [−1.35, −0.32] | 1.3 | 0.018 |
| grw_spfl_cups_poisson (= g2) | 1.25 | 872 | +0.085 [−0.052, +0.222] | 50.3 [46.6, 54.1] | +7.6 [−2.6, +18.2] | +5.6 [−4.4, +15.9] | −0.81 [−1.33, −0.31] | 1.2 | 0.017 |
| s10_m03_momentum_grw | 1.13 | 899 | +0.077 [−0.060, +0.217] | 50.9 [47.3, 54.5] | +8.3 [−1.7, +18.0] | +6.3 [−3.5, +15.8] | −0.88 [−1.35, −0.39] | 1.1 | 0.018 |
| grw_spfl_poisson (= g1) | 1.10 | 887 | +0.069 [−0.071, +0.205] | 49.5 [45.8, 53.2] | +6.4 [−3.7, +16.9] | +4.5 [−5.5, +14.7] | −0.89 [−1.38, −0.40] | 1.0 | 0.015 |

**W2 panel (56/57, 23/24–25/26, 1,070 fixtures)**

| model | slope | n bets | mean clv_pp [95% CI] | beat % | ROI 2% [CI] | ROI 5% [CI] | EV@close 2% [CI] | z_clv | β move |
|---|---|---|---|---|---|---|---|---|---|
| MARKET NULL (back all) | — | 3916 | −0.086 [−0.112, −0.060] | 47.5 [46.3, 48.7] | +0.7 [−1.7, +3.2] | +0.2 [−2.2, +2.6] | −1.26 [−1.41, −1.10] | −6.3 | — |
| PLACEBO market TWA[−180,−25] | — | 203 | +0.276 [+0.012, +0.551] | 54.7 [47.7, 61.3] | +16.7 [−16.6, +59.3] | +14.3 [−18.1, +55.8] | +0.93 [−0.81, +3.59] | 2.0 | 0.129 |
| td_ldelta | 1.53 | 1325 | +0.187 [+0.085, +0.283] | 54.2 [51.1, 57.1] | +1.8 [−7.8, +11.1] | −0.2 [−9.6, +8.9] | −0.55 [−1.01, −0.10] | 3.7 | 0.027 |
| td_a2_carry | 1.58 | 1411 | +0.164 [+0.062, +0.272] | 54.6 [51.7, 57.6] | +3.5 [−5.9, +12.4] | +1.5 [−7.7, +10.2] | −0.56 [−0.99, −0.12] | 3.0 | 0.028 |
| td_base | 1.51 | 1417 | +0.163 [+0.066, +0.260] | 53.9 [51.0, 56.7] | +1.7 [−7.2, +11.3] | −0.2 [−8.9, +9.2] | −0.59 [−1.02, −0.17] | 3.3 | 0.026 |
| td_a2_carry_jump_ldelta | 1.61 | 1335 | +0.162 [+0.065, +0.262] | 53.6 [50.4, 56.5] | +3.4 [−6.6, +13.3] | +1.4 [−8.4, +11.0] | −0.58 [−1.03, −0.14] | 3.2 | 0.029 |
| td_a2 | 1.51 | 1419 | +0.152 [+0.050, +0.258] | 54.1 [51.1, 57.0] | +3.1 [−6.3, +12.2] | +1.1 [−8.1, +10.0] | −0.60 [−1.02, −0.18] | 2.9 | 0.028 |
| td_a2_carry_jump | 1.56 | 1398 | +0.144 [+0.043, +0.251] | 53.6 [50.5, 56.6] | +3.6 [−6.0, +12.9] | +1.6 [−7.8, +10.7] | −0.64 [−1.07, −0.21] | 2.7 | 0.028 |
| **grw_step_a2_carry_jump** (co-finalist) | 1.20 | 1395 | +0.131 [+0.029, +0.233] | 53.1 [50.2, 55.9] | +3.7 [−4.8, +12.3] | +1.7 [−6.6, +10.1] | −0.68 [−1.12, −0.26] | 2.5 | 0.027 |
| **td_lower_a2full_carry_jump** (co-finalist) | 1.46 | 1321 | +0.130 [+0.030, +0.232] | 53.3 [50.1, 56.2] | +5.1 [−4.5, +14.6] | +3.0 [−6.3, +12.2] | −0.66 [−1.09, −0.23] | 2.5 | 0.023 |
| td_lower_b1lit | 1.32 | 1337 | +0.122 [+0.024, +0.224] | 53.2 [50.2, 55.9] | +2.6 [−6.3, +11.7] | +0.6 [−8.1, +9.5] | −0.65 [−1.07, −0.23] | 2.4 | 0.022 |
| grw_step_a2_carry_jump_ldelta | 1.21 | 1344 | +0.120 [+0.013, +0.221] | 52.2 [49.2, 55.1] | +2.6 [−6.9, +11.8] | +0.6 [−8.6, +9.6] | −0.70 [−1.14, −0.27] | 2.2 | 0.027 |
| grw_step | 1.15 | 1406 | +0.119 [+0.014, +0.223] | 52.2 [49.5, 54.9] | +4.1 [−4.3, +12.8] | +2.1 [−6.1, +10.6] | −0.69 [−1.13, −0.29] | 2.3 | 0.026 |
| td_lower_base | 1.60 | 1340 | +0.116 [+0.022, +0.210] | 53.1 [50.3, 55.9] | +3.3 [−5.9, +12.7] | +1.3 [−7.7, +10.5] | −0.68 [−1.12, −0.22] | 2.4 | 0.021 |
| m12_td_ref | 1.45 | 1319 | +0.110 [+0.018, +0.200] | 53.3 [50.7, 55.9] | +4.5 [−5.2, +14.9] | +2.4 [−7.1, +12.6] | −0.71 [−1.12, −0.29] | 2.4 | 0.024 |
| grw_base | 1.14 | 1436 | +0.107 [+0.009, +0.206] | 52.2 [49.4, 54.9] | +4.6 [−3.3, +13.4] | +2.7 [−5.0, +11.2] | −0.73 [−1.15, −0.35] | 2.1 | 0.024 |
| grw_step_ldelta | 1.16 | 1334 | +0.097 [−0.007, +0.201] | 50.9 [47.9, 53.9] | +4.5 [−4.5, +14.1] | +2.5 [−6.3, +11.9] | −0.74 [−1.17, −0.29] | 1.8 | 0.024 |

On the W2@710 view (24/25–25/26 only):
- every W2 arm's CLV rises to +0.18 to +0.27;
- the ranking is unchanged: pooled TD first, GRW and lower-scope TD tied (`out/table_a_all_thresholds.csv`).

### The placebo test: does the model know more than the market's own recent prices?

This is an OLS of the entry→close move on the model edge *and* the placebo edge. The model
coefficient comes with a 300-rep slate bootstrap CI. The model and placebo edges correlate only
about 0.2.

| model (710 panel unless noted) | β model alone | β model given placebo [95% CI] | β placebo given model |
|---|---|---|---|
| td_spfl_cups_joint | 0.041 | **0.037 [0.021, 0.054]** | 0.065 |
| s11_m03_negbin_pxg / grw_spfl_cups_joint / m05_joint_grw | 0.036–0.037 | 0.033 [≈0.015, ≈0.051] | ≈0.065 |
| m12_td (live) | 0.034 | **0.031 [0.018, 0.047]** | 0.072 |
| m05_joint_td / m13_joint_td | 0.033–0.034 | 0.030 [0.015, 0.047] | ≈0.075 |
| FLOOR s12_m01_td_poisson | 0.019 | 0.016 [0.000, 0.033] | 0.084 |
| goals-only GRW Poisson (g1, g2, m00, s10_m02, s10_m03) | 0.015–0.018 | 0.011–0.014 [≈−0.004, ≈0.032] | ≈0.087 |
| W2: td_* pooled arms (1,070) | 0.026–0.029 | 0.022–0.024 [≈0.010, ≈0.040] | ≈0.10 |
| W2: grw_* arms (1,070) | 0.024–0.027 | 0.018–0.022 [≈0.006, ≈0.036] | ≈0.10 |
| W2: td_lower_* / m12_td_ref (1,070) | 0.021–0.024 | 0.016–0.020 [≈0.002, ≈0.033] | ≈0.11 |

Full table: `out/table_joint_capture.csv`.

## 4. Table B: by market, e = 2, best three models plus floor and m12 (710 panel)

The best three are ranked by CLV at e = 2, which is selection-biased by construction. "all" rows
(e = −∞) are identical across models: they are the market's own spread.

| market | all: clv_pp | td_spfl_cups_joint | m05_joint_td | m13_joint_td | m12_td (live) | FLOOR | placebo |
|---|---|---|---|---|---|---|---|
| **1X2** | −0.009 | **+0.326 [+0.163, +0.495]** n 535 | +0.258 [+0.112, +0.411] n 529 | +0.256 [+0.108, +0.406] n 541 | +0.244 [+0.089, +0.393] n 532 | +0.146 [−0.005, +0.303] n 546 | +0.223 [−0.07, +0.54] n 149 |
| O/U 2.5 | −0.212 | +0.016 [−0.21, +0.23] n 169 | −0.027 n 130 | −0.077 n 135 | −0.092 [−0.35, +0.16] n 133 | −0.086 n 127 | n 7 |
| O/U 3.5 | −0.118 | +0.042 n 73 | −0.137 n 62 | −0.110 n 61 | −0.137 n 62 | −0.138 n 67 | n 5 |
| O/U 1.5 | −0.158 | −0.078 n 44 | +0.003 n 33 | −0.138 n 34 | −0.178 n 33 | −0.097 n 36 | n 1 |
| BTTS | +0.028 | +0.738 [+0.28, +1.20] n 50 | +0.776 [+0.29, +1.28] n 44 | +0.741 n 45 | +0.812 [+0.35, +1.26] n 45 | +0.664 [+0.28, +1.07] n 47 | n 2 |
| O/U 0.5 / 4.5 / 5.5 | — | n ≤ 9 each | | | | | |
| Correct score | — | no complete books | | | | | |

- **1X2 is where the CLV is.**
  - EV@close at 2% is still ≤ 0 there: td_spfl_cups_joint −0.33 [−1.05, +0.39], m12 −0.57.
  - β move is 0.035–0.038 for the joint models and 0.019 for the floor.
- **O/U 2.5 has no CLV** for any model. β is 0.01–0.04 and the EV@close is about −1.0 to −1.3.
  - This is the market we have been betting alongside 1X2.
- **BTTS looks strong but is not believable yet.**
  - About 45 bets, all on the same ~78 fixtures.
  - The floor does as well as the joint models.
  - The placebo's β on BTTS is 0.7–0.9, i.e. BTTS LTPs are very noisy.
  - Most of it is probably price reversion; the EV@close CI spans 0.

## 5. Table C: edge buckets, league, lead time

**CLV by model-edge bucket (all selections, T−25, 710 panel):** roughly monotonic for every
model. Negative edges earn negative CLV, and the top buckets earn the most.

| edge (pp) | < −6 | [−6,−4) | [−4,−2) | [−2,0) | [0,2) | [2,4) | [4,6) | [6,10) | ≥ 10 |
|---|---|---|---|---|---|---|---|---|---|
| m12_td (live) | −0.54 | −0.35 | −0.17 | +0.02 | +0.01 | +0.01 | −0.01 | +0.25 | **+0.50** [+0.24, +0.77] |
| td_spfl_cups_joint | −0.57 | −0.46 | −0.17 | −0.16 | +0.15 | +0.07 | +0.20 | +0.40 | **+0.40** [+0.09, +0.72] |
| FLOOR | −0.35 | −0.25 | −0.19 | −0.07 | +0.08 | +0.06 | −0.05 | +0.16 | +0.26 [+0.03, +0.52] |

Between −2 and +6 pp the model's edge carries essentially no CLV. It only shows at the tails.
- **Thresholds (appendix):** CLV rises with e for every model, and so does EV@close.
  - For m12, EV@close at e = 0/2/4/6 is −0.80 / −0.68 / −0.41 / −0.13%.
  - Break-even at T−25 needs e ≳ 6–8 pp on a joint model. Only m05_joint_grw (+0.21),
    grw_lower_joint (+0.03) and m13 (+0.01) touch 0 at e = 6, all with CIs spanning 0.

**League, e = 2:** League Two beats League One for every model, but the CIs overlap.

| model | 56 League One clv_pp | 57 League Two clv_pp |
|---|---|---|
| td_spfl_cups_joint | +0.193 [+0.050, +0.340] | +0.315 [+0.125, +0.497] |
| m05_joint_td | +0.155 [+0.008, +0.305] | +0.263 [+0.062, +0.461] |
| m12_td (live) | +0.129 [−0.020, +0.291] | +0.242 [+0.050, +0.429] |
| FLOOR | +0.073 [−0.068, +0.227] | +0.156 [−0.021, +0.337] |

**Lead time, e = 2, paired.** These are the same 1,316 selections, present in all three entry
books (710 panel). "Earliest" is the first instant every runner of the market had traded. Its
median is 1,255 min, about 21 h before kick-off, and it has no staleness gate.

| model | entry | n bets | clv_pp [CI] | β move | ROI 2% [CI] | EV@close 2% [CI] |
|---|---|---|---|---|---|---|
| m12_td (live) | earliest | 448 | **+1.02 [+0.55, +1.48]** | 0.235 | +23.0 [+1.1, +50.2] | **+9.5 [+2.0, +21.3]** |
| m12_td (live) | T−60 | 445 | +0.41 [+0.14, +0.68] | 0.076 | +14.5 [−0.9, +31.4] | +0.4 [−0.7, +1.7] |
| m12_td (live) | T−25 | 432 | +0.15 [−0.02, +0.32] | 0.033 | +10.8 [−4.3, +26.8] | −0.8 [−1.4, −0.2] |
| m05_joint_td | earliest / T−60 / T−25 | ~435 | +1.03 / +0.44 / +0.17 | 0.27 / 0.08 / 0.03 | +26.6 / +12.3 / +12.2 | +9.7 / +0.6 / −0.7 |
| td_spfl_cups_joint | earliest / T−60 / T−25 | ~460 | +1.04 / +0.35 / +0.23 | 0.25 / 0.07 / 0.04 | +26.9 / +12.9 / +12.3 | +9.5 / +0.3 / −0.5 |
| FLOOR | earliest / T−60 / T−25 | ~440 | +0.66 / +0.20 / +0.08 | 0.17 / 0.04 / 0.02 | +19.2 / +12.3 / +8.7 | +8.7 / −0.1 / −1.0 |

```
m12 mean clv_pp on the same selections       β move (share of disagreement the market closes)
earliest  ██████████████████████████  +1.02  earliest  ████████████████████████  0.235
T−60      ██████████                  +0.41  T−60      ████████                  0.076
T−25      ████                        +0.15  T−25      ███                       0.033
```

The market absorbs most of what our models know between the first prices and T−60. By T−25
little is left. This is the most interesting result here, and it is the one with the weakest
execution realism (see Caveats).

## 6. Compression vs CLV/ROI, one line per panel

Spearman ρ across runs of the 1X2 compression slope (market-on-model; > 1 = compressed) against
the e = 2 outcomes:

| panel | runs | ρ(slope, CLV) | ρ(slope, EV@close) | ρ(slope, realised ROI) | ρ(slope, n bets) | ρ(slope, β) |
|---|---|---|---|---|---|---|
| W2, 1,070 | 15 | **+0.64** | +0.74 | −0.43 | −0.14 | +0.34 |
| W2@710 view | 15 | +0.72 | +0.53 | +0.54 | −0.46 | +0.36 |
| W0/W1, 710 | 25 | +0.13 | −0.10 | −0.17 | −0.66 | −0.00 |

```
W2 (1,070): y = mean clv_pp at e = 2, x = 1X2 compression slope
clv_pp  +0.19
  |                                                e
  |
  |
  |
  |                                              a        c   f
  |
  |                                              b
  |                                                    d
  |
  |       i
  |                                        m
  |  h       k            n
  |                                                          l
  |g                                      o
  |
  |   j
    +0.10                                            slope 1.14 → 1.61

pooled TD:  a td_base  b td_a2  c td_a2_carry  d td_a2_carry_jump  e td_ldelta  f td_a2_carry_jump_ldelta
GRW:        g grw_base  h grw_step  i grw_step_a2_carry_jump  j grw_step_ldelta  k grw_step_a2_carry_jump_ldelta
lower TD:   l td_lower_base  m td_lower_a2full_carry_jump  n td_lower_b1lit  o m12_td_ref
```

**Does the data support the hypothesis? Mostly no. There is weak support on CLV, and none on
ROI.**
- **W2:** the less-compressed GRW arms (slope 1.14–1.21) earn less CLV than the pooled-TD arms
  (1.51–1.61), +0.115 vs +0.162 pp.
  - That is the direction the hypothesis predicts ("fixing compression costs CLV").
  - But the lower-scope TD arms (td_lower_*, m12_td_ref) have pooled-TD-like slopes (1.32–1.60)
    and GRW-like CLV (+0.119).
  - So the ρ tracks *scope/dynamics family*, not compression.
- **W0/W1:** there is no relationship (ρ = +0.13). What separates runs is the observation model.
  - Joint/pxG models average +0.18 pp CLV at every slope from 1.07 to 2.34.
  - Goals-only Poisson models average +0.11 at every slope from 1.10 to 2.79.
  - β given placebo roughly doubles, from about 0.014 to about 0.03, with pxG.
- **n bets** does not fall as compression is fixed. If anything, more-compressed models bet
  slightly less (ρ −0.66 on 710, a range of only 813–942 bets).
- **Realised ROI** flips sign between two views of the same W2 runs (−0.43 vs +0.54). It is
  noise at this sample size.
- **Takeaway:** compression is a calibration problem worth fixing for Kelly sizing. It is not
  what decides whether our disagreements are right. pxG information and entry timing are.

## 7. 54/55 monitor (pooled W1/W2 fits), e = 2, T−25

| panel | model | 54 Premiership clv_pp | 55 Championship clv_pp | β (54 / 55) |
|---|---|---|---|---|
| W2 | td_base | −0.067 [−0.111, −0.021] | +0.052 [−0.055, +0.155] | 0.001 / 0.023 |
| W2 | td_ldelta | −0.113 [−0.155, −0.071] | −0.006 [−0.112, +0.099] | −0.006 / 0.018 |
| W2 | grw_step_a2_carry_jump | −0.053 [−0.096, −0.007] | +0.067 [−0.033, +0.166] | 0.002 / 0.032 |
| W1 | td_spfl_cups_joint | −0.037 [−0.092, +0.018] | +0.044 [−0.096, +0.176] | 0.003 / 0.022 |
| W1 | grw_spfl_joint | −0.053 [−0.108, +0.005] | +0.133 [−0.004, +0.265] | 0.002 / 0.030 |
| null (back all) | — | −0.08 | −0.12 to −0.14 | — |

- **Premiership: no information** (β ≈ 0).
  - The league-δ arms bet *worse* than the null there, i.e. they are confidently wrong about
    Premiership rates.
- **Championship:** a Scottish-lower-sized β (0.02–0.03), with CIs spanning 0. The Betfair
  archive covers 22/23–24/25 only (TODO 033).
- Full per-run rows: `out/table_monitor.csv`.

## 8. Caveats

**Napkin method.**
- **Entry is a last traded price, not a back price.**
  - Where the LTP printed on the lay side, the real back price was one tick longer; where it
    printed on the back side, the LTP is the price.
  - Staleness up to 90 min is allowed at T−25.
  - The placebo exists because of this.
- **No liquidity or size check.**
  - Early-book prices in League One/Two are often small matched amounts. The earliest-book EV is
    an upper bound on what 1-unit orders would have got.
- **m12/m13 price with the actual XI.**
  - It is announced around T−60, so the *earliest*-book numbers for the lineup models use
    information that did not exist then.
  - m05_joint_td and td_spfl_cups_joint have no lineup term and show the same early-entry lift,
    so the lift is not lineup leakage. But m12's early row specifically is not clean.
- **Commission is charged per market on the flat bets.** The live system's Kelly stakes, trust
  and caps are not modelled.
- **EV@close treats the de-vigged close as the truth.** If the Betfair close in these leagues has
  a favourite–longshot bias, EV@close is biased with it.
  - The realised null ROI (+0.3 to +0.7) sits about 1.5–2 pp above its EV@close (−1.25), a hint
    of that or of noise.

**Multiple comparisons.**
- About 55 run-views × 4 thresholds, plus markets, leagues and instants, adds up to a few hundred
  slices, all on the same fixtures, so they are highly correlated.
- A rough Bonferroni over the ~160 run × threshold cells needs one-sided z > 3.4.
- **What survives:**
  1. **β given placebo > 0 for the joint/pxG models.** z ≈ 4 for td_spfl_cups_joint and m12, and
     it is consistent across all 30+ joint runs.
  2. **EV@close at T−25, e = 2, is negative.** All 40 CIs sit wholly below 0 (z −2.0 to −3.7),
     and the *direction* is consistent across every run.
  3. **CLV > 0 at e = 2** for td_ldelta (z 3.7; 4.2 on W2@710) and td_spfl_cups_joint (3.7). CLV
     alone is the weakest of the three because the placebo shows part of it is LTP reversion.
- **What doesn't survive:**
  - any realised ROI;
  - any per-market result except 1X2 CLV;
  - BTTS;
  - the League One vs League Two gap;
  - the compression ρ's.
- The earliest-book EV (+9.5, CI [+2, +21]) is five models × one instant. It clears a nominal
  test and is worth a proper study, not a conclusion.

**Data gaps.**
- **Correct score** is unevaluable: 0 panel fixtures have all 19 runners traded at both T−25 and
  the close. Longshot scorelines rarely trade, so LTP-based de-vig cannot be done.
- **DC, DNB and AH** do not exist in `betfair.markets` for tournaments 54–57.
- **O/U lines other than 2.5** have 4–153 usable fixtures. **BTTS** has 78–110.
- **1X2 coverage** is 520/710 and 784/1,070. It is limited by the 90-min staleness gate: at 240
  min it would be 571/710.
- **No Poisson floor** is fitted on the W2 1,070 panel. Compare W2 to the floor via the W2@710
  view.

## 9. Reproduce

On mcmc-beast, in `/root/BF_runs/clv_napkin_dev` with `.env` sourced:

```bash
nice -n 19 /root/.juliaup/bin/julia --project -t 8 current_development/clv_napkin/r01_clv_napkin.jl
```

- Pricing all 40 fits takes about 7 min; the tables take about 10 min.
- `CNR_REUSE_PROBS=1` reuses `out/probs.jls`, which is on the beast only (60 MB, not committed).

**Suggested next steps**, not done here:
1. A liquidity-aware early-entry study from `betfair_live` ladders (back price and size, not
   LTP), for 1X2 only, with lineup-free models.
2. A threshold/lead-time grid for flat staking with EV@close as the objective.
3. Drop O/U 2.5 from the live book unless a model shows β > 0 there.

## Appendix: Table A at every threshold (T−25, all priced markets)

Each cell is mean clv_pp / EV@close at 2% / n bets. Realised-ROI CIs are in
`out/table_a_all_thresholds.csv`.

| panel | model | e = 0 | e = 2 | e = 4 | e = 6 |
|---|---|---|---|---|---|
| W2 1,070 | td_base | +0.140 / -0.68 / 1937 | +0.163 / -0.59 / 1417 | +0.213 / -0.38 / 1006 | +0.251 / -0.14 / 685 |
| W2 1,070 | td_a2 | +0.145 / -0.68 / 1906 | +0.152 / -0.60 / 1419 | +0.181 / -0.46 / 991 | +0.211 / -0.27 / 641 |
| W2 1,070 | td_a2_carry | +0.129 / -0.71 / 1914 | +0.164 / -0.56 / 1411 | +0.189 / -0.44 / 992 | +0.217 / -0.25 / 646 |
| W2 1,070 | td_a2_carry_jump | +0.134 / -0.71 / 1915 | +0.144 / -0.64 / 1398 | +0.193 / -0.44 / 1004 | +0.189 / -0.34 / 669 |
| W2 1,070 | td_ldelta | +0.125 / -0.69 / 1903 | +0.187 / -0.55 / 1325 | +0.213 / -0.33 / 902 | +0.216 / -0.18 / 611 |
| W2 1,070 | td_a2_carry_jump_ldelta | +0.119 / -0.73 / 1890 | +0.162 / -0.58 / 1335 | +0.191 / -0.44 / 899 | +0.229 / -0.23 / 602 |
| W2 1,070 | grw_base | +0.078 / -0.84 / 1934 | +0.107 / -0.73 / 1436 | +0.168 / -0.52 / 1037 | +0.237 / -0.24 / 708 |
| W2 1,070 | grw_step | +0.068 / -0.88 / 1946 | +0.119 / -0.69 / 1406 | +0.173 / -0.44 / 998 | +0.210 / -0.24 / 674 |
| W2 1,070 | grw_step_a2_carry_jump | +0.096 / -0.78 / 1941 | +0.131 / -0.68 / 1395 | +0.183 / -0.44 / 980 | +0.238 / -0.14 / 657 |
| W2 1,070 | grw_step_ldelta | +0.090 / -0.79 / 1884 | +0.097 / -0.74 / 1334 | +0.132 / -0.50 / 954 | +0.169 / -0.41 / 647 |
| W2 1,070 | grw_step_a2_carry_jump_ldelta | +0.111 / -0.73 / 1897 | +0.120 / -0.70 / 1344 | +0.145 / -0.50 / 943 | +0.212 / -0.26 / 611 |
| W2 1,070 | td_lower_base | +0.081 / -0.82 / 1916 | +0.116 / -0.68 / 1340 | +0.128 / -0.54 / 929 | +0.176 / -0.27 / 627 |
| W2 1,070 | td_lower_a2full_carry_jump | +0.082 / -0.81 / 1895 | +0.130 / -0.66 / 1321 | +0.110 / -0.60 / 890 | +0.111 / -0.55 / 591 |
| W2 1,070 | td_lower_b1lit | +0.084 / -0.80 / 1902 | +0.122 / -0.65 / 1337 | +0.138 / -0.52 / 938 | +0.127 / -0.48 / 604 |
| W2 1,070 | m12_td_ref | +0.054 / -0.87 / 1907 | +0.110 / -0.71 / 1319 | +0.163 / -0.44 / 937 | +0.189 / -0.32 / 645 |
| W2 1,070 | PLACEBO market TWA[−180,−25] | +0.118 / -0.62 / 2082 | +0.276 / +0.93 / 203 | +0.178 / -0.28 / 38 | +0.290 / -0.66 / 9 |
| 710 | td_lower_joint | +0.150 / -0.76 / 1214 | +0.189 / -0.64 / 825 | +0.201 / -0.54 / 577 | +0.350 / -0.05 / 395 |
| 710 | FLOOR s12_m01_td_poisson | +0.103 / -0.90 / 1198 | +0.113 / -0.87 / 837 | +0.134 / -0.79 / 590 | +0.208 / -0.56 / 422 |
| 710 | grw_lower_poisson | +0.062 / -0.95 / 1207 | +0.094 / -0.85 / 884 | +0.121 / -0.71 / 631 | +0.145 / -0.64 / 440 |
| 710 | grw_spfl_poisson | +0.070 / -0.91 / 1193 | +0.069 / -0.89 / 887 | +0.124 / -0.74 / 631 | +0.140 / -0.63 / 473 |
| 710 | grw_spfl_cups_poisson | +0.059 / -0.95 / 1190 | +0.085 / -0.81 / 872 | +0.111 / -0.75 / 644 | +0.072 / -0.92 / 469 |
| 710 | grw_spfl_cups_joint | +0.143 / -0.75 / 1229 | +0.186 / -0.62 / 915 | +0.245 / -0.46 / 655 | +0.395 / -0.01 / 444 |
| 710 | td_spfl_poisson | +0.132 / -0.80 / 1195 | +0.162 / -0.74 / 840 | +0.169 / -0.66 / 590 | +0.208 / -0.55 / 413 |
| 710 | td_spfl_joint | +0.162 / -0.75 / 1272 | +0.186 / -0.65 / 942 | +0.229 / -0.54 / 711 | +0.308 / -0.17 / 504 |
| 710 | td_spfl_cups_poisson | +0.121 / -0.82 / 1186 | +0.136 / -0.82 / 826 | +0.149 / -0.80 / 579 | +0.205 / -0.55 / 409 |
| 710 | td_spfl_cups_joint | +0.222 / -0.55 / 1228 | +0.247 / -0.50 / 889 | +0.331 / -0.25 / 607 | +0.400 / -0.04 / 400 |
| 710 | grw_lower_joint | +0.147 / -0.75 / 1208 | +0.167 / -0.74 / 841 | +0.220 / -0.47 / 596 | +0.316 / +0.03 / 401 |
| 710 | grw_spfl_joint | +0.158 / -0.72 / 1242 | +0.182 / -0.62 / 923 | +0.191 / -0.61 / 668 | +0.247 / -0.37 / 484 |
| 710 | m12_td (live) | +0.125 / -0.80 / 1211 | +0.181 / -0.68 / 823 | +0.250 / -0.41 / 585 | +0.364 / -0.13 / 406 |
| 710 | m05_joint_td | +0.142 / -0.75 / 1218 | +0.204 / -0.62 / 813 | +0.216 / -0.51 / 561 | +0.336 / -0.05 / 386 |
| 710 | m13_joint_td | +0.126 / -0.80 / 1201 | +0.191 / -0.63 / 834 | +0.283 / -0.32 / 577 | +0.383 / +0.01 / 396 |
| 710 | m12_grw | +0.147 / -0.79 / 1217 | +0.149 / -0.77 / 867 | +0.253 / -0.41 / 595 | +0.272 / -0.22 / 406 |
| 710 | m05_joint_grw | +0.135 / -0.79 / 1202 | +0.169 / -0.70 / 845 | +0.257 / -0.38 / 598 | +0.369 / +0.21 / 390 |
| 710 | s10_m01_td_poisson | +0.118 / -0.86 / 1198 | +0.103 / -0.88 / 834 | +0.113 / -0.85 / 591 | +0.202 / -0.59 / 417 |
| 710 | s10_m02_grw_poisson | +0.066 / -0.94 / 1203 | +0.116 / -0.77 / 883 | +0.128 / -0.70 / 626 | +0.163 / -0.60 / 438 |
| 710 | s10_m03_momentum_grw | +0.039 / -1.02 / 1192 | +0.077 / -0.88 / 899 | +0.129 / -0.70 / 628 | +0.158 / -0.62 / 437 |
| 710 | s11_m01_td_poisson | +0.098 / -0.91 / 1201 | +0.097 / -0.92 / 840 | +0.127 / -0.81 / 591 | +0.200 / -0.58 / 421 |
| 710 | s11_m02_td_joint | +0.153 / -0.75 / 1215 | +0.191 / -0.63 / 824 | +0.206 / -0.54 / 577 | +0.346 / -0.05 / 395 |
| 710 | s11_m03_negbin_pxg | +0.145 / -0.75 / 1186 | +0.161 / -0.74 / 848 | +0.228 / -0.47 / 576 | +0.340 / -0.10 / 367 |
| 710 | s12_m03_funnel_shared | +0.151 / -0.75 / 1213 | +0.181 / -0.71 / 871 | +0.204 / -0.59 / 620 | +0.238 / -0.48 / 446 |
| 710 | s12_m04_funnel_hier | +0.166 / -0.73 / 1213 | +0.173 / -0.71 / 850 | +0.247 / -0.47 / 603 | +0.242 / -0.48 / 428 |
| 710 | PLACEBO market TWA[−180,−25] | +0.138 / -0.69 / 1358 | +0.249 / -0.20 / 164 | +0.164 / -0.24 / 35 | +0.264 / -0.63 / 6 |
