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
  - **Revised in §10 (lead-time curve):** on 1X2 alone the earliest-book EV@close is about +2%,
    not +9.5%. The +9.5% was carried by stale early O/U 2.5 prices. EV@close crosses 0 between
    T−45m and T−25m, but coverage, staleness and missing volume data leave that window
    unproven as bettable.
  - **§11–§12 (brief 03):** the early edge does *not* appear in the Premiership or
    Championship (EV@close < 0 from T−24h in both), so it is probably a thin-market artefact.
    Of six pre-registered price-movement hypotheses, only one confirmed: League Two prices
    partly revert between T−6h and T−60m. Given the close, movement predicts nothing about
    results, and no market-only rule clears commission.
  - **§13 (brief 04), 26/27 live season on the order book:** at the executable back price,
    grw_spfl_joint's L1/L2 bets lost to the close at every entry time (CLV −0.6 pp at T−60m and
    T−25m, CIs below 0). The same bets at mid or LTP show +0.05 to +0.2 pp and EV 0% to +3%, so
    the LTP-era edge fits inside the 4-tick spread. The live book opens only about 3 h out and
    offers a median £6.
  - **§16 (brief 07), half-time re-pricing:** the Betfair price during the half-time break beats
    every HT model on the full-time result. First-half BBC stats (pxG, on target, shots, corners)
    add nothing detectable on top of the HT score, or on top of the HT market itself. The best
    use of the model is weight 0.13 in a log-pool, which ties the market. Model-guided closing of a
    pre-match back does not beat holding.
  - **§17 (brief 08), is our score grid mis-weighted?** Only mildly. League One/Two narrow away
    wins (0-1, 1-2) are over-weighted and 1-1/2-2 under-weighted, in the market's direction, and
    total goals are less dispersed than our grids imply (variance ratio 0.86–0.91). No one-shape
    reshape fitted on held-out seasons fixes it out of sample, the market's own §14 shape makes
    exact scores worse, and none changes the betting picture.
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
little is left. *(This table pools all markets. §10 redoes it on 1X2 over 12 instants: the 1X2-only
earliest-book EV is about +2%.)* This is the most interesting result here, and it is the one with the weakest
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

## 10. Lead-time curve

Code: `l02_clv_leadtime.jl` and `r02_clv_leadtime.jl`. It reuses r01's `out/probs.jls`, so nothing is re-priced. The output is in `out/leadtime_*.csv` and `out/r02_run.log`.

**Answer.**
1. **CLV peaks early and decays into the off.** Across the five non-floor models (view a), mean 1X2 CLV at e = 2 is:
   - +0.7 to +0.9 pp at the earliest complete book (median about 26 h out);
   - +1.2 to +1.4 at T−12h, on n ≈ 70;
   - +0.4 to +0.8 from T−6h to T−45m;
   - +0.22 to +0.33 at T−25;
   - about 0 at T−10, which sits inside the close window.

   View (a) and the paired views b and b′ all show a second bump at T−60m, the floor included.
2. **EV@close at 2% crosses 0 between T−45m and T−25m.** Point estimates are ≥ 0 at every instant from earliest to T−45m, at +0.0% to +3.2%, then −0.3% to −0.6% at T−25 and −1.0% to −1.3% at T−10. The CI clears 0 only:
   - at earliest, for all five non-floor models (+1.9% to +2.7%, z 2.3–3.4);
   - at T−12h (n ≈ 70);
   - in scattered paired cells.
3. **The window doesn't look bettable yet.**
   - **Coverage:** before T−25 only 70–381 of 710 fixtures have a fresh, complete 1X2 book, against 520 at T−25.
   - **Staleness:** the median runner tick is 99 min old at T−24h and 241 min at T−48h. The earliest book is ungated LTP.
   - **Liquidity:** there is no volume field. Tick-count terciles show no thick-market concentration of the early edge; the edge sits in the *mid* tercile at T−6h.
4. **Multiple comparisons:** about 7 labels × 12 instants × 4 views, all on the same fixtures. Nothing in the EV@close > 0 column reaches z ≈ 3.5 (earliest max 3.4). What survives:
   - EV@close < 0 at T−10 (z −7 to −9);
   - model β given placebo > 0, with the CI above 0, for all five non-floor models at every instant from earliest to T−25 except T−48h. For the floor the CI touches 0 at most instants.
5. **Correction to §1 and §5.** On 1X2 the earliest-book EV is about +2%, not +9.5%. The +9.5% was an all-market figure and was carried by the non-1X2 markets, chiefly O/U 2.5. The earliest O/U 2.5 prices show +9% to +11% EV with CIs of about [0, +35], a heavy tail of stale early prices. See 10.4.

### 10.1 Setup and coverage

- **Entry:** the LTP at or before each instant.
  - Every runner's last tick must be ≤ max(90 min, 25% of the lead) old, the market complete, and the overround in [0.90, 1.10].
  - "Earliest" is r01's first instant at which every runner had traded (≤ T−25, no staleness gate).
- **Close:** unchanged, TWA (−20, 0] de-vigged. **T−10 overlaps the close window**, so its CLV is mechanically damped.
- **Models:** all six are read on one universe of 710 fixtures (56/57, 24/25–25/26). The two W2 co-finalists are re-read on their 24/25–25/26 subset.
- **Placebo:** the de-vigged TWA over [entry − 155, entry], which is r01's placebo at T−25. At "earliest" it is the per-market TWA over [t0 − 155, t0].
  - Before T−25 the placebo barely differs from the stale LTP, so it makes 0–17 bets at e ≥ 2. It is not a CLV control there.
  - It is still the control for β. Model β given placebo stays close to model β alone at every instant, so the early information is not LTP reversion.
- **Bets:** 1X2, e ≥ 2 pp, flat 1-unit backs, 2% commission on net per market, slate-bootstrap CIs (B = 2,000; β CIs B = 300).
- **⚠ marks m12_td rows before T−60.** m12 prices with the played XI, which isn't public before about T−60, so those rows are not clean.

| entry | staleness gate (min) | fixtures with a fresh complete 1X2 book (of 710) | median runner staleness (min) |
|---|---|---|---|
| earliest | none | 583 | — (median lead 25.9 h) |
| T−48h | 720 | 87 | 241 |
| T−24h | 360 | 166 | 99 |
| T−12h | 180 | 70 | 82 |
| T−6h | 90 | 175 | 26 |
| T−3h | 90 | 264 | 25 |
| T−2h | 90 | 287 | 27 |
| T−90m | 90 | 295 | 23 |
| T−60m | 90 | 319 | 19 |
| T−45m | 90 | 381 | 13 |
| T−25m | 90 | 520 | 4 |
| T−10m | 90 | 569 | 5 |

Coverage is not monotone: T−12h (overnight for most kick-offs) has the fewest fresh books. Each instant in view (a) is also a *different* fixture mix, which is why view (b) exists.

### 10.2 View (a): every selection priced at the instant

Each cell is **mean clv_pp / EV@close 2% (n bets)**. **Bold** means the EV@close CI is wholly above 0.

| entry | fixtures | lead | m05_joint_td | td_spfl_cups_joint | m12_td | td_lower_a2f_cj | grw_step_a2_cj | FLOOR | placebo |
|---|---|---|---|---|---|---|---|---|---|
| earliest | 583 | 25.9 h | **+0.69 / +2.0 (605)** | **+0.89 / +2.7 (601)** | **+0.86 / +2.5 (606)** ⚠ | **+0.78 / +2.3 (584)** | **+0.93 / +2.7 (596)** | +0.43 / +1.2 (621) | **+2.09 / +7.8 (75)** |
| T−48h | 87 | 48.0 h | +0.66 / +2.1 (84) | +0.44 / +1.7 (89) | +0.47 / +1.4 (83) ⚠ | +0.52 / +1.7 (80) | +0.99 / +3.2 (81) | -0.23 / -0.9 (85) | — (0) |
| T−24h | 166 | 24.0 h | +0.42 / +0.8 (179) | +0.56 / +1.4 (182) | +0.54 / +1.1 (178) ⚠ | +0.73 / +1.8 (172) | **+0.93 / +2.8 (168)** | +0.43 / +0.8 (184) | **+5.68 / +20.3 (7)** |
| T−12h | 70 | 12.0 h | **+1.43 / +3.2 (72)** | **+1.20 / +2.6 (75)** | **+1.39 / +3.1 (76)** ⚠ | **+1.18 / +2.7 (67)** | **+1.29 / +3.2 (74)** | +0.76 / +1.3 (74) | — (0) |
| T−6h | 175 | 6.0 h | +0.70 / +1.4 (185) | +0.51 / +0.9 (183) | +0.64 / +1.1 (182) ⚠ | +0.76 / +1.6 (178) | +0.66 / +1.4 (174) | +0.44 / +0.6 (177) | +0.52 / +0.1 (8) |
| T−3h | 264 | 3.0 h | +0.53 / +0.6 (256) | +0.53 / +0.7 (262) | +0.43 / +0.3 (269) ⚠ | +0.64 / +0.9 (262) | +0.43 / +0.5 (257) | +0.37 / +0.1 (269) | +1.10 / +2.8 (17) |
| T−2h | 287 | 2.0 h | +0.38 / +0.0 (296) | +0.59 / +0.7 (289) | +0.38 / +0.0 (290) ⚠ | +0.50 / +0.4 (286) | +0.44 / +0.3 (280) | +0.32 / -0.2 (295) | +1.02 / +1.8 (15) |
| T−90m | 295 | 90 m | +0.48 / +0.7 (303) | +0.45 / +0.5 (308) | +0.46 / +0.6 (303) ⚠ | +0.48 / +0.7 (300) | +0.38 / +0.5 (293) | +0.25 / -0.0 (296) | +1.40 / +3.6 (14) |
| T−60m | 319 | 60 m | +0.67 / +1.3 (334) | +0.56 / +1.0 (340) | +0.62 / +1.1 (337) | +0.71 / +1.4 (320) | +0.50 / +0.9 (316) | +0.39 / +0.5 (327) | +1.53 / +5.4 (10) |
| T−45m | 381 | 45 m | +0.51 / +0.5 (404) | +0.53 / +0.6 (398) | +0.54 / +0.6 (405) | +0.57 / +0.7 (392) | +0.42 / +0.2 (377) | +0.30 / -0.2 (403) | +0.56 / +0.2 (16) |
| T−25m | 520 | 25 m | +0.26 / -0.5 (529) | +0.33 / -0.3 (535) | +0.24 / -0.6 (532) | +0.26 / -0.5 (532) | +0.22 / -0.6 (511) | +0.15 / -0.9 (546) | +0.22 / -0.2 (149) |
| T−10m | 569 | 10 m | +0.05 / -1.2 (570) | +0.08 / -1.1 (571) | +0.02 / -1.3 (591) | +0.13 / -1.0 (558) | +0.06 / -1.2 (562) | +0.10 / -1.1 (593) | +0.18 / -0.7 (214) |

<details><summary>Full view-(a) tables per model: n, beat %, β alone and given placebo [CI], ROI [CI], staleness</summary>

**m05_joint_td**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 25.9 h | 583 | 605 | +0.690 [+0.303, +1.048] | 55.0 | 0.147 | 0.135 [+0.084, +0.180] | +1.95 [+0.20, +3.63] | +20.5 [+4.5, +36.4] | — |
| T−48h | 48.0 h | 87 | 84 | +0.662 [-0.336, +1.693] | 48.8 | 0.097 | 0.089 [-0.113, +0.397] | +2.09 [-2.76, +7.84] | +30.0 [-17.7, +85.0] | 241 |
| T−24h | 24.0 h | 166 | 179 | +0.416 [-0.141, +0.962] | 57.0 | 0.145 | 0.119 [+0.031, +0.244] | +0.84 [-1.23, +3.03] | -1.9 [-20.6, +17.7] | 99 |
| T−12h | 12.0 h | 70 | 72 | +1.431 [+0.712, +2.158] | 61.1 | 0.168 | 0.174 [+0.085, +0.311] | +3.23 [+0.62, +5.88] | -10.2 [-43.4, +21.4] | 82 |
| T−6h | 6.0 h | 175 | 185 | +0.701 [+0.287, +1.129] | 59.5 | 0.092 | 0.088 [+0.040, +0.133] | +1.39 [-0.35, +3.17] | +8.7 [-14.3, +32.4] | 26 |
| T−3h | 3.0 h | 264 | 256 | +0.528 [+0.166, +0.869] | 59.4 | 0.097 | 0.084 [+0.047, +0.121] | +0.58 [-0.88, +1.95] | +16.5 [-6.7, +40.5] | 25 |
| T−2h | 2.0 h | 287 | 296 | +0.380 [+0.026, +0.735] | 53.4 | 0.064 | 0.052 [+0.011, +0.090] | +0.01 [-1.36, +1.37] | +23.3 [+5.1, +41.1] | 27 |
| T−90m | 90 m | 295 | 303 | +0.481 [+0.085, +0.857] | 57.8 | 0.079 | 0.065 [+0.018, +0.111] | +0.67 [-1.01, +2.30] | +12.9 [-8.8, +36.3] | 23 |
| T−60m | 60 m | 319 | 334 | +0.672 [+0.324, +1.006] | 59.9 | 0.099 | 0.090 [+0.047, +0.133] | +1.30 [-0.21, +2.86] | +16.6 [-2.7, +35.9] | 19 |
| T−45m | 45 m | 381 | 404 | +0.514 [+0.219, +0.819] | 60.6 | 0.081 | 0.075 [+0.035, +0.109] | +0.50 [-0.73, +1.74] | +10.0 [-7.8, +27.3] | 13 |
| T−25m | 25 m | 520 | 529 | +0.258 [+0.112, +0.411] | 56.7 | 0.035 | 0.033 [+0.016, +0.051] | -0.53 [-1.21, +0.19] | +7.2 [-7.5, +21.5] | 4 |
| T−10m | 10 m | 569 | 570 | +0.051 [-0.017, +0.119] | 54.6 | 0.006 | 0.003 [-0.004, +0.009] | -1.21 [-1.50, -0.95] | +16.5 [+0.9, +32.3] | 5 |

**td_spfl_cups_joint**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 25.9 h | 583 | 601 | +0.891 [+0.566, +1.218] | 57.6 | 0.154 | 0.141 [+0.081, +0.191] | +2.72 [+1.12, +4.36] | +14.4 [-1.1, +30.8] | — |
| T−48h | 48.0 h | 87 | 89 | +0.443 [-0.463, +1.406] | 48.3 | 0.088 | 0.161 [-0.080, +0.408] | +1.66 [-2.92, +7.10] | +13.5 [-28.9, +57.7] | 241 |
| T−24h | 24.0 h | 166 | 182 | +0.559 [+0.025, +1.085] | 57.7 | 0.131 | 0.091 [+0.001, +0.199] | +1.36 [-0.58, +3.37] | -9.7 [-27.8, +9.1] | 99 |
| T−12h | 12.0 h | 70 | 75 | +1.197 [+0.582, +1.844] | 60.0 | 0.133 | 0.131 [+0.034, +0.293] | +2.64 [+0.17, +5.29] | -11.7 [-41.2, +19.6] | 82 |
| T−6h | 6.0 h | 175 | 183 | +0.512 [+0.073, +0.946] | 58.5 | 0.087 | 0.083 [+0.040, +0.129] | +0.92 [-0.86, +2.62] | +22.5 [-0.1, +47.6] | 26 |
| T−3h | 3.0 h | 264 | 262 | +0.534 [+0.159, +0.870] | 59.9 | 0.088 | 0.072 [+0.030, +0.116] | +0.74 [-0.71, +2.07] | +14.6 [-5.6, +33.3] | 25 |
| T−2h | 2.0 h | 287 | 289 | +0.590 [+0.222, +0.948] | 56.7 | 0.073 | 0.060 [+0.014, +0.105] | +0.66 [-0.83, +2.11] | +17.3 [+0.6, +32.9] | 27 |
| T−90m | 90 m | 295 | 308 | +0.449 [+0.048, +0.819] | 55.5 | 0.074 | 0.060 [+0.012, +0.103] | +0.52 [-1.20, +2.20] | +12.4 [-9.0, +31.8] | 23 |
| T−60m | 60 m | 319 | 340 | +0.559 [+0.229, +0.863] | 57.9 | 0.080 | 0.071 [+0.026, +0.108] | +0.95 [-0.57, +2.37] | +13.2 [-5.8, +31.8] | 19 |
| T−45m | 45 m | 381 | 398 | +0.534 [+0.253, +0.819] | 59.3 | 0.075 | 0.069 [+0.028, +0.105] | +0.56 [-0.66, +1.77] | +13.2 [-2.5, +29.6] | 13 |
| T−25m | 25 m | 520 | 535 | +0.326 [+0.163, +0.495] | 56.3 | 0.038 | 0.036 [+0.017, +0.054] | -0.33 [-1.05, +0.39] | +5.3 [-9.0, +19.6] | 4 |
| T−10m | 10 m | 569 | 571 | +0.083 [+0.011, +0.154] | 55.9 | 0.007 | 0.004 [-0.004, +0.010] | -1.12 [-1.41, -0.84] | +15.3 [+0.7, +29.7] | 5 |

**m12_td (live)**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest ⚠ | 25.9 h | 583 | 606 | +0.859 [+0.492, +1.243] | 56.9 | 0.127 | 0.116 [+0.074, +0.156] | +2.51 [+0.83, +4.21] | +18.2 [+3.8, +32.9] | — |
| T−48h ⚠ | 48.0 h | 87 | 83 | +0.468 [-0.527, +1.527] | 49.4 | 0.085 | 0.051 [-0.160, +0.277] | +1.42 [-3.40, +7.11] | +38.4 [-8.1, +91.7] | 241 |
| T−24h ⚠ | 24.0 h | 166 | 178 | +0.539 [-0.066, +1.149] | 59.0 | 0.129 | 0.094 [+0.001, +0.232] | +1.13 [-1.10, +3.43] | -0.2 [-19.7, +19.3] | 99 |
| T−12h ⚠ | 12.0 h | 70 | 76 | +1.394 [+0.671, +2.111] | 61.8 | 0.156 | 0.173 [+0.096, +0.287] | +3.13 [+0.58, +5.77] | -3.6 [-34.8, +27.1] | 82 |
| T−6h ⚠ | 6.0 h | 175 | 182 | +0.640 [+0.221, +1.069] | 57.1 | 0.093 | 0.090 [+0.046, +0.134] | +1.14 [-0.64, +2.93] | +17.5 [-4.5, +40.9] | 26 |
| T−3h ⚠ | 3.0 h | 264 | 269 | +0.434 [+0.070, +0.759] | 55.8 | 0.085 | 0.075 [+0.043, +0.110] | +0.26 [-1.16, +1.63] | +10.5 [-11.4, +33.0] | 25 |
| T−2h ⚠ | 2.0 h | 287 | 290 | +0.380 [+0.021, +0.732] | 55.2 | 0.060 | 0.049 [+0.011, +0.087] | +0.05 [-1.37, +1.41] | +19.5 [+0.9, +37.9] | 27 |
| T−90m ⚠ | 90 m | 295 | 303 | +0.459 [+0.078, +0.831] | 57.4 | 0.071 | 0.058 [+0.013, +0.102] | +0.60 [-1.02, +2.24] | +12.9 [-8.4, +35.7] | 23 |
| T−60m | 60 m | 319 | 337 | +0.615 [+0.265, +0.955] | 58.8 | 0.088 | 0.079 [+0.039, +0.118] | +1.11 [-0.37, +2.64] | +16.5 [-3.5, +36.6] | 19 |
| T−45m | 45 m | 381 | 405 | +0.536 [+0.237, +0.835] | 60.2 | 0.069 | 0.064 [+0.025, +0.097] | +0.57 [-0.65, +1.80] | +10.2 [-7.8, +28.5] | 13 |
| T−25m | 25 m | 520 | 532 | +0.244 [+0.089, +0.393] | 56.4 | 0.035 | 0.033 [+0.018, +0.049] | -0.57 [-1.26, +0.10] | +7.0 [-7.5, +21.1] | 4 |
| T−10m | 10 m | 569 | 591 | +0.020 [-0.052, +0.090] | 53.6 | 0.005 | 0.003 [-0.004, +0.009] | -1.30 [-1.59, -1.03] | +15.0 [+1.0, +28.9] | 5 |

**td_lower_a2full_carry_jump**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 25.9 h | 583 | 584 | +0.778 [+0.455, +1.105] | 57.2 | 0.153 | 0.140 [+0.080, +0.191] | +2.29 [+0.69, +3.95] | +18.2 [+1.5, +34.7] | — |
| T−48h | 48.0 h | 87 | 80 | +0.525 [-0.404, +1.483] | 50.0 | 0.108 | 0.067 [-0.206, +0.339] | +1.65 [-3.06, +7.45] | +24.8 [-22.6, +78.6] | 241 |
| T−24h | 24.0 h | 166 | 172 | +0.734 [+0.203, +1.307] | 60.5 | 0.152 | 0.125 [+0.034, +0.244] | +1.76 [-0.23, +3.92] | +3.3 [-18.0, +26.0] | 99 |
| T−12h | 12.0 h | 70 | 67 | +1.177 [+0.521, +1.832] | 59.7 | 0.155 | 0.190 [+0.095, +0.323] | +2.73 [+0.05, +5.35] | -17.2 [-51.5, +17.2] | 82 |
| T−6h | 6.0 h | 175 | 178 | +0.763 [+0.342, +1.201] | 60.7 | 0.101 | 0.096 [+0.041, +0.153] | +1.61 [-0.09, +3.40] | +19.8 [-7.3, +48.3] | 26 |
| T−3h | 3.0 h | 264 | 262 | +0.639 [+0.298, +0.970] | 60.3 | 0.104 | 0.088 [+0.043, +0.132] | +0.93 [-0.48, +2.26] | +21.4 [-1.2, +44.7] | 25 |
| T−2h | 2.0 h | 287 | 286 | +0.495 [+0.122, +0.867] | 54.9 | 0.064 | 0.048 [+0.002, +0.089] | +0.36 [-1.08, +1.80] | +18.0 [-0.5, +37.5] | 27 |
| T−90m | 90 m | 295 | 300 | +0.481 [+0.108, +0.837] | 56.3 | 0.075 | 0.058 [+0.005, +0.109] | +0.67 [-0.97, +2.28] | +12.7 [-8.9, +35.2] | 23 |
| T−60m | 60 m | 319 | 320 | +0.714 [+0.375, +1.062] | 60.0 | 0.094 | 0.083 [+0.034, +0.130] | +1.44 [-0.06, +3.10] | +17.4 [-3.1, +37.3] | 19 |
| T−45m | 45 m | 381 | 392 | +0.574 [+0.297, +0.854] | 60.5 | 0.071 | 0.064 [+0.020, +0.103] | +0.65 [-0.53, +1.85] | +12.3 [-6.6, +30.3] | 13 |
| T−25m | 25 m | 520 | 532 | +0.257 [+0.091, +0.424] | 55.8 | 0.032 | 0.030 [+0.009, +0.051] | -0.54 [-1.27, +0.18] | +8.9 [-5.4, +24.4] | 4 |
| T−10m | 10 m | 569 | 558 | +0.126 [+0.056, +0.200] | 56.8 | 0.009 | 0.005 [-0.002, +0.011] | -1.01 [-1.29, -0.73] | +12.6 [-4.7, +29.6] | 5 |

**grw_step_a2_carry_jump**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 25.9 h | 583 | 596 | +0.928 [+0.599, +1.281] | 58.4 | 0.219 | 0.203 [+0.148, +0.251] | +2.74 [+1.23, +4.58] | +15.0 [+0.4, +30.2] | — |
| T−48h | 48.0 h | 87 | 81 | +0.990 [+0.101, +1.923] | 56.8 | 0.157 | 0.136 [-0.114, +0.440] | +3.17 [-1.16, +8.72] | +14.2 [-25.0, +60.3] | 241 |
| T−24h | 24.0 h | 166 | 168 | +0.928 [+0.342, +1.508] | 63.7 | 0.183 | 0.152 [+0.050, +0.284] | +2.84 [+0.75, +5.04] | +0.1 [-20.1, +22.7] | 99 |
| T−12h | 12.0 h | 70 | 74 | +1.286 [+0.637, +1.887] | 59.5 | 0.158 | 0.195 [+0.126, +0.309] | +3.24 [+0.70, +5.70] | -25.8 [-54.3, +2.0] | 82 |
| T−6h | 6.0 h | 175 | 174 | +0.660 [+0.258, +1.054] | 59.2 | 0.121 | 0.116 [+0.060, +0.169] | +1.38 [-0.23, +2.98] | +18.3 [-2.3, +42.2] | 26 |
| T−3h | 3.0 h | 264 | 257 | +0.431 [+0.039, +0.804] | 59.1 | 0.092 | 0.075 [+0.032, +0.129] | +0.47 [-0.95, +1.84] | +24.4 [+3.1, +47.3] | 25 |
| T−2h | 2.0 h | 287 | 280 | +0.443 [+0.034, +0.800] | 57.9 | 0.065 | 0.052 [+0.005, +0.099] | +0.34 [-1.18, +1.65] | +22.1 [+3.4, +42.0] | 27 |
| T−90m | 90 m | 295 | 293 | +0.384 [-0.014, +0.748] | 56.7 | 0.083 | 0.065 [+0.010, +0.115] | +0.48 [-1.15, +2.09] | +16.4 [-6.1, +38.6] | 23 |
| T−60m | 60 m | 319 | 316 | +0.498 [+0.184, +0.801] | 57.9 | 0.083 | 0.070 [+0.024, +0.113] | +0.89 [-0.45, +2.33] | +26.4 [+5.2, +48.5] | 19 |
| T−45m | 45 m | 381 | 377 | +0.418 [+0.140, +0.676] | 59.4 | 0.064 | 0.055 [+0.015, +0.096] | +0.24 [-0.85, +1.28] | +19.9 [+2.4, +38.1] | 13 |
| T−25m | 25 m | 520 | 511 | +0.219 [+0.045, +0.384] | 54.2 | 0.041 | 0.039 [+0.014, +0.064] | -0.59 [-1.34, +0.09] | +6.1 [-9.0, +20.9] | 4 |
| T−10m | 10 m | 569 | 562 | +0.060 [-0.016, +0.132] | 54.4 | 0.007 | 0.001 [-0.008, +0.009] | -1.15 [-1.45, -0.87] | +11.1 [-3.9, +26.1] | 5 |

**FLOOR s12_m01_td_poisson**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 25.9 h | 583 | 621 | +0.434 [+0.059, +0.820] | 52.0 | 0.075 | 0.062 [+0.017, +0.097] | +1.18 [-0.41, +3.03] | +10.3 [-4.2, +25.8] | — |
| T−48h | 48.0 h | 87 | 85 | -0.234 [-1.246, +0.962] | 41.2 | 0.007 | -0.066 [-0.322, +0.246] | -0.94 [-5.57, +5.22] | +26.7 [-19.4, +80.9] | 241 |
| T−24h | 24.0 h | 166 | 184 | +0.433 [-0.119, +1.040] | 56.0 | 0.081 | 0.062 [-0.006, +0.161] | +0.80 [-1.32, +3.08] | +1.0 [-18.0, +22.4] | 99 |
| T−12h | 12.0 h | 70 | 74 | +0.756 [+0.077, +1.471] | 56.8 | 0.088 | 0.090 [+0.004, +0.199] | +1.25 [-1.41, +4.03] | -7.4 [-37.3, +21.9] | 82 |
| T−6h | 6.0 h | 175 | 177 | +0.437 [-0.005, +0.884] | 53.7 | 0.039 | 0.034 [-0.007, +0.072] | +0.60 [-1.17, +2.49] | +21.2 [-3.2, +46.1] | 26 |
| T−3h | 3.0 h | 264 | 269 | +0.367 [+0.014, +0.717] | 54.3 | 0.053 | 0.043 [+0.006, +0.080] | +0.05 [-1.40, +1.45] | +15.1 [-6.5, +36.7] | 25 |
| T−2h | 2.0 h | 287 | 295 | +0.325 [-0.063, +0.681] | 51.9 | 0.024 | 0.015 [-0.020, +0.051] | -0.19 [-1.67, +1.24] | +20.6 [+1.8, +38.9] | 27 |
| T−90m | 90 m | 295 | 296 | +0.254 [-0.128, +0.646] | 52.0 | 0.035 | 0.021 [-0.019, +0.065] | -0.04 [-1.73, +1.71] | +15.5 [-6.1, +37.5] | 23 |
| T−60m | 60 m | 319 | 327 | +0.392 [+0.050, +0.737] | 54.4 | 0.046 | 0.038 [+0.002, +0.073] | +0.47 [-1.00, +2.10] | +16.1 [-3.8, +36.0] | 19 |
| T−45m | 45 m | 381 | 403 | +0.297 [+0.022, +0.572] | 55.8 | 0.040 | 0.035 [-0.002, +0.066] | -0.15 [-1.30, +1.02] | +16.6 [-1.9, +35.1] | 13 |
| T−25m | 25 m | 520 | 546 | +0.146 [-0.005, +0.303] | 54.8 | 0.019 | 0.017 [+0.001, +0.033] | -0.86 [-1.56, -0.15] | +6.0 [-8.8, +21.1] | 4 |
| T−10m | 10 m | 569 | 593 | +0.097 [+0.031, +0.162] | 55.0 | 0.007 | 0.004 [-0.001, +0.009] | -1.11 [-1.37, -0.85] | +10.2 [-5.6, +26.0] | 5 |

**PLACEBO TWA[entry−155, entry]**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 25.9 h | 580 | 75 | +2.090 [+1.053, +3.110] | 62.7 | 0.492 | — | +7.84 [+2.16, +14.64] | -11.9 [-45.7, +23.8] | — |
| T−48h | 48.0 h | 25 | 0 | — | — | -0.913 | — | — | — | 74 |
| T−24h | 24.0 h | 82 | 7 | +5.679 [+2.339, +9.388] | 100.0 | 1.177 | — | +20.27 [+8.65, +34.44] | -61.9 [-100.0, +14.3] | 61 |
| T−12h | 12.0 h | 54 | 0 | — | — | 2.936 | — | — | — | 62 |
| T−6h | 6.0 h | 175 | 8 | +0.523 [-1.228, +2.262] | 50.0 | 0.311 | — | +0.09 [-4.61, +4.54] | -2.1 [-100.0, +95.8] | 26 |
| T−3h | 3.0 h | 264 | 17 | +1.103 [+0.234, +2.161] | 76.5 | 0.802 | — | +2.84 [-0.19, +6.76] | +39.2 [-35.9, +127.2] | 25 |
| T−2h | 2.0 h | 287 | 15 | +1.020 [-0.246, +2.258] | 73.3 | 0.831 | — | +1.83 [-1.50, +5.04] | -13.1 [-71.4, +57.2] | 27 |
| T−90m | 90 m | 295 | 14 | +1.400 [+0.172, +2.553] | 78.6 | 0.803 | — | +3.59 [-0.32, +7.38] | +50.1 [-46.2, +151.8] | 23 |
| T−60m | 60 m | 319 | 10 | +1.529 [-0.195, +3.496] | 70.0 | 0.594 | — | +5.39 [-1.07, +12.91] | +14.3 [-100.0, +118.9] | 19 |
| T−45m | 45 m | 381 | 16 | +0.563 [-0.767, +2.102] | 62.5 | 0.389 | — | +0.19 [-4.09, +5.22] | -32.6 [-87.0, +34.7] | 13 |
| T−25m | 25 m | 520 | 149 | +0.223 [-0.069, +0.537] | 53.0 | 0.063 | — | -0.21 [-1.31, +0.87] | -1.9 [-22.4, +19.9] | 4 |
| T−10m | 10 m | 569 | 214 | +0.178 [+0.072, +0.291] | 57.0 | 0.057 | — | -0.65 [-1.02, -0.25] | -13.5 [-30.3, +4.0] | 5 |

</details>

- **Every model, the floor included, shows the same shape.** The five joint/pxG/pooled models sit above the floor at most instants, by 0.1–0.9 pp.
- **β alone falls steadily:** 0.13–0.22 at earliest (floor 0.08), 0.06–0.10 from T−3h to T−45m, 0.03–0.04 at T−25 and under 0.01 at T−10. The later the entry, the less of the model's disagreement the market still has to close.
- **Realised ROI is noise.** It is −26% to +38% on 67–620 bets, and most CIs span 0.

### 10.3 View (b): paired selections

**(b) as specified.** These are the 1X2 selections that have a fresh complete book at T−24h, T−6h, T−2h, T−60m *and* T−25m. The fixtures fresh at T−24h and those fresh at T−6h barely overlap, so this is only 90 selections (30 fixtures, about 30 bets per cell). The earliest book is shown on the same set.

| entry | fixtures | lead | m05_joint_td | td_spfl_cups_joint | m12_td | td_lower_a2f_cj | grw_step_a2_cj | FLOOR | placebo |
|---|---|---|---|---|---|---|---|---|---|
| earliest | 30 | 43.5 h | +0.41 / +1.4 (30) | +0.77 / +3.0 (29) | +1.04 / +3.2 (26) ⚠ | +0.58 / +1.9 (26) | +0.87 / +3.0 (31) | -0.18 / -0.5 (28) | — (1) |
| T−24h | 30 | 24.0 h | +0.08 / -0.3 (32) | +0.72 / +1.5 (30) | +0.29 / +0.0 (29) ⚠ | +0.66 / +1.3 (31) | +0.68 / +1.6 (32) | +0.07 / -0.5 (31) | — (1) |
| T−6h | 30 | 6.0 h | +0.80 / +1.3 (30) | +0.82 / +1.3 (30) | +0.91 / +1.5 (29) ⚠ | +1.08 / +2.1 (26) | +1.07 / +2.2 (28) | +0.24 / -0.4 (28) | — (1) |
| T−2h | 30 | 2.0 h | +0.51 / +0.6 (29) | +0.52 / +0.7 (33) | +0.58 / +0.6 (30) ⚠ | +0.69 / +1.0 (31) | +0.72 / +1.3 (28) | +0.45 / +0.3 (29) | — (3) |
| T−60m | 30 | 60 m | **+1.34 / +3.6 (30)** | **+1.16 / +3.0 (32)** | **+1.19 / +3.1 (30)** | **+1.45 / +3.8 (31)** | **+1.49 / +3.9 (31)** | +0.95 / +2.5 (27) | — (2) |
| T−25m | 30 | 25 m | +0.68 / +1.2 (29) | +1.01 / +1.8 (31) | +0.59 / +0.9 (28) | +0.58 / +0.7 (31) | +0.58 / +0.8 (28) | +0.37 / +0.3 (27) | -0.02 / -1.2 (6) |

<details><summary>Full (b) tables per model</summary>

**m05_joint_td**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 43.5 h | 30 | 30 | +0.409 [-1.366, +2.182] | 56.7 | 0.188 | 0.168 [-0.135, +0.443] | +1.42 [-5.27, +9.02] | -18.4 [-60.4, +28.8] | — |
| T−24h | 24.0 h | 30 | 32 | +0.082 [-1.329, +1.316] | 62.5 | 0.175 | 0.294 [+0.101, +0.457] | -0.32 [-5.13, +3.91] | -3.6 [-42.6, +35.1] | 84 |
| T−6h | 6.0 h | 30 | 30 | +0.802 [-0.087, +1.676] | 60.0 | 0.120 | 0.116 [-0.022, +0.244] | +1.31 [-1.67, +4.02] | -42.8 [-84.0, +4.2] | 21 |
| T−2h | 2.0 h | 30 | 29 | +0.509 [-0.479, +1.453] | 58.6 | 0.099 | 0.095 [-0.029, +0.204] | +0.62 [-2.73, +3.74] | -37.9 [-81.9, +11.9] | 19 |
| T−60m | 60 m | 30 | 30 | +1.340 [+0.608, +2.134] | 83.3 | 0.178 | 0.144 [+0.067, +0.241] | +3.64 [+1.03, +6.48] | -26.4 [-72.3, +26.5] | 14 |
| T−25m | 25 m | 30 | 29 | +0.679 [-0.051, +1.459] | 65.5 | 0.102 | 0.102 [+0.026, +0.208] | +1.24 [-1.54, +4.39] | -30.3 [-74.9, +20.4] | 3 |

**td_spfl_cups_joint**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 43.5 h | 30 | 29 | +0.767 [-0.707, +2.561] | 58.6 | 0.165 | 0.149 [-0.123, +0.406] | +2.98 [-2.86, +10.85] | -33.6 [-70.8, +13.3] | — |
| T−24h | 24.0 h | 30 | 30 | +0.723 [-0.464, +1.956] | 63.3 | 0.158 | 0.185 [-0.136, +0.415] | +1.49 [-2.26, +5.29] | -15.4 [-58.2, +25.9] | 84 |
| T−6h | 6.0 h | 30 | 30 | +0.824 [-0.081, +1.704] | 66.7 | 0.114 | 0.113 [-0.014, +0.226] | +1.35 [-1.68, +3.99] | -11.7 [-56.5, +43.2] | 21 |
| T−2h | 2.0 h | 30 | 33 | +0.522 [-0.321, +1.271] | 57.6 | 0.079 | 0.077 [-0.048, +0.197] | +0.69 [-2.03, +3.02] | -13.0 [-53.6, +34.2] | 19 |
| T−60m | 60 m | 30 | 32 | +1.164 [+0.312, +1.984] | 81.2 | 0.136 | 0.106 [+0.003, +0.214] | +2.97 [+0.08, +5.78] | -20.3 [-65.3, +31.8] | 14 |
| T−25m | 25 m | 30 | 31 | +1.014 [+0.333, +1.827] | 71.0 | 0.101 | 0.090 [+0.016, +0.206] | +1.84 [-0.63, +4.84] | -20.2 [-63.4, +28.6] | 3 |

**m12_td (live)**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest ⚠ | 43.5 h | 30 | 26 | +1.037 [-0.890, +2.998] | 65.4 | 0.219 | 0.203 [-0.057, +0.445] | +3.23 [-4.04, +11.56] | -30.0 [-73.0, +23.8] | — |
| T−24h ⚠ | 24.0 h | 30 | 29 | +0.292 [-1.066, +1.545] | 65.5 | 0.199 | 0.329 [+0.156, +0.487] | +0.00 [-4.46, +4.32] | -23.3 [-63.5, +15.8] | 84 |
| T−6h ⚠ | 6.0 h | 30 | 29 | +0.912 [+0.024, +1.795] | 62.1 | 0.135 | 0.131 [+0.002, +0.252] | +1.49 [-1.56, +4.25] | -27.5 [-70.6, +19.6] | 21 |
| T−2h ⚠ | 2.0 h | 30 | 30 | +0.578 [-0.256, +1.322] | 56.7 | 0.103 | 0.100 [-0.014, +0.210] | +0.65 [-2.25, +3.27] | -40.0 [-82.1, +7.3] | 19 |
| T−60m | 60 m | 30 | 30 | +1.194 [+0.481, +2.008] | 80.0 | 0.184 | 0.152 [+0.079, +0.260] | +3.06 [+0.57, +5.94] | -37.9 [-80.0, +11.2] | 14 |
| T−25m | 25 m | 30 | 28 | +0.590 [-0.167, +1.429] | 64.3 | 0.104 | 0.105 [+0.030, +0.214] | +0.88 [-2.02, +4.22] | -51.5 [-89.7, -2.8] | 3 |

**td_lower_a2full_carry_jump**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 43.5 h | 30 | 26 | +0.578 [-1.358, +2.546] | 61.5 | 0.178 | 0.159 [-0.199, +0.422] | +1.89 [-5.65, +10.41] | -31.9 [-77.7, +27.8] | — |
| T−24h | 24.0 h | 30 | 31 | +0.656 [-0.665, +1.780] | 67.7 | 0.163 | 0.247 [+0.016, +0.448] | +1.35 [-3.19, +5.37] | +16.5 [-31.0, +76.1] | 84 |
| T−6h | 6.0 h | 30 | 26 | +1.079 [+0.260, +1.872] | 69.2 | 0.124 | 0.120 [-0.019, +0.250] | +2.14 [-1.05, +4.88] | -25.1 [-73.6, +35.5] | 21 |
| T−2h | 2.0 h | 30 | 31 | +0.693 [-0.105, +1.426] | 61.3 | 0.104 | 0.100 [-0.009, +0.217] | +1.05 [-1.77, +3.56] | -11.9 [-58.1, +43.5] | 19 |
| T−60m | 60 m | 30 | 31 | +1.446 [+0.832, +2.186] | 83.9 | 0.179 | 0.143 [+0.073, +0.228] | +3.83 [+1.59, +6.40] | -9.0 [-58.2, +46.0] | 14 |
| T−25m | 25 m | 30 | 31 | +0.581 [-0.128, +1.382] | 61.3 | 0.092 | 0.095 [+0.027, +0.185] | +0.69 [-2.04, +3.84] | -19.0 [-62.9, +35.2] | 3 |

**grw_step_a2_carry_jump**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 43.5 h | 30 | 31 | +0.867 [-0.960, +2.673] | 64.5 | 0.179 | 0.163 [-0.058, +0.355] | +3.03 [-3.48, +10.28] | -3.9 [-51.1, +55.2] | — |
| T−24h | 24.0 h | 30 | 32 | +0.681 [-0.676, +1.806] | 68.8 | 0.171 | 0.199 [+0.049, +0.308] | +1.63 [-2.47, +5.18] | +13.1 [-35.3, +63.8] | 84 |
| T−6h | 6.0 h | 30 | 28 | +1.068 [+0.228, +1.835] | 75.0 | 0.136 | 0.133 [+0.039, +0.214] | +2.19 [-0.77, +4.68] | -9.0 [-55.5, +48.9] | 21 |
| T−2h | 2.0 h | 30 | 28 | +0.721 [-0.100, +1.371] | 64.3 | 0.109 | 0.110 [+0.026, +0.195] | +1.31 [-1.56, +3.53] | -4.4 [-55.5, +58.8] | 19 |
| T−60m | 60 m | 30 | 31 | +1.492 [+0.951, +2.213] | 87.1 | 0.161 | 0.135 [+0.078, +0.197] | +3.93 [+1.92, +6.39] | +9.5 [-36.3, +65.0] | 14 |
| T−25m | 25 m | 30 | 28 | +0.585 [-0.167, +1.452] | 67.9 | 0.087 | 0.094 [+0.034, +0.165] | +0.82 [-2.07, +4.32] | -6.1 [-55.6, +49.1] | 3 |

**FLOOR s12_m01_td_poisson**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 43.5 h | 30 | 28 | -0.176 [-2.149, +1.564] | 50.0 | 0.058 | 0.035 [-0.223, +0.270] | -0.52 [-8.26, +7.27] | -38.3 [-84.2, +18.9] | — |
| T−24h | 24.0 h | 30 | 31 | +0.066 [-1.530, +1.404] | 61.3 | 0.078 | 0.132 [-0.051, +0.308] | -0.52 [-5.92, +4.08] | +3.8 [-39.8, +57.1] | 84 |
| T−6h | 6.0 h | 30 | 28 | +0.238 [-0.629, +1.048] | 53.6 | 0.038 | 0.031 [-0.086, +0.112] | -0.43 [-3.76, +2.43] | -22.1 [-75.6, +48.2] | 21 |
| T−2h | 2.0 h | 30 | 29 | +0.453 [-0.512, +1.259] | 55.2 | 0.036 | 0.032 [-0.074, +0.114] | +0.26 [-3.24, +3.03] | -0.6 [-55.5, +63.1] | 19 |
| T−60m | 60 m | 30 | 27 | +0.954 [+0.078, +1.872] | 74.1 | 0.112 | 0.076 [-0.010, +0.143] | +2.48 [-0.67, +5.61] | -13.6 [-72.8, +59.9] | 14 |
| T−25m | 25 m | 30 | 27 | +0.373 [-0.408, +1.204] | 55.6 | 0.054 | 0.051 [-0.009, +0.125] | +0.28 [-2.85, +3.63] | -25.8 [-75.1, +26.2] | 3 |

**PLACEBO TWA[entry−155, entry]**

| entry | median lead | fixtures | n bets | clv_pp [95% CI] | beat % | β alone | β given placebo [CI] | EV@close 2% [CI] | ROI 2% [CI] | stale med (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| earliest | 46.1 h | 19 | 1 | — | — | 0.381 | — | — | — | — |
| T−24h | 24.0 h | 19 | 1 | — | — | 0.439 | — | — | — | 51 |
| T−6h | 6.0 h | 19 | 1 | — | — | 0.585 | — | — | — | 23 |
| T−2h | 2.0 h | 19 | 3 | — | — | 0.699 | — | — | — | 27 |
| T−60m | 60 m | 19 | 2 | — | — | 0.988 | — | — | — | 11 |
| T−25m | 25 m | 19 | 6 | -0.019 [-1.214, +1.434] | 33.3 | 0.408 | — | -1.24 [-3.92, +2.05] | -17.3 [-100.0, +71.5] | 3 |

</details>

**Looser pairings (added because (b) is thin; not in the brief).** b′ drops T−24h (219 selections, 73 fixtures). b″ drops T−6h and T−2h (267 selections, 89 fixtures).

| entry | fixtures | lead | m05_joint_td | td_spfl_cups_joint | m12_td | td_lower_a2f_cj | grw_step_a2_cj | FLOOR | placebo |
|---|---|---|---|---|---|---|---|---|---|
| earliest | 72 | 29.6 h | **+1.76 / +7.6 (72)** | **+1.91 / +8.3 (73)** | **+2.02 / +8.3 (70)** ⚠ | **+2.18 / +9.2 (65)** | **+1.47 / +6.5 (75)** | +1.37 / +6.3 (73) | +3.47 / +29.8 (9) |
| T−6h | 73 | 6.0 h | **+0.99 / +2.6 (74)** | +0.82 / +2.1 (75) | +0.95 / +2.4 (74) ⚠ | **+1.11 / +3.0 (67)** | +0.73 / +1.7 (72) | +0.60 / +1.4 (72) | — (3) |
| T−2h | 73 | 2.0 h | +0.68 / +1.1 (70) | +0.61 / +1.0 (72) | +0.75 / +1.2 (72) ⚠ | +1.00 / +2.0 (68) | +0.46 / +0.6 (68) | +0.56 / +0.7 (73) | +0.37 / +0.3 (5) |
| T−60m | 73 | 60 m | **+1.14 / +2.8 (73)** | +0.93 / +2.1 (72) | **+1.14 / +2.6 (77)** | **+1.55 / +3.9 (68)** | **+0.94 / +2.0 (74)** | +0.98 / +2.2 (72) | +1.63 / +3.8 (5) |
| T−25m | 73 | 25 m | +0.42 / +0.4 (71) | +0.69 / +1.1 (74) | +0.45 / +0.5 (73) | +0.38 / +0.2 (72) | +0.22 / -0.1 (72) | +0.18 / -0.4 (74) | +0.22 / -0.8 (20) |

| entry | fixtures | lead | m05_joint_td | td_spfl_cups_joint | m12_td | td_lower_a2f_cj | grw_step_a2_cj | FLOOR | placebo |
|---|---|---|---|---|---|---|---|---|---|
| earliest | 89 | 37.9 h | +0.83 / +2.1 (87) | +0.76 / +2.3 (87) | +0.89 / +2.3 (86) ⚠ | +0.63 / +1.6 (83) | +1.11 / +3.2 (90) | +0.18 / +0.5 (84) | +1.62 / +8.9 (7) |
| T−24h | 89 | 24.0 h | +0.39 / +0.9 (95) | +0.62 / +1.7 (96) | +0.30 / +0.5 (93) ⚠ | +0.85 / +2.2 (95) | +0.94 / +3.0 (92) | +0.44 / +0.9 (99) | **+6.52 / +22.7 (5)** |
| T−60m | 89 | 60 m | +0.78 / +1.4 (88) | +0.58 / +0.9 (91) | +0.62 / +0.9 (85) | +0.55 / +0.8 (86) | +0.59 / +1.0 (88) | +0.24 / -0.2 (81) | — (3) |
| T−25m | 89 | 25 m | +0.43 / +0.2 (89) | +0.52 / +0.4 (92) | +0.45 / +0.2 (86) | +0.34 / -0.1 (90) | +0.27 / -0.2 (84) | +0.29 / -0.2 (88) | -0.16 / -1.5 (15) |

**Paired chart.** m05_joint_td, grw_step_a2_carry_jump and the floor on one common scale per panel and metric. Numbers are 1X2, e = 2.

```
(b) paired T−24h/T−6h/T−2h/T−60m/T−25m — 90 selections, ~30 bets per cell
entry          mean clv_pp (common scale)                EV@close 2%, % (common scale)
earliest  m05              │███          +0.41               │████         +1.4
          grw              │███████      +0.87               │█████████    +3.0
          FLR             █│             -0.18             ██│             -0.5

T−24h     m05              │█            +0.08              █│             -0.3
          grw              │█████        +0.68               │█████        +1.6
          FLR              │█            +0.07             ██│             -0.5

T−6h      m05              │██████       +0.80               │████         +1.3
          grw              │█████████    +1.07               │███████      +2.2
          FLR              │██           +0.24              █│             -0.4

T−2h      m05              │████         +0.51               │██           +0.6
          grw              │██████       +0.72               │████         +1.3
          FLR              │████         +0.45               │█            +0.3

T−60m     m05              │███████████  +1.34               │███████████  +3.6
          grw              │████████████ +1.49               │████████████ +3.9
          FLR              │████████     +0.95               │████████     +2.5

T−25m     m05              │█████        +0.68               │████         +1.2
          grw              │█████        +0.58               │███          +0.8
          FLR              │███          +0.37               │█            +0.3

(b′) paired T−6h/T−2h/T−60m/T−25m — 219 selections, ~70 bets per cell
entry          mean clv_pp (common scale)                EV@close 2%, % (common scale)
earliest  m05              │████████████ +1.76               │████████████ +7.6
          grw              │██████████   +1.47               │██████████   +6.5
          FLR              │█████████    +1.37               │██████████   +6.3

T−6h      m05              │███████      +0.99               │████         +2.6
          grw              │█████        +0.73               │███          +1.7
          FLR              │████         +0.60               │██           +1.4

T−2h      m05              │█████        +0.68               │██           +1.1
          grw              │███          +0.46               │█            +0.6
          FLR              │████         +0.56               │█            +0.7

T−60m     m05              │████████     +1.14               │████         +2.8
          grw              │██████       +0.94               │███          +2.0
          FLR              │███████      +0.98               │████         +2.2

T−25m     m05              │███          +0.42               │█            +0.4
          grw              │██           +0.22               │             -0.1
          FLR              │█            +0.18              █│             -0.4
```

- **T−60m beats T−25m on the same selections** in all three paired views, for every non-floor model. No other ordering is consistent across pairings; T−24h is the low point in (b) for m05, m12 and the floor.
- **T−60m is a local peak.** It sits above T−2h and T−25m in b and b′ for every model, the floor included, and view (a) shows the same (T−90m < T−60m > T−45m).
  - Since the floor has it too, it is a property of the T−60m price (median staleness 19 min against 4 at T−25), not of lineup information. Only m12 uses the XI, and m12 is clean from T−60 on.
- **Earliest vs T−24h depends on the fixture set.** It is +1.5 to +2.2 pp CLV and +6% to +9% EV in b′, but only +0.2 to +1.1 pp and +0.5% to +3.2% in b″. Early-entry EV is fragile to which fixtures happen to have early prices.

### 10.4 Other markets (view a, kept only where n bets ≥ 100)

Only O/U 2.5 and O/U 3.5 (plus one BTTS and one O/U 1.5 cell) reach 100 bets, and only at earliest and T−60m…T−10m. Full table: `out/leadtime_other_markets.csv`.

| market | earliest: clv_pp · EV@close 2% [CI] (n) | T−25m: clv_pp · EV@close 2% (n) | T−10m: clv_pp · EV@close 2% |
|---|---|---|---|
| O/U 2.5 | +0.90 to +1.23 · **+9.1% to +11.5%** [−0.5 to +0.6, +27 to +37] (208–272) | −0.09 to +0.07 · −0.9% to −1.3% (127–178) | −0.05 to −0.13 · −1.2% to −1.35% |
| O/U 3.5 | +0.54 to +0.88 · +0.5% to +1.3% [≈−1, ≈+2.5] (129–168) | n < 100 | −0.00 to −0.14 · −0.7% to −1.3% |

The earliest O/U 2.5 EV has a point of about +10% and a CI upper bound of +27% to +37%. Mean entry odds are about 2.4 and the hit rate is 49–53%, so the mean is carried by a heavy tail of stale early prices. This, not 1X2, is what lifted r01's all-market +9.5% earliest-book headline (§5). O/U 2.5 has no CLV from T−60m onwards, which matches Table B.

### 10.5 Liquidity (view a, 1X2, e = 2)

**`betfair.odds_history` has no volume field.** Its columns are `id`, `match_id`, `market_id` and `odds_data`. `odds_data` holds only a `timestamps` array and one LTP array per selection, and no `betfair.*` column carries matched volume or size. Liquidity is therefore proxied by **tick count**: the price rows across the market's runners up to the instant. Terciles are cut over the 710-fixture universe's 1X2 markets with a fresh book at that instant. Each cell is **n bets · clv_pp [CI] · EV@close 2%**.

**T−24h** — ticks terciles over the universe's 1X2 markets (cuts 9 / 15)

| model | thin: n · clv_pp [CI] · EV@close 2% | mid: n · clv_pp [CI] · EV@close 2% | thick: n · clv_pp [CI] · EV@close 2% |
|---|---|---|---|
| m05_joint_td | 70 · +0.61 [-0.42, +1.59] · +1.1 | 50 · -0.04 [-1.16, +0.90] · -0.0 | 59 · +0.57 [-0.38, +1.66] · +1.3 |
| td_spfl_cups_joint | 68 · +0.77 [-0.21, +1.71] · +1.5 | 50 · -0.09 [-1.13, +0.97] · -0.2 | 64 · +0.85 [-0.07, +1.68] · +2.5 |
| m12_td (live) | 70 · +0.72 [-0.29, +1.74] · +1.2 | 44 · +0.26 [-0.91, +1.48] · +0.9 | 64 · +0.53 [-0.29, +1.47] · +1.2 |
| td_lower_a2full_carry_jump | 67 · +0.97 [-0.10, +1.91] · +2.0 | 51 · +0.36 [-0.66, +1.49] · +1.1 | 54 · +0.80 [-0.09, +1.85] · +2.2 |
| grw_step_a2_carry_jump | 68 · +0.61 [-0.49, +1.66] · +1.1 | 46 · +0.79 [-0.42, +1.87] · +3.1 | 54 · +1.45 [+0.60, +2.40] · +4.7 |
| FLOOR s12_m01_td_poisson | 69 · +0.43 [-0.76, +1.50] · +0.5 | 52 · +0.42 [-0.53, +1.44] · +1.2 | 63 · +0.44 [-0.32, +1.36] · +0.9 |
| PLACEBO TWA[entry−155, entry] | 0 · — · — | 1 · — · — | 6 · +4.35 [+1.97, +8.18] · +17.8 |

**T−6h** — ticks terciles over the universe's 1X2 markets (cuts 20 / 32)

| model | thin: n · clv_pp [CI] · EV@close 2% | mid: n · clv_pp [CI] · EV@close 2% | thick: n · clv_pp [CI] · EV@close 2% |
|---|---|---|---|
| m05_joint_td | 64 · +0.56 [-0.26, +1.44] · +1.5 | 60 · +1.13 [+0.30, +2.04] · +2.8 | 61 · +0.42 [-0.26, +1.02] · -0.1 |
| td_spfl_cups_joint | 62 · +0.15 [-0.69, +1.02] · +0.6 | 57 · +1.10 [+0.30, +2.03] · +2.6 | 64 · +0.34 [-0.34, +0.94] · -0.2 |
| m12_td (live) | 64 · +0.49 [-0.33, +1.34] · +1.2 | 57 · +1.04 [+0.18, +1.99] · +2.3 | 61 · +0.42 [-0.25, +1.00] · -0.1 |
| td_lower_a2full_carry_jump | 62 · +0.66 [-0.08, +1.45] · +1.8 | 58 · +1.12 [+0.22, +2.12] · +2.6 | 58 · +0.51 [-0.16, +1.11] · +0.3 |
| grw_step_a2_carry_jump | 61 · +0.48 [-0.27, +1.20] · +1.7 | 49 · +0.91 [-0.00, +1.98] · +1.7 | 64 · +0.64 [+0.08, +1.19] · +0.8 |
| FLOOR s12_m01_td_poisson | 63 · +0.13 [-0.77, +0.99] · +0.2 | 59 · +1.13 [+0.25, +2.09] · +2.7 | 55 · +0.05 [-0.66, +0.69] · -1.2 |
| PLACEBO TWA[entry−155, entry] | 0 · — · — | 3 · — · — | 5 · +0.08 [-1.61, +2.09] · -1.0 |

**T−25m** — ticks terciles over the universe's 1X2 markets (cuts 35 / 51)

| model | thin: n · clv_pp [CI] · EV@close 2% | mid: n · clv_pp [CI] · EV@close 2% | thick: n · clv_pp [CI] · EV@close 2% |
|---|---|---|---|
| m05_joint_td | 182 · +0.21 [-0.05, +0.48] · -0.7 | 168 · +0.18 [-0.04, +0.42] · -1.0 | 179 · +0.38 [+0.12, +0.64] · +0.1 |
| td_spfl_cups_joint | 184 · +0.22 [-0.07, +0.55] · -0.7 | 174 · +0.21 [-0.07, +0.49] · -0.8 | 177 · +0.55 [+0.30, +0.81] · +0.6 |
| m12_td (live) | 186 · +0.16 [-0.10, +0.44] · -0.8 | 173 · +0.15 [-0.10, +0.40] · -1.1 | 173 · +0.43 [+0.18, +0.70] · +0.3 |
| td_lower_a2full_carry_jump | 182 · +0.22 [-0.05, +0.51] · -0.7 | 171 · +0.22 [-0.07, +0.49] · -0.9 | 179 · +0.33 [+0.07, +0.61] · -0.1 |
| grw_step_a2_carry_jump | 180 · +0.13 [-0.16, +0.41] · -0.9 | 166 · +0.15 [-0.13, +0.44] · -1.0 | 165 · +0.38 [+0.11, +0.65] · +0.2 |
| FLOOR s12_m01_td_poisson | 187 · +0.08 [-0.18, +0.37] · -1.1 | 177 · +0.04 [-0.22, +0.31] · -1.4 | 182 · +0.31 [+0.08, +0.57] · -0.1 |
| PLACEBO TWA[entry−155, entry] | 46 · +0.20 [-0.31, +0.86] · -0.2 | 42 · +0.34 [-0.24, +0.94] · +0.4 | 61 · +0.15 [-0.32, +0.62] · -0.7 |

- **T−25:** CLV and EV@close concentrate in the thickest tercile (+0.31 to +0.55 pp, EV −0.1% to +0.6%), and the thin/mid terciles are negative EV (−0.7% to −1.4%). More ticks means more trading, and our disagreements are more often right there.
- **T−6h:** the edge is in the *mid* tercile (+0.9 to +1.1 pp, EV +1.7% to +2.8%). The thick tercile is about 0 EV (−1.2% to +0.8%).
- **T−24h:** no consistent gradient. Only grw_step_a2_carry_jump's thick tercile has a CLV CI above 0.
- **Ticks are not matched money.** A 7-tick market at T−24h may have traded a few pounds. None of this shows that a meaningful stake would have been matched at these prices.

### 10.6 What would make the early window bettable

1. **Real prices and size.** The archive is LTP only. A back-price-and-size snapshot (the live `betfair_live` ladder, or Betfair's Advanced historical tier) at T−6h and T−60m would turn EV@close into an executable number.
2. **A pre-registered test** of one instant chosen from this curve on new fixtures, e.g. T−60m. The T−60m bump is consistent across models and pairings but is one of about 12 instants.
3. **A lineup-free early model.** Only m12 is flagged ⚠, and the lineup-free models show the same curve, so this is already mostly satisfied.

**Reproduce:** on mcmc-beast, after r01 has written `out/probs.jls`, run
`nice -n 19 /root/.juliaup/bin/julia --project -t 8 current_development/clv_napkin/r02_clv_leadtime.jl`. It takes about 5–10 min.

## 11. Lead time on 54/55

Code: `r03_leadtime_monitor.jl`, which reuses the l02 definitions unchanged (loaded through `l03_price_movement.jl`). It reuses r01's `out/probs.jls` (`probs_monitor`), so nothing is re-priced. Output: `out/r03a_*.csv` and `out/r03a_run.log`.

**Answer.** No: the early-entry 1X2 edge does not carry over to the Premiership or the Championship, so the League One/Two early edge of §10 should be read as a thin-market artefact until ladder data says otherwise.
- **EV@close is negative everywhere after the first day.** Across the four pooled fits' held-out 54/55 latents, EV@close at 2% is negative at every instant from T−24h to T−10m in both leagues: 80 of 80 model × instant cells, ranging from −0.1% to −5.2%. The CI is wholly below 0 in 39 of 40 Premiership cells and 24 of 40 Championship cells.
- **The only positive points** are the two GRW fits at the Premiership's earliest and T−48h books (+0.1% to +2.3%, CIs spanning 0). Those books are about 75 h out, on ungated LTPs.
- **The Premiership has already priced what the models know.** From T−12h on, CLV is −0.10 to +0.20 pp and β given placebo is ≈ 0.
- **The Championship behaves like a thinner lower-league market.** CLV is small and positive from T−3h to T−45m (up to +0.27), and β given placebo is above 0 at T−45m for three fits. But EV@close stays at −0.1% to −1.8%, and only about half as many fixtures are priced.
- **Liquidity ranks as expected.** Median 1X2 ticks by T−25m are 196 in the Premiership, 51 in the Championship and 38 in League One/Two. The 56/57 early edge sits exactly where the market is thinnest and stalest, and the liquid tier shows none of it.

### 11.1 Coverage and liquidity

Each cell is the median number of 1X2 ticks per fixture (all runners) up to the instant, with the ticks in the 60 min before it in brackets.
- **Universes:** 54/55 is the W2 monitor panel (23/24–25/26); 56/57 is §10's 710 fixtures.
- **Championship prices are missing for 25/26:** the Betfair archive for 55 covers 22/23–24/25 only (TODO 033). Of the 535 Championship fixtures in the W2 panel, only 331 (62%) have any 1X2 tick. The Premiership has 98%.

| instant | 54 Premiership | 55 Championship | 56/57 League One/Two (§10 universe) |
|---|---|---|---|
| T−48h | 12 (0) | 0 (0) | 0 (0) |
| T−24h | 29 (1) | 4 (0) | 4 (0) |
| T−12h | 60 (2) | 12 (0) | 10 (0) |
| T−6h | 96 (8) | 22 (1) | 16 (1) |
| T−3h | 133 (14) | 32 (3) | 24 (2) |
| T−2h | 149 (16) | 37 (3) | 27 (2) |
| T−90m | 158 (17) | 39 (3) | 29 (2) |
| T−60m | 171 (21) | 42 (4) | 31 (3) |
| T−45m | 184 (28) | 44 (5) | 33 (4) |
| T−25m | 196 (36) | 51 (9) | 38 (8) |
| T−10m | 210 (45) | 57 (13) | 44 (13) |
| all pre-off | 224 | 65 | 51 |

### 11.2 The curve per league (1X2, e = 2, view a)

Each cell is **mean clv_pp / EV@close 2% (n bets)**.
- **Bold** means the EV@close CI is wholly above 0 (no cell qualifies).
- † means β given placebo has its CI above 0.
- "Fixtures fresh" is the number of W2-panel fixtures with a fresh, complete 1X2 book, out of the panel.
- W1 fits are scored on 24/25–25/26 and W2 fits on 23/24–25/26.

**54 Premiership**

| entry | fixtures fresh (W2 panel) | stale med (min) | lead | td_spfl_cups_joint (W1) | grw_spfl_joint (W1) | td_base (W2) | grw_step_a2_cj (W2) | NULL back all (W2 panel) |
|---|---|---|---|---|---|---|---|---|
| earliest | 563/594 | — | 75 h | +0.02 / -1.5 (430) | +0.53 / +1.2 (393) † | -0.02 / -1.1 (647) † | +0.57 / +2.3 (579) † | -0.24 / -2.4 (1689) |
| T−48h | 235/594 | 129 | 48 h | -0.23 / -3.2 (212) | +0.28 / +0.1 (183) † | -0.31 / -3.5 (283) | +0.47 / +1.4 (230) † | -0.06 / -2.6 (705) |
| T−24h | 363/594 | 69 | 24 h | -0.29 / -5.2 (288) | +0.16 / -3.2 (262) † | -0.26 / -4.8 (426) | +0.15 / -2.5 (384) † | -0.05 / -2.9 (1089) |
| T−12h | 351/594 | 34 | 12 h | +0.00 / -2.6 (287) | +0.08 / -2.1 (256) | -0.05 / -2.3 (434) | +0.20 / -1.1 (382) | -0.00 / -1.9 (1053) |
| T−6h | 486/594 | 12 | 6 h | -0.04 / -2.8 (394) | +0.02 / -2.3 (349) | -0.05 / -2.5 (610) | +0.07 / -1.7 (527) | -0.00 / -1.8 (1458) |
| T−3h | 545/594 | 8 | 3 h | -0.02 / -2.6 (439) | -0.09 / -2.6 (393) | -0.07 / -2.7 (682) | -0.03 / -2.3 (606) | -0.02 / -1.9 (1635) |
| T−2h | 555/594 | 7 | 2 h | +0.03 / -2.5 (453) | -0.03 / -2.4 (410) | -0.03 / -2.6 (692) | -0.01 / -2.2 (606) | -0.00 / -1.8 (1665) |
| T−90m | 554/594 | 6 | 90 m | +0.03 / -2.6 (449) | +0.01 / -2.3 (408) | -0.02 / -2.6 (689) | -0.05 / -2.4 (605) | -0.02 / -1.9 (1662) |
| T−60m | 566/594 | 3 | 60 m | +0.07 / -2.3 (465) | -0.02 / -2.4 (423) | -0.01 / -2.6 (707) | -0.07 / -2.5 (626) | -0.00 / -1.8 (1698) |
| T−45m | 573/594 | 3 | 45 m | -0.08 / -2.7 (461) | -0.10 / -2.6 (427) | -0.10 / -2.8 (709) | -0.08 / -2.4 (624) | +0.00 / -1.8 (1719) |
| T−25m | 579/594 | 2 | 25 m | +0.01 / -1.7 (469) | -0.01 / -1.6 (436) | -0.04 / -2.0 (711) | -0.03 / -1.8 (641) | -0.00 / -1.5 (1737) |
| T−10m | 580/594 | 2 | 10 m | -0.01 / -1.7 (469) | +0.00 / -1.6 (433) | -0.03 / -1.8 (711) | -0.02 / -1.7 (637) | -0.01 / -1.4 (1740) |

**55 Championship**

| entry | fixtures fresh (W2 panel) | stale med (min) | lead | td_spfl_cups_joint (W1) | grw_spfl_joint (W1) | td_base (W2) | grw_step_a2_cj (W2) | NULL back all (W2 panel) |
|---|---|---|---|---|---|---|---|---|
| earliest | 321/535 | — | 38 h | -0.78 / -5.7 (148) | -0.40 / -3.8 (136) | -0.24 / -3.2 (344) | +0.35 / -0.8 (308) † | -0.20 / -2.4 (963) |
| T−48h | 57/535 | 183 | 48 h | -1.75 / -9.4 (26) | -0.86 / -3.8 (19) | -0.68 / -5.5 (61) | +0.06 / -3.1 (59) | -0.12 / -2.9 (171) |
| T−24h | 131/535 | 104 | 24 h | -0.35 / -3.5 (70) | -0.12 / -2.7 (69) | -0.28 / -3.0 (137) | +0.17 / -1.4 (128) | -0.09 / -1.8 (393) |
| T−12h | 110/535 | 42 | 12 h | -0.51 / -4.1 (56) | -0.41 / -3.8 (56) | -0.42 / -4.0 (118) | -0.23 / -3.5 (113) | -0.04 / -2.3 (330) |
| T−6h | 188/535 | 20 | 6 h | -0.30 / -3.6 (100) | -0.44 / -4.1 (89) | -0.28 / -3.3 (211) | -0.11 / -2.5 (189) | -0.05 / -1.9 (564) |
| T−3h | 220/535 | 18 | 3 h | +0.13 / -0.9 (115) | +0.13 / -0.9 (115) | +0.02 / -1.4 (249) | +0.08 / -1.1 (236) | -0.07 / -1.4 (660) |
| T−2h | 242/535 | 16 | 2 h | +0.08 / -0.8 (129) | -0.00 / -1.0 (126) | -0.01 / -1.5 (267) | -0.01 / -1.4 (249) | -0.05 / -1.3 (726) |
| T−90m | 260/535 | 14 | 90 m | +0.03 / -1.3 (141) | -0.00 / -1.4 (133) | -0.03 / -1.8 (295) | +0.02 / -1.5 (273) | -0.05 / -1.4 (780) |
| T−60m | 265/535 | 12 | 60 m | +0.18 / -0.5 (136) | +0.18 / -0.6 (133) | +0.02 / -1.3 (296) | +0.06 / -1.1 (274) | -0.06 / -1.2 (795) |
| T−45m | 288/535 | 7 | 45 m | +0.27 / -0.1 (161) † | +0.17 / -0.3 (147) † | +0.13 / -0.8 (332) | +0.15 / -0.6 (299) † | -0.07 / -1.2 (864) |
| T−25m | 307/535 | 3 | 25 m | +0.03 / -1.1 (160) | +0.10 / -0.8 (155) | -0.03 / -1.3 (345) | +0.02 / -1.1 (318) | -0.06 / -1.3 (921) |
| T−10m | 315/535 | 3 | 10 m | -0.03 / -1.5 (167) | +0.02 / -1.3 (154) | -0.04 / -1.5 (355) | -0.04 / -1.5 (321) | -0.03 / -1.4 (945) |

- **The Premiership null costs more than the League One/Two null** (EV@close −1.4% to −2.9% against −0.9% to −1.5%).
  - Commission on a fair back costs c·(1 − p), and the Premiership's Old Firm fixtures carry many long prices.
  - So compare each model with its own league's null. In 54, from T−3h on, every fit is 0.2–1.0 pp *below* its null. Only at the earliest and T−48h books do the GRW fits beat it (+2.7 to +4.6 pp, on stale LTPs).
- **β given placebo is above 0 in the Premiership only at ≥ 24 h out, and only for the two GRW fits.** From T−12h the models add nothing the market's own recent prices lack. At T−45m and T−10m, td_base's β given placebo is even slightly *negative* (CI below 0).

### 11.3 Side by side with 56/57 (§10)

The two fits that appear in both studies are compared. Each cell is **clv_pp / EV@close 2%**, and **bold** means the EV CI is above 0. The last column is the null (back-everything) EV@close in each league.

| entry | td_spfl_cups_joint 56/57 (§10) | … 54 | … 55 | grw_step_a2_cj 56/57 (§10) | … 54 | … 55 | null EV 56/57 / 54 / 55 |
|---|---|---|---|---|---|---|---|
| earliest | **+0.89 / +2.7** | +0.02 / -1.5 | -0.78 / -5.7 | **+0.93 / +2.7** | +0.57 / +2.3 | +0.35 / -0.8 | -1.4 / -2.4 / -2.4 |
| T−48h | +0.44 / +1.7 | -0.23 / -3.2 | -1.75 / -9.4 | +0.99 / +3.2 | +0.47 / +1.4 | +0.06 / -3.1 | -1.1 / -2.6 / -2.9 |
| T−24h | +0.56 / +1.4 | -0.29 / -5.2 | -0.35 / -3.5 | **+0.93 / +2.8** | +0.15 / -2.5 | +0.17 / -1.4 | -1.2 / -2.9 / -1.8 |
| T−12h | **+1.20 / +2.6** | +0.00 / -2.6 | -0.51 / -4.1 | **+1.29 / +3.2** | +0.20 / -1.1 | -0.23 / -3.5 | -0.9 / -1.9 / -2.3 |
| T−6h | +0.51 / +0.9 | -0.04 / -2.8 | -0.30 / -3.6 | +0.66 / +1.4 | +0.07 / -1.7 | -0.11 / -2.5 | -1.2 / -1.8 / -1.9 |
| T−3h | +0.53 / +0.7 | -0.02 / -2.6 | +0.13 / -0.9 | +0.43 / +0.5 | -0.03 / -2.3 | +0.08 / -1.1 | -1.4 / -1.9 / -1.4 |
| T−2h | +0.59 / +0.7 | +0.03 / -2.5 | +0.08 / -0.8 | +0.44 / +0.3 | -0.01 / -2.2 | -0.01 / -1.4 | -1.3 / -1.8 / -1.3 |
| T−90m | +0.45 / +0.5 | +0.03 / -2.6 | +0.03 / -1.3 | +0.38 / +0.5 | -0.05 / -2.4 | +0.02 / -1.5 | -1.3 / -1.9 / -1.4 |
| T−60m | +0.56 / +1.0 | +0.07 / -2.3 | +0.18 / -0.5 | +0.50 / +0.9 | -0.07 / -2.5 | +0.06 / -1.1 | -1.1 / -1.8 / -1.2 |
| T−45m | +0.53 / +0.6 | -0.08 / -2.7 | +0.27 / -0.1 | +0.42 / +0.2 | -0.08 / -2.4 | +0.15 / -0.6 | -1.1 / -1.8 / -1.2 |
| T−25m | +0.33 / -0.3 | +0.01 / -1.7 | +0.03 / -1.1 | +0.22 / -0.6 | -0.03 / -1.8 | +0.02 / -1.1 | -1.2 / -1.5 / -1.3 |
| T−10m | +0.08 / -1.1 | -0.01 / -1.7 | -0.03 / -1.5 | +0.06 / -1.2 | -0.02 / -1.7 | -0.04 / -1.5 | -1.2 / -1.4 / -1.4 |

```
grw_step_a2_carry_jump, 1X2, e = 2: EV@close at 2% (%) by entry time; one common scale, │ = 0
         56/57                       54                          55
earliest          │███████    +2.7            │██████     +2.3          ██│           -0.8
T−48h             │████████   +3.2            │████       +1.4    ████████│           -3.1
T−24h             │███████    +2.8     ███████│           -2.5        ████│           -1.4
T−12h             │████████   +3.2         ███│           -1.1   █████████│           -3.5
T−6h              │████       +1.4       █████│           -1.7      ██████│           -2.5
T−3h              │█          +0.5      ██████│           -2.3         ███│           -1.1
T−2h              │█          +0.3      ██████│           -2.2        ████│           -1.4
T−90m             │█          +0.5      ██████│           -2.4        ████│           -1.5
T−60m             │██         +0.9      ██████│           -2.5         ███│           -1.1
T−45m             │█          +0.2      ██████│           -2.4          ██│           -0.6
T−25m           ██│           -0.6       █████│           -1.8         ███│           -1.1
T−10m          ███│           -1.2        ████│           -1.7        ████│           -1.5
```

- **In League One/Two** the same two fits sit +1.4 to +4.3 pp of EV above their league's null from the earliest book to T−45m.
- **In the Premiership** they are +0.3 to +4.6 pp above the null only at ≥ 24 h out (thin, stale books), and 0.2–0.6 pp *below* it from T−3h on.
- **In the Championship** they are within ±0.9 pp of the null from T−3h on, and below it at the stale early instants (to −5.3 pp).

### 11.4 Caveats

- **Different panels.** W1 fits cover 24/25–25/26 and W2 fits 23/24–25/26 (the Championship's 25/26 has no prices). The §10 56/57 rows are 24/25–25/26 only.
- **The Premiership's "earliest" book is about 75 h out and ungated.** Its +2.3% for grw_step_a2_carry_jump rests on 579 bets at stale opening LTPs, with a CI of [−0.3, +5.1].
- **Multiple comparisons.** 4 fits × 2 leagues × 12 instants, all on the same fixtures. Nothing positive would survive a correction. What does survive: in the Premiership, EV@close < 0 at every instant from T−24h, with 39 of 40 CIs wholly below 0 (z down to −27).

## 12. Price movement as a signal

Code: `r03_price_movement.jl` (Part B) and `l03_price_movement.jl` (price-path panel, bounce-free moves, conditional logit).
- **Discovery** is 21/22–23/24 and **confirmation** is 24/25–25/26.
- Six confirmatory hypotheses were frozen and committed in §12.3 (commit 7c5e5706) before confirmation was run, once.
- Output: `out/r03b_{disc,conf}_*.csv` and `out/r03b_{disc,conf}_run.log`.
- **Phases shown:** tables give both phases, except the B1 path table and the charts (discovery). Their confirmation versions are in the CSVs.
- **Discovery only (did not replicate):**
  - the League One late reversion (T−2h→T−25m, H1, and its rule form H6);
  - the Championship's earliest→T−6h momentum (+0.10);
  - B3's two Championship g cells;
  - every discovery-best market-only EV rule (H5 and its neighbours).
- **Confirmation only (post hoc, untested):**
  - the mid-window (T−6h→T−60m) reversion in League One, the Championship and pooled;
  - the size of B4's AGAINST > WITH gap at T−3h and T−2h;
  - the two nominal AGAINST EV@close > 0 cells for m05_joint_td and m12_td.

**Answer.** Price movement is mostly a *timing* signal. The one piece that survives confirmation is **reversion in the lower leagues, not momentum**.
- **The shape of the path (B1).** About two-thirds of the earliest→close move has happened by T−6h (61–73% per league) and 88–93% by T−25m (discovery). Trading intensity rises 12–37× from T−6h to the off, and the Premiership trades 2.5–5× more than League One/Two.
- **Reversion (B2).** With LTP bounce removed (the move is never measured to the tick you would trade), prices that moved between T−6h and T−60m partly revert by the close.
  - This was pre-registered for League Two: **H2, slope −0.21 [−0.37, −0.06], Holm p 0.02**, the only one of six hypotheses confirmed.
  - In confirmation the same sign also shows in League One, the Championship and pooled (post hoc).
  - A third to a half of the raw-LTP reversion is bid/ask bounce.
- **Momentum (B2).** The Premiership shows a small late momentum (+0.02 to +0.06) that did not clear Holm.
- **The close is efficient with respect to movement (B3).** Given the close, earlier movement carries no information about *results*: g ≈ 0 in every league and both phases. So any edge is in the timing of entry.
- **Model plus movement (B4).** In League One/Two, model edge ≥ 2 bets where the price had moved **against** the model beat bets where it moved **with** it.
  - CLV is +0.6 to +0.9 pp against −0.1 to +0.6 pp from T−6h to T−60m, for all three models in confirmation. This is the *opposite* of "the model agrees with the steam".
  - The pre-registered difference (H4) missed Holm (p 0.04 → 0.20).
- **No market-only rule clears commission (B5).** The most replicable rule (back Premiership runners that shortened ≥ 1 pp since T−3h, at T−25m) earns CLV +0.12 pp [+0.06, +0.19] in confirmation but EV@close −0.9%. The discovery-best EV rule (H5) and the League One drift rule (H6) both failed.

### 12.1 B1: the price path (1X2 unless stated)

**Share of the earliest→close squared move resolved by each instant**, computed as 1 − Σ(p_close − p_t)² / Σ(p_close − p_e)². p_t is the de-vigged 45-min TWA, and markets not yet formed carry their opening price.

| market | league | discovery: % by T−24h · T−6h · T−60m · T−25m (markets) | confirmation |
|---|---|---|---|
| 1X2 | 54 | 42 · 63 · 81 · 93 (571) | 45 · 68 · 80 · 94 (388) |
| 1X2 | 55 | 28 · 63 · 84 · 90 (344) | 19 · 61 · 76 · 86 (152) |
| 1X2 | 56 | 32 · 61 · 82 · 88 (529) | 17 · 45 · 53 · 63 (353) |
| 1X2 | 57 | 43 · 73 · 88 · 90 (460) | 25 · 48 · 57 · 63 (278) |
| 1X2 | all | 37 · 66 · 84 · 90 (1904) | 28 · 55 · 65 · 75 (1171) |
| OU2.5 | 54 | 6 · 47 · 74 · 89 (567) | 31 · 63 · 77 · 91 (392) |
| OU2.5 | 55 | 3 · 38 · 76 · 83 (333) | 15 · 50 · 79 · 86 (142) |
| OU2.5 | 56 | 3 · 30 · 65 · 72 (510) | 2 · 32 · 70 · 75 (325) |
| OU2.5 | 57 | 8 · 38 · 64 · 72 (422) | 1 · 67 · 88 · 91 (249) |
| OU2.5 | all | 5 · 40 · 71 · 81 (1832) | 15 · 58 · 79 · 88 (1108) |

```
Share of the earliest→close squared move resolved by each instant (1X2, 45-min TWA, discovery)
        54                         55                         56                         57
T−48h   ▏███████               35%  ▏███                   16%  ▏██                    11%  ▏████                  18%
T−24h   ▏████████              42%  ▏██████                28%  ▏██████                32%  ▏█████████             43%
T−12h   ▏██████████            50%  ▏█████████             45%  ▏██████████            48%  ▏███████████           57%
T−6h    ▏█████████████         63%  ▏█████████████         63%  ▏████████████          61%  ▏███████████████       73%
T−3h    ▏███████████████       73%  ▏████████████████      79%  ▏███████████████       75%  ▏████████████████      81%
T−60m   ▏████████████████      81%  ▏█████████████████     84%  ▏████████████████      82%  ▏██████████████████    88%
T−25m   ▏███████████████████   93%  ▏██████████████████    90%  ▏██████████████████    88%  ▏██████████████████    90%
T−10m   ▏███████████████████   97%  ▏███████████████████   94%  ▏███████████████████   93%  ▏███████████████████   93%

1X2 ticks per hour per market (all runners), by bucket, discovery (log-ish bars: one █ ≈ 4 ticks/h)
T−24h   ▏                       0.8  ▏                       0.4  ▏                       0.4  ▏                       0.3
T−12h   ▏█                      3.0  ▏                       1.1  ▏                       0.8  ▏                       0.8
T−6h    ▏██                     6.1  ▏█                      2.7  ▏                       1.5  ▏                       1.6
T−3h    ▏███                   13.7  ▏█                      5.8  ▏█                      4.5  ▏█                      4.2
T−2h    ▏████                  17.8  ▏██                     7.2  ▏█                      5.2  ▏█                      4.6
T−90m   ▏█████                 21.0  ▏██                     8.9  ▏██                     7.0  ▏██                     6.5
T−60m   ▏███████               27.0  ▏███                   10.1  ▏██                     6.5  ▏██                     6.4
T−45m   ▏██████████            40.3  ▏████                  17.0  ▏███                   13.3  ▏███                   11.2
T−25m   ▏███████████           43.4  ▏██████                24.2  ▏████                  16.9  ▏████                  15.0
T−10m   ▏█████████████         53.0  ▏████████              32.7  ▏██████                24.9  ▏█████                 21.9
off     ▏███████████████████   76.4  ▏████████████████      64.7  ▏██████████████        55.7  ▏█████████████         53.0
```

**Mean |Δp| per bucket (pp, TWA-smoothed) · ticks per hour per market · share of markets formed (discovery).** Confirmation is in `out/r03b_conf_b1_path.csv` and looks the same except in the last half-hour (below).

| bucket | 54: \|Δp\| pp · ticks/h · formed | 55 | 56 | 57 |
|---|---|---|---|---|
| open → T−48h | 1.01 · — · 68% | 0.26 · — · 24% | 0.27 · — · 20% | 0.30 · — · 17% |
| T−48h → T−24h | 0.97 · 0.8 · 97% | 0.88 · 0.4 · 70% | 0.89 · 0.4 · 60% | 0.83 · 0.3 · 59% |
| T−24h → T−12h | 0.84 · 3.0 · 100% | 1.18 · 1.1 · 93% | 1.16 · 0.8 · 86% | 1.16 · 0.8 · 82% |
| T−12h → T−6h | 0.86 · 6.1 · 100% | 1.02 · 2.7 · 99% | 1.09 · 1.5 · 94% | 1.12 · 1.6 · 91% |
| T−6h → T−3h | 0.69 · 13.7 · 100% | 0.93 · 5.8 · 100% | 1.17 · 4.5 · 98% | 1.09 · 4.2 · 98% |
| T−3h → T−2h | 0.34 · 17.8 · 100% | 0.46 · 7.2 · 100% | 0.57 · 5.2 · 99% | 0.54 · 4.6 · 99% |
| T−2h → T−90m | 0.21 · 21.0 · 100% | 0.29 · 8.9 · 100% | 0.37 · 7.0 · 99% | 0.40 · 6.5 · 99% |
| T−90m → T−60m | 0.23 · 27.0 · 100% | 0.30 · 10.1 · 100% | 0.41 · 6.5 · 100% | 0.42 · 6.4 · 99% |
| T−60m → T−45m | 0.23 · 40.3 · 100% | 0.19 · 17.0 · 100% | 0.21 · 13.3 · 100% | 0.21 · 11.2 · 99% |
| T−45m → T−25m | 0.38 · 43.4 · 100% | 0.31 · 24.2 · 100% | 0.35 · 16.9 · 100% | 0.32 · 15.0 · 100% |
| T−25m → T−10m | 0.26 · 53.0 · 100% | 0.27 · 32.7 · 100% | 0.34 · 24.9 · 100% | 0.32 · 21.9 · 100% |
| T−10m → off | 0.16 · 76.4 · 100% | 0.21 · 64.7 · 100% | 0.27 · 55.7 · 100% | 0.27 · 53.0 · 100% |
| off → close | 0.28 · — · 100% | 0.42 · — · 100% | 0.64 · — · 100% | 0.71 · — · 100% |

**Favourites, longshots, home, draw and away** (all four leagues). Runners are ranked by the opening de-vigged price; drift is p_close − p_open.

| runner (1X2, all leagues) | discovery: open p · drift open→close pp [CI] · \|move\| · % by T−6h · % by T−60m | confirmation |
|---|---|---|
| favourite | 0.51 · +0.11 [-0.13, +0.35] · 3.82 · 65% · 84% | 0.50 · +0.44 [+0.15, +0.72] · 3.79 · 53% · 63% |
| middle | 0.27 · -0.50 [-0.66, -0.34] · 2.63 · 64% · 83% | 0.27 · -0.71 [-0.88, -0.53] · 2.65 · 58% · 65% |
| longshot | 0.22 · +0.39 [+0.24, +0.54] · 2.21 · 70% · 87% | 0.22 · +0.27 [+0.07, +0.47] · 2.24 · 55% · 69% |
| home | 0.43 · -0.14 [-0.35, +0.07] · 3.52 · 63% · 83% | 0.43 · +0.24 [-0.05, +0.52] · 3.54 · 52% · 61% |
| draw | 0.25 · +0.11 [-0.02, +0.25] · 1.91 · 78% · 89% | 0.25 · -0.16 [-0.35, +0.03] · 1.87 · 63% · 77% |
| away | 0.32 · +0.03 [-0.16, +0.25] · 3.22 · 64% · 84% | 0.32 · -0.07 [-0.34, +0.21] · 3.28 · 55% · 65% |

- **The middle-priced runner drifts and the longshot firms in both phases** (−0.5 to −0.7 pp and +0.3 to +0.4 pp). Favourites shortened in confirmation only.
  - This is the market pricing out the second-favourite, not a favourite–longshot pattern in a single direction.
  - There is no stable home/draw/away drift.
- **The draw moves least:** |move| is 1.9 pp against 3.2–3.5 pp for home and away, and more of its move is resolved early.
- **O/U 2.5 forms later** (5% of its move by T−24h in discovery) and moves mostly from T−12h to T−3h.
- **In confirmation, League One/Two resolve much less of the move by T−25m** (63% against 88–90%). Their last-25-minute |Δp| roughly doubles (T−25m→T−10m 0.53–0.63 pp against 0.32–0.34). Some of this is the close estimator: the close is a TWA over only (−20, 0], and in 24/25–25/26 the off→close gap in 56/57 is 0.9–1.0 pp. In either case the late window in the lower leagues got noisier.

### 12.2 B2: momentum vs reversion, and how much survives de-bouncing

The regression is y = the later move to the close on x = the earlier move, one row per 1X2 selection.
- **Sample:** markets that had formed by the window start and are fresh (l02 gate) at the boundary t_m. CIs are slate-clustered.
- **raw:** de-vigged LTP at both ends. The boundary tick sits in x with + and in y with −, so bid/ask bounce shows up as fake reversion.
- **smooth:** 45-min TWA prices. This variant is **biased toward momentum on short windows**, because first differences of a time-averaged random walk are positively autocorrelated (the Working effect). That is why its slopes are +0.3 to +0.8 for windows ending at T−25m. It is shown only because the brief asked for it; it is not a clean test.
- **droplast / gap15:** x ends at the tick *before* the boundary tick (droplast) or at the last tick ≤ t_m − 15 min (gap15). It never uses the boundary tick, and if no tick falls inside the window, x = 0. y = p_close − 1/odds_LTP(t_m) is the CLV of a back at that LTP. These are the bounce-free, tradeable-direction tests, on the raw 1/odds scale.
- **Bold** marks a gap15 CI that excludes 0.

| x window | league | disc raw | smooth | droplast | **gap15** [95% CI] | n | conf raw | smooth | droplast | **gap15** [95% CI] | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| earliest → T−6h | 54 | -0.022 | -0.004 | -0.007 | -0.007 [-0.036, +0.022] | 1392 | +0.005 | +0.014 | -0.006 | -0.009 [-0.059, +0.046] | 969 |
|  | 55 | +0.038 | +0.067 | +0.088 | **+0.097 [+0.008, +0.188]** | 555 | -0.080 | -0.026 | +0.064 | +0.053 [-0.136, +0.267] | 264 |
|  | 56 | +0.051 | +0.074 | +0.087 | +0.083 [-0.065, +0.259] | 429 | -0.128 | -0.082 | -0.071 | -0.067 [-0.171, +0.048] | 300 |
|  | 57 | -0.031 | -0.029 | -0.045 | -0.047 [-0.121, +0.010] | 447 | -0.073 | -0.048 | +0.004 | +0.002 [-0.120, +0.095] | 225 |
|  | all | -0.008 | +0.008 | +0.003 | +0.003 [-0.027, +0.031] | 2823 | -0.047 | -0.021 | -0.009 | -0.011 [-0.052, +0.032] | 1758 |
| T−6h → T−60m | 54 | +0.065 | +0.099 | +0.077 | +0.040 [-0.028, +0.100] | 1668 | +0.101 | +0.137 | +0.105 | +0.034 [-0.074, +0.151] | 1131 |
|  | 55 | -0.116 | -0.041 | -0.060 | -0.052 [-0.159, +0.037] | 804 | -0.250 | -0.139 | -0.148 | -0.133 [-0.270, +0.000] | 363 |
|  | 56 | -0.023 | +0.030 | +0.033 | +0.034 [-0.065, +0.136] | 993 | -0.318 | -0.304 | -0.176 | **-0.185 [-0.312, -0.045]** | 516 |
|  | 57 | -0.211 | -0.119 | -0.112 | **-0.101 [-0.188, -0.036]** | 798 | -0.318 | -0.304 | -0.236 | **-0.210 [-0.363, -0.056]** | 402 |
|  | all | -0.068 | -0.010 | -0.019 | -0.026 [-0.070, +0.020] | 4263 | -0.186 | -0.179 | -0.106 | **-0.137 [-0.204, -0.072]** | 2412 |
| T−3h → T−60m | 54 | +0.091 | +0.176 | +0.110 | +0.043 [-0.063, +0.140] | 1668 | +0.072 | +0.216 | +0.102 | -0.037 [-0.183, +0.108] | 1131 |
|  | 55 | -0.264 | -0.177 | -0.133 | -0.119 [-0.276, +0.046] | 816 | -0.357 | -0.212 | -0.168 | -0.114 [-0.310, +0.068] | 363 |
|  | 56 | -0.168 | -0.100 | -0.056 | -0.073 [-0.197, +0.055] | 1044 | -0.390 | -0.447 | -0.202 | **-0.222 [-0.402, -0.021]** | 528 |
|  | 57 | -0.255 | -0.134 | -0.117 | -0.093 [-0.203, +0.026] | 843 | -0.307 | -0.276 | -0.136 | -0.076 [-0.298, +0.140] | 417 |
|  | all | -0.142 | -0.076 | -0.047 | **-0.068 [-0.128, -0.001]** | 4371 | -0.201 | -0.219 | -0.071 | **-0.121 [-0.212, -0.031]** | 2439 |
| T−2h → T−25m | 54 | +0.040 | +0.275 | +0.056 | **+0.054 [+0.022, +0.082]** | 1704 | +0.021 | +0.295 | +0.042 | +0.024 [-0.005, +0.054] | 1158 |
|  | 55 | -0.056 | +0.082 | +0.017 | -0.018 [-0.069, +0.037] | 966 | -0.023 | +0.163 | +0.029 | -0.027 [-0.119, +0.057] | 420 |
|  | 56 | -0.179 | +0.040 | -0.123 | **-0.131 [-0.191, -0.070]** | 1314 | -0.039 | +0.053 | +0.002 | -0.024 [-0.131, +0.068] | 834 |
|  | 57 | -0.100 | -0.064 | -0.070 | -0.078 [-0.163, +0.000] | 1056 | -0.069 | +0.097 | -0.062 | -0.089 [-0.180, +0.022] | 699 |
|  | all | -0.087 | +0.083 | -0.041 | **-0.047 [-0.079, -0.017]** | 5040 | -0.035 | +0.163 | -0.007 | -0.021 [-0.067, +0.019] | 3111 |
| T−60m → T−25m | 54 | +0.019 | +0.411 | +0.060 | **+0.057 [+0.024, +0.090]** | 1704 | +0.038 | +0.426 | +0.052 | **+0.033 [+0.002, +0.068]** | 1158 |
|  | 55 | -0.043 | +0.346 | +0.034 | -0.012 [-0.063, +0.055] | 966 | +0.009 | +0.500 | +0.049 | -0.017 [-0.124, +0.087] | 420 |
|  | 56 | -0.168 | +0.367 | -0.126 | **-0.137 [-0.217, -0.044]** | 1326 | -0.008 | +0.807 | +0.032 | +0.048 [-0.072, +0.154] | 837 |
|  | 57 | -0.086 | +0.130 | -0.078 | -0.101 [-0.238, +0.010] | 1062 | -0.060 | +0.585 | -0.058 | -0.113 [-0.251, +0.021] | 702 |
|  | all | -0.091 | +0.338 | -0.034 | **-0.041 [-0.078, -0.004]** | 5058 | -0.020 | +0.542 | +0.010 | +0.004 [-0.039, +0.043] | 3117 |

**What survives de-bouncing (gap15 as a share of raw).**
- **League Two, T−6h→T−60m:** 48% of the raw reversion in discovery (−0.10 of −0.21) and 66% in confirmation (−0.21 of −0.32). This is H2, confirmed.
- **League One, late window (T−2h→T−25m):** 73% survived in discovery (−0.13 of −0.18), but the effect vanished in confirmation, raw included. H1 failed.
- **League One, T−6h→T−60m:** absent in discovery, then −0.19 [−0.31, −0.05] in confirmation. This is *post hoc* and not tested.
- **Pooled, T−6h→T−60m:** −0.03 (n.s.) in discovery and −0.14 [−0.20, −0.07] in confirmation. The mid-window reversion is stronger in the later seasons.
- **Premiership, T−2h→T−25m and T−60m→T−25m:** +0.05 to +0.06 momentum in discovery (bounce *hides* it: raw is +0.02 to +0.04). In confirmation it is +0.02 to +0.03, and H3 failed Holm. The rule form (B5) nevertheless replicates its CLV.
- **Earliest→T−6h:** nothing consistent. One discovery cell (the Championship, +0.10) did not replicate. Early drift neither continues nor reverts.

**O/U 2.5 (pooled leagues).** There is a weak T−6h→T−60m reversion in discovery (gap15 −0.06 [−0.11, −0.02]) that is not significant in confirmation (−0.04 [−0.11, +0.04]). Late windows show nothing once de-bounced.

| O/U 2.5, all leagues: x window | disc raw | disc gap15 [CI] (n) | conf raw | conf gap15 [CI] (n) |
|---|---|---|---|---|
| T−6h → T−60m | -0.078 | -0.064 [-0.112, -0.017] (1742) | -0.049 | -0.040 [-0.111, +0.036] (936) |
| T−3h → T−60m | -0.110 | -0.055 [-0.138, +0.027] (1904) | -0.076 | -0.060 [-0.153, +0.035] (992) |
| T−2h → T−25m | -0.040 | +0.020 [-0.024, +0.064] (2448) | -0.032 | +0.021 [-0.021, +0.062] (1306) |
| T−60m → T−25m | -0.063 | +0.027 [-0.026, +0.085] (2498) | -0.053 | +0.044 [-0.002, +0.090] (1346) |

### 12.3 Pre-registered confirmatory hypotheses (frozen from discovery)

- **Discovery** is 21/22–23/24 and **confirmation** is 24/25–25/26, all tournaments 54–57, 1X2.
- Code: `r03_price_movement.jl` §8, run once with `C3_PHASE=confirmation`.
- Every window, threshold and signal below was fixed on discovery.
- **Test:** each p is a one-sided slate-bootstrap tail share (B = 4,000, floor 1/B). Holm is applied
  across all six at α = 0.05.
- **Definitions:**
  - "gap15" and "droplast" are the bounce-free moves (l03 `c3_bf_move`), measured on 1/odds. They
    never use the entry tick.
  - The slope is the OLS slope of y = p_close − 1/odds_LTP(t_m) on that move, among markets that
    had formed by the window start and are fresh at t_m.

| id | hypothesis (one-sided) | discovery estimate [95% CI] | n (disc.) |
|---|---|---|---|
| H1 | League One (56): slope of the T−25m→close move on the T−2h→T−25m gap15 move **< 0** (late reversion) | −0.131 [−0.194, −0.067] | 1,314 sel. |
| H2 | League Two (57): slope of the T−60m→close move on the T−6h→T−60m gap15 move **< 0** | −0.101 [−0.190, −0.035] | 798 sel. |
| H3 | Premiership (54): slope of the T−25m→close move on the T−2h→T−25m gap15 move **> 0** (late momentum) | +0.054 [+0.023, +0.085] | 1,704 sel. |
| H4 | 56/57, grw_step_a2_carry_jump 1X2 bets with edge ≥ 2 pp at T−60m: mean CLV when the gap15 move since the earliest book went *against* the model (< −0.1 pp) minus mean CLV when it went *with* it (> +0.1 pp) **> 0** | +0.43 pp [−0.12, +1.01] | 197 bets |
| H5 | League Two (57), market only: back every 1X2 runner whose droplast move since T−6h is ≤ −2 pp, at the T−60m LTP. EV@close at 2% commission **> 0** | +0.40% [−0.81, +1.69] | 124 bets |
| H6 | League One (56), market only: back every 1X2 runner whose gap15 move since T−3h is ≤ −2 pp, at the T−25m LTP. Mean CLV **> 0** | +0.45 pp [+0.20, +0.71] | 132 bets |

**Why these six:**
- H1–H3 are the B2 cells where the bounce-free slope has a discovery |z| ≥ 2.5 and the gap15 and
  droplast variants agree.
- H4 is the B4 question in the direction discovery pointed. The *model agrees with the steam*
  direction lost in discovery.
- H5 is the single best market-only EV@close cell out of about 800 B5 cells. It is included to
  measure the winner's curse, and it is expected to fail.
- H6 is the tradeable form of H1: the CLV of a rule, not a slope.

**Not tested but reported:** everything else in §12. It is re-run on confirmation with the frozen
code (B1–B5 tables) as description, not as tests.

**Results (confirmation, run once).** The discovery column is the same frozen statistic on 21/22–23/24.

| id | hypothesis | discovery (n) | confirmation [95% CI] (n) | one-sided p | Holm p | confirmed |
|---|---|---|---|---|---|---|
| H1 | 56: slope of T−25m→close on T−2h→T−25m (gap15) < 0 | -0.131 (1314) | -0.024 [-0.129, +0.067] (834) | 0.3182 | 0.722 | no |
| H2 | 57: slope of T−60m→close on T−6h→T−60m (gap15) < 0 | -0.101 (798) | -0.210 [-0.366, -0.060] (402) | 0.0037 | 0.022 | **yes** |
| H3 | 54: slope of T−25m→close on T−2h→T−25m (gap15) > 0 | +0.054 (1704) | +0.024 [-0.005, +0.054] (1158) | 0.0473 | 0.201 | no |
| H4 | 56/57 grw_step edge≥2 at T−60m: CLV(price against model) − CLV(with) > 0 | +0.43 pp (197) | +0.52 pp [-0.06, +1.11] (292) | 0.0403 | 0.201 | no |
| H5 | 57: back drift ≤ −2 pp since T−6h (droplast) at T−60m: EV@close 2% > 0 | +0.40% (124) | +0.84% [-1.44, +3.14] (65) | 0.2407 | 0.722 | no |
| H6 | 56: back drift ≤ −2 pp since T−3h (gap15) at T−25m: CLV > 0 | +0.45 pp (132) | +0.11 pp [-0.27, +0.52] (91) | 0.2980 | 0.722 | no |

- **Only H2 is confirmed** (League Two, T−6h→T−60m reversion). Its confirmation estimate is twice its discovery one.
- **H1** (League One, late reversion) and **H6** (its rule form) failed: the late League One effect was a discovery-period feature.
- **H3** (Premiership late momentum) and **H4** (model bets where the price went against the model beat those where it went with it) are nominally significant (one-sided p 0.047 and 0.040) with the discovery sign, but not after Holm.
- **H5,** the discovery-best market-only EV rule (a winner's-curse check), failed, as expected. Its confirmation point estimate is +0.8% on 65 bets with a CI of [−1.4, +3.1].

### 12.4 B3: is the close efficient with respect to movement?

Conditional logit per 1X2 market: P(win_i) = softmax(c·log p_close_i + g·m_i), where m = log p_close − log p_t (the move from t to the close).
- g = 0 means the close has fully priced the movement.
- c is a temperature: c > 1 means the close is under-confident.
- CIs are slate-bootstrap (B = 300).
- "LL gain" is the in-sample log-likelihood gain of (c, g) over c alone, per market, ×10⁻³. It is ≥ 0 by construction.

| league | disc c | g: move from earliest | g: from T−6h | g: from T−60m | LL gain (T−6h) ×10⁻³ | markets | conf c | g: earliest | g: T−6h | g: T−60m | LL gain | markets |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 54 | 1.07 | -0.08 [-0.87, +0.80] | -0.80 [-2.10, +0.31] | -0.79 [-2.56, +0.79] | +1.24 | 561 | 0.90 | -0.01 [-0.97, +0.91] | -0.60 [-2.23, +0.85] | -1.50 [-3.48, +0.50] | +0.75 | 376 |
| 55 | 0.92 | +0.55 [-0.48, +1.56] | +1.48 [-0.28, +3.61] | +2.27 [+0.08, +4.93] | +4.34 | 330 | 1.00 | +0.60 [-1.07, +2.09] | +1.55 [-0.75, +4.74] | +0.99 [-2.07, +4.57] | +4.82 | 147 |
| 56 | 1.03 | -0.46 [-1.04, +0.06] | -1.01 [-2.07, +0.16] | -0.92 [-2.54, +0.80] | +3.25 | 498 | 0.78 | -0.07 [-1.10, +1.05] | -0.49 [-1.86, +0.77] | +0.54 [-0.81, +1.98] | +0.89 | 323 |
| 57 | 1.05 | -0.23 [-0.77, +0.30] | +0.13 [-1.28, +1.43] | +0.67 [-1.30, +2.62] | +0.04 | 425 | 0.85 | -0.27 [-1.17, +0.71] | -0.17 [-1.71, +1.26] | -0.03 [-1.29, +1.52] | +0.12 | 260 |
| all | 1.04 | -0.18 [-0.45, +0.10] | -0.31 [-0.85, +0.32] | -0.05 [-0.94, +0.76] | +0.24 | 1814 | 0.88 | -0.04 [-0.54, +0.47] | -0.23 [-0.93, +0.47] | -0.04 [-0.85, +0.75] | +0.16 | 1106 |

- **g is ≈ 0 in every league, move window and phase.** Two discovery cells have CIs clear of 0 (the Championship, moves from T−60m and T−25m); their confirmation CIs span 0.
- **The close already contains whatever the path knew about results.** Movement is a *timing* signal (take the price before it moves), not information the close misses.
- **The temperature flipped between phases:** c = 1.03–1.07 in discovery and 0.78–0.90 in confirmation for 54/56/57. The close was slightly under-confident in 21/22–23/24 and over-confident in 24/25–25/26. This is itself a warning against reading EV@close as truth in any single season.

### 12.5 B4: model edge × price direction

- **Bets:** 1X2, model edge ≥ 2 pp at the entry LTP.
- **Direction:** the gap15 move since the stated instant, on the bet runner. WITH means the move is > +0.1 pp (the runner shortened toward the model); AGAINST means it is < −0.1 pp.
- **Discovery:** one held-out season (23/24) of grw_step_a2_carry_jump. **Confirmation:** grw_step_a2_carry_jump, plus m05_joint_td and m12_td on the 710 panel as the brief asked.
- **m12_td** prices with the played XI, so its rows before T−60m are not clean.
- Each cell is **clv_pp / EV@close 2% (n)**. **Bold** means the CLV CI is above 0. League One/Two (56+57):

| entry (direction since) | phase, model | model edge ≥ 2, all | price moved WITH model | price moved AGAINST model | market only: back what shortened |
|---|---|---|---|---|---|
| T−6h (since earliest) | disc grw_step_a2_cj | +0.36 / -0.2 (101) | +0.55 / +0.9 (30) | +0.24 / -1.2 (65) | -0.13 / -1.4 (132) |
| T−6h (since earliest) | conf grw_step_a2_cj | **+0.66 / +1.4 (174)** | +0.46 / +0.5 (55) | **+0.83 / +1.6 (97)** | -0.30 / -1.7 (205) |
| T−6h (since earliest) | conf m05_joint_td | **+0.70 / +1.4 (185)** | +0.53 / +0.7 (59) | **+0.83 / +1.6 (102)** |  |
| T−6h (since earliest) | conf m12_td | **+0.64 / +1.1 (182)** | +0.63 / +1.1 (55) | **+0.73 / +1.2 (102)** |  |
| T−3h (since T−6h) | disc grw_step_a2_cj | +0.16 / -1.4 (172) | -0.00 / -1.4 (65) | +0.36 / -1.4 (85) | -0.41 / -2.2 (223) |
| T−3h (since T−6h) | conf grw_step_a2_cj | **+0.48 / +0.7 (250)** | -0.09 / -0.9 (101) | **+0.65 / +1.0 (102)** | -0.47 / -2.3 (342) |
| T−3h (since T−6h) | conf m05_joint_td | **+0.60 / +0.8 (247)** | +0.13 / -0.4 (99) | **+0.89 / +1.6 (102)** |  |
| T−3h (since T−6h) | conf m12_td | **+0.50 / +0.5 (260)** | +0.14 / -0.4 (101) | **+0.65 / +0.9 (115)** |  |
| T−2h (since T−6h) | disc grw_step_a2_cj | +0.28 / -0.3 (176) | +0.13 / -0.1 (64) | +0.41 / -0.4 (90) | -0.36 / -1.9 (223) |
| T−2h (since T−6h) | conf grw_step_a2_cj | **+0.50 / +0.6 (276)** | +0.23 / -0.0 (106) | **+0.66 / +1.2 (136)** | -0.45 / -2.0 (368) |
| T−2h (since T−6h) | conf m05_joint_td | **+0.45 / +0.3 (292)** | +0.21 / -0.1 (110) | **+0.66 / +1.0 (139)** |  |
| T−2h (since T−6h) | conf m12_td | **+0.41 / +0.2 (286)** | +0.14 / -0.3 (113) | **+0.63 / +0.9 (133)** |  |
| T−60m (since earliest) | disc grw_step_a2_cj | +0.24 / -0.9 (208) | +0.01 / -1.6 (65) | **+0.44 / +0.1 (132)** | -0.26 / -1.8 (266) |
| T−60m (since earliest) | conf grw_step_a2_cj | **+0.50 / +0.9 (316)** | +0.11 / -0.8 (99) | **+0.63 / +1.4 (193)** | -0.37 / -1.9 (412) |
| T−60m (since earliest) | conf m05_joint_td | **+0.67 / +1.3 (334)** | +0.54 / +0.4 (109) | **+0.66 / +1.3 (204)** |  |
| T−60m (since earliest) | conf m12_td | **+0.62 / +1.1 (337)** | +0.39 / +0.0 (115) | **+0.62 / +1.1 (200)** |  |
| T−60m (since T−3h) | disc grw_step_a2_cj | +0.29 / -0.4 (202) | +0.18 / -1.2 (73) | +0.32 / -0.5 (93) | -0.25 / -2.0 (235) |
| T−60m (since T−3h) | conf grw_step_a2_cj | **+0.50 / +0.9 (313)** | +0.10 / -0.7 (109) | **+0.78 / +1.9 (146)** | -0.28 / -1.8 (370) |
| T−60m (since T−3h) | conf m05_joint_td | **+0.66 / +1.3 (330)** | +0.43 / +0.1 (120) | **+0.88 / +2.1 (148)** |  |
| T−60m (since T−3h) | conf m12_td | **+0.60 / +1.1 (334)** | +0.27 / -0.4 (119) | **+0.90 / +2.1 (155)** |  |
| T−25m (since T−60m) | disc grw_step_a2_cj | **+0.22 / +0.1 (284)** | +0.15 / +1.8 (101) | **+0.31 / -0.4 (110)** | -0.28 / -0.9 (285) |
| T−25m (since T−60m) | conf grw_step_a2_cj | **+0.21 / -0.6 (501)** | -0.00 / -1.4 (156) | **+0.29 / -0.4 (183)** | -0.09 / -1.5 (502) |
| T−25m (since T−60m) | conf m05_joint_td | **+0.25 / -0.6 (519)** | +0.16 / -1.1 (164) | **+0.27 / -0.4 (181)** |  |
| T−25m (since T−60m) | conf m12_td | **+0.24 / -0.6 (522)** | +0.11 / -1.3 (168) | **+0.27 / -0.4 (183)** |  |

- **In confirmation, AGAINST beats WITH on CLV in all 18 cells shown (3 models × 6 entries).** The smallest gaps are +0.10 to +0.12 pp (m12_td at T−6h and m05_joint_td at T−60m, both since the earliest book). In discovery AGAINST won 5 of 6.
  - The AGAINST group earns +0.27 to +0.90 pp, with EV@close +0.9% to +2.1% from T−3h to T−60m. The WITH group earns −0.1 to +0.6 pp.
  - One EV@close CI clears 0 for both m05_joint_td and m12_td: AGAINST at T−60m since T−3h, +2.1% [+0.0, +4.3] and [+0.2, +4.2]. That is nominal, post hoc, and one of a few hundred B4 cells.
- **"The model agrees with the steam" is not better than either signal alone.** In League One/Two the steam alone (market only: back what shortened) *loses* CLV (−0.1 to −0.6 pp). Model plus steam is roughly the model minus a bit.
- **Independence:** "model edge ≥ 2, all" in confirmation is on the same 24/25–25/26 fixtures as §10, so it is not independent evidence for the early edge. Only the WITH/AGAINST split is new.
- **Premiership/Championship (54+55, grw_step_a2_carry_jump):** no split matters. WITH and AGAINST sit at CLV −0.3 to +0.2 pp and EV@close −1.0% to −2.8% at every entry, in both phases. See `out/r03b_*_b4_model_x_direction.csv`.

| phase | entry | 54/55 WITH model: clv / EV (n) | AGAINST model |
|---|---|---|---|
| disc | T−6h (since earliest) | -0.03 / -1.5 (94) | -0.09 / -2.0 (170) |
| disc | T−60m (since earliest) | -0.28 / -2.8 (110) | -0.12 / -2.1 (230) |
| disc | T−25m (since T−60m) | +0.03 / -1.5 (135) | -0.13 / -1.9 (194) |
| conf | T−6h (since earliest) | +0.19 / -1.0 (145) | +0.06 / -2.4 (266) |
| conf | T−60m (since earliest) | -0.01 / -1.8 (136) | +0.03 / -2.0 (379) |
| conf | T−25m (since T−60m) | +0.02 / -1.2 (206) | -0.01 / -1.7 (317) |

### 12.6 B5: a market-only rule

- **The grid:** 5 league sets × 10 (entry, since) pairs × 2 signals (droplast, gap15) × 2 directions × 4 thresholds (0.5–4 pp), about 800 cells per phase.
- **Rules:** back every 1X2 runner past the threshold at the entry LTP, flat, 2% commission.
- **Replication is weak.** Across the 605 cells with n ≥ 80 bets in discovery and ≥ 30 in confirmation, the discovery→confirmation correlation is 0.26 for CLV and 0.14 for EV@close. Selected rules:

| rule | discovery: clv_pp [CI] · EV@close 2% [CI] · ROI 2% (n) | confirmation |
|---|---|---|
| 54: back runners whose gap15 move since T−3h is ≥ +1 pp, at the T−25m LTP | +0.14 [+0.06, +0.21] · -1.0 [-1.2, -0.7] · +2 (322) | +0.12 [+0.06, +0.19] · -0.9 [-1.1, -0.7] · -7 (222) |
| 54: back runners whose gap15 move since T−60m is ≥ +1 pp, at the T−25m LTP | +0.13 [+0.05, +0.21] · -1.0 [-1.3, -0.8] · -2 (228) | +0.11 [+0.05, +0.19] · -1.0 [-1.2, -0.7] · -14 (170) |
| 56: back runners whose gap15 move since T−3h is ≤ −2 pp, at the T−25m LTP | +0.45 [+0.19, +0.70] · -0.5 [-1.3, +0.3] · +7 (132) | +0.11 [-0.32, +0.54] · -0.9 [-2.1, +0.4] · -7 (91) |
| 57: back runners whose droplast move since T−6h is ≤ −2 pp, at the T−60m LTP | +0.43 [+0.07, +0.77] · +0.4 [-0.8, +1.6] · +9 (124) | +0.68 [+0.03, +1.36] · +0.8 [-1.4, +3.2] · -2 (65) |
| 57: back runners whose gap15 move since T−6h is ≤ −2 pp, at the T−60m LTP | +0.34 [-0.03, +0.68] · -0.1 [-1.3, +1.1] · +21 (123) | +0.70 [+0.06, +1.41] · +0.9 [-1.2, +3.4] · -6 (64) |
| all: back runners whose gap15 move since T−6h is ≤ −2 pp, at the T−60m LTP | +0.02 [-0.15, +0.18] · -1.5 [-2.2, -0.9] · +0 (493) | +0.34 [+0.03, +0.67] · +0.1 [-1.0, +1.3] · +1 (229) |
| all: back runners whose gap15 move since T−6h is ≥ +1 pp, at the T−3h LTP | -0.13 [-0.27, +0.01] · -1.6 [-2.1, -1.1] · -6 (727) | -0.37 [-0.61, -0.13] · -2.2 [-2.9, -1.6] · -5 (419) |

**The best simple market-only rule, in words:** *in the Premiership, at T−25m, back any 1X2 runner whose price has shortened by at least 1 pp (bounce-free) since T−3h.*
- It is the one rule whose CLV replicates: +0.14 pp in discovery, +0.12 pp [+0.06, +0.19] in confirmation.
- Its EV@close is −0.9% [−1.1, −0.7] after 2% commission, and its realised ROI is −7%. The momentum is real and a fraction of the round-trip cost.
- **The League Two "back what drifted ≥ 2 pp since T−6h, at T−60m" rule** has EV@close > 0 point estimates in both phases (+0.4%, +0.8%), but its CIs span 0 on 65–124 bets. It is H5, which failed.
- **No market-only rule has an EV@close CI above 0 in confirmation**, except one 36-bet cell out of 659 (the pooled drift ≥ 4 pp since T−6h rule at T−2h), which is noise at that multiplicity.

### 12.7 Caveats and tradability

- **LTP only.** Every price here is a last traded price. The archive has no back/lay ladder, no spread and no volume, so tick counts are the only liquidity proxy.
  - The bounce-free variants guarantee that the *signal* never shares a tick with the entry. The *entry* is still an LTP, which may have printed on the lay side, where the executable back price is one tick longer.
- **Tradeability labels:**
  - *Not tradeable without ladder data:* the raw-LTP reversion that does not survive gap15. That is about half of every raw reversion slope, plus the discovery-only League One late reversion (H1).
  - *Survives de-bouncing, but not shown to be profitable:* H2's League Two mid-window reversion. It passes droplast, gap15 and (for what it is worth) the smooth variant. Its rule form (H5) did not clear commission.
- **Which ladder data would confirm it:** snapshots from `betfair_live` of the best 3 back and lay prices with sizes and the total matched volume, for the 1X2 runners of 54–57, at least at T−6h, T−3h, T−60m and T−25m. With these:
  - the move can be measured on the **mid-price** (bounce-free by construction);
  - the entry can be priced at the **back price actually available**, not the LTP;
  - the **size** available at that price can be checked. A 2-unit edge on a £5 queue is not an edge.
- **Samples.** Discovery has one season with model prices (23/24). The Championship's confirmation is 24/25 only (147–152 markets). League Two's confirmation windows have 225–417 selections.
- **Multiplicity.** B2 alone is about 200 cells per phase and B5 about 800. Only the six pre-registered hypotheses are tests. Everything else is description, and the post-hoc confirmation patterns (League One and pooled mid-window reversion; the B4 AGAINST split) need their own forward test.

### 12.8 What this means for betting

1. **Is there a candidate rule for a pre-registered forward paper test?** Yes, one, and it is model-based. In League One/Two, back the model's 1X2 edge ≥ 2 pp bets at **T−60m**, and pre-register the split by the bounce-free price direction since T−3h. The claim to test is that AGAINST beats WITH on CLV and that AGAINST has EV@close > 0. That split is H4 plus the reversion of H2. In confirmation the AGAINST group at T−60m (since T−3h) ran +2.1% [0.0, +4.3] for m05_joint_td, but that was post hoc and only nominal. **No market-only rule is worth a test:** none clears 2% commission, and the Premiership steam rule's replicated CLV (+0.12 pp) is about an eighth of the cost.
2. **Which league, entry time and signal?** League One/Two (56/57), not 54/55, where §11 finds no early edge at all. Entry at T−60m, not earlier: it has the best mix of fresh-book coverage (319/710 against 175 at T−6h), the §10 bump, and clean lineup information for m12. Use the lineup-free m05_joint_td (and m12_td from T−60m) with the market direction as a filter, not a market-only signal.
3. **What ladder data does it need?** The `betfair_live` best-3 back/lay prices and sizes and matched volume for 1X2 at T−3h and T−60m (plus T−25m as the control), for every 56/57 fixture over one season. That is enough to replace the LTP entry with an executable back price, measure direction on the mid-price, and log the size available. Without it, the +1–2% EV@close is an LTP number that a one-tick spread would erase.

**Reproduce** (mcmc-beast, `/root/BF_runs/clv_napkin_dev`, after r01 has written `out/probs.jls`):

```bash
nice -n 19 /root/.juliaup/bin/julia --project -t 8 current_development/clv_napkin/r03_leadtime_monitor.jl                   # §11, ~6 min
C3_PHASE=discovery    nice -n 19 /root/.juliaup/bin/julia --project -t 8 current_development/clv_napkin/r03_price_movement.jl   # §12 discovery, ~4 min
C3_PHASE=confirmation nice -n 19 /root/.juliaup/bin/julia --project -t 8 current_development/clv_napkin/r03_price_movement.jl   # §12 confirmation (the once-only run is done)
```

## 13. 26/27 live season: grw_spfl_joint on the order book

Brief 04. Code: `l04_live_orderbook.jl` (loader), `r04_extend_grw_spfl_joint.jl` (the one sampling
job), `r04_live_orderbook.jl` (evaluation). Numbers: `out/r04_*.csv`; logs `out/r04_extend_run.log`,
`out/r04_run.log`.

**Answer.** At the price you could actually have backed at, `grw_spfl_joint` lost to the close in
League One/Two this season at every entry time with data. At T−60m, 61 bets (edge ≥ 2 pp) had mean
CLV −0.59 pp [−1.19, −0.09] and EV@close −0.9% [−4.9, +1.9] after 2% commission. At T−25m, 70 bets had
CLV −0.60 pp [−0.91, −0.19] and EV −3.0% [−4.4, −1.2]. Realised ROI was −16% to −27%, but its CIs span
about ±30 points and mean nothing. **The same bets priced at the mid or at the LTP look like §10:**
CLV +0.05 to +0.2 pp and EV 0% to +3%. So the whole LTP-era edge fits inside the spread, which is
a median 4 ticks, or about 0.8 pp of probability, at the back price. The model's selection is worth
about +0.15 pp over backing everything (the null is −0.74 pp at T−60m), which is a fifth of the
spread. **Early entry is barely possible.** The live book opens a median 3 h before the off: only 4
fixtures have a book at T−24h, 8 at T−6h and 28 at T−3h. **Size is tiny.** The median offer at the
back price is £6 at T−60m and £9 at T−25m, and only 8–13% of bets had £50 or more. **The model
disagrees with the market far more than in 24/25–25/26:** the mean edge on bets is about 10 pp,
against 6–8 pp in §10. Most of that disagreement comes from the four clubs that changed tier this
summer (28 of 79 fixtures). On those fixtures the model's 1X2 log-loss is 0.99, against the mid close's
0.83; on the rest it is about level (1.03 vs 1.02). **The sample is 8 match days and 60–70 bets per
usable instant.** It can detect a CLV of about 0.5–0.8 pp and an EV of about 2–5%, and nothing smaller.
The one result that clears its CI is negative: at the executable back price, CLV < 0.

### 13.1 Step 1: the order book and coverage

- **Tables** (`betfair_live`):
  - `market_metadata` holds `market_id`, `event_id`, `event_name`, `competition(_id)`, `market_type`,
    `home_team`, `away_team` and `open_date`.
  - `order_book_1m` holds `market_id`, `symbol`, `ts`, `bid_prices[]`, `bid_volumes[]`, `ask_prices[]`,
    `ask_volumes[]`, `total_matched` (runner), `market_matched` and `last_price_traded`.
  - Prices and volumes are integers ×10⁴. `bid` is the back side and `ask` the lay side (as in
    `MatchDay.BookLevels`); every bet's back price is below its mid.
- **Depth:** the best 3 levels are present (mean array length 2.6–3.0).
- **Volume:** runner `total_matched` and `market_matched` are populated. On the 08-01 round they are
  populated on 57–88% of rows per market.
- **LTP:** `last_price_traded` is NULL throughout the 08-01 round and populated from 08-08 on. In the
  last 3 h before the off it is present on about 80% of rows.
- **Cadence:** despite the table name, snapshots arrive every **3 min** (median). The pre-off series
  has holes: the median fixture's longest gap is 18 min, and 33 of 60 fixtures have a gap longer
  than 15 min.
- **Mapping:** `market_metadata.event_id` joins to `betfair.match_meta` (70 of 70 rows verified), which
  gives the sofascore `match_id`. Runners arrive already normalised to `home`/`draw`/`away`. Betfair
  `open_date` equals the sofascore kick-off for all 70 markets.
- **Results and model inputs** (fresh `ScottishPyramid` DataStore from betdb, 5,323 matches):
  - Every 26/27 L1/L2 fixture in betdb (79, through 2026-09-19) has a result.
  - The 09-26 round is not in betdb yet, and not in `betfair_live` either.
  - BBC live-text commentary (the joint model's pxG) covers the L1/L2 league rounds through 08-22 (39
    of 79 fixtures) and **none of the league rounds from 08-28 on**. Cup midweeks are covered.
  - The joint model masks missing pxG (Poisson arm only), so this is not blocking. It affects only
    fold 44's training: 48 of its 115 26/27 fixtures carry no pxG.
- **Collector:** the newest row in `order_book_1m` is 2026-09-20 12:56 UTC, and the newest
  `market_metadata` row is 09-20. **The collector appears to have written nothing since 20 September.**

Coverage by match day (L1/L2, 26/27). "First book" is the first snapshot with all three runners
two-sided, in minutes before kick-off. Instant columns count fixtures with a complete two-sided book
no more than 15 min old.

| Date | Played | Market | Pre-off snapshots | First book (median, min) | earliest | T−24h | T−6h | T−3h | T−60m | T−25m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 08-01 | 10 | 10 | 10 | 1,685 | 10 | 4 | 8 | 10 | 9 | 10 |
| 08-08 | 10 | 10 | 10 | 120 | 10 | 0 | 0 | 0 | 10 | 10 |
| 08-15 | 9 | 9 | 9 | 113 | 9 | 0 | 0 | 0 | 7 | 9 |
| 08-22 | 10 | 10 | **0** (first tick 14:06, after the off) | — | 0 | 0 | 0 | 0 | 0 | 0 |
| 08-28 | 1 | 1 | 1 | 103 | 1 | 0 | 0 | 0 | 1 | 1 |
| 08-29 | 9 | **0** | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 |
| 09-05 | 10 | 10 | 10 | 238 | 10 | 0 | 0 | 10 | 6 | 10 |
| 09-12 | 9 | 9 | 9 | 168 | 9 | 0 | 0 | 0 | 9 | 9 |
| 09-15 | 1 | 1 | 1 | 88 | 1 | 0 | 0 | 0 | 1 | 1 |
| 09-19 | 10 | 10 | 10 | 203 | 10 | 0 | 0 | 8 | 10 | 10 |
| **total** | **79** | **70** | **60** | **185.5** (range 86–1,685) | **60** | **4** | **8** | **28** | **53** | **60** |

Book quality by time to the off (all two-sided runner rows):

| Band | Fixtures | Median spread (ticks) | Median spread (% of back) | Median best-back size (£) | Median best-3 back depth (£) | Median market matched (£) | LTP present |
|---|---:|---:|---:|---:|---:|---:|---:|
| > 24h | 6 | 9 | 12.3% | 10.7 | 81 | — | 0% |
| 6–24h | 10 | 6 | 8.3% | 6.5 | 80 | 131 | 0% |
| 3–6h | 30 | 4 | 5.0% | 5.8 | 30 | 394 | 32% |
| 1–3h | 60 | 4 | 5.1% | 5.9 | 39 | 476 | 79% |
| 25–60m | 60 | 4 | 4.9% | 7.0 | 52 | 551 | 83% |
| 0–25m | 60 | 3 | 4.2% | 15.0 | 62 | 1,191 | 82% |

### 13.2 Step 2: the walk-forward extension (the one sampling job)

- **What was sampled:**
  - model: the persisted fit's own model;
  - sampler: `QueuedNUTSConfig`, 1,000 draws, 500 warmup, 4 chains, the original settings;
  - execution: `QueuedExecution(16)`, `-t 16`, threads pinned;
  - splitter: the run's `ScopedWalkForwardCV` scope with "26/27" appended to `target_seasons`.
- **Checks before sampling:**
  - positions 1–40 of the extended splitter hold out exactly the 1,461 fixtures the persisted fit
    holds out;
  - only positions 41–44 were sampled;
  - every training fixture sits on a calendar day strictly before the fold's first held-out day
    (a Date comparison, per the LastHistorical note);
  - there is no lineup input in this model.
- **Time:** 2.4 min of wall time.
- **Persistence:** nothing was written to `mcmc_experiments`. The extension (folds, chains,
  diagnostics, latents) is serialised to `/root/BF_runs/clv_napkin_dev/out/r04_ext_grw_spfl_joint.jls`
  on the beast. The rebuilt DataStore sits next to it.
- **Deviation from the brief's wording.** The brief says "train on everything before the match
  day's date, then predict that day". W1's folds do not do that: `ScopedWalkForwardCV` holds out
  the next **biweekly** clock step and trains strictly before the step's first day. I kept W1's
  splitter, which is what `extend_fit` would use and what "the same way the W1 folds did" means in
  code. As a result, a step's later days are predicted with one or two weeks' less data. Example:
  09-05 is predicted from data through 08-22.

| Fold | Held-out L1/L2 days | Train fixtures (26/27, with pxG) | Max R̂ (param) | Min bulk / tail ESS | Divergences | Rerun? |
|---|---|---|---|---|---|---|
| 41 | 08-01, 08-08 (20) | 1,461 (0, 0) | 1.0032 (`dyn.β.σ₀`) | 1,477 / 2,116 | 0 / 4,000 | no |
| 42 | 08-15, 08-22 (19) | 1,503 (42, 42) | 1.0037 (`dyn.α.σ₀`) | 982 / 1,643 | 0 / 4,000 | no |
| 43 | 08-28, 08-29, 09-05 (20) | 1,528 (67, 67) | 1.0060 | 919 / 1,649 | 0 / 4,000 | no |
| 44 | 09-12, 09-15, 09-19 (20) | 1,576 (115, 67) | 1.0077 | 1,140 / 1,860 | 0 / 4,000 | no |

- **Diagnostics:** tree-depth saturation was 0% and min BFMI was 0.60. The run passes
  `audit_convergence` against the run's own thresholds. All 79 L1/L2 fixtures and 77 54/55 monitor
  fixtures have held-out latents.

### 13.3 Per instant: executable back price (1X2, edge ≥ 2 pp vs the de-vigged mid, 1 unit, 2% commission)

- **Close:** the de-vigged mid-price TWA over (−20, 0].
- **Size:** the size offered at the price taken. ≥ £20/£50/£100 is the share of bets whose taken
  price had at least that much on offer.
- **CIs:** slate-bootstrapped, B = 2,000.

| Entry | Fixtures with book | Bets | Slates | Mean edge (pp) | CLV pp [CI] | EV@close 2% [CI] | ROI 2% [CI] | Median back size (£) | ≥ £20 | ≥ £50 | ≥ £100 | Median spread (ticks) | Mean odds |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| earliest | 60 | 72 | 8 | 9.7 | −1.96 [−4.58, −0.52] | −3.9 [−10.1, −0.6] | −20 [−53, +7] | 5.8 | 26% | 11% | 3% | 6 | 4.14 |
| T−24h | 4 | 3 | — | — | — | — | — | 6.2 | 0% | 0% | 0% | 22 | — |
| T−6h | 8 | 7 | 1 | 14.5 | −0.57 (1 slate) | −5.5 (1 slate) | −16 (1 slate) | 4.0 | 29% | 14% | 0% | 4 | 3.64 |
| T−3h | 28 | 33 | 3 | 11.8 | −0.73 [−1.77, +0.06] | −2.0 [−7.5, +2.0] | +1 [−25, +29] | 5.0 | 24% | 9% | 0% | 4 | 3.99 |
| **T−60m** | 53 | 61 | 8 | 10.3 | **−0.59 [−1.19, −0.09]** | **−0.9 [−4.9, +1.9]** | −27 [−57, +8] | 6.0 | 20% | 8% | 0% | 4 | 4.32 |
| **T−25m** | 60 | 70 | 8 | 10.0 | **−0.60 [−0.91, −0.19]** | **−3.0 [−4.4, −1.2]** | −16 [−46, +16] | 9.4 | 37% | 13% | 4% | 4 | 4.06 |

The same bet set, priced three ways, plus the market-only null (back every selection at the back
price) and the §2 LTP-based close as a cross-check. The LTP rows are fewer because LTP is NULL on
08-01 and missing on about 20% of snapshots.

| Entry | Null bets | Null CLV pp [CI] | Null EV@close 2% [CI] | Model bets at MID: CLV / EV | At LTP: n, CLV / EV | At BACK vs the LTP close: n, CLV / EV |
|---|---:|---|---|---|---|---|
| earliest | 180 | −1.71 [−3.19, −0.93] | −4.3 [−8.1, −2.1] | +0.19 / +2.5% | 61: +0.05 / +2.8% | 61: −0.75 / −1.4% |
| T−24h | 12 | −2.55 (1 slate) | −7.1 (1 slate) | — | — | — |
| T−6h | 24 | −0.95 (1 slate) | −4.9 (1 slate) | +0.26 / −2.6% | — | — |
| T−3h | 84 | −0.90 [−1.01, −0.81] | −3.0 [−4.5, −1.1] | +0.15 / +1.5% | 23: +0.21 / +3.2% | 23: −0.20 / +0.5% |
| T−60m | 159 | −0.74 [−0.89, −0.64] | −2.2 [−3.7, −1.2] | +0.20 / +2.4% | 52: +0.10 / +2.2% | 52: −0.32 / +0.3% |
| T−25m | 180 | −0.76 [−0.84, −0.69] | −3.4 [−3.6, −2.9] | +0.19 / +0.0% | 60: +0.19 / +0.8% | 60: −0.46 / −2.6% |

- **The LTP close is kinder than the mid close.** From T−3h on it lifts back-price CLV by
  0.1–0.5 pp, on smaller bet sets. The LTP sits 0.1–0.4 pp nearer the back side than the mid does
  (§13.4).
- **Neither close makes the back-price bets clearly profitable.** Only T−3h (+0.5%, 23 bets) and
  T−60m (+0.3%) show point estimates above 0 against the LTP close.

### 13.4 Spread and size: what §10–§12 lose once the entry is executable

Paired on the same bets (those with an LTP at entry). The first two columns are how many probability
points the back price costs relative to the mid and to the LTP. The last three are EV@close (2%) at
each price.

| Entry | Bets | Back − mid (pp) | Back − LTP (pp) | EV at mid | EV at LTP | EV at back |
|---|---:|---:|---:|---:|---:|---:|
| earliest | 61 | +0.98 | +0.86 | +2.7% | +2.8% | −1.4% |
| T−3h | 23 | +0.87 | +0.49 | +4.2% | +3.2% | +0.4% |
| T−60m | 52 | +0.80 | +0.50 | +3.7% | +2.2% | +0.3% |
| T−25m | 60 | +0.79 | +0.67 | +0.6% | +0.8% | −2.5% |

**What the spread costs.**
- The LTP-based entries of §10–§12 were half to two-thirds of a spread better than any price you
  could actually back at. The back price is 0.5–0.9 pp of probability worse than the LTP and about
  0.8–1.0 pp worse than the mid.
- Measured at the LTP, this season's bets reproduce the shape of §10 (+2% to +3% EV@close from T−3h
  to T−60m, falling to about +1% at T−25m). At the back price, the same bets are −2.5% to +0.4%.
- §10 put the biggest 56/57 1X2 CLV at +0.5 to +0.9 pp. That is smaller than the median 4-tick
  spread, which is about 5% of the price at 1–3 h out and still 4% in the last 25 minutes.

**What size is on offer.** The queue is small:
- The median best-back size is £5–7 from T−6h to T−25m, rising to £15 only in the last 25 minutes.
- The best three back levels together hold a median £30–60.
- Matched volume in the whole market is a median £475 at 1–3 h out and £1,190 in the last 25 minutes.

So a 1-unit bet above about £10 would walk the book. This matches the §10 finding that early LTPs
were thin and stale: in League One/Two, "early" means about 3 h and a £6 queue, not 24 h.

### 13.5 §12's AGAINST/WITH split at T−60m (descriptive, not a test)

Direction is the change in the de-vigged mid probability of the selection, from the earlier instant
to T−60m. "WITH" means it shortened toward the model's side by more than 0.1 pp. Bets are at the back
price, edge ≥ 2 pp.

| Since | Group | Bets | CLV pp [CI] | EV@close 2% [CI] |
|---|---|---:|---|---|
| T−3h | all | 26 | −0.46 [−1.72, +0.43] | −0.8 [−7.7, +4.9] |
| T−3h | price WITH model | 9 | +0.43 [−0.48, +2.44] | −1.2 [−4.1, +3.8] |
| T−3h | price AGAINST model | 11 | −1.60 [−2.71, +0.31] | −4.5 [−10.5, +7.7] |
| T−3h | flat | 6 | +0.30 [−0.49, +1.88] | +6.4 [+3.3, +12.6] |
| earliest book | all | 61 | −0.59 [−1.19, −0.09] | −0.9 [−4.9, +1.9] |
| earliest book | price WITH model | 19 | +0.01 [−1.08, +1.21] | −1.8 [−5.9, +2.6] |
| earliest book | price AGAINST model | 34 | −1.34 [−2.11, −0.56] | −4.1 [−8.5, +0.1] |
| earliest book | flat | 8 | +1.16 [−0.78, +4.21] | +15.0 [−0.2, +42.3] |

- **The sign is the reverse of §12:** here AGAINST is worse than WITH. The T−3h version has only 20
  directional bets on 3 match days, because the book rarely exists at T−3h. That is too few to confirm
  or refute the §12 candidate.
- **Since the earliest book (a variant the brief did not ask for),** 34 AGAINST bets lost 1.3 pp of
  CLV. The transition-club fixtures of §13.6 are not what drives this: they make up 13 of the 34
  AGAINST bets and 6 of the 19 WITH bets.

### 13.6 Where the disagreement lives: clubs that changed tier

- **The clubs:** Airdrieonians and Ross County came down from the Championship into League One,
  East Kilbride came up from League Two, and Kelty Hearts went down from League One to League Two. They appear in 28 of the 79 fixtures.
- **The model's edge there is about double** (13–17 pp vs 7–8 pp). These are the fixtures where the
  model is worst: 1X2 log-loss 0.993 vs the mid close's 0.827 on 22 of them, against 1.029 vs 1.018 on
  the other 38.
- **Example:** Peterhead v Ross County. The model had Peterhead at 0.53 to win; the mid close had
  0.17.
- **Why:** this is the known W1 transition bias (see the W1 README). GRW has no summer macro step, so a
  club's rating enters the new season almost unchanged. Pooling over the SPFL does not tell the model
  that a relegated Championship side is a League One favourite.

| Entry | Group | Bets | Mean edge (pp) | CLV pp [CI] | EV@close 2% [CI] | ROI 2% |
|---|---|---:|---:|---|---|---:|
| earliest | transition fixture | 30 | 13.1 | −2.06 [−4.79, +0.23] | −3.7 [−15.2, +7.0] | −48% |
| earliest | other | 42 | 7.2 | −1.89 [−4.53, +0.23] | −4.1 [−9.0, +1.1] | +1% |
| T−3h | transition fixture | 14 | 16.9 | −2.05 [−4.75, −0.65] | −6.1 [−19.5, −0.0] | −30% |
| T−3h | other | 19 | 8.1 | +0.25 [−0.42, +1.06] | +1.0 [−1.6, +4.7] | +23% |
| T−60m | transition fixture | 25 | 14.6 | −0.99 [−2.76, +0.90] | −0.8 [−11.1, +11.7] | −42% |
| T−60m | other | 36 | 7.4 | −0.31 [−1.59, +1.04] | −1.0 [−6.3, +4.5] | −17% |
| T−25m | transition fixture | 28 | 14.2 | −0.75 [−1.28, −0.23] | −3.2 [−5.8, −0.8] | −38% |
| T−25m | other | 42 | 7.2 | −0.50 [−0.74, −0.07] | −2.9 [−3.9, −0.7] | −2% |

Excluding those fixtures does not rescue the model at the back price. At T−25m, the other fixtures
still have CLV −0.50 [−0.74, −0.07] and EV −2.9%.

### 13.7 What this sample can and cannot say

- **Size of the sample:** 60–70 bets per usable instant, clustered in 8 match days. T−3h has 33 bets
  on 3 days; T−6h and T−24h have too few to use.
- **CI widths (edge ≥ 2):**

  | Entry | CLV CI width | EV CI width | ROI CI width |
  |---|---:|---:|---:|
  | T−60m | 1.1 pp | 6.8 pts | 64 pts |
  | T−25m | 0.7 pp | 3.2 pts | 62 pts |

- **Detectable effects:** with 80% power at a two-sided 5% level, the minimum detectable effect is
  about 1.43 × the CI half-width.

  | Effect | T−60m | T−25m |
  |---|---:|---:|
  | CLV | 0.8 pp | 0.5 pp |
  | EV@close | 4.9% | 2.3% |
  | Realised ROI | 46 pts | 44 pts |

- **What it can say.** A CLV of −0.6 pp is the size of the spread, and the data can see that. A true
  EV of +1% to +2%, the size of §10's early edge, is below what it can detect.
- **What it cannot say.**
  - Realised ROI over two months says nothing.
  - An 8-cluster bootstrap understates the uncertainty.
  - Nothing here tests a hypothesis. It describes one model on one partial season.

### 13.8 Caveats

- **One model.** It is W1's `grw_spfl_joint`: GRW, the most decompressed cell (slope 1.07), with a
  known transition bias. The fixed pre-registration candidate from §12.8 (m05_joint_td) was not
  extended, since the brief allowed one job.
- **Biweekly folds, as in W1** (§13.2). The second day of a two-week step is predicted with up to two
  weeks' less data than a weekly refit would have. This matters most for the transition clubs, whose
  ratings are moving fastest.
- **Missing pxG in fold 44** (09-12 to 09-19): 42% of its 26/27 training fixtures lack BBC commentary.
  A BBC backfill would change fold 44's inputs.
- **Holes in the book.**
  - The 08-22 round was recorded only after kick-off, and the 08-29 round was not recorded at all.
  - The 08-01 round has no LTP.
  - The collector stopped on 20 September.
  - 19 of the 79 played fixtures could not be priced.
- **Execution is idealised.** The fill is assumed at the best back for the full unit, with no queue
  priority or latency, and on a 3-minute snapshot rather than the true minute. Where the unit exceeds
  the best-back size (most bets above about £6), the real price would be worse.
- **Mid close vs LTP close.** The mid close is the brief's definition. It is noisier far from the
  off, when spreads are wide, but in (−20, 0] the median spread is 3 ticks. The LTP close is shown as
  the §2 cross-check.

**Reproduce** (mcmc-beast, `/root/BF_runs/clv_napkin_dev`, `.env` sourced):

```bash
# once: rebuild the DataStore from betdb (read-only) into out/, then the one sampling job (~3 min)
julia --project -e 'using BayesianFootball, Serialization; mkpath("/root/BF_runs/clv_napkin_dev/out"); serialize("/root/BF_runs/clv_napkin_dev/out/r04_datastore_ScottishPyramid.jls", BayesianFootball.Data.load_datastore_sql(BayesianFootball.Data.ScottishPyramid()))'
julia --project -t 16 current_development/clv_napkin/r04_extend_grw_spfl_joint.jl > /root/BF_runs/logs/clv_napkin/r04_extend.log 2>&1
# evaluation (~1 min; SELECTs from betfair_live)
julia --project -t 8 current_development/clv_napkin/r04_live_orderbook.jl
```

## 14. How the market prices matches

Brief 05. Code: `l05_market_structure.jl` / `r05_market_structure.jl`; results:
`out/r05_*.csv` and `out/r05_heatmap.txt`. **No new posterior fit or MCMC.**

**Answer.** At the first available close, the market rates clubs mainly by their *previous
season's strength and tier*, not by a flat newly promoted/relegated prior: last-season goal
difference plus tier-change indicators explain 70% of the in-sample variation in early-season
market strength (173 club-seasons), and last-season *market* strength adds explanatory power on
its paired sample. Transition clubs are **already substantially re-rated at their first quoted
match**; promoted clubs' first-match strength is near their later-season level on average, while
relegated clubs can still move, but the latter estimate is too noisy to put a precise number of
weeks on adaptation. The correct-score book is not independent Poisson: it puts extra mass on
1-1 and 2-2 and less on 1-0/0-1. Of the grids tested, a **shared-goals bivariate component
(λ₃ ≈ 0.08) with mildly under-dispersed COM-Poisson marginals (ν ≈ 1.08)** best *describes* the
CS quotes; a simple bivariate Poisson (λ₃ ≈ 0.09) also beats Poisson and DC (ρ ≈ −0.04).
**Recommendation:** first fix cross-tier initial states (seed from prior-season market strength
where available, and test a tier-aware fallback); separately prototype shared-goal covariance
and test its proper scores prospectively. Merely swapping our score grid cannot repair the
much larger model–market **rate** gap on transition fixtures.

### 14.1 Data, inversion and rating definition

- **Archive:** Scottish Premiership / Championship / League One / League Two (tournaments
  54–57), 21/22–25/26. Core close is a de-vigged last-20-minute TWA of Betfair LTP from
  `betfair.odds_history`, as in §2. The CS comparison instead uses each runner's last
  pre-kick-off trade at most 120 minutes old: require at least 16 of 19 runners including
  the low-score block, normalise over the **present runners**, and compare grids on that
  same conditional set. These are asynchronous, thin exchange quotes, **not** a simultaneous
  complete de-vigged CS book; this is the most important limitation on claims about its shape.
- **Live:** 26/27 `betfair_live.order_book_1m`, two-sided best-back/best-lay mid, last-20-minute
  TWA, de-vigged for 1X2 and goal lines. As in §13, this covers dates only through 20 September.
- **Inversion:** minimise summed KL to de-vigged 1X2 and the available under 1.5/2.5/3.5
  probabilities using the chosen bivariate + COM grid. Report λ as its *actual expected
  goals* (the bivariate component contributes to both sides), not just its independent
  component rates. A two-parameter grid cannot fit all independent prices exactly. The
  independent-Poisson 1X2 + under-2.5 inversion is retained in `r05_market_lambda.csv`.
  There are 2,356 archive and 113 live match inversions; the mean summed inversion KL is
  0.00039. Only 1X2 and under 2.5 are needed for inclusion.
- **Ratings:** per league-season, least squares on
  `log λ_h = μ + H + attack_h − defence_a` and
  `log λ_a = μ + attack_a − defence_h`, with attack/defence random walks by *team match*,
  ridge identification, and a common smoothing penalty. Earlier-date one-step error chooses
  penalty 4 (log-rate RMSE 0.0954, vs 0.0959 at penalty 1); the resulting full-season
  *smoothed* paths and opponent-adjusted per-match `net = attack + defence` are in
  `r05_rating_paths.csv`. A season's full path uses later quotes to smooth earlier ratings;
  for a first-match-vs-settled comparison use the **per-match quote** `net`, not the
  full-season smoothed estimate. Different league-season centres are not comparable as an
  absolute cross-tier rating without the tier label.

| League | Played 21/22–25/26 | Core 1X2 + O/U 2.5 | Also O/U 1.5 & 3.5 | With usable CS | Complete 19-runner CS | 26/27 live core |
|---|---:|---:|---:|---:|---:|---:|
| Premiership | 990 | 897 | 641 | 764 | 485 | 27 |
| Championship | 895 | 391 | 177 | 122 | 60 | 27 |
| League One | 895 | 589 | 180 | 98 | 23 | 29 |
| League Two | 895 | 479 | 130 | 60 | 13 | 30 |

The CS results are heavily Premiership-weighted (764/1,044); lower-division estimates are
thin. The 26/27 live book has a CS set on 112 of 113 inverted fixtures.

### 14.2 What predicts the market's team ratings?

Early strength is the within-league-season-centred opponent-adjusted net rating averaged
across the first three **available** matches (at least two). Prior tier and final position,
last-season goal difference per game, and the previous market rating use the preceding
season's data. The time-variation outcome for first weeks is the change from matches 1–3
to matches 6–10; 26/27 has not reached a settled level (defined here using matches 16+).
These are descriptive, in-sample OLS R², **not** predictive or causal explanations; several
club-seasons share a team and league and the reported HC1 standard errors are not
team-clustered.

| Explanatory block | Outcome / sample | n | In-sample R² | Main coefficients (log net-strength units) |
|---|---|---:|---:|---|
| Tier-change indicators only | season start, all four leagues | 173 | 0.031 | promoted −0.09; relegated +0.13 vs stayed |
| + previous final table | same | 173 | **0.705** | prior GD/game × stayed +0.63 (SE 0.06); × promoted +0.27 (0.08); × relegated +0.23 (0.10); prior rank adds little conditional on GD |
| Previous table alone | paired with next row | 126 | 0.692 | prior GD/game × stayed +0.62 |
| + previous market rating | same 126 | **0.790** | rating carry × stayed +0.92 (SE 0.14); × tier movers +0.19 (0.37), imprecise; prior GD effects shrink |
| Previous table alone | L1/L2, *matched* lineup sample | 47 | 0.539 | — |
| + starting-XI wealth, turnover | same 47 | 0.549 | wealth +0.077 (SE 0.089); turnover −0.073 (0.169) |
| Previous table | settled strength, all leagues | 186 | 0.506 | prior GD/game × stayed +0.58 |
| Previous table | change from matches 1–3 to 6–10 | 125 | 0.060 | no robust tier-change slope |

The lineup rows use a *proxy* of `src/models/pregame/builder/components.jl`'s
`ProductionWealthFeature`: age-weighted log starting-XI `proposed_market_value`, averaged
over a team's first three covered games, and starter turnover relative to the previous
season. The production feature instead uses matchup log-wealth differences and
kickoff-safe valuation fallback; do **not** interpret this proxy as a pixel-identical
ablation of `m05_joint_production_wealth_grw`. Of 237 team-seasons, 174 have no usable
wealth measure. Comparing the 80-row L1/L2 table-only R² (0.375) to the 47-row wealth R²
(0.549) would be a sample-composition error: **the paired lift is only 0.010**. Player
ratings exist in the SofaScore lineups, but coverage and point-in-time availability did
not support a reliable extra pre-season rating regressor here. Full-time/part-time status
is **not** a populated historical betdb feature. The separate research panel in
`experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/STATUS_NOTES.md`
has 21 Verified, 8 Inferred and 223 Unknown season labels; it is retrospectively sourced,
not a point-in-time full-census input, so no FT coefficient is claimed.

**How fast for tier movers?** In historical seasons with an available match-16+ level,
mean *first-match minus settled* opponent-adjusted net strength is −0.016 for promoted
clubs (16 first-match quotes; bootstrap CI [−0.110, +0.069]), −0.126 for relegated clubs
(12; [−0.312, +0.058]), and +0.027 for stayers (109). The mean gap for promoted clubs
is near zero also across matches 6–10; relegated-club point estimates vary by round
rather than tracing a clean decay. Season-start (matches 1–3) to settled changes average
−0.019 for 23 promoted club-seasons and +0.049 for 22 relegated club-seasons with both
endpoints (`r05_transition_analogues.csv`). **Conclusion:** the first market quote
already incorporates the new tier; the data cannot identify a universal convergence
half-life. Missing early quotes and using an end-of-season reference both matter.

The four 26/27 transition clubs (rates in goals per match from the **same fixture**, first
available live close vs held-out `grw_spfl_joint`; home/away oriented to the club):

| Club, change | First quoted match | Market λ for / against | GRW λ for / against | Market net / GRW net | Later quoted market net (last available) |
|---|---|---|---|---|---|
| Airdrieonians, Champ → L1 | 08-01 at Cove (away) | 1.58 / 1.08 | 0.84 / 1.51 | +0.11 / −0.85 | +0.14 on 09-19 |
| East Kilbride, L2 → L1 | 08-01 vs Queen of the South | 2.06 / 1.04 | 1.66 / 1.23 | +0.19 / −0.19 | +0.19 on 09-19 |
| Kelty Hearts, L1 → L2 | 08-01 at Forfar (away) | 1.57 / 1.17 | 0.93 / 1.49 | +0.07 / −0.70 | −0.11 on 09-19 |
| Ross County, Champ → L1 | 08-01 at Peterhead (away) | 1.86 / 0.93 | 0.97 / 1.59 | +0.70 / −0.49 | +0.64 on 09-19 |

These *first quotes*, not retrospectively smoothed priors, already see Ross County as
strong. The different opponents/venues make a raw λ sequence **not** a rating path; the
last column uses the opponent-adjusted rating. No 26/27 club has a match-16+ settled
level yet. Full fixture-level examples (23 quotes) are in `r05_transition_2627.csv`.

On the matched held-out fixtures, model-minus-market log net strength for
`grw_spfl_joint` is +0.402 on promoted teams' first five matches (42 team-games),
−0.242 on relegated teams' first five (37), and −0.024 on stayers (281); at match 6+
it is +0.115 / −0.105 / −0.004 respectively. `m12_td` on 56/57 has +0.295
(promoted, n=13), −0.009 (relegated, n=22), −0.028 (stayed, n=73) in the first five.
These are aggregates across different seasons, and **do not** mean every transition
moves in the same direction: the individual 26/27 Ross County error is far larger.
Among ≥8-match team-seasons the largest absolute discrepancies include
`m12_td` Inverness 25/26 (−0.603), East Kilbride 25/26 (−0.359) and
`grw_spfl_joint` Celtic 24/25 (−0.374); see `r05_model_vs_market_team.csv`.
The grw model's matched CS sample has 1X2 KL(market‖model) ×1,000 of 23.5
and under-2.5 KL ×1,000 of 27.4 (463 fixtures); m12 has 39.8 and 5.7 (77).
Those marginal mismatches are far larger than the grid-shape differences below.

### 14.3 What is the market's score grid?

Per fixture, each candidate fits **two** rate parameters to 1X2 + under 2.5;
its global shape parameter is then chosen to minimise mean KL(market CS ‖ grid CS),
conditional on the CS runners present. All parameter CIs and KL-difference CIs resample
**match dates** (2,000 cluster draws), conditional on this grid of candidate values.
Thus the numbers are in-sample descriptions, not a held-out model contest, and should
not be read as the exchange's literal generative algorithm.

| Grid | Scottish n=1,044: mean CS KL ×1,000 | ΔKL vs Poisson ×1,000 [95% CI] | CS shape (95% CI) | Independently fitted from goal lines¹ |
|---|---:|---|---|---|
| Independent Poisson | 4.990 | 0 | — | — |
| Dixon–Coles | 4.486 | −0.505 [−0.541, −0.469] | ρ −0.04 [−0.04, −0.04] | ρ −0.07 |
| Bivariate Poisson | 3.849 | −1.141 [−1.213, −1.068] | shared λ₃ 0.09 [0.09, 0.09] | λ₃ 0.07 |
| Negative binomial | 4.990 | 0 | κ 0 (Poisson boundary) | κ 0 |
| COM-Poisson marginals | 4.205 | −0.785 [−0.856, −0.708] | ν 1.08 [1.08, 1.08] | ν 1.10 |
| Simple diagonal inflation | 4.277 | −0.713 [−0.778, −0.643] | δ 0.09 [0.09, 0.09] | δ 0.10 |
| DC + COM | 3.703 | −1.287 [−1.361, −1.212] | (ρ, ν) (−0.04, 1.08) | (−0.07, 0.98) |
| **Bivariate + COM** | **3.083** | **−1.908 [−2.002, −1.812]** | **(λ₃, ν) (0.08, 1.08); bootstrap λ₃ [0.08, 0.09], ν 1.08** | (0.10, 1.18) |

¹ Goal-line column: choose the shape solely by best simultaneous fit to 1X2 and under
1.5/2.5/3.5 on matches with all three lines; **not** the CS sample or the same
objective, so disagreement (especially ν) is a consistency warning. CIs that print as
a single grid step do not imply exact identification. `r05_grid_fit.csv` has all four
leagues and four expected-total bins. By league the CS-selected simple DC ρ is −0.04,
−0.05, −0.04, −0.04 and bivariate λ₃ is 0.09, 0.10, 0.10, 0.09
(Prem / Champ / L1 / L2); the combined bivariate+COM advantage is present in all
four, but L1/L2 contain only 98/60 matches. The DC correction weakens from
ρ ≈ −0.05 at expected total <2.4 to ≈ −0.02 at ≥3.2, whereas shared λ₃
remains ≈0.09–0.10. A liquid-league reference, the English Premier League
(n=1,486 CS books), has the **same** winner: KL 2.163 vs Poisson 3.890,
λ₃=0.08, ν=1.08 (`r05_grid_fit_epl.csv`).

Mean market-minus-fitted-Poisson CS residuals (percentage points; conditional on
available CS runners), and the residual after the best bivariate+COM shape:

| Residual | 0-0 | 1-0 | 0-1 | 1-1 | 2-2 | CS draw total² | 1X2 draw³ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poisson | +0.12 | −0.37 | −0.47 | **+0.71** | **+0.55** | **+1.60** | +1.55 |
| Bivariate + COM | +0.16 | −0.35 | −0.41 | +0.27 | +0.16 | +0.65 | +0.60 |

² Sum over the draw CS runners present, including Any Other Draw. ³ 1X2 market draw
minus the *grid fitted jointly to 1X2 and O/U 2.5*, not an independently fitted draw.
The Poisson CS draw residual is +2.19 pp for expected total <2.4 but only
+0.28 pp at ≥3.2; full league/total splits with date-cluster intervals are in
`r05_residuals.csv`. The 4×4 ASCII residual heatmaps (and EPL control) are in
`r05_heatmap.txt`. Even the winning candidate leaves a structured 1-0/0-1
deficit and 1-1 excess: do not hard-code it as a final observation model.

**Our models are not showing that market pattern.** Their own posterior-mean CS
minus a Poisson fit to their *own* 1X2 + under 2.5 is *negative* at 1-1
(−0.10 pp m12, −0.28 pp grw) and 2-2 (−0.06 / −0.19); market minus the
same Poisson reference is +0.71 / +0.55. These comparisons mix posterior
draws differently and are diagnostic rather than a likelihood identification.
On the **same** 77 m12 CS fixtures, market-CS KL ×1,000 is 60.76 for the model's
own grid, 61.15 if its core probabilities are priced with Poisson, and 62.87
if priced with the CS-selected shape. On the 463 grw fixtures: 69.07, 67.64,
68.23 respectively. When instead the *market rates* are inverted and priced,
Poisson / best-grid KL falls to 8.90 / 6.91 (m12 panel) and 5.28 / 3.04
(grw panel). Therefore transplanting the market grid **without** correcting
rates is not supported as an improvement. `r05_model_grid_shape.csv` and
`r05_model_grid_kl.csv` preserve both comparisons.

**Thinner-market consistency (descriptive, not a bet):** relative to the grid
fitted only to 1X2 + under 2.5, the archive's median-style *mean absolute*
deviations are 0.92 pp for BTTS (market bias −0.47 pp), 0.69 pp for under
1.5 (bias −0.55) and 0.60 pp for under 3.5 (bias +0.30). In the live
26/27 mid book, they are 1.16 / 0.55 / 0.46 pp respectively. CS-cell
absolute deviations are about 0.25–0.48 pp for the best grid, on the
conditional runner set. `r05_consistency.csv` has counts, 90th-percentile
absolute deviations and Poisson controls; its CS and goal-line numbers
cannot be directly equated to executable cross-market arbitrage once spread,
commission, missing runners and asynchrony are considered.

**Reproduce:** on `mcmc-beast`, in `/root/BF_runs/clv_napkin_dev` with the
existing `out/probs.jls`, §13's extension and rebuilt datastore available;
source `.env` without printing it, then run
`julia --project -t 16 current_development/clv_napkin/r05_market_structure.jl`.
It issues read-only betdb SELECTs and loads the persisted fit for pricing,
but does **not** run sampling. The matched wealth-sample baseline was also
recomputed in the same warm session after the original full runner run.

## 16. Half-time re-pricing

Brief 07. Code: `l07_halftime.jl` (loader) and `r07_halftime.jl` (runner). Results are in
`out/r07_*.csv` and the run log is `out/r07_run.log`. There is no MCMC; the only fits are Poisson
GLMs (IRLS) and 1-D maximum likelihood. (§15 is left for brief 06.)

**Answer.**
- **Does the first half tell us more than the HT market? No, not in anything we can measure.**
  - Scored on the full-time result, the Betfair price during the break beats every HT model on
    the 24/25–25/26 test set: 736 matches in 54–57.
  - The full model (anchor + HT score + reds + first-half pxG and on-target gaps) is **worse**
    than the market by +0.013 nats of 1X2 log-loss per match [+0.004, +0.024].
  - The first-half stats don't even beat the same model without them, fitted on the same 23/24
    season.
  - Put the stats on top of the HT market's *own* second-half rates and they add nothing
    (Δ +0.004 [−0.000, +0.009]).
  - The best the model can do is contribute weight 0.13 to a log-pool with the market, and that
    pool ties the market: Δ +0.0005 [−0.0006, +0.0017].
- **Where?** Nowhere. The shortfall is largest exactly where an edge would plausibly live:
  - level games (+0.029);
  - games where a side dominated the first half on pxG without leading (+0.032, CI above 0);
  - League One and League Two (+0.022 and +0.038).

  The market already prices the first half at least as well as our stats, and the model
  over-reacts to them.
- **Does it survive the spread?** In arithmetic only.
  - The 26/27 order book in the break has a 3–4 tick spread (1.0–1.6 pp) and £11–28 median back
    size.
  - A 2 pp-rule edge mostly clears that spread (93–96% of bets), but that edge is not real.
    Against the HT last traded price over two test seasons, the rule's bets make
    +0.002 units a bet [−0.13, +0.15] for the full model.
- **Does model-guided closing beat holding? No.**
  - "Close a pre-match back when the model values it below the lay price" is within ±0.03 units
    of "always hold" for every model, with CIs spanning 0.
  - That is before paying the in-play spread, which the archive (LTP only) cannot charge.
- **For the model:** re-pricing at half-time is not an edge. If anything is to be taken from
  this:
  - the HT market is a strong **anchor** for an in-play model, not a target to beat;
  - first-half pxG carries a small real signal about second-half scoring (β ≈ +0.23 per pxG
    above expectation, z 2.4, in-sample 23/24), but the market already prices it.

### 16.1 Data and definitions

- **HT market price:**
  - For each runner, the **median last traded price** of the ticks inside the half-time break.
    The markets are then de-vigged multiplicatively, with an overround gate of 0.9–1.1.
  - **The break comes from BBC live text:** from the first `half_end` post + 1 min to the
    second-half `kick_off` post − 0.5 min.
  - **Why not the brief's window:** `sofascore.matches.injury_time1` is 0 for most matches, so
    KO + 45 + added time would have included the end of the first half. The BBC break starts at
    median KO + 49.0 min (p10 47.1, p90 52.0) and ends at KO + 63.2 (p10 60.9, p90 66.5), against
    the sofascore kick-off.
  - **Fallback where BBC timing is missing:** [KO + 52.0, KO + 60.9], a window inside the break
    for about 80% of matches.
  - **Missing runner:** if exactly one runner has no tick in the break, it is imputed as
    1 − Σ(other implied) and flagged. Results are shown with and without those markets.
  - **Sanity check:** a side leading by 2+ at HT is the HT market favourite in **99.7%** of 380
    such matches.
- **HT state:**
  - The sofascore HT score.
  - First-half red cards (red or second yellow, minute ≤ 45) from `sofascore.match_incidents`,
    which covers every match.
- **First-half performance** from BBC live text (minute ≤ 45, added time included):
  - shots, shots on target and corners per side;
  - proxy xG from `src`'s shot-cell table (`Features.build_shots`, `fit_shot_xg`,
    `predict_xg`): 106 zone × body × context cells, base rate 0.125. It is **fitted on training
    matches only**.
  - Each stat is compared with what the pre-match anchor expected: a quasi-Poisson GLM of the
    first-half stat on log λ_for, log λ_against and home, fitted on 23/24.
  - Gaps are raw for pxG (obs − E) and Pearson for counts ((obs − E)/√E).
  - "Big chances" are not tagged in the BBC text, so they are not used.
- **Other sources checked:**
  - `sofascore.match_statistics` has per-period (`1ST`) stats **for the Premiership only**.
  - `match_shotmap` shot xG covers the Championship 23/24–25/26 and the Premiership 25/26 only.
  - BBC is the only source that covers all four leagues, from 23/24.
- **Pre-match anchor:**
  - The de-vigged close (TWA over (−20, 0]) for 1X2 + O/U 1.5/2.5/3.5, inverted with §14's best
    grid (bivariate + COM, λ₃ 0.08, ν 1.08).
  - λ_pre is the grid's goal means.
- **Model:**
  - Second-half goals of each team follow
    `g2 ~ Poisson(λ_pre · s₂ · exp(β'perf + γ'state))` (team rows, IRLS).
  - The state terms are: home; lead 1; lead 2+; trail 1; trail 2+; red for; red against; and HT
    goals (capped at 4).
  - Full-time 1X2 given HT comes from the HT score plus the two second-half Poissons, with
    full-time-draw cells inflated by δ (1-D ML on the training set). O/U 2.5 comes from the same
    grid, without δ.
- **Models compared:**
  - (0) the HT market.
  - (1) state only, fitted on 21/22–23/24.
  - (1′) state only, fitted on 23/24: the like-for-like control for (2).
  - (2) state + pxG gap for/against + on-target z for/against, fitted on 23/24 (BBC starts
    23/24).
  - (2b) (2) + shots and corners.
  - (3) a log-pool of (2) with the market, weight fitted on 23/24.
  - (4) the **HT market's own** second-half rates, from inverting its de-vigged 1X2 given the HT
    score (exactly identified; with β = 0 it reproduces the market to |ΔLL| = 2.6e-4), times
    exp(β'perf), fitted on 23/24.

  Model (4) is the direct test of "does the first half tell us more than the market".
- **Scoring:** mean FT 1X2 log-loss given HT (nats) and binary O/U 2.5 FT log-loss. Model minus
  market is resampled with a paired, match-day-clustered bootstrap (2,000 draws).

Coverage for the test seasons (`r07_coverage.csv` has every season):

| League | Season | Played | Anchor | HT 1X2 book | of which imputed runner | HT O/U 2.5 | BBC-timed break | BBC first half | SofaScore 1ST-period stats | SofaScore shot xG |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prem | 24/25 | 198 | 190 | 183 | 17 | 163 | 193 | 198 | 198 | 0 |
| Prem | 25/26 | 198 | 190 | 192 | 12 | 177 | 198 | 198 | 198 | 198 |
| Champ | 24/25 | 180 | 118 | 120 | 24 | 101 | 178 | 180 | 0 | 178 |
| Champ | 25/26 | 175 | 5 | 5 | 1 | 5 | 174 | 175 | 0 | 172 |
| L1 | 24/25 | 180 | 93 | 141 | 62 | 122 | 178 | 180 | 0 | 0 |
| L1 | 25/26 | 175 | 102 | 124 | 31 | 141 | 175 | 175 | 0 | 0 |
| L2 | 24/25 | 180 | 82 | 102 | 43 | 101 | 179 | 180 | 0 | 0 |
| L2 | 25/26 | 175 | 77 | 97 | 43 | 92 | 172 | 175 | 0 | 0 |

**Scored sample (anchor + HT book + BBC):** 736 test matches (Prem 367, Champ 104, L1 150,
L2 115). BBC covers every test match that has an HT book, so the "common" and "all with HT book"
rows in `r07_logloss.csv` are identical.

The Championship archive is nearly empty for 25/26, as in §14.

The training sets are:
- (1): 1,499 matches;
- (2), (1′) and (4): 463–542 matches, all from 23/24.

### 16.2 Does the first half beat the HT market? (FT 1X2 log-loss given HT)

Δ = model − market (positive = worse than the market); 95% match-day-clustered CI.

| Model | Pooled (n = 736) | Prem (367) | Champ (104) | L1 (150) | L2 (115) |
|---|---|---|---|---|---|
| (0) HT market, log-loss | 0.8008 | 0.7840 | 0.7552 | 0.8973 | 0.7701 |
| (1) state only (3 seasons) | +0.0060 [−0.0025, +0.0150] | +0.0004 [−0.010, +0.011] | −0.0028 [−0.016, +0.011] | +0.0169 [−0.006, +0.043] | +0.0174 [−0.011, +0.052] |
| (1′) state only (23/24) | +0.0108 [+0.0017, +0.0207] | +0.0028 | +0.0011 | +0.0198 | +0.0335 |
| (2) + pxG / on-target | **+0.0131 [+0.0035, +0.0238]** | +0.0059 [−0.007, +0.019] | −0.0023 [−0.019, +0.014] | +0.0224 [+0.003, +0.044] | +0.0383 [+0.005, +0.076] |
| (2b) + shots / corners | +0.0149 [+0.0054, +0.0250] | +0.0064 | +0.0033 | +0.0244 | +0.0403 |
| (3) log-pool (2) × market, w = 0.13 | **+0.0005 [−0.0006, +0.0017]** | −0.0001 | −0.0008 | +0.0013 | +0.0027 |
| (4) HT market rates × perf | **+0.0042 [−0.0005, +0.0089]** | +0.0022 | +0.0023 | +0.0017 | +0.0153 [+0.004, +0.027] |

Without markets that have an imputed runner (n = 597), the ordering is the same: (2) +0.016
[+0.004, +0.028], (3) +0.0009, (4) +0.0043 [−0.0005, +0.0092].

**Do the stats add to the score state?** Compare (2) and (1′), both fitted on 23/24:
+0.0131 vs +0.0108. So no.

**Coefficients** (`r07_coefficients.csv`):
- **Pxg is the only first-half stat with a signal.** In (2), the pxG gap *for* is
  +0.23 per pxG (z 2.4). Its against gap and both on-target terms are ≈ 0. With the HT market as
  the offset, (4), it falls to +0.13 (z 1.3).
- **Red cards matter:** a red for is −0.34 (z −2.2) and a red against is +0.34 (z 3.2), from
  (1) over three seasons.
- **One training season is unstable.** The "trail by 1" effect is +0.30 (z 3.0) on 23/24 alone
  but +0.06 over 21/22–23/24. That instability is why (1′) and (2) lose to (1) out of sample.
- **Second half:** its share of the match rate is s₂ = exp(−0.57) = 0.57. The full-time-draw
  inflation δ is +0.03 to +0.05, and +0.14 in (4).

**O/U 2.5 FT given HT** (n = 621 with an HT O/U book): the market's log-loss is 0.5435.

| Model | Δ vs market |
|---|---|
| (1) | +0.014 [+0.004, +0.025] |
| (2) | +0.012 [+0.001, +0.024] |
| (2b) | +0.010 [−0.001, +0.023] |
| (4) | +0.024 [+0.007, +0.045] |

Model (4) is inverted from the 1X2 alone, so its totals are not the market's O/U.

### 16.3 Where would an edge live? (common sample)

| Group | n | (2) Δ vs market | (4) Δ vs market | (3) Δ |
|---|---:|---|---|---|
| Level at HT | 272 | +0.029 [+0.007, +0.052] | +0.010 [+0.002, +0.018] | +0.0017 |
| One-goal lead | 325 | −0.0004 [−0.012, +0.010] | +0.0004 [−0.007, +0.008] | −0.0006 |
| Two+ goal lead | 139 | +0.013 [−0.008, +0.035] | +0.0019 [−0.007, +0.009] | +0.0009 |
| pxG-dominant side (\|gap\| ≥ 0.5) **not leading** | 119 | +0.032 [+0.003, +0.059] | +0.012 [−0.001, +0.026] | +0.0022 |
| Everything else | 617 | +0.0096 [−0.002, +0.022] | — | +0.0002 |

- **The "dominated but not ahead" games are where the model is worst:** it shades too far
  toward the side that dominated.
- **Level games:** the loss comes mostly from the draw. (4)'s δ = 0.14, fitted on 23/24,
  over-prices draws on the test seasons.

### 16.4 Tradability

**Archive rule, test seasons** (`r07_trade_rule.csv`): back the full-time 1X2 at the HT LTP when
the model is above the de-vigged HT market by more than 2 pp, lay when below. Stake 1 unit, 2%
commission on winnings.

| Model | Bets | Mean edge claimed | P&L per bet [CI] | ROI on risk |
|---|---:|---:|---|---:|
| (1) state only | 774 | 4.5 pp | +0.072 [−0.062, +0.203] | +3.7% |
| … of which backs | 405 | 4.5 pp | +0.178 [+0.026, +0.353] | +17.8% |
| … of which lays | 369 | 4.5 pp | −0.045 [−0.219, +0.115] | −1.6% |
| (2) + pxG / on-target | 981 | 4.9 pp | +0.002 [−0.132, +0.152] | +0.2% |
| (3) log-pool | 24 | 3.0 pp | −0.248 [−0.611, +0.189] | −24.5% |
| (4) HT market × perf | 809 | 3.2 pp | −0.033 [−0.145, +0.078] | −2.7% |

(1)'s backs are the one cell of twelve (4 models × back/lay/all) whose CI excludes 0 (p ≈ 0.02),
and they lean on level and one-goal states (`r07_trade_rule_cells.csv`). They do not survive a
multiple-comparison correction. The same model's log-loss is worse than the market's, and its
lays lose. Treat it as noise unless it replicates.

**26/27 order book during the break** (`r07_orderbook_ht_2627.csv`, 126 matches with a complete
two-sided 1X2 book; 63 with a BBC-timed break, the rest the fallback window):

| Runner | Median spread | Median spread (pp) | Median best-back size | Median best-lay size | Share of back sizes ≥ £20 / ≥ £50 |
|---|---:|---:|---:|---:|---|
| Favourite | 3 ticks | 1.6 | £28 | £49 | 60% / 37% |
| Draw | 4 ticks | 1.2 | £15 | £24 | 43% / 29% |
| Outsider | 4 ticks | 1.0 | £11 | £13 | 29% / 19% |

On 26/27, the 2 pp rule measured against the order-book mid:
- (1) makes 103 bets and (2) makes 75, with claimed edges of 4.1–5.6 pp against the mid.
- At the executable back or lay price those edges are still 2.0–3.6 pp, and 93–96% of bets keep
  a positive edge.
- Realised P&L: (1) −3.3 units over 103 bets; (2) −2.8 units over 75.

**So the spread is not what kills the idea:** the archive shows the claimed edges aren't there
to begin with. The typical 0.5–0.8 pp half-spread and £10–30 of size would cap any real edge
further.

### 16.5 Hedging a pre-match back at half-time (test seasons)

**Setup** (`r07_hedge.csv`):
- **Positions:** 1 unit backed pre-match on the home side and, separately, on the away side, at
  the close TWA odds. That gives 1,472 positions from the common sample.
- **Price at HT:** the median LTP, used as both lay and back price. This is optimistic for any
  trade at HT, since the archive has no spread.
- **Close:** lay o_pre/o_HT at o_HT. P&L = o_pre/o_HT − 1 whatever happens, less 2% commission
  on a gain.
- **Hold:** P&L = (o_pre − 1)(1 − c) if the side wins, else −1. The mean for hold is +0.060 per
  position.

Δ is the policy minus hold, per position:

| Policy | Model | Share closed | Δ vs hold [CI] |
|---|---|---:|---|
| Always close | — | 100% | −0.016 [−0.095, +0.061] |
| A: close if p_model < 1/o_HT | (1) | 46% | +0.014 [−0.038, +0.064] |
| A | (2) | 49% | −0.012 [−0.065, +0.041] |
| A | (4) | 61% | −0.030 [−0.088, +0.024] |
| B: A, plus add 1 unit if p_model > 1/o_HT + 2 pp | (1) | 46% (22% added) | +0.046 [−0.019, +0.117] |
| B | (2) | 49% (22% added) | −0.021 [−0.102, +0.060] |
| B | (4) | 61% (4% added) | −0.027 [−0.089, +0.030] |

- **Model-guided closing does not beat holding.** Every CI spans 0, and the one positive point
  estimate belongs to model (1), whose probabilities score worse than the market's (§16.2).
- **A fair price makes closing EV-neutral.** Closing at a fair market price is zero-EV, so the
  only thing a model could add is better information than the HT price, and §16.2 says it has
  none.
- **In practice, closing loses.** A real close pays the half-spread (about 0.5–0.8 pp of
  probability in 26/27), which would push every close-based policy further below hold.

### 16.6 Power

The smallest pooled log-loss difference this test sample can detect (80% power, two-sided 5%,
2.8 × the bootstrap SE):

| Comparison | Minimum detectable effect (nats per match) |
|---|---:|
| (2) vs market | 0.015 |
| (1) vs market | 0.013 |
| (4) vs market | 0.007 |
| (3) blend vs market | 0.0016 |

- A **gain** of 0.005 nats over the market, which would be a lot for a betting edge, is inside
  (4)'s and (2)'s noise.
- **But the point estimates all sit on the wrong side.** Every point estimate is ≥ 0, and (2)'s
  and (1′)'s CIs exclude 0 on the losing side.
- **Per-league results are about 2× less precise** (for example, L2 (2): SE 0.018).

### 16.7 Caveats

- **The archive is LTP only.**
  - The HT market is a median of last-traded prices in the break, not an executable price.
    Imputing a missing runner touches 1–62 markets per test league-season; results hold without them.
  - The hedging and archive rule tables therefore price at LTP, which is optimistic for the
    trader.
- **Training depth:**
  - The performance models learn from one season (23/24) because BBC live text starts then.
    One season's state coefficients are visibly unstable (trail 1: +0.30 vs +0.06).
  - A pooled 23/24–24/25 fit tested on 25/26 would roughly double the training data, but would
    halve the test set and the power.
- **Proxy xG and "dominance":**
  - Proxy xG is the `src` cell table (zone × body × context from BBC text), not an event-level
    xG. On-target and shot counts are raw.
  - The "dominance" subgroup is a post-hoc cut (|pxG supremacy gap| ≥ 0.5, dominant side not
    leading); it was not pre-registered.
- **Timing:**
  - Kick-off times are sofascore's scheduled starts. BBC timestamps pin the break for 2,238
    matches (172–198 of each test league-season); elsewhere the fallback window is used.
  - The 26/27 order book is 3-minute snapshots (median 3 in the break) and stops at
    2026-09-20, as in §13.
- **Anchor:** the §14 grid is used as a fixed transform. The anchor is the close, so the models
  know nothing the pre-match market didn't.
- **Not a betting test:** these are descriptive rules on realised outcomes, not a staking study.

**Reproduce** (on `mcmc-beast` in `/root/BF_runs/clv_napkin_dev`; source `.env` without printing
it). The run needs §13's rebuilt DataStore at `out/r04_datastore_ScottishPyramid.jls`, makes
read-only betdb SELECTs and runs no sampling:

```bash
julia --project -t 16 current_development/clv_napkin/r07_halftime.jl > /root/BF_runs/logs/clv_napkin/r07_run.log 2>&1
```

## 17. Is our score grid mis-weighted?

Brief 08. Code: `l08_score_grid.jl` (loader) and `r08_score_grid.jl` (runner). Results are in
`out/r08_*.csv`, with the full O / E / residual tables in `out/r08_residual_tables.txt`; the run
log is `out/r08_run.log`. There is no MCMC; the only fits are 1–2 parameter maximum-likelihood
fits on training seasons.

**Answer.**
- **Is the grid mis-weighted? Mildly, in League One/Two, and not in the way the headline
  "Poisson draws" story suggests.** Summed over all held-out folds (710 matches on the W0/W1
  panels, 1,070 on the W2 panels):
  - **Narrow away wins are over-weighted:** 0-1 by 1.9–2.7 pp of matches and 1-2 by about 2 pp.
    Away wins by one goal come out at 101 observed vs 130–133 expected on W0/W1, outside the 95%
    parametric band.
  - **1-1 is under-weighted** by 1.2–1.5 pp, and 2-2 by up to 1.5 pp.
  - **Big away wins are under-weighted:** GD ≤ −4 is 18 observed vs about 10.5 expected.
  - **Totals are the wrong shape:** 1-goal matches are over-predicted and 2- and 4-goal matches
    under-predicted. The variance of total goals is only **0.86–0.91×** what the grids imply;
    observed 2.43 against 2.77–2.82 on W0/W1, just outside the band.
  - **Draws overall are about right in League One/Two:** 182 observed vs 180–183 expected on
    W0/W1; 281 vs 267–268 on W2, inside the band. The missing 1-1/2-2 mass is offset by too much
    0-0 for grw_spfl_joint and the W2 GRW.
  - **The whole-table χ² is not decisive:** bootstrap p = 0.14–0.15 for the W0 runs and
    0.02–0.06 for grw_spfl_joint and W2.
- **Does it match the market's pattern?** Partly.
  - The sign matches §14's market-minus-Poisson on 1-0, 0-1 (−), 1-1 and 2-2 (+), but ours is
    about 2× larger on 1-1 and 4–6× larger on 0-1.
  - 0-0 disagrees: the market adds 0-0 mass and our results say there is too much.
  - On the Premiership/Championship monitor (both pooled GRW runs), the misfit is the draw:
    +3.5 to +3.8 pp more draws than predicted (bootstrap p 0.02 / 0.003), with 0-0 +2.2 pp. It
    comes with a goal level that is too high (2.97–2.98 predicted vs 2.67–2.69 observed), as in
    §14.
- **Which shape fixes it? None, out of sample, in League One/Two.** Keeping each posterior draw's
  goal means and fitting one shape on training seasons:
  - **Bivariate λ₃ goes to 0** in every League One/Two fit (0.003 at most).
  - **COM ν flips between training seasons**, 0.86–0.89 on 24/25 vs 1.10–1.15 on 25/26. For the
    W0/W1 runs it is significantly *worse* out of sample (exact-score log-loss +0.006 to +0.008
    nats per match).
  - **Dixon–Coles ρ:** −0.01 fitted on 24/25 and −0.13 on 25/26 for W0/W1, −0.07 for W2. It
    moves the exact-score log-loss by −0.002 to +0.0025, with every CI spanning 0.
  - **The market's §14 shape applied as-is** (bivariate + COM, λ₃ 0.08, ν 1.08) makes the exact
    score **worse** by +0.011 to +0.013 (CIs above 0). The market's ρ = −0.04 is the only
    near-neutral option (−0.001, CI spans 0).
  - On the monitor, the market's ρ and the fitted λ₃ ≈ 0.025 give small but real 1X2 and O/U
    gains (−0.0005 to −0.0025 nats).
- **Does it change the betting picture? No.**
  - Where the best out-of-sample shape is effectively "none" (W0/W1), no bet changes.
  - With DC (W2) or the market's shape, 10–30% of the 1X2 bet list at T−25/T−60 changes
    (dropped + added). Archive CLV moves by −0.06 to +0.06 pp and EV@close by −0.24 to +0.19
    points, all far inside the CIs.
  - O/U 2.5 bets barely move.
  - On the 26/27 order book, the market shape adds 3–5 bets and back-price CLV stays negative
    (−0.58 to −0.59 pp reshaped, vs −0.60 to −0.66 pp unreshaped).
- **What to do instead:** don't bolt on a global shape parameter.
  - The consistent League One/Two error is **goal difference** (too many narrow away wins, too
    few big ones, too few 1-1s), with totals under-dispersed. A scoreline-level obs-model change
    would need to be league- and season-stable to earn its keep, and on two held-out seasons none
    is.
  - The larger, stable misfits are the monitor's goal level and draw rate. Those are rate and
    league-offset problems, not shape.

### 17.1 Data and definitions

- **Grids:** all six runs are `CountLatents{Float64,Nothing}`, i.e. a posterior mixture of
  independent Poissons. Each match's grid is the src kernel (`Predictions.compute_score_grid!`)
  averaged over **every** posterior draw (1,000–4,000).
- **Panels:**
  - League One/Two held-out targets on each run's own panel: W0 and W1 on 24/25–25/26 (710
    matches), W2 on 23/24–25/26 (1,070).
  - The Premiership/Championship monitor for the two pooled GRW runs (751 and 1,129 matches).
  - Panels are never pooled.
- **O vs E:**
  - E is Σ grid over 0–4 and 5+ per side; O is the count.
  - Residuals are (O − E)/√E.
  - Pearson χ² pools the cells with E < 5 into one.
  - The p-value and every "band" come from a **parametric bootstrap**: 2,000 replicate panels,
    each match's score drawn from its own grid, so cell dependence is handled.
  - Dispersion compares the observed variance of total goals with the model's (mean per-match
    variance + variance of per-match means).
- **Reshaping:**
  - Each shape is applied **per posterior draw** to 400 evenly thinned draws, holding that
    draw's (λh, λa):
    - (a) DC ρ on 0-0/1-0/0-1/1-1, cells clamped at 0;
    - (b) bivariate: U = λh − λ₃, V = λa − λ₃, with λ₃ capped at 0.95·min(λh, λa);
    - (c) COM ν, with rates solved per draw so each marginal mean stays λ;
    - (d) bivariate + COM.
  - Each shape is fitted by exact-score MLE on training seasons.
  - W2 trains on 23/24 and tests on 24/25–25/26. The W0/W1 panels have no 23/24, so they use a
    **two-way season split**: fit 24/25 → score 25/26, fit 25/26 → score 24/25, pooled.
  - Held-out differences use a paired, match-day-clustered bootstrap against the unreshaped grid
    built from the same thinned draws.
- **Betting (§3):**
  - The best shape per run is the one with the largest out-of-sample exact-score gain. The
    market's §14 shape is also run as a fixed alternative.
  - **Archive:** T−25 and T−60 LTP entries, 1X2 and O/U 2.5, edge ≥ 2 pp, on the test matches
    (l01 method; each match uses the θ fitted on the other season).
  - **26/27:** §13's back-price order-book view for grw_spfl_joint (θ fitted on its whole
    24/25–25/26 panel).

### 17.2 Observed vs expected (League One/Two)

Standardised residuals (O − E)/√E for m12_td (710 matches). The other W0 runs are within ±0.1 of
it, and grw_spfl_joint and the W2 runs are in `r08_residual_tables.txt`:

```
        a0      a1      a2      a3      a4     a5+
h0    -0.69   -1.89   +0.23   +1.55   +2.32   +0.77
h1    -0.43   +1.00   -2.13   -0.01   +0.14   +0.84
h2    +1.26   +0.84   +0.16   -0.56   +0.72   -0.62
h3    +0.40   +1.77   -1.49   -1.01   +0.28   -0.93
h4    +0.34   -0.89   -0.69   -0.52   -0.97   -0.56
h5+   +1.74   +0.74   -1.05   -1.07   +1.03   -0.35
```

(grw_spfl_joint: 0-0 −2.05, 0-1 −2.40, 1-2 −1.48, 1-1 +1.17, 3-1 +2.16, 0-4 +2.48.
td_lower_a2full_carry_jump: 0-1 −2.29, 1-2 −2.03, 2-2 +1.50, 5+-0 +3.12.)

| Run, panel | n | χ² (bootstrap p) | Draws O / E [95% band] | Mean goals O / E | Var(total) O / model [band] (within + between) | Ratio |
|---|---:|---|---|---|---|---:|
| m12_td, L1/L2 | 710 | 28.8 (0.14) | 182 / 180.5 [159, 204] | 2.72 / 2.71 | 2.43 / 2.77 [2.47, 3.09] (2.74 + 0.03) | 0.88 |
| m05_joint_td, L1/L2 | 710 | 28.9 (0.15) | 182 / 180.1 [158, 203] | 2.72 / 2.72 | 2.43 / 2.81 [2.50, 3.14] | 0.86 |
| FLOOR s12_m01, L1/L2 | 710 | 29.2 (0.15) | 182 / 179.9 [158, 202] | 2.72 / 2.73 | 2.43 / 2.82 [2.51, 3.15] | 0.86 |
| grw_spfl_joint, L1/L2 | 710 | 36.3 (0.02) | 182 / 183.3 [161, 207] | 2.72 / 2.58 | 2.43 / 2.80 [2.49, 3.15] (2.68 + 0.12) | 0.87 |
| td_lower_a2full_carry_jump, L1/L2 | 1,070 | 35.4 (0.06) | 281 / 266.9 [239, 296] | 2.77 / 2.74 | 2.63 / 2.89 [2.63, 3.18] | 0.91 |
| grw_step_a2_carry_jump, L1/L2 | 1,070 | 38.4 (0.04) | 281 / 267.7 [242, 295] | 2.77 / 2.68 | 2.63 / 2.96 [2.68, 3.26] | 0.89 |
| grw_spfl_joint, Prem/Champ | 751 | 49.6 (0.005) | **199 / 172.4 [150, 194]** | 2.67 / **2.97** | 2.67 / 3.41 [3.04, 3.83] | 0.78 |
| grw_step_a2_carry_jump, Prem/Champ | 1,129 | 75.7 (< 0.001) | **300 / 257.1 [229, 284]** | 2.69 / **2.98** | 2.67 / 3.39 [3.08, 3.72] | 0.79 |

Total goals, O / E (m12_td, League One/Two; `r08_totals.csv` has every run):

| Total | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| O / E [band] | 44 / 48.9 | **111 / 129 [109, 149]** | **192 / 172 [151, 195]** | 154 / 155 | **122 / 105 [88, 124]** | 87 / 100 |

Goal difference (home − away), O / E:

| Run, panel | ≤ −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | ≥ +4 |
|---|---|---|---|---|---|---|---|---|---|
| m12_td, L1/L2 | **18 / 10.5** | 32 / 25.9 | 70 / 67.7 | **101 / 133** | 182 / 180 | 147 / 151 | 102 / 86 | 35 / 37 | 23 / 18 |
| td_lower (W2), L1/L2 | **29 / 19.0** | 50 / 41.5 | 102 / 104 | **154 / 199** | 281 / 267 | 224 / 222 | 139 / 129 | 51 / 58 | 40 / 31 |
| grw_step (W2), Prem/Champ | **19 / 30.9** | 50 / 49 | **79 / 108** | 194 / 194 | **300 / 257** | 229 / 221 | 144 / 140 | 62 / 73 | 52 / 55 |

(Bold = outside the 95% parametric band.)

**Side by side with the market** (pp of matches). Ours is (O − E)/n; the market's is §14's
market CS − Poisson fitted to its own 1X2 + O/U 2.5:

| Source | 0-0 | 1-0 | 0-1 | 1-1 | 2-2 | Draw |
|---|---:|---:|---:|---:|---:|---:|
| m12_td, L1/L2 | −0.68 | −0.50 | −2.07 | +1.30 | +0.14 | +0.22 |
| grw_spfl_joint, L1/L2 | −2.24 | −1.69 | −2.73 | +1.50 | +0.88 | −0.18 |
| td_lower (W2), L1/L2 | −0.43 | −0.33 | −2.04 | +1.23 | +1.05 | +1.32 |
| grw_step (W2), L1/L2 | −1.28 | −1.02 | −2.31 | +1.43 | +1.51 | +1.24 |
| grw_spfl_joint, Prem/Champ | +2.16 | +0.93 | −0.49 | +0.11 | +0.88 | +3.54 |
| grw_step (W2), Prem/Champ | +2.23 | +0.47 | −0.06 | +0.21 | +0.96 | +3.80 |
| **Market − Poisson (§14), L1** | +0.35 | −0.37 | −0.46 | +0.62 | +0.55 | +1.74 |
| **Market − Poisson (§14), L2** | +0.25 | −0.40 | −0.49 | +0.60 | +0.45 | +1.53 |

A single cell's (O − E)/n has a standard error of about 1–1.5 pp at n = 710. Only the patterns
repeated across runs (0-1 −, 1-1 +, GD −1 −) and the monitor's draw excess stand out from noise.

### 17.3 Keep the rates, change only the shape

Fitted parameters on training seasons (League One/Two), next to the market's §14 values:

| Run | Split | DC ρ | Bivariate λ₃ | COM ν | Bivariate + COM (λ₃, ν) |
|---|---|---|---|---|---|
| m12_td | 24/25 → 25/26 / 25/26 → 24/25 | −0.011 / −0.128 | 0 / 0 | 0.86 / 1.10 | (0, 0.86) / (0, 1.10) |
| m05_joint_td | same | −0.013 / −0.128 | 0 / 0 | 0.87 / 1.12 | (0, 0.87) / (0, 1.12) |
| FLOOR s12_m01 | same | −0.008 / −0.125 | 0 / 0 | 0.89 / 1.11 | (0, 0.89) / (0, 1.11) |
| grw_spfl_joint | same | −0.011 / −0.159 | 0 / 0 | 0.88 / 1.15 | (0, 0.88) / (0, 1.15) |
| td_lower (W2) | 23/24 → 24/25–25/26 | −0.070 | 0 | 1.01 | (0, 1.01) |
| grw_step (W2) | 23/24 → 24/25–25/26 | −0.071 | 0.003 | 1.02 | (0, 1.02) |
| grw_step (W2), Prem/Champ | same | −0.081 | 0.025 | 1.06 | (0.022, 1.06) |
| **Market, §14** | CS books, 54–57 | **−0.04** | **0.08–0.09** | **1.08** | **(0.08, 1.08)** |

**Held-out change vs the unreshaped grid** (nats per match, paired clustered 95% CI; negative is
better). League One/Two:

| Run | Shape | Exact score | 1X2 | O/U 2.5 | P(draw) predicted / observed |
|---|---|---|---|---|---|
| m12_td | (a) DC | +0.0017 [−0.0027, +0.0061] | +0.0029 [−0.0005, +0.0063] | 0 | 0.271 / 0.256 |
| m12_td | (c) COM | **+0.0069 [+0.0017, +0.0119]** | +0.0009 | +0.0004 | 0.254 / 0.256 |
| m12_td | market DC −0.04 (fixed) | −0.0011 [−0.0032, +0.0010] | +0.0001 | 0 | 0.264 / 0.256 |
| m12_td | market bivariate + COM (fixed) | **+0.0127 [+0.0065, +0.0196]** | −0.0001 | 0 | 0.266 / 0.256 |
| grw_spfl_joint | (a) DC | +0.0025 [−0.0027, +0.0076] | **+0.0049 [+0.0008, +0.0092]** | 0 | 0.278 / 0.256 |
| grw_spfl_joint | (c) COM | **+0.0084 [+0.0028, +0.0140]** | +0.0009 | +0.0005 | 0.259 / 0.256 |
| grw_spfl_joint | market bivariate + COM (fixed) | **+0.0133 [+0.0066, +0.0207]** | +0.0016 | +0.0002 | 0.270 / 0.256 |
| td_lower (W2) | (a) DC | −0.0015 [−0.0052, +0.0020] | +0.0003 | 0 | 0.270 / 0.256 |
| td_lower (W2) | market bivariate + COM (fixed) | **+0.0117 [+0.0057, +0.0181]** | −0.0003 | 0 | 0.265 / 0.256 |
| grw_step (W2) | (a) DC | −0.0019 [−0.0054, +0.0014] | +0.0005 | 0 | 0.268 / 0.256 |
| grw_step (W2) | market DC −0.04 (fixed) | −0.0014 [−0.0034, +0.0005] | +0.0001 | 0 | 0.261 / 0.256 |

- **Other cells:** bivariate λ₃ (→ 0) changes nothing, to within 1e-6. m05 and the floor match
  m12 to within ±0.001. The W2 COM fits (ν ≈ 1.01–1.02) change nothing (±0.0001).
- **Monitor (Premiership/Championship):**
  - grw_step's fitted λ₃ = 0.025 gives 1X2 −0.0006 [−0.0011, −0.0001] and O/U −0.0005
    [−0.0008, −0.0002].
  - The market's DC −0.04 gives 1X2 −0.0017 [−0.0033, −0.0002].
  - grw_spfl_joint's COM ν ≈ 0.95–0.98 gives O/U −0.0007 [−0.0012, −0.0003].
  - These are real but tiny (`r08_shape_test.csv`).

### 17.4 Does it change the bets?

Changes are the reshaped grid minus the unreshaped one, edge ≥ 2 pp (`r08_betting_changes.csv`;
the level of each row is in `r08_betting.csv`):

| Run | Shape | Market, instant | Bets (Poisson → reshaped) | Dropped / added | Δ CLV (pp) | Δ EV@close (pts) |
|---|---|---|---|---|---:|---:|
| W0 runs, grw_spfl_joint | best = bivariate (λ₃ → 0) | all | unchanged | 0 / 0 | 0 | 0 |
| td_lower (W2) | DC ρ −0.07 | 1X2 T−25 | 533 → 558 | 53 / 78 | +0.05 | +0.18 |
| td_lower (W2) | DC ρ −0.07 | 1X2 T−60 | 324 → 350 | 31 / 57 | −0.04 | −0.13 |
| grw_step (W2) | DC ρ −0.07 | 1X2 T−25 | 516 → 553 | 54 / 91 | +0.00 | +0.03 |
| m12_td | market bivariate + COM | 1X2 T−25 | 530 → 547 | 30 / 47 | +0.05 | +0.19 |
| grw_spfl_joint | market bivariate + COM | 1X2 T−25 | 528 → 577 | 33 / 82 | −0.01 | −0.01 |
| any run | any | O/U 2.5 | ±0–9 bets | | ≤ ±0.07 | ≤ ±0.15 |
| grw_spfl_joint, 26/27 book (back) | market bivariate + COM | 1X2 T−60 / T−25 | 62 → 65 / 70 → 75 | 3 / 6, 4 / 9 | +0.08 / +0.01 | +0.19 / −0.04 |

Every Δ is well inside the ±0.15–0.3 pp CLV and ±0.7–1.5 point EV CIs of the rows it compares.
The 26/27 back-price CLV stays at −0.58 to −0.66 pp either way.

### 17.5 Caveats

- **Power:**
  - 710 held-out matches per W0/W1 panel is small for 36 cells. Single-cell residuals of ±2 are
    expected by chance in a table this size.
  - The reliable signals are the ones repeated across runs, since they share the same matches
    and outcomes: GD −1 over-predicted, total-goal under-dispersion, 1-1 under-weight. They are
    correlated evidence from **one** set of outcomes, not independent replications.
- **Split:**
  - The W0/W1 shapes use a two-way season split (no 23/24 predictions exist on those panels), so
    each test season is scored with a θ from the other.
  - The flip in ν and ρ between 24/25 and 25/26 is itself the main finding: the League One/Two
    shape isn't season-stable.
- **Reshape approximations:**
  - Reshaping uses 400 thinned draws per match; the unreshaped comparison uses the same draws.
  - Bivariate λ₃ is capped at 0.95·min(λh, λa) per draw, which binds only for tiny λ.
  - DC cells are clamped at 0.
- **Monitor:** the monitor rows mix a goal-level bias (the models over-predict Premiership goals)
  with shape. A shape parameter cannot fix a level error, and the monitor's draw excess is partly
  that level error.
- **Market comparison:** the market column is §14's CS books, which are thin in League One/Two
  (98/60 matches), as noted there.
- **Betting:** descriptive only, and all the CLV/EV levels are §3/§10/§13's.

**Reproduce** (on `mcmc-beast` in `/root/BF_runs/clv_napkin_dev`; source `.env` without printing
it). The run needs §13's DataStore and 26/27 extension in `out/` plus r05's `out/r05_residuals.csv`,
takes about 4 minutes, makes read-only DB reads and loads the persisted fits, with no sampling:

```bash
julia --project -t 16 current_development/clv_napkin/r08_score_grid.jl > /root/BF_runs/logs/clv_napkin/r08_run.log 2>&1
```
