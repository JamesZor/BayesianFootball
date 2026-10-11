# Wave 5 report: the qs_marketobs + scenario Kelly paper-trading config (TODO 044)

## Summary

- **Frozen config** (`FREEZE.md`, commit `04fdba71`, pushed before any 25/26 wave-5 number existed): qs_marketobs + scenario Kelly, basket B1 = 1X2 H/D/A + Under 2.5, **λ 12**. It was picked on 24/25 only.
- **25/26 at T−25m:** wealth 1.624, g **9.70** [1.90, 21.89] (circular [−0.11, 19.53]), MDD **−19.26%**, 494 bets.
- m12 × canonical × λ 23 on the same days: wealth 1.327, g 5.66 [−0.98, 21.37], MDD −16.15%, 557 bets.
- Paired d vs m12 on 25/26 is **+4.04 [−10.52, 14.54]** (circular [−9.07, 13.98]). The interval is wide and does not separate the two configs.
- **Decision (ruling 8): paper-trade the frozen config.** Its 25/26 g is above 0 and its MDD (−19.26%) is inside the bound (−20.39%), with a margin of 1.13 pp.
- **The MDD match did not carry over.** On 24/25, λ 12 sat just inside m12's drawdown. On 25/26 its MDD is 3.1 pp worse than m12's. The fallback (λ 23) has 25/26 g 5.57 and MDD −11.22%.
- **The selected basket ranked first again on 25/26** among the 6 baskets looked at: B1 9.70, B3 8.26, B5 6.37, B6 6.13, B2 4.97, B4 4.41.
- **The per-direction pattern flipped.** Away carried 24/25 (Kelly ROI +26.5%) and lost on 25/26 (−6.6%), while Draw and U2.5 earned. B4, the rule-based basket that kept only H and A, had the lowest 25/26 g of the six.
- **T−6h (report only):** the T−25m-frozen config is worse than m12 on both halves: g 1.13 vs 6.53 on 24/25 and 3.31 vs 4.91 on 25/26, with MDD −17.7% / −13.0% vs −6.8% / −6.5%. λ 12 is a T−25m setting.
- **Caveats:** 6 baskets × 12 λ looked at; about 50 slates per half; m12's experiment-07 in-sample advantage on both halves; last-traded, non-executable prices. 25/26 was unseen by the selection rule, not unseen by anyone.

Units: g = mean daily log growth of the unit bankroll, ×10⁻³ per slate day. d = paired difference in g against `m12_canonical_l23` on the same days and snapshot. Intervals are 90% moving-block bootstraps: 8-week blocks within the half, 999 reps, seed 20261009. The first interval is noncircular (wave 4's `P3.block_bootstrap`); "circular" uses wrap-around block starts. Each half starts at bankroll 1.0, and MDD is measured within the half. Figures are net of 2% commission; CLV ignores commission. Sources: `results/phase1/`, `results/phase2/phase2_summary.csv` (all columns), and `phase2_decision.csv`. Runner `716c84a1`, log `logs/phase2.log`.

## Selection (24/25 only, T−25m, 49 slates)

The 6 baskets were all solved jointly with ScenarioKelly on the same days. Legs outside a basket never enter the solve (ruling 4). Directions outside the canonical tiers get trust 0.25. The matched λ is the smallest grid λ whose 24/25 MDD is within m12 canonical's −13.595% at that λ and every larger one (ruling 3).

| Basket | Directions | Matched λ | g | MDD % | g @ λ 23 | MDD @ λ 23 |
|---|---|---:|---:|---:|---:|---:|
| **B1_canonical** | H, D, A, U2.5 | **12** | **17.52** | −13.07 | 9.90 | −7.14 |
| B2_1x2 | H, D, A | 14 | 14.74 | −12.71 | 9.53 | −7.99 |
| B3_1x2_ou25 | H, D, A, O2.5, U2.5 | 14 | 17.09 | −12.16 | 10.96 | −7.64 |
| B4_eda_keep (EDA KEEP rule) | H, A | 14 | 12.94 | −13.01 | 8.36 | −8.21 |
| B5_all13 | all 13 directions | 16 | 13.83 | −12.50 | 9.96 | −8.92 |
| B6_home_u25 | H, U2.5 | 16 | 6.06 | −12.66 | 4.35 | −9.17 |

Ruling 6 picks B1, and no smaller basket comes within 0.5×10⁻³. The grid decides between B1 and B3: B3 has higher g than B1 at every common λ, but at λ 12 it misses the target by 0.19 pp (−13.78%).

Ruling-4 sensitivity (canonical basket, λ 23, 24/25):

| Variant | g | MDD % | Budget-binding days |
|---|---:|---:|---:|
| Wave-4 all-legs-then-trust | 10.25 | −7.81 | 41% |
| Basket-only legs | 9.90 | −7.14 | 12% |

Source: `results/phase1/phase1_selection.csv`, `phase1_sensitivity.csv`, `phase1_directions.csv`.

## λ curve (24/25)

Each cell is g ×10⁻³ / MDD %. Source: `results/phase1/phase1_lambda_curve.csv`.

| λ | B1 | B2 | B3 | B4 | B5 | B6 |
|---:|---|---|---|---|---|---|
| 4 | 21.1 / −21.3 | 19.7 / −23.6 | 22.9 / −21.1 | 17.5 / −21.6 | 20.8 / −22.9 | 13.0 / −25.8 |
| 8 | 20.9 / −18.7 | 19.7 / −21.0 | 22.7 / −19.0 | 17.6 / −19.9 | 20.6 / −21.5 | 11.5 / −22.8 |
| 10 | 19.6 / −15.4 | 18.4 / −17.2 | 21.5 / −15.9 | 16.4 / −17.0 | 19.4 / −18.4 | 9.3 / −19.0 |
| 12 | **17.5 / −13.1** | 16.6 / −14.6 | 19.2 / −13.8 | 14.6 / −15.0 | 17.3 / −16.2 | 7.8 / −16.3 |
| 14 | 15.4 / −11.4 | 14.7 / −12.7 | 17.1 / −12.2 | 12.9 / −13.0 | 15.5 / −14.1 | 6.8 / −14.2 |
| 16 | 13.7 / −10.0 | 13.2 / −11.2 | 15.2 / −10.7 | 11.6 / −11.5 | 13.8 / −12.5 | 6.1 / −12.7 |
| 20 | 11.2 / −8.1 | 10.8 / −9.1 | 12.5 / −8.7 | 9.5 / −9.4 | 11.3 / −10.2 | 5.0 / −10.4 |
| 23 | 9.9 / −7.1 | 9.5 / −8.0 | 11.0 / −7.6 | 8.4 / −8.2 | 10.0 / −8.9 | 4.3 / −9.2 |
| 30 | 7.7 / −5.5 | 7.5 / −6.2 | 8.6 / −5.9 | 6.5 / −6.4 | 7.8 / −6.9 | 3.4 / −7.2 |

λ 6, 18 and 26 are omitted here but are in the CSV.

Is λ the active dial for B1? Yes:
- The drawdown constraint is active on 65% of days at λ 12, rising to 92% at λ 23.
- The 0.99 budget binds on 12% of days at every λ.
- The 0.25 slate cap binds on 0% of days at λ 12 (2% at λ ≤ 8).
- Mean exposure falls from 9.6% at λ 4 to 3.3% at λ 30.

## Freeze

`FREEZE.md` holds the full config:
- the trust table, PolicySpec, BookSpec, ScenarioKelly with 1000 scenarios and seed 20261010;
- the MDD target, the 6 looks, the fallback and the ruling-8 bound.

It was committed as `04fdba71` and pushed. The git order is:
1. Phase 1 results `dd5ab8ba`;
2. FREEZE `04fdba71`;
3. the Phase 2 runner `716c84a1`, the first code to call `release!` (log: `SEAL_RELEASED freeze_commit=04fdba71…`);
4. Phase 2 results `0f600b92`.

## Evaluation (T−25m)

Integrity checks passed in the run:
- On 24/25, every basket's g equals its Phase-1 g exactly, and m12's MDD equals the target exactly.
- m12 and wave-4 SK (all legs) reproduce Phase 0's per-season wealth, MDD and bets on both halves.
- The DB audit was unchanged on all 10 tables.

**25/26 (evaluation half, 50 slates):**

| Cell | Role | Wealth | g [90%] | g circular | d vs m12 [90%] | d circular | MDD % | Sharpe | Calmar | ROI % | Turnover (units) | Bets | CLV pp | Beat close % | Budget-binding | Cap-binding | Risk-active |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **frozen_B1_l12** | frozen | **1.624** | **9.70 [1.90, 21.89]** | [−0.11, 19.53] | **+4.04 [−10.52, 14.54]** | [−9.07, 13.98] | **−19.26** | 0.158 | 3.24 | 12.6 | 4.97 | 494 | 0.29 | 56.2 | 10% | 0% | 74% |
| fallback_B1_l23 | fallback | 1.321 | 5.57 [1.09, 12.35] | [0.09, 11.06] | −0.10 [−13.87, 7.48] | [−11.47, 8.31] | −11.22 | 0.166 | 2.86 | 13.6 | 2.36 | 496 | 0.29 | 56.2 | 10% | 0% | 100% |
| m12_canonical_l23 | benchmark | 1.327 | 5.66 [−0.98, 21.37] | [−3.19, 18.46] | — | — | −16.15 | 0.116 | 2.03 | 7.8 | 4.22 | 557 | 0.07 | 53.0 | n/a | 0% | 100% |
| cand_B3_1x2_ou25_l14 | candidate | 1.511 | 8.26 [1.22, 19.40] | [−0.62, 17.05] | +2.59 [−11.20, 12.10] | [−9.87, 12.01] | −17.26 | 0.155 | 2.96 | 12.0 | 4.26 | 535 | 0.24 | 54.9 | 10% | 0% | 84% |
| cand_B5_all13_l16 | candidate | 1.375 | 6.37 [0.47, 17.48] | [−1.79, 14.66] | +0.71 [−11.95, 9.85] | [−10.87, 9.27] | −18.13 | 0.130 | 2.07 | 8.5 | 4.40 | 727 | 0.34 | 58.0 | 52% | 0% | 94% |
| cand_B6_home_u25_l16 | candidate | 1.358 | 6.13 [2.02, 12.57] | [1.04, 11.34] | +0.46 [−13.74, 8.55] | [−11.41, 9.24] | −7.75 | 0.210 | 4.62 | 20.5 | 1.75 | 218 | 0.41 | 57.2 | 0% | 0% | 88% |
| cand_B2_1x2_l14 | candidate | 1.282 | 4.97 [−1.89, 14.46] | [−3.00, 12.63] | −0.69 [−14.06, 7.06] | [−11.89, 7.30] | −19.83 | 0.099 | 1.42 | 9.4 | 3.00 | 410 | 0.31 | 56.2 | 4% | 0% | 84% |
| cand_B4_eda_keep_l14 | candidate | 1.247 | 4.41 [−2.18, 12.05] | [−2.05, 10.75] | −1.25 [−15.07, 4.74] | [−12.13, 6.29] | −19.41 | 0.097 | 1.27 | 10.3 | 2.39 | 289 | 0.44 | 58.9 | 0% | 0% | 84% |
| wave4_sk_all_legs_l23 | check | 1.303 | 5.30 [0.39, 11.74] | [−0.06, 10.72] | −0.37 [−14.67, 6.99] | [−12.32, 8.05] | −10.47 | 0.164 | 2.90 | 14.3 | 2.12 | 449 | 0.32 | 56.8 | 52% | 0% | 96% |

**24/25 (selection half, 49 slates, for completeness):**

| Cell | Role | Wealth | g [90%] | g circular | d vs m12 [90%] | d circular | MDD % | Sharpe | Calmar | ROI % | Turnover (units) | Bets | CLV pp | Beat close % | Budget-binding | Cap-binding | Risk-active |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_B1_l12 | frozen | 2.360 | 17.52 [7.90, 25.65] | [8.88, 26.61] | +2.48 [−7.61, 10.45] | [−6.83, 10.25] | −13.07 | 0.290 | 10.40 | 22.6 | 6.01 | 529 | 0.21 | 54.8 | 12% | 0% | 65% |
| fallback_B1_l23 | fallback | 1.625 | 9.90 [4.60, 14.31] | [5.13, 15.05] | −5.14 [−13.91, 1.79] | [−14.00, 1.76] | −7.14 | 0.296 | 8.75 | 23.3 | 2.68 | 529 | 0.21 | 54.8 | 12% | 0% | 92% |
| m12_canonical_l23 | benchmark | 2.090 | 15.04 [5.76, 24.00] | [5.91, 25.97] | — | — | −13.60 | 0.366 | 8.01 | 24.1 | 4.52 | 575 | 0.30 | 58.2 | n/a | 0% | 96% |
| cand_B3_1x2_ou25_l14 | candidate | 2.310 | 17.09 [7.97, 25.56] | [8.56, 26.00] | +2.05 [−8.05, 10.05] | [−7.01, 9.28] | −12.16 | 0.321 | 10.78 | 23.0 | 5.71 | 601 | 0.16 | 52.9 | 24% | 0% | 78% |
| cand_B2_1x2_l14 | candidate | 2.059 | 14.74 [7.38, 20.58] | [8.34, 21.93] | −0.30 [−9.12, 5.86] | [−8.49, 5.94] | −12.71 | 0.285 | 8.34 | 24.2 | 4.38 | 476 | 0.28 | 55.9 | 2% | 0% | 82% |
| cand_B5_all13_l16 | candidate | 1.969 | 13.83 [5.79, 21.25] | [5.86, 21.57] | −1.21 [−11.32, 6.72] | [−11.09, 6.32] | −12.50 | 0.287 | 7.75 | 19.8 | 4.89 | 693 | 0.23 | 54.5 | 41% | 0% | 88% |
| cand_B4_eda_keep_l14 | candidate | 1.886 | 12.94 [6.05, 19.18] | [6.63, 19.53] | −2.10 [−10.36, 5.28] | [−11.35, 5.06] | −13.01 | 0.263 | 6.81 | 25.0 | 3.55 | 354 | 0.26 | 53.7 | 0% | 0% | 82% |
| cand_B6_home_u25_l16 | candidate | 1.346 | 6.06 [−1.44, 10.94] | [−0.05, 11.72] | −8.98 [−20.83, −0.65] | [−20.33, −0.62] | −12.66 | 0.205 | 2.73 | 20.3 | 1.71 | 204 | 0.41 | 59.9 | 0% | 0% | 88% |
| wave4_sk_all_legs_l23 | check | 1.653 | 10.25 [4.64, 15.10] | [5.21, 15.60] | −4.79 [−13.60, 2.30] | [−13.74, 1.99] | −7.81 | 0.310 | 8.36 | 25.9 | 2.52 | 503 | 0.27 | 55.2 | 41% | 0% | 94% |

Column notes:
- ROI is money ROI: Σ P&L / Σ stake, both rescaled by the slate's opening bankroll.
- Turnover is Σ stake in units of the opening bankroll.
- Budget-binding: the joint raw total is ≥ 0.989. m12's per-match allocator has no joint budget, hence n/a.
- Cap-binding: the 0.25 slate cap rescaled the slate.
- Risk-active: k_risk < 1.

No day was unmatched in any paired comparison.

**What the 25/26 numbers say:**
- The frozen config grew fastest on 25/26, and the candidate ranking at the top held: B1 > B3 > B5.
- Every d interval includes 0, under both block schemes. 50 slates cannot separate the frozen config from m12, or from the fallback.
- The frozen config's g interval excludes 0 under the noncircular scheme but not the circular one ([−0.11, 19.53]).
- The price of λ 12 is drawdown. MDD is −19.26% at λ 12 against −11.22% at λ 23, on the same 494–496 bets.
- B6 had the smallest 25/26 drawdown among the candidates (−7.75%) and the best Sharpe. It was the worst basket on 24/25, so this is one more half-season of noise in the ranking, not a finding.

**Per-direction, T−25m** (`phase2_directions.csv`; Kelly ROI rescaled by the slate's opening bankroll):

| Cell | Half | Home | Away | Draw | U2.5 | All |
|---|---|---|---|---|---|---|
| frozen_B1_l12 | 24/25 | 173 bets, +26.5% | 191, +26.5% | 104, +7.7% | 61, +5.0% | 529, +22.6% |
| frozen_B1_l12 | 25/26 | 155, +17.0% | 142, **−6.6%** | 112, +14.4% | 85, **+28.9%** | 494, +12.6% |
| m12_canonical_l23 | 24/25 | 142, +31.8% | 183, +40.7% | 160, −1.3% | 90, −10.9% | 575, +24.1% |
| m12_canonical_l23 | 25/26 | 126, −0.1% | 157, −9.3% | 166, +15.7% | 108, +48.3% | 557, +7.8% |

The 24/25 per-direction table would have dropped Draw and U2.5; that is what B4 did. On 25/26 those were the two directions that earned, and away lost for both models. This confirms the EDA §5.1 warning: half-season per-direction ROI is not a basket-selection rule. The table has every cell's per-direction rows, CLV and standalone MDD.

## T−6h (ruling 9, report only)

The T−25m-frozen config and m12 canonical are applied to the T−6h snapshot (90-minute staleness), with no separate calibration. The T−6h snapshot covers about half the fixtures: 155 / 136 books against 309 / 302 at T−25m, on 44 slate days per half.

| Cell | Half | Wealth | g [90%] | d vs m12 [90%] | MDD % | Bets | ROI % | CLV pp |
|---|---|---:|---|---|---:|---:|---:|---:|
| frozen_B1_l12 | 24/25 | 1.051 | 1.13 [−3.60, 7.42] | −5.39 [−11.73, 2.72] | −17.73 | 176 | 4.2 | 0.62 |
| fallback_B1_l23 | 24/25 | 1.061 | 1.34 [−1.36, 5.13] | −5.19 [−11.36, 1.58] | −9.95 | 176 | 8.1 | 0.62 |
| m12_canonical_l23 | 24/25 | 1.333 | 6.53 [−0.01, 13.42] | — | −6.82 | 193 | 28.1 | 0.65 |
| frozen_B1_l12 | 25/26 | 1.157 | 3.31 [−1.99, 8.04] | −1.61 [−7.78, 3.64] | −12.99 | 151 | 12.6 | 0.71 |
| fallback_B1_l23 | 25/26 | 1.101 | 2.19 [−0.62, 4.73] | −2.73 [−7.67, 1.20] | −6.89 | 151 | 14.4 | 0.71 |
| m12_canonical_l23 | 25/26 | 1.241 | 4.91 [1.29, 9.62] | — | −6.51 | 158 | 23.3 | 0.39 |

The fallback rows at T−6h are extra. They are the same solve as the frozen config, at λ 23.

At T−6h, qs + SK trails m12 on both halves, and the frozen config's drawdown is about twice m12's. The paper-trading config is a T−25m config. A T−6h book would need its own selection.

## Decision (ruling 8)

**Paper-trade the frozen config** (qs_marketobs + scenario Kelly, B1, λ 12). On 25/26 its g is 9.70×10⁻³ > 0 and its MDD is −19.26%, inside 1.5 × target = −20.39%. Source: `results/phase2/phase2_decision.csv`.

The margin on the drawdown bound is 1.13 pp, so the rule passed narrowly. d vs m12 (+4.04, interval spans 0) is descriptive only.

## Caveats

- **Number of looks.**
  - 6 baskets × 12 λ values were scored on 24/25, plus the ungated per-direction run and the ruling-4 sensitivity.
  - The rule chose the maximum of 6 noisy estimates, so the 24/25 g of the winner (17.52) is biased upwards. Its 25/26 g (9.70) is the honest figure.
  - The grid alone decided B1 over B3.
- **About 50 slates per half** (49 and 50; 44 at T−6h). Every d interval spans 0. Neither half-season can rank the candidates with confidence.
- **m12's experiment-07 advantage on both halves.** m12's canonical policy (λ 23, basket and tiers) was chosen in experiment 07 on these same two seasons using m12. The benchmark is therefore in-sample on both halves, which favours m12 on 25/26 too. The MDD target is also an m12-in-sample number.
- **Last-traded prices, not executable.** Snapshots are Betfair last-traded prices with up to 90 minutes' staleness. The CLV napkin (2026-09-28) found back-price CLV of about −0.6 pp in the 26/27 order book, with the edge living inside the spread. Expect live fills to be worse than these backtests.
- **25/26 is unseen by the selection rule, not unseen by anyone.** Wave 4 published full-period numbers that include 25/26 (for the wave-4 SK and m12 configs, not for any wave-5 basket or λ). The brief, the rulings and the basket list were written by people who had seen them.
- **The MDD match is an in-sample calibration.** λ 12 matched m12's 24/25 drawdown, but on 25/26 it ran 3.1 pp deeper than m12. Paper trading should watch drawdown against the −20.39% ruling-8 bound and keep the λ 23 fallback ready.
