# Wave 3: fusing goals+xG with market rates, with learned market bias

1. Three frozen 40-fold grids forecast the same 710 Scottish L1/L2 fixtures; no reference refits, no prior or threshold changes after scoring.
2. Primary: fusion_qs_bias − market close, 1X2 LogLoss −0.000325 (0.612793 vs 0.613118). The noncircular 90% interval [−0.00707, 0.00113] crosses zero: **no detectable difference**. The circular interval agrees; the clustered 95% is [−0.00684, 0.00598].
3. On the standard 3-way scores (common 595 fixtures), fusion_qs_bias is 1.052405 vs the close's 1.053440 categorical LogLoss (−0.00104) and 0.221579 vs 0.221438 RPS (+0.00014). Both cross zero.
4. Against wave 2's qs_marketobs: +0.000876 1X2 LogLoss, no detectable difference. Adding xG to the market arm gives no detectable gain.
5. Bias ablation, fusion_qs_bias − fusion_qs_nobias: +0.000235 1X2 LogLoss, no detectable difference. The learned bias terms give no detectable gain, and the point estimate is slightly worse.
6. **The market does not compress (κ_D < 1). It is more spread than the fused latent:** κ_D's 90% interval is above 1 in all 40 folds for both bias arms (median 1.17 for QS, 1.33 for GRW).
7. Home tilt δ_D (QS median 0.018) and totals offset δ_M (median −0.007): every fold's 90% interval includes 0.
8. δ_M and log κ have posterior correlation −0.80. Their sum, the market's level relative to the xG latent, is about +0.10 and excludes 0 in every fold; it is pinned much more tightly than either term alone (sd 0.015 vs 0.020/0.025).
9. Learning κ_D makes predictions more compressed: the market-on-model slope is 1.277 with bias vs 1.106 without (GRW with bias: 1.483). κ_D absorbs the market's extra spread instead of passing it to the latent.
10. Six of 60 interval rows have a circular class that disagrees with the prescribed noncircular class. None is a primary pair; the one prescribed noncircular "worse" (qs_bias − grw_bias, all markets, +0.00072) is n.d. under circular.
11. The leakage audit passes: 3 arms × 40 folds have 0 market rows in target or after the cutoff, and all 1430 table rows are built from their own close only. All 120 folds have 0 divergences and R̂ ≤ 1.0194; the control's fold-40 tail ESS (330.7) is flagged.
12. **Decision: no promotion.** fusion_qs_bias does not beat the market close on the prescribed interval, and the bias terms add nothing detectable. Phase 5's frozen-seed reproductions are byte-identical (REPRODUCIBILITY.md).

## Fixed panel, score law and traceability

Namespace `scottish_lower_qs_wave3_2426`, scorecard v1.2, reference de-vigged Betfair TWA(−20,0] close. Snapshot `c786e2fc…423b4` and the frozen market table `680bc228…72549` are pinned and checked by every gate. Runs are in `results/phase3/RUNS.csv` (sampling source `dde8a441`). Each grid is 4×(500+1000) at attempt 0 with no R̂ reruns.

The scorer, `r06_score.jl` at `4f7ee7bc`, is wave 2's metric set and intervals, frozen in DECISIONS_LOG before any score was computed.

**References** are qs_marketobs `81fdf817…` and qs_joint `2a5eee90…`, loaded from wave 2's namespace and never refit. They match wave 2's committed CSVs exactly (`results/phase4/reference_parity.csv`), on all of the following:
- harness rows (598 per arm, max difference 0);
- observation frames (including market_close, 2899 rows);
- goal fixture scores (710);
- common-panel means (≤ 2.2e-16, within wave 2's r11 tolerance of 1e-14).

**Panels:**
- 1X2: 595 fixtures / 1785 selections, the same keys for every arm and the close.
- OU2.5: 379/758. BTTS: 178/356. All markets: 627/2899.
- Goal scores: all 710 fixtures.
- The harness's own close RPS uses 1528 fixtures (models 596), so the RPS here is the common 595-fixture value (`score_panel_audit.csv`).

**Intervals** (all in `paired_intervals.csv`):
- noncircular 8-week moving blocks within season, B 999, seed 20261009, 90%: **the classification**;
- circular 8-week, 90%: sensitivity check, and disagreements are flagged;
- harness fixture-clustered, B 10 000, seed 20260911, 95%.

The noncircular replicate mean drifts from the point estimate by up to 0.00564 (circular: 0.00063), as in wave 2. Its intervals are skewed accordingly.

## Outcome scores

The 1X2 columns are on the common 595 fixtures (`score_panel_audit.csv`). The rest are harness target-subset values (`harness_scores_vs_fusion_qs_bias.csv`). Binary is the harness's per-selection LogLoss; categorical is the standard 3-way −log p(outcome).

| model | 1X2 binary | 1X2 categorical | RPS (595) | OU2.5 | BTTS | Brier | ECE all | compression | model-on-market |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion_qs_bias | 0.612793 | 1.052405 | 0.221579 | 0.685758 | 0.687319 | 0.211507 | 0.00729 | 1.277 | 0.633 |
| fusion_qs_nobias | 0.612559 | 1.051655 | 0.221253 | 0.685320 | 0.687511 | 0.211366 | 0.00673 | 1.106 | 0.734 |
| fusion_grw_bias | 0.612596 | 1.052289 | 0.221438 | 0.684780 | 0.684554 | 0.211401 | 0.00894 | 1.483 | 0.534 |
| qs_marketobs (w2) | 0.611917 | 1.050477 | 0.220809 | 0.686544 | 0.688133 | 0.211061 | 0.00911 | 1.070 | 0.722 |
| qs_joint (w2) | 0.616180 | 1.058839 | 0.223881 | 0.690041 | 0.682000 | 0.213111 | 0.02355 | 1.167 | 0.578 |
| market_close | 0.613118 | 1.053440 | 0.221438 | 0.689878 | 0.683371 | 0.211564 | 0.01391 | — | — |

The compression slope is market-on-model (ideal 1). A value above 1 means the model is less spread than the market.

## Paired comparisons (arm − reference; negative = arm better)

Class = noncircular 90%. "n.d." means no detectable difference. ⚑ marks a circular class that disagrees.

| pair | metric | Δ | noncircular 90% | class | circular 90% | clustered 95% |
|---|---|---:|---|---|---|---|
| **1. qs_bias − close** | 1X2 binary | −0.000325 | [−0.00707, 0.00113] | n.d. | n.d. | [−0.00684, 0.00598] |
| | 1X2 categorical | −0.001035 | [−0.01388, 0.00188] | n.d. | n.d. | [−0.01341, 0.01089] |
| | RPS | +0.000142 | [−0.00385, 0.00144] | n.d. | n.d. | [−0.00381, 0.00394] |
| | OU2.5 | −0.004121 | [−0.00607, 0.00385] | n.d. | n.d. | [−0.01188, 0.00352] |
| | BTTS | +0.003948 | [−0.00117, 0.00851] | n.d. | n.d. | [−0.00412, 0.01194] |
| | all | −0.000792 | [−0.00463, 0.00126] | n.d. | n.d. | [−0.00531, 0.00360] |
| **2. qs_bias − qs_marketobs** | 1X2 binary | +0.000876 | [−0.00163, 0.00191] | n.d. | n.d. | [−0.00156, 0.00327] |
| | 1X2 categorical | +0.001928 | [−0.00282, 0.00398] | n.d. | n.d. | [−0.00242, 0.00622] |
| | RPS | +0.000770 | [−0.00082, 0.00155] | n.d. | n.d. | [−0.00069, 0.00220] |
| | goals joint | +0.000410 | [−0.00552, 0.00546] | n.d. | n.d. | [−0.00731, 0.00911] |
| **3. qs_bias − qs_nobias** | 1X2 binary | +0.000235 | [−0.00179, 0.00113] | n.d. | n.d. | [−0.00170, 0.00210] |
| | 1X2 categorical | +0.000751 | [−0.00304, 0.00250] | n.d. | n.d. | [−0.00265, 0.00408] |
| | RPS | +0.000327 | [−0.00089, 0.00089] | n.d. | n.d. | [−0.00078, 0.00142] |
| | goals joint | −0.000820 | [−0.00505, 0.00085] | n.d. | n.d. | [−0.00477, 0.00318] |
| **4. qs_bias − grw_bias** | 1X2 binary | +0.000198 | [−0.00049, 0.00177] | n.d. | n.d. | [−0.00137, 0.00180] |
| | all | +0.000717 | [0.00001, 0.00267] | **worse** | n.d. ⚑ | [−0.00075, 0.00225] |
| **4. qs_bias − qs_joint** | 1X2 binary | −0.003386 | [−0.00477, 0.00041] | n.d. | better ⚑ | [−0.00867, 0.00183] |
| | 1X2 categorical | −0.006433 | [−0.00878, 0.00033] | n.d. | better ⚑ | [−0.01620, 0.00325] |
| | RPS | −0.002301 | [−0.00315, 0.00001] | n.d. | better ⚑ | [−0.00567, 0.00103] |

All 60 rows, including OU2.5/BTTS/all and the goal total and allocation channels for every pair, are in `results/phase4/paired_intervals.csv`.

Tier 5 is context only and not a decision input: each other arm against the close. Every 1X2 row there is n.d. There are two flagged contradictions:
- fusion_grw_bias − close, all markets: −0.00151, noncircular **better** [−0.00541, −0.00044], circular n.d.;
- fusion_grw_bias − close, OU2.5: −0.00510, circular better.

Both clustered intervals cross zero. Neither is a prescribed pair, so they are not grounds for a claim.

## Market bias posteriors

Source: `posterior_bias_by_fold.csv`, `posterior_level_joint_by_fold.csv` and the figure `posterior_bias_by_fold.svg`. The market sees D̃ = δ_D + κ_D·D and M̃ = δ_M + M on the true goal log-rates ζ = η + log κ.

| arm | κ_D median (range) | folds κ_D 90% > 1 | δ_D median | δ_M median | δ_D / δ_M folds excl. 0 | σ_obs median |
|---|---|---:|---:|---:|---|---:|
| fusion_qs_bias | 1.171 (1.125–1.208) | 40/40 | +0.018 | −0.007 | 0 / 0 | 0.1190 |
| fusion_grw_bias | 1.334 (1.260–1.392) | 40/40 | −0.002 | −0.013 | 0 / 0 | 0.1193 |
| fusion_qs_nobias | (fixed 1) | — | (0) | (0) | — | 0.1191 |

- **Compression:** the answer is no. The market's supremacy is 17% (QS) or 33% (GRW) wider than the fused latent's, in every fold. This matches the standing finding that the models' team latents are compressed relative to the market. Because predictions price the latent's true rates, learning κ_D lets the market's spread be explained as bias rather than shared signal. The bias arms' predictions are therefore more compressed (slopes 1.277/1.483 vs 1.106), and they score no better.
- **Home tilt and totals:** neither δ_D nor δ_M is distinguishable from 0 in any fold (median P(δ_D < 0) is 0.27 for QS; median P(δ_M < 0) is 0.62).
- **Joint reading of δ_M and log κ:** they trade off along a ridge, with posterior correlation −0.80 (range −0.84 to −0.72) in both arms. log κ itself sits at 0.107 (QS) / 0.111 (GRW), against 0.099 in nobias. Their sum δ_M + log κ, the market's per-side level relative to the xG latent, is +0.101 (QS) / +0.098 (GRW), with sd 0.015. It excludes 0 in all 40 folds. The level gap is therefore between the xG latent and goals/market, and goals and market agree: δ_M ≈ 0.
- **σ_obs:** 0.119 in all three arms, the same as wave 2's qs_marketobs (0.1192). The bias terms do not reduce the measurement noise.
- **Control fold 40** (tail ESS 330.7, flagged): κ_D 1.184 [1.098, 1.285], δ_D 0.024 [−0.017, 0.061], δ_M −0.005 [−0.035, 0.026], σ_obs 0.114. These are in line with the other folds.

## Posterior r (QS arms)

Source: `posterior_r_by_fold.csv`. The median of the fold medians for r₀/rₛ/rₖ is 0.253/0.355/0.295 (bias) and 0.222/0.314/0.256 (nobias). The 90% intervals fall below the prior median 1 in 40/40/37 folds of both arms, close to wave 2's qs_marketobs (0.197/0.295/0.228).

## Leakage audit

`r10_leakage_audit.jl`, wave 2's audit on the three fusion arms. Output: `results/leakage_audit.csv`, `leakage_provenance.csv` and `leakage_logloss.csv`.
- **Folds:** in all 120 fold × arm cells, the active market likelihood rows (26 567 per arm) include 0 held-out targets and 0 rows at or after the fold's first target kickoff. No target row enters the likelihood. The fused design's market mask equals the feature's availability mask in every fold. The 710 targets match the persisted fold metadata.
- **Own-close provenance:** all 1430 table rows (1107 full books) come from their own fixture's close. There are 0 cross-match rows and 0 post-kickoff ticks, and the frozen KL is reproduced exactly.
- **Saved predictions:** repricing them gives the reported 1X2 LogLoss to ≤ 6.7e-16, with a probability gap of 0.

## Acceptance and execution

- **Sampling:** all three grids have 0 divergences and R̂ ≤ 1.0194, so no reruns were needed. The control's fold-40 tail ESS (330.7 < 400) is a review flag. Its bias posteriors above are typical.
- **Phase 4 a1** (`phase4_gate.sh a1 score leakage`, beast, fresh REPL per step): score in 2.6 min, leakage in 1.4 min, both PASS.
- **Record step:** read-only Phase 3 = Phase 4 score parity, then this experiment's register row and the canonical boards.
- **Exclusions:** no ROI, staking or favourable subsets. The flagged circular contradictions are reported, not substituted.
