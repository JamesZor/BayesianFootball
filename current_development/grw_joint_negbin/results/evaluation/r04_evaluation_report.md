# r04 proper scores — Task 014 (JointGammaNegBinObservation)

Generated 2026-09-12 07:08. Panel: 710 fixtures (24/25 + 25/26 walk-forward). Book: de-vigged Betfair TWA(−20, 0] close. Control reproduction: `m12_poisson` LogLoss 0.64437 / ECE 0.0086 vs published 0.64437 / 0.0086 on Task 013's own 1X2 + O/U 2.5 + BTTS basis (2899 rows).

Scored markets here are wider than Task 013's: 1X2, BTTS and O/U 1.5 / 2.5 / 3.5 / 4.5. The `all` scope therefore pools more rows than the 2,899 that reproduction figure is computed on, and is not comparable with it.

Read the totals and BTTS scopes as the test and the 1X2 scope as the control: a negative binomial redistributes mass within a fixed mean, so a change on O/U 3.5 with none on 1X2 is the mechanism behaving as stated.

## Scope: all

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_poisson | Joint Gamma-Poisson | 4054 | 0.61170 | 0.61908 | 0.21170 | 0.21142 | 0.22383 | 0.21110 | 0.0137 | 0.0108 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 4054 | 0.61173 | 0.61908 | 0.21169 | 0.21142 | 0.22378 | 0.21110 | 0.0146 | 0.0108 |
| m12_poisson | Joint Gamma-Poisson | 4054 | 0.61276 | 0.61908 | 0.21220 | 0.21142 | 0.22493 | 0.21110 | 0.0123 | 0.0108 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 4054 | 0.61290 | 0.61908 | 0.21226 | 0.21142 | 0.22504 | 0.21110 | 0.0107 | 0.0108 |
| m00_poisson | Poisson | 4054 | 0.61414 | 0.61908 | 0.21274 | 0.21142 | 0.22492 | 0.21110 | 0.0168 | 0.0108 |
| m00_baseline_grw_negbin | NegBin | 4054 | 0.61427 | 0.61908 | 0.21280 | 0.21142 | 0.22506 | 0.21110 | 0.0117 | 0.0108 |
| m10_poisson | Poisson | 4054 | 0.61495 | 0.61908 | 0.21316 | 0.21142 | 0.22631 | 0.21110 | 0.0120 | 0.0108 |
| m10_lineup_grw_negbin | NegBin | 4054 | 0.61502 | 0.61908 | 0.21318 | 0.21142 | 0.22621 | 0.21110 | 0.0118 | 0.0108 |

## Scope: 1X2

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 1785 | 0.61553 | 0.61312 | 0.21282 | 0.21156 | 0.22378 | 0.21110 | 0.0119 | 0.0186 |
| m05_poisson | Joint Gamma-Poisson | 1785 | 0.61558 | 0.61312 | 0.21286 | 0.21156 | 0.22383 | 0.21110 | 0.0124 | 0.0186 |
| m00_poisson | Poisson | 1785 | 0.61666 | 0.61312 | 0.21329 | 0.21156 | 0.22492 | 0.21110 | 0.0189 | 0.0186 |
| m00_baseline_grw_negbin | NegBin | 1785 | 0.61689 | 0.61312 | 0.21340 | 0.21156 | 0.22506 | 0.21110 | 0.0129 | 0.0186 |
| m12_poisson | Joint Gamma-Poisson | 1785 | 0.61701 | 0.61312 | 0.21350 | 0.21156 | 0.22493 | 0.21110 | 0.0140 | 0.0186 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 1785 | 0.61718 | 0.61312 | 0.21356 | 0.21156 | 0.22504 | 0.21110 | 0.0124 | 0.0186 |
| m10_lineup_grw_negbin | NegBin | 1785 | 0.61852 | 0.61312 | 0.21413 | 0.21156 | 0.22621 | 0.21110 | 0.0104 | 0.0186 |
| m10_poisson | Poisson | 1785 | 0.61867 | 0.61312 | 0.21421 | 0.21156 | 0.22631 | 0.21110 | 0.0108 | 0.0186 |

## Scope: OU1.5

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_poisson | Joint Gamma-Poisson | 430 | 0.52070 | 0.52728 | 0.17025 | 0.17266 | 0.0322 | 0.0103 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 430 | 0.52209 | 0.52728 | 0.17070 | 0.17266 | 0.0270 | 0.0103 |
| m12_poisson | Joint Gamma-Poisson | 430 | 0.52440 | 0.52728 | 0.17159 | 0.17266 | 0.0436 | 0.0103 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 430 | 0.52501 | 0.52728 | 0.17176 | 0.17266 | 0.0219 | 0.0103 |
| m00_poisson | Poisson | 430 | 0.53187 | 0.52728 | 0.17428 | 0.17266 | 0.0213 | 0.0103 |
| m00_baseline_grw_negbin | NegBin | 430 | 0.53238 | 0.52728 | 0.17448 | 0.17266 | 0.0358 | 0.0103 |
| m10_poisson | Poisson | 430 | 0.53271 | 0.52728 | 0.17452 | 0.17266 | 0.0318 | 0.0103 |
| m10_lineup_grw_negbin | NegBin | 430 | 0.53355 | 0.52728 | 0.17483 | 0.17266 | 0.0341 | 0.0103 |

## Scope: OU2.5

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | NegBin | 758 | 0.68709 | 0.68988 | 0.24699 | 0.24829 | 0.0254 | 0.0183 |
| m10_lineup_grw_negbin | NegBin | 758 | 0.68752 | 0.68988 | 0.24722 | 0.24829 | 0.0153 | 0.0183 |
| m00_poisson | Poisson | 758 | 0.68755 | 0.68988 | 0.24721 | 0.24829 | 0.0222 | 0.0183 |
| m10_poisson | Poisson | 758 | 0.68760 | 0.68988 | 0.24725 | 0.24829 | 0.0109 | 0.0183 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 758 | 0.68930 | 0.68988 | 0.24812 | 0.24829 | 0.0309 | 0.0183 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 758 | 0.68955 | 0.68988 | 0.24825 | 0.24829 | 0.0089 | 0.0183 |
| m12_poisson | Joint Gamma-Poisson | 758 | 0.68973 | 0.68988 | 0.24833 | 0.24829 | 0.0133 | 0.0183 |
| m05_poisson | Joint Gamma-Poisson | 758 | 0.68986 | 0.68988 | 0.24839 | 0.24829 | 0.0311 | 0.0183 |

## Scope: OU3.5

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 528 | 0.61180 | 0.61046 | 0.21036 | 0.20987 | 0.0363 | 0.0148 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 528 | 0.61213 | 0.61046 | 0.21067 | 0.20987 | 0.0239 | 0.0148 |
| m12_poisson | Joint Gamma-Poisson | 528 | 0.61230 | 0.61046 | 0.21067 | 0.20987 | 0.0180 | 0.0148 |
| m05_poisson | Joint Gamma-Poisson | 528 | 0.61235 | 0.61046 | 0.21054 | 0.20987 | 0.0465 | 0.0148 |
| m10_lineup_grw_negbin | NegBin | 528 | 0.61959 | 0.61046 | 0.21369 | 0.20987 | 0.0353 | 0.0148 |
| m10_poisson | Poisson | 528 | 0.61998 | 0.61046 | 0.21379 | 0.20987 | 0.0253 | 0.0148 |
| m00_baseline_grw_negbin | NegBin | 528 | 0.62012 | 0.61046 | 0.21367 | 0.20987 | 0.0312 | 0.0148 |
| m00_poisson | Poisson | 528 | 0.62070 | 0.61046 | 0.21383 | 0.20987 | 0.0306 | 0.0148 |

## Scope: OU4.5

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m10_poisson | Poisson | 197 | 0.32977 | 0.50791 | 0.09027 | 0.09594 | 0.0887 | 0.0963 |
| m10_lineup_grw_negbin | NegBin | 197 | 0.33120 | 0.50791 | 0.09059 | 0.09594 | 0.0950 | 0.0963 |
| m00_poisson | Poisson | 197 | 0.33176 | 0.50791 | 0.09067 | 0.09594 | 0.0928 | 0.0963 |
| m00_baseline_grw_negbin | NegBin | 197 | 0.33358 | 0.50791 | 0.09124 | 0.09594 | 0.0966 | 0.0963 |
| m12_poisson | Joint Gamma-Poisson | 197 | 0.34171 | 0.50791 | 0.09327 | 0.09594 | 0.0823 | 0.0963 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 197 | 0.34264 | 0.50791 | 0.09363 | 0.09594 | 0.0863 | 0.0963 |
| m05_poisson | Joint Gamma-Poisson | 197 | 0.34576 | 0.50791 | 0.09450 | 0.09594 | 0.0850 | 0.0963 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 197 | 0.34674 | 0.50791 | 0.09480 | 0.09594 | 0.0895 | 0.0963 |

## Scope: BTTS

| model | likelihood | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_poisson | Joint Gamma-Poisson | 356 | 0.68192 | 0.68337 | 0.24445 | 0.24516 | 0.0366 | 0.0300 |
| m05_wealth_grw_negbin | Joint Gamma-NegBin | 356 | 0.68236 | 0.68337 | 0.24465 | 0.24516 | 0.0192 | 0.0300 |
| m12_poisson | Joint Gamma-Poisson | 356 | 0.68496 | 0.68337 | 0.24591 | 0.24516 | 0.0402 | 0.0300 |
| m12_joint_hybrid_synergy_negbin | Joint Gamma-NegBin | 356 | 0.68508 | 0.68337 | 0.24596 | 0.24516 | 0.0679 | 0.0300 |
| m00_poisson | Poisson | 356 | 0.69108 | 0.68337 | 0.24897 | 0.24516 | 0.0353 | 0.0300 |
| m10_poisson | Poisson | 356 | 0.69125 | 0.68337 | 0.24906 | 0.24516 | 0.0205 | 0.0300 |
| m00_baseline_grw_negbin | NegBin | 356 | 0.69168 | 0.68337 | 0.24927 | 0.24516 | 0.0145 | 0.0300 |
| m10_lineup_grw_negbin | NegBin | 356 | 0.69177 | 0.68337 | 0.24932 | 0.24516 | 0.0128 | 0.0300 |

## Paired Δ, fixture-clustered bootstrap (B = 10000)

Each row is a NegBin rung minus its Task 013 Poisson counterpart, the two differing in exactly one component. Negative Δ favours the NegBin arm; `p_better` is the share of resamples in which it scored lower. 95% percentile interval.

### LogLoss

| contrast | scope | n_obs | n_fixtures | delta | lo | hi | p_better | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin − m00_poisson | all | 4054 | 630 | +0.00013 | -0.00059 | +0.00083 | 0.3441 | false |
| m00_baseline_grw_negbin − m00_poisson | 1X2 | 1785 | 595 | +0.00023 | -0.00033 | +0.00079 | 0.2077 | false |
| m00_baseline_grw_negbin − m00_poisson | OU1.5 | 430 | 215 | +0.00051 | -0.00201 | +0.00281 | 0.3404 | false |
| m00_baseline_grw_negbin − m00_poisson | OU2.5 | 758 | 379 | -0.00046 | -0.00176 | +0.00079 | 0.7623 | false |
| m00_baseline_grw_negbin − m00_poisson | OU3.5 | 528 | 264 | -0.00058 | -0.00196 | +0.00071 | 0.8085 | false |
| m00_baseline_grw_negbin − m00_poisson | OU4.5 | 197 | 104 | +0.00182 | -0.00130 | +0.00421 | 0.1036 | false |
| m00_baseline_grw_negbin − m00_poisson | BTTS | 356 | 178 | +0.00059 | -0.00302 | +0.00408 | 0.3754 | false |
| m05_wealth_grw_negbin − m05_poisson | all | 4054 | 630 | +0.00003 | -0.00061 | +0.00066 | 0.4556 | false |
| m05_wealth_grw_negbin − m05_poisson | 1X2 | 1785 | 595 | -0.00005 | -0.00054 | +0.00044 | 0.5760 | false |
| m05_wealth_grw_negbin − m05_poisson | OU1.5 | 430 | 215 | +0.00138 | -0.00100 | +0.00354 | 0.1240 | false |
| m05_wealth_grw_negbin − m05_poisson | OU2.5 | 758 | 379 | -0.00056 | -0.00183 | +0.00071 | 0.8141 | false |
| m05_wealth_grw_negbin − m05_poisson | OU3.5 | 528 | 264 | -0.00055 | -0.00163 | +0.00049 | 0.8507 | false |
| m05_wealth_grw_negbin − m05_poisson | OU4.5 | 197 | 104 | +0.00098 | -0.00153 | +0.00310 | 0.2006 | false |
| m05_wealth_grw_negbin − m05_poisson | BTTS | 356 | 178 | +0.00044 | -0.00312 | +0.00387 | 0.4074 | false |
| m10_lineup_grw_negbin − m10_poisson | all | 4054 | 630 | +0.00007 | -0.00063 | +0.00074 | 0.4177 | false |
| m10_lineup_grw_negbin − m10_poisson | 1X2 | 1785 | 595 | -0.00015 | -0.00068 | +0.00038 | 0.7082 | false |
| m10_lineup_grw_negbin − m10_poisson | OU1.5 | 430 | 215 | +0.00084 | -0.00179 | +0.00321 | 0.2506 | false |
| m10_lineup_grw_negbin − m10_poisson | OU2.5 | 758 | 379 | -0.00008 | -0.00139 | +0.00126 | 0.5429 | false |
| m10_lineup_grw_negbin − m10_poisson | OU3.5 | 528 | 264 | -0.00039 | -0.00172 | +0.00088 | 0.7214 | false |
| m10_lineup_grw_negbin − m10_poisson | OU4.5 | 197 | 104 | +0.00143 | -0.00101 | +0.00345 | 0.1147 | false |
| m10_lineup_grw_negbin − m10_poisson | BTTS | 356 | 178 | +0.00052 | -0.00305 | +0.00404 | 0.3908 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | all | 4054 | 630 | +0.00014 | -0.00053 | +0.00077 | 0.3331 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | 1X2 | 1785 | 595 | +0.00017 | -0.00027 | +0.00060 | 0.2241 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU1.5 | 430 | 215 | +0.00061 | -0.00192 | +0.00295 | 0.3083 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU2.5 | 758 | 379 | -0.00018 | -0.00143 | +0.00105 | 0.6121 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU3.5 | 528 | 264 | -0.00016 | -0.00131 | +0.00091 | 0.6163 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU4.5 | 197 | 104 | +0.00093 | -0.00215 | +0.00323 | 0.2367 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | BTTS | 356 | 178 | +0.00012 | -0.00341 | +0.00357 | 0.4803 | false |

### Brier

| contrast | scope | n_obs | n_fixtures | delta | lo | hi | p_better | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin − m00_poisson | all | 4054 | 630 | +0.00006 | -0.00025 | +0.00035 | 0.3453 | false |
| m00_baseline_grw_negbin − m00_poisson | 1X2 | 1785 | 595 | +0.00010 | -0.00014 | +0.00034 | 0.2069 | false |
| m00_baseline_grw_negbin − m00_poisson | OU1.5 | 430 | 215 | +0.00020 | -0.00065 | +0.00098 | 0.3180 | false |
| m00_baseline_grw_negbin − m00_poisson | OU2.5 | 758 | 379 | -0.00022 | -0.00084 | +0.00039 | 0.7600 | false |
| m00_baseline_grw_negbin − m00_poisson | OU3.5 | 528 | 264 | -0.00016 | -0.00069 | +0.00036 | 0.7224 | false |
| m00_baseline_grw_negbin − m00_poisson | OU4.5 | 197 | 104 | +0.00057 | +0.00005 | +0.00102 | 0.0173 | true |
| m00_baseline_grw_negbin − m00_poisson | BTTS | 356 | 178 | +0.00030 | -0.00145 | +0.00200 | 0.3712 | false |
| m05_wealth_grw_negbin − m05_poisson | all | 4054 | 630 | -0.00001 | -0.00029 | +0.00026 | 0.5227 | false |
| m05_wealth_grw_negbin − m05_poisson | 1X2 | 1785 | 595 | -0.00003 | -0.00024 | +0.00018 | 0.6176 | false |
| m05_wealth_grw_negbin − m05_poisson | OU1.5 | 430 | 215 | +0.00044 | -0.00038 | +0.00121 | 0.1422 | false |
| m05_wealth_grw_negbin − m05_poisson | OU2.5 | 758 | 379 | -0.00027 | -0.00088 | +0.00034 | 0.8149 | false |
| m05_wealth_grw_negbin − m05_poisson | OU3.5 | 528 | 264 | -0.00019 | -0.00063 | +0.00025 | 0.7998 | false |
| m05_wealth_grw_negbin − m05_poisson | OU4.5 | 197 | 104 | +0.00030 | -0.00022 | +0.00076 | 0.1203 | false |
| m05_wealth_grw_negbin − m05_poisson | BTTS | 356 | 178 | +0.00020 | -0.00152 | +0.00188 | 0.4115 | false |
| m10_lineup_grw_negbin − m10_poisson | all | 4054 | 630 | +0.00002 | -0.00028 | +0.00031 | 0.4524 | false |
| m10_lineup_grw_negbin − m10_poisson | 1X2 | 1785 | 595 | -0.00008 | -0.00031 | +0.00016 | 0.7357 | false |
| m10_lineup_grw_negbin − m10_poisson | OU1.5 | 430 | 215 | +0.00031 | -0.00057 | +0.00112 | 0.2364 | false |
| m10_lineup_grw_negbin − m10_poisson | OU2.5 | 758 | 379 | -0.00003 | -0.00067 | +0.00061 | 0.5368 | false |
| m10_lineup_grw_negbin − m10_poisson | OU3.5 | 528 | 264 | -0.00011 | -0.00064 | +0.00042 | 0.6581 | false |
| m10_lineup_grw_negbin − m10_poisson | OU4.5 | 197 | 104 | +0.00032 | -0.00018 | +0.00076 | 0.1024 | false |
| m10_lineup_grw_negbin − m10_poisson | BTTS | 356 | 178 | +0.00026 | -0.00148 | +0.00198 | 0.3878 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | all | 4054 | 630 | +0.00005 | -0.00022 | +0.00032 | 0.3442 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | 1X2 | 1785 | 595 | +0.00007 | -0.00012 | +0.00026 | 0.2475 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU1.5 | 430 | 215 | +0.00018 | -0.00070 | +0.00098 | 0.3380 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU2.5 | 758 | 379 | -0.00009 | -0.00069 | +0.00050 | 0.6113 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU3.5 | 528 | 264 | +0.00000 | -0.00044 | +0.00043 | 0.5001 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | OU4.5 | 197 | 104 | +0.00036 | -0.00011 | +0.00076 | 0.0609 | false |
| m12_joint_hybrid_synergy_negbin − m12_poisson | BTTS | 356 | 178 | +0.00006 | -0.00165 | +0.00174 | 0.4802 | false |

## Every arm against the Betfair close (ΔLogLoss)

| model | scope | n_obs | n_fixtures | delta | lo | hi | p_negative | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m00_baseline_grw_negbin | all | 4054 | 630 | -0.00480 | -0.02413 | +0.00798 | 0.6790 | false |
| m00_baseline_grw_negbin | 1X2 | 1785 | 595 | +0.00377 | -0.00646 | +0.01388 | 0.2334 | false |
| m00_baseline_grw_negbin | OU1.5 | 430 | 215 | +0.00510 | -0.01132 | +0.02083 | 0.2615 | false |
| m00_baseline_grw_negbin | OU2.5 | 758 | 379 | -0.00279 | -0.01629 | +0.01082 | 0.6633 | false |
| m00_baseline_grw_negbin | OU3.5 | 528 | 264 | +0.00967 | -0.00750 | +0.02695 | 0.1304 | false |
| m00_baseline_grw_negbin | OU4.5 | 197 | 104 | -0.17433 | -0.51656 | +0.00317 | 0.9537 | false |
| m00_baseline_grw_negbin | BTTS | 356 | 178 | +0.00830 | -0.00713 | +0.02387 | 0.1473 | false |
| m05_wealth_grw_negbin | all | 4054 | 630 | -0.00734 | -0.02741 | +0.00545 | 0.7839 | false |
| m05_wealth_grw_negbin | 1X2 | 1785 | 595 | +0.00242 | -0.00597 | +0.01080 | 0.2793 | false |
| m05_wealth_grw_negbin | OU1.5 | 430 | 215 | -0.00519 | -0.02131 | +0.01014 | 0.7389 | false |
| m05_wealth_grw_negbin | OU2.5 | 758 | 379 | -0.00058 | -0.01328 | +0.01210 | 0.5365 | false |
| m05_wealth_grw_negbin | OU3.5 | 528 | 264 | +0.00134 | -0.01485 | +0.01829 | 0.4385 | false |
| m05_wealth_grw_negbin | OU4.5 | 197 | 104 | -0.16117 | -0.50935 | +0.01718 | 0.7678 | false |
| m05_wealth_grw_negbin | BTTS | 356 | 178 | -0.00101 | -0.01452 | +0.01213 | 0.5643 | false |
| m10_lineup_grw_negbin | all | 4054 | 630 | -0.00406 | -0.02328 | +0.00854 | 0.6471 | false |
| m10_lineup_grw_negbin | 1X2 | 1785 | 595 | +0.00540 | -0.00550 | +0.01606 | 0.1622 | false |
| m10_lineup_grw_negbin | OU1.5 | 430 | 215 | +0.00627 | -0.00876 | +0.02051 | 0.1964 | false |
| m10_lineup_grw_negbin | OU2.5 | 758 | 379 | -0.00235 | -0.01485 | +0.01008 | 0.6453 | false |
| m10_lineup_grw_negbin | OU3.5 | 528 | 264 | +0.00913 | -0.00622 | +0.02468 | 0.1159 | false |
| m10_lineup_grw_negbin | OU4.5 | 197 | 104 | -0.17671 | -0.51934 | +0.00036 | 0.9737 | false |
| m10_lineup_grw_negbin | BTTS | 356 | 178 | +0.00840 | -0.00515 | +0.02228 | 0.1148 | false |
| m12_joint_hybrid_synergy_negbin | all | 4054 | 630 | -0.00618 | -0.02618 | +0.00667 | 0.7266 | false |
| m12_joint_hybrid_synergy_negbin | 1X2 | 1785 | 595 | +0.00406 | -0.00509 | +0.01301 | 0.1866 | false |
| m12_joint_hybrid_synergy_negbin | OU1.5 | 430 | 215 | -0.00227 | -0.01826 | +0.01290 | 0.6017 | false |
| m12_joint_hybrid_synergy_negbin | OU2.5 | 758 | 379 | -0.00033 | -0.01268 | +0.01201 | 0.5225 | false |
| m12_joint_hybrid_synergy_negbin | OU3.5 | 528 | 264 | +0.00167 | -0.01350 | +0.01743 | 0.4195 | false |
| m12_joint_hybrid_synergy_negbin | OU4.5 | 197 | 104 | -0.16527 | -0.51184 | +0.01226 | 0.8258 | false |
| m12_joint_hybrid_synergy_negbin | BTTS | 356 | 178 | +0.00171 | -0.01152 | +0.01476 | 0.4091 | false |
| m00_poisson | all | 4054 | 630 | -0.00494 | -0.02417 | +0.00778 | 0.6868 | false |
| m00_poisson | 1X2 | 1785 | 595 | +0.00354 | -0.00670 | +0.01360 | 0.2454 | false |
| m00_poisson | OU1.5 | 430 | 215 | +0.00459 | -0.01136 | +0.02015 | 0.2810 | false |
| m00_poisson | OU2.5 | 758 | 379 | -0.00233 | -0.01604 | +0.01141 | 0.6337 | false |
| m00_poisson | OU3.5 | 528 | 264 | +0.01024 | -0.00748 | +0.02817 | 0.1248 | false |
| m00_poisson | OU4.5 | 197 | 104 | -0.17615 | -0.51724 | +0.00064 | 0.9724 | false |
| m00_poisson | BTTS | 356 | 178 | +0.00771 | -0.00655 | +0.02223 | 0.1485 | false |
| m05_poisson | all | 4054 | 630 | -0.00738 | -0.02761 | +0.00545 | 0.7852 | false |
| m05_poisson | 1X2 | 1785 | 595 | +0.00247 | -0.00586 | +0.01072 | 0.2731 | false |
| m05_poisson | OU1.5 | 430 | 215 | -0.00658 | -0.02228 | +0.00853 | 0.7952 | false |
| m05_poisson | OU2.5 | 758 | 379 | -0.00002 | -0.01286 | +0.01298 | 0.4983 | false |
| m05_poisson | OU3.5 | 528 | 264 | +0.00189 | -0.01477 | +0.01925 | 0.4142 | false |
| m05_poisson | OU4.5 | 197 | 104 | -0.16214 | -0.50967 | +0.01579 | 0.7852 | false |
| m05_poisson | BTTS | 356 | 178 | -0.00145 | -0.01390 | +0.01096 | 0.5931 | false |
| m10_poisson | all | 4054 | 630 | -0.00413 | -0.02335 | +0.00849 | 0.6493 | false |
| m10_poisson | 1X2 | 1785 | 595 | +0.00555 | -0.00524 | +0.01622 | 0.1553 | false |
| m10_poisson | OU1.5 | 430 | 215 | +0.00543 | -0.00911 | +0.01946 | 0.2230 | false |
| m10_poisson | OU2.5 | 758 | 379 | -0.00228 | -0.01493 | +0.01029 | 0.6376 | false |
| m10_poisson | OU3.5 | 528 | 264 | +0.00952 | -0.00641 | +0.02566 | 0.1154 | false |
| m10_poisson | OU4.5 | 197 | 104 | -0.17814 | -0.51950 | -0.00184 | 0.9846 | true |
| m10_poisson | BTTS | 356 | 178 | +0.00788 | -0.00436 | +0.02053 | 0.1080 | false |
| m12_poisson | all | 4054 | 630 | -0.00632 | -0.02640 | +0.00657 | 0.7339 | false |
| m12_poisson | 1X2 | 1785 | 595 | +0.00389 | -0.00516 | +0.01277 | 0.1951 | false |
| m12_poisson | OU1.5 | 430 | 215 | -0.00288 | -0.01855 | +0.01193 | 0.6343 | false |
| m12_poisson | OU2.5 | 758 | 379 | -0.00015 | -0.01265 | +0.01229 | 0.5110 | false |
| m12_poisson | OU3.5 | 528 | 264 | +0.00184 | -0.01370 | +0.01802 | 0.4124 | false |
| m12_poisson | OU4.5 | 197 | 104 | -0.16620 | -0.51266 | +0.01117 | 0.8453 | false |
| m12_poisson | BTTS | 356 | 178 | +0.00159 | -0.01072 | +0.01385 | 0.4025 | false |
