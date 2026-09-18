# r04 proper scores — Task 015 (market-anchored MultiScaleGRW)

Generated 2026-09-13 20:12. Panel 710 fixtures (24/25 + 25/26 walk-forward). Book: de-vigged Betfair TWA(−20, 0] close. Baseline reproduction: LogLoss 0.64315 / ECE 0.0123 on 2899 rows (published 0.64315 / 0.0123).

Runs: `m05_joint_grw_baseline` `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`; `m05_joint_grw_supremacy_w040` `0ee58d18-b7e9-4168-8d78-93887b1a8c26`; `m05_joint_grw_smile_supremacy_w020` `fcd5e974-9a46-4a10-9828-6b987a5484d6`; `m05_joint_grw_smile_supremacy_w040` `30620d3e-e4bd-4c05-b1a1-85cefa36b728`; `m05_joint_grw_smile_supremacy_w070` `32d588f1-d666-4112-a7e1-5c9545fbbe3d`.

`all` pools 1X2 + O/U 2.5 + BTTS. OU1.5 / OU3.5 are secondary and never pooled.

## Scope: all

| model | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w040 | 2899 | 0.64092 | 0.64182 | 0.22494 | 0.22529 | 0.22128 | 0.21110 | 0.0145 | 0.0139 |
| m05_joint_grw_smile_supremacy_w070 | 2899 | 0.64097 | 0.64182 | 0.22496 | 0.22529 | 0.22128 | 0.21110 | 0.0198 | 0.0139 |
| m05_joint_grw_smile_supremacy_w020 | 2899 | 0.64116 | 0.64182 | 0.22505 | 0.22529 | 0.22164 | 0.21110 | 0.0152 | 0.0139 |
| m05_joint_grw_supremacy_w040 | 2899 | 0.64135 | 0.64182 | 0.22516 | 0.22529 | 0.22221 | 0.21110 | 0.0195 | 0.0139 |
| m05_joint_grw_baseline | 2899 | 0.64315 | 0.64182 | 0.22603 | 0.22529 | 0.22383 | 0.21110 | 0.0123 | 0.0139 |

## Scope: 1X2

| model | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w040 | 1785 | 0.61209 | 0.61312 | 0.21118 | 0.21156 | 0.22128 | 0.21110 | 0.0251 | 0.0186 |
| m05_joint_grw_smile_supremacy_w070 | 1785 | 0.61230 | 0.61312 | 0.21128 | 0.21156 | 0.22128 | 0.21110 | 0.0357 | 0.0186 |
| m05_joint_grw_smile_supremacy_w020 | 1785 | 0.61245 | 0.61312 | 0.21134 | 0.21156 | 0.22164 | 0.21110 | 0.0243 | 0.0186 |
| m05_joint_grw_supremacy_w040 | 1785 | 0.61338 | 0.61312 | 0.21181 | 0.21156 | 0.22221 | 0.21110 | 0.0234 | 0.0186 |
| m05_joint_grw_baseline | 1785 | 0.61558 | 0.61312 | 0.21286 | 0.21156 | 0.22383 | 0.21110 | 0.0124 | 0.0186 |

## Scope: OU2.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w070 | 758 | 0.68668 | 0.68988 | 0.24677 | 0.24829 | 0.0181 | 0.0183 |
| m05_joint_grw_smile_supremacy_w040 | 758 | 0.68698 | 0.68988 | 0.24691 | 0.24829 | 0.0228 | 0.0183 |
| m05_joint_grw_smile_supremacy_w020 | 758 | 0.68705 | 0.68988 | 0.24694 | 0.24829 | 0.0179 | 0.0183 |
| m05_joint_grw_supremacy_w040 | 758 | 0.68796 | 0.68988 | 0.24743 | 0.24829 | 0.0278 | 0.0183 |
| m05_joint_grw_baseline | 758 | 0.68986 | 0.68988 | 0.24839 | 0.24829 | 0.0311 | 0.0183 |

## Scope: BTTS

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 356 | 0.68192 | 0.68337 | 0.24445 | 0.24516 | 0.0366 | 0.0300 |
| m05_joint_grw_supremacy_w040 | 356 | 0.68235 | 0.68337 | 0.24467 | 0.24516 | 0.0042 | 0.0300 |
| m05_joint_grw_smile_supremacy_w070 | 356 | 0.68734 | 0.68337 | 0.24711 | 0.24516 | 0.0255 | 0.0300 |
| m05_joint_grw_smile_supremacy_w040 | 356 | 0.68741 | 0.68337 | 0.24714 | 0.24516 | 0.0095 | 0.0300 |
| m05_joint_grw_smile_supremacy_w020 | 356 | 0.68742 | 0.68337 | 0.24715 | 0.24516 | 0.0169 | 0.0300 |

## Scope: OU1.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 430 | 0.52070 | 0.52728 | 0.17025 | 0.17266 | 0.0322 | 0.0103 |
| m05_joint_grw_supremacy_w040 | 430 | 0.52285 | 0.52728 | 0.17122 | 0.17266 | 0.0234 | 0.0103 |
| m05_joint_grw_smile_supremacy_w070 | 430 | 0.52706 | 0.52728 | 0.17267 | 0.17266 | 0.0208 | 0.0103 |
| m05_joint_grw_smile_supremacy_w040 | 430 | 0.52765 | 0.52728 | 0.17286 | 0.17266 | 0.0202 | 0.0103 |
| m05_joint_grw_smile_supremacy_w020 | 430 | 0.52784 | 0.52728 | 0.17289 | 0.17266 | 0.0199 | 0.0103 |

## Scope: OU3.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w070 | 528 | 0.60853 | 0.61046 | 0.20903 | 0.20987 | 0.0288 | 0.0148 |
| m05_joint_grw_smile_supremacy_w040 | 528 | 0.60897 | 0.61046 | 0.20927 | 0.20987 | 0.0201 | 0.0148 |
| m05_joint_grw_smile_supremacy_w020 | 528 | 0.60948 | 0.61046 | 0.20953 | 0.20987 | 0.0203 | 0.0148 |
| m05_joint_grw_supremacy_w040 | 528 | 0.61131 | 0.61046 | 0.21016 | 0.20987 | 0.0306 | 0.0148 |
| m05_joint_grw_baseline | 528 | 0.61235 | 0.61046 | 0.21054 | 0.20987 | 0.0465 | 0.0148 |

## Paired Δ (candidate − reference), fixture-clustered bootstrap B = 10000

Negative Δ favours the candidate. 95% percentile interval.

### LogLoss

| contrast | scope | n_obs | n_fixtures | delta | lo | hi | p_better | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00180 | -0.00440 | +0.00082 | 0.9087 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00221 | -0.00612 | +0.00173 | 0.8623 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00190 | -0.00495 | +0.00119 | 0.8872 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00042 | -0.00300 | +0.00384 | 0.4009 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00215 | -0.00057 | +0.00479 | 0.0593 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00104 | -0.00507 | +0.00302 | 0.6890 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00199 | -0.00676 | +0.00286 | 0.7870 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00314 | -0.00738 | +0.00107 | 0.9286 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00282 | -0.01506 | +0.00957 | 0.6720 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00550 | -0.00844 | +0.01924 | 0.2215 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00713 | -0.00648 | +0.02026 | 0.1438 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00287 | -0.01984 | +0.01272 | 0.6237 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00223 | -0.00715 | +0.00269 | 0.8158 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00349 | -0.00787 | +0.00090 | 0.9370 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00288 | -0.01549 | +0.00980 | 0.6724 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00548 | -0.00824 | +0.01896 | 0.2195 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00694 | -0.00687 | +0.02020 | 0.1539 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00338 | -0.02068 | +0.01263 | 0.6440 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00219 | -0.00717 | +0.00281 | 0.8061 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00328 | -0.00803 | +0.00153 | 0.9102 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00318 | -0.01597 | +0.00959 | 0.6864 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00542 | -0.00807 | +0.01864 | 0.2149 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00636 | -0.00765 | +0.01991 | 0.1795 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00382 | -0.02141 | +0.01249 | 0.6612 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | all | 2899 | 627 | -0.00043 | -0.00489 | +0.00405 | 0.5781 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | 1X2 | 1785 | 595 | -0.00129 | -0.00376 | +0.00117 | 0.8458 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | OU2.5 | 758 | 379 | -0.00097 | -0.01366 | +0.01166 | 0.5600 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | BTTS | 356 | 178 | +0.00506 | -0.00925 | +0.01907 | 0.2450 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | OU1.5 | 430 | 215 | +0.00480 | -0.00894 | +0.01828 | 0.2395 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | OU3.5 | 528 | 264 | -0.00234 | -0.01990 | +0.01400 | 0.5962 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | all | 2899 | 627 | +0.00024 | -0.00069 | +0.00117 | 0.2999 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | 595 | +0.00036 | -0.00099 | +0.00170 | 0.2986 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | 379 | +0.00006 | -0.00114 | +0.00132 | 0.4555 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | 178 | +0.00001 | -0.00175 | +0.00175 | 0.4952 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | 215 | +0.00019 | -0.00110 | +0.00150 | 0.3899 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | 264 | +0.00051 | -0.00112 | +0.00214 | 0.2712 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | all | 2899 | 627 | +0.00005 | -0.00089 | +0.00098 | 0.4690 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | 595 | +0.00021 | -0.00113 | +0.00161 | 0.3863 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | 379 | -0.00030 | -0.00148 | +0.00086 | 0.6944 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | 178 | -0.00007 | -0.00180 | +0.00172 | 0.5359 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | 215 | -0.00059 | -0.00190 | +0.00068 | 0.8129 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | 264 | -0.00045 | -0.00193 | +0.00098 | 0.7261 | false |

### Brier

| contrast | scope | n_obs | n_fixtures | delta | lo | hi | p_better | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00087 | -0.00205 | +0.00032 | 0.9220 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00104 | -0.00279 | +0.00073 | 0.8767 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00096 | -0.00243 | +0.00053 | 0.8968 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00022 | -0.00144 | +0.00189 | 0.3945 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00096 | -0.00007 | +0.00197 | 0.0324 | false |
| m05_joint_grw_supremacy_w040 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00038 | -0.00210 | +0.00136 | 0.6656 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00098 | -0.00326 | +0.00134 | 0.7964 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00152 | -0.00343 | +0.00037 | 0.9411 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00145 | -0.00742 | +0.00460 | 0.6816 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00270 | -0.00415 | +0.00941 | 0.2215 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00264 | -0.00232 | +0.00762 | 0.1436 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00102 | -0.00783 | +0.00539 | 0.6075 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00109 | -0.00341 | +0.00126 | 0.8192 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00167 | -0.00363 | +0.00030 | 0.9512 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00148 | -0.00763 | +0.00475 | 0.6822 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00269 | -0.00404 | +0.00927 | 0.2174 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00261 | -0.00249 | +0.00766 | 0.1521 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00128 | -0.00832 | +0.00534 | 0.6324 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00107 | -0.00342 | +0.00130 | 0.8118 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00157 | -0.00369 | +0.00058 | 0.9244 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00162 | -0.00790 | +0.00463 | 0.6916 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00266 | -0.00392 | +0.00915 | 0.2143 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00241 | -0.00277 | +0.00752 | 0.1770 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00151 | -0.00868 | +0.00532 | 0.6560 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | all | 2899 | 627 | -0.00022 | -0.00236 | +0.00193 | 0.5823 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | 1X2 | 1785 | 595 | -0.00063 | -0.00169 | +0.00044 | 0.8742 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | OU2.5 | 758 | 379 | -0.00052 | -0.00669 | +0.00563 | 0.5650 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | BTTS | 356 | 178 | +0.00247 | -0.00452 | +0.00934 | 0.2436 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | OU1.5 | 430 | 215 | +0.00164 | -0.00345 | +0.00680 | 0.2574 | false |
| m05_joint_grw_smile_supremacy_w040 − m05_joint_grw_supremacy_w040 | OU3.5 | 528 | 264 | -0.00089 | -0.00799 | +0.00588 | 0.5914 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | all | 2899 | 627 | +0.00010 | -0.00032 | +0.00053 | 0.3068 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | 595 | +0.00016 | -0.00045 | +0.00076 | 0.3038 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | 379 | +0.00003 | -0.00057 | +0.00065 | 0.4625 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | 178 | +0.00000 | -0.00087 | +0.00086 | 0.4952 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | 215 | +0.00003 | -0.00046 | +0.00052 | 0.4512 | false |
| m05_joint_grw_smile_supremacy_w020 − m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | 264 | +0.00026 | -0.00046 | +0.00099 | 0.2434 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | all | 2899 | 627 | +0.00002 | -0.00041 | +0.00045 | 0.4692 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | 595 | +0.00010 | -0.00050 | +0.00072 | 0.3829 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | 379 | -0.00014 | -0.00073 | +0.00043 | 0.6891 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | 178 | -0.00003 | -0.00089 | +0.00085 | 0.5339 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | 215 | -0.00019 | -0.00070 | +0.00029 | 0.7776 | false |
| m05_joint_grw_smile_supremacy_w070 − m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | 264 | -0.00023 | -0.00090 | +0.00040 | 0.7605 | false |

## Every arm against the Betfair close (ΔLogLoss)

| model | scope | n_obs | delta | lo | hi | p_negative | significant |
|---|---|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | all | 2899 | +0.00134 | -0.00473 | +0.00744 | 0.3367 | false |
| m05_joint_grw_baseline | 1X2 | 1785 | +0.00247 | -0.00590 | +0.01053 | 0.2760 | false |
| m05_joint_grw_baseline | OU2.5 | 758 | -0.00002 | -0.01302 | +0.01320 | 0.4992 | false |
| m05_joint_grw_baseline | BTTS | 356 | -0.00145 | -0.01375 | +0.01109 | 0.5907 | false |
| m05_joint_grw_baseline | OU1.5 | 430 | -0.00658 | -0.02273 | +0.00846 | 0.8011 | false |
| m05_joint_grw_baseline | OU3.5 | 528 | +0.00189 | -0.01502 | +0.01903 | 0.4240 | false |
| m05_joint_grw_supremacy_w040 | all | 2899 | -0.00047 | -0.00608 | +0.00513 | 0.5635 | false |
| m05_joint_grw_supremacy_w040 | 1X2 | 1785 | +0.00026 | -0.00699 | +0.00728 | 0.4602 | false |
| m05_joint_grw_supremacy_w040 | OU2.5 | 758 | -0.00192 | -0.01512 | +0.01156 | 0.6096 | false |
| m05_joint_grw_supremacy_w040 | BTTS | 356 | -0.00102 | -0.01397 | +0.01208 | 0.5616 | false |
| m05_joint_grw_supremacy_w040 | OU1.5 | 430 | -0.00443 | -0.02097 | +0.01076 | 0.7046 | false |
| m05_joint_grw_supremacy_w040 | OU3.5 | 528 | +0.00085 | -0.01616 | +0.01838 | 0.4747 | false |
| m05_joint_grw_smile_supremacy_w020 | all | 2899 | -0.00066 | -0.00610 | +0.00469 | 0.5813 | false |
| m05_joint_grw_smile_supremacy_w020 | 1X2 | 1785 | -0.00067 | -0.00907 | +0.00749 | 0.5569 | false |
| m05_joint_grw_smile_supremacy_w020 | OU2.5 | 758 | -0.00283 | -0.01073 | +0.00503 | 0.7546 | false |
| m05_joint_grw_smile_supremacy_w020 | BTTS | 356 | +0.00405 | -0.00410 | +0.01208 | 0.1611 | false |
| m05_joint_grw_smile_supremacy_w020 | OU1.5 | 430 | +0.00056 | -0.01027 | +0.01070 | 0.4500 | false |
| m05_joint_grw_smile_supremacy_w020 | OU3.5 | 528 | -0.00098 | -0.00998 | +0.00771 | 0.5844 | false |
| m05_joint_grw_smile_supremacy_w040 | all | 2899 | -0.00090 | -0.00608 | +0.00417 | 0.6244 | false |
| m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | -0.00103 | -0.00888 | +0.00659 | 0.5960 | false |
| m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | -0.00290 | -0.01071 | +0.00479 | 0.7608 | false |
| m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | +0.00404 | -0.00385 | +0.01180 | 0.1538 | false |
| m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | +0.00037 | -0.01036 | +0.01024 | 0.4616 | false |
| m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | -0.00149 | -0.01014 | +0.00690 | 0.6312 | false |
| m05_joint_grw_smile_supremacy_w070 | all | 2899 | -0.00085 | -0.00575 | +0.00397 | 0.6291 | false |
| m05_joint_grw_smile_supremacy_w070 | 1X2 | 1785 | -0.00081 | -0.00805 | +0.00624 | 0.5789 | false |
| m05_joint_grw_smile_supremacy_w070 | OU2.5 | 758 | -0.00319 | -0.01099 | +0.00440 | 0.7854 | false |
| m05_joint_grw_smile_supremacy_w070 | BTTS | 356 | +0.00397 | -0.00378 | +0.01172 | 0.1580 | false |
| m05_joint_grw_smile_supremacy_w070 | OU1.5 | 430 | -0.00022 | -0.01099 | +0.00951 | 0.5055 | false |
| m05_joint_grw_smile_supremacy_w070 | OU3.5 | 528 | -0.00193 | -0.01051 | +0.00633 | 0.6733 | false |

## Home-favourite compression (1X2 home selection)

| model | market_bin | n | mean_p_market | mean_p_model | mean_gap | home_win_rate |
|---|---|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | [0.00, 0.30) | 111 | 0.2370 | 0.3022 | +0.0652 | 0.2523 |
| m05_joint_grw_smile_supremacy_w020 | [0.00, 0.30) | 111 | 0.2370 | 0.3211 | +0.0840 | 0.2523 |
| m05_joint_grw_smile_supremacy_w040 | [0.00, 0.30) | 111 | 0.2370 | 0.3105 | +0.0735 | 0.2523 |
| m05_joint_grw_smile_supremacy_w070 | [0.00, 0.30) | 111 | 0.2370 | 0.3005 | +0.0634 | 0.2523 |
| m05_joint_grw_supremacy_w040 | [0.00, 0.30) | 111 | 0.2370 | 0.2993 | +0.0622 | 0.2523 |
| m05_joint_grw_baseline | [0.30, 0.40) | 148 | 0.3556 | 0.3768 | +0.0212 | 0.4054 |
| m05_joint_grw_smile_supremacy_w020 | [0.30, 0.40) | 148 | 0.3556 | 0.3893 | +0.0337 | 0.4054 |
| m05_joint_grw_smile_supremacy_w040 | [0.30, 0.40) | 148 | 0.3556 | 0.3859 | +0.0303 | 0.4054 |
| m05_joint_grw_smile_supremacy_w070 | [0.30, 0.40) | 148 | 0.3556 | 0.3819 | +0.0263 | 0.4054 |
| m05_joint_grw_supremacy_w040 | [0.30, 0.40) | 148 | 0.3556 | 0.3833 | +0.0277 | 0.4054 |
| m05_joint_grw_baseline | [0.40, 0.50) | 178 | 0.4482 | 0.4306 | -0.0176 | 0.4944 |
| m05_joint_grw_smile_supremacy_w020 | [0.40, 0.50) | 178 | 0.4482 | 0.4289 | -0.0193 | 0.4944 |
| m05_joint_grw_smile_supremacy_w040 | [0.40, 0.50) | 178 | 0.4482 | 0.4310 | -0.0171 | 0.4944 |
| m05_joint_grw_smile_supremacy_w070 | [0.40, 0.50) | 178 | 0.4482 | 0.4325 | -0.0156 | 0.4944 |
| m05_joint_grw_supremacy_w040 | [0.40, 0.50) | 178 | 0.4482 | 0.4379 | -0.0103 | 0.4944 |
| m05_joint_grw_baseline | [0.50, 0.60) | 107 | 0.5389 | 0.4814 | -0.0575 | 0.4953 |
| m05_joint_grw_smile_supremacy_w020 | [0.50, 0.60) | 107 | 0.5389 | 0.4696 | -0.0693 | 0.4953 |
| m05_joint_grw_smile_supremacy_w040 | [0.50, 0.60) | 107 | 0.5389 | 0.4766 | -0.0623 | 0.4953 |
| m05_joint_grw_smile_supremacy_w070 | [0.50, 0.60) | 107 | 0.5389 | 0.4843 | -0.0546 | 0.4953 |
| m05_joint_grw_supremacy_w040 | [0.50, 0.60) | 107 | 0.5389 | 0.4928 | -0.0461 | 0.4953 |
| m05_joint_grw_baseline | [0.60, 1.00) | 51 | 0.6753 | 0.5546 | -0.1207 | 0.5882 |
| m05_joint_grw_smile_supremacy_w020 | [0.60, 1.00) | 51 | 0.6753 | 0.5522 | -0.1231 | 0.5882 |
| m05_joint_grw_smile_supremacy_w040 | [0.60, 1.00) | 51 | 0.6753 | 0.5686 | -0.1067 | 0.5882 |
| m05_joint_grw_smile_supremacy_w070 | [0.60, 1.00) | 51 | 0.6753 | 0.5843 | -0.0909 | 0.5882 |
| m05_joint_grw_supremacy_w040 | [0.60, 1.00) | 51 | 0.6753 | 0.5822 | -0.0930 | 0.5882 |
