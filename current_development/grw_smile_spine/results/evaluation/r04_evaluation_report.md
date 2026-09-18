# r04 proper scores — Task 016 (1-parameter smile spine)

Generated 2026-09-17 22:03 at `1d9da5e` on mcmc-beast. Panel 710 fixtures (24/25 + 25/26 walk-forward). Book: de-vigged Betfair TWA(−20, 0] close. Baseline reproduction: LogLoss 0.64315 / ECE 0.0123 on 2899 rows (published 0.64315 / 0.0123).

Runs: `m05_joint_grw_baseline` `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`; `m05_joint_grw_supremacy_w040` `0ee58d18-b7e9-4168-8d78-93887b1a8c26`; `m05_joint_grw_smile_supremacy_w020` `fcd5e974-9a46-4a10-9828-6b987a5484d6`; `m05_joint_grw_smile_supremacy_w040` `30620d3e-e4bd-4c05-b1a1-85cefa36b728`; `m05_joint_grw_smile_spine_w020` `eaf53852-a078-4190-b744-089966a306f6`; `m05_joint_grw_smile_spine_w040` `582035c0-e145-44f7-9f40-89e25388e79a`.

`all` pools 1X2 + O/U 2.5 + BTTS (Task 013's published basis). O/U 0.5, 1.5, 3.5 and 4.5 are secondary and never pooled — the spine's line forces φ₀ = 0.900 and φ₄ = 1.111 against the five-strike 0.843 and 1.069, so the ends are where the one-parameter restriction can cost something.

## Scope: all

| model | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w040 | 2899 | 0.64078 | 0.64182 | 0.22488 | 0.22529 | 0.22141 | 0.21110 | 0.0155 | 0.0139 |
| m05_joint_grw_smile_supremacy_w040 | 2899 | 0.64092 | 0.64182 | 0.22494 | 0.22529 | 0.22128 | 0.21110 | 0.0145 | 0.0139 |
| m05_joint_grw_smile_spine_w020 | 2899 | 0.64104 | 0.64182 | 0.22499 | 0.22529 | 0.22184 | 0.21110 | 0.0140 | 0.0139 |
| m05_joint_grw_smile_supremacy_w020 | 2899 | 0.64116 | 0.64182 | 0.22505 | 0.22529 | 0.22164 | 0.21110 | 0.0152 | 0.0139 |
| m05_joint_grw_supremacy_w040 | 2899 | 0.64135 | 0.64182 | 0.22516 | 0.22529 | 0.22221 | 0.21110 | 0.0195 | 0.0139 |
| m05_joint_grw_baseline | 2899 | 0.64315 | 0.64182 | 0.22603 | 0.22529 | 0.22383 | 0.21110 | 0.0123 | 0.0139 |

## Scope: 1X2

| model | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w040 | 1785 | 0.61209 | 0.61312 | 0.21118 | 0.21156 | 0.22128 | 0.21110 | 0.0251 | 0.0186 |
| m05_joint_grw_smile_spine_w040 | 1785 | 0.61213 | 0.61312 | 0.21121 | 0.21156 | 0.22141 | 0.21110 | 0.0264 | 0.0186 |
| m05_joint_grw_smile_supremacy_w020 | 1785 | 0.61245 | 0.61312 | 0.21134 | 0.21156 | 0.22164 | 0.21110 | 0.0243 | 0.0186 |
| m05_joint_grw_smile_spine_w020 | 1785 | 0.61264 | 0.61312 | 0.21144 | 0.21156 | 0.22184 | 0.21110 | 0.0227 | 0.0186 |
| m05_joint_grw_supremacy_w040 | 1785 | 0.61338 | 0.61312 | 0.21181 | 0.21156 | 0.22221 | 0.21110 | 0.0234 | 0.0186 |
| m05_joint_grw_baseline | 1785 | 0.61558 | 0.61312 | 0.21286 | 0.21156 | 0.22383 | 0.21110 | 0.0124 | 0.0186 |

## Scope: OU2.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w040 | 758 | 0.68615 | 0.68988 | 0.24650 | 0.24829 | 0.0162 | 0.0183 |
| m05_joint_grw_smile_spine_w020 | 758 | 0.68621 | 0.68988 | 0.24653 | 0.24829 | 0.0146 | 0.0183 |
| m05_joint_grw_smile_supremacy_w040 | 758 | 0.68698 | 0.68988 | 0.24691 | 0.24829 | 0.0228 | 0.0183 |
| m05_joint_grw_smile_supremacy_w020 | 758 | 0.68705 | 0.68988 | 0.24694 | 0.24829 | 0.0179 | 0.0183 |
| m05_joint_grw_supremacy_w040 | 758 | 0.68796 | 0.68988 | 0.24743 | 0.24829 | 0.0278 | 0.0183 |
| m05_joint_grw_baseline | 758 | 0.68986 | 0.68988 | 0.24839 | 0.24829 | 0.0311 | 0.0183 |

## Scope: BTTS

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 356 | 0.68192 | 0.68337 | 0.24445 | 0.24516 | 0.0366 | 0.0300 |
| m05_joint_grw_supremacy_w040 | 356 | 0.68235 | 0.68337 | 0.24467 | 0.24516 | 0.0042 | 0.0300 |
| m05_joint_grw_smile_spine_w020 | 356 | 0.68721 | 0.68337 | 0.24705 | 0.24516 | 0.0263 | 0.0300 |
| m05_joint_grw_smile_supremacy_w040 | 356 | 0.68741 | 0.68337 | 0.24714 | 0.24516 | 0.0095 | 0.0300 |
| m05_joint_grw_smile_supremacy_w020 | 356 | 0.68742 | 0.68337 | 0.24715 | 0.24516 | 0.0169 | 0.0300 |
| m05_joint_grw_smile_spine_w040 | 356 | 0.68778 | 0.68337 | 0.24733 | 0.24516 | 0.0244 | 0.0300 |

## Scope: OU0.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 533 | 0.20844 | 1.31832 | 0.05126 | 0.05434 | 0.0147 | 0.0364 |
| m05_joint_grw_supremacy_w040 | 533 | 0.20940 | 1.31832 | 0.05139 | 0.05434 | 0.0156 | 0.0364 |
| m05_joint_grw_smile_spine_w040 | 533 | 0.21805 | 1.31832 | 0.05245 | 0.05434 | 0.0342 | 0.0364 |
| m05_joint_grw_smile_spine_w020 | 533 | 0.21808 | 1.31832 | 0.05244 | 0.05434 | 0.0434 | 0.0364 |
| m05_joint_grw_smile_supremacy_w020 | 533 | 0.22267 | 1.31832 | 0.05321 | 0.05434 | 0.0440 | 0.0364 |
| m05_joint_grw_smile_supremacy_w040 | 533 | 0.22283 | 1.31832 | 0.05326 | 0.05434 | 0.0442 | 0.0364 |

## Scope: OU1.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 430 | 0.52070 | 0.52728 | 0.17025 | 0.17266 | 0.0322 | 0.0103 |
| m05_joint_grw_supremacy_w040 | 430 | 0.52285 | 0.52728 | 0.17122 | 0.17266 | 0.0234 | 0.0103 |
| m05_joint_grw_smile_supremacy_w040 | 430 | 0.52765 | 0.52728 | 0.17286 | 0.17266 | 0.0202 | 0.0103 |
| m05_joint_grw_smile_supremacy_w020 | 430 | 0.52784 | 0.52728 | 0.17289 | 0.17266 | 0.0199 | 0.0103 |
| m05_joint_grw_smile_spine_w040 | 430 | 0.53183 | 0.52728 | 0.17427 | 0.17266 | 0.0492 | 0.0103 |
| m05_joint_grw_smile_spine_w020 | 430 | 0.53207 | 0.52728 | 0.17432 | 0.17266 | 0.0436 | 0.0103 |

## Scope: OU3.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_supremacy_w040 | 528 | 0.60897 | 0.61046 | 0.20927 | 0.20987 | 0.0201 | 0.0148 |
| m05_joint_grw_smile_spine_w040 | 528 | 0.60936 | 0.61046 | 0.20944 | 0.20987 | 0.0238 | 0.0148 |
| m05_joint_grw_smile_supremacy_w020 | 528 | 0.60948 | 0.61046 | 0.20953 | 0.20987 | 0.0203 | 0.0148 |
| m05_joint_grw_smile_spine_w020 | 528 | 0.61003 | 0.61046 | 0.20977 | 0.20987 | 0.0242 | 0.0148 |
| m05_joint_grw_supremacy_w040 | 528 | 0.61131 | 0.61046 | 0.21016 | 0.20987 | 0.0306 | 0.0148 |
| m05_joint_grw_baseline | 528 | 0.61235 | 0.61046 | 0.21054 | 0.20987 | 0.0465 | 0.0148 |

## Scope: OU4.5

| model | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 197 | 0.34576 | 0.50791 | 0.09450 | 0.09594 | 0.0850 | 0.0963 |
| m05_joint_grw_smile_supremacy_w020 | 197 | 0.34713 | 0.50791 | 0.09534 | 0.09594 | 0.0889 | 0.0963 |
| m05_joint_grw_supremacy_w040 | 197 | 0.34731 | 0.50791 | 0.09486 | 0.09594 | 0.0838 | 0.0963 |
| m05_joint_grw_smile_supremacy_w040 | 197 | 0.34763 | 0.50791 | 0.09552 | 0.09594 | 0.0891 | 0.0963 |
| m05_joint_grw_smile_spine_w020 | 197 | 0.35284 | 0.50791 | 0.09714 | 0.09594 | 0.0981 | 0.0963 |
| m05_joint_grw_smile_spine_w040 | 197 | 0.35317 | 0.50791 | 0.09727 | 0.09594 | 0.0981 | 0.0963 |

## Strike ladder — UNDER selection, per O/U line

`mean_gap` is mean `p_model − p_market`; `realised_under_rate` is the outcome frequency on the same rows.

| model | strike | K | n_obs | logloss | market_logloss | delta_logloss | mean_p_model | mean_p_market | mean_gap | realised_under_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | 0.5000 | 0 | 149 | 0.15797 | 0.15574 | +0.00223 | 0.0718 | 0.0679 | +0.0040 | 0.0336 |
| m05_joint_grw_smile_spine_w020 | 0.5000 | 0 | 149 | 0.17215 | 0.15574 | +0.01641 | 0.0893 | 0.0679 | +0.0214 | 0.0336 |
| m05_joint_grw_smile_spine_w040 | 0.5000 | 0 | 149 | 0.17191 | 0.15574 | +0.01617 | 0.0894 | 0.0679 | +0.0215 | 0.0336 |
| m05_joint_grw_smile_supremacy_w020 | 0.5000 | 0 | 149 | 0.17912 | 0.15574 | +0.02338 | 0.0994 | 0.0679 | +0.0315 | 0.0336 |
| m05_joint_grw_smile_supremacy_w040 | 0.5000 | 0 | 149 | 0.17905 | 0.15574 | +0.02331 | 0.0995 | 0.0679 | +0.0316 | 0.0336 |
| m05_joint_grw_supremacy_w040 | 0.5000 | 0 | 149 | 0.15938 | 0.15574 | +0.00364 | 0.0729 | 0.0679 | +0.0050 | 0.0336 |
| m05_joint_grw_baseline | 1.5000 | 1 | 215 | 0.52070 | 0.52728 | -0.00658 | 0.2423 | 0.2305 | +0.0118 | 0.2279 |
| m05_joint_grw_smile_spine_w020 | 1.5000 | 1 | 215 | 0.53207 | 0.52728 | +0.00479 | 0.2715 | 0.2305 | +0.0411 | 0.2279 |
| m05_joint_grw_smile_spine_w040 | 1.5000 | 1 | 215 | 0.53183 | 0.52728 | +0.00455 | 0.2717 | 0.2305 | +0.0412 | 0.2279 |
| m05_joint_grw_smile_supremacy_w020 | 1.5000 | 1 | 215 | 0.52784 | 0.52728 | +0.00056 | 0.2478 | 0.2305 | +0.0174 | 0.2279 |
| m05_joint_grw_smile_supremacy_w040 | 1.5000 | 1 | 215 | 0.52765 | 0.52728 | +0.00037 | 0.2482 | 0.2305 | +0.0177 | 0.2279 |
| m05_joint_grw_supremacy_w040 | 1.5000 | 1 | 215 | 0.52285 | 0.52728 | -0.00443 | 0.2444 | 0.2305 | +0.0140 | 0.2279 |
| m05_joint_grw_baseline | 2.5000 | 2 | 379 | 0.68986 | 0.68988 | -0.00002 | 0.4784 | 0.4702 | +0.0082 | 0.4987 |
| m05_joint_grw_smile_spine_w020 | 2.5000 | 2 | 379 | 0.68621 | 0.68988 | -0.00367 | 0.4917 | 0.4702 | +0.0215 | 0.4987 |
| m05_joint_grw_smile_spine_w040 | 2.5000 | 2 | 379 | 0.68615 | 0.68988 | -0.00373 | 0.4922 | 0.4702 | +0.0220 | 0.4987 |
| m05_joint_grw_smile_supremacy_w020 | 2.5000 | 2 | 379 | 0.68705 | 0.68988 | -0.00283 | 0.4780 | 0.4702 | +0.0078 | 0.4987 |
| m05_joint_grw_smile_supremacy_w040 | 2.5000 | 2 | 379 | 0.68698 | 0.68988 | -0.00290 | 0.4784 | 0.4702 | +0.0082 | 0.4987 |
| m05_joint_grw_supremacy_w040 | 2.5000 | 2 | 379 | 0.68796 | 0.68988 | -0.00192 | 0.4823 | 0.4702 | +0.0121 | 0.4987 |
| m05_joint_grw_baseline | 3.5000 | 3 | 264 | 0.61235 | 0.61046 | +0.00189 | 0.6841 | 0.6859 | -0.0018 | 0.6932 |
| m05_joint_grw_smile_spine_w020 | 3.5000 | 3 | 264 | 0.61003 | 0.61046 | -0.00043 | 0.6733 | 0.6859 | -0.0126 | 0.6932 |
| m05_joint_grw_smile_spine_w040 | 3.5000 | 3 | 264 | 0.60936 | 0.61046 | -0.00110 | 0.6736 | 0.6859 | -0.0123 | 0.6932 |
| m05_joint_grw_smile_supremacy_w020 | 3.5000 | 3 | 264 | 0.60948 | 0.61046 | -0.00098 | 0.6777 | 0.6859 | -0.0082 | 0.6932 |
| m05_joint_grw_smile_supremacy_w040 | 3.5000 | 3 | 264 | 0.60897 | 0.61046 | -0.00149 | 0.6778 | 0.6859 | -0.0081 | 0.6932 |
| m05_joint_grw_supremacy_w040 | 3.5000 | 3 | 264 | 0.61131 | 0.61046 | +0.00085 | 0.6887 | 0.6859 | +0.0028 | 0.6932 |
| m05_joint_grw_baseline | 4.5000 | 4 | 104 | 0.34383 | 0.64710 | -0.30327 | 0.8397 | 0.8578 | -0.0180 | 0.9038 |
| m05_joint_grw_smile_spine_w020 | 4.5000 | 4 | 104 | 0.35156 | 0.64710 | -0.29554 | 0.8077 | 0.8578 | -0.0501 | 0.9038 |
| m05_joint_grw_smile_spine_w040 | 4.5000 | 4 | 104 | 0.35176 | 0.64710 | -0.29535 | 0.8078 | 0.8578 | -0.0500 | 0.9038 |
| m05_joint_grw_smile_supremacy_w020 | 4.5000 | 4 | 104 | 0.34597 | 0.64710 | -0.30114 | 0.8168 | 0.8578 | -0.0409 | 0.9038 |
| m05_joint_grw_smile_supremacy_w040 | 4.5000 | 4 | 104 | 0.34626 | 0.64710 | -0.30085 | 0.8168 | 0.8578 | -0.0410 | 0.9038 |
| m05_joint_grw_supremacy_w040 | 4.5000 | 4 | 104 | 0.34560 | 0.64710 | -0.30150 | 0.8413 | 0.8578 | -0.0165 | 0.9038 |

## Paired Δ (candidate − reference), fixture-clustered bootstrap B = 10000

Negative Δ favours the candidate; 95% percentile interval. `width` is hi − lo — a Δ near 0 with a wide interval is an unresolved test, not parity.

### LogLoss

| contrast | scope | n_obs | n_fixtures | delta | lo | hi | width | p_better | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00211 | -0.00692 | +0.00273 | 0.00965 | 0.8025 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00294 | -0.00723 | +0.00130 | 0.00853 | 0.9106 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00365 | -0.01605 | +0.00869 | 0.02474 | 0.7193 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00529 | -0.00829 | +0.01854 | 0.02683 | 0.2241 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU0.5 | 533 | 384 | +0.00963 | +0.00297 | +0.01552 | 0.01255 | 0.0028 | true |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.01136 | -0.00414 | +0.02578 | 0.02992 | 0.0741 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00232 | -0.01916 | +0.01309 | 0.03225 | 0.5981 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU4.5 | 197 | 104 | +0.00708 | -0.02443 | +0.03181 | 0.05624 | 0.2867 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00237 | -0.00736 | +0.00262 | 0.00998 | 0.8268 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00345 | -0.00787 | +0.00102 | 0.00888 | 0.9342 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00371 | -0.01663 | +0.00921 | 0.02584 | 0.7170 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00586 | -0.00761 | +0.01910 | 0.02671 | 0.2012 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU0.5 | 533 | 384 | +0.00960 | +0.00262 | +0.01575 | 0.01313 | 0.0038 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.01113 | -0.00474 | +0.02594 | 0.03067 | 0.0831 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00299 | -0.02038 | +0.01284 | 0.03322 | 0.6247 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU4.5 | 197 | 104 | +0.00741 | -0.02397 | +0.03199 | 0.05596 | 0.2771 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | all | 2899 | 627 | -0.00012 | -0.00110 | +0.00084 | 0.00194 | 0.5987 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | 1X2 | 1785 | 595 | +0.00020 | -0.00019 | +0.00060 | 0.00079 | 0.1564 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU2.5 | 758 | 379 | -0.00083 | -0.00364 | +0.00206 | 0.00570 | 0.7165 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | BTTS | 356 | 178 | -0.00020 | -0.00358 | +0.00304 | 0.00662 | 0.5538 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU0.5 | 533 | 384 | -0.00460 | -0.00714 | -0.00165 | 0.00549 | 0.9982 | true |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU1.5 | 430 | 215 | +0.00423 | -0.00264 | +0.01080 | 0.01344 | 0.1194 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU3.5 | 528 | 264 | +0.00055 | -0.00076 | +0.00183 | 0.00259 | 0.1992 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU4.5 | 197 | 104 | +0.00571 | +0.00201 | +0.00892 | 0.00691 | 0.0019 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | all | 2899 | 627 | -0.00014 | -0.00112 | +0.00082 | 0.00194 | 0.6163 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | 595 | +0.00005 | -0.00041 | +0.00052 | 0.00093 | 0.4157 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | 379 | -0.00083 | -0.00362 | +0.00209 | 0.00572 | 0.7144 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | 178 | +0.00038 | -0.00285 | +0.00351 | 0.00636 | 0.4206 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU0.5 | 533 | 384 | -0.00478 | -0.00727 | -0.00192 | 0.00535 | 0.9989 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | 215 | +0.00419 | -0.00268 | +0.01074 | 0.01343 | 0.1204 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | 264 | +0.00039 | -0.00088 | +0.00160 | 0.00248 | 0.2708 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU4.5 | 197 | 104 | +0.00554 | +0.00180 | +0.00877 | 0.00697 | 0.0028 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | all | 2899 | 627 | -0.00057 | -0.00511 | +0.00389 | 0.00900 | 0.6015 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | 1X2 | 1785 | 595 | -0.00124 | -0.00372 | +0.00123 | 0.00495 | 0.8397 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU2.5 | 758 | 379 | -0.00181 | -0.01466 | +0.01077 | 0.02543 | 0.6068 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | BTTS | 356 | 178 | +0.00544 | -0.00895 | +0.01941 | 0.02836 | 0.2299 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU0.5 | 533 | 384 | +0.00864 | +0.00148 | +0.01491 | 0.01343 | 0.0100 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU1.5 | 430 | 215 | +0.00898 | -0.00672 | +0.02374 | 0.03047 | 0.1259 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU3.5 | 528 | 264 | -0.00195 | -0.01973 | +0.01443 | 0.03416 | 0.5789 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU4.5 | 197 | 104 | +0.00586 | -0.02672 | +0.03145 | 0.05817 | 0.3190 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | all | 2899 | 627 | +0.00026 | -0.00062 | +0.00115 | 0.00177 | 0.2765 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | 1X2 | 1785 | 595 | +0.00051 | -0.00079 | +0.00179 | 0.00259 | 0.2163 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU2.5 | 758 | 379 | +0.00006 | -0.00117 | +0.00133 | 0.00250 | 0.4614 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | BTTS | 356 | 178 | -0.00057 | -0.00229 | +0.00114 | 0.00343 | 0.7459 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU0.5 | 533 | 384 | +0.00003 | -0.00052 | +0.00062 | 0.00114 | 0.4722 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU1.5 | 430 | 215 | +0.00023 | -0.00109 | +0.00153 | 0.00262 | 0.3553 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU3.5 | 528 | 264 | +0.00067 | -0.00101 | +0.00234 | 0.00335 | 0.2167 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU4.5 | 197 | 104 | -0.00033 | -0.00202 | +0.00122 | 0.00324 | 0.6515 | false |

### Brier

| contrast | scope | n_obs | n_fixtures | delta | lo | hi | width | p_better | significant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00104 | -0.00333 | +0.00127 | 0.00460 | 0.8098 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00142 | -0.00333 | +0.00049 | 0.00382 | 0.9277 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00186 | -0.00789 | +0.00417 | 0.01206 | 0.7284 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00260 | -0.00400 | +0.00906 | 0.01307 | 0.2209 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU0.5 | 533 | 384 | +0.00118 | +0.00026 | +0.00199 | 0.00173 | 0.0069 | true |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00407 | -0.00149 | +0.00928 | 0.01077 | 0.0738 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00078 | -0.00758 | +0.00555 | 0.01313 | 0.5791 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_baseline | OU4.5 | 197 | 104 | +0.00264 | -0.00382 | +0.00822 | 0.01203 | 0.1863 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | all | 2899 | 627 | -0.00115 | -0.00351 | +0.00121 | 0.00472 | 0.8304 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | 1X2 | 1785 | 595 | -0.00164 | -0.00362 | +0.00036 | 0.00399 | 0.9471 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU2.5 | 758 | 379 | -0.00189 | -0.00819 | +0.00439 | 0.01259 | 0.7244 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | BTTS | 356 | 178 | +0.00288 | -0.00367 | +0.00935 | 0.01302 | 0.1997 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU0.5 | 533 | 384 | +0.00119 | +0.00022 | +0.00204 | 0.00182 | 0.0087 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU1.5 | 430 | 215 | +0.00401 | -0.00168 | +0.00941 | 0.01110 | 0.0821 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU3.5 | 528 | 264 | -0.00110 | -0.00817 | +0.00546 | 0.01363 | 0.6119 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_baseline | OU4.5 | 197 | 104 | +0.00277 | -0.00364 | +0.00841 | 0.01206 | 0.1754 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | all | 2899 | 627 | -0.00006 | -0.00054 | +0.00041 | 0.00095 | 0.6001 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | 1X2 | 1785 | 595 | +0.00010 | -0.00007 | +0.00027 | 0.00033 | 0.1220 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU2.5 | 758 | 379 | -0.00041 | -0.00180 | +0.00102 | 0.00282 | 0.7166 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | BTTS | 356 | 178 | -0.00010 | -0.00177 | +0.00150 | 0.00326 | 0.5524 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU0.5 | 533 | 384 | -0.00078 | -0.00121 | -0.00027 | 0.00094 | 0.9979 | true |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU1.5 | 430 | 215 | +0.00143 | -0.00124 | +0.00398 | 0.00522 | 0.1527 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU3.5 | 528 | 264 | +0.00024 | -0.00033 | +0.00079 | 0.00113 | 0.2017 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_supremacy_w020 | OU4.5 | 197 | 104 | +0.00180 | +0.00063 | +0.00282 | 0.00218 | 0.0017 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | all | 2899 | 627 | -0.00007 | -0.00054 | +0.00040 | 0.00095 | 0.6121 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | 595 | +0.00003 | -0.00017 | +0.00023 | 0.00040 | 0.3711 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | 379 | -0.00041 | -0.00180 | +0.00104 | 0.00284 | 0.7150 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | 178 | +0.00019 | -0.00140 | +0.00174 | 0.00314 | 0.4188 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU0.5 | 533 | 384 | -0.00081 | -0.00124 | -0.00031 | 0.00092 | 0.9989 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | 215 | +0.00141 | -0.00126 | +0.00397 | 0.00524 | 0.1525 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | 264 | +0.00017 | -0.00037 | +0.00070 | 0.00107 | 0.2635 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_smile_supremacy_w040 | OU4.5 | 197 | 104 | +0.00175 | +0.00058 | +0.00276 | 0.00218 | 0.0025 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | all | 2899 | 627 | -0.00028 | -0.00246 | +0.00186 | 0.00432 | 0.6036 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | 1X2 | 1785 | 595 | -0.00060 | -0.00167 | +0.00048 | 0.00215 | 0.8645 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU2.5 | 758 | 379 | -0.00093 | -0.00715 | +0.00521 | 0.01236 | 0.6130 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | BTTS | 356 | 178 | +0.00266 | -0.00435 | +0.00948 | 0.01383 | 0.2286 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU0.5 | 533 | 384 | +0.00106 | +0.00009 | +0.00192 | 0.00183 | 0.0169 | true |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU1.5 | 430 | 215 | +0.00305 | -0.00261 | +0.00842 | 0.01103 | 0.1418 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU3.5 | 528 | 264 | -0.00072 | -0.00789 | +0.00603 | 0.01392 | 0.5716 | false |
| m05_joint_grw_smile_spine_w040 − m05_joint_grw_supremacy_w040 | OU4.5 | 197 | 104 | +0.00241 | -0.00434 | +0.00825 | 0.01260 | 0.2155 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | all | 2899 | 627 | +0.00011 | -0.00030 | +0.00052 | 0.00082 | 0.2919 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | 1X2 | 1785 | 595 | +0.00022 | -0.00036 | +0.00080 | 0.00116 | 0.2193 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU2.5 | 758 | 379 | +0.00003 | -0.00058 | +0.00066 | 0.00124 | 0.4694 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | BTTS | 356 | 178 | -0.00028 | -0.00114 | +0.00056 | 0.00170 | 0.7490 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU0.5 | 533 | 384 | -0.00001 | -0.00009 | +0.00008 | 0.00018 | 0.5933 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU1.5 | 430 | 215 | +0.00005 | -0.00048 | +0.00057 | 0.00104 | 0.4182 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU3.5 | 528 | 264 | +0.00032 | -0.00043 | +0.00107 | 0.00150 | 0.1971 | false |
| m05_joint_grw_smile_spine_w020 − m05_joint_grw_smile_spine_w040 | OU4.5 | 197 | 104 | -0.00013 | -0.00069 | +0.00037 | 0.00106 | 0.6873 | false |

## Every arm against the Betfair close (ΔLogLoss)

| model | scope | n_obs | delta | lo | hi | p_negative | significant |
|---|---|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | all | 2899 | +0.00134 | -0.00473 | +0.00744 | 0.3367 | false |
| m05_joint_grw_baseline | 1X2 | 1785 | +0.00247 | -0.00590 | +0.01053 | 0.2760 | false |
| m05_joint_grw_baseline | OU2.5 | 758 | -0.00002 | -0.01302 | +0.01320 | 0.4992 | false |
| m05_joint_grw_baseline | BTTS | 356 | -0.00145 | -0.01375 | +0.01109 | 0.5907 | false |
| m05_joint_grw_baseline | OU0.5 | 533 | -1.10987 | -1.65487 | -0.62887 | 1.0000 | true |
| m05_joint_grw_baseline | OU1.5 | 430 | -0.00658 | -0.02273 | +0.00846 | 0.8011 | false |
| m05_joint_grw_baseline | OU3.5 | 528 | +0.00189 | -0.01502 | +0.01903 | 0.4240 | false |
| m05_joint_grw_baseline | OU4.5 | 197 | -0.16214 | -0.50861 | +0.01596 | 0.7822 | false |
| m05_joint_grw_supremacy_w040 | all | 2899 | -0.00047 | -0.00608 | +0.00513 | 0.5635 | false |
| m05_joint_grw_supremacy_w040 | 1X2 | 1785 | +0.00026 | -0.00699 | +0.00728 | 0.4602 | false |
| m05_joint_grw_supremacy_w040 | OU2.5 | 758 | -0.00192 | -0.01512 | +0.01156 | 0.6096 | false |
| m05_joint_grw_supremacy_w040 | BTTS | 356 | -0.00102 | -0.01397 | +0.01208 | 0.5616 | false |
| m05_joint_grw_supremacy_w040 | OU0.5 | 533 | -1.10891 | -1.65400 | -0.62742 | 1.0000 | true |
| m05_joint_grw_supremacy_w040 | OU1.5 | 430 | -0.00443 | -0.02097 | +0.01076 | 0.7046 | false |
| m05_joint_grw_supremacy_w040 | OU3.5 | 528 | +0.00085 | -0.01616 | +0.01838 | 0.4747 | false |
| m05_joint_grw_supremacy_w040 | OU4.5 | 197 | -0.16060 | -0.50567 | +0.01624 | 0.7635 | false |
| m05_joint_grw_smile_supremacy_w020 | all | 2899 | -0.00066 | -0.00610 | +0.00469 | 0.5813 | false |
| m05_joint_grw_smile_supremacy_w020 | 1X2 | 1785 | -0.00067 | -0.00907 | +0.00749 | 0.5569 | false |
| m05_joint_grw_smile_supremacy_w020 | OU2.5 | 758 | -0.00283 | -0.01073 | +0.00503 | 0.7546 | false |
| m05_joint_grw_smile_supremacy_w020 | BTTS | 356 | +0.00405 | -0.00410 | +0.01208 | 0.1611 | false |
| m05_joint_grw_smile_supremacy_w020 | OU0.5 | 533 | -1.09564 | -1.64666 | -0.60857 | 1.0000 | true |
| m05_joint_grw_smile_supremacy_w020 | OU1.5 | 430 | +0.00056 | -0.01027 | +0.01070 | 0.4500 | false |
| m05_joint_grw_smile_supremacy_w020 | OU3.5 | 528 | -0.00098 | -0.00998 | +0.00771 | 0.5844 | false |
| m05_joint_grw_smile_supremacy_w020 | OU4.5 | 197 | -0.16078 | -0.51312 | +0.02260 | 0.7635 | false |
| m05_joint_grw_smile_supremacy_w040 | all | 2899 | -0.00090 | -0.00608 | +0.00417 | 0.6244 | false |
| m05_joint_grw_smile_supremacy_w040 | 1X2 | 1785 | -0.00103 | -0.00888 | +0.00659 | 0.5960 | false |
| m05_joint_grw_smile_supremacy_w040 | OU2.5 | 758 | -0.00290 | -0.01071 | +0.00479 | 0.7608 | false |
| m05_joint_grw_smile_supremacy_w040 | BTTS | 356 | +0.00404 | -0.00385 | +0.01180 | 0.1538 | false |
| m05_joint_grw_smile_supremacy_w040 | OU0.5 | 533 | -1.09549 | -1.64630 | -0.60843 | 1.0000 | true |
| m05_joint_grw_smile_supremacy_w040 | OU1.5 | 430 | +0.00037 | -0.01036 | +0.01024 | 0.4616 | false |
| m05_joint_grw_smile_supremacy_w040 | OU3.5 | 528 | -0.00149 | -0.01014 | +0.00690 | 0.6312 | false |
| m05_joint_grw_smile_supremacy_w040 | OU4.5 | 197 | -0.16028 | -0.51232 | +0.02263 | 0.7568 | false |
| m05_joint_grw_smile_spine_w020 | all | 2899 | -0.00078 | -0.00630 | +0.00455 | 0.6007 | false |
| m05_joint_grw_smile_spine_w020 | 1X2 | 1785 | -0.00047 | -0.00896 | +0.00775 | 0.5377 | false |
| m05_joint_grw_smile_spine_w020 | OU2.5 | 758 | -0.00367 | -0.01253 | +0.00529 | 0.7832 | false |
| m05_joint_grw_smile_spine_w020 | BTTS | 356 | +0.00384 | -0.00554 | +0.01300 | 0.2062 | false |
| m05_joint_grw_smile_spine_w020 | OU0.5 | 533 | -1.10024 | -1.64985 | -0.61561 | 1.0000 | true |
| m05_joint_grw_smile_spine_w020 | OU1.5 | 430 | +0.00479 | -0.01102 | +0.01933 | 0.2712 | false |
| m05_joint_grw_smile_spine_w020 | OU3.5 | 528 | -0.00043 | -0.01009 | +0.00873 | 0.5339 | false |
| m05_joint_grw_smile_spine_w020 | OU4.5 | 197 | -0.15507 | -0.50948 | +0.03086 | 0.7182 | false |
| m05_joint_grw_smile_spine_w040 | all | 2899 | -0.00104 | -0.00639 | +0.00407 | 0.6439 | false |
| m05_joint_grw_smile_spine_w040 | 1X2 | 1785 | -0.00098 | -0.00892 | +0.00673 | 0.5897 | false |
| m05_joint_grw_smile_spine_w040 | OU2.5 | 758 | -0.00373 | -0.01252 | +0.00506 | 0.7906 | false |
| m05_joint_grw_smile_spine_w040 | BTTS | 356 | +0.00441 | -0.00475 | +0.01349 | 0.1704 | false |
| m05_joint_grw_smile_spine_w040 | OU0.5 | 533 | -1.10027 | -1.64964 | -0.61547 | 1.0000 | true |
| m05_joint_grw_smile_spine_w040 | OU1.5 | 430 | +0.00455 | -0.01125 | +0.01902 | 0.2795 | false |
| m05_joint_grw_smile_spine_w040 | OU3.5 | 528 | -0.00110 | -0.01023 | +0.00768 | 0.5949 | false |
| m05_joint_grw_smile_spine_w040 | OU4.5 | 197 | -0.15474 | -0.50892 | +0.03070 | 0.7151 | false |

## Home-favourite compression (1X2 home selection)

| model | market_bin | n | mean_p_market | mean_p_model | mean_gap | home_win_rate |
|---|---|---:|---:|---:|---:|---:|
| m05_joint_grw_baseline | [0.00, 0.30) | 111 | 0.2370 | 0.3022 | +0.0652 | 0.2523 |
| m05_joint_grw_smile_spine_w020 | [0.00, 0.30) | 111 | 0.2370 | 0.3210 | +0.0840 | 0.2523 |
| m05_joint_grw_smile_spine_w040 | [0.00, 0.30) | 111 | 0.2370 | 0.3113 | +0.0743 | 0.2523 |
| m05_joint_grw_smile_supremacy_w020 | [0.00, 0.30) | 111 | 0.2370 | 0.3211 | +0.0840 | 0.2523 |
| m05_joint_grw_smile_supremacy_w040 | [0.00, 0.30) | 111 | 0.2370 | 0.3105 | +0.0735 | 0.2523 |
| m05_joint_grw_supremacy_w040 | [0.00, 0.30) | 111 | 0.2370 | 0.2993 | +0.0622 | 0.2523 |
| m05_joint_grw_baseline | [0.30, 0.40) | 148 | 0.3556 | 0.3768 | +0.0212 | 0.4054 |
| m05_joint_grw_smile_spine_w020 | [0.30, 0.40) | 148 | 0.3556 | 0.3882 | +0.0326 | 0.4054 |
| m05_joint_grw_smile_spine_w040 | [0.30, 0.40) | 148 | 0.3556 | 0.3851 | +0.0295 | 0.4054 |
| m05_joint_grw_smile_supremacy_w020 | [0.30, 0.40) | 148 | 0.3556 | 0.3893 | +0.0337 | 0.4054 |
| m05_joint_grw_smile_supremacy_w040 | [0.30, 0.40) | 148 | 0.3556 | 0.3859 | +0.0303 | 0.4054 |
| m05_joint_grw_supremacy_w040 | [0.30, 0.40) | 148 | 0.3556 | 0.3833 | +0.0277 | 0.4054 |
| m05_joint_grw_baseline | [0.40, 0.50) | 178 | 0.4482 | 0.4306 | -0.0176 | 0.4944 |
| m05_joint_grw_smile_spine_w020 | [0.40, 0.50) | 178 | 0.4482 | 0.4272 | -0.0210 | 0.4944 |
| m05_joint_grw_smile_spine_w040 | [0.40, 0.50) | 178 | 0.4482 | 0.4295 | -0.0187 | 0.4944 |
| m05_joint_grw_smile_supremacy_w020 | [0.40, 0.50) | 178 | 0.4482 | 0.4289 | -0.0193 | 0.4944 |
| m05_joint_grw_smile_supremacy_w040 | [0.40, 0.50) | 178 | 0.4482 | 0.4310 | -0.0171 | 0.4944 |
| m05_joint_grw_supremacy_w040 | [0.40, 0.50) | 178 | 0.4482 | 0.4379 | -0.0103 | 0.4944 |
| m05_joint_grw_baseline | [0.50, 0.60) | 107 | 0.5389 | 0.4814 | -0.0575 | 0.4953 |
| m05_joint_grw_smile_spine_w020 | [0.50, 0.60) | 107 | 0.5389 | 0.4675 | -0.0714 | 0.4953 |
| m05_joint_grw_smile_spine_w040 | [0.50, 0.60) | 107 | 0.5389 | 0.4744 | -0.0645 | 0.4953 |
| m05_joint_grw_smile_supremacy_w020 | [0.50, 0.60) | 107 | 0.5389 | 0.4696 | -0.0693 | 0.4953 |
| m05_joint_grw_smile_supremacy_w040 | [0.50, 0.60) | 107 | 0.5389 | 0.4766 | -0.0623 | 0.4953 |
| m05_joint_grw_supremacy_w040 | [0.50, 0.60) | 107 | 0.5389 | 0.4928 | -0.0461 | 0.4953 |
| m05_joint_grw_baseline | [0.60, 1.00) | 51 | 0.6753 | 0.5546 | -0.1207 | 0.5882 |
| m05_joint_grw_smile_spine_w020 | [0.60, 1.00) | 51 | 0.6753 | 0.5490 | -0.1263 | 0.5882 |
| m05_joint_grw_smile_spine_w040 | [0.60, 1.00) | 51 | 0.6753 | 0.5639 | -0.1113 | 0.5882 |
| m05_joint_grw_smile_supremacy_w020 | [0.60, 1.00) | 51 | 0.6753 | 0.5522 | -0.1231 | 0.5882 |
| m05_joint_grw_smile_supremacy_w040 | [0.60, 1.00) | 51 | 0.6753 | 0.5686 | -0.1067 | 0.5882 |
| m05_joint_grw_supremacy_w040 | [0.60, 1.00) | 51 | 0.6753 | 0.5822 | -0.0930 | 0.5882 |
