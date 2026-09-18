# r04 proper scores — Task 008 Phase 1

Generated 2026-09-13 00:20. Panel: 710 fixtures. Book: de-vigged Betfair TWA(−20, 0] close. Control reproduction: `m12_hybrid_td_raw` LogLoss 0.64337 / ECE 0.0100 vs published 0.64337 / 0.01. T003 unmapped-home fixtures: 3.

## Scope: all

| model | role | dynamics | n_obs | logloss | market_logloss | brier | market_brier | rps | ece | market_ece |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | candidate | TimeDecay(180) | 2899 | 0.64272 | 0.64182 | 0.22574 | 0.22529 | 0.22416 | 0.0114 | 0.0139 |
| m05_joint_td_raw | control | TimeDecay(180) | 2899 | 0.64299 | 0.64182 | 0.22586 | 0.22529 | 0.22415 | 0.0149 | 0.0139 |
| m12_joint_hybrid_synergy_hier_ha | candidate | TimeDecay(180) | 2899 | 0.64333 | 0.64182 | 0.22603 | 0.22529 | 0.22454 | 0.0068 | 0.0139 |
| m12_hybrid_td_raw | control | TimeDecay(180) | 2899 | 0.64337 | 0.64182 | 0.22605 | 0.22529 | 0.22447 | 0.0100 | 0.0139 |
| m12_grw_raw | control | MultiScaleGRW | 2899 | 0.64437 | 0.64182 | 0.22659 | 0.22529 | 0.22493 | 0.0086 | 0.0139 |
| m12_joint_hybrid_synergy_grw_hier_ha | candidate | MultiScaleGRW | 2899 | 0.64502 | 0.64182 | 0.22689 | 0.22529 | 0.22514 | 0.0111 | 0.0139 |

## Scope: 1X2

| model | role | dynamics | n_obs | logloss | market_logloss | brier | market_brier | rps | ece | market_ece |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | candidate | TimeDecay(180) | 1785 | 0.61605 | 0.61312 | 0.21298 | 0.21156 | 0.22416 | 0.0184 | 0.0186 |
| m05_joint_td_raw | control | TimeDecay(180) | 1785 | 0.61611 | 0.61312 | 0.21299 | 0.21156 | 0.22415 | 0.0216 | 0.0186 |
| m12_hybrid_td_raw | control | TimeDecay(180) | 1785 | 0.61636 | 0.61312 | 0.21311 | 0.21156 | 0.22447 | 0.0155 | 0.0186 |
| m12_joint_hybrid_synergy_hier_ha | candidate | TimeDecay(180) | 1785 | 0.61641 | 0.61312 | 0.21315 | 0.21156 | 0.22454 | 0.0107 | 0.0186 |
| m12_grw_raw | control | MultiScaleGRW | 1785 | 0.61701 | 0.61312 | 0.21350 | 0.21156 | 0.22493 | 0.0140 | 0.0186 |
| m12_joint_hybrid_synergy_grw_hier_ha | candidate | MultiScaleGRW | 1785 | 0.61739 | 0.61312 | 0.21368 | 0.21156 | 0.22514 | 0.0081 | 0.0186 |

## Scope: OU2.5

| model | role | dynamics | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | candidate | TimeDecay(180) | 758 | 0.68626 | 0.68988 | 0.24658 | 0.24829 | 0.0087 | 0.0183 |
| m05_joint_td_raw | control | TimeDecay(180) | 758 | 0.68700 | 0.68988 | 0.24695 | 0.24829 | 0.0100 | 0.0183 |
| m12_joint_hybrid_synergy_hier_ha | candidate | TimeDecay(180) | 758 | 0.68706 | 0.68988 | 0.24697 | 0.24829 | 0.0041 | 0.0183 |
| m12_hybrid_td_raw | control | TimeDecay(180) | 758 | 0.68732 | 0.68988 | 0.24711 | 0.24829 | 0.0100 | 0.0183 |
| m12_grw_raw | control | MultiScaleGRW | 758 | 0.68973 | 0.68988 | 0.24833 | 0.24829 | 0.0133 | 0.0183 |
| m12_joint_hybrid_synergy_grw_hier_ha | candidate | MultiScaleGRW | 758 | 0.69144 | 0.68988 | 0.24915 | 0.24829 | 0.0306 | 0.0183 |

## Scope: BTTS

| model | role | dynamics | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | candidate | TimeDecay(180) | 356 | 0.68373 | 0.68337 | 0.24532 | 0.24516 | 0.0164 | 0.0300 |
| m05_joint_td_raw | control | TimeDecay(180) | 356 | 0.68405 | 0.68337 | 0.24549 | 0.24516 | 0.0040 | 0.0300 |
| m12_joint_hybrid_synergy_grw_hier_ha | candidate | MultiScaleGRW | 356 | 0.68466 | 0.68337 | 0.24572 | 0.24516 | 0.0610 | 0.0300 |
| m12_grw_raw | control | MultiScaleGRW | 356 | 0.68496 | 0.68337 | 0.24591 | 0.24516 | 0.0402 | 0.0300 |
| m12_joint_hybrid_synergy_hier_ha | candidate | TimeDecay(180) | 356 | 0.68517 | 0.68337 | 0.24602 | 0.24516 | 0.0207 | 0.0300 |
| m12_hybrid_td_raw | control | TimeDecay(180) | 356 | 0.68520 | 0.68337 | 0.24604 | 0.24516 | 0.0087 | 0.0300 |

## Paired ΔLogLoss, fixture-clustered bootstrap (B = 10000)

Negative Δ favours the left arm. 95% percentile interval.

| scope | cut | left | right | n_obs | n_fixtures | delta | lo | hi | p_negative | significant |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2899 | 627 | -0.00027 | -0.00092 | +0.00038 | 0.790 | false |
| all | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2889 | 625 | -0.00021 | -0.00086 | +0.00042 | 0.737 | false |
| all | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 813 | 178 | +0.00095 | +0.00005 | +0.00181 | 0.020 | true |
| all | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2086 | 449 | -0.00074 | -0.00158 | +0.00006 | 0.962 | false |
| all | turf home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 813 | 178 | +0.00095 | +0.00005 | +0.00181 | 0.020 | true |
| all | grass home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2076 | 447 | -0.00066 | -0.00151 | +0.00015 | 0.946 | false |
| all | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 2899 | 627 | +0.00090 | -0.00534 | +0.00718 | 0.382 | false |
| all | all fixtures | m05_joint_td_raw | betfair_close | 2899 | 627 | +0.00117 | -0.00513 | +0.00753 | 0.351 | false |
| all | all fixtures | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 2899 | 627 | -0.00004 | -0.00055 | +0.00046 | 0.563 | false |
| all | excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 2889 | 625 | +0.00002 | -0.00050 | +0.00052 | 0.466 | false |
| all | turf home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 813 | 178 | +0.00052 | -0.00019 | +0.00121 | 0.074 | false |
| all | grass home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 2086 | 449 | -0.00026 | -0.00092 | +0.00038 | 0.788 | false |
| all | turf home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 813 | 178 | +0.00052 | -0.00019 | +0.00121 | 0.074 | false |
| all | grass home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 2076 | 447 | -0.00017 | -0.00081 | +0.00045 | 0.705 | false |
| all | all fixtures | m12_joint_hybrid_synergy_hier_ha | betfair_close | 2899 | 627 | +0.00151 | -0.00515 | +0.00825 | 0.320 | false |
| all | all fixtures | m12_hybrid_td_raw | betfair_close | 2899 | 627 | +0.00155 | -0.00518 | +0.00838 | 0.317 | false |
| all | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 2899 | 627 | +0.00065 | -0.00084 | +0.00215 | 0.201 | false |
| all | excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 2889 | 625 | +0.00072 | -0.00077 | +0.00224 | 0.171 | false |
| all | turf home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 813 | 178 | +0.00271 | +0.00004 | +0.00525 | 0.022 | true |
| all | grass home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 2086 | 449 | -0.00015 | -0.00193 | +0.00163 | 0.575 | false |
| all | turf home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 813 | 178 | +0.00271 | +0.00004 | +0.00525 | 0.022 | true |
| all | grass home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 2076 | 447 | -0.00005 | -0.00184 | +0.00176 | 0.530 | false |
| all | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | betfair_close | 2899 | 627 | +0.00320 | -0.00338 | +0.00976 | 0.165 | false |
| all | all fixtures | m12_grw_raw | betfair_close | 2899 | 627 | +0.00255 | -0.00379 | +0.00891 | 0.210 | false |
| 1X2 | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1785 | 595 | -0.00006 | -0.00064 | +0.00052 | 0.578 | false |
| 1X2 | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1779 | 593 | +0.00001 | -0.00056 | +0.00057 | 0.494 | false |
| 1X2 | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 495 | 165 | +0.00079 | -0.00007 | +0.00165 | 0.036 | false |
| 1X2 | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1290 | 430 | -0.00038 | -0.00111 | +0.00033 | 0.851 | false |
| 1X2 | turf home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 495 | 165 | +0.00079 | -0.00007 | +0.00165 | 0.036 | false |
| 1X2 | grass home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1284 | 428 | -0.00030 | -0.00103 | +0.00041 | 0.794 | false |
| 1X2 | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 1785 | 595 | +0.00293 | -0.00670 | +0.01223 | 0.266 | false |
| 1X2 | all fixtures | m05_joint_td_raw | betfair_close | 1785 | 595 | +0.00299 | -0.00662 | +0.01228 | 0.263 | false |
| 1X2 | all fixtures | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 1785 | 595 | +0.00005 | -0.00046 | +0.00056 | 0.419 | false |
| 1X2 | excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 1779 | 593 | +0.00012 | -0.00038 | +0.00062 | 0.317 | false |
| 1X2 | turf home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 495 | 165 | +0.00051 | -0.00025 | +0.00128 | 0.097 | false |
| 1X2 | grass home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 1290 | 430 | -0.00013 | -0.00076 | +0.00050 | 0.648 | false |
| 1X2 | turf home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 495 | 165 | +0.00051 | -0.00025 | +0.00128 | 0.097 | false |
| 1X2 | grass home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 1284 | 428 | -0.00003 | -0.00066 | +0.00058 | 0.545 | false |
| 1X2 | all fixtures | m12_joint_hybrid_synergy_hier_ha | betfair_close | 1785 | 595 | +0.00329 | -0.00688 | +0.01303 | 0.256 | false |
| 1X2 | all fixtures | m12_hybrid_td_raw | betfair_close | 1785 | 595 | +0.00324 | -0.00691 | +0.01303 | 0.257 | false |
| 1X2 | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 1785 | 595 | +0.00039 | -0.00120 | +0.00194 | 0.323 | false |
| 1X2 | excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 1779 | 593 | +0.00045 | -0.00110 | +0.00203 | 0.287 | false |
| 1X2 | turf home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 495 | 165 | +0.00088 | -0.00177 | +0.00349 | 0.269 | false |
| 1X2 | grass home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 1290 | 430 | +0.00020 | -0.00168 | +0.00205 | 0.421 | false |
| 1X2 | turf home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 495 | 165 | +0.00088 | -0.00177 | +0.00349 | 0.269 | false |
| 1X2 | grass home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 1284 | 428 | +0.00029 | -0.00163 | +0.00219 | 0.383 | false |
| 1X2 | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | betfair_close | 1785 | 595 | +0.00428 | -0.00500 | +0.01332 | 0.179 | false |
| 1X2 | all fixtures | m12_grw_raw | betfair_close | 1785 | 595 | +0.00389 | -0.00536 | +0.01277 | 0.198 | false |
| OU2.5 | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 758 | 379 | -0.00074 | -0.00230 | +0.00078 | 0.834 | false |
| OU2.5 | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 756 | 378 | -0.00052 | -0.00200 | +0.00091 | 0.754 | false |
| OU2.5 | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 220 | 110 | +0.00118 | -0.00093 | +0.00326 | 0.136 | false |
| OU2.5 | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 538 | 269 | -0.00153 | -0.00342 | +0.00043 | 0.939 | false |
| OU2.5 | turf home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 220 | 110 | +0.00118 | -0.00093 | +0.00326 | 0.136 | false |
| OU2.5 | grass home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 536 | 268 | -0.00122 | -0.00309 | +0.00059 | 0.902 | false |
| OU2.5 | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 758 | 379 | -0.00362 | -0.01454 | +0.00725 | 0.745 | false |
| OU2.5 | all fixtures | m05_joint_td_raw | betfair_close | 758 | 379 | -0.00288 | -0.01411 | +0.00826 | 0.696 | false |
| OU2.5 | all fixtures | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 758 | 379 | -0.00026 | -0.00153 | +0.00094 | 0.666 | false |
| OU2.5 | excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 756 | 378 | -0.00005 | -0.00124 | +0.00110 | 0.539 | false |
| OU2.5 | turf home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 220 | 110 | +0.00030 | -0.00153 | +0.00211 | 0.373 | false |
| OU2.5 | grass home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 538 | 269 | -0.00049 | -0.00207 | +0.00107 | 0.732 | false |
| OU2.5 | turf home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 220 | 110 | +0.00030 | -0.00153 | +0.00211 | 0.373 | false |
| OU2.5 | grass home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 536 | 268 | -0.00019 | -0.00166 | +0.00125 | 0.594 | false |
| OU2.5 | all fixtures | m12_joint_hybrid_synergy_hier_ha | betfair_close | 758 | 379 | -0.00282 | -0.01336 | +0.00764 | 0.703 | false |
| OU2.5 | all fixtures | m12_hybrid_td_raw | betfair_close | 758 | 379 | -0.00255 | -0.01330 | +0.00813 | 0.680 | false |
| OU2.5 | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 758 | 379 | +0.00172 | -0.00166 | +0.00516 | 0.168 | false |
| OU2.5 | excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 756 | 378 | +0.00197 | -0.00153 | +0.00534 | 0.133 | false |
| OU2.5 | turf home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 220 | 110 | +0.00508 | -0.00132 | +0.01134 | 0.062 | false |
| OU2.5 | grass home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 538 | 269 | +0.00034 | -0.00363 | +0.00428 | 0.432 | false |
| OU2.5 | turf home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 220 | 110 | +0.00508 | -0.00132 | +0.01134 | 0.062 | false |
| OU2.5 | grass home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 536 | 268 | +0.00070 | -0.00324 | +0.00463 | 0.361 | false |
| OU2.5 | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | betfair_close | 758 | 379 | +0.00157 | -0.01164 | +0.01518 | 0.411 | false |
| OU2.5 | all fixtures | m12_grw_raw | betfair_close | 758 | 379 | -0.00015 | -0.01288 | +0.01276 | 0.511 | false |
| BTTS | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 356 | 178 | -0.00031 | -0.00222 | +0.00156 | 0.625 | false |
| BTTS | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 354 | 177 | -0.00062 | -0.00242 | +0.00117 | 0.745 | false |
| BTTS | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 98 | 49 | +0.00122 | -0.00121 | +0.00366 | 0.168 | false |
| BTTS | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 258 | 129 | -0.00089 | -0.00332 | +0.00148 | 0.767 | false |
| BTTS | turf home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 98 | 49 | +0.00122 | -0.00121 | +0.00366 | 0.168 | false |
| BTTS | grass home excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 256 | 128 | -0.00132 | -0.00361 | +0.00092 | 0.867 | false |
| BTTS | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 356 | 178 | +0.00036 | -0.00937 | +0.01010 | 0.471 | false |
| BTTS | all fixtures | m05_joint_td_raw | betfair_close | 356 | 178 | +0.00067 | -0.00916 | +0.01043 | 0.446 | false |
| BTTS | all fixtures | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 356 | 178 | -0.00003 | -0.00161 | +0.00157 | 0.516 | false |
| BTTS | excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 354 | 177 | -0.00032 | -0.00178 | +0.00115 | 0.655 | false |
| BTTS | turf home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 98 | 49 | +0.00106 | -0.00112 | +0.00325 | 0.174 | false |
| BTTS | grass home | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 258 | 129 | -0.00045 | -0.00246 | +0.00157 | 0.670 | false |
| BTTS | turf home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 98 | 49 | +0.00106 | -0.00112 | +0.00325 | 0.174 | false |
| BTTS | grass home excluding T003 | m12_joint_hybrid_synergy_hier_ha | m12_hybrid_td_raw | 256 | 128 | -0.00084 | -0.00269 | +0.00106 | 0.808 | false |
| BTTS | all fixtures | m12_joint_hybrid_synergy_hier_ha | betfair_close | 356 | 178 | +0.00180 | -0.00824 | +0.01171 | 0.366 | false |
| BTTS | all fixtures | m12_hybrid_td_raw | betfair_close | 356 | 178 | +0.00183 | -0.00835 | +0.01178 | 0.364 | false |
| BTTS | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 356 | 178 | -0.00030 | -0.00458 | +0.00400 | 0.551 | false |
| BTTS | excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 354 | 177 | -0.00059 | -0.00484 | +0.00370 | 0.599 | false |
| BTTS | turf home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 98 | 49 | +0.00665 | -0.00144 | +0.01433 | 0.052 | false |
| BTTS | grass home | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 258 | 129 | -0.00294 | -0.00800 | +0.00193 | 0.879 | false |
| BTTS | turf home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 98 | 49 | +0.00665 | -0.00144 | +0.01433 | 0.052 | false |
| BTTS | grass home excluding T003 | m12_joint_hybrid_synergy_grw_hier_ha | m12_grw_raw | 256 | 128 | -0.00336 | -0.00827 | +0.00169 | 0.903 | false |
| BTTS | all fixtures | m12_joint_hybrid_synergy_grw_hier_ha | betfair_close | 356 | 178 | +0.00129 | -0.01168 | +0.01442 | 0.426 | false |
| BTTS | all fixtures | m12_grw_raw | betfair_close | 356 | 178 | +0.00159 | -0.01082 | +0.01403 | 0.405 | false |

## σ_γ and the turf − grass contrast

| model | fold | n_turf | n_grass | delta_mean | delta_q05 | delta_q95 | p_turf_above | gamma_base_mean | sigma_q05 | sigma_q50 | sigma_q95 | p_sigma_below_0p02 | prior_p_sigma_below_0p02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | 20 | 7 | 18 | -0.0060 | -0.0558 | +0.0323 | 0.4230 | 0.1382 | 0.0050 | 0.0490 | 0.1219 | 0.2010 | 0.1585 |
| m12_joint_hybrid_synergy_hier_ha | 20 | 7 | 18 | +0.0025 | -0.0336 | +0.0435 | 0.5305 | 0.1384 | 0.0046 | 0.0468 | 0.1161 | 0.2165 | 0.1585 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | 7 | 18 | -0.0044 | -0.0578 | +0.0504 | 0.4320 | 0.1306 | 0.0463 | 0.0980 | 0.1491 | 0.0080 | 0.1585 |
| m05_joint_production_wealth_hier_ha | 40 | 7 | 16 | -0.0019 | -0.0385 | +0.0311 | 0.4765 | 0.1196 | 0.0044 | 0.0369 | 0.0967 | 0.2700 | 0.1585 |
| m12_joint_hybrid_synergy_hier_ha | 40 | 7 | 16 | +0.0018 | -0.0305 | +0.0364 | 0.5190 | 0.1204 | 0.0038 | 0.0360 | 0.0950 | 0.2945 | 0.1585 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | 7 | 16 | -0.0323 | -0.0836 | +0.0129 | 0.1335 | 0.1172 | 0.0484 | 0.0867 | 0.1310 | 0.0055 | 0.1585 |

## Club home effects

| model | fold | team | surface | gamma_mean | gamma_sd | gamma_q05 | gamma_q95 | home_multiplier | p_above_base |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | 20 | falkirk-fc | grass | 0.1673 | 0.0804 | 0.0551 | 0.3132 | 1.1860 | 0.6410 |
| m05_joint_production_wealth_hier_ha | 20 | annan-athletic | grass | 0.1664 | 0.0709 | 0.0659 | 0.2937 | 1.1840 | 0.6660 |
| m05_joint_production_wealth_hier_ha | 20 | edinburgh-city-fc | turf | 0.1657 | 0.0686 | 0.0666 | 0.2907 | 1.1831 | 0.6675 |
| m05_joint_production_wealth_hier_ha | 20 | peterhead | grass | 0.1639 | 0.0675 | 0.0659 | 0.2824 | 1.1808 | 0.6635 |
| m05_joint_production_wealth_hier_ha | 20 | elgin-city | grass | 0.1623 | 0.0665 | 0.0639 | 0.2875 | 1.1789 | 0.6580 |
| m05_joint_production_wealth_hier_ha | 20 | east-fife | grass | 0.1516 | 0.0648 | 0.0504 | 0.2646 | 1.1662 | 0.5825 |
| m05_joint_production_wealth_hier_ha | 20 | queen-of-the-south | grass | 0.1511 | 0.0672 | 0.0496 | 0.2681 | 1.1658 | 0.5710 |
| m05_joint_production_wealth_hier_ha | 20 | hamilton-academical | grass | 0.1501 | 0.0732 | 0.0430 | 0.2764 | 1.1651 | 0.5615 |
| m05_joint_production_wealth_hier_ha | 20 | arbroath | grass | 0.1442 | 0.0682 | 0.0350 | 0.2564 | 1.1578 | 0.5485 |
| m05_joint_production_wealth_hier_ha | 20 | stirling-albion | grass | 0.1442 | 0.0656 | 0.0415 | 0.2555 | 1.1576 | 0.5360 |
| m05_joint_production_wealth_hier_ha | 20 | bonnyrigg-rose | grass | 0.1429 | 0.0650 | 0.0386 | 0.2546 | 1.1561 | 0.5370 |
| m05_joint_production_wealth_hier_ha | 20 | dunfermline-athletic | grass | 0.1413 | 0.0769 | 0.0268 | 0.2681 | 1.1552 | 0.5115 |
| m05_joint_production_wealth_hier_ha | 20 | montrose | turf | 0.1400 | 0.0664 | 0.0349 | 0.2506 | 1.1529 | 0.5185 |
| m05_joint_production_wealth_hier_ha | 20 | airdrieonians | turf | 0.1393 | 0.0789 | 0.0137 | 0.2671 | 1.1531 | 0.5025 |
| m05_joint_production_wealth_hier_ha | 20 | albion-rovers | grass | 0.1362 | 0.0781 | 0.0088 | 0.2587 | 1.1494 | 0.4995 |
| m05_joint_production_wealth_hier_ha | 20 | clyde-fc | turf | 0.1355 | 0.0651 | 0.0295 | 0.2413 | 1.1475 | 0.4775 |
| m05_joint_production_wealth_hier_ha | 20 | dumbarton | grass | 0.1292 | 0.0640 | 0.0198 | 0.2320 | 1.1403 | 0.4245 |
| m05_joint_production_wealth_hier_ha | 20 | inverness-caledonian-thistle | grass | 0.1248 | 0.0673 | 0.0114 | 0.2233 | 1.1355 | 0.4060 |
| m05_joint_production_wealth_hier_ha | 20 | stenhousemuir | grass | 0.1237 | 0.0665 | 0.0070 | 0.2293 | 1.1341 | 0.4045 |
| m05_joint_production_wealth_hier_ha | 20 | alloa-athletic | turf | 0.1213 | 0.0667 | 0.0003 | 0.2192 | 1.1315 | 0.4045 |
| m05_joint_production_wealth_hier_ha | 20 | cove-rangers | turf | 0.1183 | 0.0656 | 0.0030 | 0.2167 | 1.1279 | 0.3735 |
| m05_joint_production_wealth_hier_ha | 20 | forfar-athletic | grass | 0.1182 | 0.0670 | -0.0022 | 0.2198 | 1.1279 | 0.3925 |
| m05_joint_production_wealth_hier_ha | 20 | the-spartans-fc | turf | 0.1166 | 0.0652 | 0.0013 | 0.2099 | 1.1260 | 0.3765 |
| m05_joint_production_wealth_hier_ha | 20 | stranraer | grass | 0.1151 | 0.0681 | -0.0071 | 0.2127 | 1.1246 | 0.3625 |
| m05_joint_production_wealth_hier_ha | 20 | kelty-hearts-fc | grass | 0.0843 | 0.0816 | -0.0663 | 0.1933 | 1.0915 | 0.2235 |
| m12_joint_hybrid_synergy_hier_ha | 20 | edinburgh-city-fc | turf | 0.1710 | 0.0700 | 0.0744 | 0.3014 | 1.1894 | 0.6915 |
| m12_joint_hybrid_synergy_hier_ha | 20 | peterhead | grass | 0.1697 | 0.0682 | 0.0708 | 0.2990 | 1.1878 | 0.7070 |
| m12_joint_hybrid_synergy_hier_ha | 20 | elgin-city | grass | 0.1660 | 0.0652 | 0.0716 | 0.2897 | 1.1831 | 0.6795 |
| m12_joint_hybrid_synergy_hier_ha | 20 | annan-athletic | grass | 0.1650 | 0.0679 | 0.0722 | 0.2859 | 1.1822 | 0.6700 |
| m12_joint_hybrid_synergy_hier_ha | 20 | montrose | turf | 0.1529 | 0.0622 | 0.0569 | 0.2585 | 1.1675 | 0.6000 |
| m12_joint_hybrid_synergy_hier_ha | 20 | east-fife | grass | 0.1511 | 0.0643 | 0.0509 | 0.2674 | 1.1655 | 0.5890 |
| m12_joint_hybrid_synergy_hier_ha | 20 | queen-of-the-south | grass | 0.1498 | 0.0634 | 0.0523 | 0.2541 | 1.1640 | 0.5825 |
| m12_joint_hybrid_synergy_hier_ha | 20 | stirling-albion | grass | 0.1479 | 0.0634 | 0.0520 | 0.2549 | 1.1617 | 0.5525 |
| m12_joint_hybrid_synergy_hier_ha | 20 | arbroath | grass | 0.1421 | 0.0647 | 0.0399 | 0.2520 | 1.1552 | 0.5135 |
| m12_joint_hybrid_synergy_hier_ha | 20 | hamilton-academical | grass | 0.1412 | 0.0686 | 0.0314 | 0.2558 | 1.1543 | 0.5120 |
| m12_joint_hybrid_synergy_hier_ha | 20 | airdrieonians | turf | 0.1408 | 0.0748 | 0.0237 | 0.2676 | 1.1544 | 0.5035 |
| m12_joint_hybrid_synergy_hier_ha | 20 | dunfermline-athletic | grass | 0.1407 | 0.0762 | 0.0155 | 0.2647 | 1.1544 | 0.5215 |
| m12_joint_hybrid_synergy_hier_ha | 20 | falkirk-fc | grass | 0.1378 | 0.0693 | 0.0272 | 0.2512 | 1.1505 | 0.4920 |
| m12_joint_hybrid_synergy_hier_ha | 20 | clyde-fc | turf | 0.1370 | 0.0621 | 0.0378 | 0.2410 | 1.1490 | 0.4845 |
| m12_joint_hybrid_synergy_hier_ha | 20 | albion-rovers | grass | 0.1364 | 0.0748 | 0.0188 | 0.2518 | 1.1493 | 0.5050 |
| m12_joint_hybrid_synergy_hier_ha | 20 | bonnyrigg-rose | grass | 0.1345 | 0.0632 | 0.0305 | 0.2384 | 1.1463 | 0.4805 |
| m12_joint_hybrid_synergy_hier_ha | 20 | cove-rangers | turf | 0.1344 | 0.0622 | 0.0345 | 0.2387 | 1.1461 | 0.4655 |
| m12_joint_hybrid_synergy_hier_ha | 20 | dumbarton | grass | 0.1312 | 0.0619 | 0.0251 | 0.2328 | 1.1423 | 0.4490 |
| m12_joint_hybrid_synergy_hier_ha | 20 | alloa-athletic | turf | 0.1251 | 0.0646 | 0.0125 | 0.2282 | 1.1357 | 0.4235 |
| m12_joint_hybrid_synergy_hier_ha | 20 | inverness-caledonian-thistle | grass | 0.1243 | 0.0641 | 0.0131 | 0.2213 | 1.1347 | 0.4160 |
| m12_joint_hybrid_synergy_hier_ha | 20 | stranraer | grass | 0.1191 | 0.0642 | 0.0050 | 0.2149 | 1.1288 | 0.3795 |
| m12_joint_hybrid_synergy_hier_ha | 20 | the-spartans-fc | turf | 0.1177 | 0.0640 | 0.0076 | 0.2149 | 1.1272 | 0.3630 |
| m12_joint_hybrid_synergy_hier_ha | 20 | forfar-athletic | grass | 0.1158 | 0.0650 | -0.0026 | 0.2151 | 1.1251 | 0.3495 |
| m12_joint_hybrid_synergy_hier_ha | 20 | stenhousemuir | grass | 0.1100 | 0.0691 | -0.0134 | 0.2079 | 1.1189 | 0.3140 |
| m12_joint_hybrid_synergy_hier_ha | 20 | kelty-hearts-fc | grass | 0.0890 | 0.0781 | -0.0585 | 0.1899 | 1.0963 | 0.2345 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | airdrieonians | turf | 0.2643 | 0.1130 | 0.1030 | 0.4624 | 1.3110 | 0.9110 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | annan-athletic | grass | 0.2389 | 0.0758 | 0.1199 | 0.3704 | 1.2736 | 0.9395 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | elgin-city | grass | 0.2225 | 0.0716 | 0.1096 | 0.3476 | 1.2524 | 0.9110 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | stirling-albion | grass | 0.2016 | 0.0701 | 0.0929 | 0.3234 | 1.2264 | 0.8500 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | queen-of-the-south | grass | 0.1873 | 0.0715 | 0.0791 | 0.3084 | 1.2091 | 0.7960 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | montrose | turf | 0.1830 | 0.0674 | 0.0773 | 0.2970 | 1.2036 | 0.7795 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | bonnyrigg-rose | grass | 0.1754 | 0.0669 | 0.0737 | 0.2929 | 1.1944 | 0.7420 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | edinburgh-city-fc | turf | 0.1707 | 0.0699 | 0.0592 | 0.2893 | 1.1891 | 0.7130 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | stranraer | grass | 0.1681 | 0.0691 | 0.0594 | 0.2827 | 1.1859 | 0.7015 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | peterhead | grass | 0.1503 | 0.0687 | 0.0418 | 0.2631 | 1.1649 | 0.6060 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | dunfermline-athletic | grass | 0.1412 | 0.0929 | -0.0136 | 0.2909 | 1.1567 | 0.5440 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | forfar-athletic | grass | 0.1311 | 0.0686 | 0.0185 | 0.2509 | 1.1428 | 0.5100 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | dumbarton | grass | 0.1208 | 0.0673 | 0.0093 | 0.2319 | 1.1310 | 0.4385 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | east-fife | grass | 0.1105 | 0.0690 | -0.0024 | 0.2240 | 1.1195 | 0.3920 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | falkirk-fc | grass | 0.1089 | 0.0740 | -0.0148 | 0.2268 | 1.1181 | 0.3905 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | hamilton-academical | grass | 0.0993 | 0.0807 | -0.0382 | 0.2281 | 1.1080 | 0.3510 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | cove-rangers | turf | 0.0951 | 0.0699 | -0.0258 | 0.2089 | 1.1025 | 0.3125 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | albion-rovers | grass | 0.0862 | 0.0982 | -0.0883 | 0.2400 | 1.0953 | 0.3230 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | stenhousemuir | grass | 0.0856 | 0.0657 | -0.0294 | 0.1903 | 1.0917 | 0.2465 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | clyde-fc | turf | 0.0843 | 0.0677 | -0.0279 | 0.1950 | 1.0905 | 0.2445 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | arbroath | grass | 0.0814 | 0.0846 | -0.0634 | 0.2128 | 1.0887 | 0.2820 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | alloa-athletic | turf | 0.0636 | 0.0720 | -0.0610 | 0.1708 | 1.0684 | 0.1685 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | inverness-caledonian-thistle | grass | 0.0523 | 0.0856 | -0.0946 | 0.1859 | 1.0576 | 0.1690 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | the-spartans-fc | turf | 0.0217 | 0.0803 | -0.1211 | 0.1476 | 1.0252 | 0.0755 |
| m12_joint_hybrid_synergy_grw_hier_ha | 20 | kelty-hearts-fc | grass | -0.0120 | 0.0809 | -0.1445 | 0.1182 | 0.9913 | 0.0315 |
| m05_joint_production_wealth_hier_ha | 40 | inverness-caledonian-thistle | grass | 0.1525 | 0.0641 | 0.0654 | 0.2674 | 1.1672 | 0.7040 |
| m05_joint_production_wealth_hier_ha | 40 | queen-of-the-south | grass | 0.1372 | 0.0565 | 0.0522 | 0.2361 | 1.1489 | 0.6350 |
| m05_joint_production_wealth_hier_ha | 40 | peterhead | grass | 0.1336 | 0.0561 | 0.0475 | 0.2299 | 1.1448 | 0.6135 |
| m05_joint_production_wealth_hier_ha | 40 | montrose | turf | 0.1282 | 0.0569 | 0.0389 | 0.2227 | 1.1386 | 0.5735 |
| m05_joint_production_wealth_hier_ha | 40 | dumbarton | grass | 0.1268 | 0.0573 | 0.0392 | 0.2271 | 1.1371 | 0.5545 |
| m05_joint_production_wealth_hier_ha | 40 | falkirk-fc | grass | 0.1264 | 0.0645 | 0.0309 | 0.2353 | 1.1371 | 0.5340 |
| m05_joint_production_wealth_hier_ha | 40 | annan-athletic | grass | 0.1253 | 0.0532 | 0.0402 | 0.2139 | 1.1351 | 0.5475 |
| m05_joint_production_wealth_hier_ha | 40 | alloa-athletic | turf | 0.1245 | 0.0557 | 0.0378 | 0.2186 | 1.1344 | 0.5440 |
| m05_joint_production_wealth_hier_ha | 40 | arbroath | grass | 0.1243 | 0.0600 | 0.0295 | 0.2234 | 1.1344 | 0.5320 |
| m05_joint_production_wealth_hier_ha | 40 | edinburgh-city-fc | turf | 0.1238 | 0.0553 | 0.0350 | 0.2168 | 1.1335 | 0.5270 |
| m05_joint_production_wealth_hier_ha | 40 | east-kilbride | turf | 0.1237 | 0.0572 | 0.0356 | 0.2165 | 1.1335 | 0.5235 |
| m05_joint_production_wealth_hier_ha | 40 | stenhousemuir | grass | 0.1230 | 0.0557 | 0.0380 | 0.2171 | 1.1327 | 0.5255 |
| m05_joint_production_wealth_hier_ha | 40 | bonnyrigg-rose | grass | 0.1192 | 0.0600 | 0.0217 | 0.2168 | 1.1287 | 0.4950 |
| m05_joint_production_wealth_hier_ha | 40 | clyde-fc | turf | 0.1159 | 0.0563 | 0.0232 | 0.2052 | 1.1247 | 0.4685 |
| m05_joint_production_wealth_hier_ha | 40 | elgin-city | grass | 0.1158 | 0.0558 | 0.0223 | 0.2050 | 1.1245 | 0.4755 |
| m05_joint_production_wealth_hier_ha | 40 | stranraer | grass | 0.1154 | 0.0526 | 0.0269 | 0.1997 | 1.1238 | 0.4635 |
| m05_joint_production_wealth_hier_ha | 40 | forfar-athletic | grass | 0.1120 | 0.0576 | 0.0171 | 0.2028 | 1.1204 | 0.4425 |
| m05_joint_production_wealth_hier_ha | 40 | east-fife | grass | 0.1113 | 0.0562 | 0.0159 | 0.2016 | 1.1195 | 0.4365 |
| m05_joint_production_wealth_hier_ha | 40 | stirling-albion | grass | 0.1094 | 0.0562 | 0.0130 | 0.1943 | 1.1173 | 0.4170 |
| m05_joint_production_wealth_hier_ha | 40 | cove-rangers | turf | 0.1058 | 0.0580 | 0.0036 | 0.1913 | 1.1135 | 0.4145 |
| m05_joint_production_wealth_hier_ha | 40 | the-spartans-fc | turf | 0.1031 | 0.0568 | 0.0043 | 0.1878 | 1.1104 | 0.3800 |
| m05_joint_production_wealth_hier_ha | 40 | hamilton-academical | grass | 0.0941 | 0.0620 | -0.0171 | 0.1801 | 1.1008 | 0.3215 |
| m05_joint_production_wealth_hier_ha | 40 | kelty-hearts-fc | grass | 0.0896 | 0.0640 | -0.0295 | 0.1784 | 1.0959 | 0.2945 |
| m12_joint_hybrid_synergy_hier_ha | 40 | inverness-caledonian-thistle | grass | 0.1474 | 0.0603 | 0.0625 | 0.2575 | 1.1610 | 0.6875 |
| m12_joint_hybrid_synergy_hier_ha | 40 | montrose | turf | 0.1369 | 0.0570 | 0.0481 | 0.2369 | 1.1486 | 0.6150 |
| m12_joint_hybrid_synergy_hier_ha | 40 | queen-of-the-south | grass | 0.1350 | 0.0569 | 0.0536 | 0.2383 | 1.1464 | 0.6140 |
| m12_joint_hybrid_synergy_hier_ha | 40 | annan-athletic | grass | 0.1339 | 0.0548 | 0.0478 | 0.2307 | 1.1451 | 0.6075 |
| m12_joint_hybrid_synergy_hier_ha | 40 | alloa-athletic | turf | 0.1279 | 0.0549 | 0.0408 | 0.2186 | 1.1381 | 0.5550 |
| m12_joint_hybrid_synergy_hier_ha | 40 | east-kilbride | turf | 0.1274 | 0.0552 | 0.0417 | 0.2218 | 1.1376 | 0.5530 |
| m12_joint_hybrid_synergy_hier_ha | 40 | peterhead | grass | 0.1269 | 0.0546 | 0.0400 | 0.2163 | 1.1370 | 0.5505 |
| m12_joint_hybrid_synergy_hier_ha | 40 | dumbarton | grass | 0.1265 | 0.0554 | 0.0365 | 0.2181 | 1.1366 | 0.5450 |
| m12_joint_hybrid_synergy_hier_ha | 40 | edinburgh-city-fc | turf | 0.1246 | 0.0543 | 0.0370 | 0.2150 | 1.1344 | 0.5225 |
| m12_joint_hybrid_synergy_hier_ha | 40 | falkirk-fc | grass | 0.1224 | 0.0626 | 0.0274 | 0.2266 | 1.1324 | 0.5175 |
| m12_joint_hybrid_synergy_hier_ha | 40 | bonnyrigg-rose | grass | 0.1220 | 0.0576 | 0.0316 | 0.2165 | 1.1316 | 0.5115 |
| m12_joint_hybrid_synergy_hier_ha | 40 | arbroath | grass | 0.1216 | 0.0594 | 0.0277 | 0.2165 | 1.1313 | 0.5035 |
| m12_joint_hybrid_synergy_hier_ha | 40 | stenhousemuir | grass | 0.1205 | 0.0532 | 0.0319 | 0.2057 | 1.1296 | 0.5155 |
| m12_joint_hybrid_synergy_hier_ha | 40 | stranraer | grass | 0.1193 | 0.0537 | 0.0326 | 0.2076 | 1.1283 | 0.4730 |
| m12_joint_hybrid_synergy_hier_ha | 40 | clyde-fc | turf | 0.1171 | 0.0531 | 0.0307 | 0.2024 | 1.1258 | 0.4665 |
| m12_joint_hybrid_synergy_hier_ha | 40 | stirling-albion | grass | 0.1159 | 0.0554 | 0.0274 | 0.2047 | 1.1246 | 0.4595 |
| m12_joint_hybrid_synergy_hier_ha | 40 | east-fife | grass | 0.1116 | 0.0548 | 0.0136 | 0.1964 | 1.1197 | 0.4290 |
| m12_joint_hybrid_synergy_hier_ha | 40 | forfar-athletic | grass | 0.1113 | 0.0558 | 0.0141 | 0.1971 | 1.1195 | 0.4395 |
| m12_joint_hybrid_synergy_hier_ha | 40 | elgin-city | grass | 0.1091 | 0.0544 | 0.0150 | 0.1939 | 1.1169 | 0.4100 |
| m12_joint_hybrid_synergy_hier_ha | 40 | the-spartans-fc | turf | 0.1089 | 0.0545 | 0.0147 | 0.1922 | 1.1167 | 0.4050 |
| m12_joint_hybrid_synergy_hier_ha | 40 | cove-rangers | turf | 0.1063 | 0.0564 | 0.0130 | 0.1931 | 1.1139 | 0.3695 |
| m12_joint_hybrid_synergy_hier_ha | 40 | kelty-hearts-fc | grass | 0.0945 | 0.0616 | -0.0205 | 0.1808 | 1.1011 | 0.3310 |
| m12_joint_hybrid_synergy_hier_ha | 40 | hamilton-academical | grass | 0.0936 | 0.0636 | -0.0213 | 0.1839 | 1.1003 | 0.3180 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | annan-athletic | grass | 0.2294 | 0.0649 | 0.1250 | 0.3374 | 1.2605 | 0.9630 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | queen-of-the-south | grass | 0.1919 | 0.0628 | 0.0950 | 0.3039 | 1.2140 | 0.8885 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | peterhead | grass | 0.1918 | 0.0623 | 0.0947 | 0.2953 | 1.2138 | 0.8840 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | bonnyrigg-rose | grass | 0.1832 | 0.0690 | 0.0775 | 0.2970 | 1.2039 | 0.8395 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | edinburgh-city-fc | turf | 0.1760 | 0.0618 | 0.0795 | 0.2799 | 1.1947 | 0.8275 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | elgin-city | grass | 0.1624 | 0.0609 | 0.0663 | 0.2649 | 1.1785 | 0.7725 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | montrose | turf | 0.1596 | 0.0611 | 0.0654 | 0.2620 | 1.1753 | 0.7500 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | stirling-albion | grass | 0.1569 | 0.0597 | 0.0607 | 0.2543 | 1.1720 | 0.7510 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | dumbarton | grass | 0.1474 | 0.0595 | 0.0470 | 0.2451 | 1.1608 | 0.6865 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | forfar-athletic | grass | 0.1251 | 0.0587 | 0.0286 | 0.2221 | 1.1352 | 0.5500 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | inverness-caledonian-thistle | grass | 0.1239 | 0.0644 | 0.0154 | 0.2289 | 1.1343 | 0.5410 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | stranraer | grass | 0.1220 | 0.0593 | 0.0268 | 0.2174 | 1.1317 | 0.5365 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | east-fife | grass | 0.1023 | 0.0609 | 0.0042 | 0.2024 | 1.1098 | 0.3960 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | falkirk-fc | grass | 0.1008 | 0.0719 | -0.0246 | 0.2165 | 1.1089 | 0.4185 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | east-kilbride | turf | 0.0996 | 0.0757 | -0.0237 | 0.2221 | 1.1079 | 0.4080 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | stenhousemuir | grass | 0.0955 | 0.0592 | -0.0026 | 0.1908 | 1.1021 | 0.3730 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | cove-rangers | turf | 0.0939 | 0.0590 | -0.0022 | 0.1864 | 1.1003 | 0.3490 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | arbroath | grass | 0.0750 | 0.0737 | -0.0497 | 0.1930 | 1.0808 | 0.2775 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | alloa-athletic | turf | 0.0740 | 0.0620 | -0.0299 | 0.1728 | 1.0789 | 0.2485 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | hamilton-academical | grass | 0.0456 | 0.0705 | -0.0741 | 0.1557 | 1.0493 | 0.1445 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | clyde-fc | turf | 0.0451 | 0.0631 | -0.0601 | 0.1419 | 1.0482 | 0.1125 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | the-spartans-fc | turf | 0.0144 | 0.0674 | -0.0960 | 0.1226 | 1.0168 | 0.0560 |
| m12_joint_hybrid_synergy_grw_hier_ha | 40 | kelty-hearts-fc | grass | -0.0227 | 0.0701 | -0.1392 | 0.0890 | 0.9799 | 0.0145 |
