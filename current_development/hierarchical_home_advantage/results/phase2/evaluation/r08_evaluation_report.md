# r08 proper scores — Task 008 Phase 2

Generated 2026-09-18 15:05. Panel 710 fixtures; de-vigged Betfair TWA(−20, 0] close. Control `m05_joint_td_raw` reproduced LogLoss 0.64299 / ECE 0.0149. T003 fixtures: 3.

## Scope: all

| model | role | n_obs | logloss | market_logloss | brier | rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | reference | 2899 | 0.64272 | 0.64182 | 0.22574 | 0.22416 | 0.0114 | 0.0139 |
| m05_joint_td_turf_asym | candidate | 2899 | 0.64289 | 0.64182 | 0.22583 | 0.22403 | 0.0118 | 0.0139 |
| m05_joint_td_raw | control | 2899 | 0.64299 | 0.64182 | 0.22586 | 0.22415 | 0.0149 | 0.0139 |
| m05_joint_td_contextual | candidate | 2899 | 0.64299 | 0.64182 | 0.22587 | 0.22410 | 0.0114 | 0.0139 |
| m05_joint_td_turf_dual | candidate | 2899 | 0.64300 | 0.64182 | 0.22587 | 0.22421 | 0.0120 | 0.0139 |
| m12_hybrid_td_raw | control | 2899 | 0.64337 | 0.64182 | 0.22605 | 0.22447 | 0.0100 | 0.0139 |
| m12_joint_hybrid_contextual | candidate | 2899 | 0.64341 | 0.64182 | 0.22607 | 0.22446 | 0.0111 | 0.0139 |

## Scope: 1X2

| model | role | n_obs | logloss | market_logloss | brier | rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_td_turf_asym | candidate | 1785 | 0.61588 | 0.61312 | 0.21290 | 0.22403 | 0.0170 | 0.0186 |
| m05_joint_td_contextual | candidate | 1785 | 0.61600 | 0.61312 | 0.21296 | 0.22410 | 0.0167 | 0.0186 |
| m05_joint_production_wealth_hier_ha | reference | 1785 | 0.61605 | 0.61312 | 0.21298 | 0.22416 | 0.0184 | 0.0186 |
| m05_joint_td_raw | control | 1785 | 0.61611 | 0.61312 | 0.21299 | 0.22415 | 0.0216 | 0.0186 |
| m05_joint_td_turf_dual | candidate | 1785 | 0.61618 | 0.61312 | 0.21304 | 0.22421 | 0.0169 | 0.0186 |
| m12_joint_hybrid_contextual | candidate | 1785 | 0.61633 | 0.61312 | 0.21311 | 0.22446 | 0.0132 | 0.0186 |
| m12_hybrid_td_raw | control | 1785 | 0.61636 | 0.61312 | 0.21311 | 0.22447 | 0.0155 | 0.0186 |

## Scope: OU2.5

| model | role | n_obs | logloss | market_logloss | brier | rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | reference | 758 | 0.68626 | 0.68988 | 0.24658 | 0.22416 | 0.0087 | 0.0183 |
| m05_joint_td_turf_dual | candidate | 758 | 0.68674 | 0.68988 | 0.24682 | 0.22421 | 0.0196 | 0.0183 |
| m05_joint_td_turf_asym | candidate | 758 | 0.68678 | 0.68988 | 0.24684 | 0.22403 | 0.0142 | 0.0183 |
| m05_joint_td_contextual | candidate | 758 | 0.68692 | 0.68988 | 0.24690 | 0.22410 | 0.0112 | 0.0183 |
| m05_joint_td_raw | control | 758 | 0.68700 | 0.68988 | 0.24695 | 0.22415 | 0.0100 | 0.0183 |
| m12_joint_hybrid_contextual | candidate | 758 | 0.68723 | 0.68988 | 0.24705 | 0.22446 | 0.0185 | 0.0183 |
| m12_hybrid_td_raw | control | 758 | 0.68732 | 0.68988 | 0.24711 | 0.22447 | 0.0100 | 0.0183 |

## Scope: BTTS

| model | role | n_obs | logloss | market_logloss | brier | rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_production_wealth_hier_ha | reference | 356 | 0.68373 | 0.68337 | 0.24532 | 0.22416 | 0.0164 | 0.0300 |
| m05_joint_td_raw | control | 356 | 0.68405 | 0.68337 | 0.24549 | 0.22415 | 0.0040 | 0.0300 |
| m05_joint_td_turf_dual | candidate | 356 | 0.68432 | 0.68337 | 0.24561 | 0.22421 | 0.0072 | 0.0300 |
| m05_joint_td_contextual | candidate | 356 | 0.68478 | 0.68337 | 0.24583 | 0.22410 | 0.0090 | 0.0300 |
| m05_joint_td_turf_asym | candidate | 356 | 0.68489 | 0.68337 | 0.24589 | 0.22403 | 0.0143 | 0.0300 |
| m12_hybrid_td_raw | control | 356 | 0.68520 | 0.68337 | 0.24604 | 0.22447 | 0.0087 | 0.0300 |
| m12_joint_hybrid_contextual | candidate | 356 | 0.68591 | 0.68337 | 0.24639 | 0.22446 | 0.0219 | 0.0300 |

## Paired ΔLogLoss, fixture-clustered bootstrap (B = 10000)

Negative Δ favours the left arm. 95% percentile interval.

| scope | cut | left | right | n_obs | n_fixtures | delta | lo | hi | p_negative | significant |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | all fixtures | m05_joint_td_turf_asym | m05_joint_td_raw | 2899 | 627 | -0.00009 | -0.00082 | +0.00065 | 0.592 | false |
| all | excluding T003 | m05_joint_td_turf_asym | m05_joint_td_raw | 2889 | 625 | -0.00003 | -0.00075 | +0.00072 | 0.536 | false |
| all | turf home | m05_joint_td_turf_asym | m05_joint_td_raw | 1828 | 399 | -0.00019 | -0.00115 | +0.00079 | 0.648 | false |
| all | grass home | m05_joint_td_turf_asym | m05_joint_td_raw | 1071 | 228 | +0.00007 | -0.00100 | +0.00117 | 0.448 | false |
| all | grass visitor at turf | m05_joint_td_turf_asym | m05_joint_td_raw | 678 | 149 | -0.00015 | -0.00246 | +0.00221 | 0.551 | false |
| all | midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 227 | 43 | -0.00146 | -0.00445 | +0.00142 | 0.834 | false |
| all | not midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 2672 | 584 | +0.00002 | -0.00073 | +0.00076 | 0.471 | false |
| all | all fixtures | m05_joint_td_turf_dual | m05_joint_td_raw | 2899 | 627 | +0.00001 | -0.00082 | +0.00083 | 0.486 | false |
| all | excluding T003 | m05_joint_td_turf_dual | m05_joint_td_raw | 2889 | 625 | +0.00007 | -0.00076 | +0.00091 | 0.434 | false |
| all | turf home | m05_joint_td_turf_dual | m05_joint_td_raw | 1828 | 399 | -0.00007 | -0.00109 | +0.00098 | 0.549 | false |
| all | grass home | m05_joint_td_turf_dual | m05_joint_td_raw | 1071 | 228 | +0.00015 | -0.00128 | +0.00157 | 0.410 | false |
| all | grass visitor at turf | m05_joint_td_turf_dual | m05_joint_td_raw | 678 | 149 | -0.00010 | -0.00210 | +0.00195 | 0.540 | false |
| all | midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 227 | 43 | -0.00270 | -0.00649 | +0.00098 | 0.921 | false |
| all | not midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 2672 | 584 | +0.00024 | -0.00062 | +0.00110 | 0.289 | false |
| all | all fixtures | m05_joint_td_contextual | m05_joint_td_raw | 2899 | 627 | +0.00001 | -0.00094 | +0.00096 | 0.495 | false |
| all | excluding T003 | m05_joint_td_contextual | m05_joint_td_raw | 2889 | 625 | +0.00007 | -0.00090 | +0.00104 | 0.450 | false |
| all | turf home | m05_joint_td_contextual | m05_joint_td_raw | 1828 | 399 | +0.00028 | -0.00092 | +0.00151 | 0.325 | false |
| all | grass home | m05_joint_td_contextual | m05_joint_td_raw | 1071 | 228 | -0.00047 | -0.00199 | +0.00101 | 0.713 | false |
| all | grass visitor at turf | m05_joint_td_contextual | m05_joint_td_raw | 678 | 149 | +0.00041 | -0.00204 | +0.00286 | 0.373 | false |
| all | midweek | m05_joint_td_contextual | m05_joint_td_raw | 227 | 43 | -0.00193 | -0.00718 | +0.00310 | 0.762 | false |
| all | not midweek | m05_joint_td_contextual | m05_joint_td_raw | 2672 | 584 | +0.00017 | -0.00078 | +0.00111 | 0.358 | false |
| all | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2899 | 627 | -0.00027 | -0.00092 | +0.00038 | 0.790 | false |
| all | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2889 | 625 | -0.00021 | -0.00086 | +0.00043 | 0.740 | false |
| all | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1828 | 399 | -0.00014 | -0.00085 | +0.00057 | 0.645 | false |
| all | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1071 | 228 | -0.00048 | -0.00174 | +0.00075 | 0.773 | false |
| all | grass visitor at turf | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 678 | 149 | +0.00001 | -0.00111 | +0.00110 | 0.495 | false |
| all | midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 227 | 43 | -0.00162 | -0.00501 | +0.00143 | 0.839 | false |
| all | not midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 2672 | 584 | -0.00015 | -0.00079 | +0.00051 | 0.674 | false |
| all | all fixtures | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 2899 | 627 | +0.00004 | -0.00094 | +0.00103 | 0.464 | false |
| all | excluding T003 | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 2889 | 625 | +0.00011 | -0.00089 | +0.00109 | 0.413 | false |
| all | turf home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 1828 | 399 | +0.00034 | -0.00097 | +0.00167 | 0.310 | false |
| all | grass home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 1071 | 228 | -0.00045 | -0.00182 | +0.00091 | 0.730 | false |
| all | grass visitor at turf | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 678 | 149 | +0.00063 | -0.00207 | +0.00338 | 0.329 | false |
| all | midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 227 | 43 | -0.00212 | -0.00745 | +0.00301 | 0.777 | false |
| all | not midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 2672 | 584 | +0.00023 | -0.00075 | +0.00118 | 0.322 | false |
| all | all fixtures | m05_joint_td_turf_asym | betfair_close | 2899 | 627 | +0.00108 | -0.00516 | +0.00719 | 0.368 | false |
| all | all fixtures | m05_joint_td_turf_dual | betfair_close | 2899 | 627 | +0.00118 | -0.00509 | +0.00730 | 0.358 | false |
| all | all fixtures | m05_joint_td_contextual | betfair_close | 2899 | 627 | +0.00118 | -0.00509 | +0.00734 | 0.358 | false |
| all | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 2899 | 627 | +0.00090 | -0.00529 | +0.00697 | 0.391 | false |
| all | all fixtures | m12_joint_hybrid_contextual | betfair_close | 2899 | 627 | +0.00160 | -0.00514 | +0.00822 | 0.322 | false |
| all | all fixtures | m05_joint_td_raw | betfair_close | 2899 | 627 | +0.00117 | -0.00506 | +0.00733 | 0.359 | false |
| all | all fixtures | m12_hybrid_td_raw | betfair_close | 2899 | 627 | +0.00155 | -0.00518 | +0.00816 | 0.328 | false |
| 1X2 | all fixtures | m05_joint_td_turf_asym | m05_joint_td_raw | 1785 | 595 | -0.00023 | -0.00099 | +0.00053 | 0.717 | false |
| 1X2 | excluding T003 | m05_joint_td_turf_asym | m05_joint_td_raw | 1779 | 593 | -0.00016 | -0.00092 | +0.00060 | 0.666 | false |
| 1X2 | turf home | m05_joint_td_turf_asym | m05_joint_td_raw | 1140 | 380 | -0.00028 | -0.00129 | +0.00073 | 0.708 | false |
| 1X2 | grass home | m05_joint_td_turf_asym | m05_joint_td_raw | 645 | 215 | -0.00013 | -0.00126 | +0.00098 | 0.595 | false |
| 1X2 | grass visitor at turf | m05_joint_td_turf_asym | m05_joint_td_raw | 432 | 144 | -0.00052 | -0.00296 | +0.00185 | 0.664 | false |
| 1X2 | midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 123 | 41 | -0.00235 | -0.00551 | +0.00069 | 0.936 | false |
| 1X2 | not midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 1662 | 554 | -0.00007 | -0.00085 | +0.00070 | 0.564 | false |
| 1X2 | all fixtures | m05_joint_td_turf_dual | m05_joint_td_raw | 1785 | 595 | +0.00007 | -0.00080 | +0.00096 | 0.434 | false |
| 1X2 | excluding T003 | m05_joint_td_turf_dual | m05_joint_td_raw | 1779 | 593 | +0.00014 | -0.00074 | +0.00101 | 0.381 | false |
| 1X2 | turf home | m05_joint_td_turf_dual | m05_joint_td_raw | 1140 | 380 | +0.00015 | -0.00093 | +0.00121 | 0.392 | false |
| 1X2 | grass home | m05_joint_td_turf_dual | m05_joint_td_raw | 645 | 215 | -0.00005 | -0.00157 | +0.00143 | 0.534 | false |
| 1X2 | grass visitor at turf | m05_joint_td_turf_dual | m05_joint_td_raw | 432 | 144 | -0.00036 | -0.00222 | +0.00147 | 0.648 | false |
| 1X2 | midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 123 | 41 | -0.00411 | -0.00811 | -0.00002 | 0.976 | true |
| 1X2 | not midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 1662 | 554 | +0.00038 | -0.00048 | +0.00125 | 0.203 | false |
| 1X2 | all fixtures | m05_joint_td_contextual | m05_joint_td_raw | 1785 | 595 | -0.00010 | -0.00112 | +0.00092 | 0.580 | false |
| 1X2 | excluding T003 | m05_joint_td_contextual | m05_joint_td_raw | 1779 | 593 | -0.00004 | -0.00106 | +0.00097 | 0.538 | false |
| 1X2 | turf home | m05_joint_td_contextual | m05_joint_td_raw | 1140 | 380 | +0.00000 | -0.00127 | +0.00126 | 0.501 | false |
| 1X2 | grass home | m05_joint_td_contextual | m05_joint_td_raw | 645 | 215 | -0.00028 | -0.00196 | +0.00143 | 0.640 | false |
| 1X2 | grass visitor at turf | m05_joint_td_contextual | m05_joint_td_raw | 432 | 144 | -0.00086 | -0.00301 | +0.00130 | 0.784 | false |
| 1X2 | midweek | m05_joint_td_contextual | m05_joint_td_raw | 123 | 41 | -0.00148 | -0.00803 | +0.00544 | 0.676 | false |
| 1X2 | not midweek | m05_joint_td_contextual | m05_joint_td_raw | 1662 | 554 | +0.00000 | -0.00096 | +0.00095 | 0.498 | false |
| 1X2 | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1785 | 595 | -0.00006 | -0.00063 | +0.00052 | 0.575 | false |
| 1X2 | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1779 | 593 | +0.00001 | -0.00056 | +0.00056 | 0.495 | false |
| 1X2 | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1140 | 380 | +0.00005 | -0.00058 | +0.00069 | 0.441 | false |
| 1X2 | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 645 | 215 | -0.00025 | -0.00138 | +0.00086 | 0.664 | false |
| 1X2 | grass visitor at turf | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 432 | 144 | +0.00033 | -0.00066 | +0.00128 | 0.262 | false |
| 1X2 | midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 123 | 41 | -0.00148 | -0.00455 | +0.00156 | 0.829 | false |
| 1X2 | not midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 1662 | 554 | +0.00005 | -0.00053 | +0.00062 | 0.435 | false |
| 1X2 | all fixtures | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 1785 | 595 | -0.00003 | -0.00108 | +0.00099 | 0.526 | false |
| 1X2 | excluding T003 | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 1779 | 593 | +0.00004 | -0.00098 | +0.00105 | 0.473 | false |
| 1X2 | turf home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 1140 | 380 | +0.00013 | -0.00121 | +0.00147 | 0.430 | false |
| 1X2 | grass home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 645 | 215 | -0.00032 | -0.00187 | +0.00126 | 0.668 | false |
| 1X2 | grass visitor at turf | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 432 | 144 | -0.00045 | -0.00299 | +0.00207 | 0.639 | false |
| 1X2 | midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 123 | 41 | -0.00205 | -0.00854 | +0.00466 | 0.734 | false |
| 1X2 | not midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 1662 | 554 | +0.00012 | -0.00086 | +0.00108 | 0.404 | false |
| 1X2 | all fixtures | m05_joint_td_turf_asym | betfair_close | 1785 | 595 | +0.00276 | -0.00667 | +0.01209 | 0.284 | false |
| 1X2 | all fixtures | m05_joint_td_turf_dual | betfair_close | 1785 | 595 | +0.00306 | -0.00642 | +0.01245 | 0.263 | false |
| 1X2 | all fixtures | m05_joint_td_contextual | betfair_close | 1785 | 595 | +0.00289 | -0.00662 | +0.01231 | 0.277 | false |
| 1X2 | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 1785 | 595 | +0.00293 | -0.00652 | +0.01236 | 0.273 | false |
| 1X2 | all fixtures | m12_joint_hybrid_contextual | betfair_close | 1785 | 595 | +0.00321 | -0.00695 | +0.01316 | 0.266 | false |
| 1X2 | all fixtures | m05_joint_td_raw | betfair_close | 1785 | 595 | +0.00299 | -0.00651 | +0.01249 | 0.270 | false |
| 1X2 | all fixtures | m12_hybrid_td_raw | betfair_close | 1785 | 595 | +0.00324 | -0.00685 | +0.01315 | 0.263 | false |
| OU2.5 | all fixtures | m05_joint_td_turf_asym | m05_joint_td_raw | 758 | 379 | -0.00022 | -0.00192 | +0.00146 | 0.607 | false |
| OU2.5 | excluding T003 | m05_joint_td_turf_asym | m05_joint_td_raw | 756 | 378 | +0.00003 | -0.00160 | +0.00167 | 0.494 | false |
| OU2.5 | turf home | m05_joint_td_turf_asym | m05_joint_td_raw | 462 | 231 | +0.00002 | -0.00229 | +0.00231 | 0.500 | false |
| OU2.5 | grass home | m05_joint_td_turf_asym | m05_joint_td_raw | 296 | 148 | -0.00059 | -0.00314 | +0.00169 | 0.690 | false |
| OU2.5 | grass visitor at turf | m05_joint_td_turf_asym | m05_joint_td_raw | 176 | 88 | -0.00016 | -0.00564 | +0.00541 | 0.517 | false |
| OU2.5 | midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 70 | 35 | +0.00032 | -0.00447 | +0.00515 | 0.439 | false |
| OU2.5 | not midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 688 | 344 | -0.00027 | -0.00206 | +0.00158 | 0.616 | false |
| OU2.5 | all fixtures | m05_joint_td_turf_dual | m05_joint_td_raw | 758 | 379 | -0.00026 | -0.00217 | +0.00164 | 0.618 | false |
| OU2.5 | excluding T003 | m05_joint_td_turf_dual | m05_joint_td_raw | 756 | 378 | -0.00001 | -0.00186 | +0.00181 | 0.514 | false |
| OU2.5 | turf home | m05_joint_td_turf_dual | m05_joint_td_raw | 462 | 231 | -0.00040 | -0.00278 | +0.00198 | 0.641 | false |
| OU2.5 | grass home | m05_joint_td_turf_dual | m05_joint_td_raw | 296 | 148 | -0.00005 | -0.00319 | +0.00304 | 0.505 | false |
| OU2.5 | grass visitor at turf | m05_joint_td_turf_dual | m05_joint_td_raw | 176 | 88 | -0.00031 | -0.00552 | +0.00488 | 0.543 | false |
| OU2.5 | midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 70 | 35 | -0.00192 | -0.00825 | +0.00440 | 0.720 | false |
| OU2.5 | not midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 688 | 344 | -0.00009 | -0.00208 | +0.00193 | 0.531 | false |
| OU2.5 | all fixtures | m05_joint_td_contextual | m05_joint_td_raw | 758 | 379 | -0.00008 | -0.00231 | +0.00213 | 0.537 | false |
| OU2.5 | excluding T003 | m05_joint_td_contextual | m05_joint_td_raw | 756 | 378 | +0.00017 | -0.00197 | +0.00230 | 0.451 | false |
| OU2.5 | turf home | m05_joint_td_contextual | m05_joint_td_raw | 462 | 231 | +0.00115 | -0.00164 | +0.00401 | 0.215 | false |
| OU2.5 | grass home | m05_joint_td_contextual | m05_joint_td_raw | 296 | 148 | -0.00200 | -0.00548 | +0.00139 | 0.874 | false |
| OU2.5 | grass visitor at turf | m05_joint_td_contextual | m05_joint_td_raw | 176 | 88 | +0.00244 | -0.00325 | +0.00812 | 0.196 | false |
| OU2.5 | midweek | m05_joint_td_contextual | m05_joint_td_raw | 70 | 35 | -0.00364 | -0.01295 | +0.00543 | 0.781 | false |
| OU2.5 | not midweek | m05_joint_td_contextual | m05_joint_td_raw | 688 | 344 | +0.00028 | -0.00193 | +0.00251 | 0.399 | false |
| OU2.5 | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 758 | 379 | -0.00074 | -0.00227 | +0.00078 | 0.830 | false |
| OU2.5 | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 756 | 378 | -0.00052 | -0.00200 | +0.00096 | 0.758 | false |
| OU2.5 | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 462 | 231 | -0.00019 | -0.00198 | +0.00152 | 0.585 | false |
| OU2.5 | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 296 | 148 | -0.00160 | -0.00438 | +0.00112 | 0.872 | false |
| OU2.5 | grass visitor at turf | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 176 | 88 | -0.00125 | -0.00426 | +0.00172 | 0.792 | false |
| OU2.5 | midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 70 | 35 | -0.00354 | -0.00916 | +0.00182 | 0.899 | false |
| OU2.5 | not midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 688 | 344 | -0.00046 | -0.00203 | +0.00110 | 0.712 | false |
| OU2.5 | all fixtures | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 758 | 379 | -0.00010 | -0.00234 | +0.00217 | 0.542 | false |
| OU2.5 | excluding T003 | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 756 | 378 | +0.00015 | -0.00202 | +0.00231 | 0.460 | false |
| OU2.5 | turf home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 462 | 231 | +0.00098 | -0.00199 | +0.00401 | 0.263 | false |
| OU2.5 | grass home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 296 | 148 | -0.00178 | -0.00509 | +0.00141 | 0.860 | false |
| OU2.5 | grass visitor at turf | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 176 | 88 | +0.00246 | -0.00349 | +0.00847 | 0.212 | false |
| OU2.5 | midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 70 | 35 | -0.00326 | -0.01292 | +0.00629 | 0.745 | false |
| OU2.5 | not midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 688 | 344 | +0.00023 | -0.00201 | +0.00251 | 0.417 | false |
| OU2.5 | all fixtures | m05_joint_td_turf_asym | betfair_close | 758 | 379 | -0.00309 | -0.01416 | +0.00773 | 0.706 | false |
| OU2.5 | all fixtures | m05_joint_td_turf_dual | betfair_close | 758 | 379 | -0.00314 | -0.01420 | +0.00764 | 0.709 | false |
| OU2.5 | all fixtures | m05_joint_td_contextual | betfair_close | 758 | 379 | -0.00296 | -0.01415 | +0.00797 | 0.698 | false |
| OU2.5 | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 758 | 379 | -0.00362 | -0.01441 | +0.00694 | 0.742 | false |
| OU2.5 | all fixtures | m12_joint_hybrid_contextual | betfair_close | 758 | 379 | -0.00265 | -0.01340 | +0.00770 | 0.688 | false |
| OU2.5 | all fixtures | m05_joint_td_raw | betfair_close | 758 | 379 | -0.00288 | -0.01398 | +0.00801 | 0.692 | false |
| OU2.5 | all fixtures | m12_hybrid_td_raw | betfair_close | 758 | 379 | -0.00255 | -0.01319 | +0.00790 | 0.679 | false |
| BTTS | all fixtures | m05_joint_td_turf_asym | m05_joint_td_raw | 356 | 178 | +0.00085 | -0.00099 | +0.00272 | 0.187 | false |
| BTTS | excluding T003 | m05_joint_td_turf_asym | m05_joint_td_raw | 354 | 177 | +0.00051 | -0.00127 | +0.00227 | 0.276 | false |
| BTTS | turf home | m05_joint_td_turf_asym | m05_joint_td_raw | 226 | 113 | -0.00015 | -0.00251 | +0.00222 | 0.541 | false |
| BTTS | grass home | m05_joint_td_turf_asym | m05_joint_td_raw | 130 | 65 | +0.00258 | -0.00018 | +0.00560 | 0.032 | false |
| BTTS | grass visitor at turf | m05_joint_td_turf_asym | m05_joint_td_raw | 70 | 35 | +0.00215 | -0.00375 | +0.00805 | 0.242 | false |
| BTTS | midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 34 | 17 | -0.00189 | -0.00755 | +0.00411 | 0.742 | false |
| BTTS | not midweek | m05_joint_td_turf_asym | m05_joint_td_raw | 322 | 161 | +0.00114 | -0.00082 | +0.00315 | 0.127 | false |
| BTTS | all fixtures | m05_joint_td_turf_dual | m05_joint_td_raw | 356 | 178 | +0.00027 | -0.00182 | +0.00240 | 0.406 | false |
| BTTS | excluding T003 | m05_joint_td_turf_dual | m05_joint_td_raw | 354 | 177 | -0.00007 | -0.00210 | +0.00192 | 0.522 | false |
| BTTS | turf home | m05_joint_td_turf_dual | m05_joint_td_raw | 226 | 113 | -0.00049 | -0.00310 | +0.00216 | 0.635 | false |
| BTTS | grass home | m05_joint_td_turf_dual | m05_joint_td_raw | 130 | 65 | +0.00159 | -0.00181 | +0.00525 | 0.194 | false |
| BTTS | grass visitor at turf | m05_joint_td_turf_dual | m05_joint_td_raw | 70 | 35 | +0.00205 | -0.00344 | +0.00786 | 0.237 | false |
| BTTS | midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 34 | 17 | +0.00082 | -0.00617 | +0.00770 | 0.410 | false |
| BTTS | not midweek | m05_joint_td_turf_dual | m05_joint_td_raw | 322 | 161 | +0.00021 | -0.00204 | +0.00247 | 0.421 | false |
| BTTS | all fixtures | m05_joint_td_contextual | m05_joint_td_raw | 356 | 178 | +0.00073 | -0.00167 | +0.00313 | 0.272 | false |
| BTTS | excluding T003 | m05_joint_td_contextual | m05_joint_td_raw | 354 | 177 | +0.00039 | -0.00196 | +0.00272 | 0.365 | false |
| BTTS | turf home | m05_joint_td_contextual | m05_joint_td_raw | 226 | 113 | -0.00007 | -0.00325 | +0.00315 | 0.516 | false |
| BTTS | grass home | m05_joint_td_contextual | m05_joint_td_raw | 130 | 65 | +0.00212 | -0.00132 | +0.00573 | 0.117 | false |
| BTTS | grass visitor at turf | m05_joint_td_contextual | m05_joint_td_raw | 70 | 35 | +0.00315 | -0.00305 | +0.00930 | 0.160 | false |
| BTTS | midweek | m05_joint_td_contextual | m05_joint_td_raw | 34 | 17 | -0.00006 | -0.01021 | +0.00998 | 0.498 | false |
| BTTS | not midweek | m05_joint_td_contextual | m05_joint_td_raw | 322 | 161 | +0.00081 | -0.00165 | +0.00326 | 0.256 | false |
| BTTS | all fixtures | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 356 | 178 | -0.00031 | -0.00217 | +0.00159 | 0.622 | false |
| BTTS | excluding T003 | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 354 | 177 | -0.00062 | -0.00239 | +0.00115 | 0.748 | false |
| BTTS | turf home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 226 | 113 | -0.00099 | -0.00335 | +0.00118 | 0.810 | false |
| BTTS | grass home | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 130 | 65 | +0.00087 | -0.00240 | +0.00430 | 0.309 | false |
| BTTS | grass visitor at turf | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 70 | 35 | +0.00119 | -0.00164 | +0.00405 | 0.204 | false |
| BTTS | midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 34 | 17 | +0.00185 | -0.00180 | +0.00558 | 0.169 | false |
| BTTS | not midweek | m05_joint_production_wealth_hier_ha | m05_joint_td_raw | 322 | 161 | -0.00054 | -0.00259 | +0.00145 | 0.701 | false |
| BTTS | all fixtures | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 356 | 178 | +0.00072 | -0.00184 | +0.00322 | 0.284 | false |
| BTTS | excluding T003 | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 354 | 177 | +0.00039 | -0.00210 | +0.00280 | 0.377 | false |
| BTTS | turf home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 226 | 113 | +0.00005 | -0.00341 | +0.00352 | 0.483 | false |
| BTTS | grass home | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 130 | 65 | +0.00188 | -0.00136 | +0.00538 | 0.132 | false |
| BTTS | grass visitor at turf | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 70 | 35 | +0.00264 | -0.00439 | +0.00969 | 0.237 | false |
| BTTS | midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 34 | 17 | -0.00002 | -0.01072 | +0.01088 | 0.503 | false |
| BTTS | not midweek | m12_joint_hybrid_contextual | m12_hybrid_td_raw | 322 | 161 | +0.00080 | -0.00180 | +0.00338 | 0.266 | false |
| BTTS | all fixtures | m05_joint_td_turf_asym | betfair_close | 356 | 178 | +0.00152 | -0.00809 | +0.01102 | 0.370 | false |
| BTTS | all fixtures | m05_joint_td_turf_dual | betfair_close | 356 | 178 | +0.00095 | -0.00874 | +0.01057 | 0.419 | false |
| BTTS | all fixtures | m05_joint_td_contextual | betfair_close | 356 | 178 | +0.00141 | -0.00829 | +0.01095 | 0.381 | false |
| BTTS | all fixtures | m05_joint_production_wealth_hier_ha | betfair_close | 356 | 178 | +0.00036 | -0.00945 | +0.00997 | 0.467 | false |
| BTTS | all fixtures | m12_joint_hybrid_contextual | betfair_close | 356 | 178 | +0.00254 | -0.00742 | +0.01250 | 0.304 | false |
| BTTS | all fixtures | m05_joint_td_raw | betfair_close | 356 | 178 | +0.00067 | -0.00906 | +0.01027 | 0.444 | false |
| BTTS | all fixtures | m12_hybrid_td_raw | betfair_close | 356 | 178 | +0.00183 | -0.00812 | +0.01180 | 0.355 | false |

## Coefficients vs prior

| model | fold | site | mean | sd | q05 | q95 | p_positive | prior_p_positive | contraction | h_pass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| m05_joint_td_turf_asym | 20 | turf_asym.w | 0.0516 | 0.0413 | -0.0149 | 0.1187 | 0.897 | 0.841 | 0.174 | false |
| m05_joint_td_turf_asym | 20 | ha.γ_base | 0.1326 | 0.0307 | 0.0824 | 0.1840 | 1.000 | 0.999 | 0.387 | — |
| m05_joint_td_turf_asym | 20 | ha.σ_γ | 0.0365 | 0.0261 | 0.0025 | 0.0831 | 1.000 | 1.000 | 0.133 | — |
| m05_joint_td_turf_asym | 40 | turf_asym.w | 0.0344 | 0.0412 | -0.0331 | 0.1043 | 0.800 | 0.841 | 0.177 | false |
| m05_joint_td_turf_asym | 40 | ha.γ_base | 0.1249 | 0.0303 | 0.0755 | 0.1751 | 1.000 | 0.999 | 0.393 | — |
| m05_joint_td_turf_asym | 40 | ha.σ_γ | 0.0308 | 0.0230 | 0.0024 | 0.0741 | 1.000 | 1.000 | 0.236 | — |
| m05_joint_td_turf_dual | 20 | turf_asym.w | 0.0520 | 0.0404 | -0.0143 | 0.1190 | 0.899 | 0.841 | 0.191 | false |
| m05_joint_td_turf_dual | 20 | turf_gen.w | -0.0304 | 0.0388 | -0.0942 | 0.0349 | 0.218 | 0.500 | 0.225 | — |
| m05_joint_td_turf_dual | 20 | turf_pace.w | 0.0228 | 0.0356 | -0.0340 | 0.0821 | 0.742 | 0.500 | 0.288 | false |
| m05_joint_td_turf_dual | 20 | ha.γ_base | 0.1458 | 0.0328 | 0.0916 | 0.1997 | 1.000 | 0.999 | 0.343 | — |
| m05_joint_td_turf_dual | 20 | ha.σ_γ | 0.0376 | 0.0272 | 0.0032 | 0.0888 | 1.000 | 1.000 | 0.097 | — |
| m05_joint_td_turf_dual | 40 | turf_asym.w | 0.0446 | 0.0424 | -0.0222 | 0.1163 | 0.860 | 0.841 | 0.152 | false |
| m05_joint_td_turf_dual | 40 | turf_gen.w | -0.0334 | 0.0373 | -0.0960 | 0.0290 | 0.183 | 0.500 | 0.253 | — |
| m05_joint_td_turf_dual | 40 | turf_pace.w | -0.0111 | 0.0350 | -0.0730 | 0.0453 | 0.384 | 0.500 | 0.300 | false |
| m05_joint_td_turf_dual | 40 | ha.γ_base | 0.1362 | 0.0336 | 0.0807 | 0.1930 | 1.000 | 0.999 | 0.328 | — |
| m05_joint_td_turf_dual | 40 | ha.σ_γ | 0.0303 | 0.0221 | 0.0029 | 0.0725 | 1.000 | 1.000 | 0.268 | — |
| m05_joint_td_contextual | 20 | turf_asym.w | 0.0529 | 0.0404 | -0.0140 | 0.1178 | 0.896 | 0.841 | 0.192 | false |
| m05_joint_td_contextual | 20 | turf_gen.w | -0.0316 | 0.0386 | -0.0960 | 0.0311 | 0.203 | 0.500 | 0.227 | — |
| m05_joint_td_contextual | 20 | turf_pace.w | 0.0234 | 0.0365 | -0.0363 | 0.0836 | 0.740 | 0.500 | 0.271 | false |
| m05_joint_td_contextual | 20 | midweek.w | 0.0331 | 0.0446 | -0.0388 | 0.1060 | 0.770 | 0.841 | 0.109 | false |
| m05_joint_td_contextual | 20 | rest_diff.w | -0.0005 | 0.0094 | -0.0156 | 0.0149 | 0.480 | 0.841 | 0.530 | — |
| m05_joint_td_contextual | 20 | ha.γ_base | 0.1437 | 0.0336 | 0.0896 | 0.1973 | 1.000 | 0.999 | 0.329 | — |
| m05_joint_td_contextual | 20 | ha.σ_γ | 0.0374 | 0.0269 | 0.0034 | 0.0868 | 1.000 | 1.000 | 0.108 | — |
| m05_joint_td_contextual | 40 | turf_asym.w | 0.0457 | 0.0410 | -0.0204 | 0.1146 | 0.864 | 0.841 | 0.181 | false |
| m05_joint_td_contextual | 40 | turf_gen.w | -0.0343 | 0.0383 | -0.0949 | 0.0284 | 0.185 | 0.500 | 0.234 | — |
| m05_joint_td_contextual | 40 | turf_pace.w | -0.0107 | 0.0364 | -0.0718 | 0.0499 | 0.374 | 0.500 | 0.273 | false |
| m05_joint_td_contextual | 40 | midweek.w | 0.0130 | 0.0423 | -0.0565 | 0.0803 | 0.623 | 0.841 | 0.154 | false |
| m05_joint_td_contextual | 40 | rest_diff.w | 0.0061 | 0.0092 | -0.0092 | 0.0211 | 0.739 | 0.841 | 0.538 | — |
| m05_joint_td_contextual | 40 | ha.γ_base | 0.1358 | 0.0341 | 0.0799 | 0.1913 | 1.000 | 0.999 | 0.317 | — |
| m05_joint_td_contextual | 40 | ha.σ_γ | 0.0304 | 0.0225 | 0.0024 | 0.0722 | 1.000 | 1.000 | 0.254 | — |
| m12_joint_hybrid_contextual | 20 | turf_asym.w | 0.0534 | 0.0414 | -0.0140 | 0.1236 | 0.907 | 0.841 | 0.172 | true |
| m12_joint_hybrid_contextual | 20 | turf_gen.w | -0.0347 | 0.0390 | -0.0966 | 0.0297 | 0.188 | 0.500 | 0.219 | — |
| m12_joint_hybrid_contextual | 20 | turf_pace.w | 0.0245 | 0.0352 | -0.0315 | 0.0837 | 0.749 | 0.500 | 0.295 | false |
| m12_joint_hybrid_contextual | 20 | midweek.w | 0.0311 | 0.0453 | -0.0422 | 0.1056 | 0.761 | 0.841 | 0.094 | false |
| m12_joint_hybrid_contextual | 20 | rest_diff.w | -0.0003 | 0.0096 | -0.0163 | 0.0154 | 0.489 | 0.841 | 0.520 | — |
| m12_joint_hybrid_contextual | 20 | ha.γ_base | 0.1433 | 0.0343 | 0.0845 | 0.2006 | 1.000 | 0.999 | 0.315 | — |
| m12_joint_hybrid_contextual | 20 | ha.σ_γ | 0.0392 | 0.0266 | 0.0035 | 0.0883 | 1.000 | 1.000 | 0.119 | — |
| m12_joint_hybrid_contextual | 40 | turf_asym.w | 0.0488 | 0.0416 | -0.0216 | 0.1171 | 0.883 | 0.841 | 0.169 | false |
| m12_joint_hybrid_contextual | 40 | turf_gen.w | -0.0279 | 0.0393 | -0.0950 | 0.0352 | 0.241 | 0.500 | 0.214 | — |
| m12_joint_hybrid_contextual | 40 | turf_pace.w | -0.0119 | 0.0342 | -0.0698 | 0.0451 | 0.368 | 0.500 | 0.316 | false |
| m12_joint_hybrid_contextual | 40 | midweek.w | 0.0135 | 0.0440 | -0.0598 | 0.0831 | 0.633 | 0.841 | 0.119 | false |
| m12_joint_hybrid_contextual | 40 | rest_diff.w | 0.0072 | 0.0095 | -0.0084 | 0.0232 | 0.764 | 0.841 | 0.524 | — |
| m12_joint_hybrid_contextual | 40 | ha.γ_base | 0.1333 | 0.0340 | 0.0759 | 0.1886 | 1.000 | 0.999 | 0.320 | — |
| m12_joint_hybrid_contextual | 40 | ha.σ_γ | 0.0309 | 0.0230 | 0.0024 | 0.0748 | 1.000 | 1.000 | 0.236 | — |

## Raw panel goals by surface × timing

| turf_home | midweek | n | mean_total | mean_home | mean_away | mean_goal_diff |
|---:|---:|---:|---:|---:|---:|---:|
| false | false | 223 | 2.668 | 1.444 | 1.224 | 0.220 |
| false | true | 25 | 3.120 | 1.800 | 1.320 | 0.480 |
| true | false | 434 | 2.705 | 1.459 | 1.247 | 0.212 |
| true | true | 28 | 2.964 | 1.250 | 1.714 | -0.464 |
