# r04 proper scores — Task 013

Generated 2026-09-11 22:52. Panel: 710 fixtures (24/25 + 25/26 walk-forward). Book: de-vigged Betfair TWA(−20, 0] close. Control reproduction: `m12_hybrid_td_raw` LogLoss 0.64337 / ECE 0.0100 vs published 0.64337 / 0.01.

## Scope: all

| model | dynamics | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_td_raw | TimeDecay(180) | 2899 | 0.64299 | 0.64182 | 0.22586 | 0.22529 | 0.22415 | 0.21110 | 0.0149 | 0.0139 |
| m05_wealth_grw | MultiScaleGRW | 2899 | 0.64315 | 0.64182 | 0.22603 | 0.22529 | 0.22383 | 0.21110 | 0.0123 | 0.0139 |
| m05_joint_grw_raw | MultiScaleGRW | 2899 | 0.64316 | 0.64182 | 0.22603 | 0.22529 | 0.22385 | 0.21110 | 0.0117 | 0.0139 |
| m12_hybrid_td_raw | TimeDecay(180) | 2899 | 0.64337 | 0.64182 | 0.22605 | 0.22529 | 0.22447 | 0.21110 | 0.0100 | 0.0139 |
| m00_baseline_grw | MultiScaleGRW | 2899 | 0.64433 | 0.64182 | 0.22654 | 0.22529 | 0.22492 | 0.21110 | 0.0184 | 0.0139 |
| m12_joint_hybrid_synergy_grw | MultiScaleGRW | 2899 | 0.64437 | 0.64182 | 0.22659 | 0.22529 | 0.22493 | 0.21110 | 0.0086 | 0.0139 |
| m10_lineup_grw | MultiScaleGRW | 2899 | 0.64561 | 0.64182 | 0.22713 | 0.22529 | 0.22631 | 0.21110 | 0.0092 | 0.0139 |

## Scope: 1X2

| model | dynamics | n_obs | logloss | market_logloss | brier | market_brier | rps | market_rps | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m05_wealth_grw | MultiScaleGRW | 1785 | 0.61558 | 0.61312 | 0.21286 | 0.21156 | 0.22383 | 0.21110 | 0.0124 | 0.0186 |
| m05_joint_grw_raw | MultiScaleGRW | 1785 | 0.61561 | 0.61312 | 0.21287 | 0.21156 | 0.22385 | 0.21110 | 0.0115 | 0.0186 |
| m05_joint_td_raw | TimeDecay(180) | 1785 | 0.61611 | 0.61312 | 0.21299 | 0.21156 | 0.22415 | 0.21110 | 0.0216 | 0.0186 |
| m12_hybrid_td_raw | TimeDecay(180) | 1785 | 0.61636 | 0.61312 | 0.21311 | 0.21156 | 0.22447 | 0.21110 | 0.0155 | 0.0186 |
| m00_baseline_grw | MultiScaleGRW | 1785 | 0.61666 | 0.61312 | 0.21329 | 0.21156 | 0.22492 | 0.21110 | 0.0189 | 0.0186 |
| m12_joint_hybrid_synergy_grw | MultiScaleGRW | 1785 | 0.61701 | 0.61312 | 0.21350 | 0.21156 | 0.22493 | 0.21110 | 0.0140 | 0.0186 |
| m10_lineup_grw | MultiScaleGRW | 1785 | 0.61867 | 0.61312 | 0.21421 | 0.21156 | 0.22631 | 0.21110 | 0.0108 | 0.0186 |

## Scope: OU2.5

| model | dynamics | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_td_raw | TimeDecay(180) | 758 | 0.68700 | 0.68988 | 0.24695 | 0.24829 | 0.0100 | 0.0183 |
| m12_hybrid_td_raw | TimeDecay(180) | 758 | 0.68732 | 0.68988 | 0.24711 | 0.24829 | 0.0100 | 0.0183 |
| m00_baseline_grw | MultiScaleGRW | 758 | 0.68755 | 0.68988 | 0.24721 | 0.24829 | 0.0222 | 0.0183 |
| m10_lineup_grw | MultiScaleGRW | 758 | 0.68760 | 0.68988 | 0.24725 | 0.24829 | 0.0109 | 0.0183 |
| m12_joint_hybrid_synergy_grw | MultiScaleGRW | 758 | 0.68973 | 0.68988 | 0.24833 | 0.24829 | 0.0133 | 0.0183 |
| m05_wealth_grw | MultiScaleGRW | 758 | 0.68986 | 0.68988 | 0.24839 | 0.24829 | 0.0311 | 0.0183 |
| m05_joint_grw_raw | MultiScaleGRW | 758 | 0.68990 | 0.68988 | 0.24840 | 0.24829 | 0.0285 | 0.0183 |

## Scope: BTTS

| model | dynamics | n_obs | logloss | market_logloss | brier | market_brier | ece | market_ece |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m05_joint_grw_raw | MultiScaleGRW | 356 | 0.68179 | 0.68337 | 0.24439 | 0.24516 | 0.0266 | 0.0300 |
| m05_wealth_grw | MultiScaleGRW | 356 | 0.68192 | 0.68337 | 0.24445 | 0.24516 | 0.0366 | 0.0300 |
| m05_joint_td_raw | TimeDecay(180) | 356 | 0.68405 | 0.68337 | 0.24549 | 0.24516 | 0.0040 | 0.0300 |
| m12_joint_hybrid_synergy_grw | MultiScaleGRW | 356 | 0.68496 | 0.68337 | 0.24591 | 0.24516 | 0.0402 | 0.0300 |
| m12_hybrid_td_raw | TimeDecay(180) | 356 | 0.68520 | 0.68337 | 0.24604 | 0.24516 | 0.0087 | 0.0300 |
| m00_baseline_grw | MultiScaleGRW | 356 | 0.69108 | 0.68337 | 0.24897 | 0.24516 | 0.0353 | 0.0300 |
| m10_lineup_grw | MultiScaleGRW | 356 | 0.69125 | 0.68337 | 0.24906 | 0.24516 | 0.0205 | 0.0300 |

## Paired ΔLogLoss, fixture-clustered bootstrap (B = 10000)

Negative Δ favours the left arm. 95% percentile interval.

| scope | left | right | n_obs | n_fixtures | delta | lo | hi | p_negative | significant |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | m00_baseline_grw | betfair_close | 2899 | 627 | +0.00252 | -0.00469 | +0.00969 | 0.240 | false |
| all | m05_wealth_grw | betfair_close | 2899 | 627 | +0.00134 | -0.00495 | +0.00741 | 0.329 | false |
| all | m10_lineup_grw | betfair_close | 2899 | 627 | +0.00379 | -0.00345 | +0.01109 | 0.154 | false |
| all | m12_joint_hybrid_synergy_grw | betfair_close | 2899 | 627 | +0.00255 | -0.00396 | +0.00894 | 0.218 | false |
| all | m05_joint_td_raw | betfair_close | 2899 | 627 | +0.00117 | -0.00519 | +0.00748 | 0.361 | false |
| all | m12_hybrid_td_raw | betfair_close | 2899 | 627 | +0.00155 | -0.00527 | +0.00825 | 0.328 | false |
| all | m05_joint_grw_raw | betfair_close | 2899 | 627 | +0.00134 | -0.00494 | +0.00741 | 0.328 | false |
| all | m12_joint_hybrid_synergy_grw | m12_hybrid_td_raw | 2899 | 627 | +0.00100 | -0.00299 | +0.00497 | 0.312 | false |
| all | m05_wealth_grw | m05_joint_td_raw | 2899 | 627 | +0.00016 | -0.00368 | +0.00403 | 0.457 | false |
| all | m12_joint_hybrid_synergy_grw | m05_wealth_grw | 2899 | 627 | +0.00122 | -0.00073 | +0.00322 | 0.119 | false |
| all | m10_lineup_grw | m00_baseline_grw | 2899 | 627 | +0.00127 | -0.00104 | +0.00357 | 0.137 | false |
| all | m05_wealth_grw | m05_joint_grw_raw | 2899 | 627 | -0.00001 | -0.00024 | +0.00023 | 0.520 | false |
| all | m12_joint_hybrid_synergy_grw | m05_joint_grw_raw | 2899 | 627 | +0.00121 | -0.00078 | +0.00322 | 0.121 | false |
| 1X2 | m00_baseline_grw | betfair_close | 1785 | 595 | +0.00354 | -0.00654 | +0.01350 | 0.246 | false |
| 1X2 | m05_wealth_grw | betfair_close | 1785 | 595 | +0.00247 | -0.00593 | +0.01058 | 0.277 | false |
| 1X2 | m10_lineup_grw | betfair_close | 1785 | 595 | +0.00555 | -0.00513 | +0.01610 | 0.156 | false |
| 1X2 | m12_joint_hybrid_synergy_grw | betfair_close | 1785 | 595 | +0.00389 | -0.00502 | +0.01279 | 0.190 | false |
| 1X2 | m05_joint_td_raw | betfair_close | 1785 | 595 | +0.00299 | -0.00649 | +0.01221 | 0.261 | false |
| 1X2 | m12_hybrid_td_raw | betfair_close | 1785 | 595 | +0.00324 | -0.00677 | +0.01320 | 0.258 | false |
| 1X2 | m05_joint_grw_raw | betfair_close | 1785 | 595 | +0.00249 | -0.00586 | +0.01066 | 0.276 | false |
| 1X2 | m12_joint_hybrid_synergy_grw | m12_hybrid_td_raw | 1785 | 595 | +0.00065 | -0.00350 | +0.00484 | 0.385 | false |
| 1X2 | m05_wealth_grw | m05_joint_td_raw | 1785 | 595 | -0.00052 | -0.00480 | +0.00370 | 0.600 | false |
| 1X2 | m12_joint_hybrid_synergy_grw | m05_wealth_grw | 1785 | 595 | +0.00143 | -0.00139 | +0.00431 | 0.154 | false |
| 1X2 | m10_lineup_grw | m00_baseline_grw | 1785 | 595 | +0.00201 | -0.00119 | +0.00522 | 0.107 | false |
| 1X2 | m05_wealth_grw | m05_joint_grw_raw | 1785 | 595 | -0.00002 | -0.00027 | +0.00023 | 0.572 | false |
| 1X2 | m12_joint_hybrid_synergy_grw | m05_joint_grw_raw | 1785 | 595 | +0.00140 | -0.00141 | +0.00430 | 0.157 | false |
| OU2.5 | m00_baseline_grw | betfair_close | 758 | 379 | -0.00233 | -0.01617 | +0.01173 | 0.622 | false |
| OU2.5 | m05_wealth_grw | betfair_close | 758 | 379 | -0.00002 | -0.01330 | +0.01313 | 0.493 | false |
| OU2.5 | m10_lineup_grw | betfair_close | 758 | 379 | -0.00228 | -0.01501 | +0.01057 | 0.629 | false |
| OU2.5 | m12_joint_hybrid_synergy_grw | betfair_close | 758 | 379 | -0.00015 | -0.01278 | +0.01252 | 0.503 | false |
| OU2.5 | m05_joint_td_raw | betfair_close | 758 | 379 | -0.00288 | -0.01377 | +0.00797 | 0.692 | false |
| OU2.5 | m12_hybrid_td_raw | betfair_close | 758 | 379 | -0.00255 | -0.01307 | +0.00797 | 0.680 | false |
| OU2.5 | m05_joint_grw_raw | betfair_close | 758 | 379 | +0.00002 | -0.01322 | +0.01317 | 0.490 | false |
| OU2.5 | m12_joint_hybrid_synergy_grw | m12_hybrid_td_raw | 758 | 379 | +0.00240 | -0.00617 | +0.01114 | 0.296 | false |
| OU2.5 | m05_wealth_grw | m05_joint_td_raw | 758 | 379 | +0.00286 | -0.00493 | +0.01094 | 0.236 | false |
| OU2.5 | m12_joint_hybrid_synergy_grw | m05_wealth_grw | 758 | 379 | -0.00013 | -0.00357 | +0.00327 | 0.536 | false |
| OU2.5 | m10_lineup_grw | m00_baseline_grw | 758 | 379 | +0.00005 | -0.00365 | +0.00355 | 0.495 | false |
| OU2.5 | m05_wealth_grw | m05_joint_grw_raw | 758 | 379 | -0.00004 | -0.00058 | +0.00051 | 0.548 | false |
| OU2.5 | m12_joint_hybrid_synergy_grw | m05_joint_grw_raw | 758 | 379 | -0.00017 | -0.00367 | +0.00329 | 0.547 | false |
| BTTS | m00_baseline_grw | betfair_close | 356 | 178 | +0.00771 | -0.00691 | +0.02229 | 0.157 | false |
| BTTS | m05_wealth_grw | betfair_close | 356 | 178 | -0.00145 | -0.01447 | +0.01108 | 0.589 | false |
| BTTS | m10_lineup_grw | betfair_close | 356 | 178 | +0.00788 | -0.00488 | +0.02017 | 0.116 | false |
| BTTS | m12_joint_hybrid_synergy_grw | betfair_close | 356 | 178 | +0.00159 | -0.01117 | +0.01392 | 0.399 | false |
| BTTS | m05_joint_td_raw | betfair_close | 356 | 178 | +0.00067 | -0.00916 | +0.01056 | 0.452 | false |
| BTTS | m12_hybrid_td_raw | betfair_close | 356 | 178 | +0.00183 | -0.00811 | +0.01199 | 0.358 | false |
| BTTS | m05_joint_grw_raw | betfair_close | 356 | 178 | -0.00158 | -0.01457 | +0.01094 | 0.599 | false |
| BTTS | m12_joint_hybrid_synergy_grw | m12_hybrid_td_raw | 356 | 178 | -0.00024 | -0.01122 | +0.01087 | 0.529 | false |
| BTTS | m05_wealth_grw | m05_joint_td_raw | 356 | 178 | -0.00212 | -0.01142 | +0.00709 | 0.682 | false |
| BTTS | m12_joint_hybrid_synergy_grw | m05_wealth_grw | 356 | 178 | +0.00304 | -0.00047 | +0.00655 | 0.045 | false |
| BTTS | m10_lineup_grw | m00_baseline_grw | 356 | 178 | +0.00017 | -0.00423 | +0.00464 | 0.463 | false |
| BTTS | m05_wealth_grw | m05_joint_grw_raw | 356 | 178 | +0.00013 | -0.00047 | +0.00070 | 0.317 | false |
| BTTS | m12_joint_hybrid_synergy_grw | m05_joint_grw_raw | 356 | 178 | +0.00317 | -0.00039 | +0.00670 | 0.039 | false |
