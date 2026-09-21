# r03 draw-mixture benchmark — TODO 021

Generated 2026-09-21 16:10 at `77513ac4` on mcmc-beast. Panel: 622 buildable of 635 quoted of 710 walk-forward fixtures; 596 with an accepted market-rate inversion (supremacy slope). Bootstrap B = 1000.

Runs: `m01_poisson_grw_tight` 2b42d3bf-28d7-47ac-8706-88798c9031ac, `m02_poisson_grw_loose_var` 27c8f2f2-e130-44ba-a764-2585d39eb2de, `m03_poisson_grw_loose_tdist` dafbfe00-54c2-42a6-a87f-2c9de470bca2, `m04_poisson_grw_loose_fixed_spread` 514c5533-fea4-4786-8a58-30d8262733cf.

## Headline metrics

| label | rho | sup_slope | sup_r2 | capital_ge4_pct | capital_le18_pct | max_drawdown_pct | sharpe_ann | flat_roi_pct | total_return_pct | p_roi_positive | n_bets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m01_poisson_grw_tight | 0.00 | 0.3987 | 0.4400 | 38.3 | 3.9 | -20.3 | 1.6521 | 5.33 | 192.9 | 0.9820 | 1329 |
| m02_poisson_grw_loose_var@0.25 | 0.25 | 0.4212 | 0.4545 | 37.6 | 4.2 | -20.4 | 1.6156 | 5.99 | 184.4 | 0.9820 | 1324 |
| m02_poisson_grw_loose_var@0.50 | 0.50 | 0.4438 | 0.4638 | 36.6 | 4.4 | -19.4 | 1.5766 | 6.07 | 171.5 | 0.9860 | 1321 |
| m02_poisson_grw_loose_var@0.75 | 0.75 | 0.4658 | 0.4732 | 35.9 | 4.7 | -19.1 | 1.5158 | 6.06 | 163.1 | 0.9780 | 1314 |
| m02_poisson_grw_loose_var | 1.00 | 0.4882 | 0.4789 | 34.7 | 5.2 | -19.1 | 1.4543 | 6.20 | 152.8 | 0.9730 | 1328 |
| m03_poisson_grw_loose_tdist@0.25 | 0.25 | 0.3990 | 0.4414 | 38.4 | 3.8 | -20.7 | 1.6171 | 5.18 | 186.2 | 0.9820 | 1322 |
| m03_poisson_grw_loose_tdist@0.50 | 0.50 | 0.3995 | 0.4412 | 38.4 | 3.7 | -20.8 | 1.6691 | 4.87 | 194.0 | 0.9860 | 1326 |
| m03_poisson_grw_loose_tdist@0.75 | 0.75 | 0.3986 | 0.4406 | 38.6 | 3.8 | -21.2 | 1.5941 | 4.08 | 179.7 | 0.9870 | 1324 |
| m03_poisson_grw_loose_tdist | 1.00 | 0.3985 | 0.4391 | 38.6 | 3.7 | -22.0 | 1.5415 | 4.55 | 170.0 | 0.9790 | 1318 |
| m04_poisson_grw_loose_fixed_spread@0.25 | 0.25 | 0.4093 | 0.4203 | 37.1 | 4.2 | -20.6 | 1.7586 | 5.23 | 233.2 | 0.9920 | 1331 |
| m04_poisson_grw_loose_fixed_spread@0.50 | 0.50 | 0.4200 | 0.3941 | 35.3 | 4.5 | -21.5 | 1.7608 | 5.22 | 252.8 | 0.9900 | 1330 |
| m04_poisson_grw_loose_fixed_spread@0.75 | 0.75 | 0.4302 | 0.3662 | 33.7 | 5.1 | -24.0 | 1.8198 | 4.61 | 305.7 | 0.9920 | 1334 |
| m04_poisson_grw_loose_fixed_spread | 1.00 | 0.4408 | 0.3386 | 31.9 | 5.5 | -27.7 | 1.7470 | 4.80 | 323.6 | 0.9970 | 1342 |

## Favourite tail (1X2, market favourite side)

| label | band | n | p_market | p_model |
|---|---|---:|---:|---:|
| m01_poisson_grw_tight | [0.40, 0.50) | 259 | 0.4461 | 0.4138 |
| m01_poisson_grw_tight | [0.50, 0.60) | 149 | 0.5386 | 0.4519 |
| m01_poisson_grw_tight | [0.60, 0.70) | 46 | 0.6362 | 0.4843 |
| m01_poisson_grw_tight | [0.70, 1.00) | 18 | 0.7625 | 0.5502 |
| m02_poisson_grw_loose_var@0.25 | [0.40, 0.50) | 259 | 0.4461 | 0.4153 |
| m02_poisson_grw_loose_var@0.25 | [0.50, 0.60) | 149 | 0.5386 | 0.4551 |
| m02_poisson_grw_loose_var@0.25 | [0.60, 0.70) | 46 | 0.6362 | 0.4891 |
| m02_poisson_grw_loose_var@0.25 | [0.70, 1.00) | 18 | 0.7625 | 0.5562 |
| m02_poisson_grw_loose_var@0.50 | [0.40, 0.50) | 259 | 0.4461 | 0.4169 |
| m02_poisson_grw_loose_var@0.50 | [0.50, 0.60) | 149 | 0.5386 | 0.4585 |
| m02_poisson_grw_loose_var@0.50 | [0.60, 0.70) | 46 | 0.6362 | 0.4935 |
| m02_poisson_grw_loose_var@0.50 | [0.70, 1.00) | 18 | 0.7625 | 0.5631 |
| m02_poisson_grw_loose_var@0.75 | [0.40, 0.50) | 259 | 0.4461 | 0.4185 |
| m02_poisson_grw_loose_var@0.75 | [0.50, 0.60) | 149 | 0.5386 | 0.4617 |
| m02_poisson_grw_loose_var@0.75 | [0.60, 0.70) | 46 | 0.6362 | 0.4975 |
| m02_poisson_grw_loose_var@0.75 | [0.70, 1.00) | 18 | 0.7625 | 0.5684 |
| m02_poisson_grw_loose_var | [0.40, 0.50) | 259 | 0.4461 | 0.4200 |
| m02_poisson_grw_loose_var | [0.50, 0.60) | 149 | 0.5386 | 0.4651 |
| m02_poisson_grw_loose_var | [0.60, 0.70) | 46 | 0.6362 | 0.5019 |
| m02_poisson_grw_loose_var | [0.70, 1.00) | 18 | 0.7625 | 0.5748 |
| m03_poisson_grw_loose_tdist@0.25 | [0.40, 0.50) | 259 | 0.4461 | 0.4138 |
| m03_poisson_grw_loose_tdist@0.25 | [0.50, 0.60) | 149 | 0.5386 | 0.4515 |
| m03_poisson_grw_loose_tdist@0.25 | [0.60, 0.70) | 46 | 0.6362 | 0.4848 |
| m03_poisson_grw_loose_tdist@0.25 | [0.70, 1.00) | 18 | 0.7625 | 0.5505 |
| m03_poisson_grw_loose_tdist@0.50 | [0.40, 0.50) | 259 | 0.4461 | 0.4138 |
| m03_poisson_grw_loose_tdist@0.50 | [0.50, 0.60) | 149 | 0.5386 | 0.4517 |
| m03_poisson_grw_loose_tdist@0.50 | [0.60, 0.70) | 46 | 0.6362 | 0.4851 |
| m03_poisson_grw_loose_tdist@0.50 | [0.70, 1.00) | 18 | 0.7625 | 0.5503 |
| m03_poisson_grw_loose_tdist@0.75 | [0.40, 0.50) | 259 | 0.4461 | 0.4140 |
| m03_poisson_grw_loose_tdist@0.75 | [0.50, 0.60) | 149 | 0.5386 | 0.4511 |
| m03_poisson_grw_loose_tdist@0.75 | [0.60, 0.70) | 46 | 0.6362 | 0.4850 |
| m03_poisson_grw_loose_tdist@0.75 | [0.70, 1.00) | 18 | 0.7625 | 0.5503 |
| m03_poisson_grw_loose_tdist | [0.40, 0.50) | 259 | 0.4461 | 0.4139 |
| m03_poisson_grw_loose_tdist | [0.50, 0.60) | 149 | 0.5386 | 0.4511 |
| m03_poisson_grw_loose_tdist | [0.60, 0.70) | 46 | 0.6362 | 0.4852 |
| m03_poisson_grw_loose_tdist | [0.70, 1.00) | 18 | 0.7625 | 0.5499 |
| m04_poisson_grw_loose_fixed_spread@0.25 | [0.40, 0.50) | 259 | 0.4461 | 0.4144 |
| m04_poisson_grw_loose_fixed_spread@0.25 | [0.50, 0.60) | 149 | 0.5386 | 0.4526 |
| m04_poisson_grw_loose_fixed_spread@0.25 | [0.60, 0.70) | 46 | 0.6362 | 0.4859 |
| m04_poisson_grw_loose_fixed_spread@0.25 | [0.70, 1.00) | 18 | 0.7625 | 0.5528 |
| m04_poisson_grw_loose_fixed_spread@0.50 | [0.40, 0.50) | 259 | 0.4461 | 0.4149 |
| m04_poisson_grw_loose_fixed_spread@0.50 | [0.50, 0.60) | 149 | 0.5386 | 0.4538 |
| m04_poisson_grw_loose_fixed_spread@0.50 | [0.60, 0.70) | 46 | 0.6362 | 0.4875 |
| m04_poisson_grw_loose_fixed_spread@0.50 | [0.70, 1.00) | 18 | 0.7625 | 0.5552 |
| m04_poisson_grw_loose_fixed_spread@0.75 | [0.40, 0.50) | 259 | 0.4461 | 0.4157 |
| m04_poisson_grw_loose_fixed_spread@0.75 | [0.50, 0.60) | 149 | 0.5386 | 0.4544 |
| m04_poisson_grw_loose_fixed_spread@0.75 | [0.60, 0.70) | 46 | 0.6362 | 0.4889 |
| m04_poisson_grw_loose_fixed_spread@0.75 | [0.70, 1.00) | 18 | 0.7625 | 0.5576 |
| m04_poisson_grw_loose_fixed_spread | [0.40, 0.50) | 259 | 0.4461 | 0.4161 |
| m04_poisson_grw_loose_fixed_spread | [0.50, 0.60) | 149 | 0.5386 | 0.4555 |
| m04_poisson_grw_loose_fixed_spread | [0.60, 0.70) | 46 | 0.6362 | 0.4903 |
| m04_poisson_grw_loose_fixed_spread | [0.70, 1.00) | 18 | 0.7625 | 0.5602 |

## Derivative markets vs the close

| label | family | n_fixtures | logloss_model | logloss_close | headline | mean_p_model | mean_p_close |
|---|---|---:|---:|---:|---|---:|---:|
| m01_poisson_grw_tight | 1X2 | 595 | 1.06033 | 1.05344 | home | 0.4175 | 0.4215 |
| m01_poisson_grw_tight | OU2.5 | 379 | 0.68756 | 0.68988 | over_25 | 0.5131 | 0.5298 |
| m01_poisson_grw_tight | BTTS | 178 | 0.69079 | 0.68337 | btts_yes | 0.5376 | 0.5577 |
| m02_poisson_grw_loose_var@0.25 | 1X2 | 595 | 1.06097 | 1.05344 | home | 0.4176 | 0.4215 |
| m02_poisson_grw_loose_var@0.25 | OU2.5 | 379 | 0.68776 | 0.68988 | over_25 | 0.5132 | 0.5298 |
| m02_poisson_grw_loose_var@0.25 | BTTS | 178 | 0.69051 | 0.68337 | btts_yes | 0.5362 | 0.5577 |
| m02_poisson_grw_loose_var@0.50 | 1X2 | 595 | 1.06124 | 1.05344 | home | 0.4177 | 0.4215 |
| m02_poisson_grw_loose_var@0.50 | OU2.5 | 379 | 0.68786 | 0.68988 | over_25 | 0.5136 | 0.5298 |
| m02_poisson_grw_loose_var@0.50 | BTTS | 178 | 0.69054 | 0.68337 | btts_yes | 0.5352 | 0.5577 |
| m02_poisson_grw_loose_var@0.75 | 1X2 | 595 | 1.06198 | 1.05344 | home | 0.4177 | 0.4215 |
| m02_poisson_grw_loose_var@0.75 | OU2.5 | 379 | 0.68812 | 0.68988 | over_25 | 0.5137 | 0.5298 |
| m02_poisson_grw_loose_var@0.75 | BTTS | 178 | 0.69012 | 0.68337 | btts_yes | 0.5337 | 0.5577 |
| m02_poisson_grw_loose_var | 1X2 | 595 | 1.06279 | 1.05344 | home | 0.4178 | 0.4215 |
| m02_poisson_grw_loose_var | OU2.5 | 379 | 0.68836 | 0.68988 | over_25 | 0.5139 | 0.5298 |
| m02_poisson_grw_loose_var | BTTS | 178 | 0.69003 | 0.68337 | btts_yes | 0.5326 | 0.5577 |
| m03_poisson_grw_loose_tdist@0.25 | 1X2 | 595 | 1.06064 | 1.05344 | home | 0.4174 | 0.4215 |
| m03_poisson_grw_loose_tdist@0.25 | OU2.5 | 379 | 0.68776 | 0.68988 | over_25 | 0.5129 | 0.5298 |
| m03_poisson_grw_loose_tdist@0.25 | BTTS | 178 | 0.69101 | 0.68337 | btts_yes | 0.5372 | 0.5577 |
| m03_poisson_grw_loose_tdist@0.50 | 1X2 | 595 | 1.06057 | 1.05344 | home | 0.4175 | 0.4215 |
| m03_poisson_grw_loose_tdist@0.50 | OU2.5 | 379 | 0.68765 | 0.68988 | over_25 | 0.5129 | 0.5298 |
| m03_poisson_grw_loose_tdist@0.50 | BTTS | 178 | 0.69060 | 0.68337 | btts_yes | 0.5369 | 0.5577 |
| m03_poisson_grw_loose_tdist@0.75 | 1X2 | 595 | 1.06087 | 1.05344 | home | 0.4174 | 0.4215 |
| m03_poisson_grw_loose_tdist@0.75 | OU2.5 | 379 | 0.68789 | 0.68988 | over_25 | 0.5125 | 0.5298 |
| m03_poisson_grw_loose_tdist@0.75 | BTTS | 178 | 0.69106 | 0.68337 | btts_yes | 0.5365 | 0.5577 |
| m03_poisson_grw_loose_tdist | 1X2 | 595 | 1.06129 | 1.05344 | home | 0.4175 | 0.4215 |
| m03_poisson_grw_loose_tdist | OU2.5 | 379 | 0.68787 | 0.68988 | over_25 | 0.5124 | 0.5298 |
| m03_poisson_grw_loose_tdist | BTTS | 178 | 0.69079 | 0.68337 | btts_yes | 0.5361 | 0.5577 |
| m04_poisson_grw_loose_fixed_spread@0.25 | 1X2 | 595 | 1.06059 | 1.05344 | home | 0.4170 | 0.4215 |
| m04_poisson_grw_loose_fixed_spread@0.25 | OU2.5 | 379 | 0.68748 | 0.68988 | over_25 | 0.5112 | 0.5298 |
| m04_poisson_grw_loose_fixed_spread@0.25 | BTTS | 178 | 0.69122 | 0.68337 | btts_yes | 0.5351 | 0.5577 |
| m04_poisson_grw_loose_fixed_spread@0.50 | 1X2 | 595 | 1.06109 | 1.05344 | home | 0.4166 | 0.4215 |
| m04_poisson_grw_loose_fixed_spread@0.50 | OU2.5 | 379 | 0.68799 | 0.68988 | over_25 | 0.5097 | 0.5298 |
| m04_poisson_grw_loose_fixed_spread@0.50 | BTTS | 178 | 0.69194 | 0.68337 | btts_yes | 0.5327 | 0.5577 |
| m04_poisson_grw_loose_fixed_spread@0.75 | 1X2 | 595 | 1.06213 | 1.05344 | home | 0.4162 | 0.4215 |
| m04_poisson_grw_loose_fixed_spread@0.75 | OU2.5 | 379 | 0.68833 | 0.68988 | over_25 | 0.5079 | 0.5298 |
| m04_poisson_grw_loose_fixed_spread@0.75 | BTTS | 178 | 0.69262 | 0.68337 | btts_yes | 0.5301 | 0.5577 |
| m04_poisson_grw_loose_fixed_spread | 1X2 | 595 | 1.06389 | 1.05344 | home | 0.4157 | 0.4215 |
| m04_poisson_grw_loose_fixed_spread | OU2.5 | 379 | 0.68889 | 0.68988 | over_25 | 0.5062 | 0.5298 |
| m04_poisson_grw_loose_fixed_spread | BTTS | 178 | 0.69342 | 0.68337 | btts_yes | 0.5275 | 0.5577 |
