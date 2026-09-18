# r09 counterfactual re-pricing — 2026-09-12, T-25 — Task 008 Phase 2

Generated 2026-09-18 15:02. as_of 2026-09-12T13:35:00 UTC. Flat arm Run 67; contextual arm `m12_joint_hybrid_contextual` run 991e4991-624f-4166-a551-3233cd17ecb4 (terms: turf_asym, turf_gen, turf_pace, midweek, rest_diff). Both on Fold 43.

Reproduction by `flat_optB`: 11 live legs, 15 re-priced, 11 shared; max |Δrisk| £3.64. Live realised £-45.89.

## Rung 5 fit

| model | folds | oos | draws | max_rhat | worst_rhat_fold | min_ess_bulk | min_ess_tail | n_divergent | n_transitions | divergence_rate | treedepth_rate | min_bfmi | passed | strict_rhat_pass | failures | wall_min | run_id | max_site_rhat | n_unmapped_home |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| m12_joint_hybrid_contextual | 43 | 769 | 2000 | 1.0074 | 42 | 966.3699 | 558.2684 | 0 | 172000 | 0.0000 | 0.0000 | 0.7163 | true | true |  | 13.2021 | 991e4991-624f-4166-a551-3233cd17ecb4 | 1.0065 | 5 |

## Tearsheet

| arm | n_legs | risk | net_pnl | n_away | away_risk | away_net | n_away_on_turf | away_risk_on_turf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_raw | 19 | 124.2815 | -84.2718 | 7 | 54.7688 | -54.7688 | 6 | 45.6620 |
| flat_optB | 15 | 73.7801 | -54.5244 | 7 | 28.1718 | -28.1718 | 6 | 23.8095 |
| ctx_raw | 18 | 122.9367 | -79.2004 | 7 | 54.4936 | -54.4936 | 6 | 45.1187 |
| ctx_optB | 14 | 72.0781 | -52.6051 | 7 | 27.0954 | -27.0954 | 6 | 22.7536 |

## 1X2 by fixture

| home | away | turf_home | turf_away | score | flat_raw_p_home | ctx_raw_p_home | raw_delta_p_home | raw_delta_p_away | flat_optB_p_home | ctx_optB_p_home | optB_delta_p_home | flat_raw_p_over25 | ctx_raw_p_over25 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| queen-of-the-south | east-fife | 1.0000 | 1.0000 | 1-1 | 0.4196 | 0.4140 | -0.0056 | +0.0043 | 0.4009 | 0.3995 | -0.0014 | 0.5090 | 0.5040 |
| east-kilbride | peterhead | 1.0000 | 0.0000 | 3-1 | 0.4223 | 0.4329 | +0.0106 | -0.0076 | 0.4936 | 0.5037 | +0.0101 | 0.5213 | 0.5313 |
| montrose | cove-rangers | 1.0000 | 1.0000 | 2-1 | 0.4180 | 0.4181 | +0.0000 | -0.0010 | 0.4653 | 0.4652 | -0.0000 | 0.5142 | 0.5094 |
| airdrieonians | alloa-athletic | 1.0000 | 1.0000 | 1-0 | 0.4029 | 0.3975 | -0.0055 | +0.0034 | 0.4162 | 0.4148 | -0.0014 | 0.5091 | 0.5011 |
| edinburgh-city-fc | stirling-albion | 1.0000 | 0.0000 | 7-3 | 0.4107 | 0.4170 | +0.0064 | -0.0050 | 0.4945 | 0.4988 | +0.0043 | 0.5207 | 0.5252 |
| clyde-fc | kelty-hearts-fc | 1.0000 | 1.0000 | 4-1 | 0.4257 | 0.4211 | -0.0046 | +0.0028 | 0.4781 | 0.4734 | -0.0047 | 0.5089 | 0.5021 |
| the-spartans-fc | forfar-athletic | 1.0000 | 1.0000 | 5-1 | 0.4304 | 0.4244 | -0.0060 | +0.0034 | 0.5124 | 0.5053 | -0.0071 | 0.4896 | 0.4800 |
| annan-athletic | elgin-city | 1.0000 | 0.0000 | 2-0 | 0.4288 | 0.4398 | +0.0111 | -0.0079 | 0.3961 | 0.3989 | +0.0028 | 0.5206 | 0.5308 |
| stranraer | dumbarton | 0.0000 | 1.0000 | 3-1 | 0.4315 | 0.4287 | -0.0029 | +0.0029 | 0.5040 | 0.5031 | -0.0009 | 0.5210 | 0.5223 |

## Legs

| arm | fixture | group | line | selection | effective_odds | p_model | p_market | risk | score | outcome | net_pnl |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|---:|
| flat_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | 2.5876 | 0.4911 | 0.3836 | 12.6301 | 4-1 | lose | -12.6301 |
| flat_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | 2.3514 | 0.5104 | 0.4239 | 12.1165 | 5-1 | lose | -12.1165 |
| flat_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | 4.5000 | 0.3358 | 0.2162 | 10.9712 | 7-3 | lose | -10.9712 |
| flat_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | 5.1000 | 0.3104 | 0.1902 | 10.4374 | 5-1 | lose | -10.4374 |
| flat_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | 4.5000 | 0.3250 | 0.2177 | 9.8578 | 3-1 | lose | -9.8578 |
| flat_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | 4.5000 | 0.3163 | 0.2174 | 9.1067 | 3-1 | lose | -9.1067 |
| flat_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | 2.7400 | 0.4787 | 0.3583 | 8.4200 | 3-1 | lose | -8.4200 |
| flat_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | 2.6200 | 0.4288 | 0.3722 | 8.3250 | 2-0 | win | 13.2167 |
| flat_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | 2.4400 | 0.4794 | 0.4058 | 7.6593 | 2-0 | win | 10.8088 |
| flat_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | 3.8500 | 0.3275 | 0.2527 | 6.0669 | 2-1 | lose | -6.0669 |
| flat_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | 3.8500 | 0.3191 | 0.2508 | 5.8824 | 4-1 | lose | -5.8824 |
| flat_raw | east-kilbride v peterhead | 1X2 | 0.0000 | draw | 4.4000 | 0.2527 | 0.2226 | 3.4613 | 3-1 | lose | -3.4613 |
| flat_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | 4.2000 | 0.2522 | 0.2329 | 3.3153 | 3-1 | lose | -3.3153 |
| flat_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | draw | 4.0000 | 0.2536 | 0.2432 | 3.2562 | 7-3 | lose | -3.2562 |
| flat_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | 2.3800 | 0.4790 | 0.4123 | 3.1381 | 3-1 | lose | -3.1381 |
| flat_raw | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | 2.2400 | 0.4793 | 0.4372 | 3.1235 | 7-3 | lose | -3.1235 |
| flat_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | 3.2000 | 0.3407 | 0.3033 | 2.4464 | 1-0 | lose | -2.4464 |
| flat_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | draw | 4.0000 | 0.2592 | 0.2425 | 2.1554 | 5-1 | lose | -2.1554 |
| flat_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | 2.4800 | 0.4196 | 0.3922 | 1.9123 | 1-1 | lose | -1.9123 |
| flat_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | 2.3514 | 0.4758 | 0.4239 | 11.6813 | 5-1 | lose | -11.6813 |
| flat_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | 2.5876 | 0.4334 | 0.3836 | 9.9673 | 4-1 | lose | -9.9673 |
| flat_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | 2.7400 | 0.4295 | 0.3583 | 9.2734 | 3-1 | lose | -9.2734 |
| flat_optB | east-kilbride v peterhead | 1X2 | 0.0000 | away | 4.5000 | 0.2707 | 0.2177 | 6.0987 | 3-1 | lose | -6.0987 |
| flat_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | 5.1000 | 0.2460 | 0.1902 | 5.9710 | 5-1 | lose | -5.9710 |
| flat_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | 2.4400 | 0.4360 | 0.4058 | 4.9714 | 2-0 | win | 7.0157 |
| flat_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | 4.5000 | 0.2639 | 0.2162 | 4.9381 | 7-3 | lose | -4.9381 |
| flat_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | 4.5000 | 0.2591 | 0.2174 | 4.3624 | 3-1 | lose | -4.3624 |
| flat_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | 2.3800 | 0.4439 | 0.4123 | 3.2252 | 3-1 | lose | -3.2252 |
| flat_optB | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | 3.8500 | 0.2834 | 0.2508 | 3.0005 | 4-1 | lose | -3.0005 |
| flat_optB | annan-athletic v elgin-city | 1X2 | 0.0000 | home | 2.6200 | 0.3961 | 0.3722 | 2.8090 | 2-0 | win | 4.4596 |
| flat_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | 3.8500 | 0.2840 | 0.2527 | 2.7363 | 2-1 | lose | -2.7363 |
| flat_optB | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | 2.2400 | 0.4608 | 0.4372 | 2.6399 | 7-3 | lose | -2.6399 |
| flat_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | 3.2000 | 0.3242 | 0.3033 | 1.0649 | 1-0 | lose | -1.0649 |
| flat_optB | east-kilbride v peterhead | 1X2 | 0.0000 | draw | 4.4000 | 0.2357 | 0.2226 | 1.0407 | 3-1 | lose | -1.0407 |
| ctx_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | 2.5876 | 0.4979 | 0.3836 | 12.9181 | 4-1 | lose | -12.9181 |
| ctx_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | 2.3514 | 0.5200 | 0.4239 | 12.4534 | 5-1 | lose | -12.4534 |
| ctx_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | 5.1000 | 0.3139 | 0.1902 | 10.7252 | 5-1 | lose | -10.7252 |
| ctx_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | 4.5000 | 0.3308 | 0.2162 | 10.4038 | 7-3 | lose | -10.4038 |
| ctx_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | 2.6200 | 0.4398 | 0.3722 | 10.2237 | 2-0 | win | 16.2312 |
| ctx_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | 4.5000 | 0.3192 | 0.2174 | 9.3749 | 3-1 | lose | -9.3749 |
| ctx_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | 4.5000 | 0.3174 | 0.2177 | 9.0715 | 3-1 | lose | -9.0715 |
| ctx_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | 2.7400 | 0.4687 | 0.3583 | 7.9157 | 3-1 | lose | -7.9157 |
| ctx_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | 2.4400 | 0.4692 | 0.4058 | 7.1671 | 2-0 | win | 10.1142 |
| ctx_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | 3.8500 | 0.3219 | 0.2508 | 6.1841 | 4-1 | lose | -6.1841 |
| ctx_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | 3.8500 | 0.3264 | 0.2527 | 5.9474 | 2-1 | lose | -5.9474 |
| ctx_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | 4.2000 | 0.2521 | 0.2329 | 3.4096 | 3-1 | lose | -3.4096 |
| ctx_raw | east-kilbride v peterhead | 1X2 | 0.0000 | draw | 4.4000 | 0.2497 | 0.2226 | 3.0820 | 3-1 | lose | -3.0820 |
| ctx_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | 2.3800 | 0.4777 | 0.4123 | 3.0229 | 3-1 | lose | -3.0229 |
| ctx_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | draw | 4.0000 | 0.2522 | 0.2432 | 2.9779 | 7-3 | lose | -2.9779 |
| ctx_raw | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | 2.2400 | 0.4748 | 0.4372 | 2.8604 | 7-3 | lose | -2.8604 |
| ctx_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | 3.2000 | 0.3442 | 0.3033 | 2.7866 | 1-0 | lose | -2.7866 |
| ctx_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | draw | 4.0000 | 0.2618 | 0.2425 | 2.4122 | 5-1 | lose | -2.4122 |
| ctx_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | 2.3514 | 0.4842 | 0.4239 | 13.3272 | 5-1 | lose | -13.3272 |
| ctx_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | 2.5876 | 0.4385 | 0.3836 | 10.4418 | 4-1 | lose | -10.4418 |
| ctx_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | 2.7400 | 0.4199 | 0.3583 | 8.5226 | 3-1 | lose | -8.5226 |
| ctx_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | 5.1000 | 0.2503 | 0.1902 | 6.3204 | 5-1 | lose | -6.3204 |
| ctx_optB | east-kilbride v peterhead | 1X2 | 0.0000 | away | 4.5000 | 0.2639 | 0.2177 | 4.9339 | 3-1 | lose | -4.9339 |
| ctx_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | 2.4400 | 0.4337 | 0.4058 | 4.5343 | 2-0 | win | 6.3987 |
| ctx_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | 4.5000 | 0.2610 | 0.2162 | 4.4007 | 7-3 | lose | -4.4007 |
| ctx_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | 4.5000 | 0.2599 | 0.2174 | 4.3418 | 3-1 | lose | -4.3418 |
| ctx_optB | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | 3.8500 | 0.2865 | 0.2508 | 3.3422 | 4-1 | lose | -3.3422 |
| ctx_optB | annan-athletic v elgin-city | 1X2 | 0.0000 | home | 2.6200 | 0.3989 | 0.3722 | 3.3004 | 2-0 | win | 5.2396 |
| ctx_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | 2.3800 | 0.4436 | 0.4123 | 3.0838 | 3-1 | lose | -3.0838 |
| ctx_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | 3.8500 | 0.2837 | 0.2527 | 2.6072 | 2-1 | lose | -2.6072 |
| ctx_optB | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | 2.2400 | 0.4573 | 0.4372 | 1.7727 | 7-3 | lose | -1.7727 |
| ctx_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | 3.2000 | 0.3251 | 0.3033 | 1.1492 | 1-0 | lose | -1.1492 |
