# r05 counterfactual re-pricing — 2026-09-12, T-25

Generated 2026-09-13 00:27. as_of 2026-09-12T13:35:00 UTC. Flat arm Run 67; hierarchical arm `m12_joint_hybrid_synergy_hier_ha` run 87c1052d-a181-434c-a965-3c8c801f4142. Both on Fold 43.

Reproduction of the live orders by `flat_optB`: 11 live legs, 15 re-priced, 11 shared; max |Δrisk| £3.64, max |Δp_model| 0.0234.

## Tearsheet

| arm | n_legs | risk | net_pnl | n_away | away_risk | away_net | n_away_on_turf | n_home | home_net | n_under25 | n_wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_raw | 19 | 124.2815 | -84.2718 | 7 | 54.7688 | -54.7688 | 6 | 2 | 11.3045 | 6 | 2 |
| flat_optB | 15 | 73.7801 | -54.5244 | 7 | 28.1718 | -28.1718 | 6 | 1 | 4.4596 | 6 | 2 |
| hier_raw | 19 | 124.2571 | -83.5002 | 7 | 54.5187 | -54.5187 | 6 | 2 | 12.0616 | 6 | 2 |
| hier_optB | 14 | 73.5136 | -54.2639 | 7 | 28.0811 | -28.0811 | 6 | 1 | 4.6364 | 6 | 2 |
| live_ledger (realised fills) | 11 | 55.7300 | -45.8900 | 6 | n/a | n/a | -1 | 0 | n/a | 5 | -1 |

## 1X2 home probability

| home | away | home_surface | score | flat_raw_p_home | hier_raw_p_home | raw_delta_p_home | flat_optB_p_home | hier_optB_p_home | optB_delta_p_home |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| queen-of-the-south | east-fife | grass | 1-1 | 0.4196 | 0.4193 | -0.0004 | 0.4009 | 0.4008 | -0.0001 |
| east-kilbride | peterhead | turf | 3-1 | 0.4223 | 0.4252 | +0.0030 | 0.4936 | 0.4964 | +0.0027 |
| montrose | cove-rangers | turf | 2-1 | 0.4180 | 0.4209 | +0.0028 | 0.4653 | 0.4662 | +0.0009 |
| airdrieonians | alloa-athletic | turf | 1-0 | 0.4029 | 0.4024 | -0.0005 | 0.4162 | 0.4160 | -0.0002 |
| edinburgh-city-fc | stirling-albion | turf | 7-3 | 0.4107 | 0.4085 | -0.0021 | 0.4945 | 0.4930 | -0.0015 |
| clyde-fc | kelty-hearts-fc | turf | 4-1 | 0.4257 | 0.4262 | +0.0004 | 0.4781 | 0.4778 | -0.0002 |
| the-spartans-fc | forfar-athletic | turf | 5-1 | 0.4304 | 0.4275 | -0.0030 | 0.5124 | 0.5094 | -0.0030 |
| annan-athletic | elgin-city | grass | 2-0 | 0.4288 | 0.4315 | +0.0028 | 0.3961 | 0.3968 | +0.0007 |
| stranraer | dumbarton | grass | 3-1 | 0.4315 | 0.4293 | -0.0022 | 0.5040 | 0.5029 | -0.0012 |

## Legs

| arm | fixture | group | line | selection | side | effective_odds | p_model | p_market | risk | score | outcome | net_pnl |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|---:|
| flat_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4911 | 0.3836 | 12.6301 | 4-1 | lose | -12.6301 |
| flat_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5104 | 0.4239 | 12.1165 | 5-1 | lose | -12.1165 |
| flat_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.3358 | 0.2162 | 10.9712 | 7-3 | lose | -10.9712 |
| flat_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.3104 | 0.1902 | 10.4374 | 5-1 | lose | -10.4374 |
| flat_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.3250 | 0.2177 | 9.8578 | 3-1 | lose | -9.8578 |
| flat_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.3163 | 0.2174 | 9.1067 | 3-1 | lose | -9.1067 |
| flat_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4787 | 0.3583 | 8.4200 | 3-1 | lose | -8.4200 |
| flat_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.4288 | 0.3722 | 8.3250 | 2-0 | win | 13.2167 |
| flat_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4794 | 0.4058 | 7.6593 | 2-0 | win | 10.8088 |
| flat_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3275 | 0.2527 | 6.0669 | 2-1 | lose | -6.0669 |
| flat_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | back | 3.8500 | 0.3191 | 0.2508 | 5.8824 | 4-1 | lose | -5.8824 |
| flat_raw | east-kilbride v peterhead | 1X2 | 0.0000 | draw | back | 4.4000 | 0.2527 | 0.2226 | 3.4613 | 3-1 | lose | -3.4613 |
| flat_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2522 | 0.2329 | 3.3153 | 3-1 | lose | -3.3153 |
| flat_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | draw | back | 4.0000 | 0.2536 | 0.2432 | 3.2562 | 7-3 | lose | -3.2562 |
| flat_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4790 | 0.4123 | 3.1381 | 3-1 | lose | -3.1381 |
| flat_raw | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4793 | 0.4372 | 3.1235 | 7-3 | lose | -3.1235 |
| flat_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3407 | 0.3033 | 2.4464 | 1-0 | lose | -2.4464 |
| flat_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | draw | back | 4.0000 | 0.2592 | 0.2425 | 2.1554 | 5-1 | lose | -2.1554 |
| flat_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4196 | 0.3922 | 1.9123 | 1-1 | lose | -1.9123 |
| flat_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.4758 | 0.4239 | 11.6813 | 5-1 | lose | -11.6813 |
| flat_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4334 | 0.3836 | 9.9673 | 4-1 | lose | -9.9673 |
| flat_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4295 | 0.3583 | 9.2734 | 3-1 | lose | -9.2734 |
| flat_optB | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2707 | 0.2177 | 6.0987 | 3-1 | lose | -6.0987 |
| flat_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2460 | 0.1902 | 5.9710 | 5-1 | lose | -5.9710 |
| flat_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4360 | 0.4058 | 4.9714 | 2-0 | win | 7.0157 |
| flat_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2639 | 0.2162 | 4.9381 | 7-3 | lose | -4.9381 |
| flat_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2591 | 0.2174 | 4.3624 | 3-1 | lose | -4.3624 |
| flat_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4439 | 0.4123 | 3.2252 | 3-1 | lose | -3.2252 |
| flat_optB | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | back | 3.8500 | 0.2834 | 0.2508 | 3.0005 | 4-1 | lose | -3.0005 |
| flat_optB | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.3961 | 0.3722 | 2.8090 | 2-0 | win | 4.4596 |
| flat_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2840 | 0.2527 | 2.7363 | 2-1 | lose | -2.7363 |
| flat_optB | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4608 | 0.4372 | 2.6399 | 7-3 | lose | -2.6399 |
| flat_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3242 | 0.3033 | 1.0649 | 1-0 | lose | -1.0649 |
| flat_optB | east-kilbride v peterhead | 1X2 | 0.0000 | draw | back | 4.4000 | 0.2357 | 0.2226 | 1.0407 | 3-1 | lose | -1.0407 |
| hier_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4910 | 0.3836 | 12.5532 | 4-1 | lose | -12.5532 |
| hier_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5140 | 0.4239 | 12.1668 | 5-1 | lose | -12.1668 |
| hier_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.3372 | 0.2162 | 11.0441 | 7-3 | lose | -11.0441 |
| hier_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.3124 | 0.1902 | 10.5481 | 5-1 | lose | -10.5481 |
| hier_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.3230 | 0.2177 | 9.5892 | 3-1 | lose | -9.5892 |
| hier_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.3180 | 0.2174 | 9.2179 | 3-1 | lose | -9.2179 |
| hier_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.4315 | 0.3722 | 8.7529 | 2-0 | win | 13.8961 |
| hier_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4757 | 0.3583 | 8.2130 | 3-1 | lose | -8.2130 |
| hier_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4770 | 0.4058 | 7.5099 | 2-0 | win | 10.5980 |
| hier_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3255 | 0.2527 | 5.8375 | 2-1 | lose | -5.8375 |
| hier_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | back | 3.8500 | 0.3187 | 0.2508 | 5.7969 | 4-1 | lose | -5.7969 |
| hier_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2527 | 0.2329 | 3.3757 | 3-1 | lose | -3.3757 |
| hier_raw | east-kilbride v peterhead | 1X2 | 0.0000 | draw | back | 4.4000 | 0.2518 | 0.2226 | 3.3330 | 3-1 | lose | -3.3330 |
| hier_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | draw | back | 4.0000 | 0.2543 | 0.2432 | 3.3223 | 7-3 | lose | -3.3223 |
| hier_raw | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4823 | 0.4372 | 3.2355 | 7-3 | lose | -3.2355 |
| hier_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4806 | 0.4123 | 3.1788 | 3-1 | lose | -3.1788 |
| hier_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3413 | 0.3033 | 2.4851 | 1-0 | lose | -2.4851 |
| hier_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | draw | back | 4.0000 | 0.2602 | 0.2425 | 2.2627 | 5-1 | lose | -2.2627 |
| hier_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4193 | 0.3922 | 1.8345 | 1-1 | lose | -1.8345 |
| hier_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.4792 | 0.4239 | 12.4071 | 5-1 | lose | -12.4071 |
| hier_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4337 | 0.3836 | 9.9414 | 4-1 | lose | -9.9414 |
| hier_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4268 | 0.3583 | 9.0587 | 3-1 | lose | -9.0587 |
| hier_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2479 | 0.1902 | 6.1534 | 5-1 | lose | -6.1534 |
| hier_optB | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2689 | 0.2177 | 5.7721 | 3-1 | lose | -5.7721 |
| hier_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2649 | 0.2162 | 5.0126 | 7-3 | lose | -5.0126 |
| hier_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4354 | 0.4058 | 4.8494 | 2-0 | win | 6.8435 |
| hier_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2599 | 0.2174 | 4.4395 | 3-1 | lose | -4.4395 |
| hier_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4449 | 0.4123 | 3.2501 | 3-1 | lose | -3.2501 |
| hier_optB | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4627 | 0.4372 | 3.0053 | 7-3 | lose | -3.0053 |
| hier_optB | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | back | 3.8500 | 0.2835 | 0.2508 | 2.9942 | 4-1 | lose | -2.9942 |
| hier_optB | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.3968 | 0.3722 | 2.9204 | 2-0 | win | 4.6364 |
| hier_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2834 | 0.2527 | 2.6317 | 2-1 | lose | -2.6317 |
| hier_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3244 | 0.3033 | 1.0778 | 1-0 | lose | -1.0778 |
