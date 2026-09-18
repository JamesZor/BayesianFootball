# r05 counterfactual re-pricing — 2026-09-12, T-25

Generated 2026-09-13 20:22. as_of 2026-09-12T13:35:00 UTC. All arms Fold 43. Runs: base `b0961bc4-c40c-4dbe-9c05-57df7ae0839e`; smile020 `fcd5e974-9a46-4a10-9828-6b987a5484d6`; smile040 `30620d3e-e4bd-4c05-b1a1-85cefa36b728`; smile070 `32d588f1-d666-4112-a7e1-5c9545fbbe3d`; sup040 `0ee58d18-b7e9-4168-8d78-93887b1a8c26`.

Reproduction of the live orders by `live_optB`: 11 live legs, 15 re-priced, 11 shared; max |Δrisk| £3.64, max |Δp_model| 0.0234.

## Tearsheet (full-fill settlement)

| arm | n_legs | risk | net_pnl | n_away | away_risk | away_net | n_home | home_risk | home_net | n_under25 | n_wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| live_optB | 15 | 73.78 | -54.52 | 7 | 28.17 | -28.17 | 1 | 2.81 | 4.46 | 6 | 2 |
| base_raw | 17 | 125.00 | -98.27 | 7 | 56.86 | -56.86 | 2 | 13.69 | 0.06 | 4 | 2 |
| base_optB | 11 | 60.07 | -53.56 | 6 | 24.76 | -24.76 | 1 | 1.58 | 1.36 | 4 | 2 |
| sup040_raw | 15 | 125.00 | -102.00 | 6 | 43.27 | -43.27 | 2 | 22.39 | -19.07 | 5 | 3 |
| sup040_optB | 10 | 42.44 | -39.40 | 5 | 13.89 | -13.89 | 1 | 3.36 | -3.36 | 4 | 1 |
| smile020_raw | 18 | 124.74 | -91.04 | 6 | 46.39 | -46.39 | 3 | 22.84 | -10.46 | 5 | 4 |
| smile020_optB | 11 | 39.47 | -34.20 | 5 | 14.55 | -14.55 | 1 | 2.87 | -2.87 | 5 | 1 |
| smile040_raw | 17 | 123.94 | -78.85 | 6 | 40.60 | -40.60 | 3 | 27.37 | -11.64 | 6 | 4 |
| smile040_optB | 11 | 36.18 | -28.88 | 5 | 12.64 | -12.64 | 1 | 4.08 | -4.08 | 5 | 1 |
| smile070_raw | 15 | 124.04 | -69.40 | 5 | 33.50 | -33.50 | 2 | 31.06 | -15.68 | 6 | 3 |
| smile070_optB | 10 | 31.41 | -21.36 | 5 | 10.05 | -10.05 | 1 | 5.30 | -5.30 | 4 | 1 |
| live ledger (realised fills) | 11 | 55.73 | -45.89 | 6 | n/a | n/a | 0 | n/a | n/a | 5 | -1 |

## 1X2 P(home) at T−25

`mkt_p_home` is the de-vigged book the pipeline quoted at as_of.

| home | away | score | mkt_p_home | live_optB_p_home | base_raw_p_home | base_optB_p_home | sup040_raw_p_home | smile020_raw_p_home | smile040_raw_p_home | smile070_raw_p_home | smile040_optB_p_home |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| queen-of-the-south | east-fife | 1-1 | 0.392 | 0.401 | 0.440 | 0.406 | 0.488 | 0.483 | 0.495 | 0.506 | 0.427 |
| east-kilbride | peterhead | 3-1 | 0.560 | 0.494 | 0.478 | 0.532 | 0.532 | 0.515 | 0.526 | 0.544 | 0.555 |
| montrose | cove-rangers | 2-1 | 0.477 | 0.465 | 0.369 | 0.442 | 0.403 | 0.395 | 0.405 | 0.421 | 0.460 |
| airdrieonians | alloa-athletic | 1-0 | 0.411 | 0.416 | 0.323 | 0.386 | 0.367 | 0.361 | 0.366 | 0.376 | 0.405 |
| edinburgh-city-fc | stirling-albion | 7-3 | 0.541 | 0.494 | 0.476 | 0.523 | 0.493 | 0.471 | 0.478 | 0.480 | 0.526 |
| clyde-fc | kelty-hearts-fc | 4-1 | 0.514 | 0.478 | 0.546 | 0.535 | 0.526 | 0.533 | 0.534 | 0.526 | 0.525 |
| the-spartans-fc | forfar-athletic | 5-1 | 0.567 | 0.512 | 0.460 | 0.504 | 0.494 | 0.478 | 0.493 | 0.508 | 0.543 |
| annan-athletic | elgin-city | 2-0 | 0.372 | 0.396 | 0.352 | 0.368 | 0.383 | 0.393 | 0.401 | 0.407 | 0.385 |
| stranraer | dumbarton | 3-1 | 0.550 | 0.504 | 0.475 | 0.529 | 0.461 | 0.448 | 0.452 | 0.471 | 0.517 |

## Legs

| arm | fixture | group | line | selection | side | effective_odds | p_model | p_market | risk | score | outcome | net_pnl |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|---:|
| live_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.4758 | 0.4239 | 11.6813 | 5-1 | lose | -11.6813 |
| live_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4334 | 0.3836 | 9.9673 | 4-1 | lose | -9.9673 |
| live_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4295 | 0.3583 | 9.2734 | 3-1 | lose | -9.2734 |
| live_optB | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2707 | 0.2177 | 6.0987 | 3-1 | lose | -6.0987 |
| live_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2460 | 0.1902 | 5.9710 | 5-1 | lose | -5.9710 |
| live_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4360 | 0.4058 | 4.9714 | 2-0 | win | 7.0157 |
| live_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2639 | 0.2162 | 4.9381 | 7-3 | lose | -4.9381 |
| live_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2591 | 0.2174 | 4.3624 | 3-1 | lose | -4.3624 |
| live_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4439 | 0.4123 | 3.2252 | 3-1 | lose | -3.2252 |
| live_optB | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | away | back | 3.8500 | 0.2834 | 0.2508 | 3.0005 | 4-1 | lose | -3.0005 |
| live_optB | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.3961 | 0.3722 | 2.8090 | 2-0 | win | 4.4596 |
| live_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2840 | 0.2527 | 2.7363 | 2-1 | lose | -2.7363 |
| live_optB | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4608 | 0.4372 | 2.6399 | 7-3 | lose | -2.6399 |
| live_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3242 | 0.3033 | 1.0649 | 1-0 | lose | -1.0649 |
| live_optB | east-kilbride v peterhead | 1X2 | 0.0000 | draw | back | 4.4000 | 0.2357 | 0.2226 | 1.0407 | 3-1 | lose | -1.0407 |
| base_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5932 | 0.4239 | 20.7465 | 5-1 | lose | -20.7465 |
| base_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3821 | 0.2527 | 14.0401 | 2-1 | lose | -14.0401 |
| base_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4831 | 0.3836 | 13.5890 | 4-1 | lose | -13.5890 |
| base_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.4278 | 0.3033 | 13.4508 | 1-0 | lose | -13.4508 |
| base_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4514 | 0.3583 | 9.1896 | 3-1 | lose | -9.1896 |
| base_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2933 | 0.2162 | 7.4436 | 7-3 | lose | -7.4436 |
| base_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | home | back | 1.8800 | 0.5456 | 0.5137 | 7.3831 | 4-1 | win | 6.3672 |
| base_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2891 | 0.2174 | 7.3457 | 3-1 | lose | -7.3457 |
| base_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2646 | 0.1902 | 6.7820 | 5-1 | lose | -6.7820 |
| base_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2814 | 0.2177 | 6.4318 | 3-1 | lose | -6.4318 |
| base_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4403 | 0.3922 | 6.3025 | 1-1 | lose | -6.3025 |
| base_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4484 | 0.4058 | 5.3828 | 2-0 | win | 7.5961 |
| base_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2358 | 0.2329 | 1.6864 | 3-1 | lose | -1.6864 |
| base_raw | east-kilbride v peterhead | 1X2 | 0.0000 | draw | back | 4.4000 | 0.2409 | 0.2226 | 1.5380 | 3-1 | lose | -1.5380 |
| base_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | away | back | 2.5600 | 0.4014 | 0.3809 | 1.3638 | 2-0 | lose | -1.3638 |
| base_raw | montrose v cove-rangers | 1X2 | 0.0000 | draw | back | 3.6000 | 0.2487 | 0.2703 | 1.3047 | 2-1 | lose | -1.3047 |
| base_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | draw | back | 4.0000 | 0.2750 | 0.2425 | 1.0196 | 5-1 | lose | -1.0196 |
| base_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5170 | 0.4239 | 17.9975 | 5-1 | lose | -17.9975 |
| base_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4254 | 0.3836 | 7.9589 | 4-1 | lose | -7.9589 |
| base_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.3993 | 0.3583 | 6.2976 | 3-1 | lose | -6.2976 |
| base_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3081 | 0.2527 | 5.9471 | 2-1 | lose | -5.9471 |
| base_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2451 | 0.1902 | 5.7604 | 5-1 | lose | -5.7604 |
| base_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3551 | 0.3033 | 5.5142 | 1-0 | lose | -5.5142 |
| base_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2474 | 0.2162 | 2.7680 | 7-3 | lose | -2.7680 |
| base_optB | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2442 | 0.2177 | 2.4659 | 3-1 | lose | -2.4659 |
| base_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2438 | 0.2174 | 2.3018 | 3-1 | lose | -2.3018 |
| base_optB | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | home | back | 1.8800 | 0.5351 | 0.5137 | 1.5803 | 4-1 | win | 1.3628 |
| base_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4210 | 0.4058 | 1.4774 | 2-0 | win | 2.0849 |
| sup040_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5902 | 0.4239 | 24.7071 | 5-1 | lose | -24.7071 |
| sup040_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4882 | 0.3922 | 20.6064 | 1-1 | lose | -20.6064 |
| sup040_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4708 | 0.3836 | 13.8976 | 4-1 | lose | -13.8976 |
| sup040_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3464 | 0.2527 | 11.4336 | 2-1 | lose | -11.4336 |
| sup040_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2995 | 0.2174 | 10.4925 | 3-1 | lose | -10.4925 |
| sup040_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4343 | 0.3583 | 9.4046 | 3-1 | lose | -9.4046 |
| sup040_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3742 | 0.3033 | 8.4623 | 1-0 | lose | -8.4623 |
| sup040_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2776 | 0.2162 | 6.9269 | 7-3 | lose | -6.9269 |
| sup040_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4477 | 0.4058 | 5.8290 | 2-0 | win | 8.2259 |
| sup040_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2364 | 0.1902 | 4.3335 | 5-1 | lose | -4.3335 |
| sup040_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2397 | 0.2329 | 2.9226 | 3-1 | lose | -2.9226 |
| sup040_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | home | back | 1.8800 | 0.5264 | 0.5137 | 1.7857 | 4-1 | win | 1.5400 |
| sup040_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2386 | 0.2177 | 1.6163 | 3-1 | lose | -1.6163 |
| sup040_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | draw | back | 3.7000 | 0.2454 | 0.2629 | 1.5418 | 1-1 | win | 4.0797 |
| sup040_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4325 | 0.4123 | 1.0401 | 3-1 | lose | -1.0401 |
| sup040_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5042 | 0.4239 | 15.1191 | 5-1 | lose | -15.1191 |
| sup040_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4167 | 0.3836 | 5.5105 | 4-1 | lose | -5.5105 |
| sup040_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2916 | 0.2527 | 3.5801 | 2-1 | lose | -3.5801 |
| sup040_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2269 | 0.1902 | 3.3665 | 5-1 | lose | -3.3665 |
| sup040_optB | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4232 | 0.3922 | 3.3554 | 1-1 | lose | -3.3554 |
| sup040_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.3854 | 0.3583 | 3.3031 | 3-1 | lose | -3.3031 |
| sup040_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2478 | 0.2174 | 2.6853 | 3-1 | lose | -2.6853 |
| sup040_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3335 | 0.3033 | 2.2975 | 1-0 | lose | -2.2975 |
| sup040_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2418 | 0.2162 | 1.9583 | 7-3 | lose | -1.9583 |
| sup040_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4203 | 0.4058 | 1.2611 | 2-0 | win | 1.7797 |
| smile020_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5193 | 0.4239 | 17.7624 | 5-1 | lose | -17.7624 |
| smile020_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4830 | 0.3922 | 17.0928 | 1-1 | lose | -17.0928 |
| smile020_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4611 | 0.3836 | 12.5115 | 4-1 | lose | -12.5115 |
| smile020_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3480 | 0.2527 | 10.5993 | 2-1 | lose | -10.5993 |
| smile020_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.3035 | 0.2174 | 9.8779 | 3-1 | lose | -9.8779 |
| smile020_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4113 | 0.3583 | 7.9053 | 3-1 | lose | -7.9053 |
| smile020_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3757 | 0.3033 | 7.6657 | 1-0 | lose | -7.6657 |
| smile020_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2648 | 0.1902 | 7.4737 | 5-1 | lose | -7.4737 |
| smile020_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2866 | 0.2162 | 7.3115 | 7-3 | lose | -7.3115 |
| smile020_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4531 | 0.4058 | 6.7805 | 2-0 | win | 9.5686 |
| smile020_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4611 | 0.4123 | 3.6315 | 3-1 | lose | -3.6315 |
| smile020_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2551 | 0.2177 | 3.4574 | 3-1 | lose | -3.4574 |
| smile020_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | home | back | 1.8800 | 0.5334 | 0.5137 | 3.4448 | 4-1 | win | 2.9708 |
| smile020_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2487 | 0.2329 | 3.3527 | 3-1 | lose | -3.3527 |
| smile020_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.3933 | 0.3722 | 2.3061 | 2-0 | win | 3.6612 |
| smile020_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | draw | back | 3.7000 | 0.2478 | 0.2629 | 1.3605 | 1-1 | win | 3.5999 |
| smile020_raw | montrose v cove-rangers | 1X2 | 0.0000 | draw | back | 3.6000 | 0.2575 | 0.2703 | 1.1202 | 2-1 | lose | -1.1202 |
| smile020_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | draw | back | 4.0000 | 0.2420 | 0.2432 | 1.0832 | 7-3 | lose | -1.0832 |
| smile020_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5193 | 0.4239 | 9.7328 | 5-1 | lose | -9.7328 |
| smile020_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4611 | 0.3836 | 5.3480 | 4-1 | lose | -5.3480 |
| smile020_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2290 | 0.1902 | 3.5588 | 5-1 | lose | -3.5588 |
| smile020_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2914 | 0.2527 | 3.4613 | 2-1 | lose | -3.4613 |
| smile020_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2523 | 0.2174 | 3.1886 | 3-1 | lose | -3.1886 |
| smile020_optB | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4213 | 0.3922 | 2.8691 | 1-1 | lose | -2.8691 |
| smile020_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4113 | 0.3583 | 2.5348 | 3-1 | lose | -2.5348 |
| smile020_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3342 | 0.3033 | 2.3338 | 1-0 | lose | -2.3338 |
| smile020_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4611 | 0.4123 | 2.2512 | 3-1 | lose | -2.2512 |
| smile020_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4531 | 0.4058 | 2.1851 | 2-0 | win | 3.0837 |
| smile020_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2426 | 0.2162 | 2.0100 | 7-3 | lose | -2.0100 |
| smile040_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4948 | 0.3922 | 20.5055 | 1-1 | lose | -20.5055 |
| smile040_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5108 | 0.4239 | 16.9407 | 5-1 | lose | -16.9407 |
| smile040_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4480 | 0.3836 | 11.3798 | 4-1 | lose | -11.3798 |
| smile040_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.3011 | 0.2174 | 9.5931 | 3-1 | lose | -9.5931 |
| smile040_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3391 | 0.2527 | 9.3371 | 2-1 | lose | -9.3371 |
| smile040_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4656 | 0.4058 | 8.3407 | 2-0 | win | 11.7704 |
| smile040_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4140 | 0.3583 | 7.6939 | 3-1 | lose | -7.6939 |
| smile040_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3701 | 0.3033 | 6.9659 | 1-0 | lose | -6.9659 |
| smile040_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2782 | 0.2162 | 6.2921 | 7-3 | lose | -6.2921 |
| smile040_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2532 | 0.1902 | 6.1635 | 5-1 | lose | -6.1635 |
| smile040_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.4006 | 0.3722 | 4.0667 | 2-0 | win | 6.4562 |
| smile040_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4533 | 0.4123 | 3.2187 | 3-1 | lose | -3.2187 |
| smile040_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2468 | 0.2329 | 3.1198 | 3-1 | lose | -3.1198 |
| smile040_raw | clyde-fc v kelty-hearts-fc | 1X2 | 0.0000 | home | back | 1.8800 | 0.5336 | 0.5137 | 2.7948 | 4-1 | win | 2.4102 |
| smile040_raw | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4541 | 0.4372 | 2.7475 | 7-3 | lose | -2.7475 |
| smile040_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | draw | back | 3.7000 | 0.2480 | 0.2629 | 2.5389 | 1-1 | win | 6.7180 |
| smile040_raw | east-kilbride v peterhead | 1X2 | 0.0000 | away | back | 4.5000 | 0.2450 | 0.2177 | 2.2451 | 3-1 | lose | -2.2451 |
| smile040_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5108 | 0.4239 | 7.9658 | 5-1 | lose | -7.9658 |
| smile040_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4480 | 0.3836 | 4.2676 | 4-1 | lose | -4.2676 |
| smile040_optB | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4270 | 0.3922 | 4.0807 | 1-1 | lose | -4.0807 |
| smile040_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2881 | 0.2527 | 3.0369 | 2-1 | lose | -3.0369 |
| smile040_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4656 | 0.4058 | 3.0280 | 2-0 | win | 4.2731 |
| smile040_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2504 | 0.2174 | 2.9510 | 3-1 | lose | -2.9510 |
| smile040_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2231 | 0.1902 | 2.8417 | 5-1 | lose | -2.8417 |
| smile040_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4140 | 0.3583 | 2.4895 | 3-1 | lose | -2.4895 |
| smile040_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3325 | 0.3033 | 2.0915 | 1-0 | lose | -2.0915 |
| smile040_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2402 | 0.2162 | 1.7166 | 7-3 | lose | -1.7166 |
| smile040_optB | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4533 | 0.4123 | 1.7107 | 3-1 | lose | -1.7107 |
| smile070_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.5057 | 0.3922 | 25.1138 | 1-1 | lose | -25.1138 |
| smile070_raw | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5045 | 0.4239 | 16.9283 | 5-1 | lose | -16.9283 |
| smile070_raw | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4796 | 0.4058 | 10.5305 | 2-0 | win | 14.8607 |
| smile070_raw | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4335 | 0.3836 | 10.0963 | 4-1 | lose | -10.0963 |
| smile070_raw | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2862 | 0.2174 | 8.1120 | 3-1 | lose | -8.1120 |
| smile070_raw | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.3241 | 0.2527 | 7.8516 | 2-1 | lose | -7.8516 |
| smile070_raw | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4143 | 0.3583 | 7.4964 | 3-1 | lose | -7.4964 |
| smile070_raw | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2743 | 0.2162 | 6.2058 | 7-3 | lose | -6.2058 |
| smile070_raw | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3602 | 0.3033 | 6.0612 | 1-0 | lose | -6.0612 |
| smile070_raw | annan-athletic v elgin-city | 1X2 | 0.0000 | home | back | 2.6200 | 0.4067 | 0.3722 | 5.9438 | 2-0 | win | 9.4363 |
| smile070_raw | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2423 | 0.1902 | 5.2704 | 5-1 | lose | -5.2704 |
| smile070_raw | edinburgh-city-fc v stirling-albion | OverUnder | 2.5000 | under_25 | back | 2.2400 | 0.4665 | 0.4372 | 5.1476 | 7-3 | lose | -5.1476 |
| smile070_raw | queen-of-the-south v east-fife | 1X2 | 0.0000 | draw | back | 3.7000 | 0.2477 | 0.2629 | 3.8038 | 1-1 | win | 10.0647 |
| smile070_raw | stranraer v dumbarton | OverUnder | 2.5000 | under_25 | back | 2.3800 | 0.4457 | 0.4123 | 3.2569 | 3-1 | lose | -3.2569 |
| smile070_raw | stranraer v dumbarton | 1X2 | 0.0000 | draw | back | 4.2000 | 0.2430 | 0.2329 | 2.2197 | 3-1 | lose | -2.2197 |
| smile070_optB | the-spartans-fc v forfar-athletic | OverUnder | 2.5000 | under_25 | lay | 2.3514 | 0.5045 | 0.4239 | 6.6029 | 5-1 | lose | -6.6029 |
| smile070_optB | queen-of-the-south v east-fife | 1X2 | 0.0000 | home | back | 2.4800 | 0.4337 | 0.3922 | 5.2954 | 1-1 | lose | -5.2954 |
| smile070_optB | annan-athletic v elgin-city | OverUnder | 2.5000 | under_25 | back | 2.4400 | 0.4796 | 0.4058 | 4.1689 | 2-0 | win | 5.8831 |
| smile070_optB | clyde-fc v kelty-hearts-fc | OverUnder | 2.5000 | under_25 | lay | 2.5876 | 0.4335 | 0.3836 | 3.0228 | 4-1 | lose | -3.0228 |
| smile070_optB | montrose v cove-rangers | 1X2 | 0.0000 | away | back | 3.8500 | 0.2830 | 0.2527 | 2.3703 | 2-1 | lose | -2.3703 |
| smile070_optB | east-kilbride v peterhead | OverUnder | 2.5000 | under_25 | back | 2.7400 | 0.4143 | 0.3583 | 2.2677 | 3-1 | lose | -2.2677 |
| smile070_optB | the-spartans-fc v forfar-athletic | 1X2 | 0.0000 | away | back | 5.1000 | 0.2183 | 0.1902 | 2.2369 | 5-1 | lose | -2.2369 |
| smile070_optB | stranraer v dumbarton | 1X2 | 0.0000 | away | back | 4.5000 | 0.2443 | 0.2174 | 2.1741 | 3-1 | lose | -2.1741 |
| smile070_optB | airdrieonians v alloa-athletic | 1X2 | 0.0000 | away | back | 3.2000 | 0.3295 | 0.3033 | 1.6678 | 1-0 | lose | -1.6678 |
| smile070_optB | edinburgh-city-fc v stirling-albion | 1X2 | 0.0000 | away | back | 4.5000 | 0.2396 | 0.2162 | 1.6056 | 7-3 | lose | -1.6056 |
