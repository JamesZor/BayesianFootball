# Market pruning report

Generated `2026-09-18 16:56`. Phase 1 uses the exact operational Option B trust vector and `:excise_pruned`; Phase 2 re-solves every tier cell from the posterior books.

## Headline summary

| model | p0_return_pct | p0_roi_pct | p0_sharpe_ann | p0_max_drawdown_pct | n_accretive | n_toxic | n_cannibalizing | best_tier_return_pct | best_tier_sharpe_ann | best_tier_max_drawdown_pct | t012_delta_return_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m05_joint_production_wealth | 532.303970993003 | 13.844481293648002 | 1.4202331934942067 | -39.19669832410285 | 3 | 2 | 4 | 185.08761903879108 | 1.4829988101413452 | -23.95221590077296 | -53.024952114564144 |
| m12_joint_hybrid_synergy | 588.4651310939773 | 13.955004922194439 | 1.4508901398700011 | -42.35216515135185 | 5 | 2 | 2 | 220.69555515695387 | 1.5453013130100663 | -29.473520355557888 | -14.49391992803396 |
| m05_joint_grw_smile_spine_w040 | 576.1331019902052 | 16.407057164586277 | 1.5370174842308106 | -42.81333765155565 | 0 | 4 | 5 | 181.1213497753291 | 1.6549765715415263 | -27.10207719739407 | -126.25969798353412 |

## Verification gates

| gate | model | pass | detail |
|---|---|---|---|
| S1 published Option B reproduction | m05_joint_production_wealth | missing | no published exact-contract reference; recorded panel 632, bets 1277, return 504.881203 |
| S2 score-grid coherence | m05_joint_production_wealth | true | 628 books, 0 checks, max gap 0.000e+00 (tol 1.0e-09) |
| S1 published Option B reproduction | m12_joint_hybrid_synergy | missing | no published exact-contract reference; recorded panel 632, bets 1302, return 606.536962 |
| S2 score-grid coherence | m12_joint_hybrid_synergy | true | 628 books, 0 checks, max gap 0.000e+00 (tol 1.0e-09) |
| S1 published Option B reproduction | m05_joint_grw_smile_spine_w040 | true | panel 632/632, bets 1232/1232, return 469.353510/469.353510 |
| S2 score-grid coherence | m05_joint_grw_smile_spine_w040 | true | 628 books, 3140 checks, max gap 2.998e-15 (tol 1.0e-09) |

## T012 contrast

| model | gate | pass | shift_detected | ledgers_identical | excised_markets | retained_markets | excised_n_books | retained_n_books | excised_n_bets | retained_n_bets | excised_return_pct | retained_return_pct | delta_return_pp | excised_roi_pct | retained_roi_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m05_joint_production_wealth | S0 T012 quantified | true | true | false | ou15;ou25;x1x2 | btts;ou05;ou15;ou25;ou35;ou45;x1x2 | 628 | 628 | 1281 | 1277 | 532.303970993003 | 479.2790188784388 | -53.024952114564144 | 13.844481293648002 | 13.429005831547505 |
| m12_joint_hybrid_synergy | S0 T012 quantified | true | true | false | ou15;ou25;x1x2 | btts;ou05;ou15;ou25;ou35;ou45;x1x2 | 628 | 628 | 1310 | 1304 | 588.4651310939773 | 573.9712111659434 | -14.49391992803396 | 13.955004922194439 | 13.91827465265205 |
| m05_joint_grw_smile_spine_w040 | S0 T012 quantified | true | true | false | ou15;ou25;x1x2 | btts;ou05;ou15;ou25;ou35;ou45;x1x2 | 628 | 628 | 1188 | 1247 | 576.1331019902052 | 449.8734040066711 | -126.25969798353412 | 16.407057164586277 | 14.242207093891468 |

## Cross-paradigm line decision

| candidate | robustly_accretive | classifications | delta_return_pp |
|---|---|---|---|
| +BTTS_no | false | m05_joint_grw_smile_spine_w040=toxic; m05_joint_production_wealth=toxic; m12_joint_hybrid_synergy=toxic | m05_joint_grw_smile_spine_w040=-41.63; m05_joint_production_wealth=-15.09; m12_joint_hybrid_synergy=-3.54 |
| +BTTS_yes | false | m05_joint_grw_smile_spine_w040=cannibalizing; m05_joint_production_wealth=cannibalizing; m12_joint_hybrid_synergy=cannibalizing | m05_joint_grw_smile_spine_w040=-12.69; m05_joint_production_wealth=-23.43; m12_joint_hybrid_synergy=-5.16 |
| +O2.5 | false | m05_joint_grw_smile_spine_w040=cannibalizing; m05_joint_production_wealth=cannibalizing; m12_joint_hybrid_synergy=accretive_with_cannibalization | m05_joint_grw_smile_spine_w040=-28.42; m05_joint_production_wealth=-3.71; m12_joint_hybrid_synergy=2.49 |
| +O3.5 | false | m05_joint_grw_smile_spine_w040=cannibalizing; m05_joint_production_wealth=cannibalizing; m12_joint_hybrid_synergy=accretive_with_cannibalization | m05_joint_grw_smile_spine_w040=-141.52; m05_joint_production_wealth=-11.83; m12_joint_hybrid_synergy=24.07 |
| +O4.5 | false | m05_joint_grw_smile_spine_w040=cannibalizing; m05_joint_production_wealth=cannibalizing; m12_joint_hybrid_synergy=accretive_with_cannibalization | m05_joint_grw_smile_spine_w040=-27.28; m05_joint_production_wealth=-5.24; m12_joint_hybrid_synergy=34.83 |
| +U0.5 | false | m05_joint_grw_smile_spine_w040=toxic; m05_joint_production_wealth=toxic; m12_joint_hybrid_synergy=toxic | m05_joint_grw_smile_spine_w040=-164.91; m05_joint_production_wealth=-39.5; m12_joint_hybrid_synergy=-14.7 |
| +U1.5 | false | m05_joint_grw_smile_spine_w040=toxic; m05_joint_production_wealth=accretive_with_cannibalization; m12_joint_hybrid_synergy=accretive_with_cannibalization | m05_joint_grw_smile_spine_w040=-59.76; m05_joint_production_wealth=50.53; m12_joint_hybrid_synergy=4.58 |
| +U3.5 | false | m05_joint_grw_smile_spine_w040=toxic; m05_joint_production_wealth=accretive_with_cannibalization; m12_joint_hybrid_synergy=accretive_with_cannibalization | m05_joint_grw_smile_spine_w040=-101.93; m05_joint_production_wealth=42.37; m12_joint_hybrid_synergy=24.79 |
| +U4.5 | false | m05_joint_grw_smile_spine_w040=cannibalizing; m05_joint_production_wealth=accretive_with_cannibalization; m12_joint_hybrid_synergy=cannibalizing | m05_joint_grw_smile_spine_w040=-41.01; m05_joint_production_wealth=9.06; m12_joint_hybrid_synergy=-20.37 |

A line is `robustly_accretive` only if every requested paradigm classifies it as accretive. This deliberately prevents a gain isolated to one posterior family from becoming the shared production basket.


## Phase 1 line screening

| model | candidate_label | classification | added_n_bets | added_roi_pct | core_stake_vs_p0 | delta_core_roi_pp | delta_return_pp | total_return_pct | sharpe_ann | max_drawdown_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| m05_joint_production_wealth | P0 Option B | baseline | 0 | NaN | 1.0 | 0.0 | 0.0 | 532.303970993003 | 1.4202331934942067 | -39.19669832410285 |
| m05_joint_production_wealth | +U0.5 | toxic | 65 | -61.030416970656866 | 0.9942358439944128 | 0.08639417884501022 | -39.502138453341445 | 492.8018325396615 | 1.3822073880263257 | -39.449814665884034 |
| m05_joint_production_wealth | +U1.5 | accretive_with_cannibalization | 118 | 20.98721510867344 | 0.9823699200202818 | -0.06515990758655121 | 50.53102774043862 | 582.8349987334416 | 1.5110430719745722 | -39.926978724025645 |
| m05_joint_production_wealth | +U3.5 | accretive_with_cannibalization | 108 | 7.966860283124587 | 0.9523885835714188 | 0.2956006733211449 | 42.36843808986066 | 574.6724090828636 | 1.516385143588037 | -37.766604895538954 |
| m05_joint_production_wealth | +U4.5 | accretive_with_cannibalization | 33 | 26.555411028788043 | 0.9797435285260369 | -0.5048620293186605 | 9.062089241795434 | 541.3660602347984 | 1.4410645818560046 | -39.32568304785216 |
| m05_joint_production_wealth | +O2.5 | cannibalizing | 95 | 0.7441717556559467 | 0.9790497146597895 | 0.2308010376364038 | -3.7082930929993836 | 528.5956779000036 | 1.418899685143075 | -43.91655277922203 |
| m05_joint_production_wealth | +O3.5 | cannibalizing | 102 | 4.874420143186577 | 0.9828077525339998 | 0.08679710012944852 | -11.82658129259778 | 520.4773897004052 | 1.3784187862326516 | -44.02906220952039 |
| m05_joint_production_wealth | +O4.5 | cannibalizing | 48 | 22.070675621657468 | 0.9954530108878703 | -0.047138339340101254 | -5.243902498450211 | 527.0600684945528 | 1.391409573512397 | -42.91298500218158 |
| m05_joint_production_wealth | +BTTS_yes | cannibalizing | 69 | 2.150456735965968 | 0.9856218695368354 | -0.08695730152092906 | -23.425342014096316 | 508.87862897890665 | 1.404504640454398 | -38.48844065699047 |
| m05_joint_production_wealth | +BTTS_no | toxic | 51 | -6.122723637435029 | 0.9920806954901659 | -0.04527125407953925 | -15.086177298898747 | 517.2177936941042 | 1.4376746287915108 | -38.11208260258323 |
| m12_joint_hybrid_synergy | P0 Option B | baseline | 0 | NaN | 1.0 | 0.0 | 0.0 | 588.4651310939773 | 1.4508901398700011 | -42.35216515135185 |
| m12_joint_hybrid_synergy | +U0.5 | toxic | 64 | -46.45109748761918 | 0.9949426584273992 | 0.1721128267132208 | -14.699687803372854 | 573.7654432906045 | 1.4472308794205238 | -41.94386688007613 |
| m12_joint_hybrid_synergy | +U1.5 | accretive_with_cannibalization | 118 | 8.801047084327873 | 0.9814776541011722 | -0.12066988823805325 | 4.584612604283393 | 593.0497436982607 | 1.4957241688735465 | -42.80141340560751 |
| m12_joint_hybrid_synergy | +U3.5 | accretive_with_cannibalization | 121 | 2.724281456621875 | 0.9477522428032833 | 0.6049186284955503 | 24.7905394694277 | 613.255670563405 | 1.520638850001745 | -37.591043226954525 |
| m12_joint_hybrid_synergy | +U4.5 | cannibalizing | 37 | 19.821704250169773 | 0.9763476593440416 | -0.5849857699472842 | -20.371511462061335 | 568.093619631916 | 1.4395605005288634 | -43.00581210781793 |
| m12_joint_hybrid_synergy | +O2.5 | accretive_with_cannibalization | 79 | 2.385738384281038 | 0.983094114022751 | 0.21111789562407068 | 2.494208777003678 | 590.959339870981 | 1.4490766255519745 | -45.416855584278636 |
| m12_joint_hybrid_synergy | +O3.5 | accretive_with_cannibalization | 86 | 6.10803033692179 | 0.9870501394849552 | 0.3621253921951002 | 24.074657623743292 | 612.5397887177206 | 1.4517311340943302 | -43.64975604398857 |
| m12_joint_hybrid_synergy | +O4.5 | accretive_with_cannibalization | 38 | 75.51062493410828 | 0.9963485352279514 | -0.02168760963547456 | 34.82868707409182 | 623.2938181680692 | 1.4600865076415706 | -43.67900288618434 |
| m12_joint_hybrid_synergy | +BTTS_yes | cannibalizing | 62 | 0.29506718203342164 | 0.9870951470416195 | 0.13320769243621022 | -5.161618777581566 | 583.3035123163958 | 1.4580594295024487 | -41.94189151344425 |
| m12_joint_hybrid_synergy | +BTTS_no | toxic | 51 | -9.894195060566728 | 0.9900060198336968 | 0.16723388526462735 | -3.5400057216700134 | 584.9251253723073 | 1.4855643057673174 | -41.12685182542952 |
| m05_joint_grw_smile_spine_w040 | P0 Option B | baseline | 0 | NaN | 1.0 | 0.0 | 0.0 | 576.1331019902052 | 1.5370174842308106 | -42.81333765155565 |
| m05_joint_grw_smile_spine_w040 | +U0.5 | toxic | 143 | -57.997263919597074 | 1.003309956551258 | 0.30615357469590165 | -164.91161168477606 | 411.22149030542914 | 1.3264820584142467 | -38.96582796158081 |
| m05_joint_grw_smile_spine_w040 | +U1.5 | toxic | 213 | -1.8808545203342824 | 0.9653190422521505 | -0.1059309516905067 | -59.762029702334644 | 516.3710722878706 | 1.52437653091562 | -35.70471561666817 |
| m05_joint_grw_smile_spine_w040 | +U3.5 | toxic | 18 | -1.8886638553366735 | 1.0269396907251263 | -1.6447864494321305 | -101.93366054454725 | 474.19944144565795 | 1.4295937171540463 | -44.12772959718632 |
| m05_joint_grw_smile_spine_w040 | +U4.5 | cannibalizing | 1 | 45.08 | 1.0098663137216253 | -0.7702355672980552 | -41.00674525235854 | 535.1263567378467 | 1.5230442452889874 | -41.43140991073333 |
| m05_joint_grw_smile_spine_w040 | +O2.5 | cannibalizing | 99 | 3.7443504816486413 | 0.9857157267389715 | -0.22248726667294605 | -28.416035032301238 | 547.717066957904 | 1.4957870959906432 | -42.96892650345061 |
| m05_joint_grw_smile_spine_w040 | +O3.5 | cannibalizing | 222 | 1.343202584309217 | 1.0184219213065457 | -2.258156423911066 | -141.51763498293832 | 434.6154670072669 | 1.3839596303905697 | -42.84142928739861 |
| m05_joint_grw_smile_spine_w040 | +O4.5 | cannibalizing | 94 | 8.464165850287854 | 1.0049827065202706 | -0.8636765439223151 | -27.28490805389788 | 548.8481939363073 | 1.4949954642968002 | -43.692573892453865 |
| m05_joint_grw_smile_spine_w040 | +BTTS_yes | cannibalizing | 56 | 8.56486509652783 | 0.9879356641258086 | -0.34492653679369667 | -12.689217325694244 | 563.443884664511 | 1.550641741141577 | -40.17337334420308 |
| m05_joint_grw_smile_spine_w040 | +BTTS_no | toxic | 62 | -5.811076310312956 | 0.9950331870175141 | -0.2819195022141372 | -41.62830062346234 | 534.5048013667429 | 1.5066200169445019 | -41.23746615820371 |

## Phase 2 Pareto frontier

| model | tau1 | tau2 | tau3 | survivors | total_return_pct | sharpe_ann | max_drawdown_pct | n_bets |
|---|---|---|---|---|---|---|---|---|
| m05_joint_production_wealth | 0.3 | 0.15 | 0.1 | under_15;under_35;under_45 | 104.71563506634736 | 1.5675447508350486 | -16.448793290746703 | 1532 |
| m05_joint_production_wealth | 0.3 | 0.2 | 0.0 | under_15;under_35;under_45 | 118.33064024555489 | 1.5395056316280595 | -17.418883792732736 | 1281 |
| m05_joint_production_wealth | 0.3 | 0.2 | 0.1 | under_15;under_35;under_45 | 120.45249539256169 | 1.567120801768409 | -18.18663059704595 | 1532 |
| m05_joint_production_wealth | 0.3 | 0.25 | 0.0 | under_15;under_35;under_45 | 134.3791023278609 | 1.513616021831211 | -18.99410049037238 | 1281 |
| m05_joint_production_wealth | 0.3 | 0.25 | 0.1 | under_15;under_35;under_45 | 136.40726494337832 | 1.5351374876976718 | -19.922122841800906 | 1532 |
| m05_joint_production_wealth | 0.35 | 0.2 | 0.1 | under_15;under_35;under_45 | 134.7328977761912 | 1.5415915399056621 | -19.748511744517693 | 1532 |
| m05_joint_production_wealth | 0.35 | 0.25 | 0.0 | under_15;under_35;under_45 | 150.8215071112012 | 1.5136115732220787 | -20.619241172457997 | 1281 |
| m05_joint_production_wealth | 0.35 | 0.25 | 0.1 | under_15;under_35;under_45 | 151.76146843401406 | 1.5277917367560814 | -21.450314683593522 | 1532 |
| m05_joint_production_wealth | 0.4 | 0.25 | 0.0 | under_15;under_35;under_45 | 167.74120426220676 | 1.5020455528981338 | -22.228406540511024 | 1281 |
| m05_joint_production_wealth | 0.45 | 0.25 | 0.0 | under_15;under_35;under_45 | 185.08761903879108 | 1.4829988101413452 | -23.95221590077296 | 1281 |
| m12_joint_hybrid_synergy | 0.3 | 0.25 | 0.0 | over_25;over_35;over_45;under_15;under_35 | 148.32746378946518 | 1.5121880510189147 | -19.49475264393849 | 1310 |
| m12_joint_hybrid_synergy | 0.3 | 0.25 | 0.05 | over_25;over_35;over_45;under_15;under_35 | 145.42516067224201 | 1.493200812386861 | -19.48048753408484 | 1724 |
| m12_joint_hybrid_synergy | 0.35 | 0.25 | 0.0 | over_25;over_35;over_45;under_15;under_35 | 171.20239099599291 | 1.5371656129362103 | -22.913435754793902 | 1310 |
| m12_joint_hybrid_synergy | 0.35 | 0.25 | 0.05 | over_25;over_35;over_45;under_15;under_35 | 167.56402024429104 | 1.5214928388134363 | -22.631345902607368 | 1724 |
| m12_joint_hybrid_synergy | 0.4 | 0.25 | 0.0 | over_25;over_35;over_45;under_15;under_35 | 195.33439118310625 | 1.5466811472308373 | -26.233001524654092 | 1310 |
| m12_joint_hybrid_synergy | 0.4 | 0.25 | 0.05 | over_25;over_35;over_45;under_15;under_35 | 190.9191123108597 | 1.5346959325467346 | -25.70133289832769 | 1724 |
| m12_joint_hybrid_synergy | 0.4 | 0.25 | 0.1 | over_25;over_35;over_45;under_15;under_35 | 192.8655094826261 | 1.5375798515627164 | -26.110760303539365 | 1724 |
| m12_joint_hybrid_synergy | 0.45 | 0.25 | 0.0 | over_25;over_35;over_45;under_15;under_35 | 220.69555515695387 | 1.5453013130100663 | -29.473520355557888 | 1310 |
| m12_joint_hybrid_synergy | 0.45 | 0.25 | 0.05 | over_25;over_35;over_45;under_15;under_35 | 215.47093096006037 | 1.5370381566685474 | -28.690703561424495 | 1724 |
| m12_joint_hybrid_synergy | 0.45 | 0.25 | 0.1 | over_25;over_35;over_45;under_15;under_35 | 217.54303508562586 | 1.539383233781986 | -29.09488216115564 | 1724 |
| m05_joint_grw_smile_spine_w040 | 0.3 | 0.15 | 0.0 |  | 99.19477082383811 | 1.7149222582149417 | -18.07600136111052 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.3 | 0.2 | 0.0 |  | 115.06238951280992 | 1.6863177607307396 | -19.961604124952455 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.3 | 0.25 | 0.0 |  | 131.46005812570914 | 1.641152914007011 | -21.830017276444355 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.35 | 0.2 | 0.0 |  | 129.8742738720148 | 1.6878751224593413 | -21.77567723675781 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.35 | 0.25 | 0.0 |  | 147.31980945426605 | 1.65489498037818 | -23.60914253473996 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.4 | 0.2 | 0.0 |  | 145.34147371428844 | 1.6807330472274584 | -23.57778328527148 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.4 | 0.25 | 0.0 |  | 163.87439553303386 | 1.6585585144291568 | -25.36649844098119 | 1188 |
| m05_joint_grw_smile_spine_w040 | 0.45 | 0.25 | 0.0 |  | 181.1213497753291 | 1.6549765715415263 | -27.10207719739407 | 1188 |

## Interpretation guardrails

- T012 excision is deliberately **market-level**. A BookSpec admits a complete market, so enabling one direction necessarily admits its zero-trust complement to the Kelly geometry. The report does not call that selection-level excision.
- `accretive_with_cannibalization` means net growth improved while core stake fell; it is not equivalent to a free diversification gain.
- The sweep is descriptive on the same held-out settlement period used to select survivors. Promotion needs a later untouched period.
