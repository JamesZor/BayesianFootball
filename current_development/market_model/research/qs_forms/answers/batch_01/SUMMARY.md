# Batch 01 results

Input/output hashes, exact roots, row counts and seeds: [manifest.csv](manifest.csv).
Large tables remain on the beast, per manager instruction. Units: natural-log rates, calendar weeks, nats.

## R01 —
COMPLETE. [geometry_summary.csv](geometry_summary.csv); draw table is beast-only (manifest). Probabilities repeated per quantity; zero denominators are blank, not dropped.
eng_premier R6 10a delta=6.68° [-3.5,15.17]; P(|delta|>10°)=0.283.
sco_league_two R6 10a delta=1.44° [-3.16,5.75]; P(|delta|>10°)=0.0.
sco_premiership R6 10a delta=4.36° [-2.7,10.85]; P(|delta|>10°)=0.079.
fin_veikkausliiga R6 10a delta=9.49° [-9.53,23.01]; P(|delta|>10°)=0.524.
Three largest median tilts with ≥90% eigengap>0.1: fin_veikkausliiga 9.49° (P>10°=0.524); eng_premier 6.68° (P>10°=0.283); sco_premiership 4.36° (P>10°=0.079).
Largest R6 rotation residual-variance reduction 1−l_minus/Vs: irl_first_division 10b, median 0.0625 [0.0006,0.3441]; this is not a forecast gain or refitted R6–C0 comparison.
Largest 10a/10b median tilt change: eng_championship 3.57°. No R6 10a fit meets P(|delta|>10°)>0.9: true.

## R02 —
COMPLETE. [level_geometry.csv](level_geometry.csv), [reproduction_checks.csv](reproduction_checks.csv), [season_windows.csv](season_windows.csv); path means/draw table locations and hashes in manifest. Exact joint FFBS, 128 theta draws/fit; fixed gauges never differenced across changing rosters.
eng_premier C0 level/step: 22/23 0.37 [0.34,0.41]; 23/24 0.46 [0.41,0.5]; 24/25 0.49 [0.43,0.53]; 25/26 0.5 [0.45,0.55].
sco_league_two C0 level/step: 21/22 0.54 [0.41,0.69]; 22/23 1.35 [1.06,1.64]; 23/24 0.81 [0.64,1.02]; 24/25 0.92 [0.72,1.22]; 25/26 1.74 [1.42,2.1].
sco_premiership C0 level/step: 21/22 0.47 [0.41,0.55]; 22/23 0.43 [0.37,0.49]; 23/24 0.44 [0.37,0.51]; 24/25 0.33 [0.27,0.38]; 25/26 0.31 [0.26,0.36].
fin_veikkausliiga C0 level/step: 2021 0.22 [0.19,0.27]; 2022 0.32 [0.27,0.38]; 2023 0.41 [0.34,0.5]; 2024 0.39 [0.33,0.46]; 2025 0.33 [0.29,0.4].
Largest FFBS shortcut absolute error: sco_championship R6 21/22, 1.342177269416e8; shortcut assumes equal alpha/beta variances, not zero q/s correlation.
Level/step screen >0.9 in 230 league/rung/season/gauge cells; [level_step_screen.csv](level_step_screen.csv). Partial-season labels and roster membership are in season_windows; these are latent-population, not independent team replicates.

## R03 —
COMPLETE. [forecast_joint.csv](forecast_joint.csv), [forecast_diagnostics.csv](forecast_diagnostics.csv), [joint_comparison.csv](joint_comparison.csv), [venue_contrasts.csv](venue_contrasts.csv). D=h−a, M=(h+a)/2; determinant=1, no Jacobian constant. Published median-theta marginal scores reproduced ≤1e−8.
Pooled fixture joint C0−R6 0.00031 [-0.00018,0.00138] nats/fixture; n=3053.
Pooled equal_league joint C0−R6 0.00066 [-4.0e-5,0.00217] nats/fixture; n=3053.
eng_premier C0 D mean_square=0.958; D coverage90=0.903; M mean_square=0.622; M coverage90=0.966.
sco_league_two C0 D mean_square=1.434; D coverage90=0.871; M mean_square=0.957; M coverage90=0.919.
sco_premiership C0 D mean_square=1.306; D coverage90=0.845; M mean_square=0.839; M coverage90=0.933.
fin_veikkausliiga C0 D mean_square=1.292; D coverage90=0.845; M mean_square=1.059; M coverage90=0.915.
Serial/team lag uncertainty uses 999 week-block resamples of original lag-pair rows indexed by their later week, without fabricating cross-block neighbors. Team lag means appearance-lag one within season. Venue contrast/SE distribution is unselected with ≥8 appearances per role; tables retain all teams, not selected stars.

## R04 —
COMPLETE. [dynamics_screen.csv](dynamics_screen.csv), [step_exclusions.csv](step_exclusions.csv). Fixed active-season cohort centering; conservative eligibility requires a fixture appearance in every compared calendar slot, so missing books/byes are excluded as gaps. All G ratios use actual weekly Q (process-time ratio=h).
eng_premier q increment_lag1_correlation=0.56 (rank=0.42); q squared_increment_lag1_correlation=0.294 (rank=0.27); q OLS_delta_on_prior_level=0.001 (rank=0.19); s increment_lag1_correlation=0.7 (rank=0.68); s squared_increment_lag1_correlation=0.44 (rank=0.16); s OLS_delta_on_prior_level=-0.003 (rank=0.01).
sco_league_two q increment_lag1_correlation=0.469 (rank=0.12); q squared_increment_lag1_correlation=0.218 (rank=0.25); q OLS_delta_on_prior_level=-0.003 (rank=0.03); s increment_lag1_correlation=0.817 (rank=0.47); s squared_increment_lag1_correlation=0.606 (rank=0.23); s OLS_delta_on_prior_level=-0.005 (rank=0.01).
sco_premiership q increment_lag1_correlation=0.582 (rank=0.73); q squared_increment_lag1_correlation=0.379 (rank=0.86); q OLS_delta_on_prior_level=0.001 (rank=0.5); s increment_lag1_correlation=0.776 (rank=0.85); s squared_increment_lag1_correlation=0.546 (rank=0.43); s OLS_delta_on_prior_level=0.001 (rank=0.39).
fin_veikkausliiga q increment_lag1_correlation=0.512 (rank=0.2); q squared_increment_lag1_correlation=0.245 (rank=0.26); q OLS_delta_on_prior_level=0.002 (rank=0.71); s increment_lag1_correlation=0.668 (rank=0.53); s squared_increment_lag1_correlation=0.446 (rank=0.48); s OLS_delta_on_prior_level=-0.001 (rank=0.11).
99 shared conditional C0 null panels/sentinel, suffix only after first eight observed weeks. Null ranks are plug-in predictive ranks, not p-values; non-sentinels and R6 sensitivity have no envelopes. No half-life is estimated; isolated RTS slopes are not dynamic-law evidence.


## R05 —
COMPLETE. [volatility_screen.csv](volatility_screen.csv), [team_energy.csv](team_energy.csv), [large_moves.csv](large_moves.csv), [large_move_books.csv](large_move_books.csv), [appearance_gaps.csv](appearance_gaps.csv). All observed suffixes, same99 conditional nulls on sentinels; gaps never labelled transfer windows.
team_first_second_energy_spearman: eng_premier/q -0.019 (rank=0.37); eng_premier/s -0.023 (rank=0.23); sco_premiership/q 0.071 (rank=0.65); sco_premiership/s -0.088 (rank=0.23); sco_league_two/q 0.125 (rank=0.8); sco_league_two/s 0.095 (rank=0.5); fin_veikkausliiga/q -0.003 (rank=0.51); fin_veikkausliiga/s -0.157 (rank=0.13).
cross_team_energy_CV: eng_premier/q 0.479 (rank=0.99); eng_premier/s 0.427 (rank=0.31); sco_premiership/q 0.486 (rank=0.83); sco_premiership/s 0.548 (rank=0.6); sco_league_two/q 0.624 (rank=1.0); sco_league_two/s 0.532 (rank=0.13); fin_veikkausliiga/q 0.433 (rank=0.29); fin_veikkausliiga/s 0.471 (rank=0.19).
top5pct_energy_share: eng_premier/q 0.318 (rank=1.0); eng_premier/s 0.285 (rank=0.62); sco_premiership/q 0.295 (rank=0.71); sco_premiership/s 0.298 (rank=0.81); sco_league_two/q 0.315 (rank=0.94); sco_league_two/s 0.243 (rank=0.02); fin_veikkausliiga/q 0.311 (rank=0.95); fin_veikkausliiga/s 0.302 (rank=0.77).
weekly_mean_energy_lag1_correlation: eng_premier/q 0.535 (rank=0.98); eng_premier/s 0.527 (rank=0.86); sco_premiership/q 0.492 (rank=0.91); sco_premiership/s 0.454 (rank=0.42); sco_league_two/q 0.172 (rank=0.33); sco_league_two/s 0.268 (rank=0.03); fin_veikkausliiga/q 0.235 (rank=0.42); fin_veikkausliiga/s 0.26 (rank=0.16).
mean_offdiagonal_centered_step_covariance: eng_premier/q -0.0 (rank=0.01); eng_premier/s -0.0 (rank=1.0); sco_premiership/q -0.0 (rank=0.04); sco_premiership/s -0.0 (rank=1.0); sco_league_two/q -0.0 (rank=0.12); sco_league_two/s -0.0 (rank=0.99); fin_veikkausliiga/q -0.0 (rank=0.01); fin_veikkausliiga/s -0.0 (rank=1.0).
Calendar ratios and prior-|q| tertiles are retained per axis/season in volatility_screen; nulls condition on estimated theta and selected books, so none separates inversion noise/selection from football shocks. Available book diagnostics are KL/selection count/optimizer-start spread, not quote age or exchange depth.


## R06 —
COMPLETE descriptive screen. [nonlinear_levels.csv](nonlinear_levels.csv), [tier_gaps.csv](tier_gaps.csv). Full-season posterior geometry remains separate from matched-null suffix point means. 128 paths per fit, never paths-as-extra-teams; LOTO deltas are quadratic minus comparator per team (negative is better).
Three largest C0 FFBS |curvature|: irl_first_division/2022 c=0.0367 [0.0242,0.051]; sco_championship/24/25 c=0.0343 [0.0282,0.0395]; irl_first_division/2021 c=-0.0305 [-0.0459,-0.0173].
eng_premier gap/IQR; rank; next-season shared/same-side: 22/23 0.51;0.34;17/0.882; 23/24 0.808;0.63;17/0.824; 24/25 0.604;0.47;17/0.882; 25/26 0.603;0.41;0/NA.
sco_league_two gap/IQR; rank; next-season shared/same-side: 21/22 2.298;0.82;6/0.833; 22/23 0.776;0.46;7/0.857; 23/24 0.55;0.19;7/0.571; 24/25 0.791;0.44;6/1.0; 25/26 1.074;0.69;0/NA.
sco_premiership gap/IQR; rank; next-season shared/same-side: 21/22 1.701;0.9;11/1.0; 22/23 2.247;0.98;11/1.0; 23/24 2.798;1.0;11/1.0; 24/25 2.05;0.98;10/0.9; 25/26 0.565;0.29;0/NA.
fin_veikkausliiga gap/IQR; rank; next-season shared/same-side: 2021 1.021;0.78;11/0.818; 2022 0.379;0.06;11/0.636; 2023 0.686;0.45;10/0.8; 2024 0.505;0.25;10/0.6; 2025 0.298;0.02;0/NA.
R6/full-season curvature and both LOTO changes retained per season; consistent signs/rung sensitivity must be read jointly, not selected by largest magnitude. Null gap ranks do not establish persistent tiers; tiny n (8–24) limits shape inference. R05 prior-|q| energy is the radial screen, with no new fit.

## R07 —
PARTIAL: all published3053 fixtures reproduced (≤1e−8); [goal_reproduction_checks.csv](goal_reproduction_checks.csv), [goal_ablation_fixtures.csv](goal_ablation_fixtures.csv), [goal_ablation_summary.csv](goal_ablation_summary.csv). Integrated frozen-style NOT_AVAILABLE: no joint training-end/current conditional cache; plug-in frozen-style is available, never independently spliced.
full_minus_no_style/mixture_128x4/fixture joint=0.00671 [0.00308,0.00923],MCchange=-0.00016,OK; total=0.00659 [0.00307,0.00919],MCchange=-0.00013,OK; allocation=0.00012 [-1.0e-5,8.0e-5],MCchange=-3.0e-5,OK.
full_minus_no_style/mixture_128x4/equal_league joint=0.00845 [0.00375,0.0116],MCchange=-0.00025,OK; total=0.00833 [0.00375,0.01155],MCchange=-0.00022,OK; allocation=0.00012 [-3.0e-5,8.0e-5],MCchange=-3.0e-5,OK.
full_minus_frozen_style/plugin_median_theta/fixture joint=0.01574 [0.00766,0.01761],MCchange=NA,OK; total=0.01574 [0.00766,0.01761],MCchange=NA,OK; allocation=0.0 [-0.0,0.0],MCchange=NA,OK.
full_minus_frozen_style/plugin_median_theta/equal_league joint=0.01814 [0.01,0.02104],MCchange=NA,OK; total=0.01814 [0.01,0.02104],MCchange=NA,OK; allocation=-0.0 [-0.0,0.0],MCchange=NA,OK.
sco_league_two full−no-style joint 0.01309 [-0.00469,0.0333] OK; total 0.01296 [-0.00457,0.03334] OK; allocation 0.00012 [-0.00017,0.00022] OK.
sco_premiership full−no-style joint -0.01852 [-0.02741,-0.00183] OK; total -0.01836 [-0.02739,-0.00189] OK; allocation -0.00016 [-0.00016,6.0e-5] OK.
fin_veikkausliiga full−no-style joint 0.04916 [0.01292,0.06577] OK; total 0.04886 [0.01268,0.06575] OK; allocation 0.00031 [-8.0e-5,0.00032] OK.
Mixture allocation=joint−total exactly (not equally weighted Binomial mixing). MC_UNSTABLE means first2 versus all4 samples change delta by>0.001; no extra samples. Ablations retain C0 filter/q/mu/gamma: deletion scores do not compare refitted models.
