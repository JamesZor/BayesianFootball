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

