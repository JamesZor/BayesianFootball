# Batch 01 — geometry first, then residual mechanisms

## Objective and context

Nominate the smallest defensible extensions to C0 using **existing fits only**.
The thinker stops after this packet; Claude relays results. No new MCMC, no
candidate-model optimization, no new SQL extraction or market inversion. Kalman
filtering, RTS/FFBS conditional on saved theta, and simulation at fixed theta are
allowed. A diagnostic is not a newly fitted alternative or a prospective win.

Known baseline: 11 leagues, R2/R6/C0 × 10a/10b; r≈0.27–0.68; R6 q/s-correlation
90% intervals include zero everywhere. Published C0−R6 scores sum **marginal**
supremacy/level log densities. They are not joint predictive scores. Style goal
benefit varies across leagues. See ../THEORY.md for equations and alternatives.

## Inputs, scope and shared conventions

- Worktree: `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-qsf`,
  branch `pi/market-model-qs-forms`. Computer execution environment is managed by
  Claude; do not change or overwrite source fits or published QS results.
- Let `P=current_development/market_model`, `Q=P/results/QS`,
  `OUT=P/research/qs_forms/answers/batch_01`. CSV names below are relative to OUT.
  Diagnostic code belongs under `P/research/qs_forms/`, per manager's scope.
- Authoritative fit root on beast:
  `/root/BF_runs/market_model_qs/current_development/market_model/results/QS/fits`.
  Files: `<league>/<rung>_<protocol>.jls`. Use their existing compatible loaders,
  saved configs/panels and pinned `Q/rates_<league>.csv`; record which root supplies
  each input. Inspect serialization/column definitions rather than inventing them.
- Leagues (all requests saying ALL): `eng_premier`, `eng_championship`,
  `eng_league_one`, `eng_league_two`, `sco_premiership`, `sco_championship`,
  `sco_league_one`, `sco_league_two`, `irl_premier`, `irl_first_division`,
  `fin_veikkausliiga`.
- SENTINELS: `eng_premier`, `sco_league_two`, `sco_premiership`,
  `fin_veikkausliiga` (sharp/thin and contrasting style goal results).
- Other existing inputs: `Q/{parameters_by_league,levels_by_league,level_teams,
  identifiability_by_league,forecast_fixtures,goal_fixtures,configs,
  coverage_by_league,season_inventory}.csv`; use available columns, not guesses.
  Exact useful code: `P/l06_qs_eda.jl:316–361` (level geometry), `:463–479`
  (step rotation and goal rates); `P/l02_two_stage.jl:131–183` (filter/FFBS).
- Retain all posterior theta draws for cheap parameter algebra. For filtering or
  FFBS use 128 deterministic, evenly spaced post-warmup draws, balanced across
  chains (or all if fewer). Reuse draw indices across requests. Seed 610801;
  derive per-league/rung/protocol/replicate seeds from fixed enumerated indices,
  not runtime-dependent string hashes. Record them in `manifest.csv`.
- Units: q=(alpha−beta)/2, s=(alpha+beta)/2; D=log lambda_h−log lambda_a,
  M=(log lambda_h+log lambda_a)/2. Convert existing level conventions explicitly.
  Respect the engine's real transition intervals, breaks, priors and static states;
  do not quietly turn scheduled time into “one fixture = one week.”
- 10a is retrospective. 10b theta is training-only: forecast **every fixture of a
  test week before updating any fixture of that week**. No RTS means in honest
  predictions. Document exact training/test dates; no cross-season label guessing.
- Gauges: reproduce the existing per-side centering over the full panel roster for
  comparability. For active-team statistics also center within the fixed active
  cohort and label this sensitivity separately. Do not difference states across a
  changing centering set. Exclude inactive/gap intervals from within-season step
  diagnostics and report exclusions; retain them as separately labelled gap data.
- Intervals: posterior quantiles 5/50/95% for draw quantities; distinct block-
  bootstrap intervals for forecast statistics. Bootstrap 999 moving-block samples
  of 8 observed weeks, retaining whole weeks/all opponents, within league-season;
  use 4/12-week sensitivity for pooled score headlines. Weight league aggregates
  both by fixture count and equally. Do not call fixture SE dependence-adjusted.
- Missing metadata/unsupported conditional smoother: return `NOT_AVAILABLE` or
  `NOT_IDENTIFIABLE`, reason and impacted request. Do not fabricate fields, use
  independent marginal state draws in place of joint FFBS, or launch a replacement
  fit. Stop and report after two identical check failures.

## Reusable no-refit null (R04–R06)

Use C0 10a posterior-median theta for ALL observed RTS statistics. On SENTINELS
make 99 replicate panels at that same theta. Condition on the actual first eight
observed weeks: sample the joint filtered state at the eighth week, then simulate
all later states/observations from the engine's actual transition and observation
laws with the original fixture design and missingness. Keep the prefix unchanged;
filter and RTS-smooth each entire replicate, recomputing the same centered means
and statistics as for the real suffix. Restrict all comparisons to that suffix.
This avoids choosing an arbitrary proper replacement for diffuse initialization.
If the engine cannot simulate its conditional state correctly, flag the null as
blocked; no uncalibrated substitute verdict. No theta re-estimation.

For every statistic report observed value, null median/q05/q95, and rank
`(1 + count(null <= observed))/100`; this is a plug-in predictive rank, **not** a
calibrated frequentist p-value. Nulls condition on estimated theta and the selected
panel, so they cannot rule out selection or inversion misspecification. Reuse these
replicates across R04–R06, not a separate simulation farm per diagnostic. Store
`null_statistics.csv`: league,replicate,request,axis,stratum,statistic,value,n.

## R01 — Does a free angle add a meaningful direction?

**Question:** Is C0's ellipse genuinely aligned, or are wide zero-containing
correlation intervals concealing useful scale/angle freedom or near-rank-one fits?

**Inputs/computation:** ALL, R6/R2/C0, 10a and 10b, all theta draws. Decode scales
using existing helpers. For R6, u=sa²,v=sb²,w=rho*sa*sb;
Vq=(u+v−2w)/4, Vs=(u+v+2w)/4, C=(u−v)/4. Compute r=sqrt(Vs/Vq),
c=C/sqrt(Vq Vs), k=sa/sb, eigenvalues l+≥l− of [[Vq,C],[C,Vs]],
f_minor=l−/(l++l−), eigengap=(l+−l−)/(l++l−),
delta=atan2(2C,Vq−Vs)/2 and phi=45°−delta. Angles are axial modulo 180°;
report wrapped distance to the quality axis, not an arithmetic mean across a wrap.
Also compute variance gain from rotating `1−l−/Vs` and
`I=−0.5*log(1−c²)` (Gaussian per-step dependence information, **not forecast gain**).
C0 implies c=0,k=1; R2 implies r=1. Check covariance round-trip numerically.

**Outputs:** `geometry_draws.csv`: league,rung,protocol,chain,draw,Vq,Vs,C,r,c,k,
phi_deg,delta_deg,l_plus,l_minus,f_minor,eigengap,rotation_gain,I.
`geometry_summary.csv`: league,rung,protocol,quantity,q05,median,q95,
prob_abs_delta_gt_10deg,prob_abs_c_lt_0p1,prob_minor_lt_0p05,
prob_eigengap_lt_0p1. Probabilities may be repeated per quantity or in a separate
one-row-per-fit table; document the chosen normalized schema.

**Summary needed (≤10 lines):** one line per sentinel; three largest well-identified
tilts among ALL; largest R6–C0 residual-variance reduction; protocol sensitivity.
**Rule:** posterior P(|delta|>10°)>0.9 with eigengap>0.1 in ≥90% of draws nominates
free covariance, conditional on R03 prediction. A 90% interval wholly within ±10°
supports practical alignment. Otherwise inconclusive, not “zero correlation.”
Minor fraction<0.05 suggests a rank-reduction test, not proof style is unnecessary.

## R02 — Are level geometry and step geometry really the same?

**Question:** Does the apparent level r survive uncertainty, unequal variances and
smoother shrinkage? Is static style a better candidate than no style?

**Inputs/computation:** ALL, R2/R6/C0 **10a only**, 128 theta draws with one exact
joint FFBS path each; also median-theta RTS point means. Reproduce existing
`Q/levels_by_league.csv` and `level_teams.csv` statistics within 1e−6 first, including
the published all-roster centering and season-average window. Then compute from
each path the covariance across active teams of season-average (q,s): direct
r_level=SD(sbar)/SD(qbar), c_level, k_level=SD(alphabar)/SD(betabar), rho_ab,
shortcut_r=sqrt((1+rho_ab)/(1−rho_ab)), eigen-angle and level/step ratio
`r_level/r_step(theta)`. Also average weekly cross-sectional Var(q),Var(s),Cov(q,s)
over that season and take their implied ratio; a ratio of averages is not an
average of ratios. Label partial seasons, roster changes and season length.
This is posterior latent-population geometry, not independent team replicates.

**Outputs:** `level_geometry.csv`: league,rung,protocol,season,method,gauge,n_teams,
n_weeks,quantity,q05,median,q95 (method=RTS_point or FFBS; deterministic quantiles
identical). `level_geometry_draws.csv`: league,rung,season,draw,gauge,r_level,
c_level,k_level,rho_ab,shortcut_r,delta_level_deg,r_step,level_step_ratio,
r_weekly_cross_section. Save selected centered team-season path means for R06/R08.

**Summary needed (≤10 lines):** sentinel ratios with intervals; largest shortcut
errors; strongest level/step discrepancies; R2 versus R6/C0 sensitivity.
**Rule:** P(level_step_ratio<0.75 or >1.25)>0.9 nominates separate level/step
geometry (not necessarily OU). An error in shortcut_r implicates unequal variances,
not hidden dynamics. High sensitivity to RTS versus FFBS or rung weakens the claim
that level geometry is data-determined. Zero step style and zero level style remain
separate future models.

## R03 — What does the honest forecast fail to explain?

**Question:** Are angle freedom, different axis noise, serial dynamics, common
intensity or team×venue effects visible before the observation update?

**Inputs/computation:** ALL, C0 and R6 10b, 128 theta draws. For every test fixture
extract pre-week-update Gaussian predictive (D,M) mean m_j and covariance S_j,
including measurement noise and state uncertainty. Mixture score:
`logsumexp(log N_2(y;m_j,S_j))−log J`. Also compute each coordinate's mixture
log density and their sum; reproduce the published marginal-score convention
separately if it uses different coordinates, Jacobians or median theta. Do not
expect joint scores to equal marginal sums. Record conversions/constants.
Mixture moments m=mean(m_j), S=mean(S_j+m_j*m_j')−m*m'. Residuals u_D and u_M
are coordinatewise `(y−m)/sqrt(diag S)`, not a whitening that hides axis correlation.
Report means, mean squares, cross-correlation; 90% marginal predictive coverage
using mixture CDF quantiles. Aggregate weekly residual means and test their lag
1/2/4 correlations, signed and squared. Report per-team signed residual series:
q-score proxy = u_D for home and −u_D for away; s proxy = u_M in either role.
Compute pooled within-team lag-one covariance and home-minus-away mean contrasts
per axis; exclude teams with <8 appearances per role for role contrasts. Normalize
contrast by its week-block-bootstrap SE, report a distribution not a selected star.

**Outputs:** `forecast_joint.csv`: league,season,week,fixture_id,rung,n_draws,
y_D,y_M,mean_D,mean_M,var_D,var_M,cov_DM,u_D,u_M,logp_joint,logp_D,logp_M,
covered90_D,covered90_M,home_team,away_team.
`forecast_diagnostics.csv`: league,rung,axis,statistic,n,value,boot_q05,boot_q95.
`joint_comparison.csv`: league,weighting,n,score_kind,mean_C0_minus_R6,
boot_q05,boot_q95,block_weeks. `venue_contrasts.csv`: league,rung,team,axis,
n_home,n_away,contrast,bootstrap_se.

**Summary needed (≤10 lines):** pooled fixture/equal-league joint Δ; largest
joint/marginal disagreement; sentinel axis calibration; strongest serial and role
patterns with uncertainty (not dozens of unadjusted significance claims).
**Rule:** 90% block interval entirely inside ±0.005 nats/fixture supports practical
predictive equivalence at this tolerance; entirely below −0.005 favors testing free
covariance. Wide intervals are inconclusive. Serial signed versus squared errors
nominate trend versus volatility; systematic role errors nominate venue structure.
Axis undercoverage alone does not separate process noise from market noise.

## R04 — Reversion, momentum, or simply different clock speeds?

**Question:** Which departures from a random walk survive a matched smoother null?

**Inputs/computation:** ALL C0 10a median-theta RTS; shared null for SENTINELS.
Use within-season teams active throughout each compared interval, fixed season
cohort centering, suffix after eight observed weeks. For each k in {q,s}, pool
`G_k(h)=mean((z_i,t+h−z_i,t)^2)` for h=1,2,4,8,13 calendar-week slots, never
bridging a season boundary; report pair counts. Report G(h)/(h G(1)) where defined
and actual schedule process-time ratios if not h. For consecutive eligible weekly
steps compute corr(Delta z_t,Delta z_t−1) and corr(Delta z_t²,Delta z_t−1²).
OLS screen `Delta z_i,t = b z_i,t−1 + error`, pooling centered states with an
intercept, on the same eligible rows; b<0 is only an OU nomination. Do not regress
on estimated future states and label that honest. Repeat G and b on median-theta
R6 10a RTS on SENTINELS as covariance-model sensitivity, without extra nulls.

**Outputs:** `dynamics_screen.csv`: league,rung,axis,statistic,lag,n,value,
null_q05,null_median,null_q95,null_rank,null_status.
**Summary needed (≤10 lines):** each sentinel's q/s reversion and lag signature;
strongest differential variogram shape; R6 sensitivity; missing/weak support.
**Rule:** a coherent directional pattern outside the matched 90% null envelope
at ≥2 lags, corroborated by R03, prioritizes OU (flattening/negative drift), trend
(short-lag positive increments) or volatility (squared dependence). Isolated tail
ranks or RTS slopes alone do not qualify. Different sigma's alone are not different
correlation time scales; no calibrated half-life estimate from this screen.

## R05 — Who or when carries volatility, and are there jumps?

**Question:** Is extra motion persistent by team, predictable by calendar, clustered
in time, or a few possibly bad-book events?

**Inputs/computation:** reuse R04 C0 RTS and null; ALL observed, SENTINEL nulls.
On eligible one-week steps let E_i,k,t=(Delta z_i,k,t)²/sigma_k², using actual
scheduled step variance instead if different. For teams with ≥12 steps report
mean E by team and its cross-team coefficient of variation; compare first/second
half-season team ranks (Spearman). For season phase use normalized week position
u=(week−first)/(last−first), bins [0,.2), [.2,.8], (.8,1]; report energy ratios to
middle, separately for q/s. Separately record last-appearance→next-appearance gap
length and net state change; do not label all gaps transfer windows.
Report pooled step max/median squared magnitude and top-5%-of-steps share of total
energy (NA if denominator zero). List 10 largest |Delta| events per sentinel/axis
with team/week, surrounding fixture IDs and available inversion diagnostics.
Report weekly cross-team mean energy, lag-one correlation, and mean off-diagonal
centered-step covariance (use the null because centering induces negative values).
Finally bin prior-week |q| into within-league-season thirds and compare q/s E
across bins; state dependence competes with fixed team volatility.

**Outputs:** `volatility_screen.csv`: league,axis,stratum,statistic,n,value,
null_q05,null_median,null_q95,null_rank,null_status.
`team_energy.csv`: league,season,team,axis,n,energy_first,energy_second,energy_all.
`large_moves.csv`: league,axis,team,week,delta,scheduled_sd,previous_gap_weeks,
fixture_ids,available_book_flags (no invented manager/window labels).
**Summary needed (≤10 lines):** team stability; calendar ratios; tail concentration;
energy clustering; co-movement; whether the largest moves have book-quality flags.
**Rule:** excess dispersion **and** stable team energy ranks favors team volatility;
repeatable phase contrasts favors calendar Q_t; concentration favors jump/noise
investigation; clustered energy without calendar explanation favors SV. Uniform
q shocks are unidentifiable; no centered covariance result establishes them.

## R06 — Are tiers or nonlinear quality–style relations hiding in levels?

**Question:** Is there evidence beyond a Gaussian tilted ellipse, enough to justify
tiers, nonlinear s(q), or radial dynamics?

**Inputs/computation:** ALL C0/R6 10a median-theta RTS and R02 FFBS season means.
For each league-season with ≥8 active teams, center/scale q by that season's SD.
Fit descriptive OLS `sbar = a + b qstd + c(qstd²−1)`, separately for point means and
for each path (never treat paths as additional observations). Report b,c,R² and
leave-one-team-out squared-error change quadratic versus intercept and versus
linear; tiny samples are a warning, not a reason to fit more mixture components.
For qstd compute skewness, excess kurtosis and largest adjacent order-statistic
gap divided by IQR. Label the teams on each side of the largest gap; measure the
fraction of shared teams remaining on the same side next season, allowing label
alignment but not searching multiple gaps to maximize persistence. Apply the same
point-mean statistics to the shared C0 null suffix season means (same rows/window)
on SENTINELS; keep full-season posterior geometry separate. No mixture fitting.
Use R05's |q|-energy test as the radial-volatility screen, not a redundant new fit.

**Outputs:** `nonlinear_levels.csv`: league,rung,season,window,method,quantity,n,
q05,median,q95,null_q05,null_median,null_q95,null_rank.
`tier_gaps.csv`: league,rung,season,window,n,gap_over_iqr,left_team,right_team,
next_season_shared_n,same_side_fraction,null_rank.
**Summary needed (≤10 lines):** strongest reproducible curvature; whether it
improves leave-team-out error; tier/gap persistence versus null; rung sensitivity.
**Rule:** curvature with consistent sign across seasons/rungs and better leave-
team-out error, beyond the null, nominates s(q). Large tails without persistent
gaps favor a heavy-tailed q prior before mixtures. Stable gaps beyond null merely
nominate tier mixtures; neither proves tiers or a predictive win.

## R07 — Does style help truth, and on which outcome channel?

**Question:** Is the goal benefit real under dependence-aware uncertainty, and does
it require dynamic style or only persistent team style?

**Inputs/computation:** ALL C0 10b, same 128 draws/forward filtering as R03; goal
outcomes from the existing fixture panel / `Q/goal_fixtures.csv`. First reproduce
the published full−no-style and full−no-quality plug-in scores on exactly their
3,053-fixture intersection (or explain any mismatch before interpreting new scores).
New variants: full; no_style (s=0); no_quality (q=0); frozen_style (s_i frozen at
its training-end filtered value; newly appearing teams use training-time entry
prior, never their first test book). Keep q,mu,gamma and the forward filter from
C0 unchanged in every ablation: these are deletions, **not refitted competitors**.
For uncertainty-integrated scores, draw 4 joint Gaussian state samples per theta
per forecast week (shared across variants/fixtures), including correlated frozen
and current states via a joint conditional Gaussian construction if available.
If not, omit uncertainty-integrated frozen_style and report its plug-in version;
never splice independent frozen/current draws. Compare first 2 versus all 4 state
samples; if pooled deltas change by >0.001 nats/fixture, report MC_UNSTABLE rather
than an apparent model difference or an unbounded sample increase.
For sample j, compute lambda_h/a using existing `goal_rates`; then
N~Pois(Lambda_j) and H|N~Binomial(N,p_j), p_j=logistic(D_j).
Report `L_joint=logmeanexp(log Pois(H;lh_j)+log Pois(A;la_j))`,
`L_total=logmeanexp(log Pois(N;Lambda_j))`, and
`L_allocation=L_joint−L_total`. This last quantity is the mixture's exact
conditional log score; do not independently mix Binomial scores with equal
weights (conditioning on N reweights latent draws). Sum channel deltas exactly.

**Outputs:** `goal_ablation_fixtures.csv`: league,season,week,fixture_id,variant,
method,logp_joint,logp_total,logp_allocation,n_latent_samples.
`goal_ablation_summary.csv`: league,weighting,method,comparison,channel,n,
mean_delta,boot_q05,boot_q95,block_weeks,mc_delta_change,status.
**Summary needed (≤10 lines):** reproduction check; pooled full−no-style and
full−frozen-style by channel; Scotland/Finland contrast; equal-league versus
fixture weighting; Monte Carlo sensitivity.
**Rule:** persistent benefit in total-goal scoring supports usable style signal,
not necessarily its dynamic law. Frozen-style parity nominates static style;
full superiority nominates retaining motion. Goal heterogeneity supports league
pooling/shrinkage but not any causal market-sharpness explanation. Even a positive
ablation score is not evidence against an independently refitted simpler model.

## R08 — League hierarchy, cross-division bridges, and market confounding

**Question:** Is log r heterogeneity structured enough to prioritize pooling, and
what can existing snapshots actually say about microstructure or division links?

**Inputs/computation:** ALL C0/R6 both protocols, theta draws; R02 levels; existing
Q rates/configs/coverage/season inventory and R07 goal summaries. Tier mapping:
England 1/2/3/4, Scotland 1/2/3/4, Ireland 1/2, Finland 1. Define topflight=1 for
tier 1. For each protocol, combine independent posterior draws across leagues
using fixed-seed permutations; compute across-league SD(log r), descriptive OLS
slope log r versus tier, topflight−other mean contrast, and within-country tier
slopes for England/Scotland (Ireland two-point contrast; Finland no slope).
These quantify fit uncertainty for these 11 leagues, not population sampling
uncertainty or a fitted hierarchical model. Report leave-one-league-out ranges of
point slopes. Compute Spearman(log r, sigma_obs) using matched draws; R6 k/delta
versus tier are optional algebra-only companions. Never label sigma_obs “liquidity.”

Inventory fields actually present for overround, inversion loss, selection count,
market family availability, timestamps/age, depth/spread and quote update history.
`field_inventory.csv` must say source and present/absent; read only already saved
snapshots, no network fetch. Where available, aggregate the first four by league
and correlate with log r (descriptive Spearman, leave-one-league-out range). Do
not invent a composite sharpness score, use goal prediction error as a causal
sharpness covariate, or conflate lower book acceptance with stale totals. Without
market-family timestamps/update histories, asynchronous repricing is explicitly
NOT_IDENTIFIABLE from this batch. Report per-league r alongside R07 style benefit;
no causal regression with n=11 and multiple confounders.

Finally match stable team IDs across adjacent seasons in different divisions of
the same country using R02/path panel IDs (names only if uniquely verified).
Report movers' old/new within-division q percentile and centered s, move direction,
and uncertainty. Do not estimate absolute division offsets. Require ≥5 matched
movers for a pooled rank correlation; otherwise inventory only. If panel IDs cannot
be joined safely, return NOT_AVAILABLE rather than fuzzy name matches.

**Outputs:** `league_structure.csv`: protocol,rung,quantity,q05,median,q95,
n_leagues,leave_one_out_min,leave_one_out_max,interpretation.
`field_inventory.csv`: field,source,present,unit,coverage,limitation.
`market_confounding.csv`: league,protocol,r_median,sigma_obs_median,field,n,
field_summary,style_goal_delta,status.
`division_movers.csv`: country,team_id,from_league,to_league,from_season,to_season,
quantity,q05,median,q95,status (quantities include old/new q percentile, s and
changes; no absolute global q). `mover_summary.csv`: country,n,quantity,value,status.

**Summary needed (≤10 lines):** heterogeneity and within-country tier direction;
leave-one-out sensitivity; available/missing microstructure fields; mover counts
and continuity; explicit statement of what cannot be separated.
**Rule:** stable differences across protocols with consistent within-country tier
direction prioritize hierarchical log r; weak/no tier trend still permits exchangeable
pooling. No fitted pooling win is established here. Microstructure remains a
hypothesis unless directly timed/quality-matched observations separate it from
football. Movers motivate a bridge fit only if identities and continuity support it.

## Return, verification and grading

Create `SUMMARY.md`: status per R01–R08, at most ten summary lines per request,
links to CSVs, null/MC/coverage caveats, and a short list of nominated future tests.
No final claim that a model not fitted here beats C0. Record input hashes, draw
counts, seeds, conditional simulation count, exclusions, runtime and exact commands
in `manifest.csv` / `VERIFICATION.md`. Keep output summaries compact for Astra.

Minimum executable checks in your diagnostic runner (record the actual invocation
and assertion output, not just prose promises):
1. R01 covariance round-trip relative error ≤1e−10, R2 r=1 and C0 c=0 to tolerance;
   handle zero denominators as NA, never silently drop them.
2. R02 published point-level quantities reproduced within 1e−6; list any mismatched
   rows and resolve indexing/centering differences before using new comparisons.
3. R03 assertion that each forecast uses only observations with week strictly less
   than its own, and theta training ends before test starts; predictive covariances
   PSD, finite log scores, unique fixture keys, full paired intersection counts.
4. R04–R06 exactly 99 shared conditional null replicates per sentinel or explicit
   BLOCKED status; no fake null envelopes for leagues without simulation.
5. R07 published fixture count/headlines reproduced to their reported rounding or
   documented methodological mismatch; `joint = total + allocation` within 1e−10;
   compare MC halves and confirm whole-week pairing across variants.
6. All output tables include units/method/protocol or carry them unambiguously in
   their documented schema. Nonfinite/unavailable quantities have reason codes.
7. `git diff --check` passes; changed paths confined to `research/qs_forms/`;
   source fits and canonical QS outputs unchanged. Run `./scripts/todo.sh check`
   and report pre-existing failures without editing out-of-scope task files.

**Budget/escalation:** prioritize R01–R03, then reuse caches for R04–R08. No new
sampling runs or alternative fits to rescue a blocked diagnostic. If the planned
FFBS/simulation work is unexpectedly expensive, return completed requests and the
measured blocker to Claude; do not silently downsample or expand the budget.

**How you are graded:** pass = reproducible computations, leakage/gauge checks,
matched nulls, explicit uncertainty and availability, compact summary. Partial =
correctly reported blocked requests. Failure = invented evidence/metadata, a new
MCMC fit, smoothing leakage called forecasting, or silent substitution of posterior
means for uncertainty-integrated quantities. Stop after writing the return; Claude
will relay it. No polling and no additional batch without the thinker/manager.
