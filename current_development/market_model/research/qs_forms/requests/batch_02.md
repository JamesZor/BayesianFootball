# Batch 02 — resolve the remaining structural ambiguities

## Objective, evidence and scope

Sharpen the final recommendation, not fit a winning replacement. Batch 01 supports
45° coordinates (pooled joint C0−R6 +0.00031 nats/fixture, 90% block interval inside
±0.005) and league variation in r. It also exposes small level/step ratios, very
sparse long-horizon pairs, a Scottish Premiership quality gap, and bootstrap score
intervals whose centering needs checking. These motivate **four requests only**.

No new MCMC, candidate-model fitting, SQL, quote extraction or inversion. Reuse
existing point paths, 128 FFBS paths, forecast/goal rows and the 99 conditional-null
panels per sentinel. No replacement of unavailable integrated frozen-style with
independent state draws. Quote-level repricing and division movers remain partial:
there is no request to solve missing-data problems by widening scope.

- Worktree/branch: `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-qsf`,
  `pi/market-model-qs-forms`; source relay `a95ca9ca`.
- Let `P=current_development/market_model`, `A1=P/research/qs_forms/answers/batch_01`,
  `OUT=P/research/qs_forms/answers/batch_02`. Write diagnostics/results only under
  `P/research/qs_forms/`; do not overwrite A1 artifacts or canonical QS fits/results.
- Exact file locations and hashes: `A1/manifest.csv`. Beast cache root is
  `/root/BF_runs/market_model_qsf/current_development/market_model/research/qs_forms/answers/batch_01`.
  Caches include `cache/<league>_C0_point.jls` and
  `cache/<league>_conditional_nulls.jls`; `team_season_path_means.csv` holds R02 paths.
  Original fits remain under `/root/BF_runs/market_model_qs/current_development/market_model/results/QS/fits`.
- ALL means the same 11 leagues as batch 01. SENTINELS means eng_premier,
  sco_premiership, sco_league_two, fin_veikkausliiga. New random seed: 610802;
  record derived seeds explicitly. Existing simulation draws must be reused.
- Read schema/units before computing. D=log lambda_h−log lambda_a,
  M=(log lambda_h+log lambda_a)/2. Preserve fixed cohorts and actual calendar slots.
- Null ranks retain their batch 01 meaning: `(1+#null<=observed)/100`, conditional
  plug-in predictive ranks, not p-values. Never attach them to an unmatched window,
  gauge, cohort or statistic; label posterior intervals separately.
- All summaries ≤10 lines per request. Outputs need method, units, counts, availability
  and reasons for NA. Report failures after two identical check failures, not a third
  variation. No polling or autonomous third batch; return to Claude when finished.

## R09 — Is “style” a league scoring state, a relative team state, or both?

**Question:** Does centering discard the useful style signal, making level/step
comparisons misleading? This distinguishes a dynamic league intensity from OU on
relative team style without fitting either alternative.

**Inputs:** SENTINELS, C0 10a raw median-theta RTS caches and their 99 conditional
nulls; ALL C0 10b saved fits/panels, median theta only, and A1 goal/forecast rows.
Re-filter 10b at fixed theta if required; no new parameter draws needed.

**Retrospective calculation:** For each fixed active season roster C, n=|C|,
form b_t=mean_C(s_i,t), r_i,t=s_i,t−b_t on the R04 suffix. Keep this cohort fixed
when differencing, even when a club has no fixture that week. Use all within-season
calendar transitions; separately show the old continuous-appearance selection.
For every k=q,s calculate uncentered energy `mean_i[(Delta z_i)^2]`, common energy
`(mean_i Delta z_i)^2`, and relative energy `mean_i(Delta z_i−mean Delta z)^2`.
The first quantity means the **mean of squared increments**, not square of the mean.
Check exactly: uncentered energy = common + relative. Compare each energy and the
common fraction with the same statistic from every conditional null. These are
RTS mean energies, not latent process variance estimates. Never equate their ratio
to 1/n without the matched smoother null.

Audit the R05 off-diagonal estimator. For a complete centered increment vector v,
`mean_{i!=j}(v_i v_j)=−sum_i(v_i²)/(n(n−1))`. Check this identity per week and
explain any discrepancy from R05 (e.g. pairwise missingness or time demeaning).
If temporal sample covariances were used, give the corresponding sum-zero covariance
identity. Do not call this redundant energy statistic evidence of common shocks.

**Honest style partition:** At 10b training end freeze a reference cohort C equal
to the test season's roster from existing season metadata. If only fixture-design
membership is available, explicitly condition on that design, without using any
future rates/outcomes to select it. If neither is trustworthy, report NOT_AVAILABLE
for this part. Entrants' training-end values must use their existing entry priors.
Let b0=training-end filtered mean of s over C; for each whole-week pre-update
forecast let b_t=mean_C(filtered s), r_h=s_h−b_t, r_a=s_a−b_t. Do not smooth.
Keep q,mu,gamma and the full C0 filter fixed; compute:

    full:              M=mu+gamma/2+2b_t+r_h+r_a
    no_relative_style: M=mu+gamma/2+2b_t
    frozen_common:     M=mu+gamma/2+2b0+r_h+r_a
    neither_motion_nor_relative: M=mu+gamma/2+2b0
    D=gamma+2(q_h−q_a), lambda_h/a=exp(M ± D/2).

“Frozen common” removes league-common *motion*, not the initial scoring level.
These four definitions are invariant to a constant gauge shift s→s+c, mu→mu−2c;
assert this numerically. A dynamic change of gauge would require mu_t and is not
allowed. Report raw-posterior-gauge b0 and roster size for reproducibility.

Score actual goals via independent Poisson at these plug-in rates, retaining total
and conditional-allocation channels. D is unchanged: allocation scores must agree
within floating-point tolerance. Score fixture-specific common/relative M component
variance and their covariance across test fixtures as a descriptive schedule-weighted
decomposition, not independent variance components. R11 supplies the score intervals.

**Outputs:** `axis_projection.csv`: league,season,axis,selection,statistic,n_weeks,
n_teams,value,null_q05,null_median,null_q95,null_rank,status.
`projection_identity.csv`: league,axis,identity,max_abs_error,estimator,status.
`style_partition_fixtures.csv`: league,season,week,fixture_id,variant,D,M,b0,b_t,
relative_sum,logp_joint,logp_total,logp_allocation,method.
`style_partition_components.csv`: league,quantity,value,n,cohort_source,status.

**Summary:** common/relative energy versus null; any estimator tautology; pooled
and sentinel full−no_relative/full−frozen_common point scores; gauge/leakage checks.
**Decision:** If useful style resides mainly in common motion while relative style
is dispensable/harmful, prioritize an identified dynamic league intensity plus
strongly shrunk relative style. If relative style retains benefit, do not replace
it with league intensity alone. If the apparent level/step mismatch is largely
projection/window geometry, weaken the OU inference. These are ablations, not fits.

## R10 — Does the style time-scale signal survive comparable horizons?

**Question:** Are R04's EPL/Scottish League Two reversion screens genuine shape
signals, or primarily changing-pair selection and overall energy differences?

**Inputs:** SENTINELS C0 10a median RTS and existing 99 null panels; R6 median RTS
for observed-only sensitivity. No new simulation or FFBS required.

**Exact computations:** same fixed season cohort, suffix and centering as R04.
Reproduce the old h=1,2,4,8,13 counts and ratios first. Then add:
1. **Endpoint eligibility:** team has a fixture at t and t+h within the same season;
   no requirement to play in intervening slots. Compute G_k(h) and normalized shape
   G_k(h)/(h G_k(1)), with exact scheduled Q time if not h.
2. **Matched starting rows:** for each h in {2,4,8,13}, retain team-start rows with
   fixtures at t,t+1,t+h. Calculate both G_k(h) and G_k(1) on those **identical rows**;
   output the ratio of pooled mean squares, not the mean of individual ratios.
3. **Common horizon cohort:** retain rows with fixtures at t,t+1,t+2,t+4,t+8,t+13,
   but not necessarily in between. On these rows compare all horizons. If <30 pairs
   or <6 teams, label LOW_SUPPORT and do not use it to rank models.
Report pair/team/season counts, the fraction of eligible endpoint pairs retained
by each method, and at each h the q/s ratio sqrt(G_s(h)/G_q(h)) on identical rows.
Separate common and relative style using R09's projections where defined; original
q/s screens remain the primary reproduction. Treat h=1 normalized identity as a
check, not an extreme-rank finding.

Run the same selections/statistics on all 99 saved nulls, with selection determined
only by the fixed actual observation design. Report raw energy and normalized shape
separately. Add within-season OLS Delta z on previous z (intercept included) using
endpoint-eligible h=1 rows; no half-life calculation or alternative-model fitting.
Season-wise values make pooled cancellations visible. R6 sensitivity has no null.

**Outputs:** `horizon_screen.csv`: league,rung,season_or_pool,axis,projection,
selection,h,n_pairs,n_teams,n_seasons,retained_fraction,Gh,G1_same_rows,
shape_ratio,r_s_over_q_h,null_q05,null_median,null_q95,null_rank,quantity,status.
`horizon_reproduction.csv`: league,axis,h,quantity,old,new,abs_error,status.

**Summary:** EPL/Scottish League Two style flattening after matching; q comparison;
Scottish Premiership/Finland contrast; raw energy versus shape; support and rung
sensitivity. ≤10 lines, not one headline per exploratory tail rank.
**Decision:** Reversion remains a nominated future fit only if negative drift and
long-lag flattening persist beyond matched nulls on ≥2 supported nontrivial horizons
or in multiple supported seasons. A uniform reduction in energy without shape
change favors scale/projection/noise misallocation over a specific OU claim.
No pattern here alone demonstrates goal-predictive improvement or estimates a clock.

## R11 — Are the score conclusions robust to block resampling boundaries?

**Question:** Are the practical-equivalence and style conclusions stable, and what
explains allocation point means outside their reported percentile intervals?

**Reason, not an accusation of a bug:** R07 pooled integrated full−no-style
allocation mean is about +0.000118, but its reported 8-week 90% interval ends near
+0.00008. Percentile intervals need not contain a point estimate; ordinary moving
blocks can underweight boundary weeks, especially in short/nonstationary seasons.
We need the bootstrap estimand/bias checked, not the interval forced to contain it.

**Inputs:** A1 `forecast_joint.csv`, `joint_comparison.csv`,
`goal_ablation_fixtures.csv`, `goal_ablation_summary.csv`; R09 partition score rows.
Use ALL, existing exactly paired fixture intersections. No new latent draws.

**Calculations:** Reproduce the published point means at ≤1e−10 and use the existing
A1 resampling method, seeds and counts to explain its result (read the existing
computer implementation, not guess its sampling design). Report mean/median of
bootstrap score means, bootstrap SE and bias relative to the point estimate.
Audit selection multiplicities by original week: expected inclusion can be computed
exactly when convenient, otherwise estimate with the same reproducible draws.
Show first/last block-length weeks' contribution to each selected headline score.

Add **circular moving-block score bootstrap**: within each league-season with T
observed weeks, choose uniform starts on 1:T, append L consecutive week indices
modulo T until ≥T are collected, truncate to exactly T; preserve all fixtures in
each selected week. Use 1,999 replicates, L=4,8,12, fixed seed. Preserve each
league-season's T; pool fixture-weighted via numerator/denominator, and equal-league
via the mean of replicate league scores. Reuse index draws across models/channels.
This wraps endpoints for *score resampling only*, never invents a lag pair, and
assumes approximate within-season stationarity; present it as sensitivity, not
an unquestionable replacement. Mark T<2L as LOW_SUPPORT.

Compute percentile [q05,q95] and basic interval
`[2*point−q95, 2*point−q05]`. Report bootstrap mean and bias, without centering away
a real discrepancy silently. Comparisons: R03 C0−R6 joint; R07 full−no_style and
full−no_quality integrated mixture_128x4, full−frozen_style plug-in; R09 full minus
each partition ablation (plug-in). For goal scores retain all three channels and
assert their paired means add exactly. Do not overinterpret the tiny integrated
allocation delta relative to its already reported Monte Carlo sensitivity.

**Outputs:** `score_robustness.csv`: league,weighting,source,method,comparison,
channel,n,n_weeks,bootstrap,L,B,point,boot_mean,boot_median,boot_se,bias,
percentile_q05,percentile_q95,basic_lo,basic_hi,status.
`week_weight_audit.csv`: league,season,week,bootstrap,L,B,n_fixtures,
mean_multiplicity,region,comparison,score_sum,status.
`score_reproduction.csv`: source,comparison,league,quantity,expected,actual,
abs_error,status.

**Summary:** equivalence tolerance under sensitivity; total-goal style benefit and
Scotland/Finland signs; frozen and partition ablations; explanation of the allocation
interval mismatch. Report conclusions that depend on boundary handling explicitly.
**Decision:** Retain a strong recommendation only if its score conclusion survives
reasonable L/method sensitivity; otherwise downgrade precision, not the original
point estimate. Do not declare equivalence league by league from pooled intervals.

## R12 — Is the Scottish quality gap a stable tier, or a selected extreme?

**Question:** Does the strongest mixture nomination survive posterior/rung uncertainty,
majority-class persistence and incomplete-season effects?

**Inputs:** sco_premiership only, R02 `team_season_path_means.csv` C0/R6 10a, 128
paths per fit, median RTS paths, `A1/tier_gaps.csv`, `season_windows.csv`, existing
99 C0 conditional nulls. EPL comparison can use existing summary rows only; no
fresh all-league clustering screen. No mixture optimization or mover ID joining.

**Computations:** On exactly the old full-season and suffix windows, sort each
posterior draw's qbar and choose its largest adjacent gap (same rule as R06).
Report gap/IQR quantiles, resulting lower/upper group sizes and membership
probabilities for every within-league team slug. Labels mean low/high q, never
choose the orientation to improve accuracy. Report posterior probability that
Celtic and Rangers constitute the complete upper group; this named contrast is
**post-selected from R06**, not a prespecified hypothesis. For successive seasons,
compare only shared teams and report both original same-side fraction and high-group
Jaccard `|upper_t intersect upper_t+1| / |upper_t union upper_t+1|`, plus adjusted
Rand index (ARI). For the 2×2 cross-tab n_ab, let S=sum_ab choose(n_ab,2),
A=sum_a choose(n_a.,2), B=sum_b choose(n_.b,2), Z=choose(n_shared,2);
ARI=(S−A*B/Z)/((A+B)/2−A*B/Z). Undefined denominators are NA. If a partition
is trivial on shared teams, mark persistence NOT_IDENTIFIABLE rather than award
perfect persistence. Small/high-group membership, not majority accuracy, is key.

Recompute point-mean gap and persistence statistics under the old 99 nulls using
exactly the same largest-gap search. Count seasons with gap/IQR≥2 and next-season
high-group Jaccard≥0.8, requiring both shared-team partitions nontrivial. Compare
this multi-season count with its null, rather than treating dependent season ranks
as independent evidence. Report the denominator of eligible season transitions,
and an all-windows versus existing-complete-windows sensitivity, with the same
window masks for data and nulls. Report season date ranges, fixture/week counts and
existing partial-season labels; distinguish a real gap change in 25/26 from limited
coverage. Do not promote the Gaussian-RW conditional null to a test against every
unimodal heavy-tailed population: that competitor remains unfitted.

**Outputs:** `scottish_tier_membership.csv`: rung,season,window,team,
prob_upper_group,prob_named_pair_only,n_draws,status.
`scottish_tier_robustness.csv`: rung,season_or_pair,window,quantity,n_shared,
q05,median,q95,null_q05,null_median,null_q95,null_rank,status.
`tier_window_audit.csv`: season,first_date,last_date,n_fixtures,n_weeks,
partial_label,cohort_n,status.

**Summary:** posterior group size/membership; C0/R6 agreement; chance/size-aware
persistence; 25/26 coverage; joint season evidence and its limits (≤10 lines).
**Decision:** Stable two-team membership and persistent extreme gaps nominate a
local tier/heavy-tail level prior, not a universal mixture or different step axes.
Unstable membership or coverage dependence downgrades the nomination. Even strong
results cannot show that a mixture improves prospective goal scores.

## Delivery, checks and grading

Return `OUT/SUMMARY.md`, the listed CSVs, `manifest.csv` with input/output hashes,
and `VERIFICATION.md` with actual commands and assertion output. Keep large caches
on the beast and point to them; summaries must stand alone for the thinker.

Executable checks in your diagnostic runner must cover:
- R09 energy and gauge identities (relative/absolute tolerance recorded, ≤1e−10
  on ordinary finite inputs); unchanged D/allocation under partition ablations;
  forecast-before-whole-week-update assertion; no test outcomes in b0.
- R10 old count/ratio reproduction, deterministic pair masks shared by data/null,
  identical rows in each normalized numerator/denominator; 99 nulls per sentinel.
- R11 point-mean reproduction, unique paired fixture keys, 1,999 resamples per L,
  exactly T resampled weeks per stratum, paired channel additivity and reported
  inclusion-weight audit. No unexpected fixture loss or mixing methods/variants.
- R12 matching season windows/gauges and 128 paths (or an explicitly documented
  source exception); largest-gap rule unchanged under null; no paths-as-extra-teams.
- `git diff --check` and `./scripts/todo.sh check`; report pre-existing failures
  without out-of-scope edits. Preserve source fits, A1 files and canonical QS hashes.

**Budget:** reuse caches; only the fixed-theta 10b filtering needed by R09 is new.
No new simulation panels or integration draws. Missing cache fields or unexpectedly
expensive reconstruction: return a scoped blocker rather than a replacement fit.
**How you are graded:** pass = exact reproduction/identities, leakage-free ablations,
matched selection/nulls, transparent resampling sensitivity and explicit limits.
Partial = honestly unavailable subparts. Failure = post-selected patterns described
as prospective discoveries, changed null/score definitions without labelling them,
or a fitted-model superiority claim from these diagnostics. Stop after the return.
