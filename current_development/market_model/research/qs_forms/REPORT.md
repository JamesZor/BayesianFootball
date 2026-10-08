# Quality/style: final formulation recommendation

Completed after two computation batches. Evidence relays: `a95ca9ca` and
`4a6d0cd2`. Mathematics and candidate definitions: [THEORY.md](THEORY.md).

## Summary — answers to James

1. **Keep 45° quality/style:** it is the best-supported parsimonious expression here, not a proven universal optimum.
2. A learned rotation adds no material pooled honest market-prediction gain; C0 and R6 meet the specified equivalence tolerance.
3. Quality governs log-rate supremacy; style governs geometric intensity—not expected total goals independently of quality.
4. **League-hierarchical r is the leading extension to test**, with separate level and step geometry; its predictive gain is not yet measured.
5. Retain style: removing it loses about **0.0067 nats/fixture** in pooled goal score, almost entirely through total goals.
6. Do not shrink every league identically: style hurts in Scottish Premiership and helps strongly in Finland in these ablations.
7. Hidden structure includes a persistent—but changing—Scottish quality tier and some long-horizon style discrepancies; no universal OU law is established.
8. C0 projects out global common style: a league scoring state needs a new dynamic **mu_t** model, not reinterpretation of cached states.
9. Start Spec A with centered q/s, league-pooled r and random walks; test targeted OU/noise and quality-prior changes separately.
10. Market sharpness versus real football variation remains unidentified without better market observations and prospective goal-model comparisons.

## 1. What the comparison can establish

The data are de-vigged closing books inverted to Poisson rates, with real goals
used for a separate outcome check. Protocol 10a is retrospective; 10b fits theta
on earlier seasons and forecasts each test week before updating it. The honest
comparison covers **3,053 fixtures**. This is not evidence about trading profits.

Unless stated otherwise, score intervals below are **90% circular 8-week block
bootstrap percentile intervals**, preserving whole weeks within league-season.
R11 also checked ordinary/circular resampling, basic intervals, 4/8/12-week blocks
and fixture/equal-league weighting. Circular resampling assumes approximate
within-season stationarity; it is a sensitivity analysis, not a cure for all
selection or temporal dependence.

Posterior intervals express uncertainty under the fitted model. The 99-panel
conditional-null ranks express discrepancy from a plug-in Gaussian-RW simulation;
**they are not p-values**. Many diagnostic patterns were selected after inspection.
No OU, hierarchical-league, mixture, stochastic-volatility or alternative goal model
was newly fitted. “Promising” below never means it already beats C0.

## 2. Ranked formulations and decisive remaining tests

### 1 — Fixed 45° q/s with two dynamic axes: retain as the baseline

**Evidence (R01, R03, R07, R11).** No R6 10a league passes the prespecified
P(|tilt|>10°)>0.9 screen. Tilt is 1.44° [−3.16,5.75] in Scottish League Two,
6.68° [−3.50,15.17] in EPL and 9.49° [−9.53,23.01] in Finland: precision varies.
The proper joint market score, C0−R6, is **+0.000314
[−0.000590,+0.001185] nats/fixture**. All examined pooled intervals lie inside
±0.005, including equal-league weighting. This is pooled practical equivalence,
not equivalence in every league or exact equality of covariance matrices.

**Versus C0:** learned-angle/unequal-scale R6 is practically equivalent on this
pooled market criterion, with no demonstrated gain sufficient to displace C0.
A learned angle is simply another parameterization of R6's 2×2 covariance;
unequal attack/defence scales and q/s covariance describe the same extra freedom.

**Decisive goal-model test:** walk-forward goal scores for diagonal q/s versus a
regularized free q/s covariance, with identical data, level priors and likelihood.
Market equivalence is not automatically goal-model equivalence.

### 2 — League-hierarchical log r: highest-priority new hierarchy

Use `log r_l ~ Normal(b0 + b_t tier_l, tau_r²)` as a candidate, with a separate
quality-volatility scale. Add independently measured market covariates only when
available; do not constrain r<1 by construction.

**Evidence (R08).** Between-league SD(log r) is **0.293 [0.248,0.340]** in 10a
and **0.306 [0.254,0.361]** in 10b. England's log-r/tier slopes are −0.074 and
−0.109; Scotland's −0.198 and −0.224, with their reported posterior intervals
below zero in both protocols. These are uncertainty statements for these fitted
leagues, not a representative sample of all football leagues. Ireland is inconclusive.

**Versus C0:** not yet fitted; neither better nor equivalent nor worse prediction
has been established. The evidence supports heterogeneity and motivates partial
pooling, not a causal tier/sharpness mechanism. Avoid a universal Scottish `r≈0.3`.

**Decisive test:** hierarchical versus unpooled league scales in future-season and
leave-league-out goal prediction, using training-only hyperparameters. Compare
exchangeable pooling first, then the tier covariate; eleven leagues cannot identify
many country, tier and liquidity effects simultaneously.

### 3 — Separate level geometry from step geometry; retain style motion

**Evidence (R02).** C0 season-average `r_level/r_step` medians span **0.37–0.50
in EPL, 0.31–0.47 in Scottish Premiership, 0.22–0.41 in Finland**, versus
0.54–1.74 in Scottish League Two. The earlier Scottish near-equality is not a
cross-league law. Season averaging, finite team populations, centering and initial
conditions prevent interpreting these ratios as stationary OU variance ratios.

**Evidence against deleting style (R07/R11):**

| Goal-score comparison | Joint difference, nats/fixture | What it shows |
|---|---:|---|
| Full − no style, posterior-mixture rates | **+0.006708 [0.003469,0.009792]** | Pooled style signal is useful |
| Full − no quality, posterior-mixture rates | **+0.061274 [0.052256,0.069422]** | Quality is the dominant signal |
| Full − frozen style, plug-in rates only | **+0.015745 [0.010663,0.020888]** | Freezing the fitted style path is costly in this deletion check |

Almost all style benefit is total-goal score: **+0.006589
[0.003336,0.009679]**. Quality benefit is predominantly conditional goal allocation:
+0.060490. Style total-score benefit is **−0.018361 [−0.033192,−0.005331]**
in Scottish Premiership and **+0.048857 [0.018758,0.084262]** in Finland; those
signs survive all three circular block lengths. Scottish League Two remains
inconclusive: +0.012963 [−0.004837,0.032892].

**Versus C0:** zero/frozen-style deletions are worse pooled, but they are **not
refitted one-dimensional/static-style models**. Separate level/step priors are a
supported design safeguard, not a measured improvement. A dynamic rank-one model
with static team style is distinct from setting all style to zero. An unequal-loading
rank-one model is also distinct and has not been tested.

**Decisive test:** refit quality-only, static-style and dynamic-style goal models
with equal training windows and uncertainty integration; permit independent initial
q/s scales. Do not compare a refitted candidate to an artificially frozen baseline.
Integrated frozen-style was unavailable and remains explicitly untested.

### 4 — A local quality heavy-tail/tier prior: credible in Scottish Premiership

**Evidence (R06/R12).** In both C0 and R6, **all 128 retained path draws per fit**
place exactly Celtic and Rangers above the largest quality gap in each season
21/22–24/25. The first three shared-team transitions have posterior median Jaccard
and adjusted Rand index 1. The joint count of large-gap/persistent transitions is
2 versus conditional-null 90% range [0,1], rank 1.00. This is stronger than
uncorrected majority-group persistence, but the named pair was post-selected.

It is not permanent: in 25/26 none of those draws has exactly that upper pair;
upper-group size has median 1 and 90% range 1–3. Transition Jaccard falls to 0.50,
ARI to 0.516. **Every season uses the same pre-split rounds 1–33 window**;
uniquely missing post-split matches cannot explain the latest change. Accepted-book
selection still applies (25/26: 193 accepted fixtures of 198 raw).

**Versus C0:** descriptive level structure is supported, predictive superiority is
unresolved. The Gaussian-RW conditional null is not a comparison against every
unimodal heavy-tailed population, and no universal mixture follows from one league.

**Decisive test:** compare a Gaussian q-level prior, a unimodal heavy-tailed prior,
and a strongly pooled ordered mixture on q only. Forward-test extreme-team
calibration and goal scores; do not hard-code Celtic/Rangers into permanent tiers.

### 5 — Axis-specific time scales/OU or observation noise: targeted tests, not defaults

**Evidence (R04/R10).** After fixing horizon eligibility, style shape
`G(13)/(13 G(1))` is EPL **3.86** (null rank 0.01, 978 pairs), Finland **3.25**
(0.04, 268), Scottish Premiership **5.78** (0.23, 523), Scottish League Two
**7.40** (0.26, 268). These are **RTS-smoothed** variograms: their null shape is
not the raw-RW value 1. Scottish League Two's original long-lag flattening does
not survive the pair-selection change. EPL retains a negative style drift slope
−0.00250 (rank 0.01) and a long-horizon discrepancy; this nominates a test, not a
half-life. Finland supplies a weaker, isolated long-horizon signal.

Scale and clock are different: Scottish Premiership's raw h=13 style energy is
below its null (rank 0.01), while its normalized shape is not unusual (0.23).
R03 also shows EPL intensity overcoverage (90% interval coverage **0.966**,
standardized squared error **0.622**), compared with supremacy 0.903/0.958.
That can reflect process/noise allocation rather than a new state law.

**Versus C0:** no OU, distinct-clock or axis-noise candidate has demonstrated better
prediction. There is no robust general momentum finding from autocorrelated RTS
increments; the fitted smoother's null itself creates strong autocorrelation.

**Decisive test:** on the nominated leagues, compare RW with style-only OU, then
separate q/s OU clocks, against a simpler alternative with distinct observation-axis
noise. Use full forward filtering, forecast calibration and goal scores; retain
random walks elsewhere unless the extension generalizes. Do not infer kappa from
level/step ratios or the descriptive OLS slope.

### 6 — Time-local quality volatility/jumps: investigate events before team types

**Evidence (R05).** EPL q top-5% steps contain **31.8%** of RTS energy (null rank
1.00); weekly mean-energy lag correlation is **0.535** (0.98). Cross-team energy
CV is **0.479** (0.99), but first-/second-half team-energy rank correlation is
**−0.019** (0.37). Sentinel team rankings generally fail to persist.

**Versus C0:** localized heterogeneity is a candidate; permanent team-specific
volatility, stochastic volatility and jump/regime models are all predictively
unresolved. Bad/inverted books, calendar changes and actual football shocks remain
competing explanations. No manager-change effect was identified.

**Decisive test:** audit concentrated events against executable book quality and
known ex-ante calendar/event dates; compare a robust observation law or known
calendar Q_t before adding persistent SV or latent regimes. Require prospective
scores, not another selected tail statistic.

### 7 — Other coordinate expressions: useful interpretation, no free predictive gain

- **Probability/total coordinates:** under Poisson goals, use
  `D=log(lambda_h/lambda_a)`, `M=(log lambda_h+log lambda_a)/2`,
  `Lambda=2 exp(M) cosh(D/2)`, `p=logistic(D)`. Then total goals are Poisson(Lambda)
  and home allocation given total is Binomial(total,p). This makes the R07/R11
  channel results interpretable. It also shows why q influences expected totals.
  **Equivalent to C0** if the law is transformed exactly; an additive team model
  on log Lambda is a *different, untested* interaction model. A lone win probability
  discards draw/total information. Test changed laws on forward goal scores.
- **Factor loadings:** unrestricted one-factor alpha/beta loadings plus two residual
  variances have four parameters for three covariance entries. Equal uniqueness
  can identify a covariance rewrite; with a matching covariance/prior it is
  **equivalent, not a discovered hierarchy**. Distinct factor clocks would require
  an identified new model and prospective testing. R01/R03 give no reason to
  introduce loadings merely to relabel the ellipse.
- **Polar magnitude/angle:** **equivalent** only with correctly transformed dynamics
  and Jacobian. Independent polar walks would change the model and are untested;
  the angle is unstable near zero and depends on centering. No computed result
  establishes an advantage over q/s. Test only after a reproducible state-dependent
  variance pattern supplies a concrete motivation.
- **Nonlinear s(q):** R06's selected Irish 2022 quadratic coefficient is 0.0367,
  but quadratic−linear leave-team-out error is **+0.00017 [−0.00069,+0.00093]**
  and adjacent-season curvature changes sign. **Not shown better than C0**;
  require stable leave-season/team-out gain before fitting a nonlinear hierarchy.

### 8 — Common league states, venue strengths and division bridges: defer

**R09 is an identification result, not a fitted-model victory.** C0's observation
map applies full-roster zero-sum projection each week. Across all 11 10b fits the
raw common style direction has maximum loading **1.10e−15** and its filtered mean
remains at its prior within **5.70e−15**. Honest common/relative style score
ablations were therefore canceled as **NOT_IDENTIFIABLE**. A smaller active-roster
mean can move relative to excluded teams; it is not an observed global intensity.
The off-diagonal centered covariance identities hold to **5.42e−20** and do not
independently establish common shocks.

**Versus C0:** the canceled partition has no score and no comparison. A genuine
league-total state needs dynamic `mu_t` alongside centered style; equal q shocks
cancel from supremacy. Earlier supplied ICC≈0.10 retrospective/≈0 honest lowers
priority but was not recomputed here. A new dynamic-mu fit must earn its place via
honest league-total and goal calibration.

Venue-specific effects have no validated predictive gain in this investigation;
they require strongly pooled team×venue candidates tested against schedule
imbalance, not an unidentifiable split of global home advantage. Cross-division
pooling is also **unresolved**, because cached team identities could not be verified
for movers. Establish IDs and bridge assumptions, then compare carried versus
reset states out of time. Zero verified movers does not mean no clubs moved.

## 3. Market geometry is not yet football truth

The positive goal check prevents dismissing style as entirely meaningless market
noise. It does **not** identify why r differs across leagues. Totals staleness,
heteroskedastic inversion error, selection by book completeness, real tactical
variation and non-Poisson goal dependence remain confounded.

R08 found archived trade-price timestamps but not an executable quote-age/update
experiment. Optimizer-start spread is not bid–ask spread; sigma_obs is not liquidity.
A lower r in thin markets has no uniquely identified causal interpretation.
The decisive data are matched market-family quote histories, depth/spread/freshness
and accepted/rejected books, combined with a joint goal/market observation model
and forward prediction. No causal sharpness claim is warranted from eleven leagues.

## 4. Concrete recommendation for goal-model Spec A

1. **Keep centered 45° q/s**, documenting beta as concedes-more and applying the
   actual zero-sum observation map. Preserve coherent score-grid pricing.
2. **Retain both dynamic axes**, with quality dominant but style not fixed to zero.
   Start with random walks; the smaller style axis can still carry useful total signal.
3. **Use/compare hierarchical league log r**, learning hyperparameters on training
   data only. Treat tier as a modest candidate covariate, not causal sharpness.
4. **Do not force initial/level covariance to equal process geometry.** Keep the
   relevant priors separate and regularized, especially for entrants and new seasons.
5. **Do not add OU, SV, mixtures, venue effects and dynamic mu together.** First
   test the local quality-prior and style-clock/noise nominations separately. A
   Scottish heavy-tail prior is a simpler competitor to permanent discrete tiers.
6. **Choose by forward goal performance**, with total/allocation scores, uncertainty
   calibration, league-specific results and dependence-aware intervals. A market-fit
   ellipse or in-sample null discrepancy cannot substitute for this comparison.

No batch 03 is essential: it would not turn these unfitted alternatives into tested
models or supply missing quote histories/IDs. The remaining uncertainty belongs in
future fits and data work, not in a stronger claim from the same saved draws.

## 5. Evidence and partial-work audit

- [Batch 01 summary](answers/batch_01/SUMMARY.md): R01–R08; exact artifacts and
  beast-only draws are indexed in its [manifest](answers/batch_01/manifest.csv).
- [Batch 02 summary](answers/batch_02/SUMMARY.md): R09–R12; artifact locations and
  hashes are in its [manifest](answers/batch_02/manifest.csv).
- R07 integrated frozen-style is unavailable; only its plug-in deletion is reported.
- R08 quote repricing is unidentified and mover joins unavailable; neither was imputed.
- R09 canceled partition scores/checks are **NOT_RUN**, never claimed as passes.
- R11 is complete for existing R03/R07 scores, partial only for canceled R09 scores.
  It reproduced the old results and explained the allocation interval mismatch as
  boundary underweighting; it did not silently replace or alter their point estimates.
- R12's 21/22 suffix posterior paths are unavailable; full-window posterior and
  matched suffix point/null comparisons remain distinct. Complete-window sensitivity
  equals the main analysis because every season shares rounds 1–33, not because
  post-split data were recovered.
- No new candidate-model fits, no new predictive superiority claims, and no trading
  recommendation are implied by this report.
