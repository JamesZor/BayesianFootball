# Which quality/style formulation? — candidate theory

Status: complete after two batches. Sections 1–5 retain the candidate hypotheses
(with the R09 observation-map correction); §6 is the batch 01 interim interpretation;
§7 records the final update. Conclusions: `REPORT.md`. Request IDs refer to
`requests/batch_01.md` and `requests/batch_02.md`. C0 is a comparison, not truth.

## 1. What is known, and what “best” means

The supplied cross-league results have innovation r = sigma_s/sigma_q between
0.27 (Scottish League Two) and 0.68 (Finland). R6 q/s step-correlation 90% intervals
include zero in all 11 leagues. C0−R6 honest differences are below 0.005 nats per
fixture in absolute value, but the published score sums two marginal densities:
that is not proof of joint predictive equivalence. Full−no-style goal log score
is +0.0067 ± 0.0023 fixture SE overall, −0.0198 ± 0.0083 in Scottish Premiership,
and +0.0502 ± 0.0177 in Finland. Dependence-aware uncertainty is still needed.
Sources: context card and `results/QS/QUALITY_STYLE_REPORT.md`.

Three separate criteria: (i) interpretable coordinates; (ii) parsimonious and
identified dynamics; (iii) better prospective goal distributions. An invertible
coordinate change alone cannot improve a probability model. A changed prior,
transition law, noise law or dimensional restriction can. Screens on existing
fits can nominate such changes, not establish their prospective superiority.

## 2. Exact geometry, including a total-goals caveat

Write beta as **concedes-more**, x=(alpha,beta)', z=(q,s)', and

    z = A x,  A = (1/2)[1 -1; 1 1];    alpha=q+s, beta=s-q.
    D = log(lambda_h/lambda_a) = gamma + 2(q_h-q_a),
    M = (log lambda_h + log lambda_a)/2 = mu+gamma/2+s_h+s_a.

**R09 engine clarification:** in these observation equations q and s are the
projected team states. For full structural roster size N, P=I−11'/N and
q=P q_raw, s=P s_raw at each week. Therefore H_s 1=0: a uniform raw style
shift does not change C0's rates. A genuine league scoring state requires mu_t,
not an ablation of the prior-only raw common mode. Subset-cohort means can move
relative to the other roster teams; they are not that global common mode.

A is an orthogonal rotation times 1/sqrt(2), not a unit-length rotation.
q is log-rate supremacy; s is log geometric-mean scoring intensity. Crucially,

    Lambda = lambda_h+lambda_a = 2 exp(M) cosh(D/2),
    p = lambda_h/Lambda = logistic(D).

Thus q also changes expected total goals at fixed M. Saying “s drives the total”
is shorthand for log-rate product, not an exact claim about Lambda or O/U prices.
Under independent Poisson goals, N=G_h+G_a ~ Pois(Lambda) and
G_h | N ~ Binomial(N,p). This gives a useful alternative observation factorization.

For R6 steps let u=sigma_alpha², v=sigma_beta², w=rho_ab sigma_alpha sigma_beta.
Then Q_z=A Q_x A', with

    Vq=(u+v-2w)/4, Vs=(u+v+2w)/4, Cqs=(u-v)/4.
    r=sqrt(Vs/Vq), c=Cqs/sqrt(Vq Vs).

C0 imposes Cqs=0, equivalently u=v, but permits negative or positive rho_ab via r.
R2 imposes w=0, hence Vq=Vs and r=1 even when c is nonzero. R2 need not have
independent q/s steps. The shortcut sqrt((1+rho_ab)/(1-rho_ab)) equals r only
when attack and defence variances are equal; measure level r directly (R01–R02).

### Coordinates and dimension

| Candidate | Compact formulation and identifying restriction | Difference from C0; cheap check |
|---|---|---|
| Learned principal angle | Q_x=R_phi diag(l_plus,l_minus) R_phi', first column (cos phi,−sin phi)'; order l_plus≥l_minus and orient alpha positive. In q/s, delta=0.5 atan2(2Cqs,Vq−Vs), phi=45°−delta (mod 180°). | Exactly R6 covariance, not an additional model. Angle unstable when eigenvalues coincide. R01: posterior tilt, eigen-gap and rotation's change in residual variance; R03: joint forecast score. |
| Unequal attack/defence scales | k=sigma_alpha/sigma_beta with rho_ab free. | k≠1 is the same extra covariance freedom as c≠0, not separate evidence. Standardizing each axis can hide unequal effects on log rates. R01 reports k and c alongside phi. |
| One factor with loadings | alpha=a f+e_alpha, beta=−b f+e_beta; Var(Delta f)=1, a,b>0, independent residuals with variances tau_alpha²,tau_beta². Covariance=[a²+tau_alpha²,−ab;−ab,b²+tau_beta²]. | Four parameters for three covariance entries: not identified. Equal uniqueness tau_alpha=tau_beta identifies a rank-one-plus-isotropic representation when the leading direction has opposite signs, but this is another representation of the same ellipse. Distinct temporal kernels for f and residuals would add testable structure. R01 screens loading direction/rank; R04 screens time scales. |
| Polar states | q=R cos psi, s=R sin psi, R≥0. | Exact reparameterization if priors/Jacobians are transformed (Cartesian density gains factor R in polar measure). Gaussian walks become state-dependent polar noise; psi is singular near R=0 and depends on gauge. Independent walks on R,psi are a different, less natural model. R06 radial dependence can motivate, not identify, it. |
| Probability/total coordinates | (D,M) ↔ (p,Lambda); alternatively log(P_home/P_away) with a separate draw/total coordinate. | Same exact Poisson predictions under a bijection; 1X2 win log-odds is not D and a lone win probability loses draw/total information. Additive team effects in log Lambda rather than M impose a different interaction through cosh(D/2). R03/R07 split supremacy, intensity and goal-total errors. |
| One-dimensional dynamic limit | sigma_s=0 but s_i,0 may remain heterogeneous and static. | Removes style *change*, not style levels. Compare to fully quality-only s_i,t≡0 and to unequal-loading rank-one alpha=a f, beta=−b f. Neither a positive variance quantile under a continuous prior nor zeroing fitted s is a fitted boundary-model test. R01 rank fraction, R02 style levels, R07 frozen-style versus zero-style ablations. |

For C0 the style eigenvalue fraction is r²/(1+r²): about 0.068 at r=0.27
and 0.316 at r=0.68. These algebraic illustrations are not estimates from new
posterior draws. Small style variance is not the same as dispensable style.

## 3. Hierarchy: share information without manufacturing tiers

1. **League pooling.** eta_l=log r_l ~ N(b0+b_t tier_l+b_m sharpness_l,
   tau_r²), with a separate prior on log sigma_q,l. Allow r>1; do not build the
   desired answer into the support. Optional delta_l ~ N(d0,tau_delta²) near
   the quality axis, or an axial circular law if large rotations are plausible.
   Add country effects only when supported by more leagues; 11 leagues cannot
   separate many country/tier/liquidity effects. Prediction: smaller estimation
   variance and sensible transfer to sparse leagues, not necessarily a new mean
   within a well-observed league. R08 screens heterogeneity and tier contrasts;
   definitive test is leave-league/future-season-out pooling versus unpooled fits.
   Never treat posterior draws as independent league replicates or silently reuse
   their priors as likelihoods in a two-stage hierarchy.
2. **Team volatility.** log sigma_k,i = log sigma_k,l + u_k,i,
   u_k,i ~ N(−tau_k²/2,tau_k²), k in {q,s}; optionally correlated u's.
   Persistent dispersion of team innovation energy predicts this rather than
   calendar volatility. R05 checks excess dispersion and split-season stability
   against the fitted observation geometry. Gauge-centering induces cross-team
   dependence and must be applied to the whole covariance, not ignored.
3. **Quality tiers.** q_i,0 | c_i ~ N(m_c,v_c), ordered m_1<...<m_K;
   s stays unimodal. A dynamic label requires an explicit transition law.
   Non-Gaussian q levels or overshrunk extremes, not negative alpha/beta correlation,
   motivates tiers. R06 checks shape and persistent gaps. A heavy-tailed unimodal
   q prior is a simpler competitor; small leagues make apparent clusters cheap.
4. **Cross-division bridge.** For a mover l→m,
   q_i,new = a_lm + b_lm q_i,old + e_i,q,
   s_i,new = c_lm + d_lm s_i,old + e_i,s, with offseason variances.
   Division-relative centering must be converted to an identified global reference.
   Within-division fixtures alone do not identify absolute quality offsets;
   transition clubs or cups help only with explicit continuity assumptions.
   R08 inventories movers and relative-rank continuity, not a fictitious estimated
   tier offset. Future joint fit tests whether carried states outperform reset priors.

## 4. Hidden dynamics: competing, not automatically cumulative

Let z_i,t=(q_i,t,s_i,t) be in a documented identifiable gauge. C0 has
z_i,t+1=z_i,t+epsilon_i,t, Q=diag(sigma_q²,sigma_s²) on the engine time scale.
Schedules, season breaks and inactive teams must be respected.

| Mechanism | Formulation | Distinct prediction; first screen / decisive later test |
|---|---|---|
| Mean reversion | z_k,t+Delta−m_k = exp(−kappa_k Delta)(z_k,t−m_k)+epsilon; Var epsilon=D_k(1−exp(−2kappa_k Delta))/(2kappa_k). | Negative state→future-change relation and long-lag variogram plateau. kappa→0 gives RW. R04 calibrated path screen and R03 residual drift; later forward OU fits. Finite windows, centering and conditioning create spurious reversion. |
| Momentum/local trend | z_t+1=z_t+v_t+epsilon, v_t+1=rho_v v_t+xi. | Positive short-lag increment correlation and persistent signed forecast misses. R03/R04; future local-linear-trend versus RW forecast scores. |
| Different q/s clocks | Separate kappa_q,kappa_s or trend kernels, not merely different sigma's. | Distinct normalized variogram shapes. Under stationary OU, r_level²=(D_s/D_q)(kappa_q/kappa_s), whereas short-step r²≈D_s/D_q. R02/R04; no stationary level-r identity for RW or season-averaged smoothed estimates. |
| Season phase/transfer volatility | Q_t=diag(sigma_k² exp(b_k' h_t)), known calendar h_t. | Repeatable volatility near boundaries/windows versus regular weeks. R05 uses season phase/offseason gaps; transfer attribution needs country-year window dates, absent from the card. Future ex-ante calendar model. |
| Stochastic volatility | log v_k,t=a_k+rho_k(log v_k,t−a_k)+xi_k,t; epsilon_k,t~N(0,v_k,t). | Persistent squared innovations without persistent signed ones. R03/R04; compare SV to book-noise heteroskedasticity, not just to homoskedastic RW. |
| Jumps/regimes | epsilon~(1−pi)N(0,Q)+pi N(0,cQ), or persistent regime-specific level/drift. | Concentrated large moves, potentially persistent level shifts. R05 tails and event concentrations. No manager dates means no manager-change claim; bad books and diffuse initialization are alternatives. |
| League common shocks | Delta q_i=b_i f_q+e_i,q; Delta s_i=d_i f_s+e_i,s. | Equal b_i shocks cancel in q_h−q_a and are unobservable. Equal d_i shocks alias mu_t through 2d f_s. Only nonuniform loadings or an explicitly identified league-total state are testable. R03 weekly intensity surprises/R05 centered co-movement; prior honest ICC≈0 lowers priority. |
| Style depends on quality | s_i,t=a_l+b_l q_i,t+c_l(q_i,t²−E q²)+u_i,t; or quality-dependent Var(Delta s). | Curved level locus or radial/state-dependent noise even if corr(q,s)=0. R06 tests linear/quadratic level association and R05 bins energy by |q|. Relation in levels need not imply coupled transitions. |
| Home/away-specific strengths | e.g. D=gamma+2(q_h−q_a)+h_h+h_a; M=mu+gamma/2+s_h+s_a+g_h−g_a, with constrained team deviations. | Persistent team×venue forecast errors, distinct from the unidentifiable split of a global HA parameter. R03 role contrasts; future strongly pooled venue states, with schedule imbalance controls. |
| Level versus step geometry | Q_level not forced proportional to Q_step; initialize z_i,0~N(0,P0) separately. | Different r or angles across levels, increments and time horizons. R02 direct posterior level covariance and R04 variograms. Equal ratios alone establish neither stationarity nor a shared latent hierarchy. |

**Diagnostic discipline.** RTS means are denoised; their increments are not observed
innovations. FFBS paths restore conditional uncertainty but remain conditional on
C0/R6. Match every retrospective statistic with a no-refit simulate→smooth null
using the same schedule, centering and observation noise. Report posterior bands
and null envelopes separately. A null discrepancy nominates a future model; a
null match under a restrictive fitted model is weak evidence of absence. Forecast
errors before whole-week updates supply the less model-imposed check. Multiple
screens are exploratory; require coherent cross-protocol/league evidence, not the
smallest p-value among dozens of tests.

## 5. Does r belong to football or the bookmaker?

A more general observation model is

    y_book,t = H_t z_football,t + b_market,t + e_book,t,
    e_book,t ~ N(0,R_t), possibly stale/asynchronously updated by market,
    goals_t | z_football,t ~ goal likelihood.

Sparse totals updates can attenuate apparent style motion or produce delayed jumps;
noisy thin totals can instead inflate it. Direction is not predetermined. Inversion
couples the markets, so even the supremacy coordinate need not escape this problem.
Different R_t by axis, serial book error, gate selection and freshness can all be
misallocated to process Q. Existing C0 recovery validates a limited correctly
specified synthetic experiment, not these misspecifications.

**Existing-data separation (R03/R07/R08):** axis-specific forecast calibration;
goal total and conditional-allocation scores; league-wise goal benefit of style;
correlation of log r with sigma_obs and available inversion/book-quality metadata.
Goals are an independent outcome but noisy, and the market remains the training
signal. Liquidity, country, tier and real football differences are confounded.

**Decisive future separation:** matched timestamp ladders for 1X2/totals/BTTS,
update intervals, depth/spread and overround; within-fixture freshness-matched
inversion; repeat by time-to-kickoff and book subset; preserve the accepted/rejected
book denominator. Fit latent football dynamics jointly with market-specific errors
and goals (plus plausible non-Poisson goal dependence). Test r stability and goal
scores out of time. A lower r in thin leagues alone is not evidence of market
staleness or of simpler underlying football.

## 6. Batch 01 evidence and revised ordering

Source: `answers/batch_01/SUMMARY.md` and the explicitly identified diagnostic
CSVs below, relayed at commit `a95ca9ca`. Numbers here are computed results;
interpretations and follow-up nominations are labelled separately. Posterior,
block-bootstrap and conditional-null uncertainty are not interchangeable.

### 6.1 Supported now: interpretable axes, league variation, useful style

- **R01/R03 — retain 45°.** No league's R6 10a posterior meets the prespecified
  P(|tilt|>10°)>0.9 criterion. EPL tilt is 6.68° [−3.50,15.17], Finland 9.49°
  [−9.53,23.01], Scottish League Two 1.44° [−3.16,5.75]. The first two are not
  precise proofs of alignment. Crucially, honest *joint* C0−R6 score is +0.00031
  [−0.00018,+0.00138] nats/fixture, equal-league +0.00066 [−0.00004,+0.00217]
  (3,053 fixtures, 8-week bootstrap). Both pooled intervals meet the ±0.005
  practical-equivalence criterion. This does not establish every league's equivalence.
- **R08 — league-hierarchical r is the leading extension to test.** C0 SD(log r)
  across these leagues is 0.293 [0.248,0.340] in 10a and 0.306 [0.254,0.361] in
  10b. Within England, log-r/tier slope is −0.074 [−0.110,−0.038] / −0.109
  [−0.150,−0.068]; within Scotland −0.198 [−0.267,−0.130] / −0.224
  [−0.303,−0.145]. These are fit-uncertainty intervals, not uncertainty over a
  population of leagues. Pooling itself has not been fitted or shown to win.
- **R07 — do not delete style globally.** Integrated full−no-style goal score is
  +0.00671 [0.00308,0.00923], almost entirely total-goal score (+0.00659).
  Full−no-quality is +0.06127, mostly allocation (+0.06049). Scottish Premiership
  style benefit is negative (−0.01852 [−0.02741,−0.00183]), Finland positive
  (+0.04916 [0.01292,0.06577]). Full−frozen-style is +0.01574
  [0.00766,0.01761] **plug-in only**; integrated frozen-style is unavailable.
  These are deletion comparisons, not fitted rank-one/static-style competitors.
  R11 checks interval sensitivity before treating marginal signs as settled.

### 6.2 Newly important: distinguish common scoring level from relative style

**R02 finding:** C0 season-average level/step ratios are 0.37–0.50 in EPL,
0.31–0.47 in Scottish Premiership and 0.22–0.41 in Finland, versus a much less
stable 0.54–1.74 in Scottish League Two. A universal “same geometry at levels and
steps” claim is untenable. Ratios concern season-averaged relative states, however,
not stationary variances or identified OU half-lives. The huge shortcut-r error in
a two-team/one-week Scottish Championship window is a degeneracy, not a discovery.

**Batch 01 hypothesis, corrected by R09:** the decomposition

    sbar_t = mean_{i in C}(s_i,t),   s_i,t = sbar_t + srel_i,t,
    M_ha,t = mu + gamma/2 + 2 sbar_t + srel_h,t + srel_a,t

is algebraically valid for already projected states and a fixed subset C. It does
**not** imply an observable full-roster common style state. The earlier proposal
to apply it directly to raw uncentered engine states was inapplicable: C0 projects
those states before pricing. For C equal to the full structural roster, projected
sbar_t=0. For a smaller active roster its mean moves relative to excluded teams,
not as an independent league intercept.

R09 measured maximum common-mode design loading 1.10e−15 across all 11 honest
fits, with filtered raw common mean unchanged from its prior to 5.70e−15. The
requested honest common/relative score partition is NOT_IDENTIFIABLE and was
canceled by the manager. No such goal scores or passed allocation/leakage checks
exist. The energy decomposition is valid descriptively, but does not establish a
league-intensity mechanism or explain away level/step differences. Testing a true
league scoring state requires a future dynamic mu_t model, alongside centered
team style, with each effect represented only once.

### 6.3 Hidden dynamics: nominated, not fitted or proven

- **R04 style reversion is suggestive in EPL/Scottish League Two, not universal.**
  Their RTS style drift slopes are −0.00250 / −0.00451 with conditional-null rank
  0.01. In `dynamics_screen.csv`, EPL G_s(13)/(13 G_s(1))=1.18 versus null median
  4.88 (rank 0.01), but only **48 pairs** survive strict continuous-appearance
  eligibility; Scottish League Two has 21 pairs at eight weeks and none at 13.
  Different horizons select different teams/windows. Scottish Premiership has
  positive style slope +0.00125 (rank 0.39) and only five 13-week pairs. R10
  uses endpoints and matched horizon cohorts before promoting OU/different clocks.
- **R05 quality volatility is uneven, but not stably “a volatile team type.”** EPL
  q energy CV=0.479 (rank 0.99), top-5% energy share=0.318 (rank 1.00), and weekly
  energy lag correlation=0.535 (rank 0.98). Yet its first/second-half team energy
  rank correlation is −0.019 (rank 0.37); all sentinel values are small. Prefer an
  event/calendar/noise investigation over permanent team-volatility effects.
  RTS increment autocorrelation alone is not momentum: its matched null is also high.
- **Do not interpret centered off-diagonal covariance as a common-shock finding.**
  For n centered increments v with sum v=0, their mean off-diagonal product is
  `−sum(v_i²)/(n(n−1))`. Thus the R05 extreme ranks can simply restate energy
  differences; they do not independently identify shared shocks. R09 audits the
  exact estimator and reports identifiable common/relative projections instead.
- **R06 Scottish Premiership tiers deserve a narrow follow-up.** C0 largest q
  gap/IQR is 2.25, 2.80, 2.05 in 22/23–24/25, with null ranks 0.98,1.00,0.98.
  In 23/24 all 11 shared teams retain their gap side the next season (rank 1.00).
  `tier_gaps.csv` places those gaps below Rangers; 25/26 instead has a 0.565 gap
  between Rangers and Celtic. High same-side accuracy can favor the larger group;
  partial windows and gap membership matter. R12 tests posterior/rung robustness
  and minority membership before interpreting this as a general tier prior.
- **R06 nonlinear s(q) is not yet a useful extension.** The selected largest
  curvature (Irish First Division 2022, c=0.0367 [0.0242,0.0510]) has quadratic−
  linear leave-team-out error +0.00017 [−0.00069,0.00093]. Adjacent 2021 curvature
  is negative. A selected nonzero coefficient does not establish stable curvature.

### 6.4 Scope of the next decision

Keep 45° and league-varying r as the provisional Spec A basis; give level priors
and process scales separate parameters. Test common intensity versus relative style
before interpreting small level/step ratios as OU. Tier priors and q-event volatility
remain local nominations, not universal additions. No case has emerged for polar
states, a freely unidentified factor hierarchy, stable team-volatility ranks or
mandatory nonlinear s(q). There is no new positive case for venue-specific strengths.

Microstructure remains unresolved: archived trade-price timestamps are not executable
quote ages, optimizer-start spread is not bid–ask spread, and sigma_obs is not
liquidity. Division movers are unavailable because cached IDs are not independently
verified; no fuzzy join or claim of zero movers is warranted. Leave both gaps
explicit rather than spending batch 02 on unavailable data.

The batch 02 scope was R09 common/relative style, R10 robust horizon screens,
R11 resampling robustness and R12 Scottish tier membership. Its final interpretation
follows; no candidate model was fitted or certified superior by these diagnostics.

## 7. Final update after batch 02

Source: `answers/batch_02/SUMMARY.md`, relay `4a6d0cd2`; the manager confirmed
C0's projection and the common pre-split Premiership window. No batch 03 is needed:
remaining decisive tests require new models or unavailable data, not another
rearrangement of these posterior draws.

1. **R09 corrects the observation-map hypothesis, not the style goal result.** The
   global raw common mode is prior-only (see §§2/6.2). Active-cohort common-energy
   fractions must not be described as measured league intensity. Complete centered
   off-diagonal energy/covariance identities hold to 5.42e−20; R05's pair-selected
   estimator deviates because its rows differ, not because common shocks were found.
   Dynamic mu_t is identifiable only through a different observation/dynamics model;
   existing retrospective/honest ICC≈0.10/≈0 from the brief is not recomputed evidence.
2. **R10 weakens a general OU claim.** Matched h=13 style shape ratios
   G(13)/(13 G(1)) are EPL 3.86 (rank 0.01, 978 pairs), Scottish League Two 7.40
   (0.26, 268), Scottish Premiership 5.78 (0.23, 523), Finland 3.25 (0.04, 268).
   These are smoother statistics against a matched null, not direct RW variograms.
   Scottish League Two's old long-lag flattening is not robust to selection. EPL
   keeps a long-horizon signal plus negative drift, but not a universal two-axis
   OU law; Finland's isolated long-lag discrepancy is weaker. Scottish Premiership
   has low raw style energy at h=13 (rank 0.01) without an unusual normalized shape
   (rank 0.23): scale and clock shape must remain separate. No half-life is estimated.
3. **R11 confirms the large score conclusions and explains the tiny mismatch.**
   Circular 8-week C0−R6 interval is [−0.000590,+0.001185] around +0.000314;
   all examined methods, L=4/8/12 and pooling weights meet ±0.005. Integrated
   full−no-style is +0.006708 [0.003469,0.009792]; total channel +0.006589
   [0.003336,0.009679]. Scottish Premiership total benefit stays negative and
   Finland positive across all three circular block lengths. Full−frozen-style
   remains positive, +0.015745 [0.010663,0.020888], **plug-in only**. Ordinary
   allocation bootstrap bias was −9.27e−5 due to boundary underweighting; circular
   bias is −1.20e−6. The tiny allocation result remains Monte-Carlo-sensitive and
   is not used as a new structural discovery. Canceled R09 scores remain absent.
4. **R12 upgrades a local level-prior nomination, not a permanent tier law.** In
   both C0 and R6, all 128 retained path draws put exactly Celtic/Rangers above
   the largest q gap in each season 21/22–24/25; none does in 25/26. Shared-team
   Jaccard/ARI medians are 1 through the first three transitions. They fall to
   0.50/0.516 in 24/25→25/26; the latest upper-group size has posterior median 1
   and 90% range 1–3. The multi-season extreme-gap/persistence count is 2 versus
   conditional-null 90% range [0,1] (rank 1.00). This is post-selected, not a Bayes
   factor against heavy tails. Every season uses rounds 1–33; the 25/26 change is
   not explained by uniquely missing post-split rounds. No universal mixture follows.
5. **Final Spec A recommendation:** keep projected 45° q/s; retain dynamic style;
   allow hierarchically pooled league log r and separately specified initial/level
   geometry. Use RW as the default while testing targeted style OU/noise alternatives,
   and a Scottish quality heavy-tail/tier prior, one change at a time. None of these
   unfitted extensions can yet be called predictively better than C0. Market
   microstructure, verified cross-division bridges and integrated frozen-style
   comparisons remain explicitly unresolved. Full ranking and decisive future tests
   are in `REPORT.md`.
