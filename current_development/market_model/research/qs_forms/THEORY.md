# Which quality/style formulation? — candidate theory

Status: batch 01 specification, 2026-10-08. No new computations yet. Request IDs
refer to `requests/batch_01.md`. C0 is the comparison, not assumed ground truth.

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

## 6. Pre-result ordering and decision boundaries

- **Retain now:** q/s as exact interpretable coordinates; C0 as parsimonious baseline.
- **Highest-value screens:** R6 tilt/scale freedom (R01/R03), posterior level geometry
  (R02), and serial/axis calibration errors (R03–R05).
- **Plausible next model, not yet a win:** hierarchical league log r, with tier as a
  modest covariate and market sharpness only if independently measured (R08).
- **Conditional extensions:** OU/trend, heteroskedasticity or nonlinear style only
  after their distinct residual/path signatures survive matched null checks.
- **Not an improvement by itself:** PCA, identified factor rewrite, polar coordinates
  or probability transforms. Mixtures, manager regimes and cross-division bridges
  require stronger evidence/inputs than the card supplies.
- **Spec A, provisional:** keep 45° observation coordinates, separate level and step
  scales, allow league r rather than fixing a universal 0.3. Do not yet add all the
  dynamics or eliminate style. Goal-model prediction—not market ellipse fit—is the
  eventual decision criterion. Batch 01 will update this ordering, not certify a
  new model without fitting and forward validation.
