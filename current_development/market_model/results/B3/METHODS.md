# B3 methods and literature (2026-10-06)

## Scope and fixed decisions

The human keeps **double Poisson for Phase C**, whatever B3 finds; C2 is a later
human decision. B3 compares score grids as evidence, not as a Phase C change.
No outside managerial-change data are fetched.

## Choices made before inspecting real results

- Active step endpoints are each team's first through last **accepted-panel**
  match week within each season, inclusive, restricted to t>=2. Steps into a
  team's first observed week are included if t>=2. Between-match weeks are
  included; summer gaps and prior-only teams are excluded from the statistics.
  The state model itself still evolves through every grid week, including summer.
- Analyse centred alpha/beta innovations: remove the common-team gauge from
  each path at each week before differencing. Quality/style are (alpha-beta)/2
  and (alpha+beta)/2. The same mask and centring apply to observed and replicate.
- Use 200 deterministically stratified posterior theta draws, covering all four
  chains. At every draw: observed full-panel FFBS; replicate all fixtures/weeks
  conditional on that observed draw's initial state; regenerate all subsequent
  Gaussian innovations and noise; refilter under the unchanged initial prior;
  draw replicate FFBS. This integrates initial-state uncertainty through the
  observed posterior rather than regenerating implausible initial team levels.
- Both protocol PPCs are retrospective **full-panel** checks: 10b only limits
  the theta fit, not smoothing. They are not honest out-of-sample forecasts.
- Joint 95% events use each draw's empirical one-sided 95/5% quantiles, not an
  absolute 95% threshold: attack up/concessions down, and attack down/concessions
  up separately; report their signed difference for asymmetry. These finite
  thresholds do not estimate an asymptotic tail-dependence coefficient.
- Excess kurtosis is the raw central fourth/second-moment ratio minus 3.
  Lag-1 squared-step correlations pair adjacent weeks of the same team-season
  only. Kendall tau-a is appropriate for continuous FFBS steps (no ties).
- Profile all Phase A accepted books, not a subset with more convenient lines.
  At every parameter candidate the same starts, KL and rate gates are recorded;
  optimisation/multistart failure makes the profile value infinite. KL/rate
  failures are visible, not discarded from the sum. Final inversions must pass
  **all** gates. Local grids use ForwardDiff gradients with BFGS; G0 reuses
  Phase A's exact finite-difference inversion and score grid.
- G2's global parameter is the nonnegative shared **lambda3** in goal units.
  Its optimiser coordinates are log(lambda1), log(lambda2); every reported rate
  is expected goals lambda1+lambda3 / lambda2+lambda3. DC tau is not clipped;
  infeasible parameter/rate combinations have infinite loss.
- Golden-section refinement uses the neighbours of the best coarse candidate;
  boundaries are retained as candidates. Profiles, including all evaluated
  refinement points, are saved. Global dependence is fitted on full books, then
  held fixed for heldout-line inversion. Consequently the heldout-line check has
  shared-parameter information from heldout lines; it is **not** fixture CV.
  Similarly, G1's R6 10b limits team-model theta fitting to honest_train, but its
  global grid parameter was estimated from both seasons. That informational
  row is not a fully honest end-to-end forecast comparison.
- Heldout winner is the lowest pooled mean absolute **selection** residual
  across all heldout OU0.5/1.5/3.5 and BTTS lines (including their complements).
  A fixture with more heldout lines contributes more terms, as in Phase A's
  per-line design. Bootstrap replicates resample fixtures with all their terms.
  CIs condition on the fitted grid parameter/rates (no re-estimation within a
  bootstrap), and do not account for additional team/temporal dependence.
- All grids are renormalised on 0:10 x 0:10, exactly Phase A support. Reported
  rates are the untruncated expected goals, so finite-support means have a
  truncation error. The t04 small-rate examples measure this explicitly; the
  1e-6 mean identity is not falsely asserted when truncation exceeds 1e-6.
  Frank CDF rectangle differences clamp only negative roundoff no larger than
  2e-14; material negative mass is an error. Near-zero kappa (<=1e-7) is exactly
  independence. No probability floor is added to materially positive cells.
- Outcome grid/G0 comparisons use all 517 Phase B fixtures for both 1X2 and
  OU2.5. Paired close comparisons use only **quoted** closes: Phase B's published
  `rates_vs_goals.csv` has 517 1X2 closes but only 353 OU2.5 closes. No missing
  OU2.5 quote is imputed or silently assigned a different line. Every output row
  reports its n; raw fixture rows flag has_close. This is quote availability,
  not an outcome- or fit-selected subset.
- The largest-step table ranks **absolute posterior median signed Delta q**,
  with posterior 5/95% intervals, separately for both theta protocols. This is
  a stable ranking of estimated changes, not a search for the largest random
  posterior draw or evidence of a causal manager effect.

## Literature actually read in the personal research library

- `mchale2011-modelling-dependence-goals-scored-opposing`, source-PDF **p. 9**:
  bivariate Poisson shared component allows only nonnegative covariance; **p. 11**
  gives the Frank CDF and discusses positive/negative dependence and the discrete
  margins caveat; **p. 12** gives the four-CDF rectangle likelihood and reports
  negative fitted dependence in national-team goal data; **p. 13** warns that
  continuous Kendall identities do not hold for discrete margins. We implement
  the rectangle construction with Poisson margins, not their negative binomial
  margins, and do not transfer their empirical dependence sign to this market.
- `genest2007-primer-copulas-count-data`, source-PDF **p. 1** (abstract) warns
  that atoms/ties invalidate familiar continuous copula inference relations;
  **p. 2**, not p. 1 as stated in the brief, explicitly discusses nonuniqueness
  of Sklar's representation in the discrete case. Parametric copula grids remain
  valid constructions, but a fitted Frank parameter is not a uniquely identified
  arbitrary discrete copula. We fit the book probabilities parametrically rather
  than infer score copulas from a continuous-rank identity.
- `ruppert2010-copulas`, source-PDF **p. 4** says t-copula degrees of freedom
  control tail dependence; **p. 12** gives zero Gaussian asymptotic tail dependence
  away from perfect correlation and the t-copula coefficient; **pp. 9–10** define
  Kendall concordance/discordance and the sample pair-sign statistic; **p. 16**,
  not pp. 9–10, gives tau=(2/pi)asin(rho). That identity is used only for the
  continuous **team-step** Gaussian benchmark, never for discrete goal scores.

## Brief/DESIGN corrections

DESIGN's historical exclusion of alternative score grids and outcome scoring
is superseded locally by the explicit B3 brief; no Phase C implementation or
score-grid decision is changed. DESIGN's historical r04 Phase D filename is not
this B3 r04. The two page-reference inaccuracies above are documented rather
than copying an unsupported citation. Truncated-support means need not equal
untruncated lambda to 1e-6; the brief explicitly permits documented truncation.
The brief's wording could imply 517 paired OU2.5 close comparisons; only 353
such closes are present in the pinned Phase B population. Grid/G0 losses still
cover all 517 fixtures; quote-paired comparisons are explicitly labelled.
