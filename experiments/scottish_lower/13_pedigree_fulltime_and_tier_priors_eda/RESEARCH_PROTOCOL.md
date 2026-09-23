# Estimands, claim audit and decision protocol

Recorded 2026-09-23, before this suite's empirical results. This is exploratory research, not a pre-registration of a future trial. The work-package narrative is a set of hypotheses, not a source of verified observations.

## 1. Claims that require independent evidence

- **A Gaussian prior plus sum-to-zero does not impose a 0.76 log-goal or 54% probability ceiling.** A vector `(c,-c,0,…)` has zero sum for arbitrarily large finite `c`. Gaussian support is unbounded. Posterior shrinkage, sparse history, likelihood weighting, component competition and score-grid design can compress predictions, but that is not the asserted mathematical cap.
- **Rejecting one cut-posterior funnel does not prove that goal feedback has no role.** It shows those implemented alternatives performed worse in their measured experiment. The primary evidence is [Experiment 12](../12_decoupled_generative_xg/README.md), not its proposal.
- **The 1.7214 slope belongs to Experiment 12's `m02_joint_gamma_poisson`**, run `97c7a3d9-a05a-4029-90cb-e34279b8c791`, 710 held-out fixtures/40 folds. It is not automatically the production `m12` slope or the September 2026 slate slope. The +0.0028 LogLoss contrast is the clean-coverage folds 21–40 contrast, not the all-fold headline difference.
- **Operational status is not synonymous with professional registration, league, wealth or pedigree.** Hybrid, unknown and dated changes must survive into the dataset. Classifying a team from its later success is outcome leakage.
- **42 clubs per season does not imply 42 unique clubs over six seasons.** Membership must be reconstructed per season. Seasons beyond verified fixture/source coverage remain unverified, not copied forward as ground truth.
- **The live stake narrative is not itself an audit.** £20.35 / £43.53 = 46.75% of slate stakes, not 46.75% of bankroll. Confirm the fills, settlement, account opening balance, quote instant, run UUID and probability source before calling it realised drawdown.
- **Edge has multiple units.** At quoted raw probabilities, Cove has `0.297 - 1/5.70 = 12.16` percentage points probability edge and `0.297*5.70 - 1 = 69.29%` pre-commission expected return. QoS gives `11.69` points and `86.48%` respectively. Neither calculation directly yields the prompt's +23.8%/+14.4%; those require another probability/trust/commission convention. Do not silently label these numbers the same statistic.

## 2. Data and measurement contract

Use a fixed extraction cutoff, finished regulation-time fixtures and immutable match IDs. Record actual tournament metadata; do not trust a hand-written tier-ID mapping. League membership is season-specific. Cup entrants outside SPFL, reserve sides, unresolved identity, extra-time-only score ambiguity and missing tier assignments are exclusions with counts, not additional low-tier teams.

For prices, retain source, timestamp relative to kickoff and all three 1X2 selections. When no better supported vig-removal method is available, define `p_k=(1/o_k)/sum_j(1/o_j)`. A last archived price without a pre-kickoff guarantee is an **archive proxy**, not a verified tradable close. A bookmaker quote is not Betfair. Mixed-time selection vectors are not a coherent snapshot.

Report distinct outcomes separately:

1. Goal difference, measured in goals/match.
2. Shot and BBC proxy-xG difference, with coverage denominators and calibration provenance.
3. `log(p_home/p_away)` and/or `logit(p_home)-logit(p_away)`, explicitly named; neither is intrinsically a Poisson log-rate difference.
4. A log-rate difference only if obtained from a documented score-model inversion.

An unconditional league mean goal difference mostly reflects home advantage, not a between-league strength gap. Cross-tier bridges identify relative levels, subject to selection and home-venue confounding. Estimate distinct adjacent steps rather than assuming equally spaced tiers. Report sample sizes, uncertainty and sensitivity to cup competition, season, venue and repeated clubs. Sparse bridge cells or disconnected tier graphs are identification failures, not zero gaps.

## 3. Within-tier status comparisons

Orient mixed-status match outcomes from the FT team's perspective; control for whether FT is home. Keep FT–FT, PT–PT, hybrid and unknown separate. Publish a strict evidence-backed panel and, if useful, a clearly labelled continuity-inferred sensitivity panel. Descriptive FT advantages are not causal training-regime effects: payroll, player quality, relegation, team identity and recruitment are confounders.

A team-fixed-effect regression may remove nearly all status variation. That is a limitation of identification, not a reason to drop team effects and claim causation. Validate any status proxy with held-out **clubs or club-seasons**, not random lineup rows. Missing valuations and different scrape times can encode coverage instead of resources.

## 4. Compression attribution and transition estimands

To assess association with compression, join the **same** immutable run, held-out fixtures and price instant. Compare `s_market = a + b*s_model + season/venue terms + error` with a nested model adding point-in-time status/pedigree, then validate on future seasons or held-out clubs. Report out-of-sample changes, uncertainty, and coefficient stability. A change in `b`, partial R² or explained residual variance is not a causal percentage of the 0.7214 slope excess. Cross-sectional goal differences alone cannot answer this question.

Transition match counters require complete preceding season membership and all played league matches, not just matches with available odds. Separate promotion from relegation, distinguish entry into the observed lower-tier dataset from a genuinely unseen team, and retain censoring of incomplete 26/27 windows. Windows are matches 1–5, 6–10 and 11–20. Compare production posterior-predictive win probabilities (draw-average score grids, **not** a score grid at mean lambda) with same-time market probabilities. Market prices are a benchmark, not observed latent talent.

There is no universal number of matches until a prior 'learns the truth'. In a simple normal-normal illustration, posterior weight on a prior is `v/(v+n*tau²)` for observation variance `v` and prior variance `tau²`; the actual dynamic joint model has different effective information. Any claimed adjustment time needs an operational tolerance and measured uncertainty.

## 5. Kelly accounting and decision gate

For an isolated back selection with executable decimal odds `o`, probability `p`, commission rate `c`, net win odds `b=(o-1)*(1-c)`, full Kelly is `max(0,(p*(b+1)-1)/b)`. Fractional Kelly scales this by a specified fraction. This is **not** the existing correlated multi-market/slate allocator. Reproduce the real allocator and policy to attribute its stakes; label simplified simulations as counterfactual illustrations.

Drawdown is `1 - bankroll/running_peak` along a time-ordered settled bankroll path. Losing bets, negative ROI, total exposure and drawdown are different quantities. Closing odds can assess a hypothetical close-execution strategy but cannot reconstruct T−25 fills. No removal-of-bets counterfactual should silently redistribute the freed stake.

**Decision gate:** no production change based on EDA association alone. First obtain point-in-time status/membership coverage, then test (A) pedigree/status initial-state priors, (B) residual dynamic ratings plus fixed covariates, and (C) resource-only control on identical future held-out folds. Compare proper scores, conditional calibration, tail exposure and portfolio risk under the existing policy. Separate the informational intervention from any recalibration/trust change. This suite launches none of those grids.
