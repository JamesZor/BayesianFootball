# TODO 039 — Phase B2 report (2026-10-05)

Phase B2 only. No pooled Phase C/D implementation, `src/` changes, database writes,
package additions/updates, R3/R4 refits or changes to the published A/B artifacts.
Implemented and verified, including the human's post-processing addendum.
Both fresh full runs and both fresh addendum-only runs reproduce their CSVs byte-for-byte.

## Plain-language finding

**Persistent team quality is the supported covariance mechanism. R6 is the
simpler preferred rung; R7 is nearly tied, not a decisively inferior model.**
R6 gives the highest honest point density score, improves supremacy RMSE, and
brings both axes close to nominal 90% coverage. R5 alone helps level density and
coverage but leaves supremacy undercovered and increases residual persistence.
Adding fixture correlation to R6 buys no clear additional improvement: R7's
fixture-noise correlation interval includes zero in both protocols.

**The addendum shows quality-dominated evolution, not proof of a one-dimensional hierarchy.**
R6's style/quality innovation SD ratio is **0.315340 [0.262343, 0.374520]** in
10a and **0.282411 [0.186097, 0.396350]** in 10b; R2 fixes it at 1
(`rotation_quality_style.csv`). Strongly negative fitted alpha/beta **levels**
exist even in R2, so levels correlation alone cannot diagnose innovation covariance.

**An added common weekly league shock is not supported by the honest test.**
R6 level-error ICC is **0.097602 [0.010906, 0.179128]** retrospectively, but
**−0.005788 [−0.076693, 0.058211]** in 10b (`league_shock_test.csv`, 95% week
bootstrap CIs). Do not add a dynamic league intercept solely on the 10a result.

This is evidence about covariance in this restricted market-inversion panel,
not a causal identification of team quality or a betting/forecast promotion.

### Honest 10b results

Source: `onestep_metrics_b2.csv`, protocol=10b, subset=all, n=273 per rung/axis.
Values below are rounded from the CSV, not from targets or a selected subset.

| Rung | Supremacy RMSE | Level RMSE | Supremacy mean logpd | Level mean logpd | Supremacy 90% coverage | Level 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| R2 | 0.195579 | 0.050492 | 0.142738 | 1.450992 | 82.418% | 97.802% |
| R5 | 0.202946 | 0.050514 | 0.137941 | 1.519002 | 83.883% | 94.872% |
| R6 | 0.190775 | 0.053156 | 0.252850 | 1.525799 | 90.110% | 91.209% |
| R7 | 0.191882 | 0.052917 | 0.245587 | 1.530029 | 90.476% | 91.575% |

**Trade-off:** R6's level point RMSE is worse than R2's, despite better predictive
density and much better coverage. Improved interval calibration is not an
improvement on every metric. R2 is refitted with new seeded draws here; its tiny
numerical differences from B are not overwritten into B.

Source: `paired_vs_r2.csv`, paired by the same 273 scored fixtures. The score is
exactly the requested sum of the supremacy and level marginal log densities.

| Candidate vs R2 | Mean fixture-score difference | Fixture SE | Sum difference |
|---|---:|---:|---:|
| R5 | 0.063214 | 0.018483 | 17.257292 |
| R6 | 0.184919 | 0.038492 | 50.482924 |
| R7 | 0.181886 | 0.037623 | 49.654936 |

These SEs are for each candidate **versus R2**, not for R6 versus R7, and do not
account for additional serial dependence. A near tie is not evidence of decisive
separation. Complexity plus R7's near-zero extra covariance favour R6.

### What the correlations are

Source: `convergence_b2.csv`, rho_ab rows, posterior median and 5/95% quantiles.
The sign is negative because beta means **concedes more**: improved attack and
reduced concessions move together.

| Rung | Protocol | Innovation rho_ab median | 90% posterior interval |
|---|---|---:|---|
| R6 | 10a | −0.821935 | [−0.873899, −0.756554] |
| R6 | 10b | −0.864318 | [−0.942540, −0.742311] |
| R7 | 10a | −0.823424 | [−0.879559, −0.748933] |
| R7 | 10b | −0.845963 | [−0.939632, −0.692789] |

Source: `noise_equivalence.csv`. These are posterior transformations of each draw,
not a transformation of three separate marginal quantiles:
`s²=(sigma_sup²+4sigma_lev²)/4`,
`rho_epsilon=(4sigma_lev²−sigma_sup²)/(4sigma_lev²+sigma_sup²)`.
R2/R6 fix rho_epsilon to zero.

| Rung | Protocol | Side variance s² median | rho_epsilon median | rho_epsilon 90% interval |
|---|---|---:|---:|---|
| R5 | 10a | 0.00409710 | −0.493142 | [−0.581133, −0.393114] |
| R5 | 10b | 0.00481656 | −0.395404 | [−0.517555, −0.252966] |
| R7 | 10a | 0.00306132 | 0.005305 | [−0.161264, 0.192773] |
| R7 | 10b | 0.00425593 | −0.070175 | [−0.261567, 0.145133] |

R5's negative fixture covariance largely disappears when persistent quality
innovations are allowed. That does **not** say fixture variance is zero.

### Residual correlation and persistence

Sources: `residual_correlation.csv` and `persistence.csv`, both protocols and all
scored fixtures. Persistence signs each supremacy error toward the named team,
sorts matches by date/ID, and pairs consecutive matches **within a season**.
Percentile 95% CIs resample whole team clusters across their seasons (2000 reps,
seed 3924). The 10b calculation has 527 pairs and 19 contributing team clusters.

| Rung | 10b corr(home error, away error) | 10b team-signed lag-1 correlation | Team-cluster 95% CI |
|---|---:|---:|---|
| R2 | −0.585183 | 0.159581 | [0.002244, 0.293212] |
| R5 | −0.608384 | 0.213072 | [0.056162, 0.342405] |
| R6 | −0.535612 | 0.100298 | [−0.057605, 0.237205] |
| R7 | −0.542678 | 0.115475 | [−0.040184, 0.252673] |

R2's positive persistence is borderline under this bootstrap; R6 reduces it and
its CI includes zero. This supports, but does not prove, removal of persistence.
R6's raw home/away residual correlation remains negative: predictive state
uncertainty and actual innovations also correlate side errors. It is **not** an
estimate of the observation-noise rho_epsilon alone. Shared opponents mean the
team-cluster CI is still a diagnostic rather than a fully independent causal test.

### Smoothed fit and collapsed likelihood

Source: `ladder_summary_b2.csv` (full-panel 10a theta plug-in; likelihood integrates
states but is not a marginal likelihood integrating theta).

| Rung | Smoothed R² supremacy | Smoothed R² level | Supremacy RMSE | Level RMSE | Collapsed loglik |
|---|---:|---:|---:|---:|---:|
| R2 | 0.980629 | 0.925241 | 0.061829 | 0.022503 | 820.135480 |
| R5 | 0.961775 | 0.930247 | 0.086853 | 0.021737 | 843.150822 |
| R6 | 0.988918 | 0.839247 | 0.046764 | 0.032998 | 895.563480 |
| R7 | 0.989078 | 0.838976 | 0.046426 | 0.033026 | 895.569082 |

R6 sacrifices level smoothing fit as well as level point forecast RMSE. Do not
pick the model on in-sample R² alone. All five axes, both protocols, warm,
season-open and in-season rows are retained. For example, the 23 honest
season-open fixtures have R6 supremacy/level coverage 0.9565217391304348 /
0.9130434782608695 (`onestep_metrics_b2.csv`, season-open); this small subset is
not the basis for the all-fixture decision.

## Human addendum: quality/style hierarchy and common weekly shocks

Source: `rotation_quality_style.csv`, all retained theta draws, medians and 90%
intervals. R2/R6/R7, both protocols; no new fits. The CSV contains sigma_q²,
sigma_s², sigma_q, sigma_s, sigma_s/sigma_q and corr(Delta q,Delta s).

| Rung | Protocol | Style/quality SD ratio | 90% interval |
|---|---|---:|---|
| R2 | 10a | 1.000000 | [1.000000, 1.000000] |
| R6 | 10a | 0.315340 | [0.262343, 0.374520] |
| R7 | 10a | 0.313636 | [0.256167, 0.381723] |
| R2 | 10b | 1.000000 | [1.000000, 1.000000] |
| R6 | 10b | 0.282411 | [0.186097, 0.396350] |
| R7 | 10b | 0.299647 | [0.190937, 0.437346] |

For R6, sigma_q and sigma_s medians are 0.03188268951755056 /
0.010050417892826066 (10a), and 0.02649316811790953 /
0.007474817620520855 (10b). corr(Delta q,Delta s) is
−0.019479103434475068 [−0.2519091801911399, 0.21519765499661042] in 10a,
and −0.1800155915278403 [−0.5739798139621318, 0.26676352434256784] in 10b.
Thus the innovation ellipse is dominated by quality and close to aligned with
quality/style axes, but the sampled rungs retain style variation. No exact-zero
style model or Bayes factor was fitted; a positive lower quantile under a continuous
prior is not a formal test of that nested null. This is not proof of an exactly
one-dimensional hierarchy or a three-group population.

The manager (answering pi's question on the human's behalf; the ambiguity was in the manager's addendum) set **levels** to mean season-average RTS alpha/beta across
active teams, as a point correlation at median theta; posterior intervals apply
only to the innovation rotations. Active here means present in accepted fixtures
that season: 19 teams in each season. Average all grid weeks between first and
last accepted fixtures (40 / 39 weeks); do not replace an unobserved prior-only
state with an allegedly estimated team level. Both protocol-specific smoothers
use the full panel, so these level summaries are descriptive, not honest forecasts.

Source: `rotation_quality_style.csv`, scope=levels.

| Rung | Protocol | 24/25 levels correlation | 25/26 levels correlation |
|---|---|---:|---:|
| R2 | 10a | −0.834790 | −0.863098 |
| R6 | 10a | −0.819691 | −0.856368 |
| R7 | 10a | −0.819590 | −0.856318 |
| R2 | 10b | −0.836863 | −0.865658 |
| R6 | 10b | −0.822505 | −0.859451 |
| R7 | 10b | −0.823595 | −0.860115 |

The level anticorrelation is already present with rho_ab=0 innovations in R2;
posterior levels and dynamic innovations answer different questions.

Source: `league_shock_test.csv`. One-way random-effects ANOVA, with the unequal
cluster-size coefficient `n0=(N−sum(n_week²)/N)/(K−1)`,
`var_week=(MS_between−MS_within)/n0`,
`ICC=var_week/(var_week+MS_within)`. Retain negative signed MOM estimates rather
than truncating them. Bootstrap whole weeks, treating repeated sampled weeks
as distinct groups. All scored fixtures/opening weeks and singleton weeks are
retained, pooling configured tournaments; no favourable subset is selected.

| Rung | Protocol | Level-residual ICC | 95% week-cluster CI |
|---|---|---:|---|
| R2 | 10a | 0.103385 | [0.012598, 0.186861] |
| R5 | 10a | 0.102757 | [0.011836, 0.185841] |
| R6 | 10a | 0.097602 | [0.010906, 0.179128] |
| R7 | 10a | 0.097567 | [0.010930, 0.179035] |
| R2 | 10b | 0.030076 | [−0.070476, 0.127484] |
| R5 | 10b | 0.032271 | [−0.068525, 0.130355] |
| R6 | 10b | −0.005788 | [−0.076693, 0.058211] |
| R7 | 10b | −0.003695 | [−0.076078, 0.062229] |

10a uses 517 fixtures in 76 nonempty weeks (72 with multiple fixtures); 10b uses
273 in 38 weeks (36 replicated). Bootstrap: 2000 replicates, seed 3928. The
retrospective clustering does not persist as positive evidence in honest 10b.
Common filtering/intercept uncertainty and other same-week dependence can also
produce positive ICC; the statistic alone does not causally identify a league
shock or require a time-varying mu.

### Literature checked, and what is not claimed

- `koopman2013-dynamic-bivariate-poisson-model-analysing`, source-PDF **p. 7,
  eq. 4** specifies independent normal attack/defence disturbances. The innovation
  rotation tests that independence on market-derived rates. Their beta enters
  intensity with a minus sign (defensive strength), opposite to our concessions
  convention. Our market evidence does not refute their goal-model findings.
- `baio2010-bayesian-hierarchical-model-prediction-football`, source-PDF **p. 4**
  gives exchangeable team effects; **pp. 6–7** discuss overshrinkage and introduce
  a three-group mixture. Correlated fitted levels, a quality-dominated ellipse,
  and within-week residual correlation are distinct from identifying those groups.
  No group/tier hierarchy or mixture was fitted; those remain Phase C scope.

## Implication for Phase C (requires human sign-off)

Start from **correlated same-team attack/concessions GRW innovations**, retaining
independence across teams and the independent initial-state prior. Negative
rho_ab is important to the honest density and axis calibration here. An
independent, equal-side Gaussian fixture deviation is a reasonable initial u_m
shape: R7 provides no clear evidence for extra correlated fixture noise after
accounting for the GRW covariance. **Do not set u_m=0**, and do not copy the B2
noise variance directly into a pooled model: inversion error, score-grid misfit
and genuinely transient fixture information are confounded in this observation
noise, while Phase C's book likelihood also contributes measurement variance.
The quality/style rotation suggests a useful parameterisation, not forcing style
variance to zero or fitting a group/tier mixture. The honest ICC supplies no current
reason to add a common weekly mu shock. Keep the double-Poisson grid as agreed.
B2 does not resolve book concentration,
nested-line dependence, totals-only books or the choice of pooled inference engine.

## Acceptance evidence

Statistical source SHA: `25a408e2eb031f53d5cac2478a48ed277f4a3fc9`.
Auxiliary-runner configuration and absent-reference handling commits do not change
R03's core sampling/evaluation implementation. Addendum source is
`fd8b23a9f9b82fc75ebc7056282588bbe834bca5`: the full runner appends deterministic
hierarchy post-processing, and an addendum-only runner reads the existing fits.
No new fits were launched after the human's post-processing-only instruction.
HEAD-based convergence provenance requires the statistical SHA above for core
byte-hash reruns; use the addendum SHA for the two additional CSVs.

1. **Fresh laptop tests:** Julia 1.12.1, own `pi_julia_mm_b2`, 8 threads, BLAS=1.
   `include("current_development/market_model/t03_covariance_tests.jl")` then
   `include("current_development/market_model/t02_two_stage_tests.jl")`.
   Final addendum-source fresh pane `%31`: **131/131 t03**, **131/131 t02**,
   **66.8 s / 59.8 s**, anchored `T03_DONE`, `T02_DONE`; t03 also reaches
   `R01_DONE`. Complete output is retained in `LAPTOP_TESTS.log`. The earlier
   pre-addendum t03 passed 115/115; the added assertions did not replace old ones.
2. **Config regression:** all **16 non-runtime Phase A CSVs byte-identical**, in a
   temporary output directory; every MarketPanel field equals the frozen Phase B
   source `767dbe7ec28eefe8ac04c3a25a9e9085a8350be6`; R2 loglik agrees to ≤1e-12
   at log([.07,.05,.03]), log([.12,.01,.06]), log([.03,.09,.02]). Human explicitly
   approved excluding only nondeterministic `runtime.csv`. Published A/B files
   remain untouched, including their runtime files.
3. **Engine gates:** `engine_gates_b2.csv`: **34/34**. Both reductions, all three
   original-side batch Gaussian likelihood/RTS gates, R6 FFBS moments, and all
   Phase B/TODO 023 gates pass. Maximum new batch likelihood error
   **1.6413537196058314e-12** (limit 1e-9); maximum RTS mean error
   **1.587618925213974e-14** (limit 1e-8). R6 FFBS uses the unchanged 4000-draw,
   max-z≤4.5 and covariance-error≤0.1 criteria; no thresholds loosened.
4. **Convergence:** `convergence_b2.csv`: **32/32**, both protocols, all theta.
   Max Rhat **1.0010992161573777**, min bulk ESS **4649.036542389847**, min tail
   ESS **6306.479768118173**. Budgets all exactly 4×(2000+3000), thin 1.
   Correct pinned cache SHA256 on both hosts:
   `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
   `Distributions` stays 0.25.126. The owned beast worktree is
   `/root/BF_runs/market_model_b2`; Julia 1.12.4, 8 threads while manager review
   was active, core pinning and BLAS=1. Initial preflight load was 3.84; launch
   load was 0.12; second launch 1.12. No other session was operated.
5. **Artifacts:** every requested B2 table and both visually inspected PNGs
   exists; `noise_equivalence.csv` additionally records s²/rho_epsilon posterior
   summaries. All scientific result numbers above are from named CSVs. Wall
   times below are execution metadata from `RUN_PROVENANCE.md`, not fake seeded
   scientific measurements.
6. **Fresh full runner and reproduction:** first fresh beast pane `%250` reaches
   `R03_DONE` in **566.8761510848999 s**. Second fresh pane `%251`, same SHA,
   settings and seeds: **565.3235380649567 s**, `R03_DONE`; all **10 core CSVs
   and both PNGs byte-identical**. Logs:
   `/root/BF_runs/logs/market_model_b2/25a408e2_run1.log` and `25a408e2_run2.log`.
   First-run CSV/PNG snapshots and hashes are retained alongside those logs.
   The subsequent addendum was verified without refits: two fresh panes `%252`
   / `%253` include `r03_hierarchy_postprocess.jl` at its recorded SHA, reach
   `R03_HIERARCHY_DONE`, and give **both additional CSVs byte-identical**. Logs
   `fd8b23a9_post1.log` / `fd8b23a9_post2.log` are in the same log directory.
   `SCIENTIFIC_SHA256SUMS.txt` covers all **12 CSVs and both PNGs**; see
   `REPRODUCIBILITY.md` for the two-stage reproduction procedure.
7. **Process README:** `../../README.md` covers the note, questions, pipeline,
   inputs/outputs/gates/seeds, dated human decisions, A/B/B2 results and caveats,
   pane-ID REPL procedures, snapshot/packages, and other-league prerequisites.
8. **Tracking:** TODO 039 has dated `@pi` B2 claim/addendum/completion entries.
   `./scripts/todo.sh check` and `git diff --check` pass. TODO remains IN_PROGRESS
   for the unimplemented/unapproved Phase C/D work, not closed.
9. **Human addendum artifacts:** `rotation_quality_style.csv` has **48 rows**,
   both protocols for R2/R6/R7, posterior rotation intervals plus per-season RTS
   point levels; `league_shock_test.csv` has **8 rows**, all four rungs/protocols.
   The numbers are copied into the plain-words section above. Manager brief §7
   is copied verbatim to this branch's brief; the manager copy was read-only.
10. **Addendum simulation tests:** planted common-weekly ICC
    **0.44873747771167816**, CI **[0.3896948361754303, 0.4982498542696015]**;
    no-shock ICC **0.005882559113451667**, CI
    **[−0.020292643117333514, 0.03079024796200639]**. Both pass; see the fresh
    test log. Additional tests verify the rotation covariance algebra, signed
    negative MOM estimates, and unequal/singleton week accounting. Required
    Koopman/Baio citations were read at the specified source-PDF pages and are
    included in both this report and the README.

### First-run fit wall times (seconds; RUN_PROVENANCE.md)

Full-run times use the runner's timer after package/loader setup; they do not
include REPL startup or package precompilation. Test durations are Test's reported
suite timings. These are execution metadata, not seeded scientific CSV quantities.

| Rung | 10a | 10b |
|---|---:|---:|
| R2 | 59.377594 | 28.422563 |
| R5 | 86.006934 | 44.772617 |
| R6 | 85.769782 | 44.723747 |
| R7 | 104.996012 | 54.787202 |

## Implementation choices and brief corrections

- No modifications to TODO 023's engine: a prototype covariance arm supplies
  correlated process covariance to the reused RTS/FFBS backward pass, with an
  exact scalar-update filter for R5's rotated observation rows. The independent
  batch check uses the original home/away coordinates and full observation
  covariance, so it does not merely compare the filter against itself.
- The rotation determinant really is 1; no likelihood Jacobian correction is
  needed. Priors are exactly those specified, with log-scale and atanh Jacobians.
  The existing linked-coordinate numerical bound of ±12 is retained; it removes
  only an extremely small near-endpoint portion of the nominal correlation prior.
- Team bootstrap clusters include a team's seasons together; lag pairs never
  cross seasons. Tied dates are ordered by match ID. No data subsets or noisy
  chains were dropped to obtain a favourable result.
- B2's nonlinear **smoothed** log-total is an MC expectation under the RTS
  Gaussian, as requested. B's historical log-total smoothing transformed the
  RTS mean. Linear axes keep the B definitions. Pre-week nonlinear predictions
  keep B's seeded MC/KDE convention. This distinction is labelled in the rows,
  not silently treated as a new published B number.
- The requested sum of two marginal axis log densities is not the true joint
  pre-week density, since state uncertainty can couple the axes. Both the brief
  and the decision table are interpreted literally, with that limitation stated.
- The brief's suggestion of a literal season-boundary week in the reused engine
  was inaccurate: `MID.season_break_weeks` already derives boundaries from seasons
  and observed weeks. The config bridge makes chronological season facts explicit;
  R2/R6/R7 have normal weekly steps through the gap, not an invented summer jump.
- Literal byte identity of Phase A's `runtime.csv` cannot hold on rerun. The
  human approved the narrowly scoped exception; no wall time was forged. Tests
  reproduce the snapshot metadata on the pinned **laptop** cache. The beast copy
  is byte-identical but may have a different historical mtime; SHA is the data
  contract, not cross-host equality of wall-clock file metadata.
- The first t03 development attempt had **98 passes / 1 harness error** because
  a dynamically created Module lacked the standard `include` binding. A normal
  sandbox module fixes isolation; no statistical gate or tolerance changed.
- The addendum's `d=2(q_h−q_a)` / `ell=s_h+s_a` identities omit the static
  predictor terms: full axes are `d=gamma+2(q_h−q_a)` and
  `ell=mu+gamma/2+s_h+s_a`. The rotation describes team-state contributions;
  no mu/gamma term was dropped from any filter or prediction.
- Scottish TODO 023 reference rates are not a new-league SSE benchmark. The A
  runner reports NaN with zero common matches instead of failing on `maximum([])`;
  this does not affect the Scottish regression. English segment implementation
  remains a separate, unimplemented `src/` task.

## Blocked / handoff

No unresolved blocker. The expanded acceptance checks pass; Phase B2 only is
handed off for manager/human review. Phase C is not authorised by this report.
All owned laptop/beast REPL panes are closed after verification; unrelated
human/manager sessions were not operated. The beast checkout and logs remain
available for review; no owned running session is being left for a manager rerun.
Final artifacts are committed and pushed only to `pi/market-model-phase-b2`.
