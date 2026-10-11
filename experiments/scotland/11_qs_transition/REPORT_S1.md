# S1 — Transition EDA and a conservative market-informed prior

## Summary (10 lines)
1. W2 reproduces: cohort counts **5 / 6 / 2**, all six runs score, and all DB audits are unchanged. [q6_cohort_gate; verification]
2. W2 TD has persistent relegation/promotion bias **−4.24 / +4.46 pp**; A2 reduces it to **−0.63 / +0.88 pp**. [q1_bias_first20]
3. QS is different: **+1.50 / +1.44 pp**, not W2's negative relegation bias; its panel is shorter. [q1_bias_first20]
4. QS promotion bias is **+5.02 pp at game 1**, **−0.25 at games 6–10**, **+1.61 at 11–20**: entry error, not a monotone universal cold-start curve. [q1_bias_bins]
5. Opening market re-rating is mostly quality: relegation **+0.212**, promotion **−0.220** in opponent-anchored D units. [q7_opening_shift]
6. Cups support positive, unequal tier gaps; bookmaker q gaps are **0.397 / 0.281 / 0.185**, versus A2's **0.235**, not 0.47. [q3_tier_cup_estimates; q3_a2_units]
7. Summer q SD is **0.092** versus **0.053** within-season; calendar-adjusted excess is uncertain. [q4_paired_diffusion]
8. QS promotion absolute error rises **1.68 pp/week** since cutoff after controls; relegation does not show that pattern. [q5_conditional_age]
9. Select **A2-centred P2**, conservatively: first-quote D RMSE **0.160**, versus league mean **0.210**, A2 **0.169**; superiority over A2 is not established. [q2_prior_headline; q2_paired_error]
10. Chronology passes, but naive insertion reuses historical market evidence. Resolve that factorisation before F1/F2; build/test QS+A2 first. [q12_status; q12_overlap]

All source names in brackets mean `eda/results/<name>.csv`. Figures are in `eda/figures/`.
This is descriptive diagnosis and prior-only forward prediction, **not a new fitted-arm gate**.

## 1. Data, reproduction and provenance — Q6/Q12

### Inventory

| Tier | Accepted inverted closes |
|---|---:|
| Premiership | 956 |
| Championship | 477 |
| League One | 835 |
| League Two | 693 |

Source: `q6_quote_totals.csv`. Only accepted inversions enter the reconstruction.
`q6_coverage.csv` gives each club-season's fixture/quote/rejection counts, first observed game,
last date and diagnostic recency-weighted effective count. Its extra pre-market roster season is
kept for transition classification, not treated as observed market history.

The Championship has **no supplied accepted inversions in 25/26**; its roster contains **175**
fixtures and **10** clubs. The frozen W2 cache has **5** qualifying TWA 1X2 closes, not a complete
season. This is an estimator/cache-specific count, not a claim that the current archive has only
that many quotes. Sources: `q6_coverage.csv`, `q6_w2_close_coverage.csv`.

Forward first-quote coverage is thin: **11 / 10 / 10 / 8** club-seasons in 22/23 through 25/26;
the latest first accepted quote is game **10**, not always the opener. Four 22/23 movers have no
usable old club reconstruction. Every club is listed in `q6_transitions.csv` and
`q2_prior_predictions.csv`; minimum-count sensitivity is in `q6_bias_thresholds.csv` and
`q2_coverage_sensitivity.csv`. Missing history stays an explicit fallback, never a zero-variance
observation. `q6_2627_coverage_only.csv` contains only counts/date ranges, including scheduled
fixtures; no held-back outcomes or performance were analysed.

### Reproduction and exact vintage assignment

- All five W2 runs cover **1,070** lower-tier target fixtures; QS covers **710**. The W2 cohort gate
  is **5 relegated / 6 promoted / 2 entered-SPFL** club-seasons, with **96 / 114 / 40** cohort
  fixtures before pricing missingness. Sources: `q6_cohort_gate.csv`, `verification.csv`.
- Full UUIDs and namespaces were resolved through harness tables: `resolved_runs.csv`.
- W2 datastore checksum: `db3ed548191bcd5287a6c1a18e34a105`; only that cache priced existing fits.
- Scores/observations use the harness `_score_one` path without persistence. Decomposed gate
  means match the harness to numerical tolerance: `verification.csv`.
- Important storage trap: `match_latents.fold_id` attaches aggregate latents to the first fold;
  it is **not prediction-vintage provenance**. Cutoffs are reconstructed from saved splitter
  metadata and `Data.get_next_matches`, with unique/full latent-ID coverage. Historical and
  fitted-current-season IDs both enter the cutoff. Sources: `q12_cutoff_check_*.csv`,
  `folds_*.csv`, `history_*.csv`; implementation authority `db_storage.jl:795–796`.

### Filtration and evidence reuse

`q12_source_manifest.csv` records club/opponent endpoint IDs, hyperparameter-label IDs,
normalisation IDs, maximum source date, gauge, fallback and target boundary. All numerical prior
ingredients use seasons strictly before the target season. `q12_fold_prior_hashes.csv` hashes
actual source/parameter values and confirms identical inputs across folds of a season.
Chronology violation count: **0**. Sources: `q12_status.csv`, `verification.csv`.

The saved QS market feature has **1,107** covered rows out of **1,430**. Its exact mask, rather
than all training fixtures, is used for overlap auditing. The proposed all-history summaries
potentially reuse **509–590** covered historical quotes per QS fold; the two endpoint networks
alone reuse **140–147**. Sources: `q12_qs_market_likelihood.csv`, `q12_overlap.csv`.

**Consequences:** do not put an S−1 endpoint prior at the beginning of an older multi-season
window, and do not count the summarised closes again as independent historical likelihood
observations. A disjoint pre-window construction or a genuinely sequential boundary model must
be specified. An overlapping empirical-Bayes regulariser is a different, explicitly labelled
arm—not exact Bayes, and not repaired by enlarging covariance. The prior-only scores below test
all-history mean transport, **not** a subsequently changed disjoint implementation.

A calendar-year League Cup label initially admitted held-back rows into a raw cache export.
No outcomes were inspected/scored. Final outcome-bearing exports enforce the actual date
embargo, not string season comparisons; the correction is logged in `eda/DECISIONS_LOG.md`.

## 2. Cause verdict — Q1/Q4/Q5

### First-20 bias, transitioning club's win probability (pp)

| Existing fit | Relegated into L1 | Promoted into L1 |
|---|---:|---:|
| TD base | −4.24 [−6.28, −2.19] | +4.46 [+2.84, +5.99] |
| TD A2/full/carry/jump | −0.63 [−1.56, +0.58] | +0.88 [−0.85, +2.48] |
| GRW base | −5.44 [−9.96, −1.09] | +3.41 [+0.77, +6.79] |
| GRW step/A2/carry/jump | −2.65 [−5.20, −0.01] | +1.19 [−0.91, +3.94] |
| m12 TD reference | −2.52 [−6.32, +1.37] | +4.72 [+2.28, +7.26] |
| QS marketobs, shorter panel | +1.50 [−0.36, +4.05] | +1.44 [+0.90, +2.26] |

Source: `q1_bias_first20.csv`, canonical club-season bootstrap. W2 has **5 / 6** priced club-seasons,
QS **3 / 4**. Do not interpret these tiny-cohort intervals as a powered mechanism test.
`q1_joint_fixture_sensitivity.csv` merges club-season clusters sharing a fixture; QS has only
**2** independent connected blocks per gate direction. `q1_club_bias.csv` shows every club;
leave-one-club-season ranges are in `q1_bias_first20.csv`. `q1_common_panel.csv` makes matched
QS/W2-panel comparisons; `q1_paired_a2.csv` uses identical fixtures.

| Cause | Verdict and numerical signature |
|---|---|
| **a. Cold start** | **Contributes to QS promotion entry error; not a universal explanation.** QS promotion: +5.02 at game 1, +1.91 at 2–5, −0.25 at 6–10, +1.61 at 11–20. Relegation is −0.11, +4.08, +0.01, +1.40 respectively. Not monotone decay. Old in-scope/all-tier history cross-tabs are in `q1_history_bias.csv` and the raw club observations. |
| **b. Tier scale / transport** | **Strongest explanation of W2's directional persistence.** TD base is still −5.80 / +3.41 at games 11–20; A2 reduces these to −3.18 / +0.88. Market opening re-ratings have the expected opposite signs. This supports a transport/tier correction, but mover-only data cannot separate tier value from summer/selection jumps. QS has largely absorbed the W2 relegation sign; do not apply a blanket positive relegation probability correction to it. |
| **c. Summer step** | **Plausible uncertainty misspecification, weak explanation of mean transition bias.** QS non-movers are −0.41 pp at game 1, −0.70 at 2–5, +0.88 at 6–10, +1.04 at 11–20; there is no large uniform signed opening lag. Noise-adjusted summer q/s SD is 0.092/0.044 versus 0.053/0.026 within-season. But calendar-normalised ratios are only 1.22 [0.93,1.54] and 1.18 [0.84,1.63]. A distinct summer process is not securely identified. |
| **d. Refit lag** | **Some promotion-specific evidence; not the general cause.** QS absolute-error age slope is +1.68 pp/week [0.15,3.22] for promoted clubs, −0.18 [−1.70,1.35] for relegated clubs, +0.25 [−0.30,0.80] for non-movers. Signed slopes are inconclusive. These are cross-fixture associations, not the gain from weekly refitting. |

Sources: `q1_bias_bins.csv`, `q4_paired_diffusion.csv`, `q5_conditional_age.csv`.
Q4 compares disjoint equal-count windows with **108** paired non-moving club-seasons in **14**
league-season blocks. Gaps are different (**193** versus **94** midpoint days); uncertainty is
approximate, and no calendar-matched causal summer estimate is claimed. The current QS ordinary
step and historical macro-scale summaries are in `q4_qs_scales.csv` and `q4_qs_boundary_proxy.csv`;
the latter is a posterior-mean plug-in proxy, not full predictive covariance.
Q5 controls club-season and game bin, excludes season-opening fits and very stale gaps.
`q5_age_bias.csv` and `q5_own_since_cutoff.csv` give the full cross-tabs. There is only one saved
held-out vintage per fixture; `q5_refit_events.csv` compares different fixtures/opponents and
cannot identify a weekly-refit counterfactual.

Rate-level residuals are available in `q1_rate_summary.csv`, but are rotations of **log posterior
mean lambda**, not posterior means of isolated club q/s. D includes home-effect mismatch, and U
is a fixture sum; these caveats prevent invented latent attribution.

## 3. Market re-rating, tier value and style — Q2/Q3/Q7/Q8/Q9

### Quotes and gauges

For a quote, `D=(log λh−log λa−γ)/2` and `U=(log λh+log λa−γ)/2`, with fixed common `μ_ref=0`.
Only D/U are identified by a fixture. Old club estimates use connected constrained
league-season least squares: sum(q)=0 within division, s on the common reference, including the
league environment. The old endpoint is the fixed last **80** accepted league fixtures, not a
future-smoothed destination-season rating. Gamma used for validation is estimated from earlier
seasons only. Graph rank/conditioning, reconstruction covariance and own counts are recorded in
`q3_reconstruction_graph.csv` and `market_team_reconstruction.csv`.

Opening D shifts against frozen old club/opponent anchors are **+0.212** for relegation and
**−0.220** for promotion; by the game-5 checkpoint they are **+0.260 / −0.281**. U shifts are
**+0.057 / +0.024** initially, **+0.014 / +0.030** at game 5. These are fixture innovations, not
independently observed club states. Games 10/20 are in `q7_opening_shift.csv`.
Whole-destination-season descriptive q shifts are **+0.322 / −0.314**, with style shifts
**+0.010 / +0.019**; they are never used as opening labels. Estimated style/quality dispersion
ratios are **0.178 / 0.285**, but small cohorts and gauge/measurement noise limit precision.
Sources: `q7_mover_shifts.csv`, `q7_shift_summary.csv`.

### Tier evidence

A2's **total** step 0.47, split 0.48/0.52, means **Δq=0.235, Δs=−0.0094**.
Source: `q3_a2_units.csv`. Cup 1X2 does not independently identify precise style or attack/
concession steps; no such estimate is fabricated.

| Adjacent division | Cup bookmaker q contrast | Excluding Old Firm | Mover net contrast, relegation / promotion |
|---|---:|---:|---:|
| Premiership–Championship | 0.397 [0.365,0.431] | 0.415 | 0.536 / 0.476 |
| Championship–L1 | 0.281 [0.251,0.312] | 0.290 | 0.332 / 0.313 |
| L1–L2 | 0.185 [0.146,0.221] | 0.193 | 0.235 / 0.235 |

Sources: `q3_tier_cup_estimates.csv`, `q3_tier_mover_estimates.csv`. The mover contrasts are
**net tier rebase plus free club/summer jump**, not an independently identified tier parameter.
The cup bookmaker design is full rank. Deleting one linking club leaves rank intact and q gaps
in **[0.389,0.413] / [0.269,0.288] / [0.166,0.203]**. Sources:
`q3_bridge_club_sensitivity.csv`. Competition exclusions are in `q3_tier_cup_estimates.csv`.
A single constant cup step is **0.283**; the adjacent pattern is visibly nonlinear.

Regulation-goal exposure contrasts imply average per-tier q steps **0.315 / 0.270 / 0.374**
for gaps 1/2/3, based on **131 / 23 / 14** cross-tier ties. Their intervals are wide, and these
aggregate corrected scoring contrasts do not isolate lineup/competition effects. Cup coverage
and explicit regulation scores are recorded in `q3_cup_coverage.csv` and `q3_cup_regulation.csv`;
knockout winner flags/extra-time totals are not substituted for regulation goals.

Q9's bookmaker/Betfair bridge uses **2,955** overlapping league fixtures. It calibrates the
bookmaker 1X2 log-odds contrast into approximate D units, not a fictitious cup U inversion.
Final bookmaker quote timestamps are unavailable, so source alignment is **unverified**;
current-season league reconstructions anchor cup diagnostics descriptively. Cup priors are not
cleared by this analysis. Sources: `q9_source_bridge.csv`, `q9_source_by_tier.csv`, `q9_status.csv`.
Lineup-rotation/source separation was not completed. Q11 squad discontinuity was not run;
`q11_squad_status.csv` records the optional omission.

Q8's estimated q carry contracts from **0.734** to **0.668** as earlier training seasons expand,
but the estimated-carry candidate does not improve pooled forward first-quote error over the
selected conservative prior. Measurement-error sensitivity and source-strength overlap are in
`q8_selection_slopes.csv` and `q8_nonmover_overlap.csv`. Missing overlap is not extrapolated into
a causal selection effect. Whole-old-season endpoint and pre-market-era results-only robustness
were not run; `q8_scope_status.csv` records those limits.

## 4. Proposed prior: A2-centred P2, with explicit evidence-policy gate

### Mean rule and per-fold construction

Choose the simple P2 nested form with fixed unit carry, not an additional freely estimated
carry/summer/tier decomposition. Let `x̂_i^-=(q̂_i^-,ŝ_i^-)'` be the old endpoint estimate,
`g∈{relegated,promoted,entrant}` and S the target season:

\[
 m_{i,S}^{rel}=\widehat x_i^-+\widehat d_{g,S},\qquad
 x_{i,S}^{rel}\sim N_2(m_{i,S}^{rel},C_{i,S}).
\]

For historical first-five-game **fixture** labels in seasons `<S`, form design rows
`f_D,g=I(home∈g)−I(away∈g)` and `f_U,g=I(home∈g)+I(away∈g)`. With frozen earlier-season opponent
anchors, let `b_D=q̂_h^-−q̂_a^-`, `b_U=ŝ_h^-+ŝ_a^-`. Then separately for c=D,U:

\[
 \widehat d_{c,S}=(F_c'F_c+2I)^{-1}
 \{F_c'(y_c-b_c)+2d_{c,0}\},
\]
\[
 d_{D,0}=(+0.235,-0.235,0)',\qquad d_{U,0}=(0,0,0)'.
\]

The fixed ridge is a weak two-observation pooling rule. Directional means are learned separately,
not forced to be antisymmetric. Sparse first-SPFL means remain strongly pooled. This A2-centred
candidate was chosen after the initial EDA comparisons; its forward construction is filtered,
but the selection is **not an independent untouched-panel validation**.

Use `C_i=V_i^-+Σ_g+U_d`, with diagonal pooled boundary/predictive dispersion estimated only from
earlier mover labels and a declared weak fallback when no earlier labels exist. The observed
fixture residual dispersion contains quote/opponent error; it is a conservative working bound,
**not an identified intrinsic state-jump covariance**. Do not force its style/quality ratio to
QS's within-season ratio. The direct-fixture interval check below uses the pooled residual plus
hypermean uncertainty, not a certified production state-prior covariance. S2 must settle that
covariance/evidence factorisation before claiming calibrated state priors.

`q2_prior_parameters.csv` contains all per-season means, predictive scales, counts and source
seasons. For the panel, learned q shifts (relegated/promoted) are **+0.215/−0.292** in 23/24,
**+0.280/−0.288** in 24/25, **+0.236/−0.295** in 25/26. U shifts are small and separately learned.
No target-season quote, future cohort label or 26/27 result enters their construction.

**Missing history:** use the old division mean style and zero relative q, with the known-direction
A2-centred shift and broad uncertainty; for outside-SPFL origin use the earlier entrant parent
or an explicitly unlearned broad mean. Do not extrapolate an invented fifth-tier step. Rejection,
sparse/disconnected reconstruction and long staleness must select a low-reliability fallback.

**Production gauge:** if `θ_ℓ` is the common-pyramid quality reference, map
`m_i^G=m_i^rel+θ_new`. The learned `d_g` already contains the division rebase; do **not** add
`θ_old−θ_new` again. A2-compatible predictor terms must contribute the tier effect exactly once.
League quotes alone do not identify cross-division θ; cups suggest contrasts but have unresolved
source/rotation provenance. A pooled mapping therefore needs an explicit structural A2 assumption
or an earlier-season connected estimate, not silent per-division recentering.

Attach the prior only to a genuinely new initial node at/after its evidence cutoff, or a
well-defined season boundary. Keep ordinary innovations zero-mean and never reset incumbents
at every refit. **Reject the naive implementation on the existing historical market likelihood.**
Prefer disjoint pre-window sources; if these are too sparse, retain A2 fallback. A sequential
boundary must condition on the old posterior, not multiply a second independent market-summary
likelihood. If the manager chooses overlapping EB regularisation instead, label it and repeat
the prior-only check under that exact policy before production fitting.

### Prior-only expanding-season check

One fixture once, including fixtures with two movers. Errors are in log-rate rotation units,
not win-probability pp. “First quote” means first accepted quote at its actual league game.
“First five” means all accepted fixtures through actual game 5, not a smoothed club rating.

| Quantity | League mean | A2 step-only | Selected A2-centred P2 |
|---|---:|---:|---:|
| First-quote D RMSE | 0.210 | 0.169 | **0.160** |
| First-quote D MAE | 0.163 | 0.134 | **0.127** |
| First-quote D signed error | +0.033 | +0.059 | +0.052 |
| First-quote D interval coverage | 0.88 | 0.94 | 0.97 |
| First-five D RMSE | 0.183 | 0.172 | **0.170** |
| First-quote U RMSE | 0.084 | 0.092 | 0.084 |
| First-five U RMSE | 0.093 | 0.079 | **0.076** |

Source: `q2_prior_headline.csv`; **32** first-quote fixtures from **39** mover club-seasons,
**146** first-five fixtures from **36** covered club-seasons. Coverage is descriptive; weak
fallback intervals are especially wide. Selected D/U calibration slopes are **0.75/0.52** for
first quotes, **0.70/0.49** for first-five fixtures: there is residual mean/spread miscalibration.
Full signed error, MAE, RMSE, log score and coverage are in the same CSV.

Pre-panel 22/23 has only **10** first-quote fixtures; selected D RMSE equals A2's **0.146** because
there are no earlier mover labels and the prior uses its declared A2 fallback. Within the panel,
selected D RMSE is **0.166**, versus A2 **0.178**, league mean **0.228**. Sources:
`q2_prior_unique_fixture.csv`. This is not evidence for a learned pre-panel transport law.

Selected-minus-A2 D MSE is **−0.0029 [−0.0086,+0.0026]** at the first quote and
**−0.0007 [−0.0030,+0.0016]** across the first five: **no clear superiority over A2**.
Source: `q2_paired_error.csv`, paired fixture bootstrap; repeated fixtures within a club make
first-five uncertainty exploratory. Lower-gate directions and coverage-conditioned results are
in `q2_lower_gate_prior.csv` and `q2_coverage_sensitivity.csv`; pooled full-pyramid improvements
must not be assumed to hold identically in every lower cohort.

Uncentred P2 first-quote D RMSE is **0.198** and P1 **0.216**; neither beats the A2-centred version.
Estimated carry also fails to improve it (**0.196**). P3 with a fixed step and free residual jump
is algebraically P2 in this fixture design, not an independently identified decomposition.
Sources: `q2_prior_headline.csv`, `q2_prior_summary.csv`.

## 5. Outcomes are not a proof of mover-specific market inefficiency — Q10

On QS's aligned priced transition fixtures, market win residual is **+4.44 pp** for relegated
clubs [−8.23,+17.77], **+0.03 pp** for promoted clubs [−10.23,+10.38]. These intervals do not
identify a stable market inefficiency. Source: `q10_market_residual.csv`; season, home/price-bin
and matched-control tables are `q10_by_season.csv`, `q10_calibration.csv` and
`q10_matched_market_residual.csv`.

QS categorical fixture log loss minus market is **−0.0155 [−0.0410,+0.0103]** on mover-first20
fixtures: not evidence that QS is unambiguously worse against outcomes on this historical panel.
Source: `q10_fixture_logloss.csv`. This categorical fixture score is distinct from the harness's
per-observation Bernoulli score. Existing A2/base paired harness-style differences are in
`q10_paired_model_scores.csv`; no new winner/non-inferiority gate is declared here.
No ROI mining and no held-back performance check were performed.

## 6. Recommended S2 build list and S3 arms

### S2, in order
1. **QS+A2 composition smoke first (F3).** Verify total-step/q conversion and single tier-effect
   accounting; all original configs remain unchanged. No market-summary evidence reuse required.
2. **Prior provenance/gauge API before F1/F2.** Store source IDs/seasons/hashes, node date, exact
   market-likelihood exclusion policy, fallback/reliability, common-scope map and covariance.
   Reject chronology, double-counting and backdated initial-node violations as hard checks.
3. **Opt-in season-boundary/new-entrant mean support** for the selected P2 structure. Unit carry
   first; no extra estimated carry or free q/s covariance yet. If disjoint data are insufficient,
   fall back to A2 rather than quietly reproduce the all-history EB prior as exact updating.
4. **Optional target-season uncertainty step (F4)**, off by default, distinct from prior/jump
   uncertainty. Do not duplicate summer variance or use a variance-only step as a mean correction.
5. Defer pooled cups, league-pxG adjustments and squad-dependent priors until their independent
   source/identification checks pass. Q9 does not justify F7.

### S3, one factor at a time on the frozen W2 panel
- Control: unchanged `qs_marketobs` recipe, reproduced on the longer panel in S3 before comparisons.
- **F3: QS+A2**, the lowest-risk first corrective arm.
- **F1: eligible old-club market means, no learned transition shift**, only after the evidence-policy
  gate; isolates whether historical club information helps.
- **F1+F2: A2-centred P2 transport**, paired against the same eligible F1 arm; isolates learned shifts.
- **F4: target-season macro uncertainty only**, to test secondary summer evidence.
- **F6: weekly refits only**, a secondary lag arm; budget after the mean-transport tests.
- Combine only passing factors in S4; do not begin with full-pyramid+cup+mean+summer+weekly changes.

All arms retain W2's club-season transition-bias gate and paired fixture log-loss gate.
Existing W2 A2 finalists were selected on this same panel; their improvements are descriptive
motivation, not independent validation of the proposed QS arm.

## 7. Open questions / limits

- Can an operational F1/F2 summary be disjoint and fresh enough with the short market history?
  If not, choose sequential posterior transport or explicitly labelled EB—not a silent workaround.
- State-jump covariance is less identified than mean transport; predictive fixture residuals cannot
  be declared intrinsic q/s process noise. First-quote style calibration needs further pooling checks.
- Championship coverage, late first quotes and inversion/source changes limit the apparent transport
  law. Repeat prior-only tests if the evidence policy, source table or gauge changes.
- Cup adjacent-gap nonlinearity is credible descriptively, but source timestamps, rotation, neutral
  venues and inherited league-anchor error prevent a production cup prior from being cleared.
- Q11 and Q9 rotation bins were skipped. Q8 endpoint/older-results robustness is incomplete, explicitly
  recorded rather than converted into a negative finding.
- The existing QS run has no 23/24 predictions. S3 needs a longer-panel control and a frozen shared
  market-input recipe; the EDA did not fabricate those missing QS predictions.
- No paired successive prediction vintages exist here; Q5 cannot quantify weekly-refit benefits.
- No held-back outcome test, production fit, `src` edit or DB write occurred. See `eda/README.md`
  for execution and `eda/HANDOVER.md` for the manager's next decision.
