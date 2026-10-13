# What could reduce QS's market-tracking error on tier movers?

**Deep research and read-only EDA, completed 2026-10-13.** All probability errors and MAE changes below are **percentage points (pp)**. Negative ΔMAE is better. Paths are relative to this directory unless explicitly prefixed. No Julia, QS fits, database writes, git writes, or 26/27 outcome analysis. James explicitly approved lightweight chronological residual regressions as EDA.

## Summary — 13 lines
1. **Investigate faster, opponent-aware assimilation of recent performance before another permanent tier offset.**
2. Recent six-game goal difference plus shots/SoT reduces forward-diagnostic mover MAE **5.334→4.527 pp (−0.808)**; non-movers improve **−0.917**.
3. This explains **25.0%** of held-out mover squared residuals versus a zero correction, not 25% of causal market information.
4. The reduction survives all six QS arms (**−0.808 to −1.243 pp**) and training on non-movers alone (**−0.836**).
5. **It does not repair entry:** first-five-game ΔMAE is **+0.323 promoted / +0.025 relegated**; gains are in games 6–20.
6. A 14-day information delay shrinks its mover gain to **−0.351**, implicating freshness, but weekly QS still leaves a **−0.861** diagnostic opportunity.
7. Last-season results/shots, age, continuity, manager-change flags and resource proxies do **not** individually show useful forward mover-MAE gains.
8. Higher-tier recruits have a plausible raw association but only **−0.039 pp** forward improvement; actual announced-XI strength remains untested.
9. **SofaScore values and manager fields fail historical-vintage checks:** values are constant per player; current manager names overwrite historical ones.
10. BBC match-page managers recover real changes, but their generic change flag does not improve forward mover MAE.
11. Earlier market-residual innovations buy only **−0.169 pp** for movers, against **−0.546** for non-movers: this is not just persistent anchoring.
12. **Better tracking is not proven better forecasting:** a coherent 1X2 projection improves MAE but has outcome ΔLL **+0.0030 [−0.0038,+0.0095]** overall.
13. Build one small, coherent fast-quality/shot-information arm; separately repair entry information and provenance. No diagnostic here passes the production gates.

## 1. Population, metric and experimental discipline

### 1.1 Exact reconstruction, including the non-mover baseline

Sources read as requested: `../../REPORT_S3B.md` §§1–2; `../../MANAGER_DECISIONS.md` D21–D24; `../../DESIGN.md` §9; `../../REPORT_S1.md` and `../LOGLOSS_THEORY_REPORT.md` summaries. `docs/INDEX.md` is absent in this worktree.

`queries.sql` is the SQL entry point. It includes the bounded DB extraction and the companion SQL files by `.read`; `features.sql` creates strictly lagged club-fixtures. The extract contains **3,675** historical fixtures in tournaments 54–57, July 2021–June 2026. The extra two historical seasons supply prior-season/rolling covariates; targets are only 23/24–25/26.

The control has **4,406 all-market selections on 938 fixtures**, but the relevant club-win 1X2 coverage is **1,795 selections on 898 fixtures**. We reproduce **all 221 canonical first-20 transition observations exactly**, including game numbers and residuals: 206 promoted/relegated-into-L1 observations and 15 entrant observations. `validation.json` records zero discrepancy. Non-movers exclude relegated-into-L2 clubs rather than quietly counting them as controls.

| Cohort | Club-seasons | First-20 priced club-fixtures | MAE_club |
|---|---:|---:|---:|
| Relegated into L1 | 5 | 94 | 5.802 |
| Promoted into L1 | 6 | 112 | 4.952 |
| Non-mover | 39 | 674 | 4.724 |
| Relegated into L2, additional diagnostic | 6 | 98 | 4.993 |
| Entered SPFL | 1 | 15 | 5.088 |

Source: `baseline_summary.csv`. A missing entrant has no priced club-win observations, not zero error. `MAE_club` averages absolute errors within club-season, then averages club-seasons equally. Signed club means therefore need not equal the older row-weighted frozen-bias table. We do not alter any frozen scoring output.

### 1.2 What “explains the residual” means here

Let

\[
e_{it}=100\{P^{QS}_{it}(\text{win})-P^{mkt}_{it}(\text{win})\}.
\]

For each candidate block, a **diagnostic ridge regression** predicts `e` using context plus pre-match features. Context comprises own QS win probability and its quadratic, home/away, L1/L2, game number, and mover-direction indicators. A negative predicted residual would increase the QS probability. This is an explanatory test, **not a new QS fit** or an executable betting rule.

- **Forward splits:** train 23/24 → assess 24/25; train 23/24+24/25 → assess 25/26. No random CV; no test-season fitting or hyperparameter search.
- Train on all earlier-season priced club-fixtures, with equal club-season total weight. Test only the first 20 league games for each reported cohort.
- Fixed ridge penalty **10**; training-only mean imputation and standardisation, with explicit missingness indicators. Penalties 1 and 100 are reported as sensitivity, not used to choose a winner.
- Primary held-out mover sample: **130 club-fixtures, seven club-seasons: three relegated and four promoted**. Non-movers: **446 rows, 26 club-seasons**. Training is shared across clubs; a mover-only regression would have far too few independent transitions.
- Raw association tables use all three seasons and are **descriptive**, not held-out evidence. Squared-error “R²” in the forward tables is `1 − MSE(corrected residual)/MSE(original residual)`, not ordinary centred in-sample R².
- Paired cluster-bootstrap intervals resample club-season means, **B=10,000**, seed 20261013. They condition on the fitted regressions, omit training uncertainty and multiplicity, and do not fully resolve opponent/network dependence. With seven mover clusters, they are exploratory intervals, not promotion certificates.
- These seasons were already examined in the larger project. Chronological prediction prevents feature/label look-ahead, but does not turn this research into an untouched confirmatory experiment.

**Information-time caveat:** features are available before the target fixture, while archived QS forecasts can have an earlier fold cutoff. A new arm must explicitly choose fold-cutoff features or legal online updates between folds. Fixed 7/14/28-day delays and the weekly-arm check diagnose this issue; they are **not** an assertion that we recovered the exact control-grid information cutoff.

### 1.3 What a constant club adjustment cannot fix

An oracle that subtracts the **full first-20 median** residual of each club-season is the best unconstrained constant correction for this MAE. It uses future information and is diagnostic only:

| Cohort | Original MAE | Oracle constant-corrected MAE | Maximum constant gain |
|---|---:|---:|---:|
| Relegated into L1 | 5.802 | 5.384 | 0.418 |
| Promoted into L1 | 4.952 | 4.638 | 0.314 |
| Non-mover | 4.724 | 4.260 | 0.463 |

Between-club mean offsets explain just **13.7% relegated / 4.8% promoted** of uncentred squared residuals. A time-varying or early-only correction is not subject to this constant-offset bound. The result explains why fixing average bias need not move MAE much; it does **not** prove the remainder is irreducible noise. Sources: `oracle_constant_bound.csv`, `error_decomposition.csv`.

## 2. Squad strength, turnover, higher-tier recruits and age

### Data and availability

`queries.sql` Q2 extracts starter identities, player IDs, market-value fields and DOB from historical lineup rows, not target-match performance. `features.sql` uses the **latest previous** 10–11-starter lineup, within 120 days. No target-match XI is assumed available without an announcement timestamp.

- Previous-season continuity: share of that XI with a recorded start for the same club in the previous season.
- Higher-tier arrivals: share absent from that club's previous-season starters but observed starting in a higher Scottish tier in that previous season. This is **observed player provenance**, not a complete transfer registry; it misses overseas/non-SPFL sources and some bench players.
- Age: DOB-derived age at the historical match; at least eight valid DOBs required. DOB is an immutable attribute, unlike market value.
- Consecutive-XI continuity: overlap of the two latest eligible prior lineups; no target-match absences enter it.
- Home/away side contrasts compare the mover with its **actual opponent**, not just an isolated club feature.

Coverage on the 206 mover rows: previous-season continuity/higher-tier contrasts **196**, age contrast **199**, consecutive-XI continuity **206**. Complete-case market-value contrast has only **30/206** rows; non-movers only **48/674**. Identity coverage exists in upper tiers even though their value/DOB columns are absent. Sources: `lineup_coverage.csv`, `feature_associations.csv`, `mover_covariates.csv`, `extra_features.sql`.

### A serious market-value vintage failure

Among the **575 players with positive values**, **every player has exactly one distinct positive value across all 30,579 valued appearances** spanning the extract. There is no valuation-as-of field. Treat these as unversioned/backfilled values, not prices known at each historical match. Null and zero valuations are not evidence of low squad quality.

`value_vintage_audit.csv` records the failure. **No admissible pre-match residual effect can be estimated from these values.** After the provenance audit, valuation features were removed from all published residual associations and regressions, not just the combined blocks. The raw fields remain in the audit extract/panel for provenance inspection only. Merely lagging the match does not cure valuation look-ahead.

### Effect on the residual

| Pre-match proxy | Raw mover correlation with `e` | Forward mover ΔMAE | Non-mover ΔMAE |
|---|---:|---:|---:|
| Relative mean age | +0.033 | +0.075 | +0.008 |
| Previous-season continuity | −0.021 | +0.067 | −0.001 |
| Higher-tier arrivals | −0.185 | −0.039 | −0.007 |
| Consecutive-XI continuity | −0.233 | −0.002 | −0.254 |

Higher-tier recruits have the expected suggestive sign: greater relative recruiting from above goes with QS being lower than the market. The raw slope is **−0.827 pp per ten-percentage-point increase** in the XI share, but its very small forward gain warns against turning that slope into a confident prior. Simple churn looks useful for ordinary clubs, not for the small mover panel.

**Verdict:** current aggregate squad proxies do not account for much validated mover MAE. This does not falsify **actual announced-XI player strength**, injury/suspension news or quality-weighted departures: none is captured by an unweighted previous XI. Improve that data before promising a lineup fix.

A bounded summer-cup audit (`cups.sql`) found only **11 July events, all 2024, and one 32-row player-lineup extract** in these DB tables. This is insufficient to reconstruct movers' summer squads systematically. Do not silently treat missing cup lineups as squad continuity. External dated transfer/registration and lineup-announcement archives would be needed for game one.

## 3. Manager changes

### Why the named SofaScore columns cannot be used as history

Inverness home fixtures from **August 2021 through April 2026** all name **Scott Kellacher** in `sofascore.matches.manager_home`, including August 2024 before his appointment. Similar near-static club-level names appear elsewhere. The official club announcement dated **23 October 2024** says Duncan Ferguson leaves and Kellacher takes over [R7]. A zero-change test using the SofaScore columns would be invalid.

`audit.sql` instead extracts BBC archival match-page manager strings. They recover Ferguson→Kellacher at the correct historical boundary (`inverness_manager_timeline.csv`). The BBC rows were retrieved in July 2026: historical page content supports event-time reconstruction, but we do not claim a contemporaneously captured database vintage. Identity normalization and announcement-time validation remain production requirements.

### Test and result

Feature: whether a different non-null manager was observed in the club's **previous three league fixtures**, plus the opponent equivalent. This deliberately misses the first fixture under a new manager unless a dated announcement is separately supplied.

- Relegated first-20 rows with a recent observed change: **12**, MAE **7.82**, signed club-mean error **+7.26**; other rows: **82**, MAE **5.67**.
- Promoted change rows: **4**, MAE **6.09**; non-movers: **46**, MAE **4.99**.
- Despite that striking relegated subset, forward ΔMAE is **+0.061 movers / +0.001 non-movers**. Raw relative-manager correlation is only **+0.130 / +0.039**.

Sources: `manager_vintage_audit.csv`, `stratified_errors.csv`, `candidate_comparison.csv`. Subset club weights differ, and changes are endogenous to bad form: this is not a causal manager effect. The systematic-review literature finds mixed effects and substantial counterfactual/reversion problems [R4]. Prefer manager/news flags as possible **state-uncertainty shocks**, not an automatic positive “new-manager bounce”.

## 4. Last-season underlying performance, results and recent form

### What is actually measurable

Historical goals, points and BBC shots/SoT are available. In lower-tier target seasons, native SofaScore xG coverage is **zero**. BBC total-shot coverage is **360/360 club-fixtures** in both leagues in 23/24; **358/360 L1 and 360/360 L2** in 24/25; **350/350 each** in the extracted 25/26 schedule. These counts cover the extracted fixtures, not an assertion of complete 25/26 league scheduling.

BBC `basic` rows are used once per stat; `attack` duplicates are not added again, and `filled=true` rows are excluded. This tests the **shot/SoT information underlying a pxG proxy**, not a pre-existing learned pxG series. We do not manufacture or fit a full-panel pxG conversion that could leak future calibration. Sources: Q3–Q4, `history_coverage.csv`, `features.sql`.

### Last season is weak; recent relative performance is stronger

| Candidate block added to context | Mover ΔMAE | Non-mover ΔMAE | Mover forward squared-error R² |
|---|---:|---:|---:|
| Last-season GD + points/game | +0.076 | −0.041 | 0.002 |
| Last-season shots + SoT differential | +0.070 | −0.126 | 0.006 |
| Previous season's final-six GD | +0.139 | −0.021 | −0.010 |
| Most recent six-match relative GD | **−0.592** | **−0.677** | 0.193 |
| Most recent six-match relative shots + SoT | **−0.537** | **−0.596** | 0.208 |
| Recent relative GD + shots + SoT (`form_pair`) | **−0.808** | **−0.917** | **0.250** |

Context alone changes mover MAE **+0.056**, non-mover **−0.018**. Thus the three-feature form block improves on context by **−0.864 / −0.899 pp**, not merely on an artificially weak intercept-only comparison. Combining all admissible non-market blocks produces **−0.895** for movers, only another **0.088 pp** beyond the small form block. Do not begin with a large feature kitchen sink.

Raw mover correlations are **−0.364 relative GD**, **−0.346 relative SoT**, **−0.295 relative shots**, compared with **+0.048 previous-season relative GD** and **+0.013 end-season GD**. Improving recent relative performance accompanies market probabilities above QS. This could reflect **different assimilation speed or functional form**, not information intrinsically absent from historical goal inputs. Raw league GD is not tier-adjusted, which particularly matters near the boundary.

### Forward effect, uncertainty and robustness

| Held-out group | Club-seasons | Original MAE | Corrected diagnostic MAE | ΔMAE [cluster interval] |
|---|---:|---:|---:|---:|
| Relegated into L1 | 3 | 5.849 | 5.037 | −0.812 [−1.773,+0.238] |
| Promoted into L1 | 4 | 4.949 | 4.144 | −0.804 [−1.315,−0.294] |
| Both directions, equal club-season weight | 7 | 5.334 | 4.527 | −0.808 [−1.291,−0.309] |
| Non-mover | 26 | 4.830 | 3.913 | −0.917 [−1.257,−0.556] |

These are the **24/25–25/26** denominators, not the three-season baseline in §1. The relegated interval crosses zero. 25/26 has just **one** relegated cluster, Hamilton; no one-season bootstrap point is convincing by itself.

- Pooled-mover gain is **−0.548 in 24/25**, **−1.154 in 25/26**.
- Train only on non-movers: **−0.836** mover MAE. Train only on 23/24 for both test seasons: **−0.745**.
- Ridge penalties 1/10/100: **−0.810 / −0.808 / −0.717**.
- Excluding either held-out mover club-season leaves pooled improvement between **−0.982 and −0.647**. This is deletion influence, not seven new independent tests.
- Same features, arm-specific training residual and QS probability: control **−0.808**, A2 **−1.243**, entrant **−0.856**, summer **−0.881**, weekly **−0.861**, pooled A2 **−1.236**. These are improvements to each arm's own residuals, **not six new fitted QS arm results**.
- Delay all form inputs by 7/14/28 days: **−0.583 / −0.351 / −0.156**. Freshness matters; weekly refitting alone does not erase the diagnostic signal.

Sources: `chronological_diagnostics.csv`, `robustness_results.csv`, `leave_one_cluster_sensitivity.csv`.

### The first five games remain unsolved

| Direction | Games | Original MAE | Form-corrected MAE | ΔMAE |
|---|---|---:|---:|---:|
| Relegated | 1–5 | 8.629 | 8.654 | **+0.025** |
| Relegated | 6–10 | 3.690 | 2.734 | −0.956 |
| Relegated | 11–20 | 5.431 | 4.327 | −1.104 |
| Promoted | 1–5 | 5.402 | 5.725 | **+0.323** |
| Promoted | 6–10 | 4.243 | 2.956 | −1.286 |
| Promoted | 11–20 | 5.022 | 3.808 | −1.214 |

`form_gain_by_game_bin.csv`; equal-club weighting is repeated **within** each bin, so these deltas cannot simply be averaged to recover the overall delta. Previous-tier form contaminates the first rolling window, and current-tier information is scarce. A first-five exclusion or ramp is a plausible **new hypothesis**, not a tested winning choice selected from this table. A learned, squad-conditioned boundary transition remains complementary to a fast in-season signal.

## 5. Club resources, relegation distress, venue and employment model

No retrieved column identifies wage budget, full-/part-time status, or actual parachute receipts. Capacity extracted from match JSON is entirely missing; BBC attendance is entirely missing. A venue name or former Championship membership is **not** a defensible employment-status classifier.

A legal but crude resource proxy is the club's share of prior extracted league appearances in the top two tiers, contrasted with its opponent. It has mover raw correlation **−0.153** but forward ΔMAE only **−0.035**, versus **−0.090** for non-movers. It measures league history, not cash. Longer pre-2021 history would change this proxy. Source: `features.sql`, `feature_associations.csv`, `candidate_comparison.csv`.

**Distress is not the same as relegation.** Two dated, independently verified examples in the actual mover panel:

- Inverness administration/coaching reductions, October 2024: the official 23 October coaching announcement [R7] and contemporary reporting [R8] describe a real organisational shock. Its priced first-20 residual moves from **−1.744 pp** before 24 October (9 rows) to **+0.974** after (9 rows), while MAE **falls 5.516→4.277**. Thus a plausible news shock does **not** produce the simple narrative “administration causes all large misses.” Opponents, simultaneous squad/manager changes and small N prevent attribution (`inverness_resource_event.csv`).
- Hamilton's **17 April 2025** SPFL statement documents remuneration/tax defaults, stadium-information problems and a 15-point sanction [R9], all known before its 25/26 L1 season. Hamilton 25/26 is the highest-MAE relegated club here (**9.11 pp**), but with one distressed entrant we cannot estimate a transferable distress coefficient (`club_trajectories.csv`).

The fetched SPFL Articles define a “Parachute Payment” with respect to **Premiership→Championship** relegation [R10]. Do not import English parachute-payment assumptions into Championship→L1 without actual dated Scottish payment records. We do not assert that these documents prove the absence of every other lower-tier distribution.

**Verdict:** maintain dated news/financial/registration regimes and consider conditional uncertainty, not a universal relegation-resource bonus. Effect size is **unidentified**, not zero.

## 6. Opponent-specific effects, home/away and schedule

A club's win probability is a function of **both teams**. Attributing every mover-fixture residual to the mover is a substantial modelling trap.

- Relegated MAE: **6.13 away vs 5.60 home**; signed errors **+1.80 vs −1.37**.
- Promoted MAE: **5.19 away vs 4.70 home**; signed errors **+2.47 vs +0.37**.
- Non-mover MAE: **4.64 away vs 4.81 home**. There is no shared “away fixtures always worse” pattern.
- Against opponents with previous-season GD >0.3/game, relegated MAE is **6.48** (23 rows), versus **5.25** against middle opponents (51). Promoted error is instead **5.97** against weak opponents (GD <−0.3, 30 rows), compared with **4.80** against strong ones (17).

These opponent categories are pre-match descriptive proxies, not actual same-tier strength ratings; previous-tier schedule strength is a confound. Current QS probability strata are also exported, using **QS**, never the target price, to identify favourites/outsiders. Some categories are tiny: only **two** promoted-favourite observations exceed 55% QS win probability.

More persuasive than stratified means is the chronological contrast:

| Diagnostic | Mover ΔMAE |
|---|---:|
| Opponent previous/recent GD block | −0.280 |
| Own recent GD/shots/SoT only | −0.236 |
| Opponent recent GD/shots/SoT only | −0.313 |
| **Relative recent GD/shots/SoT** | **−0.808** |
| Both sides separately, six covariates | −0.804 |

This supports a symmetric quality/attack–defence mechanism rather than a special mover-only intercept. A capped relative gap since the last **league** game adds no mover value (**+0.035 pp**, non-movers −0.027). It is not a complete rest/congestion measure: cups, travel and training schedules are missing.

Sources: `stratified_errors.csv`, `robustness_results.csv`, `extra_features.sql`. Home/away and prior-opponent controls are already in or tested beside the baseline, not inferred from the target match price.

## 7. Market dynamics: lag, anchoring or noise?

### What can be tested without using the target price

Historical accepted rates come from `current_development/market_model/results/QS/rates_sco_*.csv`, restricted to historical seasons. We use prior-five-fixture rotated rate summaries, `D=0.5 log(λ_h/λ_a)` with club-side sign, and `S=0.5 log(λ_h λ_a)`, plus latest-minus-prior-five trend. They are **match-strength summaries, not opponent-adjusted latent club ratings**; raw D can move when schedule strength changes.

Separately define the previous-three-fixture residual innovation:

\[
z_{it}=\tfrac12\{\overline e_{i,t-3:t-1}-\overline e_{j,t-3:t-1}\},
\]

using only forecasts and market prices from fixtures already played. Each club's average uses those preceding league fixtures with a priced observation; missing residuals remain missing. No current price, rate, line movement or current-match inversion enters any feature.

- Raw correlation of `z` with the next residual: **0.262 movers vs 0.432 non-movers**; univariate slopes **0.462 vs 0.740**.
- Learned lag-residual block: **−0.169 pp movers / −0.546 non-movers**.
- Fixed 25% / 50% / 100% innovation corrections: mover **−0.119 / −0.111 / +0.094**. Chasing the entire previous gap makes movers worse.
- Historical rate-level/trend/style block: **+0.090 movers / −0.138 non-movers**. This is not evidence that properly opponent-adjusted market states are useless; this deliberately simple proxy does not identify them.
- Combining admissible non-market features with historical market/residual features gives **−1.065 mover / −1.062 non-mover pp**, versus **−0.895 / −0.928** without them. The extra opportunity is modest and not independently confirmed.

### There is no universal cold-start trajectory

Selected first-5 / games-6–10 / games-11–20 signed errors, all three seasons:

| Club-season | Direction | 1–5 | 6–10 | 11–20 |
|---|---|---:|---:|---:|
| Cove 23/24 | Relegated | +10.72 | +1.06 | −3.22 |
| Hamilton 23/24 | Relegated | −12.07 | −2.17 | −2.97 |
| Inverness 24/25 | Relegated | +0.46 | −3.51 | +0.97 |
| Hamilton 25/26 | Relegated | +4.53 | +3.74 | +3.89 |
| Stenhousemuir 24/25 | Promoted | +4.92 | +1.74 | −1.27 |
| Dumbarton 24/25 | Promoted | −0.16 | +1.08 | +4.86 |
| East Fife 25/26 | Promoted | +2.04 | −1.54 | +2.52 |

All 11 mover trajectories, priced counts, MAE and signed-bin means are in `club_trajectories.csv`; they are paired model-minus-market trajectories, **not schedule-neutral ratings**. Some entrants exhibit decaying misrating, some reverse sign, some remain persistently wrong, and some deteriorate. The evidence supports **a mixture of limited persistent state error and match-varying/new information**, not a claim that all remaining error is noise or all movers are anchored to their old league.

### Market-reference and numerical audits

`probability_mass_audit.csv` finds two incomplete 1X2 market triplets, both in training season 23/24:

- **11395580**, Cove–Edinburgh, 23 December 2023: only home/away exported; market probabilities sum to one over those two outcomes.
- **11395562**, Alloa–Falkirk, 27 January 2024: only draw/away exported; again a two-outcome market normalization.

These cannot be silently treated as complete unconditional 1X2 books. The frozen panel is preserved, but re-estimating the form diagnostic after excluding them from training still gives **−0.816 pp**, close to −0.808. Thus they are a target-quality issue, not the main signal.

Among 896 complete triplets, 49 have model mass below one by more than 1e−5; minimum sum **0.989724**, consistent with lost finite-grid tail mass. That mechanism should be confirmed in the scorer before calling it a bug. Our proper-score sanity check explicitly normalizes model probabilities; the canonical MAE reproduction remains raw and unchanged.

## 8. Does better tracking actually make the forecast better?

**Not established.** The residual regressions are not coherent multi-market generative forecasts. To do a limited check without a QS fit, `outcome_sanity.py` normalizes each complete QS 1X2 triplet, keeps the draw probability fixed, and shifts non-draw mass between home and away using half the difference between their predicted residual corrections. It uses no current market price as an input.

For the small form block:

- Coherent-projection MAE changes: **−0.754 relegated**, **−0.690 promoted**, **−0.864 non-movers**; hence the tracking gain is not solely an artefact of impossible independent win probabilities.
- Held-out complete-triplet **multiclass 1X2** outcome ΔLL is **+0.0030 [−0.0038,+0.0095]**, **595 fixtures**.
- On fixtures involving a first-20 L1 mover: **+0.0076 [−0.0088,+0.0233]**, **112 fixtures**.
- Larger combined corrections also have slightly positive outcome-LL point changes, with intervals spanning zero (`outcome_sanity.csv`).

These intervals are paired fixture bootstraps and do not resolve all temporal/team dependence. This is **not the original all-market Bernoulli scoring convention**, and no totals/BTTS score tensor has been modified. It neither passes nor formally evaluates frozen Gate 1. It does demonstrate why **market MAE alone must not determine promotion**: copying more of market behaviour can worsen outcome forecasts in a finite sample. No ROI claim or betting recommendation follows from this EDA.

## 9. Literature: what transfers, and what does not

The literature supports mechanisms, not numerical Scottish QS-MAE promises. Search-result snippets were treated as leads; sources below were fetched or read from the library. No browser tool was exposed in this session; web search/fetch and the local research-library helpers were used.

1. **Player-level information can explain match-specific deviations.** Holmes & McHale's player-rating model adjusts ratings across leagues and models lineup/opponent interactions. Their closing discussion explicitly identifies changing lineups and short-term player form as mechanisms behind team-strength dynamics, including injury-driven trader adjustments. This is a stronger mechanism for per-match residuals than static squad wealth. It requires timestamped actual/expected lineups and ratings trained only on earlier games. **Library:** `holmes2023-forecasting-football-match-results-player`, **PDF p.10**; focused page saved in `library_holmes_p10.txt`. Do not transfer its betting ROI to Scottish lower leagues.
2. **Values can carry information, but historical vintage is essential.** Peeters (2018), *Testing the Wisdom of Crowds in the field: Transfermarkt valuations and international soccer results*, finds valuation-based forecasts outperform FIFA/Elo benchmarks. We verified the abstract at the Erasmus repository [R1]; the publisher page hit a bot challenge. This supports collecting dated valuations, **not using backfilled SofaScore values** or asserting an incremental gain over our market-informed QS.
3. **Squad-value covariates and promoted priors are feasible, not automatically superior.** Ribeiro et al. (2025), *A Bayesian approach to predict performance in football: a case study*, §3.2–3.3, uses prior-season attack/defence information, diffuse priors for promoted teams and Transfermarkt squad values [R2]. It is a Brazilian application, with a different target and evaluation. Ridall's state-space treatment distinguishes within/between-season forgetting and promoted-team hyperparameters, and notes weak dependence on previous lower-tier history in its EPL setting: **`ridall2024-bayesian-state-space-models-modelling`, PDF p.11**, saved locally. That is a reason to learn transfer strength and uncertainty, not blindly reuse old-tier GD.
4. **Underlying performance can be more predictive than realised goals.** Mead, O'Hare & McMenemy (2023), *Expected goals in football: Improving model performance and demonstrating value*, reports xG predicting future success better than traditional statistics [R3]. Our data show a recent-shot signal, but **we did not test native lower-tier xG**, and the paper does not quantify Scottish model-market residual improvement.
5. **Manager-change evidence is mixed and endogenous.** Sousa et al.'s systematic review retained **24 studies**; some show small short-term gains, others little/no effect, and matched counterfactual examples show improvement without changing coach [R4]. Do not interpret our high-error changed-manager subset as causal or assign a universal bounce.
6. **Continuity can matter at season scale, without predicting residuals at match scale.** *Does Stability Foster Team Performance? A European Football (Soccer) Inquiry*, abstract and discussion, associates stability with performance after some controls [R5]. It is observational and not a point-in-time residual test. Our null-ish mover forward result is compatible with the market already pricing broad continuity.
7. **The market is an informative reference, not mathematical truth or a profit objective.** Whelan's *Risk Aversion and Favourite-Longshot Bias in a Competitive Fixed-Odds Betting Market* explains how bookmaker risk aversion and bettor disagreement can produce biased implied prices [R6]. Hubáček & Šír distinguish predictive accuracy, correlation with the market and trading profit: **`hubacek2023-beating-market-bad-predictive-model`, PDF p.1**, saved locally. This supports retaining proper-score and economic safeguards, not assuming lower market MAE guarantees higher ROI.

Library search: promoted teams; player ratings/lineups; Transfermarkt/market values; managerial changes; market efficiency; shots/Elo. The catalogue and converted text were searched. This was not a full-text search of every unconverted book. A CIES stability landing page yielded navigation only, so it is **not** used as evidence.

### Verified web references

- **R1:** Peeters, 2018, DOI `10.1016/j.ijforecast.2017.08.002`; https://repub.eur.nl/pub/102113 ; bibliographic abstract also https://econpapers.repec.org/RePEc:eee:intfor:v:34:y:2018:i:1:p:17-29 . Abstract verified, publisher full text unavailable.
- **R2:** Ribeiro et al., 2025, §§3.2–3.3: https://pmc.ncbi.nlm.nih.gov/articles/PMC11949986/ .
- **R3:** Mead et al., 2023, DOI `10.1371/journal.pone.0282295`; repository abstract/citation: https://dspace.stir.ac.uk/handle/1893/35242 .
- **R4:** Sousa et al., *Effects of changing the head coach on soccer team's performance: A systematic review*, abstract and comparative-study table: https://pmc.ncbi.nlm.nih.gov/articles/PMC10955743/ .
- **R5:** *Does Stability Foster Team Performance?*, draft, PDF p.2 and discussion: https://www.professormarkvanvugt.com/images/files/DoesStabilityFosterTeamPerformance.pdf . Publication metadata not established here; no year or causal effect is inferred.
- **R6:** Whelan, September 2023 manuscript, PDF pp.1–2: https://www.karlwhelan.com/Papers/EconomicaFinal.pdf .
- **R7:** ICTFC, **23 October 2024**, official coaching announcement: https://ictfc.com/club-update-coaching-staff/ .
- **R8:** Contemporary Inverness administration/coaching/roster reporting: https://www.skysports.com/football/news/13238256/inverness-caledonian-thistle-enter-administration-and-face-battle-for-survival .
- **R9:** SPFL, **17 April 2025**, Hamilton disciplinary statement: https://spfl.co.uk/news/spfl-statement-hamilton-academical-fc .
- **R10:** SPFL Articles, 15 April 2019, definitions of Parachute Payment and referenced articles: https://spfl.co.uk/admin/filemanager/images/shares/pdfs/SPFL%20Articles%20of%20Association%20of%2015-Apr-19%20(CURRENT).pdf . Historical document, not assumed to establish all later financial arrangements.

## 10. Ranked, concrete QS research arms

The numerical ranges below distinguish **observed residual-probe gains** from **planning targets for an unrun QS arm**. They are not claimed posterior forecasts of improvement. Do not stack all changes before identifying the first mechanism.

### 1. Fast quality/form component for both clubs — highest empirical priority

- **Mechanism:** a slow team state misses shorter-lived relative quality changes; update both mover and opponent rather than the mover alone.
- **Data:** completed historical goals, BBC shots/SoT, dates, opponent identities and current-tier exposure. Start with the small three-feature block; no backfilled values.
- **Model change:** preserve the existing QS score tensor and augment quality, for example
  \[
  q_{i,t}=q^{slow}_{i,t}+f_{i,t},\quad
  f_{i,t}=\rho f_{i,t-1}+\gamma^\top\xi_{i,t-1}+\epsilon_{i,t},
  \]
  with `ξ` a pre-match goal/shot innovation after opponent, venue and tier adjustment. Alternatively use a regularised pre-match `βᵀx` quality term for the first attribution arm. Enter `f_h−f_a` into the QS quality contrast; price all markets through the existing score kernel.
- **Expected size:** observed three-feature residual-probe gain **0.81 pp pooled movers**, about **1 pp in games 6–20**, none in games 1–5. A **0.3–0.8 pp** first-20 QS-MAE reduction is a reasonable *test target*, not an established expectation; zero or an outcome-score failure remains plausible.
- **Leakage/evidence risk:** rolling histories must stop at the declared forecast information time; shots from the target game are prohibited. If goals/shots already enter the joint likelihood, do not multiply their derived summaries as independent extra evidence. Use a sequential conditional transition or an explicitly specified joint observation system. Old-tier form requires a gauge/tier transform.
- **QS test:** one frozen fast-component arm versus control, then a shots ablation and the weekly control. Report both directions, first5/6–10/11–20, non-movers, coherent all-market LL, allocation/convergence checks and timing. Do not retrospectively choose a five-game gate from this EDA and call it confirmed.

### 2. Event-time market-innovation assimilation — modest second priority

- **Mechanism:** incorporate information revealed by **earlier fixtures'** market observations without waiting for a slow state/fold update.
- **Data:** point-in-time archived pre-match prices on completed fixtures, their QS predictions, exact market timestamps and fold cutoffs.
- **Model change:** a regularised sequential update
  \[
  q^+_{i,t}=q^-_{i,t}+K_t\{D^{mkt}_{t}-E[D_t\mid\mathcal F_{t^-}]\},
  \]
  with opposite-side identification, observation noise and a constrained gain. Treat this as assimilation of each historical market observation **once**, not a second market-derived prior on top of the same likelihood.
- **Expected size:** isolated diagnostic **0.17 pp mover** improvement, about **0.55 non-mover**; plan around **0–0.3 pp** mover MAE, not a multi-pp cure. A full previous-gap correction worsens movers.
- **Leakage risk:** never current-match prices; never duplicate an observation already assimilated. Exact online filtration matters more than nominal refit frequency.
- **QS test:** fixed-gain/learned-gain single-factor arm versus weekly and control with identical admissible historical observations; retain outcome-score gate. Do not infer a gain from schedule-confounded raw D averages alone.

### 3. Timestamped lineup-strength innovations and quality-weighted recruitment

- **Mechanism:** actual absences, replacements and cross-tier recruits change match strength after broad squad expectations have been set.
- **Data:** announced/expected XI with availability timestamps; previous-season player identities and minutes; player ratings learned across tiers on history only; dated transfers/registrations. A dated valuation series is optional, not supplied by the current field.
- **Model change:** add a lineup **deviation from the team's expected lineup**, not another unrestricted permanent quality offset:
  \[
  q^{match}_{i,t}=q_{i,t}+\beta\sum_p(w^{ann}_{ipt}-w^{expected}_{ipt})r_{p,t^-}.
  \]
  Hierarchically shrink players with little prior data and account for tier strength in `r`. Squad turnover can condition boundary variance as well as its mean.
- **Expected size:** measured coarse proxies provide **approximately zero mover gain** (higher-tier share 0.04 pp; consecutive continuity essentially zero). Actual-XI upside is **unquantified**; require a prospective improvement target such as ≥0.25 pp before investing in a full grid, not a fabricated 1-pp forecast.
- **Leakage risk:** target-match minutes, ratings, injury outcomes, retrospective team memberships and unversioned values. Historical XI rows alone do not prove the XI was available at the forecast instant.
- **QS test:** historical-expected-XI arm first; announced-XI arm only on a timestamp-audited subset. Keep a matched non-mover comparison and missing-coverage fallback, not zero-valued players.

### 4. Learned, squad-conditioned boundary transition — retain for the unsolved opening

- **Mechanism:** learn what survives promotion/relegation, especially when recruitment and organisational regimes differ; do not use one permanent tier translation for everyone.
- **Data:** old posterior state, identified tier gauge, dated roster/financial changes, old-tier opponent-adjusted underlying performance, multiple independent transition cohorts.
- **Model change:**
  \[
  q_{i,new,0}\sim N\{a_{d}+b_d q_{i,old,end}+c^\top z_{i,summer},\ \sigma_d^2\exp(\eta^\top z_{i,summer})\},
  \]
  with partial pooling over direction `d`; use carry/mean/variance with explicit identifiability. Couple this to a separate in-season component rather than forcing an enduring offset.
- **Expected size:** not newly validated here. Even an oracle constant offset buys only **0.42 relegated / 0.31 promoted pp** over first20, but an early-only correction can exceed that constant bound. Last-season GD/shots/end-form proxies failed forward; require useful new entry information. Plan conservatively, with first-five performance as the explicit attribution target.
- **Leakage risk:** estimating tier/boundary priors with target-season outcomes/prices, or reusing historical market summaries already in the old posterior as independent observations.
- **QS test:** one new boundary arm with cross-cohort learning, frozen early/late diagnostics and both existing gates. It is complementary to rank 1, not evidence that combining two unvalidated arms will pass.

### 5. Dated organisational/manager regime shocks — targeted, not universal

- **Mechanism:** administration, unpaid wages, major releases or a coaching regime can alter both state mean and uncertainty.
- **Data:** announcement timestamps, effective dates, actual affected squad; independently curated employment status/budget if available. BBC historical manager transitions are a starting point; SofaScore current-manager backfill is not.
- **Model change:** sparse, strongly pooled shock indicators in boundary/within-season innovation variance, e.g. `Var(Δq)=σ² exp(η_news I_news+η_manager I_manager)`; only add a directional mean effect if repeated cohorts support it.
- **Expected size:** **unknown** for financial shocks; generic manager flag **worsens** mover MAE by 0.06 pp here. No supported average gain can be assigned. The purpose is exceptional-club risk, not a guaranteed mean-MAE reduction.
- **Leakage risk:** using eventual administration or end-season financial reports before public announcement; confusing the sanction date, manager departure and player releases; coding a distressed club after seeing its residuals.
- **QS test:** predeclared event ledger and cross-event/club validation; compare matched poor-form controls, retain rare-event uncertainty. Do not estimate a Hamilton-only coefficient and call it a general relegation model.

### Prerequisites, not additional winning arms

1. Fix or replace historical value/manager vintages; keep a field-level availability ledger.
2. Audit complete market-triplet construction, price instant, vig removal and score-grid tail mass. Preserve the existing reference/gates while adding clearly labelled clean-market sensitivity.
3. Record actual fold information cutoffs in observation exports. The match-time versus fold-time gap is a first-class experimental factor.
4. Do not use current-match odds as a “feature”, line-movement shortcut or calibration anchor to win this target metric.

## 11. Reproduction, artifacts and limitations

### Entry points

From the repository root:

```bash
# Offline, using the saved bounded extract (no DB or credentials):
PYTHON=/tmp/qs-s1-venv/bin/python \
  bash experiments/scotland/11_qs_transition/research/mae_deep/reproduce.sh

# Optional NEW read-only source snapshot; this changes extraction vintage:
bash experiments/scotland/11_qs_transition/research/mae_deep/run_sql.sh
```

The observed environment was Python **3.13.12**, NumPy **2.5.4**, DuckDB Python/CLI **1.5.6**. `requirements.txt` records Python dependencies; another interpreter with those packages can replace `PYTHON`. The temporary venv path was discovered in this session, not assumed portable.

`run_sql.sh` loads the existing ignored environment file without printing it, attaches Postgres **READ_ONLY**, and redacts credential-bearing URL errors. DB queries are bounded to named tables, historical date ranges and Scottish tournaments. Local temporary DuckDB tables and exported files are not writes to Postgres. No `mcmc_experiments` connection was used.

Raw bounded extracts are compressed in **`extract_snapshot.tar.gz`**, not left as large loose CSV dumps. Offline reproduction restores them, creates outputs, verifies chronology/denominators, then removes only those named temporary raw CSVs. `input_manifest.csv` hashes the six observation files, four Scottish rate files, canonical transition errors and the extract archive. `verify.py` checks exact reproduction, historical feature dates (including delayed cutoffs), no future target seasons, all-arm direction, primary diagnostic values and wealth quarantine.

### Main artifacts

| Artifact | Purpose |
|---|---|
| `queries.sql`, `audit.sql`, `cups.sql` | Bounded source extraction, manager/provenance and summer-cup coverage |
| `features.sql`, `extra_features.sql`, `sensitivity.sql` | Lagged covariates, churn/end-form/rest, delayed information and audits |
| `panel.csv`, `residual_predictions.csv` | Joined analysis panel and forward predictions; raw current outcomes are not predictor columns |
| `baseline_summary.csv`, `validation.json`, `verification.csv` | Denominators, exact primary-metric reproduction and checks |
| `candidate_comparison.csv`, `chronological_diagnostics.csv` | Every candidate's forward effect and non-mover comparison |
| `feature_associations.csv`, `diagnostic_coefficients.csv` | Raw associations, feature availability and train-only coefficients |
| `robustness_results.csv`, `leave_one_cluster_sensitivity.csv` | Six arms, delay/ridge/training-scope and influence checks |
| `club_trajectories.csv`, `form_gain_by_game_bin.csv`, `stratified_errors.csv` | Early/late paths, opponent/home/away and subgroup diagnostics |
| `error_decomposition.csv`, `oracle_constant_bound.csv` | Why club offsets cannot remove most per-match MAE |
| `value_vintage_audit.csv`, `manager_vintage_audit.csv`, `probability_mass_audit.csv` | Data-quality findings, including inadmissible proxies |
| `outcome_sanity.csv`, `coherent_1x2_mae.csv` | Proper-score caution after coherent 1X2-only projection |
| `analyze.py`, `robustness.py`, `outcome_sanity.py`, `final_tables.sql` | Reproducible diagnostic calculations, no QS model fits |

### What remains unresolved

- No current-lineup injury/availability signal with a verified historical announcement time; no historical valuations, budgets or full-/part-time classifications.
- No native lower-league xG; tested shots/SoT rather than an unverified reconstructed pxG series.
- Only 11 priced primary transition club-seasons, seven in chronological assessment, and one priced entrant. More rows do not create more independent transitions.
- No exact archived control-fold information-time match; all-match historical eligibility is tested, with delay/weekly sensitivities instead.
- No new coherent QS score tensor, all-market outcome gate, convergence checks or portfolio simulation. The strongest tracking signal **still needs a real arm and the unchanged gates**.
- No use of 26/27 outcomes for estimation, validation or model selection. No recommendation to unblind that season merely to choose between these exploratory feature blocks.

**Bottom line:** the observed opportunity is mainly **faster, bilateral recent-performance tracking after entry**, not a larger permanent mover correction. Entry remains a separate information problem. The data needed to test richer squad/resource stories must first become genuinely point-in-time.
