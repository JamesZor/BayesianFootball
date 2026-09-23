# Scottish club pedigree, operational status and tier priors

**Research date / extraction cutoff:** 23 September 2026<br>
**Tracking:** [TODO 027](../../../todos/027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md)
**Decision:** retain the production model; develop a point-in-time pedigree-prior ablation only after the evidence gaps below are resolved. **No MCMC or ADVI grids were run.**

## Abstract and executive conclusions

This investigation audits the motivating live slate, extracts all four Scottish league tiers from `betdb`, constructs a source-referenced operational-status panel, measures available outcome/market/proxy-xG contrasts, and specifies informative-prior and covariate alternatives. It does **not** establish that full-time status caused the model's favourite compression.

The principal findings are:

1. **The upper leagues are present in SQL.** The available normal-time outcome panel contains **4,068 fixtures** across tournaments 54–57 from January 2021 to the cutoff. The production model's lower-league training scope must not be confused with database availability. However, the queried archive supplies **no cross-tier cup bridge observations**, and its league histories are incomplete. Adjacent-tier talent steps are therefore **not estimated** here.
2. **The proposed mathematical ceiling is false.** Zero-centred Gaussian team effects and sum-to-zero identification do not bound team contrasts or impose a 54% win-probability ceiling. Shrinkage can compress predictions without imposing a hard cap.
3. **The motivating status labels are unsafe.** Cove announced a hybrid/full-time transition for 2023/24. Queen of the South is explicitly reported to have remained full-time after relegation in 2022. Neither should simply be designated a part-time opponent in an explanation of the September 2026 losses; their exact 26/27 status still requires contemporary evidence.
4. **The live ledger materially revises the incident narrative.** Cove was **home** to Ross County. The quoted **£20.35** combines two underdog bets **and two draw bets**; their filled loss was **£16.29**. The entire m12-labelled slate settled at **−£5.09**, equivalent to **−1.018%** of its stated £500 opening bankroll—not a demonstrated historical maximum drawdown.
5. **The status census is incomplete, transparently so.** The delivered CSV covers **252 club-seasons / 45 distinct clubs**, with **21 verified season labels, 8 inferred labels and 223 Unknowns**. This is a usable evidence register, **not the definitive six-season classification requested in the mandate**. The strict League One FT–PT sample contains only **8 matches**, too few and too selected to estimate a general operational-status premium.
6. **A mock prior/covariate implementation passes deterministic gradient checks**, including a perturbed compiled tape. It allocates **2,432 bytes per warmed gradient**; allocation-free execution and production Turing integration have **not** been demonstrated.

The deliverable is a reproducible, qualified EDA and implementation design. Missing status evidence, cup bridges, complete transition histories and immutable live-m12 provenance remain open dependencies, not zero effects or successful acceptance gates.

---

## 1. Research design and evidence hierarchy

The [work package](WORK_PACKAGE_PROMPT.md) supplies hypotheses and incident claims, not verified observations. The [research protocol](RESEARCH_PROTOCOL.md) defines estimands and claim boundaries. Evidence is ordered as follows:

- **Operational facts:** saved paper orders, fills and settlements in `betdb.paper_runbook`.
- **Historical football observations:** provider match identities, seasons, regulation scores, BBC commentary and archived odds.
- **Operational classifications:** fetched official statements or explicit credible reporting, with inferred continuity and missing evidence distinguished.
- **Model results:** one immutable saved experiment run and its saved draws/portfolio, not a mutable run name or a newly reconstructed fit.
- **Mechanisms:** mathematical proposals and synthetic checks, not measured predictive improvements.

All SQL activity is read-only. Neither paper ledgers nor experiment storage were altered. No production `src/` code was changed. Scripts and tabular outputs are suite-local; [the manifest](results/artifact_manifest.json) records CSV checksums and row counts.

### Quantities that must not be conflated

- Goals per match within a division are **not** that division's strength relative to another.
- `log(p_home/p_away)` is an outcome-probability contrast, **not** automatically a Poisson log-rate ratio; it also differs from `logit(p_home) − logit(p_away)`.
- A last archived pre-kickoff vector is **not necessarily the closing price or an executable quote**.
- A verified retrospective status label is **not necessarily information available before the fixture**.
- Probability edge, expected return, requested stake, filled risk, net P&L and drawdown have different units and meanings.
- Market probabilities are a benchmark, not observed latent talent or a causal counterfactual.

## 2. Reconciliation of the motivating live slate

Source: [SLATE_AUDIT.md](SLATE_AUDIT.md), [raw ledger extract](data/slate_2026-09-19.csv) and [derived audit](results/slate_2026-09-19.csv).

There were **13 m12-labelled orders** and **14 m05-labelled orders**, across separate £500 paper accounts. Quotes were stamped **13:34 UTC**, slates **13:35 UTC**, and kickoffs **14:00 UTC** on 19 September. These are approximately T−26 quotes, not demonstrated closing prices.

| Actual fixture / selection | Decimal odds | Saved model probability | Order-time market probability | Probability edge | Requested risk | Filled risk / loss |
|---|---:|---:|---:|---:|---:|---:|
| Cove Rangers–Ross County: Cove | 7.20 | 37.3981% | 13.5734% | +23.8247 pp | £11.06 | £7.00 |
| Cove Rangers–Ross County: draw | 4.90 | 26.4232% | 19.9446% | +6.4786 pp | £2.96 | £2.96 |
| Hamilton–Queen of the South: Queen | 9.00 | 25.3971% | 10.9750% | +14.4221 pp | £4.52 | £4.52 |
| Hamilton–Queen of the South: draw | 5.50 | 24.5207% | 17.9590% | +6.5617 pp | £1.81 | £1.81 |

All four selections lost. Their **requested** £20.35 is **46.75% of £43.53 requested slate risk**. The underdogs alone requested **£15.58**, with **£11.52** filled; all four filled for **£16.29**. Other slate settlements offset part of these losses, leaving **−£5.09** across the 13 m12 orders. £494.91 is the resulting one-slate balance equivalent, not a reconstructed full account ledger.

The work package's quoted odds and probabilities are not these saved order values. All extracted CLV/close fields are empty, so its closing-price claims cannot be verified from this ledger. A lost bet does not itself prove a bad probability forecast.

**Provenance limitation:** the m12 slate stores `run_name=m12_joint_hybrid_synergy`, fold 43, but **`model_run_id=NULL`**. The separately located historical m12 run in §6 must not be asserted to be the exact live posterior. The m05 UUID resolves, but cannot repair m12 lineage.

## 3. Stage 1 — operational-status panel

### Construction and coverage

[data/spfl_club_operational_status.csv](data/spfl_club_operational_status.csv) joins actual SQL season membership to a curated [evidence interval table](data/operational_status_evidence.csv) and [20-source register](data/status_sources.csv). It retains `Full-Time`, `Part-Time`, `Hybrid` and `Unknown` separately. The 42 memberships in each season represent **45 unique clubs** over the six-season union.

| Season | Verified | Inferred | Unknown | Total |
|---|---:|---:|---:|---:|
| 21/22 | 2 | 1 | 39 | 42 |
| 22/23 | 2 | 2 | 38 | 42 |
| 23/24 | 5 | 1 | 36 | 42 |
| 24/25 | 4 | 1 | 37 | 42 |
| 25/26 | 6 | 2 | 34 | 42 |
| 26/27 | 2 | 1 | 39 | 42 |
| **Total** | **21** | **8** | **223** | **252** |

All 42 current club/tier assignments agree with the independently fetched [official SPFL tables](https://spfl.co.uk/clubs/ross-county/fixtures), preserved in [the web membership snapshot](data/spfl_membership_web_2627.csv). That verifies membership, **not employment arrangements**.

### Material findings from fetched sources

| Club | Supported finding | Primary evidence / qualification |
|---|---|---|
| Cove Rangers | PT in 22/23; Hybrid transition in 23/24 | [Club, 18 May 2023](https://coverangersfc.com/2023/05/18/club-statement-full-time-football/); [implementation confirmation, 1 January 2024](https://coverangersfc.com/2024/01/01/chairmans-new-year-message-3/). Later status is not established by that transition announcement. |
| Queen of the South | FT continuously following 2022 relegation through the reporting period in 25/26 | [BBC financial review](https://www.bbc.co.uk/news/articles/cpvxpgpdw9ro) explicitly states retrospective continuity; [Murphy appointment, May 2024](https://www.bbc.com/sport/football/articles/c97z27mqjmro) corroborates FT football. Not advance knowledge of all earlier fixtures. |
| Hamilton | Remained FT for 23/24 | [BBC, 12 June 2023](https://www.bbc.com/sport/football/65881664). Prior-season continuity is separately inferred. |
| Inverness CT | Remained FT for 24/25 despite considering alternatives | [Inverness Courier, 21 May 2024](https://www.inverness-courier.co.uk/sport/ict-confirm-they-will-remain-full-time-351275/). Financial distress alone is not PT status. |
| Airdrieonians | Explicit hybrid model announced for 19/20 | [Club, 18 April 2019](https://www.airdriefc.com/1819-news/180419/hybrid-model-the-future-for-airdrieonians). Study-window continuity is **inferred**, not annually verified. |
| Arbroath | PT in 24/25 and 25/26 | [Flynn signing](https://arbroathfc.co.uk/transfer-deadline-day-action-at-arbroath/); [Herald 25/26 report](https://www.heraldscotland.com/sport/25659884.arbroath-became-premiership-promotion-contenders-playing-part-time/). The latter explicitly identifies Queen's Park, Ross County and St Johnstone as FT opponents. |
| Montrose | PT in 21/22 and 26/27 | [Petrie retrospective](https://www.thecourier.co.uk/fp/sport/football/2784072/stewart-petrie-on-5-years-at-montrose-still-striving-for-improvement-greatest-memory-and-why-jobs-elsewhere-have-never-been-considered/) and [2026 preview](https://www.theterrace.scot/news/26254406.stewart-petrie-montroses-big-summer-recruitment/). Intervening seasons are not automatically verified. |
| Alloa | Explicit PT in 21/22 | [Contemporary BBC cup report](https://www.bbc.co.uk/sport/football/60006014). Later loan-report testimony is retained with inferred season dating. |

Other register entries support selected seasons for Peterhead, Queen's Park, Falkirk and Stenhousemuir. The Kelty ownership quotation is retained but not assigned to a season until dated. Dunfermline and many other club-seasons remain unknown; this is unfinished verification, not evidence that they were PT.

**Source conflict:** the December 2021 Petrie interview loosely calls several opponents FT, including Alloa and Cove. That list conflicts with more direct club-specific evidence. We retain his first-person Montrose statement but do **not** classify every named opponent from it. This illustrates why league position, reputation, financial distress and generic “professional club” descriptions cannot substitute for operational evidence.

### Point-in-time limitations

The constructor assigns research season labels using inclusive evidence intervals at 1 July. It is not a production match-level status extractor and does not prove that status remained unchanged mid-season. Production needs separate **effective** and **known-at/publication** timestamps, revisions, explicit unknown handling and contract/training-regime scope. Full-time loanees at a PT club do not automatically make its operational model Hybrid.

### Resource proxies

The lineup valuation field has severe tier-dependent availability:

| Tier | Team-match rows with any value / extracted rows | Coverage |
|---|---:|---:|
| Premiership | 46 / 2,210 | 2.1% |
| Championship | 32 / 2,026 | 1.6% |
| League One | 1,931 / 1,934 | 99.8% |
| League Two | 1,936 / 1,936 | 100.0% |

These denominators are **lineup-enriched rows**, not all scheduled club matches. Values are EUR sums over available listed players, not audited payroll or complete squad wealth, and there is no pre-kickoff scrape guarantee. In the strict eight FT–PT fixtures, FT lineups have an average **€324,375 larger listed-value sum**, but this tiny retrospective comparison is not a validated status classifier. No out-of-club predictive accuracy, age-profile classifier, squad-size classifier or kickoff-time classifier is established. Missingness itself could otherwise become a spurious “upper-tier” signal.

## 4. Stage 2 — all-tier outcomes, prices and BBC proxy xG

### SQL universe and archive gaps

`r01` uses `sofascore.events`, joins provider seasons, and requires finished fixtures with `homeScore.normaltime` and `awayScore.normaltime`. It does not silently mix extra-time or penalty-shootout scores with regulation outcomes. Membership uses both fixture sides. Retrospective membership is appropriate for EDA but cannot itself prove pre-match knowledge.

The 4,068 extracted records are the available **qualifying archive**, not a complete fixture census. In particular:

- Each full Premiership season contributes **198 rather than 228** matches: post-split coverage is absent from this extract.
- Other completed seasons also have gaps, e.g. Championship 23/24 contributes **178/180** and 25/26 **173/180**; League One 25/26 contributes **175/180**.
- 20/21 starts at the January 2021 boundary and 26/27 is in progress. Status-panel analyses begin at 21/22.

These gaps limit transition ordinals, representativeness and historical case studies. A record absent from the normal-time panel is not necessarily an unplayed match.

### League scoring environments — not cross-tier strength estimates

Source: [tier goals](results/r02_tier_goal_descriptives.csv); [season detail](results/r02_tier_season_goal_descriptives.csv).

| Tier | Matches | Home goals | Away goals | Total goals | Home−away goals | 95% home-club bootstrap sensitivity interval |
|---|---:|---:|---:|---:|---:|---|
| Premiership (54) | 1,105 | 1.536 | 1.177 | 2.713 | +0.358 | [−0.017, +0.801] |
| Championship (55) | 1,008 | 1.379 | 1.172 | 2.551 | +0.207 | [+0.014, +0.411] |
| League One (56) | 976 | 1.500 | 1.315 | 2.815 | +0.185 | [−0.052, +0.432] |
| League Two (57) | 979 | 1.463 | 1.237 | 2.700 | +0.226 | [+0.108, +0.367] |

Means are per fixture. The 2,000-resample intervals cluster by home club; they do not fully address shared opponents, changing composition or temporal dependence. They are sensitivity summaries, not causal tier-effect intervals. The reported win-rate intervals elsewhere are ordinary Wilson intervals and assume more independence than this repeated-club panel actually provides.

### Archived market coverage

A valid snapshot contains all three 1X2 prices at one timestamp at or before kickoff. Proportional de-vigging uses

\[
p_k=\frac{o_k^{-1}}{\sum_j o_j^{-1}}.
\]

| Tier | Coherent vectors / fixtures | Coverage | Median minutes before kickoff | Mean log(p_home/p_away) |
|---|---:|---:|---:|---:|
| Premiership | 956 / 1,105 | 86.5% | 5.16 | +0.408 |
| Championship | 479 / 1,008 | 47.5% | 5.95 | +0.333 |
| League One | 840 / 976 | 86.1% | 11.83 | +0.283 |
| League Two | 699 / 979 | 71.4% | 9.25 | +0.292 |

[Timing detail](results/r02_betfair_1x2_coverage_and_supremacy.csv) includes a long stale-price tail: lower-tier tenth-percentile signed offsets are approximately **−215 and −242 minutes**. These observations are therefore labelled **last coherent pre-kickoff archive proxies**, not uniformly close or T−25 prices. Timing and coverage selection can materially affect any market-on-model comparison.

### BBC commentary proxy

The [proxy notes](PXG_NOTES.md) define an empirical-Bayes shot conversion table using BBC text zone, body part and context:

\[
\widehat{xG}_{c}=\frac{g_c+25\bar p}{n_c+25},
\]

with a separate penalty conversion and global fallback for unparsed attempts. Coefficients and counts are saved in [the conversion table](results/r06_bbc_proxy_xg_conversion_coefficients.csv). They are estimated on the extraction window for **descriptive measurement**, not historical prediction; future information relative to an individual fixture is consequently present in this measurement calibration.

The implementation uses the project's commentary parsing/conversion kernel, but intentionally omits the broader feature ladder's shot-count/goals fallbacks. No commentary is missing data, not zero xG. Side availability requires resolved commentary; a true zero-attempt side may therefore be excluded. This is a further selection limitation.

The original matches-enriched proxy universe contained 4,066 fixtures, while the events universe contains 4,068. The difference is **10 events-only and 8 enrichment-only records**, not merely two missing fixtures. The final proxy panel is reconciled to the 4,068 events IDs; [the reconciliation](results/r06_bbc_proxy_xg_fixture_reconciliation.csv) preserves provenance. Reported coverage uses that reconciled universe.

The final extraction contains **46,424 shot-event rows** and **2,249/4,068 fixtures (55.29%)** with both sides available. Coverage is zero before 23/24; the overall percentage should not be read as the contemporary provider failure rate.

| Tier | Both-side proxy fixtures | Coverage | Mean proxy xG per side | Proxy minus goals per covered side |
|---|---:|---:|---:|---:|
| Premiership | 605 | 54.75% | 1.642 | +0.251 |
| Championship | 541 | 53.67% | 1.117 | −0.159 |
| League One | 552 | 56.56% | 1.153 | −0.254 |
| League Two | 551 | 56.28% | 1.095 | −0.259 |

The tier-varying proxy-minus-goal residuals caution against assuming a uniformly calibrated national xG ruler. They are not tier-strength steps. Independent Julia verification matched all **46,424 parsed descriptors** and all **4,068 availability masks**; maximum side-total discrepancy was **4.998×10⁻⁷**, within the unchanged **5×10⁻⁷** CSV serialization tolerance. This verifies the parser/conversion kernel and independent aggregation, not full DataStore feature-route identity.

### Cross-tier bridge conclusion

The database inventory and same-season cross-tier candidate query yield **zero usable cup bridges**. Consequently this report does **not** provide numerical estimates of Premiership→Championship, Championship→League One or League One→League Two strength steps in goals, proxy xG or market log-rates. Treating different league goal totals as those steps would answer the wrong question.

A future bridge extraction needs verified competition IDs, regulation scores, season-specific memberships, explicit reserve/non-SPFL exclusions, venue and competition controls, and dated market/commentary coverage. July cup fixtures must use the coming season's membership—not the most recent league match's old tier. Until such a connected comparison graph or a defensible cross-season invariance model exists, tier hyperparameters are prior assumptions rather than empirically identified steps.

## 5. Stage 3 — within-League-One FT versus PT

The strict panel admits only verified season labels. The sensitivity panel additionally admits clearly marked inferred continuity/date assignments. Hybrid and Unknown are never recoded to PT. Mixed matches are oriented toward the FT side; same-status matches retain home orientation.

| Panel / match category | Matches | Oriented goal difference | Oriented win rate | 95% Wilson interval |
|---|---:|---:|---:|---|
| Strict FT–PT | 8 | −0.750 | 25.0% | [7.1%, 59.1%] |
| Strict FT–FT | 8 | +1.000 | 62.5% | [30.6%, 86.3%] |
| Strict PT–PT | 4 | +0.250 | 25.0% | [4.6%, 69.9%] |
| Inferred sensitivity FT–PT | 11 | −0.364 | 36.4% | [15.2%, 64.6%] |

[Counts and exclusions](results/r02_league_one_ft_pt_status_descriptives.csv) are part of the result. Of 976 League One fixture records, the strict analysis excludes 43 outside the status panel, 905 without accepted evidence and 8 involving excluded status categories. Its eight mixed matches are the **24/25 Arbroath–Inverness / Arbroath–Queen of the South fixtures**; they are not eight independent club-level experiments.

For those eight FT-oriented comparisons:

| Metric | Mean FT−PT contrast | Covered matches |
|---|---:|---:|
| Goals | −0.750 | 8 |
| BBC shots | +0.875 | 8 |
| BBC commentary proxy xG | +0.019 | 8 |
| Fair win probability difference | −0.008 (−0.8 pp) | 8 |
| log(fair FT win probability / fair PT win probability) | −0.019 | 8 |
| Listed lineup value sum | +€324,375 | 8 |

Source: [aligned metrics](results/r02_league_one_ft_pt_aligned_metrics.csv). The sensitivity panel's goal, shot and proxy differences are **−0.364, +2.182 and +0.104**, respectively. These results neither establish a negative FT effect nor support the hypothesised overwhelming FT advantage. Opponent strength, venue, squad quality and financial selection remain confounded, and the evidence-backed sample is tiny. A regression would not make the missing club-season information appear.

### How much of the 1.72 slope is explained?

**Not identified.** The recorded **1.7214** slope belongs to [Experiment 12's](../12_decoupled_generative_xg/README.md) `m02_joint_gamma_poisson`, run `97c7a3d9-a05a-4029-90cb-e34279b8c791`, over 710 held-out fixtures. It is not automatically the production m12 slope or the September slate slope. Adding status to a different cohort cannot yield a defensible fraction of its 0.7214 excess.

A valid follow-up would hold the run, fixture cohort, outcome-price transformation and price instant fixed; regress market supremacy on model supremacy with/without point-in-time status and pedigree; and compare future-season or held-out-club residual performance. Even then, slope change or partial R² would measure association, not the causal fraction attributable to training arrangements.

## 6. Stage 4 — tier transitions and financial exposure

### Observed transition trajectories

`r08_transition_descriptives.py` uses both home and away fixtures and requires membership in consecutive observed seasons. “Downward” means a larger tier number; “upward” the reverse. Administrative relegations and sporting promotions are not separately verified. Windows count **observed qualifying fixtures**, not guaranteed complete calendar histories.

| Direction | Window | Club-fixture rows | Transition club-seasons represented | Mean goal difference | Win rate |
|---|---|---:|---:|---:|---:|
| Downward | 1–5 | 140 | 28 | +0.264 | 42.1% |
| Downward | 6–10 | 129 | 28 | +0.403 | 45.0% |
| Downward | 11–20 | 240 | 24 | +0.263 | 45.4% |
| Upward | 1–5 | 140 | 28 | −0.136 | 38.6% |
| Upward | 6–10 | 126 | 28 | −0.143 | 35.7% |
| Upward | 11–20 | 240 | 24 | −0.037 | 35.8% |

Source: [window descriptives](results/r08_transition_window_descriptives.csv) and [club-fixture panel](results/r08_transition_all_fixture_panel.csv). Only **65/140 downward** and **40/140 upward** first-window rows come from tier-seasons meeting the script's archive-completeness test. Later windows also change cohort through censoring. These cross-sectional curves are **not a posterior learning curve** and do not identify how many matches a flat prior needs to adjust.

Selected observed League One entries illustrate heterogeneity:

| Club / entry season | Observed first ≤20 fixtures | Mean goal difference |
|---|---:|---:|
| Dunfermline 22/23 | 20 | +1.050 |
| Hamilton 23/24 | 20 | +1.500 |
| Inverness 24/25 | 20 | +0.100 |
| Hamilton 25/26 | 20 | +0.650 |
| Ross County 26/27 | 7 | +2.571 |

Ross County's row is heavily censored and cannot be extrapolated to 20 fixtures. Falkirk's entry into League One predates this extraction window; the observed transitions here instead include its later promotions. It is not assigned a fictitious relegation case inside 2021–26.

### Saved model and portfolio lineage

Historical m12 run **`928dad3b-ccaf-4909-b6b7-4f1a815e1cab`**, integer ID **114**, namespace **`scottish_lower_joint_player_2426`**, contains **710 relational latent rows**. Saved portfolio **`a7c4c55b-f8d2-416e-ba85-9c7fe9bedc1a`** reports **1,328 bets**, 99 slates, 622 staked fixtures and **−19.629% maximum drawdown** under its recorded policy and Betfair de-vigged **TWA[−20,0]** pricing. This is a whole-portfolio historical statistic, not transition-attributable drawdown and not the live slate's missing UUID.

### Historical posterior versus market: paired transition windows

Audited `r09` pricing checks the saved `PoissonCountModel` / `JointGammaPoissonObservation` configuration, decodes stored BFCL draws, and averages independent Poisson 1X2 probabilities over draws on the repository's **0:11 goal support without renormalization**. Minimum retained mean score mass is **0.9999988299**. The Python recurrence is independently tested, but is not claimed bit-identical to a Julia fit reload.

Saved draws cover **220 club-side rows / 196 unique fixtures** from 1,015 transition rows. Both sides survive when two transitioning teams meet. **186 club-side rows** also have coherent archive prices; every model and market mean below uses exactly those paired rows.

| Transition | Window | Paired n | Model team-win probability | Market team-win probability | Model−market |
|---|---|---:|---:|---:|---:|
| Downward | 1–5 | 24 | 37.46% | 36.52% | +0.94 pp |
| Downward | 6–10 | 31 | 37.01% | 36.71% | +0.30 pp |
| Downward | 11–20 | 59 | 36.47% | 37.52% | −1.05 pp |
| Upward | 1–5 | 16 | 41.30% | 33.58% | +7.72 pp |
| Upward | 6–10 | 19 | 39.42% | 35.35% | +4.08 pp |
| Upward | 11–20 | 37 | 40.51% | 36.79% | +3.71 pp |

Source: [paired summaries](results/r09_m12_transition_market_summary.csv). The pooled downward entrants are **not generally underpriced** in the first ten observed matches of this selected historical panel. This does not dismiss specific favourite failures: averaging weak and strong entrants can conceal conditional compression. Upward entrants show larger positive model-minus-market differences. Neither trajectory establishes a learning timescale because opponents, price age, archive completeness and represented clubs change between windows.

### Bets against transitioning favourites

Define a transitioning **archive favourite** as having strictly the largest de-vigged 1X2 probability at the archived snapshot. Analyse its opponent separately—not bets on the transitioning team. This gives **91 paired observations**, including the following downward-entrant subset:

| Window | Opponent observations | Mean opponent model−market | Actual saved opponent bets | Saved stake sum | Saved net P&L |
|---|---:|---:|---:|---:|---:|
| 1–5 | 11 | +6.14 pp | 8 | 106.88 | +1.86 |
| 6–10 | 14 | +6.16 pp | 12 | 103.47 | +73.91 |
| 11–20 | 31 | +7.72 pp | 27 | 468.99 | +211.60 |

Amounts are the saved portfolio's currency units. Source: [opponent summary](results/r09_m12_opponent_underdog_summary.csv), [actual opponent ledger subset](results/r09_m12_existing_portfolio_opponent_bets.csv). Across these **47 downward-opponent bets**, saved stakes total **679.34** and net P&L **+287.37**. The full selected panel contains 65 opponent bets. Thus the historical extract does **not** substantiate a blanket claim that betting against transitioning favourites caused a realised bankroll loss. Positive selected historical P&L also does not prove correct probabilities or a prospective edge.

The saved portfolio used **TWA[−20,0]** prices; this suite's favourite classification uses last coherent archive snapshots. Those can disagree in timing and odds. Saved stakes/P&L are preserved, not reconstructed from the different snapshots.

The supplemental quarter-Kelly calculation uses `b=(odds−1)×0.98` and `f=0.25×max(0,[p(b+1)−1]/b)`, with **independent 100-unit capital per club-side**. It is an exposure/settlement diagnostic, not the correlated portfolio allocator or a sequential bankroll simulation. We deliberately do **not** label its aggregate P&L as strategy return or drawdown. Neither it nor the ledger subset identifies drawdown *caused by* a cold-start mechanism. Such attribution needs a complete common-timing bankroll path and a frozen, explicitly specified counterfactual allocation policy.

## 7. Stage 5 — model formulations and architectural recommendation

Full equations and integration seams: [MODEL_DESIGN.md](MODEL_DESIGN.md). Runnable deterministic mock: [l04_pedigree_tier_components.jl](l04_pedigree_tier_components.jl), exercised by [r04](r04_pedigree_tier_components.jl).

### Why centring is not a probability cap

The vector `(c, −c, 0, …)` sums to zero for arbitrarily large finite `c`. Gaussian support permits such contrasts. The resulting win probability also depends on the common scoring level and score model, not just one log-rate contrast. The rejected decoupled funnel demonstrates that those tested alternatives worsened measured compression; it does not logically prove that goal feedback can play no role in any specification.

### A. Informative prior means — preferred first ablation

Let `r_i` denote tier, `c_i` operational category, and `p_i(t)` a strictly history-based pedigree measure. Define

\[
m_i(t)=\tau_{r_i(t)}+\delta_{c_i(t)}+\kappa p_i(t),\qquad
\delta_{PT}=0,\quad \widetilde m_i=m_i-\bar m.
\]

With the project's **additive defence-weakness** convention,

\[
\eta_h=\mu+H+\alpha_h+\beta_a,\qquad
\eta_a=\mu+\alpha_a+\beta_h,
\]

use raw effects with means `+m_i/2` for attack and `−m_i/2` for defence, then centre each vector. The expected supremacy contribution is exactly **`m_h − m_a`**. Centring removes location, not the desired contrast.

A reference-tier hierarchy is

\[
\tau_4=0,\qquad \tau_r=\sum_{j=r}^{3}d_j,\qquad
d_j\sim\operatorname{HalfNormal}(s_d).
\]

Positive adjacent increments encode a substantive prior ordering, not a measured result of this EDA. The reference fixes location but cannot identify gaps in a disconnected match graph. Illustrative regularisers are `δ_FT, δ_Hybrid ~ Normal(0,0.20²)`, `κ ~ Normal(0,0.30²)` and residual attack/defence scales `HalfNormal(0.30)`; these are **not fitted recommendations**. Hybrid gets its own coefficient, not an unsupported fixed half-FT score.

Current tier cancels within a same-tier match. Thus a current-tier-only predictor cannot distinguish a newly relegated League One club from its League One opponents. The relevant cold-start information is **previous-tier pedigree, resources or prior upper-tier performance**. A candidate imported pedigree signal can decay as

\[
p_i(t)=p_{i,0}\exp[-\log(2)D_i(t)/h_p],
\]

with a predeclared or regularised half-life and no use of subsequent results to construct `p_i,0`.

### B. Linear supremacy covariates — competing ablation

Use finite, precomputed category contrasts such as

\[
x_{FT,m}=\tfrac12(I_{h,FT}-I_{a,FT}),\qquad
q_m=w_{FT}x_{FT,m}+w_{Hybrid}x_{Hybrid,m}+w_p x_{p,m},
\]

and add `+q_m` to home and `−q_m` to away. The factor one-half avoids doubling the intended home–away log-rate effect under `SupremacyRole`.

Fit dynamic team ratings as **residuals around the supplied mean**, or use strongly regularised covariates with explicit residual diagnostics. Do not initially fit the same freely weighted status/pedigree signal both in prior means and in the predictor: the team effects can absorb it and the resulting coefficients need not be separately identified. Club fixed effects similarly remove much of a nearly time-invariant status signal.

### Architecture, AD and validation

- Feature preparation—not `@model`—handles effective dates, knowledge timestamps, missingness, centring constants and categorical designs.
- Model input vectors must be finite `Float64`/`Int`; production must declare the feature dependency and reproduce OOS extraction.
- Use vectorised arithmetic and indexed gathers in the taped likelihood. No parameter-dependent control flow or mutation is introduced.
- Informative means belong in a dynamics component; the competing fixed contrast belongs in the existing covariate interface. The suite's mock and documented API sketch are **not drop-in production components**.

The synthetic 4-team/6-fixture mock passed: **31 tape instructions**, compiled-versus-fresh relative error **0**, compiled-versus-ForwardDiff **1.075×10⁻¹⁶**, perturbed compiled-versus-fresh **0**. Warmed allocation was **2,432 bytes/gradient**. This verifies the tested mock's calculus, not allocation-free Turing integration, model convergence, predictive improvement or correct production feature filtration.

## 8. Decision, outstanding gates and next experiment

**Do not promote a pedigree/status component on these results.** Prioritise a trustworthy national history and initial-state prior over an unsupported binary “FT always beats PT” correction.

| Work-package requirement | Delivered / remaining gap |
|---|---|
| Six-season, all-SPFL status dataset | 252 memberships and explicit evidence classes delivered; **223 Unknown labels remain**. Definitive status criterion not met. |
| All-tier empirical comparison | Goals, market and proxy coverage/descriptives delivered; **cross-tier cup bridges absent**, so adjacent strength steps unestimated. |
| League One FT/PT delta | Strict/sensitivity matched summaries delivered; small selected panel, no causal effect or compression attribution identified. |
| Transition lag and financial cost | Observed all-side windows, saved m12/portfolio lineage and live settlement audit delivered; archive gaps and price/provenance differences limit attribution. No universal adaptation time established. |
| Informative priors versus covariates | Equations, sign/centring conventions, collinearity discussion and mock gradient checks delivered; production integration and zero allocation not verified. |
| Compute discipline | Read-only extraction, descriptive calculations and deterministic AD only; no sampling grids. |

The next controlled study should compare **(0) unchanged control, (A) pedigree/status initial-state priors, (B) covariate-only, and (C) resource-only control**, on identical future-held-out fixtures, unchanged score kernels and unchanged staking policy. Preregister transition-age strata, status-missingness treatment and market instant. Evaluate proper scores, conditional calibration, extreme-underdog exposure and portfolio risk with club/fixture dependence respected. Keep informational changes separate from trust/recalibration changes. That study is proposed, **not run here**.

## 9. Reproduction and verification

Run from repository root. Keep `BF_DB_URL` and optional `BF_EXPERIMENTS_DB_URL` in the normal credential environment; never put secrets into commands or committed files. The saved CSVs support offline re-analysis. Re-extraction overwrites snapshot files, so preserve the existing manifest if comparing versions.

```bash
SUITE=experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda
# This worktree has no Manifest; the already-instantiated main checkout was used.
JULIA_PROJECT=/home/james/bet_project/BayesianFootball

python "$SUITE/r01_extract_empirical.py"
# Curated source CSVs are checked in; this idempotently applies the later tranche.
python "$SUITE/r11_extend_status_evidence.py"
julia --project="$JULIA_PROJECT" "$SUITE/r03_status_construction.jl"
bash "$SUITE/r06_run_bbc_proxy_xg.sh"
julia --project="$JULIA_PROJECT" "$SUITE/r06_verify_bbc_proxy_xg_kernel.jl"
python "$SUITE/r02_empirical_eda.py"
python "$SUITE/r08_transition_descriptives.py"
python "$SUITE/r08_query_saved_m12.py"
python "$SUITE/r09_m12_transition_pricing.py"
python "$SUITE/r09_m12_pricing_tests.py"
python "$SUITE/r05_slate_audit.py"

julia --project="$JULIA_PROJECT" "$SUITE/r04_pedigree_tier_components.jl"
julia --project="$JULIA_PROJECT" "$SUITE/r10_status_contract_tests.jl"
python "$SUITE/r07_validate_artifacts.py"
./scripts/todo.sh check
```

Python extraction requires `psycopg`; saved-draw pricing additionally requires `zstandard`. Julia uses the existing project dependencies; no package versions were changed. `r09_fetch_status_metadata.py` separately retrieves publication metadata omitted from readable web extraction; it is not needed for offline EDA.

The status-join regression suite checks known-ID matching, conflicting-ID rejection, documented name fallback, interval boundaries and conflict refusal. `r07` checks unique memberships, source references, agreement with all 42 current official tier assignments, unique ledger orders and the critical settlement arithmetic. The `r02` home-only transition candidate CSV is a **superseded diagnostic**, not the basis of §6; use `r08` instead.

### Artifact map

- [STATUS_NOTES.md](STATUS_NOTES.md): classification scope and source limitations.
- [EMPIRICAL_NOTES.md](EMPIRICAL_NOTES.md): SQL and empirical workflow notes.
- [PXG_NOTES.md](PXG_NOTES.md): proxy definition, coefficients, reconciliation and kernel verification.
- [SLATE_AUDIT.md](SLATE_AUDIT.md): live odds/probability/stake/settlement audit.
- [MODEL_DESIGN.md](MODEL_DESIGN.md): complete modelling specification and integration sketch.
- `data/`: membership, evidence, fixture/price/lineup extracts and proxy panels.
- `results/`: all tabulated calculations and [checksum manifest](results/artifact_manifest.json).

**Reporting rule:** use the final checked CSVs and this synthesis for headline numbers. Early intermediate notes or the original work-package narrative are not evidence of completed acceptance criteria.
