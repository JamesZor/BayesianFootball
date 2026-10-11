# Stage 1 — 12 testable questions

**★ = five priorities (Q1–Q5).** Run the coverage/filtration checks Q6/Q12 first as prerequisites. These are proposed diagnostics, not new research findings or permission for new QS/Turing fits. Source keys resolve in [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

## Common contract

- Use the supplied 21/22–25/26 market panel and existing fit outputs; older results back to 2008 only for explicitly separate results-only comparisons. Never tune on 26/27; coverage descriptions only. Follow DESIGN.md's cohort definitions and held-out panel.
- All market-derived prior ingredients for season S, including hyperparameters, tier offsets and normalisation, use **seasons `<S`**. Past cohorts' destination-season observations may be training labels only when that season is `<S`. Evaluate forward by season, not random fixture splits or leave-one-club-out with later seasons retained.
- A quote identifies `D=(log λh−log λa−γ)/2=qh−qa` and `U=(log λh+log λa−2μ−γ)/2=sh+sa`, not an independent club rating. Use connected, opponent-adjusted reconstructions with explicit gauge/error estimates. First-quote tests should predict the fixture's D/U directly or use opponent anchors available before the quote, never future-smoothed opponents. [report §2 derivation]
- Report club-season counts, fixtures and missingness per statistic. Use club-season-clustered uncertainty for the first-20-game transition-bias gate; paired fixture-clustered uncertainty for model log-loss differences. For tiny cohorts show every club, leave-one-club sensitivity and descriptive intervals rather than declare a powered null. Fixtures with two movers must not be treated as two independent outcomes. [DESIGN.md §4; proposed safeguards]
- “Supports/refutes” below means a directional diagnostic with uncertainty; it is not a fitted-arm gate. A wide interval is inconclusive, not evidence that a fix is unnecessary. No new production model fits are needed for these questions.

## Q1 ★ — Does entry bias decay with information, or persist by direction?

- **Hypothesis:** cold start produces a large early error that diminishes with own-match information; missing tier scale leaves a persistent direction-specific residual.
- **Statistic:** transitioning club's mean `P_model(win)−P_market(win)` in games 1, 2–5, 6–10, 11–20, plus the prespecified first-20 aggregate; cross-tabulate old in-scope match count and old all-tier market count. Show paired differences between W2 base/A2 counterparts on identical fixtures and the separate qs_marketobs panel. Also plot q/s rate residuals where existing artifacts permit; do not fabricate missing latents.
- **Data:** existing fits named in DESIGN.md §6, held-out fixture predictions, de-vigged closes, club-season tiers and training-history counts. Compare QS/W2 only on common coverage, not unmatched headline means.
- **Support/refute:** early errors concentrated among history-poor clubs and declining toward zero support (a); stable negative relegated/positive promoted errors after 10–20 games support (b). Lack of either pattern weakens those explanations; A2 improvement is descriptive, not independent validation of W2-selected finalists.
- **Informs:** causes **a/b**, fixes **F1/F2/F3/F5**. Basis: [OW6; RI11; project design].

## Q2 ★ — Does a genuinely prior-season market prior predict the first new-season quote?

- **Hypothesis:** old club strength plus a learned direction-specific adjustment predicts the opening market better than zero or an undifferentiated entrant mean.
- **Statistic:** expanding-season prior-only comparison of P1/P2/P3 in PRIOR_FORMS.md: signed error, MAE/RMSE for match D/U, predictive interval coverage and predictive log score including quote/opponent uncertainty. Show promotion/relegation/first-SPFL strata, estimated carry slopes and transition offsets; compare with prior-season goal-difference-plus-tier baseline and no-shift baseline. Re-evaluate fixed-prior errors at games 5/10/20 to describe settling, not to retrain the season's prior.
- **Data:** all-tier old closes and tiers, first accepted destination-season quote (record actual game number), later checkpoint quotes and previous-season results; training labels strictly before each validation season. Report how few genuinely pre-23/24 transitions have both sides covered.
- **Support/refute:** lower forward error with correct sign and calibrated coverage supports F1/F2; improvement only in pooled/in-sample results refutes the evidence for transport. If club history adds no predictive value over cohort means, prefer stronger pooling. Reject forced antisymmetry if direction-specific offsets materially differ after common-scale adjustment; otherwise report uncertainty.
- **Informs:** **a/b**, **F1/F2**; candidate prior selection, not a trading-edge claim. Basis: [RI11; OW6; EG].

## Q3 ★ — Is a tier step identified, and is 0.47 being compared in the right units?

- **Hypothesis:** connected cross-tier evidence supports a mostly-quality step, but spacing may differ between adjacent Scottish divisions.
- **Statistic:** graph of league-season connectivity through movers and cups; rank/conditioning of the constrained market-rating reconstruction; adjacent-tier `Δα`, `Δβ`, `Δq`, `Δs` contrasts with uncertainty. Compare constant-step versus three separate adjacent-step descriptive residuals; sensitivity excluding Old Firm and one linking club/competition at a time. Report A2's total-step/split convention explicitly before comparing 0.47 with `2Δq` or any other quantity.
- **Data:** prior-season all-tier inverted closes, club-season movement, cup results and timestamped bookmaker odds; use cup and mover estimates separately before combining. Existing A2 documentation/outputs only as available to Stage 1.
- **Support/refute:** connected, stable signed contrasts across independent link types support (b)/F3/F5; disconnection or collapse when one bridge is removed means the step is not securely estimated. Large adjacent-gap differences weaken a single linear step. A mover-only estimate cannot by itself distinguish league offset from a free mover jump.
- **Informs:** **b**, **F2/F3/F5**. Basis: [DC7; report §1 unit conversion and §2 identification].

## Q4 ★ — Do non-movers need a distinct summer uncertainty step?

- **Hypothesis:** non-mover end-to-start variation exceeds ordinary within-season variation after accounting for elapsed time and estimation noise.
- **Statistic:** q/s mean changes and covariance of late-season→opening-window residuals versus within-season windows matched on quote count and, where possible, calendar gap; compare residual dispersion with existing QS boundary predictive variance. Show a per-calendar-day and a per-refit-step comparison. Propagate market reconstruction uncertainty, including shared-opponent covariance; do not just subtract two noisy rating means and call their SD process noise.
- **Data:** non-movers' prior/current-season quotes through 25/26, dates, tier/season μ and γ estimates, existing QS state/boundary summaries if available. Keep league-season centring consistent across the boundary.
- **Support/refute:** extra dispersion with little mean drift supports F4 variance inflation; systematic mean reversion supports a separate mean-dynamics question, not just larger variance. Similar dispersion after noise adjustment weakens (c). Signed mover-only errors with stable non-movers instead support F2.
- **Informs:** **c versus b**, **F4**, boundary uncertainty for **F1/F2**. Basis: [KL12; RI8; RI11; EG eq.(7)].

## Q5 ★ — Is error tied to time since refit rather than games since promotion?

- **Hypothesis:** market updates arrive before a biweekly fit incorporates them, causing a refit-age pattern independent of cold start.
- **Statistic:** signed/absolute model−market residual by days since training cutoff and count of own matches since cutoff, stratified by game-number bin and cohort; before/after-refit event plots on comparable fixtures, with non-movers as control. Relate errors to the market movement since the last included quote. Use only prediction vintages whose cutoffs precede their fixture; where successive vintages for the same future fixture exist, pair them.
- **Data:** existing fit cutoff metadata, dated predictions and quotes through 25/26; no weekly refits requested during EDA.
- **Support/refute:** residual build-up within folds followed by reduction after refitting, persisting after game-number stratification, supports (d)/F6. Persistent same-sign errors immediately after refits weaken lag as the sole cause. This diagnostic does not quantify a weekly-refit improvement without a later paired arm.
- **Informs:** **d versus a/b**, **F6**. Basis: [project design; PI §2 sequential update precedent].

## Q6 — Is missing or stale market history itself selecting the apparent effect?

- **Hypothesis:** entrant prior reliability varies substantially by tier, gap length and accepted quote coverage.
- **Statistic:** per club-season counts, last accepted quote date, first observed game number, effective recency-weighted sample size, inversion rejection rate where metadata exist, and availability of both sides of a transition. Compare bias/prior errors for complete coverage versus all available coverage; show results under minimum-count thresholds rather than silently dropping clubs.
- **Data:** 21/22–25/26 all-tier quote inventory and fixture/tier rosters, specifically the Championship 25/26 gap; 26/27 inventory only, no outcomes or performance diagnostics.
- **Support/refute:** large coverage-conditioned error differences or late “first quotes” support a reliability-aware fallback and weaken unqualified opening-day claims. Stable comparisons across coverage cuts weaken missingness as the explanation. No-history entrants must remain in an explicit fallback group.
- **Informs:** **a**, **F1/F2/F5**. Basis: [EG §4.2; supplied data card].

## Q7 — Is the transition predominantly quality, and does it share QS's style scale?

- **Hypothesis:** mean transition shifts are mainly q, but boundary uncertainty need not have the within-season style/quality ratio ≈0.3.
- **Statistic:** separate q/s shift means, noise-adjusted variances and covariance by direction; report uncertainty on `SD(s)/SD(q)` and on attack/concession split. Compare changes in D with changes in U after μ/γ adjustment. Show sensitivity to reconstruction/gauge choices and shrink sparse covariance estimates toward pooled values.
- **Data:** linked old/new market summaries, estimated reconstruction covariance, existing QS scale summaries if available; not raw goals-for/against averages presented as q/s.
- **Support/refute:** near-zero s means with stable q shifts support a quality-focused F2; reproducible s shifts or larger style spread weaken a fixed near-pure-q correction or forced 0.3 boundary ratio. If uncertainty overwhelms these contrasts, retain simpler covariance with an explicit uncertainty floor.
- **Informs:** **a/b/c**, **F1/F2/F3/F4**. Basis: [DC7-derived rotation; RI11; report §1 TODO 029 covariance caveat].

## Q8 — Is carry non-unit because moving clubs are selected extremes?

- **Hypothesis:** a fixed tier subtraction misses regression from strong/weak source-league extremes; destination performance depends on source strength with carry less than one.
- **Statistic:** cohort-centred destination opening D/q proxy versus old-season adjusted q, with measurement-error-aware slope or reliability sensitivity; compare end-window versus whole-old-season summaries and source rank/goal difference. Compare movers with source-strength-matched non-movers where overlap exists; report lack of overlap rather than extrapolate. Score carry=0, carry=1 and shrunken estimated carry chronologically as Q2.
- **Data:** old closes, results/rank, movers and non-movers, dates and uncertainty; old results-only cohorts back to 2008 as a separate robustness panel, not fabricated market ratings.
- **Support/refute:** improved forward performance from partial carry supports it; an apparent slope below one that vanishes after old-rating measurement-error adjustment is not evidence of genuine mean reversion. Full carry working equally well weakens this extension; no overlap prevents a causal mover interpretation.
- **Informs:** **a/b**, **F1/F2/F3** (A2 carry comparison). Basis: [RI11 exploratory finding; proposed selection/reliability diagnostic].

## Q9 — Do cups add cross-tier information, or a competition/source distortion?

- **Hypothesis:** cup-based tier contrasts survive adjustment for venue, lineup rotation and bookmaker source; otherwise they are a biased anchor.
- **Statistic:** cup-versus-mover contrast differences by competition, tier gap and available lineup-retention bins; separately compare bookmaker and Betfair implied D/U on **league fixtures with both sources**, if such overlap exists. Check quote timestamp, de-vigging, regulation-time settlement and inversion residuals. If cup 1X2-only rates are weakly identified, use 1X2 contrasts rather than pretend totals/style are precisely observed.
- **Data:** Scottish/League/Challenge Cup results, bookmaker 1X2 odds and lineups; league quotes for source bridge only where available; no assumed Betfair cup feed.
- **Support/refute:** agreement on overlapping support supports F5; systematic source/rotation shifts or absent overlap weaken pooling. Do not label disagreement a tier effect if source and competition are inseparable.
- **Informs:** **b**, **F3/F5**. **F7 is conditional only:** if later pooled arms use pxG, request the existing league-specific pxG-vs-goals audit; market/cup differences alone cannot justify a pxG scale change. Basis: [DC7; supplied data card].

## Q10 — Is the model wrong against outcomes, or is the market wrong on movers?

- **Hypothesis:** apparent QS “edges” are mainly model error, not a persistent mover-specific market inefficiency.
- **Statistic:** cohort `I(win)−P_market(win)` and `I(win)−P_model(win)`, calibration by market-probability/home-away bins, and paired model-minus-market proper scores on the same fixtures. Compare movers with price/home-status-matched non-movers; show per-season signs and club-season uncertainty. Any offered-odds ROI is secondary, with commission/spread conventions stated and no strategy mining.
- **Data:** held-out 23/24–25/26 results, existing model predictions and aligned closes; never the held-back 26/27 outcomes.
- **Support/refute:** market-calibrated outcomes but systematically displaced model probabilities support fixing QS; a stable market residual across forward seasons motivates investigation of market bias, not an immediate contrarian rule. Sparse noisy ROI alone supports neither claim.
- **Informs:** **a/b diagnostic validity**, **F1/F2/F3** evaluation against outcomes. Basis: [WM abstract; BS limitations; project gates].

## Q11 — Is summer dispersion related to squad discontinuity?

- **Hypothesis:** low retained-player minutes correlate with larger opening q/s innovations and reduced carry, including among non-movers.
- **Statistic:** previous-to-opening-window retained-minute share or lineup overlap versus absolute and signed market innovations; stratify by direction/old strength and show matched non-mover comparison. Distinguish increased variance from consistently positive/negative means; show missing-lineup coverage.
- **Data:** lineups/minutes and old/new market summaries; manager-change covariates only if independently timestamped data are already available. Opening-season lineups are diagnostic labels, **not inputs to the prior** under the strict earlier-seasons rule.
- **Support/refute:** larger residual dispersion at low retention supports a heterogeneity diagnostic for F4/F2; disappearance after strength/coverage adjustment weakens the mechanism. Association is not a causal effect of transfers or managers.
- **Informs:** **c**, **F4**, optional uncertainty structure for **F1/F2**. Basis: [ST abstract; RI11].

## Q12 — Can each proposed prior pass a time-and-evidence provenance audit?

- **Hypothesis:** a usable F1/F2 prior can be constructed without future labels, backdating or duplicate use of market observations.
- **Statistic:** for every validation fold, manifest of source seasons/IDs, maximum timestamp, target node date, hyperparameter-training seasons, reference/gauge definition, fallback branch and overlap with QS's market-likelihood row IDs. Count violations. Verify bit-identical prior inputs across folds of one target season; record effective historical sample counts after disjointness rules.
- **Data:** EDA-derived prior summaries, quote IDs/dates and existing fit training manifests; missing manifests are an unresolved implementation dependency, not an automatic pass.
- **Support/refute:** zero chronology violations and a declared disjoint/sequential evidence policy support implementability. Any target-season prior input, hyperparameter fitted on validation labels, future-informed initial node or unacknowledged duplicate market rows blocks that construction. An empirical-Bayes overlapping-history alternative must be explicitly labelled and separately justified, not called exact Bayesian updating.
- **Informs:** validity of **F1/F2/F3/F5** for all causes; targets **a/b** without leakage. Basis: [OW6; RI11; report §3 filtration derivation; DESIGN.md §7].
