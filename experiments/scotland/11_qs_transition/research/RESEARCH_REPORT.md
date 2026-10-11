# Stage 0 — research summary
- Use a mover-specific initial mean, not merely more zero-centred noise: Scottish precedent already replaces the league-average entrant prior. [OW6]
- Separate a change of league reference from a change in the club itself; cross-division identification needs links, not just more disconnected matches. [DC7; derivation in §1]
- Published English division contrasts imply quality gaps of about 0.14–0.27 in QS units, **not** a universal Scottish mover shift. [DC7; calculation in §1]
- Cohort priors and explicit summer uncertainty have direct football-model precedents. [RI11; RI8]
- Do not assume the relegation correction is the negative of the promotion correction; the reviewed studies do not establish that symmetry. [OW6; RI11]
- Market-rate observations have a close precedent, but using a future fixture's odds is different from an earlier-seasons-only prior. [EG, §§3–4]
- No verified evidence here establishes an exploitable Scottish promoted/relegated-club market bias or a universal settling time. [scope below; WM, abstract]
- Prioritise chronological prior-only checks, cohort bias curves, cross-tier identification, non-mover summer dispersion and refit-age diagnostics. [proposals in EDA_QUESTIONS.md]

## Evidence conventions and scope

**Established** below means a method or result verified in the named source, not universal truth or a causal estimate. **Suggestive** means limited empirical evidence or an analogy. **Proposal/derivation** means our adaptation, not a published QS result. Citation keys resolve to library **doc_id + source-PDF page**, or **title + URL**, in the source register below. Project facts come only from the supplied [brief](ASTRA_RESEARCH_BRIEF.md) and [DESIGN.md](../DESIGN.md), not an independent data audit.

## 1. Promotion/relegation effects: magnitude, asymmetry and settling

### Established: useful precedents, but no transferable transition constant

- Dixon–Coles jointly modelled four English divisions and cups. Their Table 3 has mean multiplicative attack/concession parameters `(1.38, 0.68)`, `(1.07, 0.86)`, `(0.83, 1.14)`, `(0.73, 1.32)`, from highest to lowest division. Lower concession parameters mean better defence. These are **division averages**, not effects estimated on moving clubs. [DC7]
- Owen's Scottish Premier League study used the previous season's fitted attack/defence means for incumbents and **the relegated club's prior for the promoted replacement**. The paper explicitly calls this simplistic. Its 2003/04 optimisation season had no mover because Falkirk could not be admitted; consequently it is not a validation of entrant-prior quality. [OW6]
- Ridall et al. use a separate common Gamma attack/defence prior for newly promoted EPL clubs, estimating its hyperparameters on the first 18 seasons and evaluating online on the next 14. Their exploratory analysis says promoted-club performance did not depend strongly on Championship history; this is a reason to **test carry**, not assume it is one. It is not a demonstrated absence of value in Scottish market history. [RI11]

### Derived numerical comparisons — check units before using any of them

From QS's supplied equations, `α=q+s`, `β=s−q`. For a higher-minus-lower division contrast, transform Dixon–Coles's **ratios of arithmetic means** as

\[
\Delta\alpha=\log(\bar\alpha_H/\bar\alpha_L),\quad
\Delta\beta=\log(\bar\beta_H/\bar\beta_L),\quad
\Delta q=(\Delta\alpha-\Delta\beta)/2,\quad
\Delta s=(\Delta\alpha+\Delta\beta)/2.
\]

| English division contrast | Δα | Δβ (concession) | Δq | Δs | Δα−Δβ |
|---|---:|---:|---:|---:|---:|
| Premier − Division 1 | +0.254 | −0.235 | +0.245 | +0.010 | 0.489 |
| Division 1 − Division 2 | +0.254 | −0.282 | +0.268 | −0.014 | 0.536 |
| Division 2 − Division 3 | +0.128 | −0.147 | +0.138 | −0.009 | 0.275 |

These are calculations from [DC7], not estimates of mean log-strength differences, with no propagated standard errors. They suggest predominantly **quality**, rather than style, differences and non-uniform spacing. Their English historical scope prevents treating them as Scottish estimates. [DC7; algebra above]

**A2 comparison, conditional on the context card's split convention:** if the supplied 0.47 is a *total* attack-plus-defensive-improvement step split 48/52, then an upward strength contrast is `(Δα,Δβ)=(0.2256,−0.2444)`, hence `(Δq,Δs)=(0.235,−0.0094)`. That is comparable to the last column, **not directly to Δq**, in the table. Stage 1 must confirm A2's reported convention; Stage 0 did not inspect implementation. A club moving **up** into stronger opposition has a negative *league-relative* q rebase, even if its absolute strength is unchanged. [supplied brief; derivation from QS equations]

Ridall's entrant hyperparameters are `Gamma(19.3,23.9)` for multiplicative attack and `Gamma(30,26.4)` for multiplicative concession. The shape/rate means are about `0.808` and `1.136`; logging those means and rotating yields approximately `(q,s)=(−0.171,−0.043)`. These are **not** `E(log strength)`, not QS prior SDs, and not a league-centred Scottish correction: Ridall uses a different identification convention. They provide a sign/scale illustration of a below-average entrant prior only. [RI3; RI8; RI11; RI12; calculation]

The supplied TODO 029 relegation prior rotates to `(E q,E s)=(+0.14,−0.02)`. Its marginal SDs cannot determine q/s uncertainty without `Cov(α,β)`: if independence were assumed, `SD(q)=SD(s)=0.164`, `Cov(q,s)=−0.0069`. Thus it would **not** reproduce the supplied QS style/quality scale ratio ≈0.3. Do not carry over its leaked fitted numbers as hyperparameters. [supplied brief; linear covariance transformation]

### Suggestive, not established

A within-league prior for selected winners from below is not the inverse of one for selected losers from above; the reviewed entrant models do not estimate both directions jointly. The relevant EDA is separate **origin/destination × entry-type** means and spreads, with pooling when samples are small. A league-average gap is a rebase, whereas a mover's residual change can include summer recruitment, retention and regression from selection extremes. [OW6; RI11; proposal/identification reasoning]

No universal “settles by game 5/10/20” result was verified. The pi-rating paper's cold-start experiment suggests roughly **two seasons**, with a further season for the most extreme teams, to reach acceptable ratings from zero under its learning rule. This is **algorithmic convergence**, not a biological or market adaptation time for promoted clubs, and cannot be mapped to a QS fortnight. [PI, §4.2]

## 2. How rating systems handle movers and link divisions

### Established methods

| Family | Verified handling | Implication for QS (proposal) |
|---|---|---|
| Elo / ordered logit | Hvattum–Arntzen use Elo differences as ordered-logit covariates. Their later detailed implementation establishes initial ratings using two seasons spanning divisions; entrants to the lowest observed division accumulate matches before getting a performance rating. [HA; AH, §A.1, PDF pp.2–3] | Retain a common scale for observed movers; an entrant warm-up method does not solve a first-match prediction unless a fallback is specified. Do not backdate the established rating to warm-up matches. |
| pi-ratings | New teams start at zero; distinct home/away ratings learn from goal-difference forecast errors. The mean-zero reference can be one league or multiple leagues linked by inter-league/cup matches. [PI, §2] | A new rating family does not automatically fix cold starts. Its ratings are diminished goal-difference units, not log-rate q/s. |
| Practitioner Elo | SinceAWin's English/Scottish league-only system reallocates relegated clubs' points to promoted clubs and vice versa to preserve divisional point totals. [SA] | Explicit rebasing is operational precedent, not an empirically estimated club jump. |
| Dynamic Poisson | Owen uses last-season initial means; Koopman–Lit handle inactive teams through missing observations in a common state vector. [OW6; KL12] | Distinguish unseen entrants from returning clubs with stale information; retain uncertainty during absence. |
| Multi-division Poisson | Dixon–Coles identify relative divisions using movers over time and inter-division cup results. [DC7] | Full-pyramid scope alone is insufficient if components remain disconnected or continuity assumptions are too weak. |

**ClubElo limitation:** the requested system page redirected to a fixtures homepage during retrieval; no original ClubElo promotion/relegation algorithm was verified. SinceAWin says its approach is based on ClubElo **with differences**; do not relabel its replacement rule as ClubElo's. [SA; retrieval record: `http://clubelo.com/System` → `https://clubelo.com/`]

### Derived identification check: what a market quote actually identifies

For a home/away fixture, the supplied QS equations imply

\[
D={\log\lambda_h-\log\lambda_a-\gamma\over2}=q_h-q_a,
\qquad
U={\log\lambda_h+\log\lambda_a-2\mu-\gamma\over2}=s_h+s_a.
\]

Thus one inverted close is **not** an independently observed club q/s. It contains opponent, home-advantage and goal-environment information. Recover club summaries jointly on a connected schedule with stated centring and uncertainty, or construct a first-quote diagnostic against an opponent estimate fixed before that quote. A disconnected league's quality offset is unidentified from its internal matches; global style shifts are confounded with μ. [derivation from supplied QS equations; connection to DC7]

A mover links disconnected tiers only through an assumption about its persistence/jump. If both tier offsets and free club jumps are inferred from movers alone, they can trade off. Cups offer additional links, but cup lineups and bookmaker-versus-exchange pricing may change the estimand. Treat cups as a **cross-check**, stratified by competition and available lineup strength, not an unquestioned anchor. [DC7; supplied data card; proposed sensitivity analysis]

## 3. Market-informed priors and market efficiency

### Established precedents

Egidi–Pauli–Torelli infer scoring rates from de-vigged bookmaker probabilities, introduce those as auxiliary observations with bookmaker-level noise, and combine historical and market rates in the Poisson intensity. Their model has season-to-season Gaussian attack/defence evolution and zero-sum identification. This is a close conceptual precedent for market-rate information, but **not the same** as QS's KL inversion or a Gaussian prior on a club's q/s. They explicitly do not implement a relegation system, and sparse-history teams have wider posterior intervals. [EG, §§3.1–3.2, 4.2]

The arXiv v1 displayed inversion equations (4) are complementary probabilities and therefore supply only one independent constraint as written. Do not copy that inversion recipe as proof of unique identification of two rates; retain the project's accepted inversion/QA and test sensitivity to the input market. This is an algebraic caveat about the fetched version, not a claim about its implementation or final published correction. [EG, eq.(4); derivation]

Pre-tournament outright odds can also imply strengths: Zeileis–Leitner–Hornik combine de-margined odds from 26 bookmakers/exchanges and invert tournament simulations to obtain team abilities. Outright odds reflect the tournament path as well as strength; their inversion is not equivalent to taking a log of title odds. This is a **method precedent**, not a recommendation to procure new Scottish outright data overnight. [ZL, abstract and §1]

### Required chronology for this experiment — proposed stricter protocol

The literature distinguishes train/test periods (e.g. Ridall's 18/14-season split and Owen's round-ahead forecasts); the project additionally requires **seasons strictly before the target season** for every market-derived prior. A source's statement “out of sample” is not sufficient evidence of that stricter restriction. [RI11; OW6; DESIGN.md §7]

1. For target season S, build all prior means, re-ratings, uncertainty calibration, tier links and centring constants only from seasons `<S`. Freeze them across S's biweekly folds. A training mover label may use its destination-season opening quote only if that destination season is itself `<S`. [project rule; proposed implementation contract]
2. First target-season quotes and games 5/10/20 are **outcomes for checking the prior**, never inputs to that prior. Archive timestamps; a closing line is not available at an earlier execution time. Current-season observations can update the ordinary QS likelihood after they occur without updating the earlier-season prior recipe. [project rule; proposed filtration contract]
3. Do not use whole-season smoothed opponent ratings to label an “opening” club quote. Prefer direct match-level `D,U` prediction or a pre-quote opponent anchor; label any retrospective reconstruction descriptive only. [§2 derivation]
4. Audit **double use**: a prior built from closes already present in QS's historical market likelihood reuses evidence. Safest options are disjoint pre-window summaries or a boundary posterior used with only subsequent observations. Otherwise label it empirical-Bayes regularisation and assess sensitivity; widening the variance does not magically restore exact Bayes. [proposed Bayesian factorisation safeguard; market auxiliary-observation precedent EG §3]
5. A season-start market mean must not be attached to a state at the **beginning of a multi-season training window** if its inputs occur after that state time. Specify the intended boundary/initial node before implementation. [proposed temporal safeguard; OW6 season-start construction]

### What is known about exploitable mover biases?

**Established broad caution, not mover evidence:** Winkelmann et al.'s published abstract reports 14 consecutive seasons and finds isolated betting-market inefficiencies that are not persistent/systematic across leagues; simulations can produce significant single-season effects even under efficiency. It does not establish efficiency of Scottish mover prices. [WM, abstract only]

**Weak direct evidence:** a Bocconi student-club analysis of EPL/Serie A 2005/06–2023/24 reports about +2.45% average seasonal return betting against promoted EPL teams away, concentrated more in earlier seasons, and no positive average strategy for Serie A. This is non-peer-reviewed, with strategy selection and limited uncertainty reporting. Its assertion that a positive realised return must exceed the bookmaker margin to count as inefficiency is not adopted here: an offered-odds return already incorporates the price. No transportable prior or betting rule follows. [BS, “Checking for Profitable Strategies”; methodological assessment]

**Research conclusion:** within the reviewed material, no robust promoted-versus-relegated Scottish market inefficiency was verified. Therefore diagnose `model−market` and separately `outcome−market`, matched on price/home status; disagreeing with the market is not itself evidence of an edge. No universal market efficiency claim is made. [WM; BS; proposed EDA Q10]

## 4. Cold-start and partial pooling for zero-centred walks

### Established modelling lessons

Baio–Blangiardo explicitly identify overshrinkage: a common population distribution can under-rate very good teams and over-rate poor teams. They propose top/middle/bottom mixtures and non-central t attack/defence distributions to reduce it. This is a warning against pooling all entry types toward an undifferentiated zero; it is **not** a validation of a particular transition-cohort partition or Student-t QS transition. [BB6–7]

Owen separates a non-zero prior mean vector from zero-centred Gaussian variation and uses a constrained parametrisation to preserve identifiability. Ridall directly distinguishes incumbent evolution from promoted-entrant priors and trains the latter on past seasons. These are more immediate precedents for F1/F2 than introducing an entirely new rating system. [OW6; RI11]

### Proposal for QS

Keep residual innovations zero-centred, but locate the **initial/boundary state** at an informed mean: `x_i,0 = m_i + L_i z_i`, `z_i ~ N(0,I)`, `x=(q,s)`. A zero-centred random *innovation* does not require zero prior *strength*. Carry a separate covariance for measurement error, entry heterogeneity and uncertain hypermeans. In particular, do not automatically impose the within-season ratio `σ_s/σ_q≈0.3` on a transition prior. [adaptation of OW6 and RI11; algebra in §1; supplied QS ratio]

Use a fallback ladder: reliable old market history → direction/tier cohort distribution → pooled entry-type distribution with greater uncertainty → broad declared zero-centred baseline when no older cohort exists. The entrant class is known from promotion/relegation status, unlike a latent class inferred from the season's final table. Avoid learning a separate covariance for a two-club cohort without shrinkage. These are proposed safeguards for the supplied small cohorts, not established optimal cut-offs. [RI11; BB6–7; supplied DESIGN.md §4]

Three explicit alternatives, including uncertainty and exact information cut-offs, are in [PRIOR_FORMS.md](PRIOR_FORMS.md). Prior-only validation should score a future quote as a noisy match-level observation, not pretend reconstructed club ratings are error-free. [proposal derived from §2]

## 5. Between-season dynamics: variance versus mean movement

### Established: models differ on this boundary

- Koopman–Lit explicitly index consecutive active football weeks: the final week of one season and first of the next are adjacent, and summer/winter breaks are **not accounted for**. Do not cite this model as evidence that a summer macro step was already handled. [KL12]
- Ridall uses a distinct between-season forgetting factor for surviving clubs and separate promoted-club priors. Multiplying Gamma shape and rate by `ω_b` preserves the mean and multiplies variance by `1/ω_b`; that is uncertainty inflation, not mean reversion. Hyperparameters are trained before the online test period. [RI8; RI11; Gamma moment derivation]
- Egidi gives seasonal Gaussian evolution around the previous seasonal effect plus a mean component. Its discussion notes that summer player/manager transfers are not accounted for in its historical extrapolation. These are alternative dynamics assumptions, not measured Scottish summer variance. [EG, §3.2 eq.(7), §4.4]

### Suggestive covariates, not causal summer coefficients

The author-hosted study *Does Stability Foster Team Performance?* reports that more stable EPL/Serie A squads perform better on several measures, after controlling for factors including previous performance and wealth; manager stability has a weaker independent association. Its abstract supplies an observational motivation to inspect lineup retention, not a log-rate innovation variance or a causal recruitment effect. Do not import its coefficient or assume manager-change timestamps are available here. [ST, abstract]

### Proposed discriminating tests

Compare non-movers' end-to-start market residual dispersion with **equally constructed within-season gaps**, subtracting/propagating rating-estimation noise. Report means separately from dispersion and q separately from s. A large extra variance supports F4; a persistent signed mover offset supports F1/F2/F3 instead; a stale-information/refit-age pattern supports F6. Apparent boundary changes in raw odds can also be opponent or μ/γ changes, which the §2 decomposition must remove. [proposals from RI8/RI11, KL12 and §2 derivation]

Use retained-minute/lineup-overlap groups as explanatory diagnostics; next-season lineups are not known in the previous season and therefore cannot enter the stipulated earlier-seasons-only prior. Diagnose genuine mean changes and uncertainty changes separately before deciding whether to add mean reversion, a summer shock or both. [ST; project filtration rule; proposed EDA Q11]

## Source register and search limits

### Verified library pages (source-PDF numbering)

- **[DC7]** `dixon1997-modelling-association-football-scores`, p.7 — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, §4.2/Table 3.
- **[OW6]** `owen2011-dynamic-bayesian-forecasting-models-football`, pp.6–7 (opening of p.7) — *Dynamic Bayesian forecasting models of football match outcomes with estimation of the evolution variance parameter*, initial means and Scottish entrant construction.
- **[KL12]** `koopman2013-dynamic-bivariate-poisson-model-analysing`, p.12 — *A Dynamic Bivariate Poisson Model for Analysing and Forecasting Match Results in the English Premier League*, §3.2. This is the library's 2013 version; no claim that the date matches the final journal version.
- **[RI3], [RI8], [RI11], [RI12]** `ridall2024-bayesian-state-space-models-modelling`, respectively pp.3, 8, 11, 12 — *Bayesian state-space models for the modelling and prediction of the results of English Premier League football*. Table 2's converted column headers repeat symbols; no numerical forgetting factor is assigned from that ambiguous header.
- **[BB6–7]** `baio2010-bayesian-hierarchical-model-prediction-football`, pp.6–7 — *Bayesian hierarchical model for the prediction of football results*, §4.

### Verified web sources (relevant sections/abstracts only)

- **[EG]** Egidi, Pauli & Torelli, *Combining historical data and bookmakers' odds in modelling football scores*, [arXiv v1 HTML](https://arxiv.org/html/1802.08848v1), §§3.1–3.2, 4.2, 4.4; [abstract](https://arxiv.org/abs/1802.08848).
- **[HA]** Hvattum & Arntzen, *Using ELO ratings for match result prediction in association football*, [publisher-supplied abstract via RePEc](https://ideas.repec.org/a/eee/intfor/v26yi3p460-470.html), 2010. Abstract only; detailed initialization is sourced separately below.
- **[AH]** Arntzen & Hvattum, *Supplementary material for “Predicting match outcomes in association football using team ratings and player ratings”*, [PDF](https://www.statmod.org/smij/Vol21/Iss5/Arntzen/Supplement.pdf), §A.1, PDF pp.2–3.
- **[PI]** Constantinou & Fenton, *Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries*, [author-hosted pre-publication draft](http://www.constantinou.info/downloads/papers/pi-ratings.pdf), §§2, 4.2; published citation on draft: JQAS 2013, 9(1), 37–50.
- **[SA]** SinceAWin, *Elo Match Rating System*, [method page](https://sinceawin.com/elo-match-rating-system), 26 April 2014, “Relegation And Promotion”. Practitioner source, not peer-reviewed.
- **[ZL]** Zeileis, Leitner & Hornik, *Probabilistic forecasts for the 2018 FIFA World Cup based on the bookmaker consensus model*, [working-paper PDF](https://www2.uibk.ac.at/downloads/c4041030/wpaper/2018-09.pdf), abstract/§1.
- **[WM]** Winkelmann et al., *Are Betting Markets Inefficient? Evidence From Simulations and Real Data*, [publisher-supplied abstract via RePEc](https://ideas.repec.org/a/sae/jospec/v25y2024i1p54-97.html), 2024. Abstract only; publisher and earlier SSRN pages blocked retrieval.
- **[BS]** Bocconi Students Investment Club, *Odds at Play: Testing Efficiency in the Premier League and Serie A*, [article](https://bsic.it/odds-at-play-testing-efficiency-in-the-premier-league-and-serie-a/), “Data & Methodology” and “Checking for Profitable Strategies”. Weak evidence, retained to show the limitations of direct mover-bias claims.
- **[ST]** *Does Stability Foster Team Performance? A European Football (Soccer) Inquiry*, [author-hosted manuscript](https://www.professormarkvanvugt.com/images/files/DoesStabilityFosterTeamPerformance.pdf), abstract (PDF p.2). Bibliographic year/authorship not independently established here; treated as suggestive only.

**Searched:** library `promot|relegat`, season/summer/evolution variance, mixture/overshrinkage, and catalogue author/title/abstract queries for football, bookmaker/odds, ratings, Koopman and Rue; web leads for the named football models, ClubElo, pi-ratings, promoted-club betting biases, squad stability and bookmaker consensus. No full books, theses or whole papers were read; no exhaustive claim about unconverted books is made. Rue–Salvesen was located through references but not independently retrieved, so no primary empirical claim relies on it. General state-space books were not opened. No betdb queries, Julia, fits or implementation inspection occurred. Unverified search snippets were not used as evidence.
