# WORK PACKAGE PROMPT: Cross-Tier Scottish Cup and Pyramid Hierarchy EDA (TODO 029)

> **Agent Assignment**: Claude Code CLI (`claude-opus-5-5`)  
> **Workspace**: `/home/james/bet_project/.worktrees/BayesianFootball-scottish-cups-eda`  
> **Branch**: `feat/scottish-cross-tier-cups-eda`  
> **Tracking**: `todos/029_cross_tier_scottish_cup_and_pyramid_hierarchy_eda.md`  
> **Output Directory**: `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/`  
> **Context Predecessor**: `todos/027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md` (commit `f82599e1`)

---

## 1. Executive Summary & Objective

In **TODO 027**, an extensive empirical analysis of Scottish club pedigree and full-time status revealed that newly relegated clubs in Scottish League One (e.g. Ross County, Hamilton) trigger a cold-start prior blindfold under standard flat priors $\mathcal{N}(0, \sigma^2)$, manufacturing spurious Kelly edges on longshot underdogs (e.g. Cove Rangers @ 7.20 on 2026-09-19).

TODO 027 concluded with an explicit recommendation:
> *"Prioritize ingesting Scottish Cup fixtures into betdb to provide the necessary cross-tier bridges before fitting an informative tier-prior model."*

Scottish cup competitions have now been ingested into `betdb`:
- **Tournament 73**: Scottish Cup (520 matches)
- **Tournament 982**: Scottish League Cup / Premier Sports Cup (87 matches)
- **Tournament 1520**: Challenge Cup / SPFL Trust Trophy (372 matches)

Combined with the 4 SPFL league divisions (Premiership `54`: 1,230 matches; Championship `55`: 1,065 matches; League One `56`: 1,024 matches; League Two `57`: 1,025 matches), `betdb` now contains over **5,300 Scottish matches**, including **~979 cross-tier head-to-head fixtures** spanning 2021 to 2026.

### Primary Mission:
Conduct an institutional, rigorous statistical and econometric exploratory data analysis (EDA) across the entire Scottish football pyramid using all league and cup competitions. Quantify the latent tier step differentials, test the linearity vs non-linearity (chasm) of the Scottish pyramid, audit closing-line market efficiency in cup mismatches, and deliver calibrated prior distributions $(\mu, \sigma)$ for use in Bayesian hierarchical time-decay models.

---

## 2. Infrastructure, Tooling & Databases

- **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-scottish-cups-eda`
- **Environment**: `.env` is present in the worktree root.
- **Database (`betdb`)**: Connected via `BF_DB_URL` (`archpc:5433`).
- **Python Environment**: System Python 3.13 with `psycopg`, `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn` available.
- **Julia Environment**: System Julia 1.12 with `BayesianFootball`, `DataFrames`, `Dates`, `LibPQ` available.

### Database Tables in `betdb`:
- `sofascore.matches`: `match_id`, `tournament_id`, `season_id`, `home_team`, `away_team`, `home_score`, `away_score`, `start_timestamp`, `round`.
- `sofascore.tournaments`: `tournament_id`, `name`, `sport`, `country`.
- `sofascore.match_odds`: Closing / pre-match bookmaker odds.
- `sofascore.match_statistics`: In-game team statistics (shots, shots on target, possession, corners, big chances).
- `bbc.match_meta` & `bbc.live_text`: Commentary streams for proxy xG extraction.
- `betfair.match_meta`, `betfair.markets`, `betfair.odds_history`: Exchange closing line archives and market metadata.

---

## 3. Methodology & Work Package Requirements

Your work must follow a clean, reproducible multi-script pipeline organized in `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/`:

```
experiments/scotland/02_cross_tier_cups_and_pyramid_eda/
├── README.md                               # Canonical institutional report
├── data/                                   # Extracted tabular data (CSV/Parquet)
├── results/                                # Summary tables, figures, regression outputs
├── r01_extract_scottish_pyramid_dataset.py # Extracts & links all league + cup fixtures
├── r02_descriptive_supremacy_eda.py        # Goal, shot, and pxG supremacy matrices
├── r03_econometric_tier_glms.py            # Poisson / NegBin GLMs & linearity tests
├── r04_dixon_coles_network_ratings.py      # Bradley-Terry / Dixon-Coles pyramid ratings
├── r05_market_efficiency_cup_pricing.py    # De-vigged closing odds vs realized outcomes
└── r06_prior_calibration_recommendations.py# Numerical parameters for Bayesian priors
```

### Stage 1: Data Extraction, Tier Classification & Nuance Disaggregation (`r01_...`)
1. **Fixture Extraction**: Extract all matches across tournaments `[54, 55, 56, 57, 73, 982, 1520]`.
2. **Point-in-Time Tier Assignment**:
   - Determine each club's primary league tier as of the match date:
     - **Tier 1**: Scottish Premiership (`54`)
     - **Tier 2**: Scottish Championship (`55`)
     - **Tier 3**: Scottish League One (`56`)
     - **Tier 4**: Scottish League Two (`57`)
     - **Tier 5+**: Non-SPFL (Highland League, Lowland League, East/West of Scotland, Juniors, Guest Clubs).
3. **Cup Competition Nuances & Segmentation**:
   - **Scottish Cup (`73`)**: Full-strength knockout. Flag Tier 5+ non-league entrants.
   - **Scottish League Cup (`982`)**: Group stages + knockouts. Early season (July) fitness/rotation effects.
   - **Challenge Cup (`1520`)**:
     - **CRITICAL NUANCE**: Explicitly identify and flag **Premiership B-teams / U21s** (e.g. *Celtic B, Rangers B, Hearts B, Aberdeen B, Hibernian B*). Do **not** classify B-teams as Tier 1! Treat them as a distinct category (e.g. `Tier 1-B` or exclude from senior tier estimation).
     - Flag non-Scottish guest clubs (e.g. The New Saints, Bala Town, Cliftonville, Linfield).

### Stage 2: Empirical Supremacy Matrices (`r02_...`)
1. **Tier Differential Matrices**:
   - For every tier delta $\Delta \text{Tier} = \text{Tier}_{\text{away}} - \text{Tier}_{\text{home}} \in \{-4, -3, -2, -1, 0, 1, 2, 3, 4\}$:
     - Sample size ($N$ matches).
     - Home win %, Draw %, Away win %.
     - Mean goal difference ($\bar{Y}_H - \bar{Y}_A$) and total goals ($\bar{Y}_H + \bar{Y}_A$).
     - Shot difference ($\Delta \text{Shots}$), shots on target difference ($\Delta \text{SoT}$).
     - BBC proxy xG differential ($\Delta \text{pxG}$) where commentary exists.
2. **Segmented vs Pooled Comparison**:
   - Contrast cross-tier margins in Scottish Cup (`73`) vs League Cup (`982`) vs Challenge Cup (`1520`).
   - Measure whether Challenge Cup results exhibit lower supremacy differentials due to squad rotation.

### Stage 3: Econometric & Statistical Modeling (`r03_...`, `r04_...`)
1. **Latent Tier Step GLMs**:
   - Fit Poisson and Negative Binomial goal models:
     $$\log \mu_{ij} = \beta_0 + \beta_{\text{home}} \cdot \text{Home}_{ij} + \sum_{k=1}^4 \tau_k \cdot \text{Tier}_{ij}^{(k)} + \alpha_i - \beta_j$$
   - Estimate the latent tier step parameters $\tau_k$ with clustered standard errors by team/season.
2. **Hypothesis Testing for Pyramid Linearity**:
   - Formally test the null hypothesis of uniform step size:
     $$H_0: \tau_1 - \tau_2 = \tau_2 - \tau_3 = \tau_3 - \tau_4$$
   - Wald test / Likelihood Ratio Test comparing the linear tier model against an unconstrained tier model.
   - Does the Premiership-to-Championship step represent a distinct "financial/sporting chasm" compared to Championship-to-League One or League One-to-League Two?
3. **Network Rating Model (Dixon-Coles / Bradley-Terry)**:
   - Fit a unified Dixon-Coles model across all ~5,300 matches.
   - Extract team attack $\alpha_i$ and defense $\beta_i$ ratings.
   - Examine how the distribution of ratings overlaps between adjacent tiers (e.g. top of Championship vs bottom of Premiership, top of League One vs bottom of Championship).

### Stage 4: Market Efficiency & Pricing Audit (`r05_...`)
1. **Market Implied Supremacy**:
   - Invert closing 1X2 market odds (using Shin or multiplicative de-vigging) to extract market-implied goal supremacy $\Delta \lambda_{\text{mkt}}$.
2. **Realized Outcome vs Market Expectation**:
   - Group by tier matchup (e.g. T1 vs T2, T1 vs T3, T2 vs T3, T3 vs T4).
   - Compute the market pricing error: $\text{Error} = (\text{Goals}_H - \text{Goals}_A) - \Delta \lambda_{\text{mkt}}$.
   - Test for Favorite-Longshot Bias: Does the market systematically overprice higher-division favorites or lower-division underdogs in cup mismatches?

### Stage 5: Prior Calibration for Bayesian Models (`r06_...`)
Translate the empirical findings into concrete numerical prior specifications for the L1 Bayesian time-decay models being developed in TODO 028:
1. **Option A (Hierarchical Tier Steps)**:
   - Prior distribution for tier steps $\tau_{\text{tier}}$: Recommended prior family (e.g. $\text{HalfNormal}(\sigma)$ vs $\text{TruncatedNormal}(\mu, \sigma)$) and hyperparameter values.
2. **Option B (Relegated Structural Prior Offset)**:
   - When a relegated club drops from Tier $k-1$ to Tier $k$ with $<5$ matches observed, what is the empirical mean and standard deviation of its true attack advantage $\Delta \alpha$?
   - Is +0.90 (used in preliminary tests) empirically justified, or does the data indicate a different value (e.g. +0.65 or +0.75)?

---

## 4. Deliverables & Acceptance Standards

1. **Scripts & Data**: All analysis scripts (`r01` to `r06`) and intermediate datasets in `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/`.
2. **Comprehensive Report (`README.md`)**:
   - Executive Summary with key findings.
   - Data provenance and sample breakdown table across all 7 tournaments.
   - Empirical supremacy matrix table (Home/Draw/Away %, Goal Diff, Shot Diff, pxG Diff).
   - GLM regression table (coefficients, SEs, p-values, tier steps).
   - Linearity test statistics (Wald/LRT $\chi^2$, p-value).
   - Market pricing audit and underdog bias analysis.
   - Concrete mathematical prior recommendations for Option A and Option B.
3. **Repository Bookkeeping**:
   - Update `todos/029_cross_tier_scottish_cup_and_pyramid_hierarchy_eda.md` work log and findings.
   - Verify `./scripts/todo.sh check` passes cleanly.

Commit your work with a clear descriptive message: `feat(cross-tier-eda): deliver Scottish cup pyramid hierarchy and tier prior EDA (TODO 029)`.
