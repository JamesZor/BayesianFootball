# 027 — EDA: Scottish club pedigree, full-time status, and tier priors

| Field | Value |
|---|---|
| ID | 027 |
| Title | EDA: Scottish club pedigree, full-time status, and tier priors |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-23 |
| Updated | 2026-09-23 |
| Related Files / Commits / PRs | `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/` |

## Context & Problem Statement

Current production and prototype Scottish Lower models (Tournaments 56 [League 1] and 57 [League 2]) train on a 2-to-3 season rolling window in isolation from the upper tiers (Tournaments 54 [Premiership] and 55 [Championship]). Under zero-mean Gaussian shrinkage priors ($\alpha_i \sim \mathcal{N}(0, \sigma^2)$), all clubs start with an identical league-average talent expectation ($0.0$). 

When clubs transition between tiers—especially full-time professional clubs dropping into lower leagues (e.g. Ross County, Hamilton, Falkirk, Dunfermline, Queen of the South, Inverness)—the model suffers from a severe "cold-start" transition lag. The model has zero awareness of whether a club operates as Full-Time Professional or Part-Time Semi-Professional, nor does it know division pedigree. 

In live MatchDay operations (e.g. 2026-09-19 Ross County vs Cove, Hamilton vs Queen of the South), this creates severe favourite underpricing (~50-54% model win probability vs 67-71% market closing), forcing residual probability onto underdogs and generating toxic phantom Kelly bets.

This research task conducts an empirical and statistical exploratory data analysis (EDA) across all 4 SPFL tiers to quantify empirical tier gaps, measure within-tier FT vs PT performance deltas, analyze relegation transition lag, and formulate concrete mathematical priors and covariate architectures.

## Acceptance Criteria

- [ ] Comprehensive research report delivered in `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/README.md`.
- [ ] Historical SPFL operational status dataset constructed (Full-Time, Part-Time, Hybrid) across all 42 SPFL clubs for seasons 21/22 through 26/27 using web research and cross-referenced against `betdb`.
- [ ] Empirical tier supremacy gaps quantified across all 4 SPFL tiers (Premiership 54, Championship 55, League 1 56, League 2 57) using goal differentials, proxy xG, and closing Betfair market implied probabilities.
- [ ] Within-tier FT vs PT performance delta quantified (specifically in League One where both coexist) across goal rates, proxy xG, and Betfair supremacy.
- [ ] Relegation / transition lag documented: Performance and market pricing trajectory for the first 10 matches of relegated/promoted clubs.
- [ ] Mathematical formulation of candidate solutions drafted with mock Julia implementations:
  1. Informative prior on team ratings ($\alpha_i, \beta_i \sim \mathcal{N}(\mu_{\text{tier}}, \sigma^2)$) with zero-sum identifiability.
  2. Linear covariate in the predictor ($\eta = \dots + w_{\text{tier}} \Delta\text{Tier} + w_{\text{status}} \Delta\text{Status}$) preserving ReverseDiff zero-allocation tape compilation.
- [ ] Strict compute discipline: NO large MCMC sampling grids. Compute restricted to SQL queries, descriptive statistics, OLS/GLM regressions, and web research.

## Ideas & Candidate Solutions

- **Tier-Anchored Hierarchical Priors**: Instead of centering $\alpha_i$ at 0.0, center team priors at their division or pedigree tier mean ($\mu_{\text{tier}}$) with hierarchical shrinkage toward a national Scottish baseline.
- **Full-Time vs Part-Time Status Covariate**: An explicit categorical or binary indicator in the linear predictor with a positive prior on supremacy.
- **Continuous Squad Market Value Calibration**: Evaluating whether uncompressed Transfermarkt/Sofascore player valuations accurately proxy full-time professional status.
- **Cross-Tier Cup Bridge Data**: Investigating whether domestic cups (Challenge Cup, Scottish League Cup) provide direct bridge observations between tiers.

## Work Log & Progress

- [2026-09-23 @antigravity] Conducted design interview via `/grill-me` with human user. Agreed on scope: all 4 SPFL tiers, combined web + data status sourcing, comprehensive EDA report, strict restriction against heavy MCMC grids. Created worktree `/home/james/bet_project/.worktrees/BayesianFootball-pedigree-fulltime-eda` on branch `feat/scottish-pedigree-fulltime-eda`. Allocated TODO 027.

## Verification & Findings

Not run yet. Record empirical tier gaps, within-tier FT/PT performance deltas, transition lag statistics, and candidate mathematical specifications in the EDA deliverable.

