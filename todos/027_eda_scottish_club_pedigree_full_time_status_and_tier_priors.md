# 027 — EDA: Scottish club pedigree, full-time status, and tier priors

| Field | Value |
|---|---|
| ID | 027 |
| Title | EDA: Scottish club pedigree, full-time status, and tier priors |
| Status | BLOCKED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-23 |
| Updated | 2026-09-23 |
| Related Files / Commits / PRs | `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/` |

## Context & Problem Statement

> Original motivation below contains hypotheses, not established causes. The final report corrects the asserted probability cap, club-status assumptions and live-slate stake interpretation; see **Verification & Findings**.

Current production and prototype Scottish Lower models (Tournaments 56 [League 1] and 57 [League 2]) train on a 2-to-3 season rolling window in isolation from the upper tiers (Tournaments 54 [Premiership] and 55 [Championship]). Under zero-mean Gaussian shrinkage priors ($\alpha_i \sim \mathcal{N}(0, \sigma^2)$), all clubs start with an identical league-average talent expectation ($0.0$). 

When clubs transition between tiers—especially full-time professional clubs dropping into lower leagues (e.g. Ross County, Hamilton, Falkirk, Dunfermline, Queen of the South, Inverness)—the model suffers from a severe "cold-start" transition lag. The model has zero awareness of whether a club operates as Full-Time Professional or Part-Time Semi-Professional, nor does it know division pedigree. 

In live MatchDay operations (e.g. 2026-09-19 Ross County vs Cove, Hamilton vs Queen of the South), this creates severe favourite underpricing (~50-54% model win probability vs 67-71% market closing), forcing residual probability onto underdogs and generating toxic phantom Kelly bets.

This research task conducts an empirical and statistical exploratory data analysis (EDA) across all 4 SPFL tiers to quantify empirical tier gaps, measure within-tier FT vs PT performance deltas, analyze relegation transition lag, and formulate concrete mathematical priors and covariate architectures.

## Acceptance Criteria

- [x] Comprehensive research report delivered in `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/README.md`.
- [ ] Historical SPFL operational status dataset constructed (Full-Time, Part-Time, Hybrid) across all 42 SPFL clubs for seasons 21/22 through 26/27 using web research and cross-referenced against `betdb`.
- [ ] Empirical tier supremacy gaps quantified across all 4 SPFL tiers (Premiership 54, Championship 55, League 1 56, League 2 57) using goal differentials, proxy xG, and closing Betfair market implied probabilities.
- [x] Within-tier FT vs PT performance delta quantified (specifically in League One where both coexist) across goal rates, proxy xG, and Betfair supremacy.
- [ ] Relegation / transition lag documented: Performance and market pricing trajectory for the first 10 matches of relegated/promoted clubs.
- [ ] Mathematical formulation of candidate solutions drafted with mock Julia implementations:
  1. Informative prior on team ratings ($\alpha_i, \beta_i \sim \mathcal{N}(\mu_{\text{tier}}, \sigma^2)$) with zero-sum identifiability.
  2. Linear covariate in the predictor ($\eta = \dots + w_{\text{tier}} \Delta\text{Tier} + w_{\text{status}} \Delta\text{Status}$) preserving ReverseDiff zero-allocation tape compilation.
- [x] Strict compute discipline: NO large MCMC sampling grids. Compute restricted to SQL queries, descriptive statistics, OLS/GLM regressions, and web research.

## Ideas & Candidate Solutions

- **Tier-Anchored Hierarchical Priors**: Instead of centering $\alpha_i$ at 0.0, center team priors at their division or pedigree tier mean ($\mu_{\text{tier}}$) with hierarchical shrinkage toward a national Scottish baseline.
- **Full-Time vs Part-Time Status Covariate**: An explicit categorical or binary indicator in the linear predictor with a positive prior on supremacy.
- **Continuous Squad Market Value Calibration**: Evaluating whether uncompressed Transfermarkt/Sofascore player valuations accurately proxy full-time professional status.
- **Cross-Tier Cup Bridge Data**: Investigating whether domestic cups (Challenge Cup, Scottish League Cup) provide direct bridge observations between tiers.

## Work Log & Progress

- [2026-09-23 @antigravity] Conducted design interview via `/grill-me` with human user. Agreed on scope: all 4 SPFL tiers, combined web + data status sourcing, comprehensive EDA report, strict restriction against heavy MCMC grids. Created worktree `/home/james/bet_project/.worktrees/BayesianFootball-pedigree-fulltime-eda` on branch `feat/scottish-pedigree-fulltime-eda`. Allocated TODO 027.

- [2026-09-23 @pi] Claimed execution in the allocated worktree. Delegated independent evidence-backed club-status research, read-only SQL/EDA, and mathematical/AD design. Treating the work-package causal claims and operational classifications as hypotheses; no MCMC grids authorised or launched.

- [2026-09-23 @pi] Delivered final institutional-style README and reproducible read-only EDA. Extracted 4,068 normal-time fixtures across all four tiers; reconciled BBC proxy coverage to the same universe and verified 46,424 parsed events against Julia kernels. Extended the source register to 20 fetched sources and the panel to 21 verified / 8 inferred / 223 Unknown club-seasons.
- [2026-09-23 @pi] Audited live slate: Cove was home; £20.35 includes two draws plus two underdogs; four-leg filled loss £16.29, whole m12-labelled slate net −£5.09. Historical saved-m12 pricing retains 220 club-side rows / 196 fixtures and 186 market pairs, with exact portfolio-UUID opponent-bet attribution. Added and passed status-ID and draw-pricing regression tests. No new sampling or production code changes.

## Verification & Findings

Final report: `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/README.md`.

- **BLOCKED, not completed:** completing the original criteria requires the remaining verified operational-status census, cross-tier cup bridge ingestion, complete historical fixtures/transition ordinals, immutable live-m12 provenance and production integration/allocation gates. The final report and usable partial artifacts are delivered; absent evidence is not imputed as a result.
- All four Scottish leagues exist in SQL; production lower-league scope is a separate fact. No usable cup bridges were found, so adjacent tier-strength steps remain unestimated. The outcome archive is not complete (e.g. Premiership 198 rather than 228 fixtures per full season).
- Strict FT–PT League One sample: 8 fixtures, FT-oriented goal difference −0.750, win rate25.0% (Wilson95%7.1–59.1%); not a population or causal effect. Inferred sensitivity11fixtures. Zero-sum Gaussian priors do not impose a hard probability ceiling.
- Existing m12 run `928dad3b-ccaf-4909-b6b7-4f1a815e1cab`, portfolio `a7c4c55b-f8d2-416e-ba85-9c7fe9bedc1a`: posterior-draw pricing and paired transition summaries recorded without fitting. Opponent bets against downward-transition archive favourites had positive selected historical P&L, not blanket evidence of toxic losses; no causal drawdown attribution.
- Verification: `r10_status_contract_tests.jl`12/12; `r09_m12_pricing_tests.py`3/3; `r06_verify_bbc_proxy_xg_kernel.jl`parser/masks/numerical parity pass; `r04`compiled/fresh/ForwardDiff and perturbed-gradient gates pass, **2,432bytes/gradient**, not zero allocation; `r07_validate_artifacts.py`membership/source/ledger checks pass. Julia used the already-instantiated main checkout project; no package changes. Final `./scripts/todo.sh check` required before commit.
- Next action: supply/verify missing status sources and cup/fixture histories; then compare flat control, pedigree/status prior, covariate-only and resource-only on identical future held-out cohorts, keeping staking and market instant fixed.

