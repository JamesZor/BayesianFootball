# 029 — Cross-tier Scottish Cup and pyramid hierarchy EDA

| Field | Value |
|---|---|
| ID | 029 |
| Title | Cross-tier Scottish Cup and pyramid hierarchy EDA |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-24 |
| Updated | 2026-09-24 |
| Related Files / Commits / PRs | `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/`, `todos/027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md`, `todos/028_cross_tier_scottish_pyramid_and_informative_priors_time_decay_models.md`, commit `f82599e1` |

## Context & Problem Statement

In TODO 027, an extensive EDA on Scottish club pedigree and full-time status revealed that newly relegated clubs in Scottish League One (e.g. Ross County, Hamilton) trigger a cold-start prior blindfold under standard flat priors $\mathcal{N}(0, \sigma^2)$, manufacturing spurious Kelly edges on longshot underdogs (e.g. Cove Rangers @ 7.20 on 2026-09-19).

TODO 027 recommended: *"Prioritize ingesting Scottish Cup fixtures into betdb to provide the necessary cross-tier bridges before fitting an informative tier-prior model."*

Scottish Cup (Tournament 73, 520 matches), Scottish League Cup (Tournament 982, 87 matches), and Challenge Cup (Tournament 1520, 372 matches) have now been ingested into `betdb`, providing ~979 head-to-head fixtures where clubs across different SPFL tiers and non-league divisions play each other.

This task conducts a comprehensive empirical, econometric, and statistical EDA across all Scottish tournaments (54, 55, 56, 57, 73, 982, 1520) using Claude CLI (`claude-opus-5-5`) to quantify the cross-tier pyramid steps, measure goal/shot/pxG supremacy differentials, evaluate market pricing efficiency in cross-tier cup ties, and deliver calibrated prior parameters for Bayesian hierarchical models.

## Acceptance Criteria

- [ ] **Data Extraction & Linking**: Extract all ~5,300+ Scottish matches across league (54, 55, 56, 57) and cup (73, 982, 1520) competitions from `betdb`, mapping each club's primary league tier (T1: Premiership, T2: Championship, T3: League One, T4: League Two, T5+: Highland/Lowland/Non-SPFL) at match date.
- [ ] **Cup Nuance Disaggregation**: Explicitly identify and isolate Premiership B/U21 teams and cross-border guest clubs in the Challenge Cup (1520) to prevent distortion of first-team tier ratings.
- [ ] **Supremacy Matrices**: Compute empirical goal difference, total goals, shot supremacy, BBC proxy xG, and closing market odds across all tier pairings $(\Delta \text{Tier} \in \{-4, \dots, +4\})$.
- [ ] **Econometric & Statistical Modeling**:
  - Fit Poisson and Negative Binomial GLMs estimating latent tier step parameters $\tau_k$ with clustered standard errors.
  - Test the linearity hypothesis ($H_0: \tau_1 - \tau_2 = \tau_2 - \tau_3 = \tau_3 - \tau_4$) against non-linear/chasm alternatives.
  - Fit paired Dixon-Coles / Bradley-Terry ratings across the unified Scottish network.
- [ ] **Market Efficiency Audit**: Invert closing odds to obtain market-implied goal supremacy $\Delta \lambda_{\text{mkt}}$ and compare against realized goal differentials and model predictions to audit whether the market has systematic biases in cup mismatches.
- [ ] **Prior Calibration Artifacts**: Produce calibrated numerical estimates (means, standard deviations, shrinkage factors) directly usable for Option A (hierarchical tier offsets) and Option B (structural/market priors) in L1 models.
- [ ] **Institutional Report**: Deliver a comprehensive `README.md` in `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/` with narrative findings, tables, charts, and actionable model recommendations.

## Ideas & Candidate Solutions

- **Segmented vs Pooled**: Analyze Scottish Cup (73) and League Cup (982) separately from Challenge Cup (1520), then pool first-team fixtures to avoid B-team dilution.
- **Home Advantage Adjustment**: Cup ties are played at home venues without second legs; neutral/replay venues and home advantage must be controlled for in all supremacy estimates.
- **BBC Proxy xG Integration**: Incorporate commentary-derived proxy xG where available to test if cup goal differences reflect genuine chance-creation divergence.

## Work Log & Progress

- [2026-09-24 @antigravity] Scaffolded worktree `/home/james/bet_project/.worktrees/BayesianFootball-scottish-cups-eda` on branch `feat/scottish-cross-tier-cups-eda`. Allocated TODO 029. Initialized experiment directory `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/`. Prepared work package prompt for Claude CLI (`claude-opus-5-5`).

## Verification & Findings

*Pending agent execution.*
