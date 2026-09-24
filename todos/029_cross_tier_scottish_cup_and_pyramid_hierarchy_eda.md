# 029 — Cross-tier Scottish Cup and pyramid hierarchy EDA

| Field | Value |
|---|---|
| ID | 029 |
| Title | Cross-tier Scottish Cup and pyramid hierarchy EDA |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-24 |
| Updated | 2026-09-24 |
| Related Files / Commits / PRs | `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/` (README, r01–r06, l03, l04, `results/r06_prior_recommendations.json`), `todos/027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md`, `todos/028_cross_tier_scottish_pyramid_and_informative_priors_time_decay_models.md`, commit `f82599e1` |

## Context & Problem Statement

In TODO 027, an extensive EDA on Scottish club pedigree and full-time status revealed that newly relegated clubs in Scottish League One (e.g. Ross County, Hamilton) trigger a cold-start prior blindfold under standard flat priors $\mathcal{N}(0, \sigma^2)$, manufacturing spurious Kelly edges on longshot underdogs (e.g. Cove Rangers @ 7.20 on 2026-09-19).

TODO 027 recommended: *"Prioritize ingesting Scottish Cup fixtures into betdb to provide the necessary cross-tier bridges before fitting an informative tier-prior model."*

Scottish Cup (Tournament 73, 520 matches), Scottish League Cup (Tournament 982, 87 matches), and Challenge Cup (Tournament 1520, 372 matches) have now been ingested into `betdb`, providing ~979 head-to-head fixtures where clubs across different SPFL tiers and non-league divisions play each other.

This task conducts a comprehensive empirical, econometric, and statistical EDA across all Scottish tournaments (54, 55, 56, 57, 73, 982, 1520) using Claude CLI (`claude-opus-5-5`) to quantify the cross-tier pyramid steps, measure goal/shot/pxG supremacy differentials, evaluate market pricing efficiency in cross-tier cup ties, and deliver calibrated prior parameters for Bayesian hierarchical models.

## Acceptance Criteria

- [x] **Data Extraction & Linking**: Extract all ~5,300+ Scottish matches across league (54, 55, 56, 57) and cup (73, 982, 1520) competitions from `betdb`, mapping each club's primary league tier (T1: Premiership, T2: Championship, T3: League One, T4: League Two, T5+: Highland/Lowland/Non-SPFL) at match date.
- [x] **Cup Nuance Disaggregation**: Explicitly identify and isolate Premiership B/U21 teams and cross-border guest clubs in the Challenge Cup (1520) to prevent distortion of first-team tier ratings.
- [x] **Supremacy Matrices**: Compute empirical goal difference, total goals, shot supremacy, BBC proxy xG, and closing market odds across all tier pairings $(\Delta \text{Tier} \in \{-4, \dots, +4\})$.
- [x] **Econometric & Statistical Modeling**:
  - Fit Poisson and Negative Binomial GLMs estimating latent tier step parameters $\tau_k$ with clustered standard errors.
  - Test the linearity hypothesis ($H_0: \tau_1 - \tau_2 = \tau_2 - \tau_3 = \tau_3 - \tau_4$) against non-linear/chasm alternatives.
  - Fit paired Dixon-Coles / Bradley-Terry ratings across the unified Scottish network.
- [x] **Market Efficiency Audit**: Invert closing odds to obtain market-implied goal supremacy $\Delta \lambda_{\text{mkt}}$ and compare against realized goal differentials and model predictions to audit whether the market has systematic biases in cup mismatches.
- [x] **Prior Calibration Artifacts**: Produce calibrated numerical estimates (means, standard deviations, shrinkage factors) directly usable for Option A (hierarchical tier offsets) and Option B (structural/market priors) in L1 models.
- [x] **Institutional Report**: Deliver a comprehensive `README.md` in `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/` with narrative findings, tables, charts, and actionable model recommendations.

## Ideas & Candidate Solutions

- **Segmented vs Pooled**: Analyze Scottish Cup (73) and League Cup (982) separately from Challenge Cup (1520), then pool first-team fixtures to avoid B-team dilution.
- **Home Advantage Adjustment**: Cup ties are played at home venues without second legs; neutral/replay venues and home advantage must be controlled for in all supremacy estimates.
- **BBC Proxy xG Integration**: Incorporate commentary-derived proxy xG where available to test if cup goal differences reflect genuine chance-creation divergence.

## Work Log & Progress

- [2026-09-24 @antigravity] Scaffolded worktree `/home/james/bet_project/.worktrees/BayesianFootball-scottish-cups-eda` on branch `feat/scottish-cross-tier-cups-eda`. Allocated TODO 029. Initialized experiment directory `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/`. Prepared work package prompt for Claude CLI (`claude-opus-5-5`).
- [2026-09-24 @claude] Executed the work package. At the user's instruction the six-stage pipeline is written in Julia (warm REPL), not the Python the prompt specified: `r01`–`r06` plus loaders `l03_tier_design.jl` (tier GLM design) and `l04_dixon_coles_core.jl` (analytic-gradient Dixon–Coles network), and `run_all.jl`. Fixture universe = `sofascore.events` (16,119 finished fixtures 2008/09–2026/27; primary window 21/22–26/27 = 4,798 fixtures, 411 senior cross-tier ties). B/U21 sides (38) and guest clubs (18) are separate categories. Full rerun reproduces every result table byte-identically in about 3 minutes.

## Verification & Findings

Full report: [`experiments/scotland/02_cross_tier_cups_and_pyramid_eda/README.md`](../experiments/scotland/02_cross_tier_cups_and_pyramid_eda/README.md). Log goal-rate units on the L1 scale (θ = α − β).

- **Pyramid is linear once Celtic/Rangers are club effects.** Primary-window tier steps (Poisson, two-way club-season clustered SEs): T1→T2 0.51 ± 0.12, T2→T3 0.48 ± 0.11, T3→T4 0.44 ± 0.11, T4→non-league 0.44 ± 0.10. H₀ τ₁₂ = τ₂₃ = τ₃₄: Wald χ²₂ = 0.16, p = 0.92 (LRT p = 0.92; NB2 identical). Pooled step 0.473 ± 0.041. With the Old Firm pooled into T1 the first step is 0.80 (p = 0.066); they sit 1.15 ± 0.08 above the rest of the Premiership.
- **Era dependence.** Pre-2020 the rest of the Premiership was only about 0.18 above the Championship and T2→T3 was about 0.70; the long-window (2008–26) linearity test rejects (p = 0.011). Post-2020 steps are 0.54 / 0.45 / 0.48 / 0.47.
- **Goal levels are flat across the SPFL** (g_T within ±0.06); home advantage is the same in cups (h_cup = −0.02 ± 0.05).
- **Challenge Cup compresses margins** by 0.31 ± 0.13 goals vs the Scottish Cup at equal gap and venue (2008–26, p = 0.016). B-teams play at T4/T5 level.
- **Dixon–Coles network** (4,798 fixtures, no tier input): tier-mean steps 0.49 / 0.54 / 0.42 / 0.54; adjacent-tier AUC 0.81–0.90; a third of Championship club-seasons out-rate the Premiership's bottom club.
- **Market:** the closing 1X2 prices cross-tier goal margins correctly (error +0.04 ± 0.09, MZ slope 0.95 ± 0.09, N = 403). Walk-forward tier GLM and DC network add no information (encompassing p > 0.8; log-loss 0.855 vs 0.798). There is a favourite–longshot bias steeper than in the league: selections under 15% return −52% (league −23%) and the draw −34%; favourites above 60% return +2.6% ± 4% (not significant). The walk-forward DC network without tier offsets compresses cross-tier supremacy.
- **Option A prior:** d_j ~ TruncatedNormal(0.47, 0.16; 0, ∞) per SPFL step, applied 48% to α and 52% to β; Old Firm as team effects; A1 δ_league ≈ 0 (|δ| ≤ 0.07).
- **Option B prior:** relegated into League One α₀ ~ N(+0.12, 0.20²), β₀ ~ N(−0.16, 0.26²) (net +0.28; 26 structural transitions + first-5-game market view). **+0.90 is not supported** (z = 9.3 as α; only 2.8% of relegated clubs exceed it as net θ); +0.65/+0.75 are also rejected. Ross County 26/27 (market θ +1.09) is a 2.5σ₀ outlier, so pair B1 with B2 or the TODO 027 wealth/full-time covariate.
- Limits: League Cup group stage absent from betdb; no Betfair archive for any cup tie (the SofaScore close is used); 90-minute scores.
- Bookkeeping: `./scripts/todo.sh check` passes.
