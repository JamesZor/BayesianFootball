# 021 — Prototype fast-slow GRW rate pooling and decompression

| Field | Value |
|---|---|
| ID | 021 |
| Title | Prototype fast-slow GRW rate pooling and decompression |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-21 |
| Updated | 2026-09-21 |
| Related Files / Commits / PRs | `current_development/fast_slow_grw/FAST_SLOW_GRW_REPORT.md`, `current_development/fast_slow_grw/WORK_PACKAGE_PROMPT.md`, branch `feat/scottish-lower-fast-slow-grw-poc` |

## Context & Problem Statement

Scottish Lower football models suffer from Bayesian shrinkage compression on team quality:
Gaussian shrinkage pulls team latent separation towards parity, compressing favourite win
probabilities to ~50-58% when closing market prices imply 70-85%. Following the empirical
finding that market closing prices want team attack/defence contrast ($\alpha/\beta$) scaled up
by $2.43\times$ while player ratings ($\Delta L$) are already at exact 1.00 market parity, we
design a dual "Fast and Slow" model architecture inspired by the navigation concept of
"handrailing and aiming off":
1. **Tight / Slow Model (The Handrail)**: High shrinkage, long memory, tight Gaussian priors on team innovations. Provides stability, guards downside risk, and prevents overfitting on small samples.
2. **Loose / Fast Model (The Scout / Innovator)**: Relaxed shrinkage ($2.5\times$ prior scale on $\sigma_0$) or heavy-tailed Student-$t$ innovations (`TDist(4.0)`). Decompresses favourite win probabilities towards market realities.
3. **Geometric Rate Pooling ($\lambda$-space)**: $\log \lambda_{\text{blend}} = (1-w)\log \lambda_{\text{tight}} + w \log \lambda_{\text{loose}}$, preserving single score-grid coherence (1X2, Totals, BTTS) without distorting derivative markets.

This task delivers a clean proof-of-concept on minimal Poisson GRW models (zero complex covariates: no player lineups, no squad wealth, no smiles) before extending to two-arm joint observations.

## Acceptance Criteria

- [x] Dedicated worktree `/home/james/bet_project/.worktrees/BayesianFootball-fast-slow-grw` on branch `feat/scottish-lower-fast-slow-grw-poc` provisioned.
- [x] Prototype loader `l01_fast_slow_grw_loader.jl` implemented with clean minimal Poisson GRW architectures:
  - `m01_tight`: baseline `MultiScaleGRW` with standard priors.
  - `m02_loose_var`: `MultiScaleGRW` with $\sim 2.5\times$ scaled $\sigma_0$ prior scale.
  - `m03_loose_tdist`: `MultiScaleGRW` with heavy-tailed Student-$t$ ($z_0 \sim \text{TDist}(4.0)$).
  - `m04_loose_fixed_spread` (added 2026-09-21 at the user's direction): $\sigma_0$ pinned at 0.48.
- [x] Stage 1 smoke test (`r01_fast_slow_smoke.jl`) passes on 3–5 folds, confirming 0 divergences, R̂ < 1.05, ESS > 200, and verifies supremacy slope expansion on the loose models. (Mechanical gates pass 4/4; m04 fails the directional slope gate, recorded.)
- [x] Stage 2 production grid (`r02_fast_slow_production_grid.jl`) executed on `mcmc-beast` across the 40-fold Scottish Lower walk-forward cohort (710 matches).
- [x] Evaluation script (`r03_fast_slow_evaluation_and_blend.jl`) sweeps the combination weight, reporting the 6 headline metrics (Supremacy Slope, Capital $\le 1.8$, Capital $\ge 4.0$, Max DD, Sharpe, Flat ROI). **Amended 2026-09-21 by the user:** the combination is posterior draw concatenation at ratio $\rho \in \{0, .25, .5, .75, 1\}$, not geometric rate pooling; a draw mixture *is* the linear probability pool at fixture level, so no separate rate-vs-probability comparison was run.
- [x] Formal research findings report `FAST_SLOW_GRW_REPORT.md` documented with tables and CSV outputs in `results/`.

## Ideas & Candidate Solutions

- **Geometric Rate Pooling vs Linear Probability Mixtures**: Rate pooling in $\log \lambda$ space preserves bivariate Poisson / copula score-grid coherence and scales supremacy linearly: $\text{Supremacy}_{\text{blend}} = (1-w)\text{Supremacy}_{\text{tight}} + w \text{Supremacy}_{\text{loose}}$.
- **Variance Expansion vs Heavy-Tailed Innovations**: Test whether Gaussian variance expansion ($\sigma_0 \sim \text{Gamma}(2, 0.15)$) or heavy-tailed Student-$t$ ($z_0 \sim \text{TDist}(4)$) gives cleaner MCMC convergence while liberating the favourites.
- **Pure Poisson Baseline First**: Eliminate all confounding feature effects (RAPM lineups, wealth curves) to isolate the exact impact of team latent shrinkage on decompression.

## Work Log & Progress

- [2026-09-21 @antigravity] Interviewed user via `/grill-me`. Aligned on fast/slow dual GRW concept, geometric rate pooling, minimal Poisson GRW baseline, and two loose candidates (variance scaling and Student-$t$). Allocated TODO 021. Assigned to `@claude`.
- [2026-09-21 @claude] Rebuilt the scaffolded loader/runners on the Task 013 `gph_*` machinery (the scaffold called non-existent APIs and had no fold limit); instantiated the beast worktree's environment (Manifest from `BF_grw_player_hybrid`, Distributions 0.25.126). Stage 1 smoke PASS on mechanics.
- [2026-09-21 @claude] User replaced geometric rate pooling with draw concatenation and added m04 (σ₀ = 0.48). Smoke: 4/4 mechanics pass, m04 fails G6. Stage 2 grid PASS 4/4 (runs `2b42d3bf`, `27c8f2f2`, `dafbfe00`, `514c5533` in `fast_slow_grw_scottish_lower`). Stage 3 evaluation and report written.

## Verification & Findings

Full write-up: [`current_development/fast_slow_grw/FAST_SLOW_GRW_REPORT.md`](../current_development/fast_slow_grw/FAST_SLOW_GRW_REPORT.md).

- **Convergence:** 40 folds / 710 fixtures per arm, 0 divergences in 128k transitions, max R̂ 1.0115–1.0141, min bulk ESS 410–732.
- **Decompression: negative.** Supremacy slope vs the inverted Betfair close (596 fixtures): tight 0.399; best arm m02 0.488 (market = 1.0); m03 0.399; m04 0.441 with R² falling 0.44 → 0.34. Favourites priced ≥ 0.70 (n = 18): market 0.763, tight 0.550, best 0.575.
- **Mechanism:** the data identify σ₀ (2.5× prior → posterior 0.192 → 0.209); pinning σ₀ at 0.48 adds spread orthogonal to the market and worsens 1X2 / BTTS log loss.
- **Portfolio:** loose mixtures cut capital at odds ≥ 4.0 from 38% to 32–35%; m04 mixtures raise Sharpe (up to 1.82 vs 1.65) alongside deeper drawdowns and worse log loss, not bootstrapped for significance; m02 lowers Sharpe to 1.45.
- **Derivatives:** mixtures do not distort totals/BTTS (mean P(over 2.5) 0.506–0.514; O/U log loss within 0.0015). 1X2 log loss rises monotonically with the loose share.
- **Recommendation:** do not carry the fast-slow mixture into Phase 2 as a decompression device; use market-anchored generative rate calibration, or a market-directed prior on team levels rather than a wider σ₀.
