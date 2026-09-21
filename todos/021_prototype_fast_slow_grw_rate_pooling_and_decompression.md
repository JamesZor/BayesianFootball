# 021 — Prototype fast-slow GRW rate pooling and decompression

| Field | Value |
|---|---|
| ID | 021 |
| Title | Prototype fast-slow GRW rate pooling and decompression |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-21 |
| Updated | 2026-09-21 |
| Related Files / Commits / PRs | `current_development/fast_slow_grw/WORK_PACKAGE_PROMPT.md` |

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

- [ ] Dedicated worktree `/home/james/bet_project/.worktrees/BayesianFootball-fast-slow-grw` on branch `feat/scottish-lower-fast-slow-grw-poc` provisioned.
- [ ] Prototype loader `l01_fast_slow_grw_loader.jl` implemented with clean minimal Poisson GRW architectures:
  - `m01_tight`: baseline `MultiScaleGRW` with standard priors.
  - `m02_loose_var`: `MultiScaleGRW` with $\sim 2.5\times$ scaled $\sigma_0$ prior scale.
  - `m03_loose_tdist`: `MultiScaleGRW` with heavy-tailed Student-$t$ ($z_0 \sim \text{TDist}(4.0)$).
- [ ] Stage 1 smoke test (`r01_fast_slow_smoke.jl`) passes on 3–5 folds, confirming 0 divergences, R̂ < 1.05, ESS > 200, and verifies supremacy slope expansion on the loose models.
- [ ] Stage 2 production grid (`r02_fast_slow_production_grid.jl`) executed on `mcmc-beast` across the 40-fold Scottish Lower walk-forward cohort (710 matches).
- [ ] Geometric Rate Pooling evaluation script (`r03_fast_slow_evaluation_and_blend.jl`) sweeps $w \in [0.0, 1.0]$, reporting the 6 headline metrics (Supremacy Slope, Capital $\le 1.8$, Capital $\ge 4.0$, Max DD, Sharpe, Flat ROI), and compares rate pooling against probability pooling.
- [ ] Formal research findings report `FAST_SLOW_GRW_REPORT.md` documented with tables and CSV outputs in `results/`.

## Ideas & Candidate Solutions

- **Geometric Rate Pooling vs Linear Probability Mixtures**: Rate pooling in $\log \lambda$ space preserves bivariate Poisson / copula score-grid coherence and scales supremacy linearly: $\text{Supremacy}_{\text{blend}} = (1-w)\text{Supremacy}_{\text{tight}} + w \text{Supremacy}_{\text{loose}}$.
- **Variance Expansion vs Heavy-Tailed Innovations**: Test whether Gaussian variance expansion ($\sigma_0 \sim \text{Gamma}(2, 0.15)$) or heavy-tailed Student-$t$ ($z_0 \sim \text{TDist}(4)$) gives cleaner MCMC convergence while liberating the favourites.
- **Pure Poisson Baseline First**: Eliminate all confounding feature effects (RAPM lineups, wealth curves) to isolate the exact impact of team latent shrinkage on decompression.

## Work Log & Progress

- [2026-09-21 @antigravity] Interviewed user via `/grill-me`. Aligned on fast/slow dual GRW concept, geometric rate pooling, minimal Poisson GRW baseline, and two loose candidates (variance scaling and Student-$t$). Allocated TODO 021. Assigned to `@claude`.

## Verification & Findings

Not run yet.
