# 015 — Resolve Bayesian shrinkage compression and favorite underpricing in Scottish Lower

| Field | Value |
|---|---|
| ID | 015 |
| Title | Resolve Bayesian shrinkage compression and favorite underpricing in Scottish Lower |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-19 |
| Updated | 2026-09-20 |
| Related Files / Commits / PRs | `experiments/scottish_lower/09_bayesian_shrinkage_decompression/WORK_PACKAGE_PROMPT.md` |

## Context & Problem Statement

In production models (`m12_joint_hybrid_synergy` and `m05_joint_production_wealth_grw`), zero-mean Gaussian shrinkage priors on team attack/defense innovations, Richards sigmoid saturation on squad market values, and small 10-team sample sizes create an aggressive shrinkage compression effect. The maximum net latent supremacy is bounded at ~0.79 log-rate difference, creating an empirical ceiling where win probabilities rarely exceed ~60% even when closing Betfair odds imply 67-71% dominance (e.g. Ross County vs Cove, Hamilton vs Queen of the South on 2026-09-19). This underconfidence forces residual probability mass onto underdogs, generating phantom positive Kelly edges (+14% to +24%) on extreme longshots (odds >4.0) that consume substantial bankroll and cause portfolio drawdowns.

## Acceptance Criteria

- [ ] Stage 1 EDA (`r09_eda_latent_compression.jl`) completed across all 43 folds for `m12` and `m05`, mathematically documenting the supremacy slope, tail probability ceiling, parameter variances, and Kelly longshot exposure in `EDA_REPORT.md`.
- [ ] Stage 2 candidate models implemented in `l09_decompression_models.jl`: team-level dynamic variance heterogeneity ($\sigma_{i}$), hierarchical finishing factor ($\kappa_{\text{team}}$), uncompressed wealth & pedigree covariates, and fatter-tailed dynamic innovations.
- [ ] Single-fold verification ladder (`r09_smoke.jl`) passes all 8 gates (AD tape compilation, 0 divergences, $\hat{R} < 1.05$, ESS > 300, score-grid pricing, storage roundtrip).
- [ ] 40-fold walk-forward grid (`r09_production_grid.jl`) executed on `mcmc-beast` in detached tmux session across seasons 24/25 and 25/26 (710 matches).
- [ ] Out-of-sample evaluation (`r09_evaluate.jl`) and portfolio simulation (`r09_portfolio.jl`) demonstrate eliminated favorite underpricing (empirical supremacy slope $\ge 0.85$), improved Tail Brier Score, and curtailed longshot Kelly drawdown.

## Ideas & Candidate Solutions

- **Team-level dynamic heterogeneity**: Hierarchical team-specific GRW variances $\sigma_{i, a}, \sigma_{i, d} \sim \text{LogNormal}$ and team-specific home advantage $h_i$.
- **Hierarchical finishing factor $\kappa_{\text{team}}$**: Allow clinical/dominant clubs to sustain conversion multipliers above league average with zero-sum identification.
- **Uncompressed financial covariates**: Direct log-ratio wealth and division pedigree/relegation indicators instead of squashed Richards sigmoid.
- **Fatter-tailed dynamic priors**: Student-$t$ or Horseshoe innovations on attack/defense ratings to prevent shrinking genuine outliers.
- **Calibrator tail reform**: Correct `InverseGaussianLaw` asymptote so extreme discrepancies do not default to 100% model belief.

## Work Log & Progress

- [2026-09-19 @antigravity] Diagnosed probability ceiling and phantom longshot Kelly edge on 2026-09-19 live slate settlement. Aligned research plan with user via `/grill-me`. Created TODO 015.
- [2026-09-20 @antigravity] Created `experiments/scottish_lower/09_bayesian_shrinkage_decompression/WORK_PACKAGE_PROMPT.md`. Provisioned worktree `/home/james/bet_project/.worktrees/BayesianFootball-shrinkage-decompression` on branch `feat/scottish-lower-shrinkage-decompression`. Handing off autonomous execution to `@pi` (`openai-codex/gpt-6-astra`) in tmux session `agent_pi_shrinkage_decompression`.

## Verification & Findings

Not run yet. Stage 1 EDA to be executed first.
