# 025 — Prototype Decoupled Generative xG-Primary Model with Subordinate Goals

| Field | Value |
|---|---|
| ID | 025 |
| Title | Prototype Decoupled Generative xG-Primary Model with Subordinate Goals |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-22 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `src/models/pregame/observations/`, `current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`, `experiments/scottish_lower/12_decoupled_generative_xg/` |

## Context & Problem Statement

Standard Bayesian football models (including Gen 3 `JointGammaPoissonObservation`) treat discrete goal counts ($0, 1, 2, \dots$) as the primary truth, tying proxy xG into the same tightly regularized latent team rating $\mu_s$. Because goals have high Poisson noise and small sample sizes, hierarchical shrinkage contracts team ratings toward the league mean, causing the 1.41–1.66 under-scaling slope on favourites.

Option 2 decouples the generative process into two sequential physical layers:
1. **Primary Chance Creation Layer (Continuous)**: A team's true capability is its rate of chance creation and concession. The latent ratings $\alpha_{\text{xg}, i}, \beta_{\text{xg}, i}$ directly drive the proxy-xG intensity:
   $$\log \mu_{\text{xg}, h} = \mu_{\text{xg}} + \gamma_{\text{home}} + \alpha_{\text{xg}, h} + \beta_{\text{xg}, a}$$
   $$\text{pxg}_h \sim \text{Gamma}(\nu, \mu_{\text{xg}, h} / \nu)$$
   Because proxy xG is continuous and provides dense match-by-match feedback, the priors on team rating innovation variance can be significantly wider ($\sigma \approx 0.15\text{--}0.20$), allowing dominant teams to express strong supremacy without being suppressed by goal-level shrinkage.
2. **Subordinate Goal Realization Layer (Discrete)**: Goals are realized conditionally on expected chances, subject to finishing/conversion:
   $$y_h \sim \text{Poisson}(\lambda_h), \quad \lambda_h = \kappa \cdot \mu_{\text{xg}, h} \cdot \exp(\xi_h)$$
   where $\kappa$ is the finishing factor. We evaluate two structural variants for $\kappa$:
   - **Shared League $\kappa$ (`m03`)**: A global league-wide conversion scalar ($\log \kappa \sim \mathcal{N}(0, 0.20)$).
   - **Hierarchical Team $\kappa$ (`m04`)**: Team-specific finishing skill $\kappa_i = \kappa \exp(\sigma_\kappa \tilde{\kappa}_i)$ with $\sigma_\kappa \sim \text{truncated}(\mathcal{N}(0, 0.10), 0, \infty)$ and $\sum_i \tilde{\kappa}_i = 0$.

This architecture breaks the Bayesian shrinkage bottleneck: team supremacy is driven by chance generation (where data is rich and favourites stand out clearly), while goal likelihood ensures scoreline compatibility for betting markets.

Primary scope: Scottish Lower (tournaments 56/57, seasons 24/25 + 25/26, 40-fold walk-forward grid, 710 matches).

## Acceptance Criteria

- [ ] **Stage 0 (Mathematical Formulation & Likelihood Architecture)**:
  - Implement decoupled chance-primary generative architecture in `experiments/scottish_lower/12_decoupled_generative_xg/l12_loader.jl`.
  - Ensure zero allocations in ReverseDiff gradient evaluation with compiled tapes.
  - Implement score-grid integration kernels converting $(\mu_{\text{xg}}, \kappa)$ into exact match scoreline probability matrices ($12 \times 12$).
- [ ] **Stage 1 (Smoke Gate, Folds 1/20/40)**:
  - Run 4 chains $\times$ 400 warmup + 400 draws on folds 1, 20, 40 across all arms:
    - `m01_poisson_time_decay` (control 1)
    - `m02_joint_gamma_poisson` (control 2, compressed benchmark)
    - `m03_funnel_shared_kappa` (candidate 1, shared league conversion)
    - `m04_funnel_hierarchical_kappa` (candidate 2, hierarchical team finishing)
  - Verify zero NUTS divergences; $\hat{R} \le 1.05$; bulk/tail ESS $\ge 200$.
- [ ] **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
  - Benchmark across all 40 folds (710 fixtures) on `mcmc-beast`.
  - Persist runs to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_decoupled_xg`.
- [ ] **Stage 3 (Evaluation & Market Comparison)**:
  - Measure supremacy slope vs Betfair close; verify whether supremacy decompresses towards 1.00.
  - Compare shared $\kappa$ (`m03`) vs hierarchical $\kappa$ (`m04`) on proper scores (LogLoss, CRPS, RPS, ECE) and portfolio returns.
  - Full portfolio simulation on the common tradeable panel with Baker-McHale shrinkage and FlatTrust policy.
- [ ] **Stage 4 (Findings Report)**:
  - Deliver comprehensive report in `experiments/scottish_lower/12_decoupled_generative_xg/README.md`.

## Ideas & Candidate Solutions

- **Gamma vs Log-Normal Chance Likelihood**: Compare $\text{pxg} \sim \text{Gamma}(\nu, \mu/\nu)$ vs $\log(\text{pxg} + \epsilon) \sim \mathcal{N}(\log \mu, \sigma^2)$. Log-Normal has simpler conjugate properties, while Gamma preserves the non-negative support naturally.
- **Finishing Variance Structure**: Compare shared league $\kappa$ vs hierarchical team finishing $\kappa_i = \kappa \exp(\delta_i)$. In lower leagues, team finishing skill is known to regress heavily toward zero.

## Work Log & Progress

- [2026-09-22 @antigravity] Structured task specification for decoupled chance-primary generative architecture. Set status IN_PROGRESS, allocated to `@pi` using model `openai-codex/gpt-5.6-sol` in dedicated worktree `.worktrees/BayesianFootball-decoupled-xg-funnel`.

## Verification & Findings

*(To be filled upon completion of experimental stages)*
