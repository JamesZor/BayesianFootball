# 025 — Prototype Decoupled Generative xG-Primary Model with Subordinate Goals

| Field | Value |
|---|---|
| ID | 025 |
| Title | Prototype Decoupled Generative xG-Primary Model with Subordinate Goals |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-22 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `src/models/pregame/observations/`, `current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md` |

## Context & Problem Statement

Standard Bayesian football models (including Gen 3 `JointGammaPoissonObservation`) treat discrete goal counts ($0, 1, 2, \dots$) as the primary truth, tying proxy xG into the same tightly regularized latent team rating $\mu_s$. Because goals have high Poisson noise and small sample sizes, hierarchical shrinkage contracts team ratings toward the league mean, causing the 1.41–1.66 under-scaling slope on favourites.

Option 2 decouples the generative process into two sequential physical layers:
1. **Primary Chance Creation Layer (Continuous)**: A team's true capability is its rate of chance creation and concession. The latent ratings $\alpha_{\text{xg}, i}, \beta_{\text{xg}, i}$ directly drive the proxy-xG intensity:
   $$\log \mu_{\text{xg}, h} = \mu_{\text{xg}} + \gamma_{\text{home}} + \alpha_{\text{xg}, h} + \beta_{\text{xg}, a}$$
   $$\text{pxg}_h \sim \text{Gamma}(\nu, \mu_{\text{xg}, h} / \nu)$$
   Because proxy xG is continuous and provides dense match-by-match feedback, the priors on team rating innovation variance can be significantly wider ($\sigma \approx 0.15\text{--}0.20$), allowing dominant teams to express strong supremacy without being suppressed by goal-level shrinkage.
2. **Subordinate Goal Realization Layer (Discrete)**: Goals are realized conditionally on expected chances, subject to shot conversion noise $\xi$:
   $$\log \lambda_{h} = \log \mu_{\text{xg}, h} + \xi_h, \quad \xi_h \sim \mathcal{N}(0, \sigma_{\text{fin}}^2)$$
   $$y_h \sim \text{Poisson}(\lambda_h)$$
   where $\sigma_{\text{fin}} \approx 0.05$ restricts finishing noise to a modest variation around chance creation.

This architecture breaks the Bayesian shrinkage bottleneck: team supremacy is driven by chance generation (where data is rich and favourites stand out clearly), while goal likelihood ensures scoreline compatibility for betting markets.

Primary scope: Scottish Lower (tournaments 56/57, seasons 24/25 + 25/26, 40-fold walk-forward grid, 710 matches).

## Acceptance Criteria

- [ ] **Stage 0 (Mathematical Formulation & Likelihood Architecture)**:
  - Implement `DecoupledGenerativeXGObservation` or composite model builder in `current_development/decoupled_xg_generative/`.
  - Ensure zero allocations in ReverseDiff gradient evaluation with compiled tapes.
  - Implement score-grid integration kernels converting $(\mu_{\text{xg}}, \sigma_{\text{fin}})$ into exact match scoreline probability matrices ($12 \times 12$).
- [ ] **Stage 1 (Smoke Gate, Folds 1/20/40)**:
  - Run 4 chains $\times$ 400 warmup + 400 draws on folds 1, 20, 40.
  - Verify zero NUTS divergences; $\hat{R} \le 1.05$; bulk/tail ESS $\ge 200$.
  - Confirm finishing variance $\sigma_{\text{fin}}$ is well-identified and does not collapse or blow up.
- [ ] **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
  - Benchmark against `m01_poisson_time_decay` and `m02_joint_gamma_poisson` across all 710 fixtures.
  - Persist runs to PostgreSQL `mcmc_experiments`.
- [ ] **Stage 3 (Evaluation & Market Comparison)**:
  - Measure supremacy slope vs Betfair close; verify reduction of favourite compression.
  - Measure proper scores (LogLoss, CRPS, RPS, ECE) on 1X2 and totals.
  - Portfolio backtest with Baker-McHale shrinkage and FlatTrust policy.
- [ ] **Stage 4 (Findings Report)**:
  - Deliver comprehensive report in `experiments/scottish_lower/12_decoupled_generative_xg/README.md`.

## Ideas & Candidate Solutions

- **Gamma vs Log-Normal Chance Likelihood**: Compare $\text{pxg} \sim \text{Gamma}(\nu, \mu/\nu)$ vs $\log(\text{pxg} + \epsilon) \sim \mathcal{N}(\log \mu, \sigma^2)$. Log-Normal has simpler conjugate properties, while Gamma preserves the non-negative support naturally.
- **Finishing Variance Structure**: Finishing parameter $\xi_h$ can be match-specific (shot-level luck) or carry a hierarchical team finishing skill $\theta_{\text{finish}, i} \sim \mathcal{N}(0, \tau^2)$. In lower leagues, team finishing skill is known to regress heavily toward zero.

## Work Log & Progress

- [2026-09-22 @antigravity] Structured task specification for decoupled chance-primary generative architecture. Set status BACKLOG for future dispatch.

## Verification & Findings

*(To be filled upon completion of experimental stages)*
