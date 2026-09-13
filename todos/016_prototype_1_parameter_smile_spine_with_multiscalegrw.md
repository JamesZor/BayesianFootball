# 016 — Prototype 1-Parameter Smile Spine with MultiScaleGRW

| Field | Value |
|---|---|
| ID | 016 |
| Title | Prototype 1-Parameter Smile Spine with MultiScaleGRW |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-13 |
| Updated | 2026-09-13 |
| Related Files / Commits / PRs | [current_development/grw_smile_spine/](../current_development/grw_smile_spine/); [docs/tickets/T010-postgres-storage-refuses-smile-latents.md](../docs/tickets/T010-postgres-storage-refuses-smile-latents.md); [docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md](../docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md); [current_development/grw_market_smile/](../current_development/grw_market_smile/) |

## Context & Problem Statement

In Task 015, the market smile model fitted 5 independent unconstrained strike parameters ($\log \phi \sim \text{Normal}(0, 0.5)^5$) alongside market supremacy anchoring on `MultiScaleGRW`. While the model demonstrated substantial portfolio alpha on the Option B closing line (+545% to +588% bankroll growth and 15.8% flat ROI) by improving 1X2 away-bet sizing and pruning toxic bets, it revealed critical limitations:

1. **Massive MCMC Sampling Drag**: Wall time ballooned from 60 min (supremacy only) to 185 min (`w040`) and 438 min (`w070`), with minimum bulk ESS dropping from 814 down to 312, requiring a $4 \times (1000\text{w} + 2000\text{s})$ doubled budget to clear convergence.
2. **Redundant Degrees of Freedom**: The empirical posterior medians for $\log \phi(K)$ across strikes $K \in \{0, 1, 2, 3, 4\}$ follow an almost perfectly linear monotonic curve centered at $K=2$ (the 2.5 line): $[-0.170, -0.024, +0.001, +0.026, +0.069]$.
3. **Ticket T011 (Portfolio Staking Disconnect)**: `Portfolio.build_books_reported` reports `p_model` using $\phi(K)$, but the Kelly allocator solves the stake vector strictly from the un-smiled $(\lambda_h, \lambda_a)$ Poisson score grid. As a result, $\phi(K)$ never directly sized stakes.

Task 016 prototypes a **1-parameter smile spine**:
$$\log \phi(K) = \beta \cdot (K - 2) \iff \phi(K) = \exp\big(\beta \cdot (K - 2)\big) \quad \text{for } K \in \{0, 1, 2, 3, 4\}$$
with prior $\beta \sim \text{Normal}(0.04, 0.05)$, fixing $\phi(2) = 1.000$ strictly by construction.

In addition, the prototype pricer directly addresses Ticket T011 by reweighting the $12 \times 12$ scoreline grid anti-diagonals (where total goals $G = h + a$) to match the spine's marginal totals CDF before Kelly optimization.

## Acceptance Criteria

- [ ] Implement prototype in `current_development/grw_smile_spine/`:
  - `l01_loader.jl`: 1-parameter `MarketSmileSpinePillar` ($\beta$ scalar parameter), log-density, and anti-diagonal grid reweighting.
  - `r01_smoke.jl`: 2-fold smoke gate verifying ReverseDiff compiled tape against ForwardDiff ($\le 10^{-6}$), exact gradients, parameter recovery, and anti-diagonal grid reweighting accuracy.
  - `r02_production_grid.jl`: 43-fold walk-forward grid for `m05_joint_grw_smile_spine_w020` and `m05_joint_grw_smile_spine_w040` on `mcmc-beast` (-t 16).
  - Benchmark computation time and bulk/tail ESS per fold against Task 015's 5-parameter model.
  - `r04_evaluate.jl`: Proper scores (LogLoss, Brier, RPS, ECE) vs Betfair close and Task 015 rungs over the 710 walk-forward matches.
  - `r06_portfolio.jl`: Closing-line Option B portfolio with anti-diagonal reweighted staking (resolving T011).
  - `r07_t25_portfolio.jl`: T−25 tradeable portfolio with and without Option B L2 calibrator (`scot_lower_t25_inv`).
  - `r08_trust_sweep.jl`: Market expansion test evaluating `Under 1.5` and `Under 4.5`.
  - `README.md`: Comprehensive documentation.
- [ ] Model Ablation Ladder (Scottish Lower, 43 folds):
  1. `m05_joint_grw_baseline`: Pure GRW control (Task 013 `b0961bc4`)
  2. `m05_joint_grw_supremacy_w040`: Supremacy anchor only (Task 015 `0ee58d18`)
  3. `m05_joint_grw_smile_supremacy_w020`: Full 5-parameter smile @ 0.20 (Task 015 `fcd5e974`)
  4. `m05_joint_grw_smile_supremacy_w040`: Full 5-parameter smile @ 0.40 (Task 015 `30620d3e`)
  5. `m05_joint_grw_smile_spine_w020`: 1-param spine @ 0.20 (new sampling)
  6. `m05_joint_grw_smile_spine_w040`: 1-param spine @ 0.40 (new sampling)
- [ ] 6-part convergence audit passed on all 43 folds ($\hat{R} \le 1.05$, 0 divergences, ESS $\ge 400$).
- [ ] Store fits in `PostgresStorage("scottish_lower_grw_smile_spine")`.

## Ideas & Candidate Solutions

- **Anti-Diagonal Grid Reweighting (T011 Fix)**: Given joint scoreline probabilities $P_{\text{grid}}(h, a)$, total goals $G = h + a$ has marginal $P_{\text{grid}}(G) = \sum_{h+a=G} P_{\text{grid}}(h, a)$. The spine defines $P_{\text{spine}}(G \le K) = \text{cdf}(\text{Poisson}(\lambda_{\text{tot}} \cdot e^{\beta(K-2)}), K)$. Rescaling anti-diagonals by $P_{\text{spine}}(G) / P_{\text{grid}}(G)$ produces a valid $12 \times 12$ joint scoreline distribution whose totals marginal matches the smile spine, enabling coherent Kelly optimization.
- **Computation Time Reduction**: Dropping from 5 strike parameters to 1 scalar parameter $\beta$ simplifies the NUTS Riemannian geometry and should restore sampling speed close to the ~60-minute supremacy baseline.

## Work Log & Progress

- [2026-09-13 @antigravity] Aligned task via `/grill-me`. Created Task 016, initialized worktree at `/home/james/bet_project/.worktrees/BayesianFootball-grw-smile-spine` on branch `feat/grw-smile-spine`, prepared work package prompt `SMILE_SPINE_WORK_PACKAGE.md`, and launched `claude_smile_spine` in tmux.
- [2026-09-13 @claude] Phase 1 authored, NOT yet executed: `current_development/grw_smile_spine/l01_loader.jl` (includes Task 015's loader unchanged; new `SpineAnchoredCountModel` + `MarketSmileSpinePillar`, site `β_spine`; anti-diagonal reweighting `gss_reweight_grid!` and `gss_build_books_reweighted`) and `r01_smoke.jl` (gates GA, GB, G4a pre-sampling, G0a/b/c, G1, G6, G2, G3, G4b, G4c, G5). Work-package interpretations recorded in the loader header: all 23 anti-diagonals of the 12×12 grid are rescaled (not 12); mass above K = 4 follows the grid's own diagonal proportions; a non-monotone smile draw is refused, not clipped; φ ≡ 1 draws are left bit-identical. G0c added: spine ≡ five-strike model on the line log φ = β(K−2). Smoke budget 4×(500+1000) because Task 015's smoke failed its baseline on tail ESS at 500 draws. Caveat: the line through Task 015's φ medians (β_LS ≈ 0.052) misses K = 0 by ≈ −0.065 in log φ.
- [2026-09-13 @claude] Committed `0286ebd2`, cloned to mcmc-beast `/root/BF_grw_smile_spine` (Task 015's ScottishLower cache of 2026-09-12, `.env` and pinned `Manifest.toml` copied from `/root/BF_grw_market_smile`; Distributions 0.25.126). First r01 launch stopped at G4a before sampling: with φ ≡ 1 and the identity shortcut disabled the reweighted grid moved by 2.23e-4 against a fixed 1e-6 tolerance. Cause is the grid's own truncation mass (goals ≥ 12 per side, ~0.1–0.2% at the synthetic 4.0 rates), which the Σ = 1 reweighting relocates onto totals ≥ 5; shortcut bit-identity, totals-CDF match, mass and non-monotone refusal all passed. With user approval the check now bounds every cell's move by that draw's truncation mass (+1e-14 float slack), a derived bound; other G4 tolerances unchanged.

## Verification & Findings

Not run yet. Record sampling wall time and ESS vs Task 015 5-parameter model, proper scores, Option B portfolio metrics, and T−25 calibration results.
