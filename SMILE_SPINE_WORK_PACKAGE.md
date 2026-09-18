# Work Package: Task 016 — Prototype 1-Parameter Smile Spine with MultiScaleGRW

## Mission Objective
Design, benchmark, and validate a **1-Parameter Market Smile Spine** coupled with `MultiScaleGRW` and market supremacy anchoring for Scottish Lower (tournaments 56 & 57).

In Task 015, the 5-parameter smile model ($\log \phi_K \sim \text{Normal}(0, 0.5)^5$) achieved significant portfolio ROI gains on 1X2 away bets and Kelly sizing, but:
1. Ballooned MCMC sampling time from 60m to 185m–438m with bulk ESS dropping to 312 due to high-dimensional parameter interaction.
2. The empirical posterior medians for $\log \phi(K)$ across strikes $K \in \{0, 1, 2, 3, 4\}$ were found to lie along a monotonic line centered at $K=2$ (the 2.5 line): $[-0.170, -0.024, +0.001, +0.026, +0.069]$.
3. Ticket T011 revealed that `Portfolio.build_books_reported` reports `p_model` using $\phi(K)$ but currently solves Kelly stakes off the un-smiled $(\lambda_h, \lambda_a)$ grid.

This task resolves both issues:
1. Replaces the 5 free parameters with a **1-parameter linear log-slope spine**:
   $$\log \phi(K) = \beta \cdot (K - 2) \iff \phi(K) = \exp\big(\beta \cdot (K - 2)\big) \quad \text{for } K \in \{0, 1, 2, 3, 4\}$$
   with prior $\beta \sim \text{Normal}(0.04, 0.05)$, fixing $\phi(2) = 1.000$ strictly by construction.
2. Resolves Ticket T011 in the prototype pricer by **reweighting the 12×12 scoreline grid anti-diagonals** ($G = h + a$) to match the spine's marginal totals CDF before Kelly optimization.

---

## 1. Mathematical Specification

### A. Supremacy & Spine Likelihood Pillars
On top of the Gen 3 Team-Level Joint foundation (`MultiScaleGRW` + `ProductionWealthCovariate(SupremacyRole)` + `JointGammaPoissonObservation`):

```
C1 Supremacy:  η_h − η_a                      ~ Normal(log λ̂_h − log λ̂_a, σ_sup)       × w_sup
C2 SmileSpine: log κ + log(μ_h + μ_a) + β·(K-2) ~ Normal(log Λ̂_K, σ_smile), K = 0…4     × w_smile

Priors:
  σ_sup, σ_smile ~ truncated(Normal(0.15, 0.10), lower = 0.02)
  β              ~ Normal(0.04, 0.05)
```
- Both pillars read **training fixtures only**.
- At $K=2$ (the 2.5 line), $\beta \cdot (2 - 2) = 0$, so $\phi(2) = 1.000$ unconditionally.
- For $K < 2$, $\phi(K) < 1$; for $K > 2$, $\phi(K) > 1$.

### B. Anti-Diagonal Grid Reweighting (Ticket T011 Fix)
In the prototype portfolio pricer:
1. Compute the bivariate Poisson scoreline grid $P_{\text{grid}}(h, a)$ for $0 \le h, a \le 11$.
2. For each total goals level $G \in \{0, \dots, 11\}$:
   $$P_{\text{grid}}(\text{Total} = G) = \sum_{h + a = G} P_{\text{grid}}(h, a)$$
3. Compute the marginal total probability under the smile spine:
   - For $K \in \{0, 1, 2, 3, 4\}$: $F_{\text{spine}}(K) = \text{cdf}(\text{Poisson}(\lambda_{\text{tot}} \cdot e^{\beta(K-2)}), K)$.
   - Derive discrete spine mass: $P_{\text{spine}}(0) = F_{\text{spine}}(0)$, $P_{\text{spine}}(G) = F_{\text{spine}}(G) - F_{\text{spine}}(G-1)$ for $G \in \{1, 2, 3, 4\}$.
   - For $G > 4$, preserve the relative Poisson tail normalized so $\sum_{G=0}^{11} P_{\text{spine}}(G) = 1$.
4. Reweight each scoreline $(h, a)$ on anti-diagonal $G = h + a$:
   $$P_{\text{reweighted}}(h, a) = P_{\text{grid}}(h, a) \cdot \frac{P_{\text{spine}}(G)}{P_{\text{grid}}(G)}$$
5. Pass $P_{\text{reweighted}}$ to the Kelly allocator (`allocate` and `grid_shrink_factor`). Now the allocator solves stakes directly off the smiled distribution.

---

## 2. Model Ladder & Work Split

### Rungs
1. `m05_joint_grw_baseline`: Pure GRW control (pinned Task 013 `b0961bc4`)
2. `m05_joint_grw_supremacy_w040`: Pure supremacy anchor (pinned Task 015 `0ee58d18`)
3. `m05_joint_grw_smile_supremacy_w020`: Full 5-param smile @ 0.20 (pinned Task 015 `fcd5e974`)
4. `m05_joint_grw_smile_supremacy_w040`: Full 5-param smile @ 0.40 (pinned Task 015 `30620d3e`)
5. **`m05_joint_grw_smile_spine_w020`**: **NEW 1-param spine @ 0.20** (sample on beast)
6. **`m05_joint_grw_smile_spine_w040`**: **NEW 1-param spine @ 0.40** (sample on beast)

**Crucial**: Only rungs 5 and 6 are sampled. Rungs 1–4 are loaded directly from `mcmc_experiments`.

---

## 3. Prototype Structure (`current_development/grw_smile_spine/`)

- `l01_loader.jl`: Data loading, `MarketSmileSpinePillar` ($\beta$ scalar), log-density, detached panel persistence (T010), and anti-diagonal grid reweighting.
- `l02_evaluation.jl`: Out-of-sample proper scoring, book building, fixture-clustered bootstrap.
- `l04_portfolio_calibration.jl`: Option B portfolio simulation, T−25 tradeable book, and L2 calibration (`scot_lower_t25_inv`).
- `r01_smoke.jl`: Smoke test on Folds 1–2 (G0 likelihood parity, G1 ReverseDiff compiled tape vs ForwardDiff $\le 10^{-6}$, G2 sampling, G4 smile pricing, G5 persistence).
- `r02_production_grid.jl`: 43-fold walk-forward grid for the two spine rungs on `mcmc-beast` (-t 16). Persist to `scottish_lower_grw_smile_spine`.
- `r04_evaluate.jl`: Out-of-sample proper scores (LogLoss, Brier, RPS, ECE) vs Betfair close and Task 015 rungs over the 710 walk-forward matches.
- `r06_portfolio.jl`: Option B closing-line portfolio with anti-diagonal reweighted staking.
- `r07_t25_portfolio.jl`: T−25 tradeable portfolio with and without Option B L2 calibrator.
- `r08_trust_sweep.jl`: Market expansion test evaluating `Under 1.5` and `Under 4.5`.
- `README.md`: Full documentation, benchmark tables, and tearsheet.

---

## 4. Hypotheses to Validate

- **H1 (Computational Efficiency)**: The 1-parameter spine significantly reduces MCMC wall time per fold (targeting $\le 90$ min for 43 folds) and restores minimum bulk ESS $\ge 600$ (compared to 185m–438m and bulk ESS 312 in the 5-param model).
- **H2 (Parameter Recovery)**: $\beta$ converges stably to $\sim 0.03\text{--}0.05$ across folds with low posterior variance.
- **H3 (Predictive Parity)**: Out-of-sample proper scores for the 1-parameter spine are statistically indistinguishable from the 5-parameter smile ($\Delta\text{LogLoss} \approx 0$).
- **H4 (Portfolio Alpha & Sizing)**: With anti-diagonal reweighted staking (fixing T011), the 1-parameter spine preserves the +19% away-bet ROI and +15.8% flat ROI while providing coherent Kelly sizing on totals.
- **H5 (T−25 Calibration & Market Expansion)**: Evaluates whether `Under 1.5` and `Under 4.5` remain accretive under the reweighted spine.

---

## 5. Verification Gates
- **Smoke Gate (Folds 1–2)**: G0 exact log-density re-derivation $\le 10^{-14}$; G1 ReverseDiff tape exact vs ForwardDiff $\le 10^{-6}$; G2 0 divergences; G4 anti-diagonal grid marginal totals match spine CDF $\le 10^{-6}$.
- **Production Gate (43 Folds)**: $\hat{R} \le 1.05$, 0 divergences, min bulk/tail ESS $\ge 400$.
- **Filtration**: 43 folds over 24/25–26/27. Training fixtures only for market pillars. Common evaluation panel exactly 710 fixtures in 24/25 + 25/26.
