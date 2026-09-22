# Decoupled Generative xG-Primary Funnel — Scottish Lower Decompression

**TODO [025](../../../todos/025_prototype_decoupled_generative_xg_primary_model.md)** ·
namespace `scottish_lower_decoupled_xg` · **IN_PROGRESS — 2026-09-22**

## Scientific Question

Can a decoupled generative chance-creation funnel ($\text{Team Ratings} \to \text{xG} \to \text{Goals}$) decompress favourite pricing (closing the 1.724 slope gap towards 1.00) while preserving totals and BTTS calibration, and does hierarchical team finishing ($\kappa_i$) provide any out-of-sample benefit over a shared league conversion factor ($\kappa$)?

## Decision & Headline Results

*(To be recorded upon completion of Stage 3 & 4)*

| Arm | Market on model slope ↓ | Favourite P(win) | Overall LogLoss ↓ | 1X2 LogLoss ↓ | O/U 2.5 LogLoss ↓ | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m01_poisson_time_decay` (Control 1) | 2.5288 | 48.62% | 0.646809 | 0.620351 | 0.689973 | +139.3% | 1.204 | −22.84% |
| `m02_joint_gamma_poisson` (Control 2) | 1.7240 | 51.93% | 0.643730 | 0.617436 | 0.687000 | +128.0% | 1.270 | −22.43% |
| `m03_funnel_shared_kappa` (Candidate 1) | — | — | — | — | — | — | — | — |
| `m04_funnel_hierarchical_kappa` (Candidate 2) | — | — | — | — | — | — | — | — |
| Betfair close (Reference) | 1.0000 | 76.25% | 0.641816 | 0.613118 | 0.689878 | — | — | — |

## Formulation

$$\log \mu_{\text{xg}, h} = \mu_{\text{xg}} + \gamma_{\text{home}} + \alpha_{\text{xg}, h} + \beta_{\text{xg}, a}$$
$$\text{pxg}_h \sim \text{Gamma}(\nu, \mu_{\text{xg}, h} / \nu)$$
$$y_h \sim \text{Poisson}(\lambda_{\text{goal}, h}), \quad \lambda_{\text{goal}, h} = \kappa_h \cdot \mu_{\text{xg}, h}$$

- `m03`: Shared $\kappa \sim \text{LogNormal}(0, 0.20)$
- `m04`: Hierarchical team $\kappa_i = \kappa \exp(\delta_i)$ with $\sum_i \delta_i = 0$

## Verification Checklist

- [ ] Stage 0: Zero allocations on compiled ReverseDiff gradient tapes.
- [ ] Stage 1: Smoke gate passed on folds 1, 20, 40 (0 divergences, R̂ ≤ 1.05, ESS ≥ 200).
- [ ] Stage 2: 40-fold walk-forward grid executed and persisted to PostgreSQL `scottish_lower_decoupled_xg`.
- [ ] Stage 3: Supremacy regression, proper scores, and portfolio backtest on common tradeable panel.
- [ ] Stage 4: Signed off in `todos/025_prototype_decoupled_generative_xg_primary_model.md` and `./scripts/todo.sh check`.
