# Negative Binomial + Linear Proxy-xG Form Covariate — Scottish Lower Decompression

**TODO [024](../../../todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md)** ·
namespace `scottish_lower_decompression` · **IN_PROGRESS — 2026-09-22**

## Scientific Question

Can an antisymmetric linear proxy-xG form supremacy covariate bypass hierarchical Bayesian shrinkage and decompress favourite pricing (improving the empirical supremacy slope from $1.41\text{--}1.66$ towards $1.00$) in Scottish Lower football (tournaments 56 & 57, 40-fold walk-forward grid, seasons 24/25 and 25/26, 710 fixtures)?

## Decision & Headline Results

*(To be recorded upon completion of Stage 3 & 4)*

| Arm | Supremacy slope | Favourite P(win) | Overall LogLoss ↓ | 1X2 LogLoss ↓ | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m01_poisson_time_decay` (Control 1) | — | — | — | — | — | — | — |
| `m02_joint_gamma_poisson` (Control 2) | — | — | — | — | — | — | — |
| `m03_negbin_pxg_covariate` (Candidate) | — | — | — | — | — | — | — |
| Betfair close (Reference) | 1.0000 | — | — | — | — | — | — |

## Formulation & Covariate Structure

$$\log \lambda_{h} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h} + \beta_{\text{def}, a} + \frac{1}{2} w_{\text{pxg}} \cdot (\text{pxg\_form}_h - \text{pxg\_form}_a)$$
$$\log \lambda_{a} = \mu + \alpha_{\text{att}, a} + \beta_{\text{def}, h} - \frac{1}{2} w_{\text{pxg}} \cdot (\text{pxg\_form}_h - \text{pxg\_form}_a)$$
$$y_h \sim \text{NegBin}(\lambda_h, \phi), \quad y_a \sim \text{NegBin}(\lambda_a, \phi)$$
Prior: $w_{\text{pxg}} \sim \mathcal{N}(0.60, 0.20^2)$ informed by Market-Inverse Phase 2 attribution findings.

## Verification Checklist

- [ ] Stage 0: Zero allocations on compiled ReverseDiff gradient tapes.
- [ ] Stage 1: Smoke gate passed on folds 1, 20, 40 (0 divergences, R̂ ≤ 1.05, ESS ≥ 200, positive $w_{\text{pxg}}$).
- [ ] Stage 2: 40-fold walk-forward grid executed and persisted to PostgreSQL `scottish_lower_decompression`.
- [ ] Stage 3: Supremacy regression and portfolio backtest completed on common tradeable panel.
- [ ] Stage 4: Signed off in `todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md` and `./scripts/todo.sh check`.
