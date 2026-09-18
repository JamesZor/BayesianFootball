# MatchDay Operational Note: Upgrading Default `SlateDrawdown` Risk Policy ($\lambda = 8.0 \to 28.0$)

**Date:** 2026-09-18  
**Scope:** `src/MatchDay/calibration.jl`, `current_development/match_day_inference/`  
**Reference Tasks:** Task 020 (`todos/020_sweep_slatedrawdown_lambda_risk_budgets_on_grw_models.md`), Ticket T012  
**Associated Artifacts:**
- [`current_development/grw_risk_sweep/results/LAMBDA_RISK_SWEEP_REPORT.md`](../../current_development/grw_risk_sweep/results/LAMBDA_RISK_SWEEP_REPORT.md)
- [`current_development/grw_risk_sweep/results/lambda_sweep_summary.csv`](../../current_development/grw_risk_sweep/results/lambda_sweep_summary.csv)
- [`current_development/grw_risk_sweep/results/slate_pnl_moments.csv`](../../current_development/grw_risk_sweep/results/slate_pnl_moments.csv)

---

## 1. Executive Summary

This note documents the rationale, empirical evidence, and code changes for upgrading the default tail-risk parameter in `MatchDay`:

* **Change:** `option_b_scottish_lower_policy` and `option_b_system` in [`src/MatchDay/calibration.jl`](../../src/MatchDay/calibration.jl) now default to **`Portfolio.SlateDrawdown(28.0)`** instead of the historical `Portfolio.SlateDrawdown(8.0)`.
* **Keyword Argument Support:** Both functions now accept `lambda::Real = 28.0` as an optional parameter, enabling runtime risk budget tuning without monkey-patching.
* **Impact:** On the 100-slate Scottish Lower walk-forward panel (24/25 + 25/26), this upgrade compresses peak-to-trough realised max drawdown from **−42.81% to −17.08%** (comfortably under the repository's $\le 20\%$ risk ceiling), increases the annualised Sharpe ratio from **1.537 to 1.744**, reduces average slate capital lockup from **14.55% to 4.79%**, while preserving a doubling of bankroll (**+109.92% terminal return**, **+53.71% CAGR**).

---

## 2. Context & The Drawdown Problem

When the **Option B** basket (`1X2` + `Under 2.5` at full trust; `Draw`, `Away`, and `Over 1.5` at `1/1.4`) was integrated into `MatchDay`, it was configured with `SlateDrawdown(8.0)` to match the unconstrained Kelly growth benchmark from Suite 07.

However, comprehensive testing on the held-out walk-forward panel (710 matches, 628 buildable close fixtures, 100 daily slates) revealed that $\lambda = 8.0$ incurs severe downside volatility:

| Environment | Model Variant | $\lambda$ | Terminal Return | Realised Max Drawdown | Annual Sharpe | Mean Slate Exposure |
|---|---|---|---|---|---|---|
| Close | `m05_joint_grw_smile_spine_w040` (raw) | 8.0 | +576.13% | **−42.81%** | 1.537 | 14.55% |
| T−25 | `m05_joint_grw_smile_spine_w040` (raw) | 8.0 | +417.04% | **−44.19%** | 1.226 | 13.79% |
| Close | `m05_joint_grw_baseline` (raw) | 8.0 | +431.30% | **−41.96%** | 1.554 | 16.73% |

A **−43% to −44% peak-to-trough drawdown** is unacceptable for live paper execution on `paper_runbook` and breaches the operational portfolio risk ceiling of $\le 20\%$.

---

## 3. Empirical Evidence: The Task 020 Lambda Sweep

A grid sweep over $\lambda \in [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0, 35.0, 45.0]$ was executed across all 60 combinations of models and market environments on `mcmc-beast`.

### A. Performance Progression across $\lambda$ (`m05_joint_grw_smile_spine_w040` on Close Raw)

| $\lambda$ | Terminal Return | Realised Max DD | Annual Sharpe | Mean Exposure | Slate Log Growth $g$ (nats/slate) | Geom. Growth ($e^g - 1$) |
|---|---|---|---|---|---|---|
| **8.0** *(Old Default)* | +576.13% | **−42.81%** | 1.537 | 14.55% | 0.01911 | +1.930% / slate |
| **12.0** | +337.90% | −35.37% | 1.593 | 10.62% | 0.01477 | +1.488% / slate |
| **18.0** | +197.70% | −25.26% | 1.690 | 7.33% | 0.01091 | +1.097% / slate |
| **20.0** | +171.03% | −23.05% | 1.705 | 6.62% | 0.00997 | +1.002% / slate |
| **23.0** *(Old Scottish Policy)* | +141.84% | −20.38% | 1.722 | 5.79% | 0.00883 | +0.887% / slate |
| **28.0** *(New Default)* | **+109.92%** | **−17.08%** | **1.744** | **4.79%** | **0.00742** | **+0.744% / slate** |
| **35.0** | +83.21% | −13.92% | 1.763 | 3.86% | 0.00605 | +0.607% / slate |
| **45.0** *(Max Sharpe)* | +61.54% | −11.00% | **1.781** | 3.02% | 0.00480 | +0.481% / slate |

### B. The Operational Knee at $\lambda = 28.0$
* **Ceiling Enforcement:** $\lambda = 28.0$ is the **first parameter value on the grid** that forces realised max drawdown strictly below the 20% threshold on all raw boards (−17.08% on smile spine, −19.65% on baseline, −17.81% on T−25 raw). Note that $\lambda = 23.0$ (previously in `canonical_scottish_lower_policy`) breaches the 20% ceiling at −20.38%.
* **Superior Sharpe Ratio:** Sharpe increases from 1.537 to **1.744** because downside tail variance is dampened much faster than expected returns.
* **Elimination of Volatility Drag:** In Kelly portfolio theory, expected log growth satisfies $g \approx \mu - \frac{1}{2}\sigma^2$.
  * At $\lambda = 8.0$: Slate variance is high ($\sigma = 9.88\%$), resulting in a Jensen volatility drag of $\approx 0.49\%$ per slate (~20% of arithmetic edge vaporized).
  * At $\lambda = 28.0$: Slate volatility drops to $\sigma = 3.29\%$, reducing volatility drag to just $0.05\%$. **93% of arithmetic slate return is converted directly into geometric compounding growth** (+0.744% per slate).

### C. Four Statistical Moments of Slate PnL
Analysis across all 100 slates confirms the structural stability of the portfolio:

| Moment | $\lambda = 8.0$ | $\lambda = 28.0$ (New Default) | $\lambda = 45.0$ | Property |
|---|---|---|---|---|
| **1st: Mean ($\mu$)** | +2.388% | **+0.797%** | +0.502% | Scales linearly with $1/\lambda$ exposure |
| **2nd: Std Dev ($\sigma$)** | 9.882% | **3.294%** | 2.074% | Contraction slashes portfolio risk |
| **3rd: Skewness ($S$)** | +0.650 | **+0.654** | +0.654 | **Positively skewed everywhere**; invariant to $\lambda$ |
| **4th: Excess Kurtosis ($K_{\text{ex}}$)** | +0.499 | **+0.778** | +0.780 | Leptokurtic; upside win runs exceed bounded losses |

### D. Behavior under T−25 L2 Generative Rate Calibration
Under T−25 L2 generative rate calibration (Inverse Gaussian law $w_{\text{base}}=0.25, \sigma=0.35$), market inversion eliminates noisy low-conviction edges:
* Slate variance drops by **~70%**.
* Positive skewness jumps to **$S = +1.37$** and excess kurtosis reaches **$K_{\text{ex}} = +2.71$**.
* At the new default $\lambda = 28.0$, max drawdown is suppressed to just **−5.44%** with an annual Sharpe of **1.864**.
* For operators prioritizing higher growth specifically in calibrated live execution, setting `lambda = 20.0` yields **−7.46% drawdown** and **+67.21% terminal return** (Sharpe 1.851).

---

## 4. Implementation Changes

### In `src/MatchDay/calibration.jl`:

```julia
"""
    option_b_scottish_lower_policy(; lambda::Real = 28.0) -> Portfolio.PolicySpec

The Option B basket and trust vector: Home and Under 2.5 at full trust; Draw, Away and
Over 1.5 at `1/1.4`; every other canonical selection gated.

The slate-wide risk defaults to `SlateDrawdown(28.0)` (the Task 020 operational knee
enforcing realised max drawdown strictly under the 20% ceiling on raw/close boards, with
Sharpe climbing to 1.74–1.86). For T−25 L2-calibrated pipelines, `lambda = 20.0` or `28.0` is
recommended. The hard simultaneous-exposure cap remains `FixedCap(0.25)`.
"""
option_b_scottish_lower_policy(; lambda::Real = 28.0) = Portfolio.PolicySpec(
    trust = Portfolio.TieredTrust(Dict(
        ("1x2", 0.0, :home)         => 1.0,
        ("over_under", 2.5, :under) => 1.0,
        ("1x2", 0.0, :draw)         => 1.0 / 1.4,
        ("1x2", 0.0, :away)         => 1.0 / 1.4,
        ("over_under", 1.5, :over)  => 1.0 / 1.4,
    ); default = 0.0),
    risk = Portfolio.SlateDrawdown(Float64(lambda)),
    cap = Portfolio.FixedCap(0.25),
    grouping = Portfolio.DailySlate(),
)

"The complete Scottish Lower Option B pricing and staking system."
option_b_system(; lambda::Real = 28.0) = Portfolio.PortfolioSystem(
    option_b_book_spec(),
    option_b_scottish_lower_policy(; lambda = lambda),
)
```

### In `test/test_matchday_live_pipeline.jl`:
- Updated assertion: `@test system.policy.risk.lambda == 28.0`
- Added test asserting that `MD.option_b_system(; lambda = 8.0)` properly passes user-specified overrides.

---

## 5. Operational Guidelines for Live & Replay Consoles

1. **Saturday Live Console (`r07_serve_console.jl` / `r09_live_calibrated_slate.jl`)**:
   - Uses `MD.option_b_system()` directly; it now automatically inherits $\lambda = 28.0$.
   - **Operator Benefit:** Average stake commitment per slate drops from ~15% of bankroll to ~4.8%, reducing Betfair market exposure and margin requirements while eliminating the threat of deep drawdowns.
   - If pricing exclusively through T−25 calibrated books, an operator may optionally specify `option_b_system(lambda = 20.0)` for slightly higher exposure (~6.6%) while keeping drawdown $< 8\%$.
2. **Replay Console (`r08_replay_console.jl`)**:
   - Automatically loads the updated policy, ensuring historical counterfactual replays reflect realistic, risk-managed staking behavior rather than near-full-Kelly spikes.
3. **Paper Ledger Safety (`paper_runbook`)**:
   - The reservation transaction in `account_ledger` with `FixedCap(0.25)` is now rarely hit (zero slates capped at $\lambda = 28.0$, compared to 15 slates capped at $\lambda = 8.0$). This eliminates artificial stake clipping and preserves the optimal convex Kelly ratios.
