# 012 — Standardise Model Attribution and Capture Ratio in Portfolio Engine

| Field | Value |
|---|---|
| ID | 012 |
| Title | Standardise Model Attribution and Capture Ratio in Portfolio Engine |
| Status | COMPLETED |
| Priority | P2 |
| Assignee | pi |
| Created | 2026-09-11 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `src/Portfolio/attribution.jl`, `src/Portfolio/types.jl`, `src/Portfolio/display.jl`, `src/Portfolio/reporting.jl`, `test/test_portfolio_attribution.jl`, `todos/007_prototype_gaussian_random_walk_state_space_dynamics_with_reversediff.md` |

## Context & Problem Statement

When benchmarking Bayesian football forecasting models (e.g. comparing MultiScaleGRW against TimeDecay controls, or evaluating new likelihood and covariate architectures against production champions), comparing headline ROI, annual Sharpe, or LogLoss alone conceals the underlying mathematical mechanisms driving performance.

In Task 007 follow-up analysis ($r04$, `current_development/multiscale_grw/`), an attribution methodology was developed that decomposed portfolio outperformance into three distinct, separable mechanisms:

1. **Confidence & Capture Ratio ($e_{\text{win}} / e_{\text{loss}}$)**:
   - Evaluates whether model edge ($e_i = p_{\text{model}, i} - p_{\text{market}, i}$) is systematically larger on winning bets than on losing bets.
   - On staked bets (where fractional Kelly enforces $p_{\text{model}} > p_{\text{market}}$), every staked bet is positive-edge. Losing bets represent edges that failed to materialize, rather than negative edges.
   - Mean winning edge $e_{\text{win}} = \mathbb{E}[e \mid \text{won}]$ and mean losing edge $e_{\text{loss}} = \mathbb{E}[e \mid \text{lost}]$ are both positive.
   - The ratio $\text{Capture Ratio} = e_{\text{win}} / e_{\text{loss}}$ cleanly measures whether the model sits further from the market when it is right than when it is wrong.
   - In $r04$, MultiScaleGRW achieved Capture Ratios of **1.07 to 1.16** across all matched pairs, while TimeDecay models scored **0.90 to 0.97** (suffering larger errors when wrong).

2. **Controlled Sizing Alpha on Shared Bets ($B_{\text{both}} = B_A \cap B_B$)**:
   - Isolates the exact subset of bets executed by both models on identical fixtures, market families, selections, and prices.
   - Because fixtures, odds, and settlement outcomes are identical, any PnL/ROI difference on $B_{\text{both}}$ is **pure sizing alpha** ($\sum \Delta s_i \times \text{settle}_i$).
   - In $r04$, on ~1,030 shared bets, MultiScaleGRW earned **+19.50% ROI vs TimeDecay's +13.98% ROI** (+5.52 pp pure sizing edge) due to Kelly stakes scaling with superior model probability calibration on winners.

3. **Selectivity Alpha on Disjoint Bets ($B_{A \setminus B}$ vs $B_{B \setminus A}$)**:
   - Directly measures the turnover, hit rate, and ROI of bets selected exclusively by one model versus the other.

4. **Capital-Weighted Win Rate**:
   - $\text{WinRate}_{\text{cap}} = \sum (s_i \cdot \mathbf{1}_{\text{win}, i}) / \sum s_i$ compared against unweighted win rate $\frac{1}{N}\sum \mathbf{1}_{\text{win}, i}$.
   - Measures whether the sizing engine concentrates capital on winners or drags capital into losing positions.

### Current Problem
Currently, these analyses exist only in prototype research scripts (`current_development/multiscale_grw/l04_attribution_markets.jl` and `r04_attribution_and_markets.jl`). In `src/Portfolio/`, `attribution(t::Trajectory)` only produces a rudimentary breakdown by selection family (`stake`, `pnl`, `med_odds`, `hit`, `roi`). The core engine lacks standardized types, methods, and reporting tools to compute Capture Ratio or perform pairwise model-vs-model ledger attribution directly on `PortfolioResult` or `Trajectory`.

## Acceptance Criteria

- [x] **First-Class Attribution Types in `src/Portfolio/types.jl`**:
  - Define `EdgeSummary` holding: `n_bets`, `n_wins`, `win_rate`, `cap_weighted_win_rate`, `stake_sum`, `pnl_sum`, `roi`, `edge_mean`, `edge_win`, `edge_loss`, `capture_ratio`, `stake_mean`, `odds_mean`, `p_model_mean`, `p_market_mean`.
  - Define `ModelComparisonAttribution` holding: `shared_a`, `shared_b`, `exclusive_a`, `exclusive_b`, `sizing_delta_pnl`, `shared_roi_a`, `shared_roi_b`, `summary_a`, `summary_b`.
- [x] **Single-Portfolio Confidence & Capture Ratio APIs**:
  - Implement `edge_summary(bets::AbstractDataFrame) -> EdgeSummary`.
  - Implement `capture_ratio(bets::AbstractDataFrame) -> Float64`.
  - Dispatch methods for portfolio types:
    - `capture_ratio(t::Trajectory) -> Float64`
    - `capture_ratio(r::PortfolioResult) -> Float64`
    - `edge_summary(t::Trajectory) -> EdgeSummary`
    - `edge_summary(r::PortfolioResult) -> EdgeSummary`
  - Safeguard numerical edge cases: return `NaN` when $e_{\text{loss}} \le 0$, when bets are empty, or when no wins/losses exist.
- [x] **Pairwise Model Bet Partitioning & Sizing Attribution**:
  - Implement `partition_bets(a::AbstractDataFrame, b::AbstractDataFrame) -> (both_a, both_b, only_a, only_b)` asserting duplicate-free keys and exact row alignment on shared bets.
  - Implement `shared_bet_sizing_attribution(both_a::AbstractDataFrame, both_b::AbstractDataFrame)` computing $\Delta s_i = s_{A, i} - s_{B, i}$ and sizing PnL contribution $\sum \Delta s_i \times \text{settle}_i$.
  - Implement high-level dispatch:
    - `compare_portfolios(res_a::PortfolioResult, res_b::PortfolioResult; name_a="Model A", name_b="Model B") -> ModelComparisonAttribution`
- [x] **Breakdown Extensions**:
  - Support breakdown by odds buckets (e.g. `< 2.0`, `2.0 - 3.5`, `≥ 3.5`) to test longshot bias.
  - Support breakdown by market family (`1X2`, `OverUnder`, `BTTS`).
- [x] **Display & Reporting Integration**:
  - Add `Base.show` and clean tabular summary formatting for `ModelComparisonAttribution` in `src/Portfolio/display.jl` and `reporting.jl`.
- [x] **Unit Tests & Regression Safety**:
  - Add test suite `test/test_portfolio_attribution.jl` verifying:
    - Capture ratio calculations across synthetic edge distributions (positive, zero, undefined).
    - Partitioning correctness, duplicate bet key detection, and empty set handling.
    - Sizing identity: $\Delta \text{PnL}_{\text{shared}} = \sum (\Delta s_i \cdot \text{settle}_i)$.
  - Ensure `./scripts/todo.sh check` and `julia --project -e 'using Pkg; Pkg.test()'` pass cleanly.

## Ideas & Candidate Solutions

### Candidate A: Dedicated `src/Portfolio/attribution.jl` (Recommended)
- Create `src/Portfolio/attribution.jl` containing all bet-level attribution logic, keeping `metrics.jl` focused on scalar portfolio wealth metrics (Sharpe, Sortino, Calmar).
- Include `attribution.jl` in `src/Portfolio/portfolio-module.jl` and export:
  `capture_ratio`, `edge_summary`, `partition_bets`, `compare_portfolios`, `shared_bet_sizing_attribution`.
- *Pros*: Modular, zero disruption to existing `simulation.jl` or `simulate.jl` code, highly testable.

### Candidate B: Inline Summary Fields in `PortfolioSummary`
- Add `capture_ratio::Float64` and `cap_weighted_win_rate::Float64` directly into `PortfolioSummary` during `simulate_portfolio`.
- *Pros*: Capture ratio is immediately accessible on `res.summary.capture_ratio`.
- *Cons*: Modifies `PortfolioSummary` struct definition, which might require cache invalidation or schema migration if stored in databases.
- *Recommendation*: Keep `PortfolioSummary` intact initially; provide property access or helper methods `capture_ratio(res)`.

## Work Log & Progress

- [2026-09-11 @antigravity] Task created following user request and completion of Task 007 / $r04$ attribution analysis. Scoped mathematical definitions of Capture Ratio ($e_{\text{win}} / e_{\text{loss}}$), shared-bet sizing decomposition ($\Delta s_i \times \text{settle}_i$), and first-class `Portfolio` API integration.
- [2026-09-18 @pi] Claimed in worktree `BayesianFootball-task012-attribution` on branch `feat/task012-portfolio-attribution`; implementing the dedicated production attribution module, typed summaries, breakdowns, displays, exports, and regression tests.
- [2026-09-18 @pi] Completed Candidate A: added typed summaries and pairwise decomposition, canonical odds/family breakdowns, legacy-safe reporting, top-level exports, and the dedicated regression suite without changing `PortfolioSummary` or persistence layouts.

## Verification & Findings

- `test/test_portfolio_attribution.jl`: **96 / 96 passed**.
- `test/unified_portfolio_tests.jl`: **792 / 792 passed**.
- `test/portfolio_tests.jl`: **92 / 92 passed**.
- `test/test_extension.jl`: **4 / 4 passed** (independent reviewer run).
- Full sequential suite (`julia +1.12.6 --startup-file=no --project -t 8 test/runtests.jl`): **4,249 passed, 1 pre-existing broken, exit 0**. Database-backed MatchDay ledger tests skipped because local `betdb` was unreachable; pure MatchDay layers passed.
- Parallel runner: **18 / 19 suites passed**; the sole failure was the documented T007 `SplitClockProbe` hidden dependency in `features_tests.jl`, while the sequential full suite passed.
- Independent principal review: **APPROVED** after legacy-schema, union-merge, push, bucket-alignment, and display regressions were added.
- `./scripts/todo.sh check`: passed at completion.

Semantic note: production defines a loss as `payoff < 0`, excluding pushes from both conditional edge means. The prototypes used `.!won` and therefore counted pushes as losses. Production also returns `NaN` when mean losing-bet edge is non-positive rather than emitting a negative Capture Ratio. Push-bearing or negative-edge prototype figures must therefore be recomputed before direct comparison; the published $r04$ ranges remain historical prototype results.
