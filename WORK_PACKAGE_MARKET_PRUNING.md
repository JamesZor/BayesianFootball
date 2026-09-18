# WORK PACKAGE — Abstract and Automate Portfolio Market Trust Pruning Sweeps (Task 018)

> **Assignee:** `pi` (OpenAI Codex / `gpt-5.6-sol` with `--thinking high`)  
> **Session Target:** `pi_market_pruning` in tmux  
> **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-pruning`  
> **Branch:** `feat/market-pruning-harness`  
> **Canonical Guide:** `AGENTS.md`, `docs/prototype_runner_style_guide.md`

---

## 1. Executive Summary & Objective

In this repository, Bayesian predictive models generate joint score distributions or marginal intensities, which are converted into probability grids and allocated across betting markets via Kelly log-utility optimization (`src/Portfolio/`). 

Extensive research across multiple streams (`eda/`, `current_development/grw_smile_spine/r08_trust_sweep.jl`) demonstrated that **market trust and selection pruning are crucial determinants of portfolio alpha**:
1. **Knapsack Shadow Price & Capacity Cannibalization** (`eda/`): Under a portfolio slate cap (e.g. 20% of bankroll), fringe markets consume scarce bankroll capacity and crowd out the highest-margin core alpha (Home and Under 2.5).
2. **Jensen Tail Distortion** (`eda/`): Because $E[e^{-\Lambda}] \ge e^{-E[\Lambda]}$, uncalibrated models over-price deep under markets (Under 0.5, Under 1.5), generating severe negative ROI (−30% to −100%) on 0–0 and 1–0 scorelines.
3. **Smile Model Pruning Reality** (`r08_trust_sweep.jl`): Testing 66 fringe market additions showed that 45/66 degraded portfolio performance. Under 1.5 manufactured a +4.1 pp phantom edge under the smile, losing heavily (−6.3% ROI vs +37.7% in baseline), while BTTS No was accretive (+3.5% to +8.2% ROI).
4. **Ticket T012 (Shrinkage Dilution)**: Declaring a market in `BookSpec` with `trust = 0` widens the payoff matrix $R$, changing Baker-McHale's per-fixture shrinkage factor $k$ and arbitrarily shifting core stakes by up to 22 pp.

**Your Goal:**
Abstract market trust pruning out of monolithic, ad-hoc runner scripts into a clean, reusable evaluation and portfolio pruning harness under `current_development/market_pruning_harness/`. Automate the execution of market sweeps across different market categories and evaluate the cross-paradigm benchmark models on Scottish Lower 24/26.

---

## 2. Architectural Blueprint & Design Choices

Following user alignment via `/grill-me`, implement the system as a prototype pair:
- **Loader (`l01_pruning.jl`)**: The algorithmic machinery, data types, market menus, policy generators, and solver dispatch.
- **Runner (`r01_sweep.jl`)**: The executable research notebook adhering strictly to `docs/prototype_runner_style_guide.md`.

### A. Modular Market Catalog
Create an extensible market menu definition that cleanly groups markets, lines, and selection directions:
- **Core Basket**: 1X2 (`:home`, `:draw`, `:away`), O/U 2.5 (`:under_25`, `:over_25`).
- **Totals Ladder**: O/U 0.5 (`:under_05`, `:over_05`), O/U 1.5 (`:under_15`, `:over_15`), O/U 3.5 (`:under_35`, `:over_35`), O/U 4.5 (`:under_45`, `:over_45`).
- **Both Teams To Score**: BTTS (`:btts_yes`, `:btts_no`).
- *(Extensibility)*: Design the catalog data structure so additional markets (Draw No Bet, Asian Handicap, Team Totals) can be configured for leagues with higher liquidity in the future.

### B. Solution for Ticket T012 (`:excise_pruned` vs `:retain_zero_trust`)
Implement an explicit toggle in the harness:
- `:excise_pruned` (Recommended Default): When building the `BookSpec` for a given trust policy, dynamically filter `BookSpec.markets` so that any market where all selections have zero trust is **completely excised from the payoff matrix**. This guarantees Baker-McHale's shrinkage scalar $k$ and the Kelly solver coordinate space are not polluted by inactive columns.
- `:retain_zero_trust` (Legacy Diagnostic Mode): Keeps the full market menu in `BookSpec` with zero-trust downstream, enabling direct numerical parity checks against legacy runs in `eda/` and `r08_trust_sweep.jl`.
- Provide an automated gate comparing `:excise_pruned` vs `:retain_zero_trust` to quantify the exact T012 shift.

### C. Two-Stage Pruning Protocol
1. **Phase 1 — Stepwise Line Screening**:
   - Baseline $P_0$: Core basket (1X2 + Under 2.5) at Option B trust (`1.0` or `1.0 / 1.4`).
   - For each candidate line (e.g. `+U0.5`, `+U1.5`, `+U3.5`, `+U4.5`, `+O2.5`, `+O3.5`, `+O4.5`, `+BTTS_yes`, `+BTTS_no`):
     - Solve joint Kelly allocation.
     - Compute: `added_roi_pct`, `core_stake_vs_p0`, `delta_core_roi_pp`, `delta_return_pp`.
     - Flag lines as **accretive**, **neutral**, or **toxic/cannibalizing**.
2. **Phase 2 — Conviction Tier Optimization**:
   - On the surviving basket of accretive selections, evaluate multi-tier conviction structures:
     - Tier 1 (High conviction / Alpha core): e.g. Under 2.5, Home $\tau \in [0.30, 0.45]$.
     - Tier 2 (Diversifiers): e.g. Draw, Away, BTTS No $\tau \in [0.15, 0.25]$.
     - Tier 3 (Fringe / Tail probes): surviving totals $\tau \in [0.00, 0.10]$.
   - Report the Pareto frontier of Growth vs Max Drawdown.

---

## 3. Benchmark Target Models (Scottish Lower 24/26)

Evaluate the harness across three canonical model paradigms loaded from `mcmc_experiments` (or local artifact cache):
1. **`m05_joint_grw_smile_spine_w040`**: Task 016 GRW smile model (uses anti-diagonal reweighted `SmileScoreGrid` via PR #32).
   - Run UUID: `582035c0-e145-44f7-9f40-89e253883a4c` (experiment: `scottish_lower_grw_smile_spine`).
2. **`m05_joint_production_wealth`**: Un-smiled Gen 1/3 baseline control.
   - Run UUID: `5eff755c-3591-48d1-a2cc-5fc2744ddf88` (experiment: `scottish_lower_joint_2426`).
3. **`m12_joint_hybrid_synergy`**: Gen 4 RAPM player lineup hybrid.
   - Run UUID: `132df5c2-c742-4e95-8693-3aeb2b2cbaef` (experiment: `scottish_lower_joint_player_2426`).

---

## 4. Verification Gates & Standards

- **Gate S0 (Ticket T012 Parity & Quantification)**: Measure and document the difference between `:excise_pruned` and `:retain_zero_trust` on the baseline core policy.
- **Gate S1 (Option B Reproducibility)**: Verify that running the baseline policy $P_0$ reproduces the published Option B performance metrics for the tested models.
- **Gate S2 (Score-Grid Coherence)**: For `SmileLatents`, verify that all totals pricing and sizing pass through the unified `SmileScoreGrid` reweighted tensor (PR #32).
- **Performance**: Zero heap allocations in the inner loop `simulate_portfolio` evaluation; use `ThreadPinning.pinthreads(:cores)` and `BLAS.set_num_threads(1)`.
- **Output Artifacts**: Save all results to `current_development/market_pruning_harness/results/`:
  - `sweep_summary.csv`
  - `line_screening.csv`
  - `conviction_tiers.csv`
  - `t012_contrast.csv`
  - `MARKET_PRUNING_REPORT.md`

---

## 5. Execution Instructions

1. Work exclusively within `/home/james/bet_project/.worktrees/BayesianFootball-market-pruning`.
2. Inspect prior art:
   - `eda/eda_asymmetric_selection_trust.jl`
   - `eda/eda_multitier_trust.jl`
   - `eda/stochastic_control_common.jl`
   - `current_development/grw_smile_spine/r08_trust_sweep.jl`
   - `docs/tickets/T012-zero-trust-market-reprices-the-portfolio.md`
3. Implement `l01_pruning.jl` and `r01_sweep.jl`.
4. Run the benchmark sweep and verify all gates.
5. Update `todos/018_abstract_and_automate_portfolio_market_trust_pruning_sweeps.md` with full findings.
6. Ensure `./scripts/todo.sh check` passes cleanly.
