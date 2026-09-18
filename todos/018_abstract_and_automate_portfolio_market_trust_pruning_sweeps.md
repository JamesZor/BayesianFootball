# 018 — Abstract and automate portfolio market trust pruning sweeps

| Field | Value |
|---|---|
| ID | 018 |
| Title | Abstract and automate portfolio market trust pruning sweeps |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-18 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `eda/eda_asymmetric_selection_trust.jl`, `eda/eda_multitier_trust.jl`, `eda/stochastic_control_common.jl`, `current_development/grw_smile_spine/r08_trust_sweep.jl`, `docs/tickets/T012-zero-trust-market-reprices-the-portfolio.md` |

## Context & Problem Statement

Historical research across multiple streams demonstrated that market trust and selection pruning
are crucial determinants of portfolio alpha:
1. **Knapsack Shadow Price & Capacity Cannibalization** (`eda/`): Marginal fringe markets consume
   bankroll capacity under the slate cap (20%) and crowd out core 1X2 and Under 2.5 alpha.
2. **Jensen Tail Inflation** (`eda/`): Deep under markets (e.g. Under 0.5, Under 1.5) suffer from
   $E[e^{-\Lambda}] \ge e^{-E[\Lambda]}$, causing uncalibrated Poisson/smile models to perceive
   massive phantom edges on 0–0 and 1–0 scorelines (−30% to −100% ROI).
3. **Task 016 Trust Sweep** (`current_development/grw_smile_spine/r08_trust_sweep.jl`): Evaluated 66
   fringe additions to Option B and proved that 45/66 degraded portfolio performance.
4. **Ticket T012**: Merely including a market with `trust = 0` widens the payoff matrix and
   dilutes Baker-McHale's per-fixture shrinkage factor $k$, repricing core stakes by up to 22 pp.

Currently, pruning studies are implemented as monolithic, ad-hoc 500-line scripts tied to specific
model folders. We need an abstracted, modular portfolio market pruning harness (prototyped in
`current_development/market_pruning_harness/` with loader `l01_pruning.jl` and runner `r01_sweep.jl`)
that can rapidly screen and optimize market trust policies across any Bayesian model.

## Acceptance Criteria

- [x] Prototype modular harness in `current_development/market_pruning_harness/` with:
  - `l01_pruning.jl`: Abstracted market catalog (1X2, O/U 0.5 through 4.5, BTTS Yes/No), policy
    generators (stepwise addition, multi-tier conviction), and execution runner with T012 toggle.
  - `r01_sweep.jl`: Human-readable research runner executing the cross-paradigm benchmark.
- [x] Address Ticket T012 via explicit mode toggle (`:excise_pruned` vs `:retain_zero_trust`):
  - `:excise_pruned` removes wholly zero-trust markets from `BookSpec` dynamically to avoid payoff
    matrix expansion and shrinkage/allocator dilution. Selection-level excision is not claimed:
    `BookSpec` admits complete markets.
  - `:retain_zero_trust` retains legacy matrix expansion to allow direct parity checks against
    historical EDA reports.
- [x] Two-phase pruning protocol:
  - Phase 1: Stepwise single-line addition against core basket (1X2 + Under 2.5) to isolate added ROI,
    core stake cannibalization, and net return.
  - Phase 2: Categorical conviction tiering optimization on the surviving accretive basket.
- [x] Cross-paradigm benchmark across 3 canonical models on Scottish Lower 24/26:
  - `m05_joint_grw_smile_spine_w040`
  - `m05_joint_production_wealth` (un-smiled baseline)
  - `m12_joint_hybrid_synergy`
- [x] Performance: Native zero-allocation score-grid/book path preserved; model paradigms execute in
  parallel on the 40/43-fold posterior panels. The sequential and threaded runs produced
  bit-identical CSV artefacts.
- [x] Reporting: Generates summary CSVs, gate verification tables, and a structured Markdown report
  contrasting accretive vs toxic lines across models.
- [x] Repository integrity: `./scripts/todo.sh check` passes.

## Ideas & Candidate Solutions

- **Configurable Catalog**: Define market menu via typed specifications so other leagues (e.g.
  English leagues with Asian Handicaps / DNB) can reuse the same pruning pipeline.
- **Dynamic BookSpec vs Downstream Mask**: Implement `:excise_pruned` by filtering the input
  `BookSpec.markets` prior to `build_books` or inside `build_books`, ensuring the payoff matrix
  dimension matches only the active tradeable lines.
- **Re-use anti-diagonal reweighting**: For `SmileLatents` models, ensure `SmileScoreGrid`
  anti-diagonal reweighting (PR #32) is preserved for all totals.

## Work Log & Progress

- [2026-09-18 @antigravity] Initialized Task 018. Merged PR #32 scoregrid reweighting branch with
  grw-smile-spine prototype scripts in worktree `BayesianFootball-market-pruning`.
- [2026-09-18 @pi] Claimed in session `pi_market_pruning`, worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-pruning`.
- [2026-09-18 @pi] Added the prototype pair under
  `current_development/market_pruning_harness/`: a typed/extensible market catalog, explicit
  `:excise_pruned` / `:retain_zero_trust` BookSpec construction, Phase-1 directional screening,
  Phase-2 conviction grids with Pareto marking, S0/S2 gates, CSV/report writers, and the
  cross-paradigm runner.
- [2026-09-18 @pi] Synced the prototype pair to `/root/BF_market_pruning` on `mcmc-beast` and
  executed `r01_sweep.jl` in tmux session `market_pruning_sweep` with 16 Julia threads. No MCMC was
  launched; the runner read relational `CountLatents` for the historical standard fits and rebuilt
  the detached `SmileLatents` panel from the completed spine chains.
- [2026-09-18 @pi] Re-ran the complete study with model-paradigm jobs threaded. Every numerical CSV
  was bit-identical to the sequential run. Synced the final artefacts back into the worktree.

## Verification & Findings

### Gates and integrity

- Loader parses and loads on `mcmc-beast`; the catalog smoke gate confirms seven markets and the
  exact active Option B market set (`1X2`, O/U 1.5, O/U 2.5).
- **S0 passed / T012 reproduced for all three paradigms.** Keeping the full zero-trust menu changed
  terminal return by **−53.02 pp** (`m05`), **−14.49 pp** (`m12`), and **−126.26 pp** (smile spine)
  relative to market excision. Every ledger changed despite identical trust weights.
- **S1 passed where an exact-contract publication exists.** The spine reproduced Task 016 exactly:
  632 books, 1,232 bets, +469.353510% return. The standard `m05` and `m12` rows are recorded as new
  Option B benchmarks rather than falsely compared with published runs under different policies.
- **S2 passed.** The smile allocator's 628 built books produced 3,140 strike checks with maximum
  score-tensor/CDF gap **2.998e−15** (tolerance `1e−9`). Count arms correctly report no smile checks.
- `test/unified_portfolio_tests.jl`: **707 / 707** passed on `mcmc-beast`.
- Sequential vs threaded result CSVs: all six numerical artefacts were bit-identical.
- `./scripts/todo.sh check`: clean.

### Scientific findings

- **No candidate line was robustly accretive across all three paradigms.** The structured report
  therefore promotes no shared fringe basket.
- **Under 0.5 was toxic in all three models** (net return changes −39.50, −14.70, −164.91 pp for
  `m05`, `m12`, spine respectively), confirming the deep-under/Jensen warning.
- **BTTS No was toxic in all three close-book sweeps** (−15.09, −3.54, −41.63 pp). The prior
  accretive BTTS-No result does not transfer automatically to this excised close-book contract.
- Under 1.5 and Under 3.5 were accretive but cannibalizing for both standard count paradigms, yet
  toxic for the smile spine. The spine had **zero** accretive additions: positive standalone ROI
  on U4.5/O2.5/O3.5/O4.5/BTTS Yes was overwhelmed by portfolio repricing and capacity effects.
- The lineup hybrid alone found O2.5/O3.5/O4.5 net-accretive; the team-state control and smile did
  not. This is model dependence, not a production market menu.
- In Phase 2, every model's maximum-growth grid cell used `tau3 = 0.0`; the low-trust fringe tier
  did not beat excision. The Pareto table records the growth/drawdown trade-off rather than naming
  one unconstrained winner.

### Artefacts

- `current_development/market_pruning_harness/results/sweep_summary.csv`
- `current_development/market_pruning_harness/results/line_screening.csv`
- `current_development/market_pruning_harness/results/conviction_tiers.csv`
- `current_development/market_pruning_harness/results/t012_contrast.csv`
- `current_development/market_pruning_harness/results/gates.csv`
- `current_development/market_pruning_harness/results/model_inventory.csv`
- `current_development/market_pruning_harness/results/MARKET_PRUNING_REPORT.md`

### Address correction

The work-package prompt's spine UUID (`582035c0-e145-44f7-9f40-89e253883a4c`) does not match the
completed Task-016 artefact recorded by r06/r08 (`582035c0-e145-44f7-9f40-89e25388e79a`). The
runner resolves by canonical name and requires the latter immutable UUID, rather than silently
accepting either address.
