# 017 — Graduate SmileScoreGrid Anti-Diagonal Reweighting and Close T011

| Field | Value |
|---|---|
| ID | 017 |
| Title | Graduate SmileScoreGrid Anti-Diagonal Reweighting and Close T011 |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-18 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `src/predictions/score_grids/types.jl`; `src/predictions/score_grids/kernels.jl`; `src/Portfolio/pricing.jl`; `test/test_score_grids.jl`; `docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md`; `current_development/grw_smile_spine/l01_loader.jl` |

## Context & Problem Statement

Ticket T011 revealed that in `src/predictions/score_grids/` and `src/predictions/score_computation/smile_poisson.jl`, `SmileScoreGrid` and `SmileScoreMatrix` treated the smile as a pricing-only sidecar (`grid` + `(λ_tot, φ)`). When pricing Over/Under, it evaluated the analytical Poisson CDF, but when pricing 1X2, BTTS, or sizing Kelly stakes in `Portfolio`, it read the un-smiled $(\lambda_h, \lambda_a)$ grid. This produced a 4.7–7.1 pp marginal incoherence and meant that Kelly never sized stakes off the smile.

In Task 016 (`current_development/grw_smile_spine/`), anti-diagonal grid reweighting was prototyped and verified to machine precision ($< 10^{-15}$ margin gap, sum=1, bit-identical $\phi \equiv 1$ identity). Staking from the reweighted grid lifted 5-parameter smile return from +588% to +606% and Sharpe from 1.495 to 1.611.

This task refactors the score grid architecture into an `AbstractScoreGrid` type hierarchy in `src/predictions/score_grids/` where `SmileScoreGrid` natively produces and encapsulates the anti-diagonal reweighted joint score tensor, unifies Over/Under pricing on tensor anti-diagonal summation, eliminates smile-specific ad-hoc branching in `Portfolio`, and permanently closes Ticket T011.

## Acceptance Criteria

- [x] Define `abstract type AbstractScoreGrid` with `StandardScoreGrid` and `SmileScoreGrid <: AbstractScoreGrid` in `src/predictions/score_grids/types.jl`.
- [x] Integrate preallocated anti-diagonal accumulation vectors (`grid_mass`, `ratio`) into `GridWorkspace` (or `SmileGridWorkspace`) to guarantee zero allocations in inner loops.
- [x] Implement in-place anti-diagonal reweighting kernel in `src/predictions/score_grids/kernels.jl` such that `compute_score_grid!(grid::SmileScoreGrid, ws, latents, i)` produces a valid joint score distribution $P_{\text{smile}}(h, a)$.
- [x] Invariants verified: total probability mass sums strictly to $1.0$ ($< 10^{-14}$), marginal totals match smile totals CDF ($\le 10^{-9}$), $\phi \equiv 1$ identity shortcut is bit-identical and un-shortcut path is bounded by draw truncation mass ($1 - \sum \text{grid}$), non-monotone $\phi$ curves are refused.
- [x] Unify `price_market!` for `AbstractScoreGrid` to read Over/Under directly by summing anti-diagonals of the reweighted tensor.
- [x] Remove ad-hoc smile branching from `src/Portfolio/pricing.jl` (`_fill_extra!`, etc.), allowing the Kelly allocator and Baker-McHale shrinkage to naturally size from `SmileScoreGrid`.
- [x] Add comprehensive unit tests in `test/test_score_grids.jl` verifying all invariants and pricing consistency.
- [x] Run verification tests ensuring no allocation regressions in `Evaluation` or `Portfolio`.
- [x] Mark Ticket T011 as `closed` in `docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md` and update `docs/tickets/README.md`.
- [x] `./scripts/todo.sh check` passes.

## Ideas & Candidate Solutions

- Embed anti-diagonal reweighting directly into `SmileScoreGrid`'s fill path: when `compute_score_grid!` is called, the underlying double-Poisson PMF is generated and then rescaled along anti-diagonals $G = h + a$ using `ratio[G] = target_cdf_diff / grid_mass[G]`.
- Truncation mass redistribution: For $G \ge 5$, tail mass $(1 - F_{\text{smile}}(4))$ is distributed across the grid tail $(G \ge 5)$, guaranteeing that $\sum_{h, a} S = 1.000$ exactly.

## Work Log & Progress

- [2026-09-18 @antigravity] Task claimed for `@pi` running `openai-codex/gpt-5.6-sol` in tmux session `pi_smile_scoregrid`, worktree `/home/james/bet_project/.worktrees/BayesianFootball-smile-scoregrid` on branch `feat/smile-scoregrid-reweighting`. Design aligned via /grill-me.
- [2026-09-18 @pi] Graduated the prototype into the typed score-grid kernels, unified Portfolio pricing/allocation on the reweighted tensor, added regression tests, and closed T011.

## Verification & Findings

- `test/test_score_grids.jl`: 69/69 assertions. Poisson and NegBin reweighted draws sum to one within $10^{-14}$; learned-strike CDF gaps are $\le 10^{-9}$; varying fixture/draw curves index correctly; typed and legacy MatchDay tensors/prices agree; identity tensors and Portfolio ledgers are bit-identical; forced identity reweighting is bounded by truncation mass; non-monotone curves hard-refuse typed and legacy builds; score-grid, Evaluation and Portfolio fixture hot paths allocate zero bytes.
- `test/latents_tests.jl`: 467/467.
- `test/evaluation_tests.jl`: 424/424.
- `test/unified_portfolio_tests.jl`: 707/707.
- `julia --project -t 8 test/runtests.jl`: 4,031 passed, one database-dependent test skipped. The unthreaded `Pkg.test()` run passed all task-related suites but hit the unrelated flaky player-lineup 100 µs timing gate at 122 µs; its isolated rerun passed, including the timing gate (5/5).
- CountLatents keeps the same raw kernel and passes the existing bit-identical Portfolio regressions. Smile `p_model`, Kelly `p_grid`, and Baker-McHale draw grids now all derive from `BookWorkspace.S` after anti-diagonal reweighting. The legacy `SmileScoreMatrix` path used by MatchDay applies the same kernel and reads every derivative from that tensor.
