# 020 — Sweep SlateDrawdown lambda risk budgets on GRW models

| Field | Value |
|---|---|
| ID | 020 |
| Title | Sweep SlateDrawdown lambda risk budgets on GRW models |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-18 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `src/Portfolio/implementations/risk.jl`, `src/Portfolio/calibrate.jl`, `eda/README.md`, Task 016, Task 018 |

## Context & Problem Statement

The Gaussian Random Walk (GRW) state-space models (`m05_joint_grw_smile_spine_w040` and
`m05_joint_grw_baseline`) have demonstrated significant predictive sharpness and high compounding
returns (+450% to +575%) under Option B (`1X2` + `Under 2.5`). However, they also exhibited steep
maximum drawdowns of **−42.8% to −43.8%** at the closing line under the default `SlateDrawdown(8.0)`.

As demonstrated by the scale-invariance law in `eda/README.md` and documented in
`src/Portfolio/calibrate.jl`, scalar trust modifications (`FlatTrust(0.20)` vs `FlatTrust(0.40)`)
are absorbed by `SlateDrawdown` and do not alter realized exposure. **The risk parameter $\lambda$ in
`SlateDrawdown(lambda)` is the sole active master dial that controls portfolio exposure and tail risk.**

This task conducts a systematic portfolio sweep across a range of risk parameters $\lambda$ to map
the Pareto frontier of **Terminal Return vs Annual Sharpe vs Max Drawdown vs Realized Slate Exposure**
for GRW models on Scottish Lower (tournaments 56/57, seasons 24/26, 628 buildable fixtures), evaluating
both close and T−25 environments.

No MCMC sampling is needed: the sweep executes against the completed, audited posterior latents
stored on `mcmc-beast`.

## Acceptance Criteria

- [ ] Prototype modular harness in `current_development/grw_risk_sweep/`:
  - `l01_risk_sweep.jl`: Helper functions to construct `SlateDrawdown(lambda)` policies, execute portfolio simulations across slates, and extract exposure, drawdown, Sharpe, and ROI metrics.
  - `r01_lambda_sweep.jl`: Multi-threaded research runner executing the $\lambda$ grid across models and environments.
- [ ] Sweep grid covering:
  - $\lambda \in [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0, 35.0, 45.0]$
  - Models: `m05_joint_grw_smile_spine_w040` and `m05_joint_grw_baseline` (un-smiled control).
  - Environments: Closing line and T−25 order book (raw and L2-calibrated).
- [ ] Verify execution:
  - Uses `:excise_pruned` to avoid Ticket T012 matrix dilution.
  - Zero heap allocation standard preserved in inner loop evaluation.
  - Bit-identical reproducibility across sequential and threaded execution.
- [ ] Deliverables:
  - `results/lambda_sweep_summary.csv`
  - `results/pareto_frontier.csv`
  - `results/LAMBDA_RISK_SWEEP_REPORT.md` documenting the optimal operational $\lambda$ to achieve target drawdown (e.g. $\le 20\%$) with maximal Sharpe.
- [ ] `./scripts/todo.sh check` passes cleanly.

## Ideas & Candidate Solutions

- Use `calibrate_lambda` from `src/Portfolio/calibrate.jl` as an analytical reference point for targeting specific exposure levels (e.g. 10%, 15%, 20%).
- For T−25 calibrated books, test whether higher $\lambda$ further stabilizes the already reduced −16.8% drawdown found in Task 016.
- Record the ratio of realized maximum drawdown to nominal target $1 - D = 1 - e^{\log(\beta)/\lambda}$ to verify the empirical 1.15× overshoot constant noted in `src/Portfolio/implementations/risk.jl`.

## Work Log & Progress

- [2026-09-18 @antigravity] Initialized Task 020 in worktree `/home/james/bet_project/.worktrees/BayesianFootball-grw-risk-sweep` on branch `feat/grw-slatedrawdown-lambda-sweep`.
- [2026-09-18 @pi] Claimed in session `pi_grw_risk_sweep`.

## Verification & Findings

Not run yet.
