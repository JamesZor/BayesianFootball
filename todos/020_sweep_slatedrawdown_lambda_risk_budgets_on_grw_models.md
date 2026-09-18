# 020 — Sweep SlateDrawdown lambda risk budgets on GRW models

| Field | Value |
|---|---|
| ID | 020 |
| Title | Sweep SlateDrawdown lambda risk budgets on GRW models |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-18 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `current_development/grw_risk_sweep/`, `src/Portfolio/implementations/risk.jl`, `src/Portfolio/calibrate.jl`, `eda/README.md`, Task 016, Task 018 |

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

- [x] Prototype modular harness in `current_development/grw_risk_sweep/`:
  - `l01_risk_sweep.jl`: Helper functions to construct `SlateDrawdown(lambda)` policies, execute portfolio simulations across slates, and extract exposure, drawdown, Sharpe, and ROI metrics.
  - `r01_lambda_sweep.jl`: Multi-threaded research runner executing the $\lambda$ grid across models and environments.
- [x] Sweep grid covering:
  - $\lambda \in [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0, 35.0, 45.0]$
  - Models: `m05_joint_grw_smile_spine_w040` and `m05_joint_grw_baseline` (un-smiled control).
  - Environments: Closing line and T−25 order book (raw and L2-calibrated).
- [x] Verify execution:
  - Uses `:excise_pruned` to avoid Ticket T012 matrix dilution.
  - Zero heap allocation standard preserved in inner loop evaluation.
  - Bit-identical reproducibility across sequential and threaded execution.
- [x] Deliverables:
  - `results/lambda_sweep_summary.csv`
  - `results/pareto_frontier.csv`
  - `results/LAMBDA_RISK_SWEEP_REPORT.md` documenting the optimal operational $\lambda$ to achieve target drawdown (e.g. $\le 20\%$) with maximal Sharpe.
- [x] `./scripts/todo.sh check` passes cleanly.

## Ideas & Candidate Solutions

- Use `calibrate_lambda` from `src/Portfolio/calibrate.jl` as an analytical reference point for targeting specific exposure levels (e.g. 10%, 15%, 20%).
- For T−25 calibrated books, test whether higher $\lambda$ further stabilizes the already reduced −16.8% drawdown found in Task 016.
- Record the ratio of realized maximum drawdown to nominal target $1 - D = 1 - e^{\log(\beta)/\lambda}$ to verify the empirical 1.15× overshoot constant noted in `src/Portfolio/implementations/risk.jl`.

## Work Log & Progress

- [2026-09-18 @antigravity] Initialized Task 020 in worktree `/home/james/bet_project/.worktrees/BayesianFootball-grw-risk-sweep` on branch `feat/grw-slatedrawdown-lambda-sweep`.
- [2026-09-18 @pi] Claimed in session `pi_grw_risk_sweep`.
- [2026-09-18 @pi] Added `l01_risk_sweep.jl` and the numbered `r01_lambda_sweep.jl` workflow:
  immutable run verification, relational baseline loading, detached smile reconstruction,
  market-level Option-B excision, cached books, sequential/threaded sweeps, canonical exact-result
  comparison, Pareto/overshoot analysis, and report/CSV writers.
- [2026-09-18 @pi] Synced the repository (excluding `.git`, `.cache`, `data`, and `.env`) to
  `/root/BF_grw_risk_sweep` on `mcmc-beast`, reused the canonical cache, and executed the runner in
  tmux session `grw_lambda_sweep` with 16 Julia threads. No MCMC was launched.
- [2026-09-18 @pi] Synced the four final artifacts back to this worktree and recorded the measured
  panel-coverage limitation rather than padding fixtures absent from the T−25 order book.
- [2026-09-18 @antigravity] Evaluated per-slate logarithmic compounding growth rate $g_{\text{slate}}$ and calculated the four statistical moments (mean, variance/std, skewness, excess kurtosis) across all 60 cells, archiving `slate_pnl_moments.csv`.
- [2026-09-18 @antigravity] Updated `MatchDay` operational default in `src/MatchDay/calibration.jl` to `SlateDrawdown(28.0)` with keyword argument support, verified test suite (193/193 pass), and authored `NOTE_DEFAULT_CONFIGURATION_UPDATE.md`.

## Verification & Findings

### Gates and execution

- Immutable run addresses resolved and **86/86 folds were converged**. The database persists the
  baseline under its historical name `m05_wealth_grw`; the research-facing label remains
  `m05_joint_grw_baseline`.
- The common fold-held-out latent panel was 710 fixtures. The exact `:excise_pruned` Option-B book
  built **628 close fixtures** as required. T−25 contained only 611 quoted fixtures and 594 with
  complete active markets, so the T−25 sweep honestly uses **594 buildable fixtures** rather than
  fabricating the requested 628. Close and T−25 results are therefore not fixture-paired.
- T−25 inversion shifted **573/594 (96.46%)** fixtures; the other 21 used the calibrator's declared
  `fallback = :identity`. The calibrated smile arm drops φ at pricing time, matching Task 016's
  validated `t25_inv_grid` control.
- The production `Portfolio.price_fixture!` kernel allocated **0 bytes** for raw CountLatents,
  raw SmileLatents, and both φ-dropped calibrated CountLatents checks.
- The λ=8 close smile row exactly reproduced Task 018's excised Option-B reference:
  **+576.133102%**, 1,188 bets, **−42.813338%** max drawdown.
- All **60/60** model × environment × λ cells completed. Sequential execution took 5.05 s and
  threaded execution 0.33 s after book construction; every summary value, daily state, trajectory,
  ledger/attribution column and canonical result SHA-256 was bit-identical.
- Mean realised exposure was non-increasing in λ for every model/environment cell.
- `Meta.parseall` passed for both new Julia files; `./scripts/todo.sh check` passes cleanly.

### Scientific findings

- **Raw-book operational knee: λ=28.** It is the first grid point below a realised 20% max
  drawdown in all four raw model/book cells. At λ=23 all four remain just above the ceiling
  (20.38%–23.38%). At λ=28:
  - close baseline: **−19.65% DD**, Sharpe 1.633, +92.26% terminal return;
  - close smile: **−17.08% DD**, Sharpe 1.744, +109.92%;
  - T−25 baseline: **−17.22% DD**, Sharpe 1.657, +88.97%;
  - T−25 smile: **−17.81% DD**, Sharpe 1.404, +84.89%.
- **Literal maximum-Sharpe-under-20% choice: λ=45.** Sharpe continues rising as exposure falls, so
  the two-objective drawdown/Sharpe frontier collapses to the tightest tested cell in every
  environment. λ=45 yields 11.00%–12.73% raw drawdowns but only +49.19%–61.54% terminal return.
  Thus λ=45 is the strict criterion winner; λ=28 is the less conservative growth-preserving knee.
- L2 calibration already reduces T−25 risk substantially: the smile meets 20% at **λ=8**
  (−16.86% DD, Sharpe 1.780), while the baseline first meets it at **λ=10** (−17.35%, Sharpe 2.000).
  Applying λ=45 as well drives exposure to 1.23%–1.57% and drawdown to 3.45%–4.31%.
- The historical **1.15× overshoot is not universal**. It is a useful approximation for raw smile
  and raw T−25 paths at tighter λ, but the close baseline reaches ~1.31× while L2-calibrated paths
  are only ~0.35×–0.48×. Risk calibration must remain book/calibration-regime specific.
- Every λ is non-dominated when terminal return and exposure are included alongside drawdown and
  Sharpe: tightening λ exchanges growth for lower exposure. The two-objective drawdown/Sharpe
  frontier alone selects λ=45 because both metrics improve over this grid.

### Artifacts

- `current_development/grw_risk_sweep/results/lambda_sweep_summary.csv`
- `current_development/grw_risk_sweep/results/slate_pnl_moments.csv`
- `current_development/grw_risk_sweep/results/pareto_frontier.csv`
- `current_development/grw_risk_sweep/results/overshoot_calibration.csv`
- `current_development/grw_risk_sweep/results/LAMBDA_RISK_SWEEP_REPORT.md`
- `current_development/match_day_inference/NOTE_DEFAULT_CONFIGURATION_UPDATE.md`
