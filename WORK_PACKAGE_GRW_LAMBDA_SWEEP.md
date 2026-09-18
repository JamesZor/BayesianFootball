# Work Package: Task 020 — Sweep SlateDrawdown lambda risk budgets on GRW models

You are **Pi**, assigned to execute **Task 020** in worktree `/home/james/bet_project/.worktrees/BayesianFootball-grw-risk-sweep` on branch `feat/grw-slatedrawdown-lambda-sweep`.

---

## 1. Context & Objective

The Gaussian Random Walk (GRW) state-space models (`m05_joint_grw_smile_spine_w040` and `m05_joint_grw_baseline`) have demonstrated excellent predictive sharpness and high compounding returns (+450% to +575%) under Option B (`1X2` + `Under 2.5`). However, they exhibited steep maximum drawdowns of **−42.8% to −43.8%** at the closing line under the default `SlateDrawdown(8.0)`.

As demonstrated by the scale-invariance law in `eda/README.md` and documented in `src/Portfolio/calibrate.jl`:
- Modifying trust weights proportionally (`FlatTrust(0.20)` vs `FlatTrust(0.40)`) is completely absorbed by `SlateDrawdown`.
- **The risk parameter $\lambda$ in `SlateDrawdown(lambda)` is the sole active master dial that controls portfolio exposure and tail risk.**

Your objective is to systematically sweep $\lambda$ across the full Scottish Lower 24/26 evaluation panel (628 buildable fixtures) on both the closing-line and T−25 books (raw and L2-calibrated) to map the **Pareto frontier of Terminal Return vs Annual Sharpe vs Max Drawdown vs Realized Exposure**.

**NO MCMC SAMPLING IS REQUIRED.** The 43-fold walk-forward posterior latents are already fitted, audited, and stored on `mcmc-beast`.

---

## 2. Models & Data Panels

Evaluate the following model fits across the common Scottish Lower 24/26 panel (628 buildable fixtures):
1. **`m05_joint_grw_smile_spine_w040`**:
   - Spine run UUID: `582035c0-e145-44f7-9f40-89e25388e79a`
   - Reconstruct or load detached `SmileLatents` using the patterns established in `current_development/grw_smile_spine/` and `current_development/market_pruning_harness/l01_pruning.jl`.
2. **`m05_joint_grw_baseline`**:
   - The un-smiled GRW control fit (UUID `b0961bc4` / canonical baseline fit).
3. **`m12_joint_hybrid_synergy`** (optional baseline reference for cross-paradigm contrast).

---

## 3. The Experimental Grid

### A. Lambda Risk Parameter Grid
$$\lambda \in [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0, 35.0, 45.0]$$
- Note: $\lambda = 8.0$ reproduces our published Option B baseline.
- Higher $\lambda$ tightens the tail budget: $\mathbb{E}[(1 + k R)^{-\lambda}] \le 1$.
- Calculate nominal target floor $D = e^{\log(\beta)/\lambda}$ (with $\beta = 0.01$) vs actual realized max drawdown to check the empirical 1.15× overshoot constant.

### B. Environments
1. **Closing line** (`ds.odds`, de-vigged Betfair time-weighted average close).
2. **T−25 raw** (`point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))`).
3. **T−25 L2-calibrated** (using `GenerativeRateCalibrator` with `InverseGaussianLaw(w_base = 0.25, sigma = 0.35)` and `book_as_of_minutes = -25.0`, dropping $\phi$ at pricing time as validated in Task 016).

### C. Market Configuration & Staking Policy
- Core basket: Option B (`1X2` and `Under 2.5`).
- Ensure `:excise_pruned` is used (so no zero-trust markets contaminate the payoff matrix, resolving Ticket T012).
- Zero heap allocation standard in inner-loop portfolio evaluation.

---

## 4. Implementation Structure

Create the prototype pair in `current_development/grw_risk_sweep/`:
- **`l01_risk_sweep.jl`**:
  - Model loader and latent extractor (supporting both `CountLatents` and `SmileLatents`).
  - Policy constructor for `SlateDrawdown(lambda)` with `:excise_pruned`.
  - Simulation runner extracting: terminal return %, CAGR %, annual Sharpe, Sortino, Calmar, max drawdown %, win rate, total bets, and mean slate exposure.
  - Report / CSV serialization helpers.
- **`r01_lambda_sweep.jl`**:
  - Automated runner executing the grid.
  - Support multithreaded execution across models/environments.
  - Verify bit-identical results between sequential and threaded paths.

Outputs should be written to `current_development/grw_risk_sweep/results/`:
- `lambda_sweep_summary.csv`
- `pareto_frontier.csv`
- `overshoot_calibration.csv`
- `LAMBDA_RISK_SWEEP_REPORT.md`

---

## 5. Execution Protocol on `mcmc-beast`

1. Sync local prototype files to `mcmc-beast`:
   ```bash
   rsync -avz --exclude='.git' current_development/grw_risk_sweep/ root@mcmc-beast:/root/BF_grw_risk_sweep/
   ```
2. Execute the runner in a remote tmux session:
   ```bash
   ssh root@mcmc-beast "tmux new-session -d -s grw_lambda_sweep 'cd /root/BF_grw_risk_sweep && julia --project -t 16 current_development/grw_risk_sweep/r01_lambda_sweep.jl 2>&1 | tee /root/BF_grw_risk_sweep/run.log'"
   ```
3. Monitor completion, verify gates, and sync results back:
   ```bash
   rsync -avz root@mcmc-beast:/root/BF_grw_risk_sweep/results/ current_development/grw_risk_sweep/results/
   ```

---

## 6. Completion & Verification Checklist

- [ ] `l01_risk_sweep.jl` and `r01_lambda_sweep.jl` implemented.
- [ ] Sweep executed across all $\lambda$ values and environments.
- [ ] Pareto frontier identified: what $\lambda$ delivers target drawdown $\le 20\%$ while maximizing Sharpe?
- [ ] Output CSVs and `LAMBDA_RISK_SWEEP_REPORT.md` committed.
- [ ] `todos/020_sweep_slatedrawdown_lambda_risk_budgets_on_grw_models.md` and `todos/README.md` updated to `COMPLETED`.
- [ ] `./scripts/todo.sh check` passes.
