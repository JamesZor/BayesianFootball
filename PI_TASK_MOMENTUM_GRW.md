# Momentum MultiScale GRW Dynamics (Scottish Lower Phase 1) — Pi Task

You are tasked with executing **TODO 022: Prototype Momentum MultiScale GRW Dynamics** on Scottish Lower football (tournaments 56 & 57, 40-fold walk-forward cohort over seasons 24/25 and 25/26, 710 fixtures).

### Your Model & Operating Mode
- You are running on **`openai-codex/gpt-6-astra` with high thinking**.
- **SOLO AGENT**: Execute all research, mathematical modeling, Turing coding, and verification directly. **Do NOT spawn subagents.**
- Use your deep mathematical reasoning to derive the 2nd-order state-space momentum dynamics, verify stationarity and damping conditions, and design elegant, AD-safe Turing.jl code.

### Canonical Work Package Prompt:
Read your complete specification, context, mathematical definitions, and execution plan in:
[`experiments/scottish_lower/10_momentum_multiscale_grw/WORK_PACKAGE_PROMPT.md`](experiments/scottish_lower/10_momentum_multiscale_grw/WORK_PACKAGE_PROMPT.md)

### Key Files Already Scaffolded:
- Loader: [`experiments/scottish_lower/10_momentum_multiscale_grw/l10_momentum_grw_loader.jl`](experiments/scottish_lower/10_momentum_multiscale_grw/l10_momentum_grw_loader.jl)
- Stage 1 Smoke Runner: [`experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl`](experiments/scottish_lower/10_momentum_multiscale_grw/r10_momentum_smoke.jl)
- Stage 2 Production Grid Runner: [`experiments/scottish_lower/10_momentum_multiscale_grw/r20_momentum_production_grid.jl`](experiments/scottish_lower/10_momentum_multiscale_grw/r20_momentum_production_grid.jl)
- Stage 3 Evaluation Runner: [`experiments/scottish_lower/10_momentum_multiscale_grw/r30_momentum_evaluation.jl`](experiments/scottish_lower/10_momentum_multiscale_grw/r30_momentum_evaluation.jl)
- Benchmark Report: [`experiments/scottish_lower/10_momentum_multiscale_grw/README.md`](experiments/scottish_lower/10_momentum_multiscale_grw/README.md)
- TODO Tracking: [`todos/022_prototype_momentum_multiscale_grw_dynamics.md`](todos/022_prototype_momentum_multiscale_grw_dynamics.md)

### Remote Compute Node (`mcmc-beast`):
- Compute node is ready and idle.
- A matched worktree is checked out at `/root/BF_momentum_grw` on branch `feat/scottish-lower-momentum-grw` with `.env` and `Manifest.toml` configured.
- Julia executable on beast is `/root/.juliaup/bin/julia`.
- When running heavy MCMC sampling across the 40 folds (Stage 2), sync code to `mcmc-beast` (`git push` / `git -C /root/BF_momentum_grw pull`) and execute via `ssh root@mcmc-beast "..."` in a detached tmux session.

### Execution Workflow:
1. **Stage 0 (Mathematical Research & Design)**:
   - Study `src/models/pregame/components/dynamics/team_level/multiscale.jl`.
   - Formulate `MomentumMultiScaleGRW`:
     $$\alpha_t = \alpha_{t-1} + v_{t-1} + \sigma_\alpha \epsilon_t, \quad v_t = \phi v_{t-1} + \sigma_v \eta_t$$
     where $v_t$ is team directional velocity (form momentum) and $\phi \in [0, 1)$ governs persistence.
   - Implement the component, submodel function, and trajectory reconstruction in `l10_momentum_grw_loader.jl`.
   - Ensure zero heap allocations on compiled ReverseDiff gradient tapes.
2. **Stage 1 (Smoke Gate)**:
   - Run `r10_momentum_smoke.jl` on folds 1, 20, 40 across the 3 arms:
     - `m01_poisson_time_decay`
     - `m02_poisson_grw_1st_order`
     - `m03_poisson_momentum_grw`
   - Verify ReverseDiff compilation, 0 divergences, R̂ ≤ 1.05, ESS ≥ 200, and score-grid coherency.
3. **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
   - Run `r20_momentum_production_grid.jl` on `mcmc-beast`.
   - Persist fits to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_momentum_grw`.
4. **Stage 3 (Evaluation & Benchmark)**:
   - Run `r30_momentum_evaluation.jl`.
   - Compare proper scoring (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS), supremacy slope, favourite win probability ($\ge 0.70$), and portfolio metrics.
5. **Stage 4 (Report & Sign-off)**:
   - Write full findings to `experiments/scottish_lower/10_momentum_multiscale_grw/README.md`.
   - Update `todos/022_prototype_momentum_multiscale_grw_dynamics.md` to `COMPLETED` and verify `./scripts/todo.sh check`.
