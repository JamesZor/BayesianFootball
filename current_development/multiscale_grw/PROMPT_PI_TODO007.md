# Task 007 Work Package: Prototype MultiScaleGRW State-Space Dynamics with ReverseDiff

## Assignment Overview
You are assigned to implement, verify, and benchmark **Task 007** in `todos/007_prototype_gaussian_random_walk_state_space_dynamics_with_reversediff.md`.

## Infrastructure & Environment
* **Local Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-grw-dynamics`
  * Branch: `feat/multiscale-grw-dynamics` (already pushed and tracking `origin/feat/multiscale-grw-dynamics`).
  * Please `cd /home/james/bet_project/.worktrees/BayesianFootball-grw-dynamics` for all local file operations and git commands.
* **Remote Compute**: `root@mcmc-beast` (32 cores, 64 GB RAM).
  * Working directory on beast: `/root/BF_multiscale_grw` (cloned on `feat/multiscale-grw-dynamics`).
  * Julia command: `/root/.juliaup/bin/julia --project -t 16`.
  * Threads & BLAS: Pinned physical cores (`pinthreads(:cores)`), `BLAS.set_num_threads(1)`.

## Locked Architecture & Scope (Agreed via `/grill-me`)
1. **Directory Structure**:
   * Create prototype pair in `current_development/multiscale_grw/`:
     * `l01_loader.jl`: Data loading, feature extraction, Turing model definition adapting `MultiScaleGRW`.
     * `r01_runner.jl`: Execution pipeline (preflight, sampling, persistence, proper scoring).
2. **Mathematical Formulation**:
   * Revive and adapt `MultiScaleGRW` from `src/models/pregame/components/dynamics/team_level/multiscale.jl`:
     * Macro step per season ($z_{\text{season}} \sim \mathcal{N}(0, 1) \cdot \sigma_s$).
     * Micro step per match in target season ($z_{\text{target}} \sim \mathcal{N}(0, 1) \cdot \sigma_k$).
     * Non-centered parameterization with cumulative summation along time axis:
       $$\alpha_{\text{raw}} = \text{cumsum}([z_{\text{init}} \cdot \sigma_0, z_{\text{season}} \cdot \sigma_s, z_{\text{target}} \cdot \sigma_k], \text{dims}=2)$$
       $$\alpha = \alpha_{\text{raw}} - \text{mean}(\alpha_{\text{raw}}, \text{dims}=1)$$
   * Ensure zero runtime heap allocations in the ReverseDiff gradient evaluation.
3. **Execution Phasing**:
   * **Preflight**: 2-fold CV test to compile ReverseDiff tape, test gradients, verify 0 divergences.
   * **Phase 1 (Poisson Foundation)**:
     * Model 1: `m00_baseline_grw` (Global Interception + Home Advantage + `MultiScaleGRW` + Poisson).
     * Model 2: `m05_production_wealth_grw` (Interception + Home Advantage + Production Wealth + `MultiScaleGRW` + Poisson).
     * Run 40-fold walk-forward grid on `mcmc-beast` (`24/25` + `25/26`, 710 held-out matches).
     * Persist to PostgreSQL `mcmc_experiments` (`ad_backend = 'reversediff'`, tag `multiscale_grw`, `todo007`).
     * Evaluate OOS proper scores (LogLoss, Brier, RPS) vs `TimeDecayDynamics` baseline.
   * **Phase 2 (Overnight Autonomy - Two-Arm Joint $\Gamma$-Poisson)**:
     * If Phase 1 passes convergence gates ($\hat{R} \le 1.01$, $\text{ESS} \ge 400$, minimal divergences), automatically proceed to:
     * Model 3: `m05_joint_production_wealth_grw` (Joint Gamma-Poisson likelihood + Production Wealth + `MultiScaleGRW`).
     * Run 40-fold grid, persist, evaluate proper scores, and compare against Gen 3 `m05_joint_production_wealth`.

4. **Deliverables & Verification**:
   * `current_development/multiscale_grw/l01_loader.jl`
   * `current_development/multiscale_grw/r01_runner.jl`
   * `current_development/multiscale_grw/README.md` with speedup and proper score comparison tables.
   * Update `todos/007_prototype_gaussian_random_walk_state_space_dynamics_with_reversediff.md`.
   * Run `./scripts/todo.sh check` to verify consistency.
