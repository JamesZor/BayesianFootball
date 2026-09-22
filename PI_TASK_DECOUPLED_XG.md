# Decoupled Generative xG-Primary Funnel — Pi Task

You are tasked with executing **TODO 025: Prototype Decoupled Generative xG-Primary Model with Subordinate Goals** on Scottish Lower football (tournaments 56 & 57, 40-fold walk-forward cohort over seasons 24/25 and 25/26, 710 fixtures).

### Your Model & Operating Mode
- You are running on **`openai-codex/gpt-5.6-sol` with `--thinking high`**.
- **SOLO AGENT**: Execute all mathematical modeling, likelihood coding, Turing compilation, and verification directly. **Do NOT spawn subagents.**
- Use your deep mathematical reasoning to formulate the decoupled generative funnel ($\text{Team Quality} \to \text{xG} \to \text{Goals}$), verify AD safety under compiled ReverseDiff gradient tapes, and systematically benchmark decompression against the controls.

### Canonical Work Package Prompt:
Read your complete specification, context, mathematical definitions, and execution plan in:
[`experiments/scottish_lower/12_decoupled_generative_xg/WORK_PACKAGE_PROMPT.md`](experiments/scottish_lower/12_decoupled_generative_xg/WORK_PACKAGE_PROMPT.md)

### Key Questions to Answer:
1. Does decoupling chance creation from goal realization decompress the supremacy slope towards 1.00 (improving on the 1.724 slope of Gen 3/4)?
2. Does the decoupled funnel restore the Over/Under 2.5 and BTTS accuracy and portfolio growth (+128%) that were compromised by the single-arm NegBin model in TODO 024 (+83%)?
3. **Hierarchical $\kappa$ vs Shared $\kappa$**: Does team-level hierarchical finishing skill (`m04`) add any predictive value or portfolio profit over a shared league conversion factor (`m03`), or does it confirm the null result of Experiment 06?

### Remote Compute Node (`mcmc-beast`):
- Compute node is ready and idle.
- A matched worktree is checked out at `/root/BF_decoupled_xg_funnel` on branch `feat/scottish-lower-decoupled-xg-funnel` with `.env` and `Manifest.toml` configured.
- Julia executable on beast is `/root/.juliaup/bin/julia`.
- When running heavy MCMC sampling across the 40 folds (Stage 2), sync code to `mcmc-beast` (`git push` / `git -C /root/BF_decoupled_xg_funnel pull`) and execute via `ssh root@mcmc-beast "..."` in a detached tmux session.

### Execution Workflow:
1. **Stage 0 (Mathematical Formulation & Loader Design)**:
   - Scaffold `experiments/scottish_lower/12_decoupled_generative_xg/l12_loader.jl` and `r00_preflight.jl`.
   - Verify ReverseDiff gradient tape compilation and zero-allocation warmed execution across all 4 arms.
2. **Stage 1 (Smoke Gate)**:
   - Run `r10_smoke.jl` on folds 1, 20, 40 across the 4 arms:
     - `m01_poisson_time_decay` (control 1)
     - `m02_joint_gamma_poisson` (control 2, compressed benchmark)
     - `m03_funnel_shared_kappa` (candidate 1, shared league conversion)
     - `m04_funnel_hierarchical_kappa` (candidate 2, hierarchical team finishing)
   - Verify ReverseDiff compilation, 0 divergences, R̂ ≤ 1.05, ESS ≥ 200, and score-grid coherency.
3. **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
   - Run `r20_production_grid.jl` on `mcmc-beast`.
   - Persist fits to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_decoupled_xg`.
4. **Stage 3 (Evaluation & Benchmark)**:
   - Run `r30_evaluation.jl`.
   - Measure supremacy slope vs Betfair close ($y = \beta x$). Verify whether $\beta$ decompresses towards $1.00$.
   - Compare proper scoring (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS) and full portfolio simulation on the common 622-fixture tradeable panel.
5. **Stage 4 (Report & Sign-off)**:
   - Write full findings to `experiments/scottish_lower/12_decoupled_generative_xg/README.md`.
   - Update `todos/025_prototype_decoupled_generative_xg_primary_model.md` to `COMPLETED` and verify `./scripts/todo.sh check`.
