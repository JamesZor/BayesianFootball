# Negative Binomial + Linear Proxy-xG Form Covariate — Pi Task

You are tasked with executing **TODO 024: Prototype Negative Binomial with Linear Proxy-xG Form Covariate** on Scottish Lower football (tournaments 56 & 57, 40-fold walk-forward cohort over seasons 24/25 and 25/26, 710 fixtures).

### Your Model & Operating Mode
- You are running on **`openai-codex/gpt-5.6-sol` with `--thinking high`**.
- **SOLO AGENT**: Execute all mathematical modeling, feature extraction, Turing coding, and verification directly. **Do NOT spawn subagents.**
- Use your deep mathematical reasoning to formulate the linear proxy-xG form covariate, verify AD safety under compiled ReverseDiff gradient tapes, and systematically benchmark decompression against the 1.41–1.66 compression slope of Gen 3/4.

### Canonical Work Package Prompt:
Read your complete specification, context, mathematical definitions, and execution plan in:
[`experiments/scottish_lower/11_decompression_pxg_covariate/WORK_PACKAGE_PROMPT.md`](experiments/scottish_lower/11_decompression_pxg_covariate/WORK_PACKAGE_PROMPT.md)

### Key Files & References:
- Work Package Prompt: [`experiments/scottish_lower/11_decompression_pxg_covariate/WORK_PACKAGE_PROMPT.md`](experiments/scottish_lower/11_decompression_pxg_covariate/WORK_PACKAGE_PROMPT.md)
- Empirical Evidence & Attribution: [`current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`](current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md)
- Rolling Proxy-xG Lookup: `Features._pxg_rolling_lookup` (see usage in `current_development/market_inverse_dynamics/l01_market_inverse_loader.jl`)
- Covariate Reference: [`src/models/pregame/components/covariates/`](src/models/pregame/components/covariates/)
- Prior Paradigm Reference: [`experiments/scottish_lower/03_joint_gamma_poisson/`](experiments/scottish_lower/03_joint_gamma_poisson/)
- TODO Tracking: [`todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md`](todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md)

### Remote Compute Node (`mcmc-beast`):
- Compute node is ready and idle (load avg ~0.05).
- A matched worktree is checked out at `/root/BF_negbin_pxg_covariate` on branch `feat/scottish-lower-negbin-pxg-covariate` with `.env` and `Manifest.toml` configured.
- Julia executable on beast is `/root/.juliaup/bin/julia`.
- When running heavy MCMC sampling across the 40 folds (Stage 2), sync code to `mcmc-beast` (`git push` / `git -C /root/BF_negbin_pxg_covariate pull`) and execute via `ssh root@mcmc-beast "..."` in a detached tmux session.

### Execution Workflow:
1. **Stage 0 (Mathematical Formulation & Feature Design)**:
   - Formulate `ProxyXGFormCovariate` in `experiments/scottish_lower/11_decompression_pxg_covariate/l11_decompression_loader.jl`.
   - Ensure rolling window (10–16 matches) has zero future leakage.
   - Verify ReverseDiff gradient tape compilation and zero-allocation warmed execution.
2. **Stage 1 (Smoke Gate)**:
   - Run `r10_smoke.jl` on folds 1, 20, 40 across the 3 arms:
     - `m01_poisson_time_decay` (control 1)
     - `m02_joint_gamma_poisson` (control 2, compressed benchmark)
     - `m03_negbin_pxg_covariate` (candidate)
   - Verify ReverseDiff compilation, 0 divergences, R̂ ≤ 1.05, ESS ≥ 200, and $w_{\text{pxg}} \in [0.40, 0.80]$.
3. **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
   - Run `r20_production_grid.jl` on `mcmc-beast`.
   - Persist fits to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_decompression`.
4. **Stage 3 (Evaluation & Benchmark)**:
   - Run `r30_evaluation.jl`.
   - Measure supremacy slope vs Betfair close ($y = \beta x$). Verify whether $\beta$ decompresses towards $1.00$.
   - Compare proper scoring (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS) and full portfolio simulation on the common tradeable panel.
5. **Stage 4 (Report & Sign-off)**:
   - Write full findings to `experiments/scottish_lower/11_decompression_pxg_covariate/README.md`.
   - Update `todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md` to `COMPLETED` and verify `./scripts/todo.sh check`.
