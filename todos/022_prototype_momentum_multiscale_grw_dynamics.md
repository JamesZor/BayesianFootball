# 022 — Prototype momentum multiscale GRW dynamics

| Field | Value |
|---|---|
| ID | 022 |
| Title | Prototype momentum multiscale GRW dynamics |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-21 |
| Updated | 2026-09-21 |
| Related Files / Commits / PRs | `experiments/scottish_lower/10_momentum_multiscale_grw/` |

## Context & Problem Statement

First-order Gaussian Random Walk models (`MultiScaleGRW`) exhibit Bayesian shrinkage compression on team ratings, pulling dominant teams back towards zero at every independent step. As demonstrated in TODO 021, static prior adjustments (scaling variance by 2.5x, Student-t, or forcing $\sigma_0 = 0.48$) fail because goals data pin independent step variance, and draw-mixture combinations cannot evaluate beyond their constituent arms.

Directional momentum / 2nd-order GRW ("The Second Term" / velocity) provides a generative mechanism where consecutive strong performances compound into persistent velocity ($v_t$), accelerating dominant teams into the heavy-favourite regime without adding unguided isotropic noise.

Phase 1 scope: Prototype pure Poisson `MomentumMultiScaleGRW` on Scottish Lower (tournaments 56/57, 40-fold walk-forward grid, 710 matches) and benchmark against Time Decay (`m01`) and 1st-Order `MultiScaleGRW` (`m02`).

## Acceptance Criteria

- [x] **Stage 0 (Mathematical Research & Design)**:
  - Formulate and compare 2nd-order state-space representations (damped velocity vs kinematic acceleration) in Turing.jl.
  - Implement multiscale architecture: macro season step + micro matchday steps with momentum dynamics.
  - Ensure zero allocations in the inner loop and AD compatibility with compiled ReverseDiff tapes.
- [x] **Stage 1 (Smoke Gate, Folds 1/20/40)**:
  - ReverseDiff gradient tape compilation passes without allocations.
  - 0 NUTS divergences; $\hat{R} \le 1.05$; bulk/tail ESS $\ge 200$.
  - Trajectory reconstruction passes; production market partitions agree with analytic retained Poisson mass to 1e-12, with truncation reported separately (user-approved clarification, 2026-09-21).
- [ ] **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
  - Sample all 40 folds (710 fixtures, 4 chains $\times$ 800 warmup + 800 draws) across:
    - `m01_poisson_time_decay` (control)
    - `m02_poisson_grw_1st_order` (control)
    - `m03_poisson_momentum_grw` (candidate)
  - Persist runs to PostgreSQL `mcmc_experiments` (namespace `scottish_lower_momentum_grw`).
- [ ] **Stage 3 (Evaluation & Benchmark)**:
  - Proper scoring (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS, ECE) vs Betfair closing odds.
  - Supremacy slope vs Betfair close and favourite-tail calibration on fixtures $\ge 0.70$.
  - Full portfolio backtest under `BookSpec(1X2, OU2.5, BakerMcHale)` and `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`.
- [ ] **Stage 4 (Findings Report)**:
  - Comprehensive report in `experiments/scottish_lower/10_momentum_multiscale_grw/README.md`.
  - Task completion signed off; `./scripts/todo.sh check` passes.

## Ideas & Candidate Solutions

- **Candidate A (Damped Velocity State-Space)**:
  $$\alpha_t = \alpha_{t-1} + v_{t-1} + \sigma_\alpha \epsilon_{\alpha, t}, \quad v_t = \phi v_{t-1} + \sigma_v \epsilon_{v, t}$$
  where $\phi \in [0, 1)$ governs momentum persistence.
- **Candidate B (Kinematic 2nd-Difference Acceleration)**:
  $$\Delta^2 \alpha_t = \sigma \epsilon_t \implies \alpha_t = 2\alpha_{t-1} - \alpha_{t-2} + \sigma \epsilon_t$$
- **Solo Execution**: Pi runs solo with `openai-codex/gpt-6-astra` and `--thinking high`; no subagents.

## Work Log & Progress

- [2026-09-21 @pi] Began solo execution in the provisioned momentum worktree. Read the specification, AD/model, runner, database and remote execution guides; verified beast connectivity and loader construction. User approved conditional-mean OOS forecasts (last level plus inferred velocity), zero boundary velocity, and matched first-order forecast convention. Design: unchanged macro/level priors; polynomial AR velocity convolution on micro match-biweeks; omit terminal prior-only innovation. Stationary velocity does not imply a stationary level or guaranteed decompression.

- [2026-09-21 @pi] Stage 1 PASS after exact-chain re-audit: accepted UUIDs `a05fb858-8033-4d4f-a805-509c5b5daab4`, `02c1d10a-515f-4592-ba97-c895f8b38895`, `112bb865-c0e5-470a-b53a-619909367ced`. All portfolio ledgers survive persistence/repricing identically. User approved common book-coverage exclusions and full-local/stride-4-DB production storage. Nine per-fold diagnostics and posterior summaries committed as CSVs; phi remains weakly identified in smoke.
- [2026-09-21 @pi] Executed Stage 1 on beast at commit `7bea5069`. Sampling and numerical convergence satisfy the requested limits. Found own adapter's strict-zero threshold bug; prepared zero-only positive-float adapter and regression tests, plus exact-chain re-audit/persistence runner. Stage 2/3 runners are now implemented but not yet executed.
- [2026-09-21 @pi] Completed Stage 0 and prepare-only preflight. User requested AD optimization rather than relaxing the zero-allocation gate; achieved zero full-tape replay allocations via prototype-local array broadcasts, centering and scalar lifting. User approved retained-mass score-grid verification with separately reported truncation. User subsequently authorized Stage 1 sampling and gated progression to Stage 2.
- [2026-09-21 @antigravity] Created branch `feat/scottish-lower-momentum-grw`, provisioned worktree and remote compute directory on `mcmc-beast`, scaffolded experiment suite in `experiments/scottish_lower/10_momentum_multiscale_grw/`, claimed for @pi in session `agent_pi_momentum_grw`.

## Verification & Findings

- Stage 0: 202/202 deterministic assertions passed on beast (including the strict-zero audit adapter regression). Full linked-space compiled-gradient replay allocates **0 B** for all three arms on folds 1/20/40. Matched-site density parity ≤4.7e-10 and gradient relative error ≤3.6e-15, including ±3-coordinate-scale probes. No threshold waiver. Details and reproduction: experiment README and DESIGN.md.
- Prepare-only smoke: exact 40-fold/710-fixture inventory, filtration, all nine AD gates, registry registration and run-hash preflight passed. No existing completed smoke recipes found.
- Stage 1 sampling completed at 4×(400+400), 50 OOS fixtures across folds 1/20/40. All arms have zero divergences; max R-hat 1.0118/1.0252/1.0209, minimum bulk/tail ESS 552.8/399.0/359.9. Initial runner verdict was falsely negative because its strict divergence-rate threshold was configured as zero (`0 < 0`). Corrected the adapter and re-audited immutable chains without resampling. All six convergence gates and persistence/portfolio round-trips now pass. User approved an explicit common 44/50 tradeable portfolio panel; all 50 remain in latent/grid checks. Original run UUIDs are pinned in `r12_reaudit_smoke.jl`.
- Production all-fold prepare-only preflight passed. User approved full local fits/full-draw diagnostics and one-in-four PostgreSQL persistence for every arm to avoid the single-artifact size limit. Storage preflight and production launch are next. No full-cohort performance claim or completion sign-off.
