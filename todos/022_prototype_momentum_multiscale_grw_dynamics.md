# 022 — Prototype momentum multiscale GRW dynamics

| Field | Value |
|---|---|
| ID | 022 |
| Title | Prototype momentum multiscale GRW dynamics |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-21 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `experiments/scottish_lower/10_momentum_multiscale_grw/` |

## Context & Problem Statement

First-order Gaussian Random Walk models (`MultiScaleGRW`) exhibit compressed team-strength forecasts relative to the market. Their expected next **increment** is zero; they do not reset an attained rating level to zero. TODO 021 motivated testing directional dynamics after wider or forced-spread priors failed to resolve the compression.

Directional momentum / 2nd-order GRW ("The Second Term" / velocity) was hypothesized to support continued directional movement through inferred velocity ($v_t$). Phase 1 found modest decompression, not the hypothesized leap into the heavy-favourite regime; stronger forecasts and better betting performance were not guaranteed by the formulation.

Phase 1 scope: Prototype pure Poisson `MomentumMultiScaleGRW` on Scottish Lower (tournaments 56/57, 40-fold walk-forward grid, 710 matches) and benchmark against Time Decay (`m01`) and 1st-Order `MultiScaleGRW` (`m02`).

## Acceptance Criteria

- [x] **Stage 0 (Mathematical Research & Design)**:
  - Formulate and compare 2nd-order state-space representations (damped velocity vs kinematic acceleration) in Turing.jl.
  - Implement multiscale architecture: macro season step + micro matchday steps with momentum dynamics.
  - Ensure zero allocations in the inner loop and AD compatibility with compiled ReverseDiff tapes.
- [x] **Stage 1 (Smoke Gate, Folds 1/20/40)**:
  - ReverseDiff gradient tape compilation passes; warmed compiled-gradient replay has zero heap allocations (compilation itself is not claimed allocation-free).
  - 0 NUTS divergences; $\hat{R} \le 1.05$; bulk/tail ESS $\ge 200$.
  - Trajectory reconstruction passes; production market partitions agree with analytic retained Poisson mass to 1e-12, with truncation reported separately (user-approved clarification, 2026-09-21).
- [x] **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
  - Sample all 40 folds (710 fixtures, 4 chains $\times$ 800 warmup + 800 draws) across:
    - `m01_poisson_time_decay` (control)
    - `m02_poisson_grw_1st_order` (control)
    - `m03_poisson_momentum_grw` (candidate)
  - Persist runs to PostgreSQL `mcmc_experiments` (namespace `scottish_lower_momentum_grw`); user-approved full local fits/full-draw diagnostics and one-in-four DB draws, with exact reconstructed latents.
- [x] **Stage 3 (Evaluation & Benchmark)**:
  - Proper scoring (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS, ECE) vs Betfair closing odds.
  - Supremacy slope vs Betfair close and favourite-tail calibration on fixtures $\ge 0.70$.
  - Full portfolio backtest under `BookSpec(1X2, OU2.5, BakerMcHale)` and `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`, on the user-approved common tradeable panel with all refusals recorded.
- [x] **Stage 4 (Findings Report)**:
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

- [2026-09-22 @pi] Completed Stage 4 and signed off Phase 1. Stage 2 (`a6cca1bc`) and Stage 3 (`168d90c4`) both exited 0. Recorded production/evaluation tables, model and portfolio UUIDs, paired bootstrap intervals, weak momentum identification, market-coverage exclusions, storage semantics and material worst-draw grid truncation in the experiment README. Archived CSV evidence and added a read-only standard-library verifier. Research recommendation: retain the validated prototype, do not replace first-order GRW on this evidence. No live-model change or new sampling.

- [2026-09-21 @pi] Began solo execution in the provisioned momentum worktree. Read the specification, AD/model, runner, database and remote execution guides; verified beast connectivity and loader construction. User approved conditional-mean OOS forecasts (last level plus inferred velocity), zero boundary velocity, and matched first-order forecast convention. Design: unchanged macro/level priors; polynomial AR velocity convolution on micro match-biweeks; omit terminal prior-only innovation. Stationary velocity does not imply a stationary level or guaranteed decompression.

- [2026-09-21 @pi] Stage 2 launched after storage and all-fold preflights passed. Production recipe hashes: TimeDecay `3581efb64192b81e178999a2a6a23d0a9c2377b19a5e5855170c006bbc86115a`; first-order `d34bbfe4a6d19999d6a2f549d32c518dfd833482a31833cebdd8e44c907fc15b`; momentum `e056815bd235bf895c821145537d0703e390f28c396fefdbd869f4ad90f1917c`. No completed matches existed at lookup; native queue launched with recipe-addressed fold checkpoints.
- [2026-09-21 @pi] Stage 1 PASS after exact-chain re-audit: accepted UUIDs `a05fb858-8033-4d4f-a805-509c5b5daab4`, `02c1d10a-515f-4592-ba97-c895f8b38895`, `112bb865-c0e5-470a-b53a-619909367ced`. All portfolio ledgers survive persistence/repricing identically. User approved common book-coverage exclusions and full-local/stride-4-DB production storage. Nine per-fold diagnostics and posterior summaries committed as CSVs; phi remains weakly identified in smoke.
- [2026-09-21 @pi] Executed Stage 1 on beast at commit `7bea5069`. Sampling and numerical convergence satisfy the requested limits. Found own adapter's strict-zero threshold bug; prepared zero-only positive-float adapter and regression tests, plus exact-chain re-audit/persistence runner. Stage 2/3 runners are now implemented but not yet executed.
- [2026-09-21 @pi] Completed Stage 0 and prepare-only preflight. User requested AD optimization rather than relaxing the zero-allocation gate; achieved zero full-tape replay allocations via prototype-local array broadcasts, centering and scalar lifting. User approved retained-mass score-grid verification with separately reported truncation. User subsequently authorized Stage 1 sampling and gated progression to Stage 2.
- [2026-09-21 @antigravity] Created branch `feat/scottish-lower-momentum-grw`, provisioned worktree and remote compute directory on `mcmc-beast`, scaffolded experiment suite in `experiments/scottish_lower/10_momentum_multiscale_grw/`, claimed for @pi in session `agent_pi_momentum_grw`.

## Verification & Findings

- Stage 0: 202/202 deterministic assertions passed on beast (including the strict-zero audit adapter regression). Full linked-space compiled-gradient replay allocates **0 B** for all three arms on folds 1/20/40. Matched-site density parity ≤4.7e-10 and gradient relative error ≤3.6e-15, including ±3-coordinate-scale probes. No threshold waiver. Details and reproduction: experiment README and DESIGN.md.
- Prepare-only smoke: exact 40-fold/710-fixture inventory, filtration, all nine AD gates, registry registration and run-hash preflight passed. No existing completed smoke recipes found.
- Stage 1 sampling completed at 4×(400+400), 50 OOS fixtures across folds 1/20/40. All arms have zero divergences; max R-hat 1.0118/1.0252/1.0209, minimum bulk/tail ESS 552.8/399.0/359.9. Initial runner verdict was falsely negative because its strict divergence-rate threshold was configured as zero (`0 < 0`). Corrected the adapter and re-audited immutable chains without resampling. All six convergence gates and persistence/portfolio round-trips now pass. User approved an explicit common 44/50 tradeable portfolio panel; all 50 remain in latent/grid checks. Original run UUIDs are pinned in `r12_reaudit_smoke.jl`.
- Production all-fold prepare-only preflight passed. User approved full local fits/full-draw diagnostics and one-in-four PostgreSQL persistence for every arm to avoid the single-artifact size limit. Storage preflight passed for all three arms: reconstructed thinned rates equal the exact original columns and full-draw diagnostics are preserved. Stage 2 completed on beast at commit `a6cca1bc`, exit 0. Each arm has 40 folds / 710 fixtures / 128,000 audited retained transitions and zero divergences. Max R-hat 1.01178/1.01469/1.01357; minimum bulk/tail ESS 978.0/590.4/831.4. All fit/latent/chain round-trips passed.
- Production model UUIDs: TimeDecay `33d85b4a-e929-4738-8125-706e0dc26de1`; first-order `f8da493d-db2c-42d5-85e2-0f19d0107b1d`; momentum `3e06683b-de96-431f-843d-f98619d9fc13`. Portfolio UUIDs and original smoke lineage are in the experiment README and CSV evidence.
- Stage 3 completed at `168d90c4`, exit 0. All arms score identical 2,899 selections across 627 fixtures; supremacy uses 623 accepted inversions; portfolio uses 622 common tradeable fixtures (75 missing quotes + 13 unusable selections). Persistence/repricing produces identical portfolio ledgers.
- Momentum versus first-order: slope 0.3880→0.4338; favourite P(win) 55.01%→56.43% versus market 76.25% (18 fixtures). Overall binary LogLoss 0.644518→0.644441; paired Δ −0.000078, 95% interval [−0.001181,+0.001071]. Momentum worsens 1X2 LogLoss/RPS and return (+172.41% versus +197.10%), despite slightly better drawdown. Phi remains largely prior-driven. No production-replacement recommendation.
- Limitations retained explicitly: incomplete market coverage, conditional-mean forecasts, thinned evaluation draws, small favourite sample, and fixed score-grid truncation (momentum worst omitted mass 2.27% of a persisted draw / 5.31% of a full-panel draw). The retained-mass partition gate passes; this does not establish negligible truncation.
- Final verification: `python experiments/scottish_lower/10_momentum_multiscale_grw/verification/verify_results.py` passes the CSV cohort/score/ledger/lineage checks. `./scripts/todo.sh check` and `git diff --check` both passed on 2026-09-22. All scoped stages are complete; prototype remains outside `src/`.
