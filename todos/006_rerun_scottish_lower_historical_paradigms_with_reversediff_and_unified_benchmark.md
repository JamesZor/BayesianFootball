# 006 — Rerun Scottish Lower historical paradigms with ReverseDiff and unified benchmark

| Field | Value |
|---|---|
| ID | 006 |
| Title | Rerun Scottish Lower historical paradigms with ReverseDiff and unified benchmark |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [r80_train_historical_paradigms_reversediff.jl](../experiments/scottish_lower/r80_train_historical_paradigms_reversediff.jl); [r90_compare_reversediff_paradigms.jl](../experiments/scottish_lower/r90_compare_reversediff_paradigms.jl); [compare_scottish_experiments.jl](../experiments/scottish_lower/compare_scottish_experiments.jl); [r31_train_5models_2426_negbin.jl](../experiments/scottish_lower/02_negbin_2426_grid/r31_train_5models_2426_negbin.jl); [01_poisson_2426_grid/](../experiments/scottish_lower/01_poisson_2426_grid/); [02_negbin_2426_grid/](../experiments/scottish_lower/02_negbin_2426_grid/); [03_joint_gamma_poisson/](../experiments/scottish_lower/03_joint_gamma_poisson/); [06_joint_player_lineup_fusion/](../experiments/scottish_lower/06_joint_player_lineup_fusion/) |

## Context & Problem Statement

Following the successful ReverseDiff parity verification of Experiment 01 (Poisson grid, TODO 005) where 40-fold sampling dropped to ~2 minutes per model, the user requested that `@pi` do the same for the other historical models run under earlier versions (ForwardDiff fallback) up to the recent paradigms:
1. Negative Binomial (Exp 02: `m00`, `m02`, `m03`, `m04`, `m05` in `scottish_lower_negbin_2426`)
2. Two-arm Joint Gamma-Poisson (Exp 03: `m00`, `m02`, `m03`, `m04`, `m05` in `scottish_lower_joint_2426`)
3. Joint Player-Lineup RAPM Fusion (Exp 05/06: `m09`, `m10`, `m11`, `m12_joint_hybrid_synergy` in `scottish_lower_joint_player_2426`)

The user requested a script to run them all with ReverseDiff, persist them to PostgreSQL `mcmc_experiments` (with `ad_backend = 'reversediff'`), and provide a script to compare them against the earlier runs / published baselines.

## Acceptance Criteria

- [x] Author or orchestrate a multi-paradigm runner script for Scottish Lower historical models (NegBin, Joint Gamma-Poisson, Player RAPM) using compiled ReverseDiff.
- [x] Run sampling across the walk-forward grid on `mcmc-beast` pinned to 16 cores.
- [x] Persist all resulting fits, latents, and out-of-sample proper scores to PostgreSQL `mcmc_experiments` with `ad_backend = 'reversediff'` and distinct tags.
- [x] Author or update a cross-paradigm SQL/evaluation comparison script (based on or extending `compare_scottish_experiments.jl`) that pulls both the historical and new ReverseDiff runs from `mcmc_experiments`.
- [x] Generate a comprehensive comparative report covering sampling speedups, convergence diagnostics ($R̂$, ESS, divergences), proper scores (LogLoss, Brier, RPS), and key parameter estimates.
- [x] Document findings and update work log in this TODO.

## Ideas & Candidate Solutions

- **Batching & Modularity**: Rather than one monolithic script that risks failing halfway, organize the runner into clear sequential stages (e.g. Stage 1: NegBin 5 models; Stage 2: Joint Gamma-Poisson 5 models; Stage 3: Player RAPM 4 models).
- **Postgres Deduplication**: Ensure every run sets `tags = [..., "reversediff"]` so each has a unique `config_hash`.
- **Comparison Script**: Update `compare_scottish_experiments.jl` or create a targeted `r90_compare_reversediff_paradigms.jl` that queries `mcmc_experiments` directly for runs with `ad_backend = 'reversediff'` vs older runs (`ad_backend IS NULL` or `'synthetic'`).
- **Chosen approach (2026-09-10, pi)**: Use one restartable runner with independent `negbin`, `joint`, and `player` stages and environment-variable model/stage filters. Persist and score after every model; only a 40/40 converged, 710-latent, tagged ReverseDiff row is reusable. A retry keeps the scientific recipe fixed and adds an attempt tag. Keep the existing broad portfolio/RQR comparison intact and add a focused database-only runtime/parity report rather than mixing execution-provenance questions into it.
- **Trade-off**: The selected player scope is the four mature Experiment 06 two-arm models (`m09`–`m12`), not the earlier single-arm Experiment 05 prototypes. This matches the canonical `scottish_lower_joint_player_2426` namespace and makes the historical/new pairs recipe-aligned.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task and assigned to `@pi` following user request. Outlined multi-paradigm scope and acceptance criteria. Prepared prompt for `pi_solo:exp01_poisson`.
- [2026-09-10 @pi] Audited canonical Experiment 02, 03, and 06 constructors and the existing database. Historical production evidence exists for all 14 target models; its `ad_backend` is null/unknown and runtimes are real (roughly 1.2–3.4 hours/model), unlike Experiment 01's synthetic imported baselines.
- [2026-09-10 @pi] Added `r80_train_historical_paradigms_reversediff.jl`: exact 40-fold/710-OOS preflight, queued 4×800 NUTS, pinned-core/BLAS isolation, stage/model filters, attempt tags, per-model local/PostgreSQL persistence, score persistence, strict relational gate reporting, and exact Fit/latent round-trip verification.
- [2026-09-10 @pi] Added `r90_compare_reversediff_paradigms.jl`: read-only PostgreSQL selection of historical/new pairs plus Poisson and goal-decomposition context, measured runtime speedups, convergence/proper-score tables, and deserialized fold-1 parameter summaries. Synthetic imports are explicitly excluded as timing/posterior evidence.
- [2026-09-10 @pi] Remote prepare-only preflight passed on `/root/BF_goal_decomposition` at `2adc1d7f`: Julia `-t 16`, BLAS 1, 40 folds and 710 OOS fixtures, all 14 model recipes/registry entries constructed. Launched the base production run in tmux `todo006_r80` on `mcmc-beast`.
- [2026-09-10 @pi] Base execution completed all 14 Fits. Eleven passed immediately. Preserved the failed-attempt rows, then retried only NegBin `m03`/`m04` and joint `m04` without changing the scientific recipe. Joint `m04` passed `retry_2`; both NegBin arms passed `retry_3`.
- [2026-09-10 @pi] Re-scored the four player Fits against the canonical Experiment 06 Betfair TWA-close contract (rather than bookmaker `ds.odds`) and persisted those comparable LogLoss/Brier/RPS values.
- [2026-09-10 @pi] Verified the selected local and PostgreSQL Fits chain-by-chain and latent-by-latent: **14/14 exact round-trips passed**. A separate relational audit confirmed **14/14** rows at `ad_backend='reversediff'`, 40/40 converged folds, 710 latent fixtures, non-null proper scores, and `todo006` plus attempt provenance tags.
- [2026-09-10 @pi] Executed the read-only comparison runner on `mcmc-beast` and copied its report/CSVs into the worktree. Closed the task as `COMPLETED`; no commit was created.

## Verification & Findings

### Selected production runs

| Paradigm | Model | Selected run UUID | Attempt | Runtime |
|---|---|---|---|---:|
| NegBin | `m00_negbin_baseline` | `92ef0a22-159f-488a-b874-86919cacf4ca` | base | 8m46s |
| NegBin | `m02_negbin_wealth` | `d6a58414-f4e1-426d-b80a-4a6cdbe82e77` | base | 8m48s |
| NegBin | `m03_negbin_distance` | `a3ded367-e436-40b0-b00b-4ea54e24dc21` | retry 3 | 9m40s |
| NegBin | `m04_negbin_joint` | `142a2946-346f-4a02-8c3f-92f17a121df1` | retry 3 | 9m25s |
| NegBin | `m05_negbin_production_wealth` | `f65a2fa9-a064-4db9-923c-5cfd3402aa9b` | base | 8m35s |
| Joint | `m00_joint_baseline` | `ac571e3a-28d9-468e-acf2-a679fec57afb` | base | 4m41s |
| Joint | `m02_joint_squad_wealth` | `92f43430-54d4-4b09-973d-4caf2c4c099e` | base | 4m58s |
| Joint | `m03_joint_distance` | `7cb1b734-3be7-4ab0-beb9-23439e9e9b7d` | base | 5m01s |
| Joint | `m04_joint_wealth_distance` | `f36e227f-3601-4509-97f8-82959517d79b` | retry 2 | 5m15s |
| Joint | `m05_joint_production_wealth` | `92a55d0b-86ba-4f08-a706-0684c95bec75` | base | 5m01s |
| Player | `m09_joint_player_shots_outfield` | `3f287c23-1090-42c7-b863-4da8b62378ca` | base | 6m43s |
| Player | `m10_joint_player_shots_bench` | `50c68816-ea62-43a8-8691-0d9b80445197` | base | 6m40s |
| Player | `m11_joint_player_pxg_bench` | `a92cafeb-4ad4-45a0-ba0f-a99dc0d2e1d2` | base | 6m45s |
| Player | `m12_joint_hybrid_synergy` | `928dad3b-ccaf-4909-b6b7-4f1a815e1cab` | base | 7m08s |

### Main findings

- All selected runs passed the strict relational contract. Across them, max aggregate $\hat R$ was 1.0154, minimum bulk/tail ESS was 714/426, and the largest total divergence count was 11 out of 128,000 retained draws.
- Like-for-like historical timing pairs show a median **23.8×** speedup (range **11.3×–34.2×**). The old NegBin `m00` and player `m12` rows had been extended to 42/43 folds, so their runtime ratios are deliberately omitted rather than called comparable.
- Proper-score parity is strong. Across comparable pairs, the largest absolute change was **0.000088 LogLoss**, **0.000043 Brier**, and **0.000062 RPS**. Player results reproduce the published Betfair-close values to the shown precision.
- Fold-1 parameter estimates also reproduce where both artifacts deserialize. For `m12`, lineup attack moved 0.2121 → 0.2128 and lineup defence 0.2569 → 0.2567; $\kappa$ moved 1.1292 → 1.1291 and $\nu$ 3.9724 → 3.9719. New joint arms retain $\kappa=1.1264$–1.1291 and $\nu=3.9105$–3.9450, matching the published historical ranges.
- Historical joint `m00`–`m05` and player `m09`–`m11` blobs predate the added kappa-mode type parameter and cannot be deserialized by the current Julia type. Their relational evidence remains queryable. The report uses explicitly labelled published Experiment 03 parameter summaries and Experiment 06 score baselines; it does not invent unavailable historical lineup-loading values.
- Failed base/retry attempts remain in PostgreSQL under distinct attempt tags. The selected UUIDs above are the rows that passed 40/40; selection does not erase negative convergence evidence.

### Reproduction and artifacts

```bash
# Heavy execution (mcmc-beast only)
julia --project -t 16 experiments/scottish_lower/r80_train_historical_paradigms_reversediff.jl

# Read-only database comparison; launches no MCMC
julia --project -t 2 experiments/scottish_lower/r90_compare_reversediff_paradigms.jl

# Repository task registry validation
./scripts/todo.sh check
```

- [ReverseDiff comparison report](../experiments/scottish_lower/REVERSEDIFF_PARADIGMS_COMPARISON.md)
- `experiments/scottish_lower/results/reversediff_paradigm_runs.csv`
- `experiments/scottish_lower/results/reversediff_paradigm_parameters.csv`
- Remote execution logs: `/root/BF_goal_decomposition/experiments/scottish_lower/results/todo006_reversediff/r80_base.log`, `r80_retry2.log`, and `r80_retry3.log`.
